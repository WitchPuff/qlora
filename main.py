from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel
import evaluate
import wandb
import time
import os
import torch


class EfficientMetricsCallback(TrainerCallback):
    def __init__(self, eval_dataset, model_dir="model"):
        self.eval_dataset = eval_dataset
        self.model_dir = model_dir
        self.train_start_time = None


    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started.")
        self.train_start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_train_end(self, args, state, control, **kwargs):
        print("Training ended.")
        train_time = time.time() - self.train_start_time
        model = kwargs["model"]

        # Log training time
        print(f"Total training time: {train_time:.2f} sec")
        wandb.log({"train/total_time_sec": train_time})

        # Log parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        wandb.log({"model/total_params": total_params})

        # Log model size
        model_size = self.get_model_size(self.model_dir)
        print(f"Model size: {model_size:.2f} MB")
        wandb.log({"model/disk_size_mb": model_size})

        # Log max memory
        max_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Max VRAM used: {max_mem:.2f} GB")
        wandb.log({"memory/max_vram_gb": max_mem})


    def get_model_size(self, path):
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)  # MB






class EfficientEvalCallback(TrainerCallback):
    def __init__(self, name="eval"):
        self.name = name
        self.eval_start_time = None
        self._armed = False

    def on_prediction_step(self, args, state, control, **kwargs):
        if not self._armed:
            self._armed = True
            self.eval_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_time = time.time() - self.eval_start_time

        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

        model = kwargs.get("model", None)
        total_params = sum(p.numel() for p in model.parameters()) if model is not None else None

        print(f"Evaluation ended for {self.name}.")
        print(f"[{self.name}] Eval time: {eval_time:.2f} sec")
        print(f"[{self.name}] Max VRAM used: {max_mem:.2f} GB")
        print(f"[{self.name}] Total parameters: {total_params}")

        log_data = {}
        log_data[f"{self.name}/eval_time_sec"] = eval_time
        log_data[f"{self.name}/max_vram_gb"] = max_mem
        log_data[f"{self.name}/total_params"] = total_params
        log_data.update({f"{self.name}/{k}": v for k, v in metrics.items()})
        wandb.log(log_data, step=state.global_step)
        print(log_data)
        self.eval_start_time = None
        self._armed = False
        
        
def get_dataset(dataset, model_name="roberta-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tok(ex): 
        return tokenizer(ex["sentence"], truncation=True, padding="max_length", max_length=128)
    
    dataset = dataset.map(tok, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    print(dataset.keys())
    print('Train set:', len(dataset['train']))
    print('Validation set:', len(dataset['validation']))
    print('Sample:', dataset['train'][0])
    return dataset

def load_backbone(model_name="roberta-base", precision="fp16"):
    print(f"Loading {model_name} with {precision} precision")

    if precision == "fp16":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, device_map="auto")
    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_has_fp16_weight=False,
            llm_int8_skip_modules=["classifier", "pre_classifier"]
        )
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, quantization_config=bnb_config, device_map="auto")
    elif precision in ["nf4", "fp4"]:
        qtype = "nf4" if precision=="nf4" else "fp4"
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # gradient issue fixed as referenced: https://github.com/huggingface/peft/issues/1720
            llm_int8_skip_modules=["classifier", "pre_classifier"] 
        )
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, quantization_config=bnb_cfg, device_map="auto")
    else:
        raise ValueError


def compute_metrics(logits):
    acc = evaluate.load("accuracy")
    return acc.compute(predictions=logits.predictions.argmax(-1), references=logits.label_ids)




def train(model, dataset, r=8, lora_alpha=16, target_modules=["query", "value"],
        learning_rate=1e-3, epochs=25, batch_size=256, weight_decay=1e-2):
    

    
    config = {
        "target_modules": target_modules,
        "r": r,
        "alpha": lora_alpha,
        "lr": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay
    }
    output_dir = "_".join([f"{k}_{v if k != 'target_modules' else '_'.join(v)}" for k, v in config.items()])
    output_dir += f"_{int(time.time())}"
    wandb.init(project="qlora", name=output_dir, config=config)

    peft_cfg = LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=target_modules,
        bias="none", task_type="SEQ_CLS"
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    os.makedirs(os.path.join('log', output_dir), exist_ok=True)

    args = TrainingArguments(
        output_dir=os.path.join('log', output_dir),
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=int(batch_size*2),
        learning_rate=learning_rate, num_train_epochs=epochs, weight_decay=weight_decay,
        eval_strategy="epoch", save_strategy="epoch",
        logging_steps=50, bf16=False, fp16=True, report_to="wandb",
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EfficientMetricsCallback(eval_dataset=dataset["validation"], 
                            model_dir=args.output_dir)]
    )

    trainer.train()
    trainer.evaluate(eval_dataset=dataset["validation"], 
                            metric_key_prefix="inference")
    output_dir = os.path.join('ckpt', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    return output_dir

def eval(model, dataset, precision, batch_size=256, output_dir='ckpt'):
    model.eval()

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"{precision}_eval"),
        per_device_eval_batch_size=int(batch_size*2),
        eval_strategy="no", logging_steps=50, 
        bf16=False, fp16=False, report_to="none",
    )


    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EfficientEvalCallback(name=precision+"_eval")]
    )
    with torch.no_grad():
        # Int8 Issue: Error occurred during evaluation: 'MatmulLtState' object has no attribute 'memory_efficient_backward'
        # Solution: pip install bitsandbytes==0.44.0 accelerate==1.0.1 peft==0.13.0 transformers==4.46.3
        # reference: https://github.com/tloen/alpaca-lora/issues/271
        metrics = trainer.evaluate(metric_key_prefix=precision+"_eval")
    print(metrics)
    wandb.log(metrics)
    
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs='+', default=["query", "value"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    dataset = load_dataset("glue", "sst2")
    dataset = get_dataset(dataset, model_name)
    if not args.eval:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        output_dir = train(model, dataset, r=args.r, lora_alpha=args.lora_alpha,
                        target_modules=args.target_modules, epochs=args.epochs, batch_size=args.batch_size,
                        learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    elif os.path.exists(args.ckpt):
        output_dir = args.ckpt
        wandb.init(project="qlora", name=output_dir.split("/")[-1]+"_eval", config=args)
    else:
        raise ValueError("Please provide a checkpoint directory for evaluation only mode.")
    try:
        print(f"Loading checkpoints from {output_dir}")
        for precision in ["nf4", "fp4", "int8", "fp16"]:
            print("Testing precision:", precision)
            model = load_backbone(model_name=args.model_name, precision=precision)
            model = PeftModel.from_pretrained(model, output_dir, is_trainable=False)
            eval(model, dataset, precision=precision, batch_size=args.batch_size, output_dir=output_dir)
    except Exception as e:
        print("Error occurred during evaluation:", e)

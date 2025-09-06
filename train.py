from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import evaluate, torch
from torch.utils.data import DataLoader
import wandb

import time
import os
import torch
import wandb
from transformers import TrainerCallback

class EfficientMetricsCallback(TrainerCallback):
    def __init__(self, eval_dataset, batch_size=64, model_dir="model"):
        self.eval_dataset = eval_dataset
        self.model_dir = model_dir
        self.train_start_time = None
        self.eval_dataloader = DataLoader(
            dataset["validation"],
            batch_size=batch_size,
            shuffle=False
        )

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



def get_dataset(dataset, model_name="roberta-base"):

    def tok(ex): 
        return tokenizer(ex["sentence"], truncation=True, padding="max_length", max_length=128)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dataset = dataset.map(tok, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    print(dataset.keys())
    print('Train set:', len(dataset['train']))
    print('Validation set:', len(dataset['validation']))
    print('Sample:', dataset['train'][0])
    return dataset



def train(model, dataset, r=8, lora_alpha=16, target_modules=["query", "value"],
          learning_rate=1e-3, epochs=20, batch_size=32, weight_decay=1e-2):
    
    def compute_metrics(logits):
        
        acc = evaluate.load("accuracy")
        return acc.compute(predictions=logits.predictions.argmax(-1), references=logits.label_ids)
    
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
    wandb.init(project="quantized_lora", name=output_dir, config=config)

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
                            batch_size=args.per_device_eval_batch_size, model_dir=args.output_dir)]
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=dataset["validation"], 
                               metric_key_prefix="inference")
    wandb.log(metrics)
    print(metrics)
    os.makedirs(os.path.join('ckpt', output_dir), exist_ok=True)
    trainer.save_model(os.path.join('ckpt', output_dir))


if __name__ == '__main__':
    
    model_name = "roberta-base"
    dataset = load_dataset("glue", "sst2")
    dataset = get_dataset(dataset, model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    train(model, dataset, r=8, lora_alpha=16, target_modules=["query","value"], epochs=25)
    



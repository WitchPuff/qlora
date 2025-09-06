# from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from torch.utils.data import DataLoader
# from transformers import TrainerCallback
# import time
# import torch
# import wandb


# model_name = "roberta-base"


# def load_backbone(model_name="roberta-base", precision="fp16"):
#     print(f"Loading {model_name} with {precision} precision")
#     if precision == "fp16":
#         return AutoModelForSequenceClassification.from_pretrained(
#             model_name, num_labels=2, device_map="auto")
#     elif precision == "int8":
#         return AutoModelForSequenceClassification.from_pretrained(
#             model_name, num_labels=2, load_in_8bit=True, device_map="auto")
#     elif precision in ["nf4", "fp4"]:
#         qtype = "nf4" if precision=="nf4" else "fp4"
#         bnb_cfg = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type=qtype,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             # gradient issue fixed as referenced: https://github.com/huggingface/peft/issues/1720
#             llm_int8_skip_modules=["classifier", "pre_classifier"] 
#         )
#         return AutoModelForSequenceClassification.from_pretrained(
#             model_name, num_labels=2, quantization_config=bnb_cfg, device_map="auto")
#     else:
#         raise ValueError




# class EfficientEvalCallback(TrainerCallback):
#     def __init__(self, name="eval"):
#         self.name = name
#         self.eval_start_time = None

#     def on_evaluate(self, args, state, control, **kwargs):
#         self.eval_start_time = time.time()
#         torch.cuda.reset_peak_memory_stats()

#     def on_evaluate_end(self, args, state, control, **kwargs):
#         eval_time = time.time() - self.eval_start_time
#         max_mem = torch.cuda.max_memory_allocated() / (1024**3)

#         # Log to console
#         print(f"[{self.name}] Eval time: {eval_time:.2f} sec")
#         print(f"[{self.name}] Max VRAM used: {max_mem:.2f} GB")

#         # Log to wandb
#         wandb.log({
#             f"{self.name}/eval_time_sec": eval_time,
#             f"{self.name}/max_vram_gb": max_mem,
#         })

# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, default="roberta-base")
#     parser.add_argument("--path", type=str, help="Path to the fine-tuned LoRA model")
#     parser.add_argument("--batch_size", type=int, default=256)
#     args = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     args = parse_args()



#     dataset = load_dataset("glue", "sst2")
#     dataset = get_dataset(dataset, model_name)
#     dl = DataLoader(dataset["validation"], batch_size=256, shuffle=False)
    
#     for precision in ["fp16", "int8", "nf4", "fp4"]:
#         print("Testing precision:", precision)
#         model = load_backbone(model_name=args.model_name, precision=precision)
#         model = PeftModel.from_pretrained(model, args.path, is_trainable=False)
#         model.eval()
#         correct = total = 0
        
#         training_args = TrainingArguments(
#             output_dir="./results",
#             per_device_eval_batch_size=args.batch_size,
#             dataloader_drop_last=False,
#         )
#         torch.cuda.reset_peak_memory_stats()
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             eval_dataset=dataset["validation"],
#             tokenizer=tokenizer,
#             compute_metrics=compute_metrics,
#             callbacks=[
#                 EfficientEvalCallback(name="val"),
#                 EfficientMetricsCallback(eval_dataset=dataset["validation"], model_dir="./model")
#             ]
#         )

#         # 在训练结束或中途验证时触发：
#         trainer.evaluate()
#         with torch.no_grad():
#             for batch in dl:
#                 batch = {k:v.to(model.device) for k,v in batch.items()}
#                 out = model(**batch)
#                 pred = out.logits.argmax(-1)
#                 correct += (pred == batch["labels"]).sum().item()
#                 total += batch["labels"].size(0)

#         acc_val = correct / total
#         vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
#         print(f"{precision}:", acc_val, "Acc,", round(vram_gb,2), "GB")
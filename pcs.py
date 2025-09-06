from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer



model_name = "roberta-base"


def load_backbone(model_name="roberta-base", precision="fp16"):
    print(f"Loading {model_name} with {precision} precision")
    if precision == "fp16":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, device_map="auto")
    elif precision == "int8":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, load_in_8bit=True, device_map="auto")
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

precision = "nf4"
backbone = load_backbone(precision=precision)
path = 'ckpt/target_modules_query_value_r_8_alpha_16_lr_0.001_epochs_25_batch_size_32_weight_decay_0.01'
model = PeftModel.from_pretrained(backbone, path, is_trainable=False)
model.eval()


from train import get_dataset
dataset = load_dataset("glue", "sst2")
dataset = get_dataset(dataset, model_name)
from torch.utils.data import DataLoader
dl = DataLoader(dataset["validation"], batch_size=64, shuffle=False)
correct = total = 0
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    for batch in dl:
        batch = {k:v.to(model.device) for k,v in batch.items()}
        out = model(**batch)
        pred = out.logits.argmax(-1)
        correct += (pred == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

acc_val = correct / total
vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
print(f"{precision}:", acc_val, "Acc,", round(vram_gb,2), "GB")
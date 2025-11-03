import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import torch

# Load config
with open("config/model_config.json") as f:
    model_config = json.load(f)

model_name = model_config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
output_dir = model_config.get("output_dir", "models/karen-lora")

print("[*] Loading base model:", model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # float32 ffor CPU
    device_map={"": "cpu"},      # force model ke CPU
)

# don't prepare_model_for_kbit_training on CPU-only

# Setup LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['v_proj', 'q_proj', 'k_proj'],  # sesuaikan target modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


# Load dataset
print("[*] Loading processed dataset...")
dataset = load_from_disk("data/processed/combined_dataset") # old data/processed
train_data = dataset

# Training args
args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    max_steps=100,
    learning_rate=2e-4,
    fp16=False,                # dddeactivate fp16
    logging_steps=5,
    output_dir=output_dir,
    save_total_limit=2,
    save_steps=50,
    optim="adamw_torch",      # optimizer CPU friendly
    report_to="none",
    no_cuda=True,             # No CUDA
)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=args,
)

print("[*] Starting LoRA fine-tuning on CPU...")
trainer.train()

print("[+] Saving trained model...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"[DONE] Training complete. Model saved to: {output_dir}")

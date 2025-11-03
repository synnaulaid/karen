import json
from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("[*] Loading dataset...")
dataset = load_dataset("json", data_files="dataset/initial_karen_dialog.jsonl")

def tokenize_fn(example):
    # Gunakan speaker dan text untuk membuat satu string percakapan
    text = f"{example['speaker']}: {example['text']}"
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("[*] Tokenizing dataset...")
tokenized = dataset["train"].map(tokenize_fn, batched=False)

print("[*] Saving processed dataset...")
tokenized.save_to_disk("data/processed/initial_karen")

print("[DONE] Dataset KAREN siap digunakan untuk training.")

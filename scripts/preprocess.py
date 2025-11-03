import json
from datasets import Dataset
from transformers import AutoTokenizer
import os

# Load configs
with open("config/dataset_config.json") as f:
    dataset_cfg = json.load(f)
with open("config/model_config.json") as f:
    model_cfg = json.load(f)

# Init tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])

print("[+] Loading dataset from local file...")
with open(dataset_cfg["url"], "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"[*] Dataset loaded, total entries: {len(data)}")

tokenized_samples = []
count = 0

for example in data:
    conv = example.get("conversations", [])
    if len(conv) < 2:
        continue

    text = f"<human>: {conv[0]['value']}\n<assistant>: {conv[1]['value']}"
    tokenized = tokenizer(
        text,
        max_length=model_cfg["max_length"],
        padding=model_cfg["padding"],
        truncation=model_cfg["truncation"],
        return_tensors=None,
    )
    # Copy input_ids to labels for causal LM training
    tokenized["labels"] = tokenized["input_ids"].copy()

    tokenized_samples.append(tokenized)
    count += 1

    if count % 100 == 0:
        print(f"[*] Processed {count} samples...")
    if count >= 1000:  # limite try
        break

print(f"Total samples processed: {count}")

dataset = Dataset.from_list(tokenized_samples)

os.makedirs("data/processed", exist_ok=True)
dataset.save_to_disk("data/processed")

print("[DONE] Dataset saved to disk at data/processed")

from datasets import load_from_disk, concatenate_datasets

print("[*] Loading both datasets...")
old_dataset = load_from_disk("data/processed")
new_dataset = load_from_disk("data/processed/initial_karen")

print("[*] Combining datasets...")
combined = concatenate_datasets([old_dataset, new_dataset])

output_path = "data/processed/combined_final"
print(f"[*] Saving combined dataset to {output_path} ...")
combined.save_to_disk(output_path)

print("[DONE] Combined dataset saved successfully!")

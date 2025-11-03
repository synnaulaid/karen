from datasets import load_dataset, concatenate_datasets

def main():
    # Load old dataset (arrow)
    old_dataset = load_dataset('arrow', data_files='data/processed/data-00000-of-00001.arrow')['train']

    # Load new dataset (jsonl)
    new_dataset = load_dataset('json', data_files='data/new_dialog.jsonl')['train']

    # join dataset
    combined_dataset = concatenate_datasets([old_dataset, new_dataset])

    # save
    combined_dataset.save_to_disk('data/processed/combined_dataset')

    print(f"Old Dataset: {len(old_dataset)} samples")
    print(f"New Dataset: {len(new_dataset)} samples")
    print(f"Dataset Join: {len(combined_dataset)} samples")
    print("Dataset joined 'data/processed/combined_dataset'")

if __name__ == "__main__":
    main()

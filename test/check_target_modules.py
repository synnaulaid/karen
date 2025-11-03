from transformers import AutoModelForCausalLM, AutoTokenizer

def find_target_modules(model, keywords=None):
    if keywords is None:
        keywords = ["q_proj", "v_proj", "k_proj", "out_proj", "query_key_value"]

    found = set()
    for name, module in model.named_modules():
        for kw in keywords:
            if kw in name:
                found.add(name.split('.')[-1])  # ambil nama terakhir sebagai modul target
    return list(found)

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Scanning model modules...")
    target_modules = find_target_modules(model)

    if target_modules:
        print("Found possible target modules for LoRA:")
        for tm in target_modules:
            print(f" - {tm}")
        print("\nContoh konfigurasi LoRA dengan target_modules ini:")
        print(f"""
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules={target_modules},
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
""")
    else:
        print("Tidak ditemukan modul target yang cocok. Coba cek nama modul lain atau model yang berbeda.")

if __name__ == "__main__":
    main()

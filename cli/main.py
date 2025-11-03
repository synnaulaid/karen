import os
import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === Konfigurasi ===
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "KAREN"

# Pastikan path model relatif ke file ini
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_MODEL_PATH = os.path.join(CURRENT_DIR, "../models/karen-lora")

print("[*] Loading tokenizer dan model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

# Gunakan local_files_only=True agar tidak nyari ke HF Hub
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH, local_files_only=True)

print(f"\n[+] Model {MODEL_NAME} siap digunakan! Ketik 'exit' untuk keluar.\n")

# === Fungsi animasi ngetik ===
def type_effect(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()  # newline di akhir

# === Loop interaktif ===
chat_history = ""

while True:
    try:
        user_input = input("Kamu: ")
    except (KeyboardInterrupt, EOFError):
        type_effect(f"\n{MODEL_NAME}: Sampai jumpa ya 💙")
        break

    if user_input.lower() in ["exit", "quit"]:
        type_effect(f"{MODEL_NAME}: Sampai jumpa ya 💙")
        break

    # Tambahkan konteks percakapan
    chat_history += f"Kamu: {user_input}\n{MODEL_NAME}:"

    # Tokenisasi
    inputs = tokenizer(chat_history, return_tensors="pt")

    # Generate respons model
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ambil hanya bagian jawaban terbaru
    if f"{MODEL_NAME}:" in response:
        response = response.split(f"{MODEL_NAME}:")[-1].strip()
    elif "KAREN:" in response:
        response = response.split("KAREN:")[-1].strip()

    type_effect(f"{MODEL_NAME}: {response}\n")

    # Update history
    chat_history += f" {response}\n"

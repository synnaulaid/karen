import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_model_path = "../models/karen-lora"
model_name = "KAREN"

print("[*] Loading tokenizer dan model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, lora_model_path)

print(f"\n[+] Model {model_name} siap digunakan! Ketik 'exit' untuk keluar.\n")

def type_effect(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

system_prompt = "Kamu adalah KAREN, asisten AI berbahasa Indonesia yang ramah, jujur, dan natural."

while True:
    user_input = input("Kamu: ")
    if user_input.lower() in ["exit", "quit"]:
        type_effect(f"{model_name}: Sampai jumpa ya 💙")
        break

    prompt = f"{system_prompt}\nKamu: {user_input}\n{model_name}:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split(f"{model_name}:")[-1].strip()
    type_effect(f"{model_name}: {response}\n")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_model_path = "models/karen-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(base_model, lora_model_path)

prompt = "Halo, siapa kamu?"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

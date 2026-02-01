from transformers import AutoTokenizer, AutoModelForCausalLM

model = "distilgpt2"

AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)

print("HF cache downloaded successfully")

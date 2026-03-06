import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "tsai_model"

# ======================
# Model
# ======================

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
    low_cpu_mem_usage=True,
)
model.eval()
print("Model loaded successfully")
print("Model input embeddings: ", model.get_input_embeddings().weight.shape)

# Model diagnostics:
total = 0
loaded = 0
for n,p in model.named_parameters():
    total += p.numel()
    if p.abs().mean() > 1e-6:
        loaded += p.numel()
print("fraction loaded:", loaded/total)

# ======================
# Tokenizer
# ======================

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    use_fast=False
    # fix_mistral_regex=False # didn;t work here
)
tokenizer.fix_mistral_regex = False
inputs = tokenizer("Hello world", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
print("logits mean:", logits.mean().item())
print("logits std :", logits.std().item())
print(tokenizer.decode(logits.argmax(-1)[0]))

next_token = logits[:, -1, :].argmax(dim=-1)
print(outputs.logits.shape) # (batch, seq_len, vocab_size) -> (1, sequence_length, 131072)
print("Next token:", tokenizer.decode(next_token))

# A slightly longer test
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
out = model(**inputs).logits
print("Capital of france: ", tokenizer.decode(out[0,-1].argmax().unsqueeze(0)))
# test_memory_gpu.py - 3B Model Memory Test
import gc

import torch
from src.data import get_tokenizer
from src.models.model_3b import Model3B, ModelConfig


def print_memory(stage):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(
        f"{stage:30s} | Allocated: {allocated:5.2f} GB | Reserved: {reserved:5.2f} GB"
    )


# Clear memory
torch.cuda.empty_cache()
gc.collect()

print("\n" + "=" * 80)
print("Memory Test for 3B Model (3.9B params, ~1.74B active) on GPU")
print("=" * 80 + "\n")

print_memory("Initial")

# Create model using the 3B model from src/models/model_3b.py
print("\nCreating model...")
print("  Loading TSAI 131K tokenizer...")
tokenizer = get_tokenizer()

# Create model configuration
config = ModelConfig()
config.vocab_size = len(tokenizer)  # 131,072 tokens

# Create model with standard embeddings (faster for testing)
# Note: Use embedding_type="kronecker" for production with Kronecker embeddings
print(f"  Creating 3B Model (vocab_size={config.vocab_size:,})...")
model = Model3B(
    config=config,
    embedding_type="standard",  # Standard embeddings for simpler testing
    bpe_vocab=None,
    pf_codec=None,
)
model = model.to("cuda")
print_memory("After model load")

# Create input
print("\nTesting forward pass...")
vocab_size = config.vocab_size
x = torch.randint(0, vocab_size, (1, 1024)).cuda()
print_memory("After input creation")

# Forward pass
print("  Running forward pass with mixed precision (bfloat16)...")
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # Model returns (logits_ntp, logits_mtp, aux_loss) or (logits_ntp, logits_mtp)
    logits_ntp, logits_mtp, aux_loss = model(x, next_token_ids=x, return_loss=True)
print_memory("After forward pass")

# Backward pass
print("\nTesting backward pass...")
# Compute simple loss for testing
loss = logits_ntp.mean()
if aux_loss is not None and aux_loss.numel() > 0:
    loss = loss + aux_loss
loss.backward()
torch.cuda.synchronize()
print_memory("After backward pass")

# Peak memory
peak = torch.cuda.max_memory_allocated() / 1e9
print("\n" + "=" * 80)
print(f"PEAK GPU MEMORY: {peak:.2f} GB")
print("\nGPU Compatibility:")
print(
    f"  T4 (16 GB):    {'✅ FITS' if peak < 14.0 else '❌ TOO LARGE'} (Headroom: {16.0 - peak:.2f} GB)"
)
print(
    f"  L4 (24 GB):    {'✅ FITS' if peak < 22.0 else '❌ TOO LARGE'} (Headroom: {24.0 - peak:.2f} GB)"
)
print(
    f"  A10G (24 GB):  {'✅ FITS' if peak < 22.0 else '❌ TOO LARGE'} (Headroom: {24.0 - peak:.2f} GB)"
)
print(
    f"  V100 (32 GB):  {'✅ FITS' if peak < 30.0 else '❌ TOO LARGE'} (Headroom: {32.0 - peak:.2f} GB)"
)
print(
    f"  A100 (40 GB):  {'✅ FITS' if peak < 38.0 else '❌ TOO LARGE'} (Headroom: {40.0 - peak:.2f} GB)"
)
print(
    f"  A100 (80 GB):  {'✅ FITS' if peak < 78.0 else '❌ TOO LARGE'} (Headroom: {80.0 - peak:.2f} GB)"
)

print("\nMemory Status:")
if peak < 12.0:
    print("  ✅ VERY SAFE - Works on most GPUs with huge headroom!")
elif peak < 16.0:
    print("  ✅ SAFE - Works on T4 and larger GPUs")
elif peak < 24.0:
    print("  ⚠️  MODERATE - Requires L4/A10G or larger")
elif peak < 32.0:
    print("  ⚠️  HIGH - Requires V100 32GB or larger")
else:
    print("  ❌ VERY HIGH - Requires A100 40GB or larger")

print("=" * 80 + "\n")

print("Note: This test uses standard embeddings. Kronecker embeddings may have")
print("      different memory characteristics. For production training, use")
print("      DeepSpeed ZeRO-2/3 for multi-GPU memory optimization.")
print("=" * 80 + "\n")

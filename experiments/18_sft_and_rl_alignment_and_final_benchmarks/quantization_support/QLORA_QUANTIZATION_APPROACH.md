# QLoRA Quantization Approach for Team 18

This document provides a comprehensive guide to quantization strategies for SFT and RL alignment training. It addresses GitHub Issue #333: "quantization formats are guaranteed supported end-to-end".

---

## Table of Contents

1. [Quantization Formats](#1-quantization-formats)
2. [Layer-Specific Quantization Strategy](#2-layer-specific-quantization-strategy)
3. [Training Pipeline Stages](#3-training-pipeline-stages)
4. [Hardware-Specific Considerations](#4-hardware-specific-considerations)
5. [End-to-End Validation Checklist](#5-end-to-end-validation-checklist)
6. [Code Templates](#6-code-templates)
7. [Common Pitfalls and Troubleshooting](#7-common-pitfalls-and-troubleshooting)

---

## 1. Quantization Formats

### 1.1 4-bit Quantization

#### NF4 (Normal Float 4-bit) - Recommended

NF4 is specifically designed for normally distributed weights, which is typical in neural networks.

**Characteristics:**
- Optimal for weights with Gaussian distribution
- Better preservation of outliers than uniform quantization
- Recommended default for QLoRA

**Memory Savings:**
- FP32 → NF4: 8x reduction (32 bits → 4 bits)
- BF16 → NF4: 4x reduction (16 bits → 4 bits)

```python
# NF4 quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

#### FP4 (Float Point 4-bit) - Alternative

FP4 uses a uniform quantization scheme.

**When to use FP4:**
- When weights are not normally distributed
- For specific layer types with uniform weight distributions
- As a fallback if NF4 causes issues

```python
# FP4 quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Float Point 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

#### Double Quantization

Double quantization applies quantization to the quantization constants themselves, providing additional memory savings.

**Benefits:**
- ~0.4 bits/parameter additional savings
- Minimal impact on model quality
- Recommended for memory-constrained environments

```python
# Double quantization enabled
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Enable double quantization
)
```

### 1.2 8-bit Quantization (INT8)

#### LLM.int8() Method

8-bit quantization with mixed-precision decomposition for outliers.

**Characteristics:**
- Higher precision than 4-bit
- Automatic handling of outlier features
- Larger memory footprint than 4-bit

**When to prefer 8-bit over 4-bit:**
- When model quality is critical and memory allows
- For models sensitive to quantization (smaller models)
- When 4-bit shows quality degradation
- On older NVIDIA GPUs without good 4-bit support

```python
# 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Threshold for outlier detection
)
```

### 1.3 Format Comparison Table

| Format | Bits | Memory Reduction | Quality Impact | Hardware Support | Use Case |
|--------|------|------------------|----------------|------------------|----------|
| NF4 | 4 | 8x from FP32 | Low | CUDA (Ampere+) | Default for QLoRA |
| FP4 | 4 | 8x from FP32 | Low-Medium | CUDA (Ampere+) | Alternative to NF4 |
| NF4 + Double Quant | ~3.6 | ~9x from FP32 | Low | CUDA (Ampere+) | Memory-constrained |
| INT8 | 8 | 4x from FP32 | Very Low | CUDA (all), limited MPS | Quality-critical |
| BF16 | 16 | 2x from FP32 | Negligible | CUDA, MPS | No quantization needed |
| FP16 | 16 | 2x from FP32 | Negligible | CUDA | Legacy support |

---

## 2. Layer-Specific Quantization Strategy

### 2.1 Layers to QUANTIZE (Base Model)

These layers contain the bulk of parameters and benefit most from quantization:

#### Attention Layers
```
q_proj  - Query projection      (d_model × d_model)
k_proj  - Key projection        (d_model × d_model)
v_proj  - Value projection      (d_model × d_model)
o_proj  - Output projection     (d_model × d_model)
```

**Reasoning:** Attention layers represent ~30-40% of model parameters and are well-suited for quantization due to relatively uniform weight distributions.

#### MLP/FFN Layers
```
gate_proj / fc1  - Gate/First projection    (d_model × d_ff)
up_proj          - Up projection            (d_model × d_ff)
down_proj / fc2  - Down projection          (d_ff × d_model)
```

**Reasoning:** MLP layers represent ~60% of model parameters. The intermediate dimension (d_ff) is typically 4x the model dimension, making these the largest layers.

### 2.2 Layers to KEEP in Full Precision

#### Embedding Layers (CRITICAL)

```
embed_tokens  - Input embeddings     (vocab_size × d_model)
lm_head       - Output projection    (d_model × vocab_size)
```

**Why NOT quantize:**
1. **First/last layer sensitivity**: Input and output layers are most sensitive to quantization error
2. **Discrete token mapping**: Embeddings map discrete tokens to continuous space; quantization can corrupt this mapping
3. **Gradient stability**: During training with LoRA, embeddings may receive gradients; keeping them precise ensures stability
4. **Tied weights**: Many models tie embed_tokens and lm_head; quantizing affects both

#### Normalization Layers (CRITICAL)

```
input_layernorm   - Pre-attention norm
post_attention_layernorm - Post-attention norm
ln_f / norm       - Final layer norm
```

**Why NOT quantize:**
1. **Small parameter count**: Only 2 × d_model parameters per layer (negligible memory savings)
2. **Critical for stability**: Normalization layers control activation magnitudes
3. **High sensitivity**: Small changes in norm parameters cause large output changes

#### LoRA Adapter Weights (NEVER QUANTIZE)

```
lora_A  - Down projection (d_model × r)
lora_B  - Up projection   (r × d_model)
```

**Why NEVER quantize:**
1. **Active training**: LoRA adapters receive gradient updates during training
2. **Gradient precision**: Quantized weights cannot receive precise gradient updates
3. **Small footprint**: LoRA adapters are already low-rank, minimal memory benefit
4. **This is the core QLoRA principle**: Quantized base + full-precision adapters

### 2.3 Layer Quantization Summary

```
Model Architecture (Transformer Block):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ embed_tokens          [FULL PRECISION - FP32/BF16]  │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│  ╔═══════════════════════════════════════════════════════╗ │
│  ║              Transformer Block (×N)                   ║ │
│  ║  ┌─────────────────────────────────────────────────┐  ║ │
│  ║  │ input_layernorm    [FULL PRECISION]             │  ║ │
│  ║  └─────────────────────────────────────────────────┘  ║ │
│  ║  ┌─────────────────────────────────────────────────┐  ║ │
│  ║  │ q_proj, k_proj, v_proj, o_proj  [QUANTIZED]     │  ║ │
│  ║  │ + LoRA adapters                 [FULL PRECISION]│  ║ │
│  ║  └─────────────────────────────────────────────────┘  ║ │
│  ║  ┌─────────────────────────────────────────────────┐  ║ │
│  ║  │ post_attention_layernorm [FULL PRECISION]       │  ║ │
│  ║  └─────────────────────────────────────────────────┘  ║ │
│  ║  ┌─────────────────────────────────────────────────┐  ║ │
│  ║  │ gate_proj, up_proj, down_proj   [QUANTIZED]     │  ║ │
│  ║  │ + LoRA adapters                 [FULL PRECISION]│  ║ │
│  ║  └─────────────────────────────────────────────────┘  ║ │
│  ╚═══════════════════════════════════════════════════════╝ │
│                            │                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ln_f / norm           [FULL PRECISION]              │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ lm_head               [FULL PRECISION - FP32/BF16]  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Training Pipeline Stages

### Stage 1: Model Loading with Quantization

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

**What happens:**
1. Model weights are loaded from disk
2. Linear layers are quantized to 4-bit NF4
3. Quantization constants are stored alongside weights
4. Embeddings and norms remain in original precision

### Stage 2: Prepare for k-bit Training

```python
from peft import prepare_model_for_kbit_training

# Prepare model for training with quantized weights
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

**What happens:**
1. Enables gradient computation for quantized layers
2. Sets up gradient checkpointing (memory optimization)
3. Freezes base model parameters
4. Prepares model to accept LoRA adapters

### Stage 3: Apply LoRA Adapters

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to model
model = get_peft_model(model, peft_config)
```

**What happens:**
1. LoRA adapters (A and B matrices) are added to target modules
2. Adapters are initialized (A: random, B: zeros typically)
3. Only adapter parameters are trainable
4. Adapters remain in full precision (FP32/BF16)

### Stage 4: Training Loop

```python
# Forward pass
outputs = model(input_ids, labels=labels)
loss = outputs.loss

# Backward pass
loss.backward()

# Optimizer step (only updates LoRA parameters)
optimizer.step()
```

**What happens during forward pass:**
1. Inputs pass through quantized base model
2. For each target layer: `output = W_quantized @ x + (B @ A) @ x`
3. Computation happens in compute_dtype (BF16)

**What happens during backward pass:**
1. Gradients computed for LoRA adapters only
2. Base model gradients are not computed (frozen)
3. Adapter gradients are in full precision

### Stage 5: Inference Options

#### Option A: Keep Quantized Base + LoRA (Recommended)

```python
# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "base_model",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "path/to/adapters")

# Generate
outputs = model.generate(input_ids, max_new_tokens=100)
```

**Pros:** Memory efficient, no additional processing needed
**Cons:** Requires bitsandbytes at inference time

#### Option B: Merge and Re-quantize

```python
# Load in full precision
model = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(model, "path/to/adapters")

# Merge adapters into base model
model = model.merge_and_unload()

# Re-quantize for deployment (optional)
# Can use GGUF, AWQ, GPTQ for inference-optimized quantization
```

**Pros:** No PEFT dependency at inference, can use different quantization
**Cons:** Requires full precision load temporarily, extra processing

---

## 4. Hardware-Specific Considerations

### 4.1 NVIDIA CUDA (Ampere and newer - RTX 30xx/40xx, A100, H100)

**Recommended Configuration:**
```yaml
quantization:
  enabled: true
  bits: 4
  quant_type: "nf4"
  compute_dtype: "bfloat16"
  double_quant: true
```

**Notes:**
- Full support for 4-bit quantization
- BF16 compute dtype recommended (native support)
- Flash Attention 2 compatible
- Best performance/quality trade-off

### 4.2 NVIDIA CUDA (Pre-Ampere - RTX 20xx, V100, T4)

**Recommended Configuration:**
```yaml
quantization:
  enabled: true
  bits: 8  # 8-bit recommended
  compute_dtype: "float16"  # BF16 not natively supported
```

**Notes:**
- 4-bit quantization has limited support
- Use FP16 compute dtype (no native BF16)
- 8-bit INT8 works well
- May need to disable Flash Attention

### 4.3 Apple Silicon (M1/M2/M3)

**Recommended Configuration:**
```yaml
quantization:
  enabled: false  # Disable quantization
model:
  torch_dtype: "bfloat16"
  device_map: "mps"
  attn_implementation: "eager"  # No Flash Attention on MPS
```

**Notes:**
- bitsandbytes has very limited MPS support
- Use BF16 without quantization instead
- M3 Pro Max (96GB) can fit 7B models in BF16
- Avoid `device_map="auto"`, explicitly use "mps"

### 4.4 Google Colab (T4 GPU)

**Recommended Configuration:**
```yaml
quantization:
  enabled: true
  bits: 4
  quant_type: "nf4"
  compute_dtype: "float16"  # T4 doesn't have native BF16
  double_quant: true  # Important for 16GB limit
```

**Notes:**
- T4 has 16GB VRAM
- Double quantization helps fit larger models
- Use FP16 compute dtype
- Gradient checkpointing essential

### 4.5 CPU Only

**Configuration:**
```yaml
quantization:
  enabled: false
model:
  device_map: "cpu"
  torch_dtype: "float32"
```

**Notes:**
- bitsandbytes quantization is GPU-only
- Training on CPU is very slow (not recommended)
- For inference, use GGUF/GGML formats with llama.cpp

### 4.6 Hardware Detection Logic

```python
def detect_hardware():
    """Detect available hardware and return recommended config."""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:  # Ampere or newer
            return "cuda_ampere"
        else:
            return "cuda_pre_ampere"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

---

## 5. End-to-End Validation Checklist

This section addresses **Issue #333: quantization formats are guaranteed supported end-to-end**.

### 5.1 Pre-Training Validation

Run these checks before starting training:

#### Check 1: Model Loading
```bash
python validate_quantization.py --check loading
```

Verifies:
- [ ] Model loads with specified quantization config
- [ ] No OOM errors during loading
- [ ] Correct layers are quantized
- [ ] Embeddings and norms are in full precision

#### Check 2: Memory Profile
```bash
python validate_quantization.py --check memory
```

Verifies:
- [ ] Memory usage matches expected reduction
- [ ] No memory leaks during forward pass
- [ ] Gradient checkpointing works correctly

#### Check 3: LoRA Application
```bash
python validate_quantization.py --check lora
```

Verifies:
- [ ] LoRA adapters applied to correct modules
- [ ] Adapter parameters are trainable
- [ ] Base model parameters are frozen
- [ ] Adapter dtype is correct (not quantized)

### 5.2 Training Validation

#### Check 4: Gradient Flow
```bash
python validate_quantization.py --check gradients
```

Verifies:
- [ ] Gradients flow to LoRA adapters
- [ ] No NaN/Inf gradients
- [ ] Base model gradients are None or zero
- [ ] Gradient magnitudes are reasonable

#### Check 5: Training Step
```bash
python validate_quantization.py --check training_step
```

Verifies:
- [ ] Forward pass completes without error
- [ ] Loss is finite
- [ ] Backward pass completes
- [ ] Optimizer step updates only LoRA params

### 5.3 Post-Training Validation

#### Check 6: Checkpoint Saving/Loading
```bash
python validate_quantization.py --check checkpoint
```

Verifies:
- [ ] Adapter checkpoint saves correctly
- [ ] Checkpoint loads on same hardware
- [ ] Loaded model produces same outputs

#### Check 7: Inference
```bash
python validate_quantization.py --check inference
```

Verifies:
- [ ] Model generates coherent text
- [ ] No NaN/Inf in logits
- [ ] Generation completes without errors

#### Check 8: Cross-Hardware Compatibility
```bash
python validate_quantization.py --check compatibility
```

Verifies:
- [ ] Adapters can load on different GPU
- [ ] Adapters can load on CPU
- [ ] Base model can be swapped (same architecture)

### 5.4 Reproducibility Validation

#### Check 9: Seed Reproducibility
```bash
python validate_quantization.py --check reproducibility
```

Verifies:
- [ ] Same seed produces same loss values
- [ ] Same seed produces same generations
- [ ] Results reproducible across runs

### 5.5 Format Compatibility Matrix

| Operation | NF4 | FP4 | INT8 | BF16 |
|-----------|-----|-----|------|------|
| Load model | Yes | Yes | Yes | Yes |
| Apply LoRA | Yes | Yes | Yes | Yes |
| Train (CUDA Ampere+) | Yes | Yes | Yes | Yes |
| Train (CUDA Pre-Ampere) | Limited | Limited | Yes | Yes |
| Train (MPS) | No | No | No | Yes |
| Save adapters | Yes | Yes | Yes | Yes |
| Load adapters | Yes | Yes | Yes | Yes |
| Merge adapters | Yes | Yes | Yes | Yes |
| GGUF export | Yes* | Yes* | Yes* | Yes |

*After merging to full precision

---

## 6. Code Templates

### 6.1 Complete QLoRA Setup

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_qlora_model(
    model_name: str,
    quantization_bits: int = 4,
    lora_r: int = 64,
    lora_alpha: int = 128,
    target_modules: list = None,
):
    """
    Set up a model for QLoRA training.
    
    Args:
        model_name: HuggingFace model name or path
        quantization_bits: 4 or 8 (0 for no quantization)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        target_modules: List of modules to apply LoRA to
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Default target modules for common architectures
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    # Configure quantization
    bnb_config = None
    if quantization_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
    )
    
    # Prepare for k-bit training if quantized
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model, tokenizer
```

### 6.2 Training with TRL SFTTrainer

```python
from trl import SFTTrainer, SFTConfig

def train_sft(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    output_dir="./outputs",
    max_steps=1000,
    learning_rate=2e-5,
):
    """Train with SFT using TRL."""
    
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=100 if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        gradient_checkpointing=True,
        max_seq_length=512,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    trainer.save_model()
    
    return trainer
```

### 6.3 Training with TRL GRPOTrainer

```python
from trl import GRPOTrainer, GRPOConfig

def train_grpo(
    model,
    tokenizer,
    train_dataset,
    reward_func,
    output_dir="./outputs",
    max_steps=500,
    num_generations=4,
):
    """Train with GRPO using TRL."""
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_generations=num_generations,
        max_completion_length=256,
        max_prompt_length=512,
        temperature=0.7,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        gradient_checkpointing=True,
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
    )
    
    trainer.train()
    trainer.save_model()
    
    return trainer
```

---

## 7. Common Pitfalls and Troubleshooting

### 7.1 Why Certain Layers Should Not Be Quantized

| Layer | Why Not Quantize | Impact if Quantized |
|-------|------------------|---------------------|
| embed_tokens | First layer, maps tokens to embeddings | Corrupted token representations |
| lm_head | Last layer, maps to vocabulary | Wrong token predictions |
| LayerNorm | Stability-critical, few params | Training instability, NaN loss |
| LoRA adapters | Need gradient updates | Cannot train |

### 7.2 Handling NaN/Inf During Training

**Symptoms:**
- Loss becomes NaN
- Gradients contain Inf values
- Model outputs are all NaN

**Solutions:**

1. **Check compute dtype**
```python
# Use BF16 instead of FP16 for phi-2 and similar models
bnb_4bit_compute_dtype=torch.bfloat16
```

2. **Reduce learning rate**
```python
learning_rate=1e-5  # Start low
```

3. **Enable gradient clipping**
```python
max_grad_norm=1.0
```

4. **Check for problematic samples**
```python
# Filter out very long sequences
max_seq_length=512
```

### 7.3 Memory Fragmentation Issues

**Symptoms:**
- OOM errors despite having enough total memory
- Memory usage increases over time

**Solutions:**

1. **Set memory fraction**
```python
torch.cuda.set_per_process_memory_fraction(0.9)
```

2. **Clear cache periodically**
```python
torch.cuda.empty_cache()
```

3. **Use gradient checkpointing**
```python
gradient_checkpointing=True
```

4. **Reduce batch size**
```python
per_device_train_batch_size=1
gradient_accumulation_steps=8  # Maintain effective batch size
```

### 7.4 Checkpoint Compatibility Issues

**Problem:** Checkpoint trained on one machine won't load on another.

**Solutions:**

1. **Save adapter-only checkpoints**
```python
model.save_pretrained("adapter_path")  # Saves only LoRA weights
```

2. **Use safetensors format**
```python
model.save_pretrained("adapter_path", safe_serialization=True)
```

3. **Document base model version**
```yaml
# checkpoint_info.yaml
base_model: "microsoft/phi-2"
base_model_revision: "main"
quantization: "4bit-nf4"
```

### 7.5 Common Error Messages and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Model too large | Enable double_quant, reduce batch size |
| `Expected all tensors to be on the same device` | Mixed device placement | Use device_map="auto" |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Wrong target modules | Check model architecture |
| `ValueError: Target modules not found` | Incorrect module names | Print model structure, verify names |
| `bitsandbytes not found` | Missing dependency | `pip install bitsandbytes` |
| `MPS backend doesn't support quantization` | Using bnb on Apple Silicon | Disable quantization for MPS |

---

## References

1. **QLoRA Paper**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
2. **GRPO Paper**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
3. **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
4. **bitsandbytes**: [GitHub - TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
5. **PEFT Documentation**: [HuggingFace PEFT](https://huggingface.co/docs/peft)
6. **TRL Documentation**: [HuggingFace TRL](https://huggingface.co/docs/trl)

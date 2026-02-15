# Training Pipeline — What Happens Offline vs Online?

## Question 1: Can we pre-compute Embeddings, Positional Embeddings, and Multi-Head Attention offline along with tokenization?

**No.** Tokenization is the only step that can be pre-computed offline. Everything after tokenization involves learnable parameters that change at every training step.

### What Can vs Cannot Be Pre-computed

```
┌──────────────────────────────────────────────────────────────────────┐
│                    THE TRANSFORMER PIPELINE                          │
│                                                                      │
│  ┌─────────────┐                                                     │
│  │ Tokenization │  Text → Token IDs (integers)                      │
│  │              │  "Hello world" → [15496, 995]                     │
│  └──────┬───────┘                                                    │
│         │                                                            │
│    ✅ CAN pre-compute offline                                        │
│    • Output is FIXED integers that never change                     │
│    • Independent of model weights                                    │
│    • Independent of training state                                   │
│                                                                      │
│  ════════════════════  WALL  ═══════════════════════════════         │
│                                                                      │
│    ❌ CANNOT pre-compute offline (everything below)                  │
│                                                                      │
│         │                                                            │
│  ┌──────▼───────┐                                                    │
│  │  Embedding    │  Token IDs → Dense Vectors                       │
│  │  Lookup       │  [15496] → [0.12, -0.34, 0.56, ...]            │
│  └──────┬───────┘                                                    │
│         │                                                            │
│  ┌──────▼────────────┐                                               │
│  │  Positional        │  Add position information                    │
│  │  Embedding         │  embed + pos_embed → positioned vectors     │
│  └──────┬────────────┘                                               │
│         │                                                            │
│  ┌──────▼────────────┐                                               │
│  │  Multi-Head        │  Self-attention across all positions         │
│  │  Attention         │  Q, K, V projections + attention scores     │
│  └──────┬────────────┘                                               │
│         │                                                            │
│  ┌──────▼───────┐                                                    │
│  │  FFN + Norm   │  Feed-forward + layer norm                       │
│  └──────┬───────┘                                                    │
│         │                                                            │
│  ┌──────▼───────┐                                                    │
│  │  Loss + Back- │  Compute loss, backpropagate gradients           │
│  │  propagation  │  Update ALL weights above                        │
│  └──────────────┘                                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Why Each Layer Cannot Be Pre-computed

#### Embedding Lookup — ❌ Cannot pre-compute

```python
# Embedding is a LEARNABLE weight matrix
self.embedding = nn.Embedding(vocab_size=50280, embed_dim=4096)
#                              ↑ this is a 50280 × 4096 matrix of WEIGHTS

# During training:
#   Step 0:    token 15496 → [0.12, -0.34, 0.56, ...]
#   Step 1000: token 15496 → [0.08, -0.41, 0.63, ...]  ← CHANGED!
#   Step 5000: token 15496 → [-0.15, 0.22, 0.91, ...]  ← CHANGED AGAIN!
```

**Why**: The embedding matrix is **updated by gradient descent at every training step**. The vector for token `15496` at step 0 is completely different from the vector at step 5000. Pre-computed embeddings would be stale after the first gradient update.

#### Positional Embedding — ❌ Cannot pre-compute

| Type | Why it can't be pre-computed |
|------|------------------------------|
| **Learned positional embeddings** (GPT-2 style) | Same reason as token embeddings — weights change every step |
| **Fixed positional encodings** (sinusoidal / RoPE) | The positional encoding itself is fixed, BUT it's **added to the token embedding** which changes every step. RoPE is applied as a rotation to Q and K *inside* attention — it can't be separated from the model forward pass. |

#### Multi-Head Attention — ❌ Cannot pre-compute

```python
# Attention involves FOUR learnable weight matrices per layer:
W_Q = nn.Linear(4096, 4096)  # Query projection   ← WEIGHTS CHANGE
W_K = nn.Linear(4096, 4096)  # Key projection     ← WEIGHTS CHANGE
W_V = nn.Linear(4096, 4096)  # Value projection   ← WEIGHTS CHANGE
W_O = nn.Linear(4096, 4096)  # Output projection  ← WEIGHTS CHANGE
```

**Why**:
1. The Q, K, V, O projection matrices are **learnable weights** that change every step
2. Attention is **context-dependent** — the output for each token depends on *all other tokens in the same sequence*
3. During training, gradients flow back through attention to update embeddings

### The Fundamental Reason: Backpropagation

```
Forward pass:  Token IDs → Embedding → Pos Embed → Attention → FFN → Loss
                               ↑            ↑           ↑         ↑
Backward pass: Gradients ←────┘────────────┘───────────┘─────────┘
               update ALL weight matrices
```

**Training = forward pass + backward pass.** During backpropagation, gradients flow backwards through *every layer* and update the weights in *every layer*. If you pre-computed any layer's output:

1. The pre-computed values would be **stale** after the first weight update
2. Gradients could not flow through the pre-computed layer → the layers *before* it would **stop learning**
3. The model would effectively be **frozen** up to that point

### What About Inference/Feature Extraction?

| Scenario | Can pre-compute? | Why |
|----------|:-:|-----|
| **Pre-training an LLM** | ❌ | Weights change every step |
| **Fine-tuning** (full) | ❌ | All weights still change |
| **Fine-tuning** (LoRA/frozen backbone) | ⚠️ Partially | *Could* cache frozen layer outputs, but attention still depends on full sequence context — not practical |
| **Inference** (after training is done) | ✅ KV-Cache | This is what KV-cache does — caches computed K and V for already-seen tokens |
| **Feature extraction** (using a trained model) | ✅ | Freeze model, pre-compute embeddings for downstream tasks |

---

## Question 2: Do we need to tokenize in real-time during training? Apart from tokenization, what else should we do as part of training?

### Answer: No real-time tokenization needed

**No.** We already pre-tokenized offline into `.npy` shards. During training, the data is already integer token IDs sitting in `.npy` files on NVMe. The training loop never sees raw text.

```
What training receives:     [15496, 995, 314, 42, 8901, ...]   ← integers, ready to go
What training does NOT do:  "Hello world" → tokenizer → [15496, 995]   ← this was done offline
```

### Complete Picture: Everything That Happens During Training

```
╔══════════════════════════════════════════════════════════════════════╗
║                    OFFLINE — Done Once, Before Training             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Data Collection        Dolma + Sangraha + NCERT + IndicNLP      ║
║  2. Quality Filtering      Remove low-quality, dedup                ║
║  3. Tokenization           Text → token IDs (uint32)                ║
║  4. Sharding               Token stream → fixed-size .npy shards   ║
║  5. Upload to S3           Shards ready for training                ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                    ONLINE — Every Training Step on GPU               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  A. DATA LOADING (CPU + DMA)                                        ║
║     ├─ S3 → NVMe staging (background thread)                       ║
║     ├─ mmap read from NVMe (zero-copy)                              ║
║     ├─ Batch collation (stack sequences into a batch)               ║
║     ├─ Pin memory                                                    ║
║     └─ H2D transfer (CPU → GPU, async on CUDA stream)              ║
║                                                                      ║
║  B. FORWARD PASS (GPU)                                               ║
║     ├─ Embedding lookup          token IDs → dense vectors          ║
║     ├─ Positional encoding       add position information           ║
║     ├─ × N Transformer layers:                                      ║
║     │    ├─ Layer Norm                                               ║
║     │    ├─ Multi-Head Attention (Q, K, V projections + softmax)    ║
║     │    ├─ Residual connection                                      ║
║     │    ├─ Layer Norm                                               ║
║     │    ├─ Feed-Forward Network (MLP)                               ║
║     │    └─ Residual connection                                      ║
║     ├─ Final Layer Norm                                              ║
║     └─ LM Head (project to vocab → logits)                          ║
║                                                                      ║
║  C. LOSS COMPUTATION (GPU)                                           ║
║     └─ Cross-entropy loss between logits and shifted labels         ║
║                                                                      ║
║  D. BACKWARD PASS (GPU)                                              ║
║     └─ Backpropagate gradients through all layers                   ║
║                                                                      ║
║  E. OPTIMIZER STEP (GPU)                                             ║
║     ├─ Gradient clipping (max norm)                                  ║
║     ├─ Adam/AdamW weight update                                      ║
║     └─ Learning rate scheduler step                                  ║
║                                                                      ║
║  F. DISTRIBUTED SYNC (GPU ↔ GPU via NVLink/EFA)                     ║
║     ├─ Gradient all-reduce (ZeRO Stage 2)                           ║
║     │   or Parameter partitioning (ZeRO Stage 3)                    ║
║     └─ Loss averaging across ranks                                   ║
║                                                                      ║
║  G. HOUSEKEEPING (CPU)                                               ║
║     ├─ Logging (loss, learning rate, throughput)                     ║
║     ├─ Checkpointing (every N steps → save to S3)                   ║
║     ├─ Evaluation (every N steps → run on eval set)                 ║
║     └─ Memory management (GC, cache clearing)                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### What Does DeepSpeed Handle vs What Do We Handle?

| Responsibility | Who Handles It | Details |
|---|---|---|
| **Data loading & batching** | **Us** (data pipeline) | `StreamingTokenDataset` → `DataLoader` → `PrefetchDataLoader` |
| **Embedding lookup** | Model (auto) | Part of `model.forward()` |
| **Positional encoding** | Model (auto) | RoPE or learned — built into model architecture |
| **Attention + FFN** | Model (auto) | Part of `model.forward()` |
| **Loss computation** | **Us** (train loop) | `loss = model_engine(input_ids, labels=labels)` |
| **Backward pass** | **DeepSpeed** (auto) | `model_engine.backward(loss)` |
| **Gradient clipping** | **DeepSpeed** (config) | Set in `deepspeed_config.json` |
| **Optimizer step** | **DeepSpeed** (auto) | `model_engine.step()` |
| **Gradient sync (multi-GPU)** | **DeepSpeed** (auto) | ZeRO handles all-reduce/partitioning |
| **Learning rate schedule** | **DeepSpeed** (config) | Warmup + cosine/linear decay |
| **Checkpointing** | **Us** (checkpoint manager) | `S3CheckpointManager.save_checkpoint()` |
| **Mixed precision (FP16/BF16)** | **DeepSpeed** (config) | Set in `deepspeed_config.json` |
| **Logging** | **Us** (train loop) | Loss, throughput, GPU utilization |

### The Actual Training Loop — What's Left for Us to Write

Given that tokenization is offline and DeepSpeed handles most of the heavy lifting, here's what our training loop actually does:

```python
# This is what our train_epoch() function does per step:

for batch in prefetch_dataloader:          # ← Data already on GPU (prefetched)

    # Step B+C: Forward pass + loss (one call)
    loss = model_engine(
        input_ids=batch["input_ids"],      # ← Pre-tokenized integers from .npy shards
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],            # ← input_ids shifted by 1
    ).loss

    # Step D: Backward pass
    model_engine.backward(loss)            # ← DeepSpeed handles gradient sync

    # Step E: Optimizer step
    model_engine.step()                    # ← DeepSpeed handles optimizer + LR schedule

    # Step G: Housekeeping
    if step % log_interval == 0:
        log_metrics(loss, throughput, lr)

    if step % checkpoint_interval == 0:
        save_checkpoint(model_engine, shard_progress)
```

**Four lines of core training logic.** Everything else is handled by:
- **Pre-tokenization** (offline) — no text processing during training
- **Data pipeline** (our code) — shards → mmap → prefetch → GPU
- **DeepSpeed** (framework) — gradients, optimizer, distributed sync, mixed precision

### Operational Things Needed During Training

Apart from the model forward/backward, there are critical **operational** concerns:

| What | When | Why |
|------|------|-----|
| **Learning rate warmup** | First ~2000 steps | Prevents training instability at start |
| **Gradient clipping** | Every step | Prevents exploding gradients |
| **Loss spike detection** | Every step | Alert if loss suddenly jumps (data corruption or bug) |
| **GPU memory monitoring** | Periodically | Detect OOM before it crashes |
| **Checkpoint saving** | Every N steps | Fault recovery for spot instances |
| **Evaluation runs** | Every N steps | Track validation loss / perplexity |
| **Throughput tracking** | Every step | Ensure GPUs aren't being starved for data |
| **Shard progress tracking** | Every step | For exact checkpoint/resume |

All of these are already handled in our codebase — `train_epoch()` in `src/train.py` and `S3CheckpointManager` in `src/checkpoint.py` do the housekeeping, and DeepSpeed handles gradient clipping, LR scheduling, and mixed precision through config.

---

## Summary

```
What can be pre-computed offline for training:

  ✅ Tokenization       →  Text → integer IDs (fixed, no learned weights)
  ✅ Data quality filtering, dedup, mixing, sharding

  ❌ Embedding lookup   →  Learned weights, change every training step
  ❌ Positional embed   →  Either learned (changes) or applied during forward pass
  ❌ Multi-Head Attn    →  Learned weights + context-dependent + needs gradients
  ❌ FFN layers         →  Learned weights + needs gradients
  ❌ Layer norm         →  Learned weights + needs gradients
```

**The tokenization step is the natural and only boundary** between "offline pre-processing" and "online training." Everything after tokenization involves learnable parameters that are updated by gradient descent, so it *must* happen on the GPU during training.

The pipeline:

```
OFFLINE (CPU):  Raw text → Tokenize → Shard → S3
ONLINE (GPU):   S3 → NVMe → mmap → [Embedding → Attention → FFN → Loss → Backprop]
                                     └────────── ALL on GPU, every step ──────────┘
```

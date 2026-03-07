# Experiment 11: Model Growth & Weight Transfer (Dense → MoE)

This experiment investigates how to initialize a **3B Mixture-of-Experts (MoE) model** from the weights of a pre-trained **1B Dense model** without suffering a training loss spike.

The core idea is to:
1. Copy all non-FFN weights directly (embeddings, attention, norms, LM head).
2. Copy the 1B dense FFN to the **shared expert** of each MoE layer.
3. Use **SVD-based spectral compression** to create diverse **routed experts** at a smaller intermediate dimension.
4. Apply **structured rotation noise** to break symmetry between routed experts.
5. Set **router biases** to suppress or enable routing at initialization.

---
R
## Directory Structure

```
experiments/11_growth_and_weight_transfer/
├── spectral_moe_initializer.py   # Core 1B→3B weight transfer engine
├── README.md                     # This file
├── logs/                         # Training and evaluation logs
└── ablation/
    ├── __init__.py
    ├── common.py                  # Shared utilities (device, data, logging)
    ├── smart_scale.py             # Alternative: raw tensor-level model scaling
    ├── moe_diagnostics.py         # MoE routing diagnostics tools
    ├── init_exp1_null_routing.py  # Exp 1: weight copy + forced null routing
    ├── init_exp2_svd_null_routing.py  # Exp 2: SVD init + forced null routing
    ├── init_exp3_full_clone_active.py # Exp 3: full-dim clone + active routing
    ├── init_exp4_svd_active.py    # Exp 4: SVD compression + active routing
    ├── eval_exp1_equivalence.py   # Eval: verify 3B ≈ 1B with null routing
    ├── eval_exp2_svd_equivalence.py # Eval: verify SVD doesn't corrupt shared expert
    ├── train_dense_1b_baseline.py # Phase 1: train the 1B dense baseline
    ├── train_exp3_full_clone.py   # Train Exp 3 (no SVD compression)
    └── train_exp4_svd_moe.py      # Train Exp 4 (full dense-to-MoE pipeline)
```

---

## Required File Dependencies

All scripts are designed to run from the `endGame/` parent directory (one level above `experiments/`). The following files and directories must exist at runtime:

### Model Definitions

| File | Location | Description |
|------|----------|-------------|
| `recurrence_model_1b.py` | `endGame/` | 1B dense model class (`Model1B`, `ModelConfig`, `KroneckerEmbeddings`, `KroneckerConfig`) |
| `recurrence_model_3b.py` | `endGame/` | 3B MoE model class (`Model3B`, `ModelConfig`) |

### Supporting Code

| File | Location | Description |
|------|----------|-------------|
| `data_utils.py` | `endGame/` | `SYNTHStream` dataset class |
| `training.py` | `endGame/` | `save_checkpoint`, `load_checkpoint`, `set_moe_freeze_state` |
| `tokenizer.json` | `endGame/` | Pre-trained BPE tokenizer (loaded by `common.py`) |

### Dataset

| Path | Description |
|------|-------------|
| `endGame/../synth_local_en/` | Local copy of `PleIAs/SYNTH` dataset (English, used by `SYNTHStream`) |

The `SYNTHStream` in `common.py` is configured as:
```python
SYNTHStream(
    dataset_name="PleIAs/SYNTH",
    local_path="../synth_local_en",   # relative to endGame/
    seq_len=64
)
```

### Checkpoints

All checkpoint paths are relative to `endGame/`. They must exist **before** running each script:

| Checkpoint Path | Created By | Required By |
|-----------------|------------|-------------|
| `endGame/checkpoints/dense_1b_baseline/kronecker_latest.pt` | `train_dense_1b_baseline.py` | All init & eval scripts |
| `endGame/checkpoints/exp1_null_routing/init.pt` | `init_exp1_null_routing.py` | `eval_exp1_equivalence.py` |
| `endGame/checkpoints/exp2_svd_null/init.pt` | `init_exp2_svd_null_routing.py` | `eval_exp2_svd_equivalence.py` |
| `endGame/checkpoints/exp3_full_clone/init.pt` | `init_exp3_full_clone_active.py` | `train_exp3_full_clone.py` |
| `endGame/checkpoints/exp4_svd_moe/init.pt` | `init_exp4_svd_active.py` | `train_exp4_svd_moe.py` |

---

## Execution Order

Run all commands from the `endGame/` directory:

```bash
cd endGame/

# Phase 1: Train the 1B dense baseline (must complete first)
python -m ablation.train_dense_1b_baseline

# Phase 2: Ablation experiments (independent, can run in any order after Phase 1)

# Experiment 1: Architecture equivalence check
python -m ablation.init_exp1_null_routing
python -m ablation.eval_exp1_equivalence

# Experiment 2: SVD + null routing equivalence check
python -m ablation.init_exp2_svd_null_routing
python -m ablation.eval_exp2_svd_equivalence

# Experiment 3: Full-dimension clone + active routing
python -m ablation.init_exp3_full_clone_active
python -m ablation.train_exp3_full_clone

# Experiment 4: SVD compression + active routing (production pipeline)
python -m ablation.init_exp4_svd_active
python -m ablation.train_exp4_svd_moe
```

---

## Ablation Experiments Summary

| Exp | SVD Compression | Routing at Init | Purpose |
|-----|-----------------|-----------------|---------|
| 1   | No (simple copy) | Forced Null   | Verify 3B + null routing == 1B (architecture sanity) |
| 2   | Yes (1024→512)  | Forced Null    | Verify SVD init doesn't corrupt shared expert |
| 3   | No (1024 full)  | Active (0.0/0.0) | Isolate routing effect without compression |
| 4   | Yes (1024→512)  | Active (-1.0/+1.0) | Full dense-to-MoE production pipeline |

---

## Key Config Values

These values come from `ModelConfig` in `recurrence_model_1b.py` and `recurrence_model_3b.py`:

| Parameter | 1B Dense | 3B MoE |
|-----------|----------|--------|
| `hidden_size` | 512 | 512 |
| `num_layers` | varies | varies |
| `shared_expert_intermediate_size` | 1024 (2x hidden) | 1024 (copied from 1B) |
| `expert_intermediate_size` | — | 512 (default, SVD-compressed) |
| `num_real_experts` | — | 8 (routed) |
| SVD compression ratio | — | 1024 → 512 (2×) |
| Rotation epsilon | — | 0.005 |

---

## File-by-File Description

### `spectral_moe_initializer.py`

The core weight-transfer engine. See the detailed section below.

### `ablation/common.py`

**Shared utilities** used by all experiment scripts:

- `detect_device()` — Detects MPS (Apple Silicon) > CUDA > CPU.
- `load_tokenizer()` — Loads `tokenizer.json` from `endGame/` using HuggingFace `PreTrainedTokenizerFast`. Also decodes all token IDs into a `bpe_vocab` list.
- `create_kronecker_codec(vocab_size)` — Creates a `KroneckerEmbeddings` codec (256×32 = 8192 dims) used by both models.
- `create_1b_model(device, bpe_vocab, pf_codec)` — Instantiates `Model1B` with `ModelConfig` defaults.
- `create_3b_model(device, bpe_vocab, pf_codec, config_overrides)` — Instantiates `Model3B`; supports overriding config fields (e.g., setting `expert_intermediate_size=1024` for Exp 3).
- `create_data_loader(tokenizer, seq_len, batch_size, start_step)` — Wraps `SYNTHStream` in a `DataLoader` with `seed=42` for reproducibility.
- `get_reference_batch(tokenizer, device)` — Returns a single fixed batch `(x_input, y_ntp, y_mtp)` at `start_step=0` for deterministic evaluation comparisons across experiments.
- `prepare_inputs(input_ids)` — Splits `input_ids` into `x=input[:-2]`, `y_ntp=input[1:-1]`, `y_mtp=input[2:]` (NTP + MTP targets).
- `compute_losses(logits_ntp, logits_mtp, y_ntp, y_mtp, aux_loss)` — Returns `total = NTP + 0.3*MTP + aux`.
- `compute_moe_metrics(model)` — Extracts `null_rate` from `model.layers[*].mlp_block.sublayer.moe.last_indices`.
- `setup_logging(log_path)` — Creates a dual console+file logger.
- `log_step_moe(...)` — Logs step metrics including `null_rate` for MoE experiments.
- `force_null_routing(model, logit_bias=-100.0, null_logit=100.0)` — Forces all MoE gates to select null experts (used in Exp 1 and 2).
- `set_active_routing_bias(model, logit_bias, null_logit)` — Sets moderate routing bias for active-but-null-biased routing (used in Exp 3 and 4).
- `random_small_rotation(dim, eps, device)` — Generates a small orthogonal rotation matrix via skew-symmetric matrix exponential (used in Exp 3).

**Environment variables set at import:**
```python
PYTORCH_MPS_HIGH_WATERMARK_RATIO = "1.0"
PYTORCH_MPS_LOW_WATERMARK_RATIO  = "0.9"
PYTORCH_MPS_PREFER_METAL         = "1"
```

---

### `ablation/train_dense_1b_baseline.py`

**Purpose:** Train a clean 1B dense baseline that serves as the source checkpoint for all MoE initialization experiments.

**Config:**
| Parameter | Value |
|-----------|-------|
| `NUM_UPDATES` | 1000 |
| `BATCH_SIZE` | 4 (physical) |
| `ACCUM_STEPS` | 8 → effective batch = 32 |
| `SEQ_LEN` | 64 |
| `LR_MAX` | 3e-4 |
| `WARMUP_UPDATES` | 100 (cosine schedule) |
| `GRAD_CLIP` | 1.0 |
| `CHECKPOINT_INTERVAL` | 100 steps |

**Output:** `checkpoints/dense_1b_baseline/kronecker_latest.pt`  
**Log:** `logs/dense_1b_baseline.log`

---

### `ablation/init_exp1_null_routing.py`

**Purpose:** Initialize 3B MoE from 1B weights, then **force-null all routing** (biases set to `logit_bias=-100, null_logit=+100`). No SVD compression — routed experts are SVD-initialized but will never fire.

**Expected behavior at eval:** `loss_3b ≈ loss_1b` (diff < 1e-4), confirming the architecture is correct.

**Output:** `checkpoints/exp1_null_routing/init.pt`

---

### `ablation/eval_exp1_equivalence.py`

**Purpose:** Load both the 1B baseline and the Exp 1 initialized 3B model and compare forward-pass outputs on a fixed reference batch.

**Pass criterion:** NTP loss diff < 1e-4, MTP loss diff < 1e-4  
**Log:** `logs/exp1_null_routing.log`

---

### `ablation/init_exp2_svd_null_routing.py`

**Purpose:** Same as Exp 1 but with **full `SpectralMoEInitializer` SVD compression** (1024→512) applied to routed experts, then routing is force-nulled afterward.

**Ablation question:** Does SVD initialization corrupt the shared expert path?  
**Output:** `checkpoints/exp2_svd_null/init.pt`

---

### `ablation/eval_exp2_svd_equivalence.py`

**Purpose:** Same equivalence test as Exp 1 but for the SVD-initialized model.

**Pass criterion:** same as Exp 1 (diff < 1e-4)  
**Log:** `logs/exp2_svd_null.log`

---

### `ablation/init_exp3_full_clone_active.py`

**Purpose:** Clone the 1B dense FFN weights to **all routed experts at full dimension** (no SVD compression, `expert_intermediate_size=1024`). Each expert gets a small rotation (`eps=0.005`) for diversity. Routing is set to **active with neutral bias** (0.0/0.0).

**Key difference from Exp 4:** No compression — all experts are the same size as the shared expert. This isolates the effect of routing from compression.

**3B model config override:** `expert_intermediate_size = 1024` (forces each W_gate to be `(8, 512, 1024)` instead of `(8, 512, 512)`).

**Output:** `checkpoints/exp3_full_clone/init.pt`

---

### `ablation/train_exp3_full_clone.py`

**Purpose:** Train the Exp 3 initialized model.

**Config:** 200 updates, LR 3e-4, effective batch 32, 20-step warmup.  
**Extra:** Runs `moe_diagnostics.run_all_diagnostics()` at init and final step, saves detailed token-expert mapping to `logs/diagnostics/`.  
**Log:** `logs/exp3_full_clone.log`

---

### `ablation/init_exp4_svd_active.py`

**Purpose:** Full production-grade dense-to-MoE initialization:
- SVD compression 1024 → 512 (`svd_mode="independent"`)
- Structured rotation noise `eps=0.005`
- Active routing bias `logit_bias=0.0, null_logit=-1.0`

**Output:** `checkpoints/exp4_svd_moe/init.pt`

---

### `ablation/train_exp4_svd_moe.py`

**Purpose:** Train the Exp 4 initialized model with an expert warmup-freeze period.

**Config:**
| Parameter | Value |
|-----------|-------|
| `NUM_UPDATES` | 500 |
| `WARMUP_UPDATES` | 50 (routed experts frozen via `set_moe_freeze_state`) |
| `LR_MAX` | 3e-4 |
| Effective batch | 32 |

**Key feature:** Calls `set_moe_freeze_state(model, step, warmup_steps=50)` each update to keep routed expert parameters frozen during warmup, allowing shared experts to stabilize first.  
**Log:** `logs/exp4_svd_moe.log`

---

### `ablation/moe_diagnostics.py`

**Purpose:** Comprehensive MoE routing diagnostics, called during training in Exp 3 and Exp 4.

**Functions:**

| Function | Description |
|----------|-------------|
| `expert_token_distribution(model)` | Per-expert token counts (real + null) for each MoE layer and in aggregate |
| `token_expert_mapping(model, input_ids, tokenizer)` | Maps decoded token text → expert assignment per layer |
| `router_entropy(model)` | Shannon entropy of average routing probabilities (requires `gate.last_probs`) |
| `routing_weight_stats(model)` | Mean/std/min/max of actual routing weights; fraction of tokens with zero real expert assignment |
| `expert_load_balance(model)` | Coefficient of variation (CV) and max/min ratio of expert token counts |
| `run_all_diagnostics(model, input_ids, tokenizer)` | Runs all five analyses and returns a consolidated report dict |
| `log_diagnostics(report, logger, step, verbose)` | Logs report to logger in human-readable format |
| `log_compact_diagnostics(model, logger, step)` | Single-line summary per step (null%, expert counts, CV, avg weight) |
| `save_detailed_diagnostics(report, save_dir, step)` | Saves complete token-expert tables to `logs/diagnostics/token_map_<step>.txt` |
| `expert_output_scales(model)` | Extracts learnable `expert_output_scale` values from all MoE layers |

**Routing health targets:**
- `null_rate`: should decrease from near 1.0 (pure null) as experts activate
- Expert cosine similarity: target 0.95–0.98 (diverse but not random)
- Load balance CV: target < 1.0 (not all tokens going to one expert)
- Router entropy: higher = more uniform routing

---

### `ablation/smart_scale.py`

**Purpose:** An alternative, **architecture-agnostic** weight transfer tool. Instead of mapping FFN layers to MoE experts, `smart_scale.py` directly expands all tensor dimensions of a checkpoint to match a larger model using FLM-101B-style initialization.

**Approach:**
- Detects `hidden_size`, `intermediate_size`, and `latent_size` automatically from checkpoint tensor shapes.
- Copies original weights into the top-left block of the expanded tensor.
- Initializes new rows/columns with Xavier-like or attention-scaled random values.
- Preserves compression ratio for latent attention dimensions.

**Usage:**
```bash
python smart_scale.py \
    --in-ckpt  checkpoints/dense_1b_baseline/kronecker_latest.pt \
    --out-ckpt checkpoints/scaled_3b/init.pt \
    --old-hidden 512 --new-hidden 768 \
    --old-intermediate 1024 --new-intermediate 1536
```

Or use `--target-params` to auto-calculate dimensions:
```bash
python smart_scale.py \
    --in-ckpt  checkpoints/dense_1b_baseline/kronecker_latest.pt \
    --out-ckpt checkpoints/scaled_3b/init.pt \
    --target-params 3000000000 \
    --keep-comp-ratio
```

**Key constraint enforced:** `new_hidden` must be divisible by `num_heads * 2 = 18` (hardcoded for 9-head attention).

**Outputs:**
- `<out-ckpt>` — scaled state dict (raw tensors, no wrapper)
- `<out-ckpt>.meta.json` — dimension mapping metadata
- `--mask-json <path>` — optional JSON describing which tensor regions were newly initialized

**Difference from `SpectralMoEInitializer`:** `smart_scale.py` grows the model in-place by expanding existing tensor dimensions; it does **not** create a new MoE architecture with separate shared/routed experts.

---

## `spectral_moe_initializer.py` — Detailed Explanation

`SpectralMoEInitializer` is the primary engine for transferring weights from a 1B dense model into the 3B MoE architecture. It is implemented as a Python class and operates in four stages:

### Constructor Parameters

```python
SpectralMoEInitializer(
    dense_model,           # Loaded Model1B instance
    moe_model,             # Loaded Model3B instance (empty or random weights)
    num_routed_experts=20, # Number of routed experts per MoE layer
    intermediate_dense=2048, # Intermediate dim in 1B FFN (gate_proj out dim)
    intermediate_moe=1024,   # Target intermediate dim for routed experts
    rotation_eps=0.02,     # Magnitude of rotation noise for expert diversity
    device="cuda",
    svd_mode="joint",      # "joint" or "independent" (see below)
)
```

---

### Stage 1: SVD Compression (`_compress_swiglu`)

This is the mathematical core of the initializer. A SwiGLU FFN has three weight matrices:
- `Wg` (gate_proj): `(intermediate_dense, hidden)` — e.g., `(1024, 512)`
- `Wu` (up_proj): `(intermediate_dense, hidden)` — e.g., `(1024, 512)`
- `Wd` (down_proj): `(hidden, intermediate_dense)` — e.g., `(512, 1024)`

The goal is to find the **top-k directions in intermediate space** that best preserve the function of all three matrices simultaneously, compressing from `intermediate_dense=1024` to `intermediate_moe=512`.

There are two SVD modes:

#### `svd_mode="joint"` (default)

1. Stack the three matrices into a joint matrix: `M = [Wg.T; Wu.T; Wd]` → shape `(3×hidden, intermediate_dense)` = `(1536, 1024)`
2. Perform SVD: `M = U S Vh` where `Vh` rows are the principal directions in intermediate space.
3. Take the top-k rows: `V_k = Vh[:k, :]` → shape `(512, 1024)`
4. Project each matrix into this shared subspace:
   - `Wg_base = V_k @ Wg` → `(512, 512)`
   - `Wu_base = V_k @ Wu` → `(512, 512)`
   - `Wd_base = Wd @ V_k.T` → `(512, 512)`

This ensures all three matrices vote equally on which directions in intermediate space are most important.

**Explained variance** is logged: `sum(S[:k]²) / sum(S²)`. A warning is raised if < 90%.

#### `svd_mode="independent"` (used in Exp 4)

1. Each matrix independently finds its best k directions via separate SVDs:
   - `Wg`: left singular vectors `U_g[:, :k]` (shape `(1024, k)`)
   - `Wu`: left singular vectors `U_u[:, :k]`
   - `Wd`: right singular vectors `Vh_d[:k, :].T` (shape `(1024, k)`)
2. Concatenate all nominees: `candidates` shape `(1024, 3k)`
3. Run a **consensus SVD** on `candidates` and take the top-k left singular vectors as the shared basis `V_k`.
4. Project all matrices using `V_k` (same final step as joint).

**Why not fully independent SVD?** SwiGLU computes `silu(x @ Wg.T) * (x @ Wu.T)` — the element-wise multiply requires gate and up activations to share the same coordinate system. Fully separate SVD bases would make the product meaningless (observed loss spike from 4 → 11). The consensus basis finds a shared subspace by letting each matrix nominate its best directions.

---

### Stage 2: Structured Rotation Noise (`_random_small_rotation`)

Each routed expert is created by applying a different small rotation to the compressed base weights:

```python
A = torch.randn(dim, dim)
A = A - A.T          # skew-symmetric matrix (A^T = -A)
R = torch.matrix_exp(rotation_eps * A)  # R is orthogonal: R^T @ R = I
```

The rotation is then applied to preserve the SwiGLU function:
```
Wg_i = R @ Wg_base      # rotate gate rows
Wu_i = R @ Wu_base      # rotate up rows
Wd_i = Wd_base @ R.T    # rotate down cols (inverse rotation)
```

**Why does this work?** The SwiGLU output is:
```
silu(x @ Wg.T @ R.T) * (x @ Wu.T @ R.T) @ R @ Wd.T
```
The `R` and `R.T` approximately cancel for small `eps`, so each expert starts computing nearly the same function as the base, but in a slightly rotated coordinate system. This breaks the symmetry that would otherwise make all experts identical, while keeping the initial loss close to the 1B baseline.

---

### Stage 3: Build Routed Experts (`_build_routed_experts`)

Calls `_compress_swiglu` once to get `(Wg_base, Wu_base, Wd_base)`, then generates `num_routed_experts` different rotation matrices and applies them. Returns a list of `(Wg_i, Wu_i, Wd_i)` tuples.

---

### Stage 4: `initialize()` — Full Model Transfer

This is the main method that orchestrates the complete weight transfer:

1. **Embeddings**: Copies `kronecker_embeddings` (or `token_embed`) and `embed_norm`.
2. **Final norm and LM head**: Direct `load_state_dict` copies.
3. **Auxiliary components**: `pf_to_model` (Kronecker projection), `memory_gate_proj`.
4. **MTP block** (Multi-Token Prediction): Copies attention, fusion, norms, then applies SVD init to the MTP's MoE FFN layer using the same pipeline.
5. **Per-layer loop** (for each transformer layer):
   - Copies the full `attn_block` (includes attention, coefficients, norms).
   - Copies `mlp_block.coeffs` and `mlp_block.norm` (but not the sublayer, since 1B has dense FFN and 3B has MoE).
   - For the MoE sublayer:
     - Copies `Wg, Wu, Wd` from the 1B dense FFN's `shared_gate/up/down` directly to the 3B's shared expert.
     - Calls `_build_routed_experts` to generate compressed+rotated copies.
     - Assigns routed expert weights (handles both batched `W_gate` parameter tensors and `nn.ModuleList` formats).
   - Sets router biases: `logit_bias = 0.0`, `null_logit = 0.0` (neutral at init; upstream callers in `init_expN.py` override these).

**Shared expert note:** The 1B dense FFN weights are copied **at full 1024 dimension** to the shared expert. The 1B model never performs SVD compression on the shared path — only the routed experts are compressed.

---

### `validate_expert_diversity()`

After initialization, reports the pairwise cosine similarity between routed expert gate weights per layer. 

- **Target range:** 0.95–0.98 (marked `✅ OPTIMAL`)
- **Outside range:** `⚠️ ADJUST EPS`

If similarity is too high (> 0.98), experts are effectively identical and won't specialize. If too low (< 0.95), experts start too differently and initial loss will spike.

---

## Training Configuration (common across experiments)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) |
| Scheduler | Cosine with warmup (`transformers.get_cosine_schedule_with_warmup`) |
| Loss | NTP + 0.3 × MTP + aux (load balancing) |
| Gradient clip | 1.0 |
| Sequence length | 64 |
| Physical batch size | 4 |
| Gradient accumulation | 8 → effective batch = 32 |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss spike at step 0 (e.g., 4 → 12) | Routed experts contributing too much at init | Increase null_logit bias or reduce `rotation_eps` |
| Null rate > 99% throughout training | Router biases too aggressive | Reduce null_logit (e.g., from +100 → +2.65) |
| "Baseline checkpoint not found" | `train_dense_1b_baseline.py` not run yet | Run Phase 1 first |
| SVD explained variance < 0.90 Warning | Aggressive compression ratio | Try `svd_mode="independent"` or increase `intermediate_moe` |
| Expert cosine sim all 1.0 | `rotation_eps` too small | Increase `rotation_eps` (default 0.005, try 0.02) |
| `tokenizer.json` not found | Script run from wrong directory | Must run from `endGame/` |
| `../synth_local_en/` not found | Dataset not downloaded | Download PleIAs/SYNTH locally |

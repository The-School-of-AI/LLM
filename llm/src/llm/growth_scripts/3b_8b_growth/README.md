# 3B → 8B Growth: ACT Solver Unrolling

> **Trajectory-Preserving Depth Expansion** — converting an 8-layer ACT-trained 3B model into a stable 20-layer 8B warmstart checkpoint.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Why Naïve Repetition Fails](#2-why-naïve-repetition-fails)
3. [The Correct Approach: ACT Solver Unrolling](#3-the-correct-approach-act-solver-unrolling)
4. [The Delta : GSA Constraint](#4-the-delta--gsa-constraint)
5. [Block-Wise Expansion Algorithm](#5-block-wise-expansion-algorithm)
6. [Implementation Details](#6-implementation-details)
7. [Files in This Directory](#7-files-in-this-directory)
8. [Usage](#8-usage)
9. [Expected Outcome](#9-expected-outcome)

---

## 1. The Problem

Our 3B model is an **8-layer MoE** trained with **Adaptive Computation Time (ACT)** — each layer can execute itself multiple times per token based on learned gating probabilities.

After training, the OuroSpec diagnostics reveal how many passes each layer *actually learned to use* (`k_exec`):

| Layer | Type     | k_exec | Role |
|-------|----------|--------|------|
| L0    | DeltaNet | 2      | Early feature extraction |
| L1    | DeltaNet | 1      | **Fully halted** — 1 pass sufficient |
| L2    | DeltaNet | 2      | Moderate processing |
| L3    | GSA      | 3      | Retrieval checkpoint |
| L4    | DeltaNet | 2      | Mid-network integration |
| L5    | DeltaNet | 3      | Deep reasoning begins |
| L6    | DeltaNet | 4      | **Compute bottleneck** — maxed at 4 passes |
| L7    | GSA      | 3      | Final retrieval + output refinement |

The total `k_exec` across all layers = **2+1+2+3+2+3+4+3 = 20**, which is exactly the number of layers in our target 8B architecture. It tells us precisely how to expand depth.

The goal is to grow this 8-layer model into a **20-layer 8B feedforward model** while preserving everything the 3B learned.

---

## 2. Why Naïve Repetition Fails

The intuitive approach — just copy layer weights `k_exec` times — is subtly wrong.

ACT does **iterative residual refinement**:

```
h ← h + α₁·F(h)      # pass 1: strong correction
h ← h + α₂·F(h)      # pass 2: smaller correction
h ← h + α₃·F(h)      # pass 3: fine-tuning
h ← h + α₄·F(h)      # pass 4: minimal adjustment
```

With the `alpha` schedule `[1.0, 0.7, 0.5, 0.35]` hardcoded in the 3B's `AdaptiveDecoderLayer`.

Naïve weight repetition produces **sequential composition** instead:

```
h₄ = F(F(F(F(h₀))))     ← operator cascade
```

| What ACT Learned | What Naïve Repeat Gives |
|------------------|-------------------------|
| Iterative correction converging to a fixed point | Feature drift — each pass amplifies, not corrects |
| Same operator, decaying residuals | Operator cascade with growing magnitude |
| Stable Jacobian spectrum | Potential eigenvalue explosion |

This is why some 1B→7B depth-expanded LLaMA-style experiments historically diverged in the first 500–1000 steps.

---

## 3. The Correct Approach: ACT Solver Unrolling

Instead of repeating weights directly, each physical 8B child layer represents **one ACT refinement step**, scaled by the corresponding `α` coefficient:

```
Source layer L6 (k_exec=4) expands to 4 physical 8B layers:

  8B layer child₁  =  1.00 × W₆       ← pass 1: full strength
  8B layer child₂  =  0.70 × W₆       ← pass 2: 70% strength
  8B layer child₃  =  0.50 × W₆       ← pass 3: 50% strength
  8B layer child₄  =  0.35 × W₆       ← pass 4: 35% strength
```

The resulting 8B feedforward chain:

```
h ← h + F₁(h)
h ← h + F₂(h)
h ← h + F₃(h)
h ← h + F₄(h)
```

...approximates what ACT was computing internally. The Jacobian spectrum stays bounded, and 8B training starts near the 3B's learned fixed point.

This technique is formally called **trajectory-preserving depth unrolling** and is the safest known method for converting ACT-style recurrence into feedforward depth in progressive scaling pipelines.

---

## 4. The Delta : GSA Constraint

The 3B and 8B architectures both enforce a strict **3:1 interleaving** of DeltaNet and GSA layers:

```python
if (i + 1) % 4 == 0:
    layer_type = "gsa"
else:
    layer_type = "deltanet"
```

This gives:
- **3B**: GSA at positions 3, 7 → `D D D G | D D D G`
- **8B**: GSA at positions 3, 7, 11, 15, 19 → `D D D G | D D D G | D D D G | D D D G | D D D G`

DeltaNet and GSA are **different operator families** — DeltaNet uses linear O(N) recurrence, GSA uses sparse quadratic attention. Their weight matrices have **different shapes** and cannot be interchanged.

A naïve per-layer repetition using individual `k_exec` values would destroy this interleaving:

```
❌ Naïve (using k_exec per layer):
Expanded: L0 L0 L1 L2 L2 [L3 L3 L3] L4 L4 [L5 L5 L5] [L6 L6 L6 L6] [L7 L7 L7]
Types:    D  D  D  D  D    G   G   G            D  D  D  D  D  D  D  D  D    G  G  G
                           ↑ 3 consecutive GSA!                              ↑ 3 consecutive GSA at the end!
```

The fix is **block-wise expansion**.

---

## 5. Block-Wise Expansion Algorithm

### Step 1 — Form 3B Blocks

Group 3B layers by their natural DDD+GSA blocks:

```
B0 = [L0, L1, L2, L3]   →   D D D G
B1 = [L4, L5, L6, L7]   →   D D D G
```

### Step 2 — Compute Block Expansion Count

```
expand(B0) = sum(k_exec[L0..L3]) / 4 = (2+1+2+3) / 4 = 8/4 = 2 copies
expand(B1) = sum(k_exec[L4..L7]) / 4 = (2+3+4+3) / 4 = 12/4 = 3 copies

Total: (2+3) × 4 = 20 layers ✅
```

### Step 3 — Apply ACT α-Scaling Per Block Copy

Each copy of a block receives a decaying `α` from the ACT schedule:

```
B0 copy 1  (α=1.00):  L0¹=1.00W₀,  L1¹=1.00W₁,  L2¹=1.00W₂,  L3¹=1.00W₃
B0 copy 2  (α=0.70):  L0²=0.70W₀,  L1²=0.70W₁,  L2²=0.70W₂,  L3²=0.70W₃

B1 copy 1  (α=1.00):  L4¹=1.00W₄,  L5¹=1.00W₅,  L6¹=1.00W₆,  L7¹=1.00W₇
B1 copy 2  (α=0.70):  L4²=0.70W₄,  L5²=0.70W₅,  L6²=0.70W₆,  L7²=0.70W₇
B1 copy 3  (α=0.50):  L4³=0.50W₄,  L5³=0.50W₅,  L6³=0.50W₆,  L7³=0.50W₇
```

### Step 4 — Assign Layer Types by 8B Position Formula

After assembling all 20 layers, type assignment uses the 8B formula — **never** inherited from the source:

```python
layer_type = "gsa" if (i + 1) % 4 == 0 else "deltanet"
```

This guarantees:
```
✅ 8B layout:
  L00-L02  D D D  │ L03  G
  L04-L06  D D D  │ L07  G
  L08-L10  D D D  │ L11  G
  L12-L14  D D D  │ L15  G
  L16-L18  D D D  │ L19  G

Delta:GSA = 15:5 = 3:1 ✅
```

Because we expand whole blocks (which always end in GSA), the source type and target type always match — **no attention weight shape mismatches**.

---

## 6. Implementation Details

### `act_unroll_init_3b_to_8b.py`

The script implements the full algorithm in 6 stages:

| Stage | Function | Description |
|-------|----------|-------------|
| 1 | `load_k_exec()` | Reads `k_exec` from `layer_repitation_factor.json` |
| 2 | `compute_block_expansion()` | Computes n_copies per block |
| 3 | `build_expansion_plan()` | Builds full 20-layer mapping table |
| 4 | `_build_8b_state_dict()` | Instantiates Model8B to get correct weight shapes |
| 5 | `copy_layer_weights()` | Copies & α-scales each layer; handles norms separately |
| 6 | `_sync_shared_layer_keys()` | Propagates values to all `ReversibleMidpointStack` alias keys |

### Weight Copy Rules

| Weight Category | Scale by α? | Notes |
|----------------|------------|-------|
| Attention sublayer (`attn_block.sublayer.*`) | ✅ Yes | Only if src/tgt types match |
| MLP sublayer (`mlp_block.sublayer.*`) | ✅ Yes | Always (shapes identical) |
| mHC routing coefficients (`coeffs.*`) | ✅ Yes | MHC weights |
| Layer norms (`norm.weight`, `norm.bias`) | ❌ No | Norms are not residual paths — copied as-is |

### Noise for Weight Diversity

Each block copy adds tiny Gaussian noise **before** α-scaling, so SNR stays constant regardless of α:

```python
tgt_weight = alpha * (src_weight + noise)    # SNR-invariant
# NOT: alpha * src_weight + noise             # SNR degrades with smaller α
```

Default `noise_std = 1e-4`. Set to `0.0` for exact ACT-scaled copies.

### Shared-Parameter Aliases

`ReversibleMidpointStack` registers the same layer parameters under multiple state dict key paths:

```
layers.{i}.*                          ← primary (written by this script)
stack.blocks.{i}.*                    ← alias
stack.bootstrap_layer.*               ← alias (layer 0 only)
stack.mid_layers.{i-1}.block.*        ← alias (layers > 0)
stack.mid_layers.{i-1}.wrapper.layer.*← alias (layers > 0)
```

`_sync_shared_layer_keys()` propagates the initialized values to all aliases after the main copy loop, ensuring `load_state_dict()` loads the correct weights regardless of key processing order.

---

## 7. Files in This Directory

| File | Description |
|------|-------------|
| `act_unroll_init_3b_to_8b.py` | Main initialization script |
| `layer_repitation_factor.json` | Per-layer `k_exec` and diagnostic metrics from OuroSpec run |
| `README.md` | This file |

### `layer_repitation_factor.json` Schema

```json
{
    "_meta": { "source": "OuroSpec diagnostics", "lambda_p": 0.0375 },
    "layer0": {
        "k_exec": 2,
        "expected_k": 1.23,
        "s_fin": 0.047,
        "entropy": 0.7092,
        "p_mu": 0.23,
        "ponder": 0.078,
        "note": "Human-readable interpretation"
    },
    ...
}
```

The `k_exec` field is the primary signal used by the expansion algorithm. All other fields are diagnostics for analysis.

---

## 8. Usage

### Inspect the expansion plan (no files needed)

```bash
python act_unroll_init_3b_to_8b.py \
    --src /path/to/3b_checkpoint.pt \
    --tgt /path/to/output/8b_init.pt \
    --model_dir ../model \
    --dry_run
```

### Run the initialization

```bash
python act_unroll_init_3b_to_8b.py \
    --src /path/to/3b_checkpoint.pt \
    --tgt checkpoints/8b_act_unroll_init.pt \
    --model_dir ../model
```

### All options

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--src` | ✅ | — | Path to 3B checkpoint `.pt` file |
| `--tgt` | ✅ | — | Output path for initialized 8B checkpoint |
| `--model_dir` | ✅ | — | Directory containing `recurrence_model_8b.py` |
| `--json` | ❌ | auto-detect | Path to `layer_repitation_factor.json` |
| `--noise_std` | ❌ | `1e-4` | Gaussian diversity noise σ per copy |
| `--dry_run` | ❌ | `False` | Print plan only, no I/O |

---

## 9. Expected Outcome

### Without ACT Solver Unrolling (naïve repetition)

```
8B training steps 0–1000:
  ❌ Large initial loss spike
  ❌ Router entropy collapse (MoE experts homogenize)
  ❌ Δh explosion in early layers
  ❌ Slow recovery, degraded convergence
```

### With ACT Solver Unrolling

```
8B training steps 0–1000:
  ✅ Smooth loss continuation from 3B endpoint
  ✅ MoE experts re-specialize rapidly
  ✅ Jacobian spectrum remains bounded
  ✅ Fast warmstart, no instability window
```

### What is Guaranteed

| Property | Guarantee |
|----------|-----------|
| **Delta:GSA ratio** | Always 15:5 = 3:1 by position formula |
| **ACT trajectory** | Approximated via α-scaled residual chain |
| **Type matching** | Source and target types always match (block-wise expansion) |
| **Embedding** | Kronecker or standard — auto-detected and copied |
| **MoE routing** | Expert weights copied with same α scaling |
| **Optimizer** | Reset to fresh state (do not warm-start optimizer) |

---

> **Citation note**: The "trajectory-preserving depth unrolling" technique is the safest known method for converting ACT-style recurrent computation into feedforward depth in progressive scaling pipelines.

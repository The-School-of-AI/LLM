# 8B → 70B Growth: Function-Preserving Expert Explosion

> **Round-Robin Tiling + Output-Nullspace Noise + Mass Correction + Net2Wider Scaling + Top-K Warmstart** — expanding 20 routed MoE experts into 260 while preserving the learned routing function under Top-K change (2 → 8).

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Why Random Initialization Fails](#2-why-random-initialization-fails)
3. [Why Pure Duplication Also Fails](#3-why-pure-duplication-also-fails)
4. [Softmax Mass Fragmentation](#4-softmax-mass-fragmentation)
5. [Routing Drift Under Top-K Change](#5-routing-drift-under-top-k-change)
6. [The Five Corrections](#6-the-five-corrections)
7. [Round-Robin Assignment](#7-round-robin-assignment)
8. [Output-Nullspace Projected Noise (LiGO-style)](#8-output-nullspace-projected-noise-ligo-style)
9. [Router Logit Mass Correction](#9-router-logit-mass-correction)
10. [Down-Projection Scaling (Before Noise)](#10-down-projection-scaling-before-noise)
11. [Top-K Warmstart Schedule](#11-top-k-warmstart-schedule)
12. [Router Gate Weight Initialization](#12-router-gate-weight-initialization)
13. [Complete Weight Copying Strategy](#13-complete-weight-copying-strategy)
14. [Code Walkthrough](#14-code-walkthrough)
15. [Files in This Directory](#15-files-in-this-directory)
16. [Usage](#16-usage)
17. [Validation](#17-validation)
18. [Hyperparameter Tuning Guide](#18-hyperparameter-tuning-guide)
19. [Expected Outcome](#19-expected-outcome)
20. [Alternatives Considered](#20-alternatives-considered)

---

## 1. The Problem

Our 8B MoE model has **20 routed experts** per layer (plus 1 shared expert), with each expert being a SwiGLU FFN of shape `(4096, 1024)`. The 70B model scales to **260 routed experts** per layer while keeping all other dimensions identical:

| Parameter | 8B Model | 70B Model | Change |
|-----------|----------|-----------|--------|
| Hidden size | 4096 | 4096 | Same |
| Layers | 20 | 20 | Same |
| Expert intermediate | 1024 | 1024 | Same |
| **Routed experts** | **20** | **260** | **13x explosion** |
| Null experts | 20 | 260 | 13x (mirrors real) |
| Total MoE slots | 40 | 520 | 13x |
| **Top-k** | **2** | **8** | **4x increase** |
| Shared expert intermediate | 2048 | 2048 | Same |
| Total parameters | 8.29B | ~70B | ~8.4x |
| Active parameters | 3.27B | ~4.08B | ~1.25x |

The goal: initialize the 70B model from a trained 8B checkpoint such that:
1. The 70B model produces **near-identical outputs** to the 8B model at initialization
2. The 260 experts have enough **structural diversity** to specialize during training
3. The router **preserves parent-level routing** despite the Top-K change (2 → 8)

---

## 2. Why Random Initialization Fails

If we copy all non-expert weights from 8B but randomly initialize the 260 new experts:

```
Forward pass token → Attention (copied, good) → MoE (260 random experts) → garbage output
```

- **Catastrophic loss spike** — the model's output distribution is suddenly random
- **Router confusion** — gradients from random expert outputs teach garbage routing patterns
- **Slow recovery** — must re-learn expert functions from scratch

---

## 3. Why Pure Duplication Also Fails

Copying each source expert exactly 13 times with no noise creates a fatal **gradient symmetry** problem:

```
Expert 0 → Clones [0, 1, 2, ..., 12] — all identical weights
```

Since all 13 clones have identical weights AND identical router gate vectors:
- They receive **identical gradients** at every training step
- They update in **perfect lockstep** forever
- They **never differentiate** — 260 experts that behave as 20

---

## 4. Softmax Mass Fragmentation

Even with perfect weight cloning and tiny noise, naive expert tiling **breaks the MoE function** due to how softmax routing works.

In a routed MoE: `y = Σ_i p_i(x) · f_i(x)` where `p_i = softmax(gate(x))_i`

After cloning expert `e` into 13 copies with identical gate logits:

```
p_eₖ = exp(z_e) / (13·exp(z_e) + others)
```

The total probability mass for all clones: `Σₖ p_eₖ ≠ p_e` — the denominator grows by `12·exp(z_e)`, diluting the mass.

Result: **output magnitude drops ~13×** even with identical weights. This is the **#1 failure mode** in expert explosion.

---

## 5. Routing Drift Under Top-K Change

Even after fixing mass fragmentation, there's a subtler problem when **Top-K changes** (2 → 8).

### The problem

Softmax is invariant to additive shifts per-logit, but **Top-K is not**. After explosion, each original expert produces 13 noisy logits. Even with `−log(13)` correction preserving mass:

```
max(e2_clones) = 7.83       ← gate noise shifts rankings
max(e1_clones) = 8.01
max(e3_clones) = 7.85       ← e3 beats e2 now!
```

Original routing path `{e1, e2}` may become `{e1, e3}` — the computation graph changes at step 0.

### Why this gets worse with K increase

With Top-2 → Top-8, routing sensitivity increases ~4× because the probability of selecting a "wrong" sibling clone increases combinatorially. If sibling clones sit adjacent in index space, Top-K easily co-selects multiple siblings of the same parent, changing the effective computation graph.

### The solution: round-robin assignment

Spread siblings across the full index range so that parent-level competition happens before sibling-level, preserving the original routing structure. See [§7](#7-round-robin-assignment).

---

## 6. The Five Corrections

| # | Correction | What it fixes |
|---|-----------|---------------|
| 1 | **Round-robin assignment** | Siblings index-separated → preserves parent-level Top-K |
| 2 | **Output-nullspace noise** (LiGO) | `(W+ΔW)x ≈ Wx` in activation space, not just parameter space |
| 3 | **Router logit mass correction** | `bias -= log(13)` → `Σ p_clone ≈ p_original` |
| 4 | **W_down / 13 BEFORE noise** | Correct magnitude + full-strength noise for specialization |
| 5 | **Top-K warmstart schedule** | Start K=2 → 4 → 8 to stabilize routing before widening |

### Balanced Tiling

260 / 20 = **13 copies per expert, exactly**. We chose 260 (instead of 256) specifically to make this division exact.

---

## 7. Round-Robin Assignment

### Layout

```
Target [  0,  1,  2, ..., 19]  ← Sources 0..19 (copy 0)
Target [ 20, 21, 22, ..., 39]  ← Sources 0..19 (copy 1)
...
Target [240, 241, ..., 259]     ← Sources 0..19 (copy 12)
```

Formula: `assignment[j] = j % 20`

Siblings of source expert 0: indices `[0, 20, 40, 60, ..., 240]` — spread every 20 positions.

### Why round-robin works

| Property | Effect |
|----------|--------|
| Sibling index distance | 20 (interleaved) |
| Top-K co-selects siblings? | Unlikely |
| Parent-level competition | Before siblings |
| Routing diversity at init | Good |
| Need for logit ranking noise? | No |
| Early-step load balance | Naturally balanced |

With round-robin, the `−log(13)` bias correction alone is sufficient — no artificial ranking noise needed.

---

## 8. Output-Nullspace Projected Noise (LiGO-style)

### Why output-space projection matters

Parameter-space orthogonality (`⟨ΔW, W⟩_F = 0`) does not guarantee output preservation:

```
ΔW ⊥ W  in parameter space  ≠  ΔW·x ⊥ W·x  in activation space
```

The noise must be projected into the **output null-space** of W to ensure `(W+ΔW)x ≈ Wx`.

### The projection

Project noise into `ker(Wᵀ)` — the output null-space of W:

```
Q ← orthogonal(shape)              # random orthogonal matrix via QR
ΔW = Q − W(WᵀW)⁻¹WᵀQ              # project into ker(Wᵀ)
ΔW = ΔW / ||ΔW||_F                  # re-normalize
W_clone = W_base + ε · ||W_base||_F · ΔW
```

This ensures `ΔW` only adds components in output directions that W doesn't use. For any input `x`:

```
(W + ε·ΔW)x = Wx + ε·ΔW·x
```

where `ΔW·x` lies in the null-space of Wᵀ — orthogonal to all outputs W can produce.

### Properties at Initialization

| Property | Value |
|----------|-------|
| Distance from source: `‖W_clone - W_source‖_F / ‖W_source‖_F` | ≈ `eps` (= 0.01) |
| Distance between clones: `‖clone_i - clone_j‖_F / ‖W_source‖_F` | ~`sqrt(2) * eps` |
| ΔW in ker(Wᵀ) | Yes |
| Output deviation at init | Near-zero |
| Parameter orthogonality (⟨ΔW, W⟩_F) | Also holds |
| Symmetry breaking | Yes |

---

## 9. Router Logit Mass Correction

When duplicating expert `i` into `{i₁...i₁₃}`, shift each clone's logit bias:

```
logit_bias_clone = logit_bias_source − log(13)
```

Because `softmax(z − log(k)) = exp(z) / (k · denom')`, so:

```
Σₖ p_iₖ ≈ p_i    ✅ mass preserved
```

Without this correction:
- Each original expert's total contribution drops ~13×
- Top-k ranking changes even with zero expert noise
- **You are doing implicit expert dropout, not function-preserving expansion**

---

## 10. Down-Projection Scaling (Before Noise)

### Why W_down must be scaled before noise

The Net2Wider transformation requires dividing W_down by 13 so that the 13 clones sum to the original output magnitude. This scaling must happen **before** noise generation:

```python
W_base = W_down / 13                  # scale first (Net2Wider)
ΔW = output_nullspace_noise(W_base)   # noise on the scaled base
W_clone = W_base + ε · ||W_base|| · ΔW
```

If scaling is applied after noise, the noise magnitude is also divided by 13, making siblings too similar in output space and slowing specialization.

### Why only W_down?

The SwiGLU FFN computes: `y = W_down · (SwiGLU(W_gate·x) ⊙ W_up·x)`

Scaling `W_down` alone preserves internal activation dynamics while compensating for the increased number of parallel paths at the output.

---

## 11. Top-K Warmstart Schedule

### Why this is mandatory

Even with all weight-level corrections, training the 70B model with K=8 from step 0 is **not** function-preserving because:

- The original 8B model learned routing patterns optimized for K=2
- Jumping to K=8 immediately changes the computation graph
- Specialization begins before routing stabilizes → clone collapse

### Recommended schedule

| Steps | Top-K | Rationale |
|-------|-------|-----------|
| 0–1000 | 2 | Match 8B's learned routing. Clones begin differentiating. |
| 1000–3000 | 4 | Gradual increase. Router adapts to selecting from more experts. |
| 3000+ | 8 | Full 70B routing. Experts are now sufficiently differentiated. |

### Implementation

The training script should support a `top_k_schedule` parameter:

```python
def get_top_k(step):
    if step < 1000:
        return 2
    elif step < 3000:
        return 4
    else:
        return 8
```

This is **not** implemented in the init script (it's a training-time concern), but is documented here as a mandatory training requirement.

---

## 12. Router Gate Weight Initialization

### Gate Weight: Tiny Gaussian Noise

```
gate_70b.weight[j] = gate_8b.weight[src_idx] + eps_gate * N(0, 1)
```

With `eps_gate = 0.0005` (20x smaller than expert perturbation).

**Why 0.0005?** The Top-K increase (2 → 8) and round-robin layout together increase routing competition. Smaller gate noise prevents early-step routing entropy from being too high, which could cause some experts to starve.

**Why Gaussian instead of orthogonal?** Orthogonal perturbations in gate space would push clones toward very different regions of token space, potentially causing some clones to never receive any tokens.

---

## 13. Complete Weight Copying Strategy

### Per-Layer (20 backbone layers + 1 MTP block)

| Component | 8B Shape | 70B Shape | Method |
|-----------|----------|-----------|--------|
| `moe.W_gate` | (20, 4096, 1024) | (260, 4096, 1024) | Round-robin + output-nullspace noise |
| `moe.W_up` | (20, 4096, 1024) | (260, 4096, 1024) | Round-robin + output-nullspace noise |
| `moe.W_down` | (20, 1024, 4096) | (260, 1024, 4096) | **÷13 first**, then output-nullspace noise |
| `moe.gate.weight` | (20, 4096) | (260, 4096) | Round-robin + Gaussian noise (eps=0.0005) |
| `moe.gate.logit_bias` | (20,) | (260,) | Round-robin **− log(13)** |
| `moe.shared_*` | (2048, 4096) | (2048, 4096) | Direct copy |
| Attention weights | identical | identical | Direct copy |
| RMSNorm weights | identical | identical | Direct copy |
| mHC coefficients | identical | identical | Direct copy |

### Global (non-layer) Components

| Component | Method |
|-----------|--------|
| Token embeddings / Kronecker embeddings | Direct copy (auto-detected) |
| LM head | Direct copy |
| Final RMSNorm | Direct copy |
| Memory stream (lambda_r_raw, memory_ln, etc.) | Direct copy |
| MTP fusion projection | Direct copy |
| MTP attention (GSA) | Direct copy |

### Shared-Parameter Aliases

The `ReversibleMidpointStack` registers each layer under multiple state dict key paths:

```
layers.{i}.*                              ← primary (written by this script)
stack.blocks.{i}.*                        ← alias
stack.bootstrap_layer.*                   ← alias (layer 0 only)
stack.mid_layers.{i-1}.block.*            ← alias (layers > 0)
stack.mid_layers.{i-1}.wrapper.layer.*    ← alias (layers > 0)
```

After all weights are initialized, `sync_shared_layer_keys()` propagates values from the primary path to all aliases.

---

## 14. Code Walkthrough

### `expert_explosion_init_8b_to_70b.py`

#### `build_expert_assignment()`

Builds round-robin mapping: `assignment[j] = j % 20`. Siblings of source expert `e` are at indices `[e, e+20, e+40, ..., e+240]`.

#### `generate_orthogonal_noise()` + `output_nullspace_project()`

- `generate_orthogonal_noise()` — Random matrix with unit Frobenius norm via QR decomposition.
- `output_nullspace_project(Q, W)` — LiGO-style projection into ker(Wᵀ): `ΔW = Q − W(WᵀW)⁻¹WᵀQ`. Uses `torch.linalg.lstsq` for numerical stability.

#### `tile_expert_weights()`

For each target expert:
1. Look up source via round-robin assignment
2. **For W_down**: divide by 13 first (Net2Wider scaling)
3. Generate orthogonal noise, project into output null-space of (scaled) base
4. Add perturbation: `W_clone = W_base + eps * ||W_base||_F * ΔW`

#### `tile_gate_weights()` and `tile_gate_bias()`

- Gate weights: round-robin tile with Gaussian noise (`eps_gate=0.0005`)
- Logit biases: round-robin tile with `−log(13)` mass correction

#### `explode_8b_to_70b()` — Main orchestrator

1. Set random seed
2. Build round-robin expert assignment
3. Load 8B checkpoint
4. Instantiate 70B model for target shapes
5. Adapt embedding keys
6. Copy non-layer weights
7. Initialize all 20 layers (tile experts + copy rest)
8. Initialize MTP block's routed experts
9. Sync shared-parameter aliases
10. Sanity check (no NaN/Inf)
11. Save with provenance metadata

---

## 15. Files in This Directory

| File | Description |
|------|-------------|
| `expert_explosion_init_8b_to_70b.py` | Main initialization script |
| `validate_explosion_init.py` | Post-init validation script |
| `README.md` | This file |

---

## 16. Usage

### Inspect the assignment plan (no files needed)

```bash
python expert_explosion_init_8b_to_70b.py \
    --src /path/to/8b_checkpoint.pt \
    --tgt /path/to/output/70b_init.pt \
    --model_dir ../ \
    --dry_run
```

### Run the initialization

```bash
python expert_explosion_init_8b_to_70b.py \
    --src checkpoints/8b_trained.pt \
    --tgt checkpoints/70b_expert_explosion_init.pt \
    --model_dir ../
```

### All options

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--src` | Yes | — | Path to trained 8B checkpoint `.pt` file |
| `--tgt` | Yes | — | Output path for initialized 70B checkpoint |
| `--model_dir` | Yes | — | Directory containing `recurrence_model_70b.py` |
| `--eps_expert` | No | `0.01` | Output-nullspace perturbation scale for expert weights |
| `--eps_gate` | No | `0.0005` | Gaussian noise scale for router gate weights |
| `--seed` | No | `42` | Random seed for reproducibility |
| `--dry_run` | No | `False` | Print assignment plan only, no I/O |

---

## 17. Validation

### Why validate before training

Expert explosion can silently fail — the loss might look "reasonable" at init but the router distribution is completely wrong, leading to slow convergence or expert collapse later.

### Running validation

```bash
python validate_explosion_init.py \
    --src checkpoints/8b_trained.pt \
    --tgt checkpoints/70b_expert_explosion_init.pt \
    --model_dir ../
```

### Metrics and thresholds

| Metric | Formula | PASS | WARN | FAIL |
|--------|---------|------|------|------|
| Loss difference | `\|L_70B − L_8B\| / L_8B` | < 3% | < 10% | > 10% |
| Logit cosine similarity | `cos(logits_70B, logits_8B)` | > 0.995 | > 0.98 | < 0.98 |
| Router KL divergence | `KL(p_70B_agg ‖ p_8B)` | < 0.02 | < 0.05 | > 0.05 |

### How router KL is computed

The validation script aggregates clone probabilities back to parent groups using the round-robin mapping:

```
p_70b_aggregated[i] = Σ p_70b[i, i+20, i+40, ..., i+240]
```

This sums the 13 sibling probabilities (every 20th index) back to their parent. With `−log(13)` correction, this should closely match the 8B's original distribution.

If router KL is large but loss looks ok → the mass correction is likely missing or incorrect.

### Validation options

| Argument | Default | Description |
|----------|---------|-------------|
| `--src` | Required | Path to 8B checkpoint |
| `--tgt` | Required | Path to initialized 70B checkpoint |
| `--model_dir` | Required | Directory with model files |
| `--batch_size` | `2` | Batch size for validation |
| `--seq_len` | `64` | Sequence length |
| `--data_file` | None | Optional `.pt` file of real token IDs |
| `--seed` | `42` | Random seed |
| `--device` | `cpu` | Device (`cpu` or `cuda`) |

---

## 18. Hyperparameter Tuning Guide

### `eps_expert` (Expert Weight Perturbation)

| Value | Behavior |
|-------|----------|
| `0.001` | Ultra-conservative. Near-perfect function preservation. Experts may take many steps to differentiate. |
| `0.005` | Conservative. Safe choice if 0.01 causes loss spike >3%. |
| **`0.01`** | **Default. Recommended starting point.** |
| `0.02` | Aggressive. Use if experts don't differentiate after 1000+ steps at 0.01. |
| `0.05` | Very aggressive. May cause noticeable loss spike. Only for experimentation. |

### `eps_gate` (Router Gate Perturbation)

Should be **10-20x smaller** than `eps_expert`. With Top-K=8 and round-robin layout, routing competition is higher than in a Top-K=2 model:

| Value | Behavior |
|-------|----------|
| `0.0002` | Very conservative. Siblings barely differentiate in routing. |
| **`0.0005`** | **Default. Recommended for Top-K=8 + round-robin.** |
| `0.001` | More aggressive. May cause routing entropy spike with K=8. |

### `seed`

Controls the random number generator for reproducible initialization. Use the same seed to recreate the exact same checkpoint.

---

## 19. Expected Outcome

### At Initialization (Step 0)

```
70B forward pass on validation batch:
  Loss:  < 3% above 8B converged loss
  Logit cosine sim: > 0.995
  Router KL: < 0.02
  Router: distributes tokens roughly uniformly among 13 clones of each source expert
  Shared expert: exact copy from 8B
```

### Top-K Warmstart Phase (Steps 0-3000)

```
  Steps 0-1000 (K=2):
    Loss continues from 8B endpoint
    Router patterns match 8B behavior
    Siblings begin differentiating via weight noise

  Steps 1000-3000 (K=4):
    Router explores selecting 2 additional experts
    Expert diversity increases measurably
    Some sibling pairs begin specializing

  Steps 3000+ (K=8):
    Full 70B routing capacity activated
    260 experts have differentiated
    Loss improves beyond 8B converged level
```

### What is Guaranteed

| Property | Guarantee |
|----------|-----------|
| **Parent-level Top-K** | Round-robin → siblings index-separated, parent ranking preserved |
| **Softmax mass** | `−log(13)` bias correction → `Σ p_clone ≈ p_original` |
| **Output magnitude** | W_down/13 before noise → Net2Wider-correct |
| **Activation preservation** | Output-nullspace noise → `(W+ΔW)x ≈ Wx` locally |
| **Noise strength** | W_down scaled before noise → full ε for specialization |
| **No gradient symmetry** | Orthogonal perturbation → 260 distinct gradient signals |
| **Balanced tiling** | Every source expert contributes exactly 13 clones |
| **Reproducibility** | Same seed → identical checkpoint |

---

## 20. Alternatives Considered

| Alternative | Why Not Used |
|-------------|-------------|
| **Random init for new experts** | Catastrophic loss spike — random experts destroy learned representations. |
| **Pure duplication (no noise)** | Gradient symmetry — clones never differentiate. |
| **Contiguous block assignment** | Siblings adjacent in index space → Top-K co-selects multiple siblings → routing drift. |
| **Parameter-space projection (Frobenius)** | `⟨ΔW, W⟩_F = 0` ≠ `ΔW·x ⊥ W·x`. Output-nullspace preserves function in activation space. |
| **W_down scaling after noise** | Suppresses noise magnitude → siblings too similar → slower specialization. |
| **Gaussian noise (not orthogonal)** | Weaker symmetry breaking. Random perturbations can accidentally align. |
| **No logit mass correction** | Without `−log(13)`, softmax mass fragments → ~13× output drop → loss spike. |
| **Logit ranking noise** | Hacky workaround for contiguous blocks. Round-robin makes this unnecessary. |
| **Training with K=8 from step 0** | Specialization begins before routing stabilizes → clone collapse. |
| **K-means clustering of experts** | Only 20 source experts — not enough data for meaningful clustering into 260 groups. |
| **Unbalanced copying** | Amplifies existing router bias. Balanced tiling lets the router re-learn optimal distribution. |

---

> **Prior art**: This approach combines function-preserving model growth techniques (Net2Net, Net2Wider, Bert2BERT, LiGO) with MoE-specific corrections (logit mass preservation, output-nullspace perturbation, round-robin assignment). The key insight is that MoE expert explosion under Top-K change requires not just weight-level but **routing-level** preservation — round-robin layout + Top-K warmstart ensures the grown model starts from the same computation graph as the source.

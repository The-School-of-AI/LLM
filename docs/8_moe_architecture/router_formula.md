# Router Logic (Task 8.3)

## 1. Objective

This section defines the router formulation and interface for the FFN-only MoE architecture. The design ensures native expert competition (including Null), loss-free routing, and full compatibility with Team 7 telemetry and steering mechanisms.

---

## 2. Router Formula

Each token representation `x ∈ R^d` is routed using a learned linear projection followed by softmax and Top-K selection.

### 2.1 Gating Logits

For each expert `i`:

```
zi = x · Wg[i] + bi
```

Where:

* `x` : token hidden state
* `Wg ∈ R^{E × d}` : router weight matrix
* `bi` : per-expert bias term (externally controllable)
* `E` : total number of experts (including Null)

### 2.2 Routing Probabilities

```
pi = softmax(z)i
```

Softmax is applied across all experts, including Null.

### 2.3 Top-K Selection

```
S = TopK(p, K)
```

Where:

* `K` is fixed per layer
* `S` is the selected expert index set

Only experts in `S` receive token assignments. All others receive zero weight.

### 2.4 Expert Weights

For selected experts:

```
wi = pi / Σj∈S pj ,  i ∈ S
```

This ensures normalized mixture weights within Top-K.

---

## 3. Bias Interface (Steering API)

### 3.1 Purpose

Bias terms provide the sole mechanism for external routing control and steering. No auxiliary losses or heuristic constraints are used.

### 3.2 Interface Definition

Each router exposes a mutable bias vector:

```
b ∈ R^E
```

Accessible via the routing control API:

```
set_router_bias(layer_id, expert_id, value)
get_router_bias(layer_id, expert_id)
```

### 3.3 Steering Semantics

* Positive bias → favors expert selection
* Negative bias → suppresses expert selection
* Zero bias → neutral routing

Bias updates are additive to learned logits and applied before softmax.

---

## 4. Null Expert Mechanism

### 4.1 Definition

The Null expert is implemented as a standard expert with a fixed index:

```
expert_id = 0  (reserved)
```

### 4.2 Competition Rules

* Null participates in softmax and Top-K identically to real experts
* No thresholding, gating, or special-case logic is applied
* Selection is purely probability-driven

### 4.3 Semantics

When selected, the Null expert performs an identity/no-op transformation and contributes zero or minimal compute cost.

This enables natural capacity shedding and implicit sparsity.

---

## 5. Loss and Optimization Constraints

### 5.1 Prohibited Mechanisms

The following are explicitly disallowed:

* Load-balancing losses
* Entropy regularization
* Importance loss
* Capacity loss
* Z-loss
* Any auxiliary routing objective

### 5.2 Training Objec[router_formula.md](router_formula.md)tive

Routing parameters (`Wg`, base `b`) are trained exclusively through end-task loss backpropagation.

All load control and balancing is handled externally via bias steering.

---

## 6. Telemetry Interface (Team 7 Compatibility)

### 6.1 Required Outputs

Each router must emit per-layer, per-step telemetry:

```
{
  token_count[E],
  selection_rate[E],
  mean_weight[E],
  null_rate,
  overflow_rate,
  entropy
}
```

Where metrics include the Null expert.

### 6.2 Reporting Granularity

* Per-layer
* Per-training-step (aggregated)
* Optional per-batch sampling

### 6.3 Integration

Telemetry streams are consumed by Team 7 controllers to compute bias updates.

No internal feedback loops are implemented inside the router.

---

## 7. Reference Implementation (Pseudocode)

```python
def route(x, Wg, b, K):
    # x: [d]
    # Wg: [E, d]
    # b: [E]

    z = matmul(Wg, x) + b          # [E]
    p = softmax(z)                # [E]

    idx = topk(p, K)              # [K]
    w = p[idx]
    w = w / w.sum()               # renormalize

    return idx, w
```

---

## 8. Compliance Summary

| Requirement             | Status            |
| ----------------------- | ----------------- |
| TopK(softmax(x·Wg + b)) | ✔ Implemented     |
| Bias steering API       | ✔ Exposed         |
| Native Null competition | ✔ Standard expert |
| No auxiliary losses     | ✔ Enforced        |
| Telemetry compatibility | ✔ Provided        |

-
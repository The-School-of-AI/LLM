## Router Logic (Task 8.3)

### Router score + TopK formula

For each token hidden state \(x \in \mathbb{R}^{d_\text{model}}\), compute router logits over **all experts including Null**:

- Let \(E\) be the number of experts **including** Null, where Null is a standard expert index (see below).
- Router weights: \(W_g \in \mathbb{R}^{d_\text{model} \times E}\)
- Router bias vector (the *only* steering lever): \(b \in \mathbb{R}^{E}\)

Logits:
\[
\ell = x W_g + b
\]

Routing probabilities:
\[
p = \mathrm{softmax}(\ell)
\]

Expert selection and weights:
\[
\text{idx} = \mathrm{TopK}(p, k) \quad;\quad w = \mathrm{normalize}(p[\text{idx}])
\]

Where `normalize` renormalizes the selected probabilities to sum to 1 (common topK gating convention), and all non-selected experts have weight 0.

### Null expert mechanism (native competition)

- **Null is a standard expert index** `null_expert_id  [0, E-1]` that is present in the same logit vector as real experts and participates in the same `softmax` and `TopK`.
- Null has a valid output function but is compute-free by definition:
  - `Expert(null).forward(x)` returns **zeros** with the expert output shape (or an equivalent no-op that produces a zero contribution).
- There is **no threshold logic**, no if p <  then drop, and no external path; Null wins/loses purely through softmax competition and TopK selection.

### Bias steering interface (Team 7 API)

Expose the per-expert bias vector as a first-class control plane surface so Team 7 can steer routing *without* losses:

**API shape (conceptual):**
- `set_router_bias(layer_id, b: float[E])`
- `add_router_bias(layer_id, delta: float[E])`
- `get_router_bias(layer_id) -> float[E]`

**Rules:**
- Bias is applied **additively** to logits before softmax: \(\ell = xW_g + b\).
- Bias entries may target any expert including Null (e.g., positive bias toward Null to reduce compute, negative bias to discourage Null).
- No other steering knobs are allowed in the router (e.g., temperature schedules, stochastic noise, thresholds, expert masks) unless separately approved by the architecture spec.

### Telemetry outputs (interface points for Team 7)

Router must emit per-layer, per-step aggregates sufficient for routing health gates and telemetry consumption.

Minimum required signals (all computed *after* bias application):
- `router_logits_summary`: mean/std/min/max of \(\ell\) across tokens.
- `router_prob_summary`: mean entropy of \(p\); optionally top1 prob stats.
- `topk_expert_ids`: selected indices per token (or histogram only if privacy/size constrained).
- `topk_expert_weights`: selected normalized weights per token (or summary stats).
- `expert_selection_histogram[E]`: count of selections per expert (Null included).
- `null_selection_rate`: fraction of tokens where Null appears in TopK; and fraction where Null is Top1.
- `effective_active_experts`: average number of **non-Null** experts in TopK per token.

(Team 7 may define exact schema/names; this section defines what the router must be able to produce.)

### Loss constraint (explicit ban)

- **BANNED:** Any auxiliary routing/load-balancing loss terms (e.g., importance loss, load loss, aux balancing loss) that add gradients to the router.
- Router behavior control must be achieved **only** via the exposed bias vector \(b\) (including Nulls bias) and Team 7s external control logic/health gates.

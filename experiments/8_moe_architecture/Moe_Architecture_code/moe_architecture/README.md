# Team 8 MoE Architecture - Mixture of Experts Transformer

A production-ready Mixture of Experts (MoE) transformer implementation featuring GSA-style routing, loss-free load balancing, null expert data sparsity, and comprehensive telemetry for multi-stage growth from 1B to 70B parameters.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Complete Architecture Diagram](#complete-architecture-diagram)
3. [Component Details with Code](#component-details-with-code)
   - [Router (GSA & Null Expert)](#1-router-gsa--null-expert)
   - [Experts (Gated, Null, Shared)](#2-experts-gated-null-shared)
   - [Load Balancer](#3-load-balancer)
   - [MoE Block](#4-moe-block)
   - [Attention (GQA & GSA)](#5-attention-gqa--gsa)
4. [Configuration Reference](#configuration-reference)
5. [Working Flow](#working-flow)
6. [Growth Stages](#growth-stages)
7. [Quick Start](#quick-start)

---

## Architecture Overview

This MoE architecture implements a transformer model with:

- **GSA-style Multi-head Sigmoid Router** - Token routing with bounded scores
- **Null Expert Data Sparsity** - Zero-compute pathway for junk token absorption
- **Loss-Free Load Balancing** - Bias-only adjustment without auxiliary loss
- **Shared Experts** - Always-active experts for common patterns
- **Dual Gating (Optional)** - G1/G2 gates for collapse prevention
- **Gated Sparse Attention (GSA)** - Adaptive attention with learned sparsity

### Key Features

| Feature | Description |
|---------|-------------|
| **Null Expert Router** | Token-choice routing with null copies (arXiv:2601.15370v1) - **PRIMARY ROUTER** |
| DeepSeek-V3 Load Balancing | Bias-only adjustment for real experts |
| Null Expert Data Sparsity | M null copies for top-k selection (M = N(1-ρ)/ρ) |
| Dual Gating (G1+G2) | Optional collapse prevention from GSA paper |
| Team 7 Telemetry | Comprehensive routing health monitoring |
| CUDA Kernels | High-performance Triton implementations |

---

## Complete Architecture Diagram

### Full Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MoE TRANSFORMER MODEL                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    Input IDs [batch, seq]                                                       │
│         │                                                                        │
│         ▼                                                                        │
│    ┌─────────────────────┐                                                      │
│    │   Token Embedding    │  vocab_size × hidden_size                           │
│    │   (32000 × 2048)     │                                                      │
│    └──────────┬──────────┘                                                      │
│               │                                                                  │
│               ▼                                                                  │
│    ╔═══════════════════════════════════════════════════════════════════╗       │
│    ║                    TRANSFORMER LAYER × N                           ║       │
│    ║  ┌─────────────────────────────────────────────────────────────┐  ║       │
│    ║  │                                                              │  ║       │
│    ║  │   hidden_states ──┬─────────────────────────────────────┐   │  ║       │
│    ║  │        │          │                                     │   │  ║       │
│    ║  │        ▼          │                                     │   │  ║       │
│    ║  │   ┌─────────┐     │                                     │   │  ║       │
│    ║  │   │ RMSNorm │     │                                     │   │  ║       │
│    ║  │   └────┬────┘     │                                     │   │  ║       │
│    ║  │        ▼          │                                     │   │  ║       │
│    ║  │   ┌─────────────────────────────────────────────────┐   │   │  ║       │
│    ║  │   │          GQA / GSA ATTENTION                    │   │   │  ║       │
│    ║  │   │   ┌─────────────────────────────────────────┐   │   │   │  ║       │
│    ║  │   │   │  Q: [batch, 16 heads, seq, 128]         │   │   │   │  ║       │
│    ║  │   │   │  K: [batch, 4 heads, seq, 128]  (GQA)   │   │   │   │  ║       │
│    ║  │   │   │  V: [batch, 4 heads, seq, 128]          │   │   │   │  ║       │
│    ║  │   │   │           + RoPE                         │   │   │   │  ║       │
│    ║  │   │   │           + [Value Gate G2] (GSA only)   │   │   │   │  ║       │
│    ║  │   │   │           + [Sparse Indexer] (GSA only)  │   │   │   │  ║       │
│    ║  │   │   │           + [Output Gate G1] (GSA only)  │   │   │   │  ║       │
│    ║  │   │   └─────────────────────────────────────────┘   │   │   │  ║       │
│    ║  │   └────────────────────────┬────────────────────────┘   │   │  ║       │
│    ║  │        │                   │                            │   │  ║       │
│    ║  │        └───── (+) ◄────────┘  (residual)               │   │  ║       │
│    ║  │                │                                        │   │  ║       │
│    ║  │   hidden_states──┬─────────────────────────────────┐   │   │  ║       │
│    ║  │        │         │                                 │   │   │  ║       │
│    ║  │        ▼         │                                 │   │   │  ║       │
│    ║  │   ┌─────────┐    │                                 │   │   │  ║       │
│    ║  │   │ RMSNorm │    │                                 │   │   │  ║       │
│    ║  │   └────┬────┘    │                                 │   │   │  ║       │
│    ║  │        ▼         │                                 │   │   │  ║       │
│    ║  │   ┌─────────────────────────────────────────────┐ │   │   │  ║       │
│    ║  │   │              MoE BLOCK                      │ │   │   │  ║       │
│    ║  │   │  (See detailed diagram below)               │ │   │   │  ║       │
│    ║  │   └────────────────────┬────────────────────────┘ │   │   │  ║       │
│    ║  │        │               │                          │   │   │  ║       │
│    ║  │        └───── (+) ◄────┘  (residual)             │   │   │  ║       │
│    ║  │                │                                  │   │   │  ║       │
│    ║  └────────────────┼──────────────────────────────────┘   │   │  ║       │
│    ╚═══════════════════│══════════════════════════════════════╝       │
│                        ▼                                               │
│               ┌─────────────┐                                          │
│               │   RMSNorm   │                                          │
│               └──────┬──────┘                                          │
│                      ▼                                                 │
│               ┌─────────────┐                                          │
│               │   LM Head   │  hidden_size × vocab_size                │
│               └──────┬──────┘                                          │
│                      ▼                                                 │
│               Logits [batch, seq, vocab_size]                          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### MoE Block Detail

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MoE BLOCK DETAIL                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    Input [batch, seq, hidden]                                                   │
│         │                                                                        │
│         ├──────────────────────────────────────────────────────────┐            │
│         │                                                          │            │
│         ▼                                                          ▼            │
│    ┌─────────────────────────────────────────┐          ┌────────────────────┐ │
│    │     NULL EXPERT ROUTER (Primary)        │          │   SHARED EXPERTS   │ │
│    │     (arXiv:2601.15370v1 + DeepSeek)     │          │   (Always Active)  │ │
│    │                                          │          │                    │ │
│    │  ┌─────────────────────────────────┐    │          │  Expert_S1(x)      │ │
│    │  │ 1. Linear Router                │    │          │  Expert_S2(x)      │ │
│    │  │    h → logits [B, S, N+1]       │    │          │       ...          │ │
│    │  │    (N real + 1 null)            │    │          │                    │ │
│    │  │                                 │    │          │  Output:           │ │
│    │  │ 2. Apply DeepSeek Bias          │    │          │  avg(S1, S2, ...)  │ │
│    │  │    real_logits += expert_bias   │    │          │                    │ │
│    │  │    (bias on REAL experts only)  │    │          └─────────┬──────────┘ │
│    │  │                                 │    │                    │            │
│    │  │ 3. Expand Null Copies           │    │                    │            │
│    │  │    null → M copies (M=N(1-ρ)/ρ) │    │                    │            │
│    │  │    expanded [B, S, N+M]         │    │                    │            │
│    │  │                                 │    │                    │            │
│    │  │ 4. Softmax over N+M slots       │    │                    │            │
│    │  │    probs = softmax(expanded)    │    │                    │            │
│    │  │                                 │    │                    │            │
│    │  │ 5. Top-K Selection              │    │                    │            │
│    │  │    indices = topk(expanded, k)  │    │                    │            │
│    │  │                                 │    │                    │            │
│    │  │ 6. Renormalize over REAL only   │    │                    │            │
│    │  │    weights = probs[real] / sum  │    │                    │            │
│    │  │    (null gets weight = 0)       │    │                    │            │
│    │  └─────────────────────────────────┘    │                    │            │
│    │                                          │                    │            │
│    │  Output: indices [B, S, K]               │                    │            │
│    │          weights [B, S, K]               │                    │            │
│    └───────────────┬──────────────────────────┘                    │            │
│                    │                                               │            │
│                    ▼                                               │            │
│    ┌───────────────────────────────────────────────┐              │            │
│    │              ROUTED EXPERTS                    │              │            │
│    │                                                │              │            │
│    │   ┌─────────┐ ┌─────────┐     ┌─────────┐    │              │            │
│    │   │Expert 0 │ │Expert 1 │ ... │Expert N │    │              │            │
│    │   │(Gated)  │ │(Gated)  │     │(Gated)  │    │              │            │
│    │   └────┬────┘ └────┬────┘     └────┬────┘    │              │            │
│    │        │           │               │         │              │            │
│    │   ┌─────────────────────────────────────┐    │              │            │
│    │   │         NULL EXPERT                 │    │              │            │
│    │   │   Returns x × scale × 0.01          │    │              │            │
│    │   │   (Zero-compute, junk absorption)   │    │              │            │
│    │   │   Competes via M copies in routing  │    │              │            │
│    │   └─────────────────────────────────────┘    │              │            │
│    │                                                │              │            │
│    │   Output: Σ weight_i × Expert_i(x)            │              │            │
│    │   (null expert has weight=0, contributes 0)   │              │            │
│    └────────────────────┬───────────────────────────┘              │            │
│                         │                                          │            │
│                         ▼                                          ▼            │
│                    routed_output                            shared_output       │
│                         │                                          │            │
│                         └──────────────── (+) ◄────────────────────┘            │
│                                            │                                    │
│                                            ▼                                    │
│                                     Output [batch, seq, hidden]                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Gated Expert Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GATED EXPERT (SwiGLU + Dual Gating)                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    Input x [batch, seq, hidden]                                                 │
│         │                                                                        │
│         ├───────────────────────────────────┐  (save for G1)                   │
│         │                                   │                                   │
│         ▼                                   │                                   │
│    ┌─────────────────────────┐              │                                   │
│    │ G2: Input Gate (Optional)│              │                                   │
│    │ g2 = σ(x × W_input_gate) │              │                                   │
│    │ x' = x ⊙ g2              │              │                                   │
│    └───────────┬─────────────┘              │                                   │
│                │                            │                                   │
│                ▼                            │                                   │
│    ┌─────────────────────────────────────┐  │                                   │
│    │           SwiGLU FFN                 │  │                                   │
│    │                                      │  │                                   │
│    │   gate = x' × W1    [hidden → inter] │  │                                   │
│    │   up   = x' × W3    [hidden → inter] │  │                                   │
│    │                                      │  │                                   │
│    │   hidden = SiLU(gate) ⊙ up           │  │                                   │
│    │                                      │  │                                   │
│    │   output = hidden × W2 [inter → hidden]│ │                                   │
│    └───────────┬─────────────────────────┘  │                                   │
│                │                            │                                   │
│                ▼                            │                                   │
│    ┌─────────────────────────┐              │                                   │
│    │ G1: Output Gate (Optional)│◄────────────┘  (uses original x)              │
│    │ g1 = σ(x × W_output_gate) │                                                │
│    │ output = output ⊙ g1      │                                                │
│    └───────────┬─────────────┘                                                  │
│                │                                                                 │
│                ▼                                                                 │
│    Output [batch, seq, hidden]                                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details with Code

### 1. Router - NullExpertRouter (Primary)

**File:** [model/router.py](model/router.py)

Your architecture uses the **NullExpertRouter** based on the null expert paper (arXiv:2601.15370v1) with DeepSeek-V3 style load balancing. This is the **primary router** used in your configuration.

#### NullExpertRouter - Token-choice with Null Copies & DeepSeek Load Balancing

```python
class NullExpertRouter(nn.Module):
    """
    Token-choice router with null expert copies (arXiv:2601.15370v1).
    Combined with DeepSeek-V3 style bias-only load balancing.

    Key Innovation:
    - N real expert logits + 1 null logit
    - Duplicate null logit M times for top-k selection
    - M = N(1-ρ)/ρ where ρ is data_sparsity (e.g., ρ=0.8 → M=10 for N=40)
    - Softmax over expanded logits (N + M slots)
    - Renormalize gating weights over selected REAL experts only
    - DeepSeek-V3 bias adjustment for load balancing (bias only on real experts)
    """

    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        self.router_config = config.router

        self.num_real = config.num_routed_experts  # N real experts
        self.null_index = self.num_real  # Single null expert index

        # Compute M null copies based on data sparsity ρ
        self.num_null_copies = self._compute_null_copies()
        self.num_slots = self.num_real + self.num_null_copies  # N + M total slots

        # Simple linear router: hidden → N+1 logits (N real + 1 null)
        self.router = nn.Linear(config.hidden_size, self.num_real + 1, bias=True)

        # DeepSeek-V3 style: Biases for REAL experts only (null untouched)
        self.register_buffer('expert_bias', torch.zeros(self.num_real))

        # Tracking for load balancing
        self.register_buffer('expert_counts', torch.zeros(self.num_real + 1))
        self.register_buffer('total_tokens', torch.tensor(0, dtype=torch.long))

        self.null_indices = {self.null_index}

    def _compute_null_copies(self) -> int:
        """
        Compute M null copies based on data sparsity ρ.
        Formula: M = N × (1-ρ) / ρ

        Example: N=40 experts, ρ=0.8 → M = 40 × 0.2 / 0.8 = 10 null copies
        """
        if self.router_config.null_copies > 0:
            return self.router_config.null_copies
        rho = self.router_config.data_sparsity
        if rho is None or rho >= 1.0:
            return 0
        rho = max(rho, 1e-6)
        return max(0, int(round(self.num_real * (1 - rho) / rho)))

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape

        # Step 1: Compute logits for N real experts + 1 null expert
        logits = self.router(hidden_states)  # [B, S, N+1]
        real_logits = logits[..., :self.num_real]
        null_logit = logits[..., self.num_real:self.num_real + 1]

        # Step 2: Apply DeepSeek-V3 bias to REAL experts only (null untouched)
        real_logits = real_logits + self.expert_bias.view(1, 1, -1)

        # Step 3: Expand null logit M times for data sparsity
        if self.num_null_copies > 0:
            null_logits = null_logit.expand(batch_size, seq_len, self.num_null_copies)
            expanded_logits = torch.cat([real_logits, null_logits], dim=-1)
        else:
            expanded_logits = real_logits

        # Step 4: Softmax over expanded logits (N + M slots)
        probs = F.softmax(expanded_logits, dim=-1)

        # Step 5: Top-k selection over expanded logits
        top_k = min(self.router_config.top_k, expanded_logits.shape[-1])
        _, top_k_indices = torch.topk(expanded_logits, k=top_k, dim=-1)

        # Step 6: Get probabilities for selected experts
        selected_probs = probs.gather(-1, top_k_indices)

        # Step 7: Identify real vs null selections
        is_real = top_k_indices < self.num_real

        # Step 8: Renormalize weights over REAL experts only
        # Null experts get weight 0 in final gating
        real_weights = selected_probs * is_real
        denom = real_weights.sum(dim=-1, keepdim=True)
        gating_weights = real_weights / (denom + 1e-9)

        # Step 9: Map all null copies to single null expert index
        mapped_indices = top_k_indices.clone()
        mapped_indices[mapped_indices >= self.num_real] = self.null_index

        return mapped_indices, gating_weights, aux_info

    def update_expert_bias(self) -> Dict:
        """
        DeepSeek-V3 style bias update for load balancing.

        Algorithm:
            target = total_tokens / num_real_experts
            if expert_count > target × 1.1: decrease bias
            if expert_count < target × 0.9: increase bias
            bias = clamp(bias + adjustment, bias_min, bias_max)
        """
        if self.total_tokens == 0:
            return {'updated': False}

        utilization = self.expert_counts / self.total_tokens.float()
        real_util = utilization[:self.num_real]  # Only real experts
        target = 1.0 / max(self.num_real, 1)

        adjustments = torch.zeros_like(self.expert_bias)

        with torch.no_grad():
            # Decrease bias for overloaded experts
            overloaded = real_util > target * 1.1
            adjustments[overloaded] = -self.router_config.bias_update_speed

            # Increase bias for underloaded experts
            underloaded = real_util < target * 0.9
            adjustments[underloaded] = self.router_config.bias_update_speed

            # Apply with clamping
            self.expert_bias.add_(adjustments)
            self.expert_bias.clamp_(
                self.router_config.bias_clamp_min,
                self.router_config.bias_clamp_max
            )

        # Reset tracking
        self.expert_counts.zero_()
        self.total_tokens.zero_()

        return metrics
```

**Key Design Decisions:**

1. **Simple Linear Router**: Just `nn.Linear(hidden_size, N+1)` - no multi-head complexity
2. **Null Copies for Data Sparsity**: M copies of null logit compete in top-k selection
3. **Bias on Real Experts Only**: DeepSeek-V3 style - null logit untouched by bias
4. **Renormalize over Real**: Final gating weights only count real experts (null gets 0 weight)

---

### 2. Experts (Gated, Null, Shared)

**File:** [model/expert.py](model/expert.py)

#### GatedExpert - SwiGLU FFN with Optional Dual Gating

```python
class GatedExpert(nn.Module):
    """
    Expert FFN with SwiGLU and optional dual gating (G1+G2).

    Dual gating from GSA paper:
    - G2 (Input Gate): Suppresses uninformative input dimensions BEFORE computation
    - G1 (Output Gate): Provides "do nothing" pathway AFTER computation
    """

    def __init__(self, hidden_size: int, intermediate_size: int, config: ExpertConfig):
        super().__init__()

        # SwiGLU FFN projections
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # Gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # Down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # Up

        self.act = SwiGLU()  # SiLU(gate) * up

        # Dual gating (optional)
        if config.use_dual_gating:
            self.input_gate = nn.Linear(hidden_size, hidden_size, bias=True)   # G2
            self.output_gate = nn.Linear(hidden_size, hidden_size, bias=True)  # G1
            # Initialize for σ(·) ≈ 0.5 at start
            nn.init.constant_(self.input_gate.bias, config.gate_bias_init)
            nn.init.constant_(self.output_gate.bias, config.gate_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_original = x

        # G2: Input gating (suppress uninformative dimensions)
        if self.use_dual_gating:
            g2 = torch.sigmoid(self.input_gate(x))
            x = x * g2

        # SwiGLU FFN
        gate = self.w1(x)
        up = self.w3(x)
        hidden = self.act(gate, up)  # SiLU(gate) * up
        output = self.w2(hidden)

        # G1: Output gating (provides "do nothing" pathway)
        if self.use_dual_gating:
            g1 = torch.sigmoid(self.output_gate(x_original))  # Uses ORIGINAL input
            output = output * g1

        return output
```

#### NullExpert - Zero-Compute Pathway

```python
class NullExpert(nn.Module):
    """
    Null Expert: Zero-compute pathway for junk token absorption.

    Purpose: Provides explicit "do nothing" that competes in routing.

    Target null routing rates:
    - Junk tokens: 60-80% should route to null
    - Signal tokens: <10% should route to null

    How router learns null routing:
    - Important tokens → null hurts loss → gradient pushes away from null
    - Junk tokens → null doesn't hurt loss → no gradient pressure
    """

    def __init__(self, hidden_size: int, learnable_scale: bool = True):
        super().__init__()
        if learnable_scale:
            # Tiny learnable scale for gradient flow
            self.scale = nn.Parameter(torch.tensor(0.001))
        else:
            self.register_buffer('scale', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Near-zero output: allows gradient flow while effectively zero
        return x * (self.scale * 0.01)

    @property
    def num_params(self) -> int:
        return 1 if isinstance(self.scale, nn.Parameter) else 0
```

#### SharedExpert - Always-Active Expert

```python
class SharedExpert(GatedExpert):
    """
    Shared Expert: Always-active expert for common patterns.

    Handles patterns that apply to most tokens:
    - Common syntax (articles, prepositions)
    - Formatting patterns
    - Language structure

    NOT selected by router - always active for ALL tokens.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, config: ExpertConfig):
        super().__init__(hidden_size, intermediate_size, config)
        self.is_shared = True
```

#### ExpertContainer - Expert Management

```python
class ExpertContainer(nn.Module):
    """Container for all experts in a layer."""

    def __init__(self, config: MoEModelConfig):
        super().__init__()

        # Routed experts (selected by router)
        self.routed_experts = nn.ModuleList([
            GatedExpert(config.hidden_size, config.expert.intermediate_size, config.expert)
            for _ in range(config.num_routed_experts)
        ])

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            SharedExpert(config.hidden_size, config.expert.intermediate_size, config.expert)
            for _ in range(config.num_shared_experts)
        ])

        # Null experts (zero-compute)
        self.null_experts = nn.ModuleList([
            NullExpert(config.hidden_size)
            for _ in range(config.num_null_experts)
        ])

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """All shared experts process ALL tokens, outputs averaged."""
        if len(self.shared_experts) == 0:
            return torch.zeros_like(x)
        outputs = [expert(x) for expert in self.shared_experts]
        return sum(outputs) / len(self.shared_experts)

    def forward_routed(self, x, expert_indices, gating_weights) -> torch.Tensor:
        """Compute weighted combination of selected routed experts."""
        output = torch.zeros_like(x)
        flat_x = x.view(-1, hidden)

        for k in range(top_k):
            expert_ids = expert_indices[:, k]
            weights = gating_weights[:, k:k+1]

            for expert_id in expert_ids.unique():
                if expert_id >= self.routed_end:  # Skip null experts
                    continue
                mask = (expert_ids == expert_id)
                expert = self.get_expert(expert_id.item())
                expert_output = expert(flat_x[mask])
                output[mask] += weights[mask] * expert_output

        return output
```

---

### 3. Load Balancer

**File:** [model/load_balancer.py](model/load_balancer.py)

```python
class LoadBalancer:
    """
    Loss-free load balancer for MoE routing (DeepSeek-V3 approach).

    Key Formula:
        bias_i(t+1) = clamp(bias_i(t) + γ × sign(target - count_i), bias_min, bias_max)

    Where:
    - γ = 0.001 (update speed)
    - target = total_tokens / num_experts
    - bias affects routing selection but NOT gating weights
    """

    def __init__(self, num_experts: int, config: LoadBalanceConfig):
        self.expert_bias = torch.zeros(num_experts)
        self.expert_counts = torch.zeros(num_experts)
        self.config = config

    def accumulate(self, expert_indices: torch.Tensor):
        """Accumulate expert usage counts from routing decisions."""
        flat_indices = expert_indices.view(-1)
        counts = torch.bincount(flat_indices.long(), minlength=self.num_experts).float()
        self.expert_counts += counts
        self.total_tokens += flat_indices.numel()

    def update(self) -> LoadBalanceMetrics:
        """Update expert biases based on accumulated counts."""
        utilization = self.expert_counts / self.total_tokens
        target = 1.0 / self.num_experts

        adjustments = torch.zeros_like(self.expert_bias)

        with torch.no_grad():
            # Decrease bias for overloaded experts (> 110% of target)
            overloaded = utilization > target * 1.1
            adjustments[overloaded] = -self.config.bias_update_speed

            # Increase bias for underloaded experts (< 90% of target)
            underloaded = utilization < target * 0.9
            adjustments[underloaded] = self.config.bias_update_speed

            # Apply with clamping
            self.expert_bias += adjustments
            self.expert_bias.clamp_(self.config.bias_min, self.config.bias_max)

        # Handle dead experts (auto-revival)
        dead_experts = (utilization < 0.01).nonzero().squeeze(-1).tolist()
        if self.config.auto_revive:
            for expert_id in dead_experts:
                self.expert_bias[expert_id] += self.config.revive_bias_boost

        # Reset counters
        self.expert_counts.zero_()
        self.total_tokens = 0

        return self._compute_metrics(utilization, adjustments, dead_experts)


class NullRoutingMonitor:
    """
    Monitor null expert routing for junk token absorption.

    Targets:
    - Junk tokens: 60-80% should route to null
    - Signal tokens: <10% should route to null
    """

    def __init__(self, null_expert_ids, junk_token_ids):
        self.null_expert_ids = set(null_expert_ids)
        self.junk_token_ids = set(junk_token_ids)
        self.target_junk_rate = (0.6, 0.8)
        self.target_signal_rate = (0.0, 0.1)

    def update(self, token_ids: torch.Tensor, expert_indices: torch.Tensor):
        """Update null routing statistics."""
        junk_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for tid in self.junk_token_ids:
            junk_mask |= (token_ids == tid)

        # Check if routed to null
        null_mask = torch.zeros_like(expert_indices[:, 0], dtype=torch.bool)
        for nid in self.null_expert_ids:
            null_mask |= (expert_indices == nid).any(dim=-1)

        self.junk_total += junk_mask.sum().item()
        self.junk_to_null += (junk_mask & null_mask).sum().item()
        self.signal_total += (~junk_mask).sum().item()
        self.signal_to_null += ((~junk_mask) & null_mask).sum().item()

    def check_health(self) -> Dict:
        junk_rate = self.junk_to_null / (self.junk_total + 1e-9)
        signal_rate = self.signal_to_null / (self.signal_total + 1e-9)

        alerts = []
        if junk_rate < 0.6:
            alerts.append(f"Junk→null rate {junk_rate:.1%} below target 60%")
        if signal_rate > 0.1:
            alerts.append(f"Signal→null rate {signal_rate:.1%} above threshold 10%")

        return {'is_healthy': len(alerts) == 0, 'alerts': alerts}
```

---

### 4. MoE Block

**File:** [model/moe_block.py](model/moe_block.py)

```python
class MoEBlock(nn.Module):
    """
    Complete Mixture of Experts block combining:
    1. GSA Router - Token to expert routing
    2. Shared Experts - Always active
    3. Routed Experts - Selected by router
    4. Null Experts - Zero-compute pathway
    5. Telemetry - Health monitoring
    """

    def __init__(self, config: MoEModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Router (GSA or NullExpert based on config)
        if config.router.router_type == RouterType.NULL_EXPERT:
            self.router = NullExpertRouter(config)
        else:
            self.router = GSARouter(config)

        # Expert container
        self.experts = ExpertContainer(config)

        # Telemetry (Team 7 integration)
        self.telemetry = MoETelemetry(config)

    def forward(self, hidden_states, token_ids=None, return_router_info=False):
        """
        Forward pass:
        1. Shared experts process ALL tokens
        2. Router selects top-k experts per token
        3. Routed experts process selected tokens
        4. Combine: shared_output + routed_output
        """
        # Shared experts (always compute)
        shared_output = self.experts.forward_shared(hidden_states)

        # Routing
        expert_indices, gating_weights, router_aux = self.router(
            hidden_states, return_all_scores=return_router_info
        )

        # Routed experts
        routed_output = self.experts.forward_routed(
            hidden_states, expert_indices, gating_weights
        )

        # Combine outputs
        output = shared_output + routed_output

        # Collect telemetry
        if return_router_info:
            aux_info = self._collect_aux_info(
                expert_indices, gating_weights, router_aux, token_ids
            )

        return output, aux_info

    def post_training_step(self) -> Dict:
        """Update expert biases after training step (loss-free balancing)."""
        return self.router.update_expert_bias()


class MoETelemetry:
    """Telemetry for MoE health monitoring (Team 7 Integration)."""

    def check_health(self, expert_indices, gating_weights, expert_counts, total_tokens):
        health = {'is_healthy': True, 'alerts': [], 'metrics': {}}

        utilization = expert_counts / total_tokens.float()
        expected = 1.0 / len(utilization)

        # Check 1: Dead Experts (< 1% utilization)
        dead_mask = utilization < self.config.dead_expert_threshold
        dead_count = dead_mask.sum().item()
        if dead_count > 0:
            health['is_healthy'] = False
            health['alerts'].append({
                'type': 'dead_experts',
                'message': f'{dead_count} experts below 1% utilization'
            })

        # Check 2: Router Entropy
        probs = utilization + 1e-10
        entropy = -(probs * probs.log()).sum().item()
        normalized_entropy = entropy / math.log(len(utilization))
        if normalized_entropy < self.config.min_router_entropy:
            health['is_healthy'] = False
            health['alerts'].append({
                'type': 'low_entropy',
                'message': f'Router entropy {normalized_entropy:.2f} below threshold'
            })

        # Check 3: Load Balance (Gini Coefficient)
        sorted_util = torch.sort(utilization)[0]
        n = len(sorted_util)
        gini = (2 * torch.arange(1, n+1) - n - 1).float() @ sorted_util
        gini = gini / (n * sorted_util.sum())
        if gini > self.config.max_gini_coefficient:
            health['alerts'].append({
                'type': 'load_imbalance',
                'message': f'Gini coefficient {gini:.2f} above threshold'
            })

        health['metrics'] = {
            'dead_experts': dead_count,
            'normalized_entropy': normalized_entropy,
            'gini_coefficient': gini.item(),
        }

        return health
```

---

### 5. Attention (GQA & GSA)

**File:** [model/attention.py](model/attention.py)

#### Grouped-Query Attention (GQA)

```python
class GQAttention(nn.Module):
    """
    Grouped-Query Attention with RoPE.

    GQA uses fewer KV heads than query heads, reducing memory for KV cache.
    Configuration: 16 query heads, 4 KV heads (4:1 ratio)
    """

    def __init__(self, config: MoEModelConfig, layer_idx: int = 0):
        super().__init__()

        self.num_heads = config.attention.num_attention_heads      # 16
        self.num_kv_heads = config.attention.num_kv_heads          # 4
        self.head_dim = config.attention.head_dim                  # 128
        self.num_key_value_groups = self.num_heads // self.num_kv_heads  # 4

        # Projections (Q has more heads than K/V)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)       # 16 heads
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)    # 4 heads
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)    # 4 heads
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)

        self.rotary_emb = RotaryEmbedding(head_dim, max_position_embeddings, rope_theta)

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads for attention computation."""
        # [batch, 4, seq, 128] -> [batch, 16, seq, 128]
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        return hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim
        ).reshape(batch, self.num_heads, seq_len, head_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch, seq, num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len=seq)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Repeat KV heads to match Q heads
        k = self._repeat_kv(k)  # [batch, 16, seq, 128]
        v = self._repeat_kv(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        output = attn_weights @ v

        # Reshape and project
        output = output.transpose(1, 2).reshape(batch, seq, hidden_size)
        return self.o_proj(output)
```

#### Gated Sparse Attention (GSA)

```python
class GatedSparseAttention(nn.Module):
    """
    Gated Sparse Attention (arXiv:2601.15305v1).

    Key components:
    - Value gate (G2): V' = V ⊙ σ(h Wg_V)
    - Gated lightning indexer: low-dim scoring + top-k selection
    - Output gate (G1): O_gated = O_sparse ⊙ σ(h Wg_O)
    """

    def __init__(self, config: MoEModelConfig, layer_idx: int = 0):
        super().__init__()

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)

        # Gates (G2: value, G1: output)
        self.v_gate_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_gate_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)

        # Gated Lightning Indexer (low-dim sparse selection)
        self.indexer_q_proj = nn.Linear(hidden_size, indexer_heads * indexer_dim)
        self.indexer_k_proj = nn.Linear(hidden_size, indexer_heads * indexer_dim)
        self.indexer_head_weight_proj = nn.Linear(hidden_size, indexer_heads)
        self.indexer_head_bias = nn.Parameter(torch.zeros(indexer_heads))

        # EMA of variance for adaptive k (Equation 8)
        self.register_buffer("variance_ema", torch.tensor(1.0))

    def _compute_adaptive_k(self, scores, k_base, k_min, k_max):
        """k_t = clamp(k_base × Var(I_t) / V̄, k_min, k_max)"""
        var = scores.var(dim=-1, unbiased=False)
        with torch.no_grad():
            self.variance_ema.copy_(0.99 * self.variance_ema + 0.01 * var.mean())
        ratio = var / (self.variance_ema + 1e-9)
        return (k_base * ratio).clamp(k_min, k_max).long()

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Standard projections + RoPE
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Value Gate (G2) - suppress uninformative values
        v_gate = torch.sigmoid(self.v_gate_proj(hidden_states))
        v = v * v_gate

        # Gated Lightning Indexer (sparse selection)
        index_q = self.indexer_q_proj(hidden_states)
        index_k = self.indexer_k_proj(hidden_states)
        head_weights = torch.sigmoid(self.indexer_head_weight_proj(hidden_states))

        scores = torch.einsum("bthd,bshd->bhts", index_q, index_k)
        scores = torch.sigmoid(scores + self.indexer_head_bias)
        indexer_scores = (scores * head_weights).sum(dim=1)

        # Adaptive top-k selection
        k_t = self._compute_adaptive_k(indexer_scores, k_base, k_min, k_max)
        _, topk_indices = torch.topk(indexer_scores, k=k_max, dim=-1)

        # Sparse attention over selected tokens
        k_selected = torch.gather(k, dim=3, index=topk_indices)
        v_selected = torch.gather(v, dim=3, index=topk_indices)
        attn_output = sparse_attention(q, k_selected, v_selected)

        # Output Gate (G1) - "do nothing" pathway
        o_gate = torch.sigmoid(self.o_gate_proj(hidden_states))
        attn_output = attn_output * o_gate

        return self.o_proj(attn_output)
```

---

## Configuration Reference

**File:** [model/config.py](model/config.py)

### MoEModelConfig - Main Configuration

```python
@dataclass
class MoEModelConfig:
    """Complete MoE Model Configuration."""

    # Model identification
    model_name: str = "moe_model"
    model_type: ModelType = ModelType.MOE
    stage: int = 2  # Growth stage (1-4)

    # Core dimensions
    hidden_size: int = 2048
    num_layers: int = 20

    # MoE Configuration
    num_routed_experts: int = 40
    num_shared_experts: int = 2
    num_null_experts: int = 1
    moe_layer_frequency: int = 1  # MoE every N layers (1 = all)

    # Sub-configurations
    tokenizer: TokenizerConfig
    router: RouterConfig
    expert: ExpertConfig
    attention: AttentionConfig
    compute_budget: ComputeBudget
    telemetry: TelemetryConfig
```

### RouterConfig

```python
@dataclass
class RouterConfig:
    router_type: RouterType = RouterType.NULL_EXPERT

    # GSA Router Architecture
    num_router_heads: int = 4           # H_I indexer heads
    router_dim: int = 64                # d_I low-dim projection

    # Top-K Configuration
    top_k: int = 2                      # Base active experts
    top_k_min: int = 1                  # Min for adaptive
    top_k_max: int = 4                  # Max for adaptive
    use_adaptive_top_k: bool = True

    # Load Balancing (Loss-Free)
    use_aux_loss: bool = False          # NO auxiliary loss
    bias_update_speed: float = 0.001    # γ in DeepSeek-V3
    bias_clamp_min: float = -2.0
    bias_clamp_max: float = 2.0

    # Null Expert Configuration
    null_bias_init: float = 0.1
    data_sparsity: float = 1.0          # ρ (1.0 = no null copies)
    null_copies: int = 0                # M copies (overrides ρ)
```

### ExpertConfig

```python
@dataclass
class ExpertConfig:
    intermediate_size: int = 512        # FFN intermediate dimension
    use_dual_gating: bool = False       # Enable G1+G2 gates
    gate_bias_init: float = 0.0         # σ(0) = 0.5 at init
    expert_init_std: float = 0.02
    noise_std_for_expansion: float = 1e-4
```

### TelemetryConfig (Team 7 Integration)

```python
@dataclass
class TelemetryConfig:
    log_every_n_steps: int = 100

    # Health check thresholds
    dead_expert_threshold: float = 0.01     # <1% = dead
    overload_expert_threshold: float = 3.0  # >3× average = overloaded
    min_router_entropy: float = 0.7
    max_gini_coefficient: float = 0.5

    # Null routing alerts
    junk_null_rate_alert_low: float = 0.5   # Alert if <50%
    junk_null_rate_alert_high: float = 0.9  # Alert if >90%
    signal_null_rate_alert: float = 0.15    # Alert if >15%

    # Auto-correction
    enable_auto_correction: bool = True
```

### Stage-Specific Configurations

| Parameter | Stage 1 (1B Dense) | Stage 2 (3B MoE) | Stage 3 (8B MoE) | Stage 4 (70B MoE) |
|-----------|-------------------|------------------|------------------|-------------------|
| hidden_size | 2048 | 2048 | 2048 | 2048 |
| num_layers | 20 | 20 | 40 | 40 |
| num_routed_experts | - | 40 | 40 | 512 |
| num_shared_experts | - | 2 | 2 | 4 |
| num_null_experts | - | 20 | 20 | 256 |
| top_k | - | 4 | 4 | 8 |
| intermediate_size | 4096 | 512 | 512 | 512 |
| Total Params | ~1B | ~3B | ~8B | ~70B |
| Active Params | ~1B | ~0.7B | ~1.4B | ~2B |

---

## Working Flow

### Router Forward Pass Flow

```
Input: hidden_states [batch, seq, 2048]
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ 1. HEAD WEIGHTS: w = σ(h × W_weight)                │
│    Shape: [batch, seq, 4]  (4 indexer heads)        │
│    Values: (0, 1) - query-dependent head importance │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ 2. QUERY PROJECTION: q = h × W_query                │
│    Shape: [batch, seq, 4, 64]  (4 heads × 64 dim)   │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ 3. DOT PRODUCT WITH EXPERT KEYS                     │
│    Expert Keys: [60, 64] (40 routed + 20 null)      │
│    dot = q · K_exp^T                                │
│    Shape: [batch, seq, 4, 60]                       │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ 4. SIGMOID + HEAD BIAS                              │
│    head_scores = σ(dot + head_bias)                 │
│    Shape: [batch, seq, 4, 60]                       │
│    Values: (0, 1) - bounded per head                │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ 5. AGGREGATE ACROSS HEADS                           │
│    affinity = Σⱼ w_j × head_scores_j                │
│    Shape: [batch, seq, 60]                          │
│    Values: (0, 4) - bounded by num_heads            │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ 6. ADD EXPERT BIAS (Load Balancing)                 │
│    adjusted = affinity + expert_bias                │
│    expert_bias: updated algorithmically, not grads  │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ 7. TOP-K SELECTION (using adjusted scores)          │
│    indices = topk(adjusted, k=4)                    │
│    Shape: [batch, seq, 4]                           │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ 8. GATING WEIGHTS (from ORIGINAL scores!)           │
│    original = affinity[indices]  # NOT adjusted!    │
│    weights = original / sum(original)               │
│    Shape: [batch, seq, 4]                           │
│    Sum: 1.0 (normalized)                            │
└─────────────────────────────────────────────────────┘
          │
          ▼
Output: indices [batch, seq, 4], weights [batch, seq, 4]
```

### Training Loop

```python
from configs.config_3b_moe import get_config
from model.transformer import MoETransformer, create_model

# Create model
config = get_config()
model = create_model(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in epochs:
    for step, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            return_router_info=True
        )

        loss = outputs['loss']

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Post-training step (update load balancing biases)
        if step % 100 == 0:
            metrics = model.post_training_step()

            # Log telemetry
            for layer_name, layer_metrics in metrics.items():
                if layer_metrics.get('dead_experts', 0) > 0:
                    print(f"{layer_name}: {layer_metrics['dead_experts']} dead experts - auto-reviving")
```

### Inference with KV Cache

```python
model.eval()
with torch.no_grad():
    past_key_values = None
    generated_tokens = []

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )

        # Get next token
        next_token = outputs['logits'][:, -1, :].argmax(dim=-1)
        generated_tokens.append(next_token)

        # Update for next iteration
        input_ids = next_token.unsqueeze(-1)
        past_key_values = outputs['past_key_values']
```

---

## Growth Stages

### Stage 1 → Stage 2: Dense to MoE Expansion

```python
@torch.no_grad()
def expand_dense_to_moe(dense_model, moe_model, noise_std=1e-4):
    """
    Initialize MoE from Dense via "explosion".

    Guarantees lossless initialization:
    - All experts identical → uniform routing gives same output as dense
    - MoE_output = Σᵢ wᵢ × Expert_i(x) = FFN(x) × Σᵢ wᵢ = FFN(x)
    """
    # Copy embeddings and LM head
    moe_model.embed_tokens.weight.data.copy_(dense_model.embed_tokens.weight.data)
    moe_model.lm_head.weight.data.copy_(dense_model.lm_head.weight.data)

    # For each layer
    for moe_layer, dense_layer in zip(moe_model.layers, dense_model.layers):
        # Copy attention
        moe_layer.self_attn.q_proj.weight.data.copy_(dense_layer.self_attn.q_proj.weight.data)
        moe_layer.self_attn.k_proj.weight.data.copy_(dense_layer.self_attn.k_proj.weight.data)
        moe_layer.self_attn.v_proj.weight.data.copy_(dense_layer.self_attn.v_proj.weight.data)
        moe_layer.self_attn.o_proj.weight.data.copy_(dense_layer.self_attn.o_proj.weight.data)

        # Copy norms
        moe_layer.input_layernorm.weight.data.copy_(dense_layer.input_layernorm.weight.data)
        moe_layer.post_attention_layernorm.weight.data.copy_(dense_layer.post_attention_layernorm.weight.data)

        # Copy FFN to ALL routed experts
        if moe_layer.is_moe:
            for expert in moe_layer.ffn.experts.routed_experts:
                expert.w1.weight.data.copy_(dense_layer.ffn.w1.weight.data)
                expert.w2.weight.data.copy_(dense_layer.ffn.w2.weight.data)
                expert.w3.weight.data.copy_(dense_layer.ffn.w3.weight.data)

                # Add symmetry-breaking noise
                expert.w1.weight.data.add_(torch.randn_like(expert.w1.weight) * noise_std)
                expert.w2.weight.data.add_(torch.randn_like(expert.w2.weight) * noise_std)
                expert.w3.weight.data.add_(torch.randn_like(expert.w3.weight) * noise_std)

            # Copy to shared experts (no noise)
            for shared in moe_layer.ffn.experts.shared_experts:
                shared.w1.weight.data.copy_(dense_layer.ffn.w1.weight.data)
                shared.w2.weight.data.copy_(dense_layer.ffn.w2.weight.data)
                shared.w3.weight.data.copy_(dense_layer.ffn.w3.weight.data)
```

### Stage 3 → Stage 4: Expert Expansion (8 → 64)

```python
@torch.no_grad()
def expand_experts(source_moe, target_moe, children_per_parent=8, noise_std=1e-3):
    """
    Hierarchical expert expansion.
    Each parent expert creates N children.
    8 experts × 8 children = 64 experts
    """
    for parent_idx, parent in enumerate(source_moe.experts.routed_experts):
        for child_idx in range(children_per_parent):
            global_idx = parent_idx * children_per_parent + child_idx
            child = target_moe.experts.routed_experts[global_idx]

            # Copy parent weights
            child.w1.weight.data.copy_(parent.w1.weight.data)
            child.w2.weight.data.copy_(parent.w2.weight.data)
            child.w3.weight.data.copy_(parent.w3.weight.data)

            # Add divergence noise
            child.w1.weight.data.add_(torch.randn_like(child.w1.weight) * noise_std)
            child.w2.weight.data.add_(torch.randn_like(child.w2.weight) * noise_std)
            child.w3.weight.data.add_(torch.randn_like(child.w3.weight) * noise_std)

    # Hierarchical router key initialization
    for parent_idx in range(len(source_moe.router.expert_keys)):
        parent_key = source_moe.router.expert_keys[parent_idx]
        for child_idx in range(children_per_parent):
            global_idx = parent_idx * children_per_parent + child_idx
            target_moe.router.expert_keys[global_idx].copy_(parent_key)
            target_moe.router.expert_keys[global_idx].add_(
                torch.randn_like(parent_key) * 0.1
            )
```

---

## Quick Start

### Installation

```bash
cd moe_architecture
pip install torch triton
```

### Basic Usage

```python
from configs.config_3b_moe import get_config
from model.transformer import MoETransformer, create_model

# Create 3B MoE model
config = get_config()
model = create_model(config)

# Print configuration
print(config.summary())

# Forward pass
input_ids = torch.randint(0, config.tokenizer.vocab_size, (2, 512))
outputs = model(input_ids, return_router_info=True)

# Access outputs
logits = outputs['logits']            # [batch, seq, vocab_size]
router_info = outputs['router_info']  # Routing statistics per layer

# Check routing health
for layer_info in router_info:
    health = layer_info['health']
    if not health['is_healthy']:
        for alert in health['alerts']:
            print(f"Layer {layer_info['layer_idx']}: {alert['message']}")
```

### Command Line

```bash
# Show configuration
python main.py --config 3b_moe --action summary

# Create model checkpoint
python main.py --config 3b_moe --action create --device cpu --output /tmp/3b_test.pt
```

---

## References

1. **GSA Paper**: arXiv:2601.15305v1 - Gated Sparse Attention
2. **Null Expert Data Sparsity**: arXiv:2601.15370v1 - Token-choice with null copies
3. **DeepSeek-V3**: Loss-free load balancing via bias adjustment
4. **RoFormer**: Rotary Position Embedding (RoPE)
5. **SwiGLU**: GLU Variants Improve Transformer (Shazeer 2020)

---

## Team Integration Points

- **Team 6 (Tokenizer)**: `TokenizerConfig` defines token ID bands for junk/signal classification
- **Team 7 (Telemetry)**: `MoETelemetry` class provides health monitoring and alerts

---

## License

MIT License - Team 8 MoE Architecture

---

## Contributors

Team 8 - MoE Architecture: Expert Expansion & Routing

"""
DeepSeek-style Gated Sparse Attention (GSA) Integration
========================================================

This module integrates the DeepSeek GSA implementation into the llm_architecture.
Based on paper: arXiv:2601.15305v1

Key differences from the original gated_sparse_attention.py:
1. Proper scaling (1/sqrt(d_indexer)) in indexer
2. Inverse variance-k relationship (high variance → fewer tokens)
3. Modular gate classes with configurable initialization
4. Support for multiple adaptive-k methods (variance, entropy, learned)
5. Correct tensor shapes throughout

Usage:
    from components.attention.deepseek_gsa import DeepSeekGSA, DeepSeekGSAConfig

    config = DeepSeekGSAConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    gsa = DeepSeekGSA(config)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import RoPE from your components
try:
    from components.embeddings.rotary_embedding import (
        RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    HAS_ROPE = True
except ImportError:
    HAS_ROPE = False

# Try to import YaRN embedding
try:
    from components.embeddings.yarn_embedding import (
        DynamicYaRNEmbedding,
        YaRNRotaryEmbedding,
    )

    HAS_YARN = True
except ImportError:
    HAS_YARN = False

# Try to import Triton kernels
try:
    from components.kernels import (
        HAS_TRITON,
        pytorch_sparse_attention,
        triton_sparse_attention,
    )
except ImportError:
    HAS_TRITON = False
    triton_sparse_attention = None
    pytorch_sparse_attention = None


@dataclass
class DeepSeekGSAConfig:
    """Configuration for DeepSeek-style GSA."""

    # Model dimensions
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None  # Auto-computed if None

    # Indexer configuration
    indexer_dim: int = 64
    num_indexer_heads: int = 4
    indexer_activation: str = "sigmoid"  # "sigmoid" or "relu"

    # Sparsity configuration
    k_base: int = 2048
    k_min: int = 256
    k_max: int = 4096
    use_adaptive_k: bool = True
    adaptive_k_method: str = "variance"  # "variance", "entropy", or "learned"
    adaptive_k_temperature: float = 1.0

    # Gating configuration
    use_value_gate: bool = True
    use_output_gate: bool = True
    gate_activation: str = "sigmoid"
    gate_bias_init: float = 0.5  # sigmoid(0.5) ≈ 0.62 (moderate gating)

    # Position encoding
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0

    # YaRN configuration for extended context
    use_yarn: bool = False  # Whether to use YaRN instead of standard RoPE
    yarn_scale: float = 8.0  # Context extension factor (e.g., 8 for 4k->32k)
    yarn_original_max_position: int = 4096  # Original training context length
    yarn_beta_fast: float = 32.0  # Fast frequency threshold
    yarn_beta_slow: float = 1.0  # Slow frequency threshold
    yarn_mscale: float = 1.0  # Attention scaling factor
    yarn_mscale_all_dim: float = 0.0  # Scale based on full dimension
    use_dynamic_yarn: bool = False  # Use dynamic scaling based on input length

    # Training
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # Layer info (for proper initialization)
    num_layers: int = 32
    layer_idx: Optional[int] = None

    # Triton kernel optimization
    use_triton_kernels: bool = (
        True  # Use Triton kernels when available for long sequences
    )

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.k_min <= self.k_base <= self.k_max


class GatedLightningIndexer(nn.Module):
    """
    DeepSeek-style Gated Lightning Indexer.

    Key features:
    - Proper 1/sqrt(d_indexer) scaling
    - Sigmoid gating for bounded scores
    - Query-dependent head weighting
    """

    def __init__(
        self,
        hidden_size: int,
        indexer_dim: int = 64,
        num_indexer_heads: int = 4,
        activation: str = "sigmoid",
        use_causal_mask: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.indexer_dim = indexer_dim
        self.num_indexer_heads = num_indexer_heads
        self.activation = activation
        self.use_causal_mask = use_causal_mask

        # Indexer projections
        self.q_proj = nn.Linear(
            hidden_size, num_indexer_heads * indexer_dim, bias=False
        )
        self.k_proj = nn.Linear(hidden_size, indexer_dim, bias=False)

        # Query-dependent head weights
        self.weight_proj = nn.Linear(hidden_size, num_indexer_heads, bias=True)

        # Learnable bias per head
        self.bias = nn.Parameter(torch.zeros(num_indexer_heads))

        # CRITICAL: Scale factor for dot product (missing in original implementation)
        self.scale = 1.0 / math.sqrt(indexer_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize with Xavier for stable training."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.weight_proj.weight, gain=0.1)
        nn.init.zeros_(self.weight_proj.bias)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gated indexer scores.

        Args:
            hidden_states: Query hidden states [batch, seq_q, hidden_size]
            kv_hidden_states: KV hidden states [batch, seq_kv, hidden_size]
                             If None, uses hidden_states (self-attention)
            attention_mask: Optional mask [batch, seq_q, seq_kv] or [batch, 1, seq_q, seq_kv]

        Returns:
            scores: [batch, seq_q, seq_kv] importance scores
        """
        if kv_hidden_states is None:
            kv_hidden_states = hidden_states
        k_idx = self.k_proj(kv_hidden_states)  # [batch, seq_kv, d_idx]
        return self.forward_with_precomputed_k(
            hidden_states=hidden_states,
            k_idx=k_idx,
            attention_mask=attention_mask,
        )

    def forward_with_precomputed_k(
        self,
        hidden_states: torch.Tensor,
        k_idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gated indexer scores from query hidden states and precomputed indexer keys.

        Args:
            hidden_states: [batch, seq_q, hidden_size]
            k_idx: Precomputed indexer keys [batch, seq_kv, indexer_dim]
            attention_mask: Optional additive mask [batch, seq_q, seq_kv] or [batch, 1, seq_q, seq_kv]

        Returns:
            scores: [batch, seq_q, seq_kv]
        """
        batch_size, seq_q, _ = hidden_states.shape
        seq_kv = k_idx.shape[1]

        # Project queries to indexer space
        q_idx = self.q_proj(hidden_states)  # [batch, seq_q, n_heads * d_idx]
        q_idx = q_idx.view(batch_size, seq_q, self.num_indexer_heads, self.indexer_dim)

        # Query-dependent head weights: [batch, seq_q, n_heads]
        weights = torch.sigmoid(self.weight_proj(hidden_states))

        # QK scores per indexer head: [batch, n_heads, seq_q, seq_kv]
        raw_scores = torch.einsum("bqhd,bkd->bhqk", q_idx, k_idx) * self.scale

        bias_expanded = self.bias.view(1, -1, 1, 1)
        if self.activation == "sigmoid":
            gated_scores = torch.sigmoid(raw_scores + bias_expanded)
        elif self.activation == "relu":
            gated_scores = F.relu(raw_scores + bias_expanded)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Weight by query-dependent importance
        weights_expanded = weights.permute(0, 2, 1).unsqueeze(-1)
        weighted_scores = gated_scores * weights_expanded
        final_scores = weighted_scores.sum(dim=1)  # [batch, seq_q, seq_kv]

        # Causal masking supports decode where seq_kv > seq_q.
        if self.use_causal_mask and seq_q <= seq_kv:
            kv_offset = seq_kv - seq_q
            query_positions = kv_offset + torch.arange(
                seq_q, device=hidden_states.device
            )
            key_positions = torch.arange(seq_kv, device=hidden_states.device)
            causal_invalid = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            final_scores = final_scores.masked_fill(
                causal_invalid.unsqueeze(0), float("-inf")
            )

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask[:, 0, :, :]
            if attention_mask.size(-2) != seq_q:
                attention_mask = attention_mask[:, -seq_q:, :]
            if attention_mask.size(-1) != seq_kv:
                attention_mask = attention_mask[:, :, -seq_kv:]
            final_scores = final_scores + attention_mask

        return final_scores


class AdaptiveTopKSelector(nn.Module):
    """
    Adaptive Top-K Token Selector with DeepSeek improvements.

    Key fix: INVERSE relationship between variance and k
    - High variance → confident → fewer tokens needed
    - Low variance → uncertain → more tokens needed
    """

    def __init__(
        self,
        k_base: int = 2048,
        k_min: int = 256,
        k_max: int = 4096,
        use_adaptive: bool = True,
        temperature: float = 1.0,
        method: str = "variance",  # "variance", "entropy", or "learned"
    ):
        super().__init__()

        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.use_adaptive = use_adaptive
        self.temperature = temperature
        self.method = method

        # For learned method
        if method == "learned":
            self.k_predictor = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        scores: torch.Tensor,  # [batch, seq_q, seq_kv]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k tokens for each query.

        Returns:
            indices: [batch, seq_q, k_effective]
            mask: [batch, seq_q, k_effective] - True for valid positions
            k_values: [batch, seq_q] - actual k per token
        """
        batch_size, seq_q, seq_kv = scores.shape

        if self.use_adaptive:
            k_values = self._compute_adaptive_k(scores)
            # Use static k_max as upper bound instead of data-dependent k_values.max().item()
            # This avoids GPU->CPU sync and torch.compile graph breaks
            k_effective = min(self.k_max, seq_kv)
        else:
            k_values = torch.full(
                (batch_size, seq_q),
                min(self.k_base, seq_kv),
                device=scores.device,
                dtype=torch.long,
            )
            k_effective = min(self.k_base, seq_kv)

        k_effective = max(1, k_effective)

        # Get top-k indices
        scores_for_topk = scores.masked_fill(scores == float("-inf"), -1e9)
        _, indices = torch.topk(scores_for_topk, k_effective, dim=-1)

        # Create mask for variable k per token
        position_indices = torch.arange(k_effective, device=scores.device)
        position_indices = position_indices.view(1, 1, -1).expand(batch_size, seq_q, -1)
        mask = position_indices < k_values.unsqueeze(-1)

        # Also mask positions that were -inf in original scores
        gathered_scores = torch.gather(scores, -1, indices)
        mask = mask & (gathered_scores != float("-inf"))

        return indices, mask, k_values

    def _compute_adaptive_k(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute adaptive k values based on score distribution."""
        batch_size, seq_q, seq_kv = scores.shape

        # Mask invalid scores
        valid_mask = scores != float("-inf")
        masked_scores = scores.masked_fill(~valid_mask, 0)
        valid_counts = valid_mask.sum(dim=-1).float().clamp(min=1)

        if self.method == "variance":
            # Compute variance
            score_var = masked_scores.var(dim=-1)
            var_normalized = score_var / (score_var.mean() + 1e-8)

            # CRITICAL FIX: INVERSE relationship
            # High variance → confident → low k (fewer tokens)
            # Low variance → uncertain → high k (more tokens)
            k_scale = 1.0 / (1.0 + var_normalized * self.temperature)
            k_adaptive = self.k_base * (
                0.5 + k_scale
            )  # Range: [0.5*k_base, 1.5*k_base]

        elif self.method == "entropy":
            # Compute softmax entropy
            probs = F.softmax(
                masked_scores.masked_fill(~valid_mask, float("-inf")), dim=-1
            )
            probs = probs.masked_fill(~valid_mask, 0)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)

            # Normalize by max possible entropy
            max_entropy = torch.log(valid_counts + 1)
            normalized_entropy = entropy / (max_entropy + 1e-8)

            # High entropy (uniform) → need more tokens
            k_adaptive = self.k_base * (0.5 + normalized_entropy * self.temperature)

        elif self.method == "learned":
            score_mean = masked_scores.mean(dim=-1, keepdim=True)
            score_std = masked_scores.std(dim=-1, keepdim=True)
            score_max = masked_scores.max(dim=-1, keepdim=True).values
            stats = torch.cat([score_mean, score_std, score_max], dim=-1)

            k_scale = self.k_predictor(stats).squeeze(-1)
            k_adaptive = self.k_min + k_scale * (self.k_max - self.k_min)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Clamp and ensure <= valid positions
        k_values = torch.clamp(k_adaptive, self.k_min, self.k_max)
        k_values = torch.minimum(k_values, valid_counts)

        return k_values.long()


class FusedGates(nn.Module):
    """
    Fused G1 (output gate) + G2 (value gate) projection.

    Instead of two separate linear projections from hidden_states:
        g2 = value_gate_proj(h)    # hidden -> num_kv_heads * head_dim
        g1 = output_gate_proj(h)   # hidden -> num_heads * head_dim

    We fuse into a single projection and split:
        g_both = fused_proj(h)     # hidden -> (num_kv_heads + num_heads) * head_dim
        g2, g1 = split(g_both)

    This halves kernel launch overhead and improves memory access patterns.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias_init: float = 0.5,
        activation: str = "sigmoid",
        use_value_gate: bool = True,
        use_output_gate: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.activation = activation
        self.use_value_gate = use_value_gate
        self.use_output_gate = use_output_gate

        # Compute fused output size
        self.g2_dim = num_kv_heads * head_dim if use_value_gate else 0
        self.g1_dim = num_heads * head_dim if use_output_gate else 0
        total_dim = self.g2_dim + self.g1_dim

        if total_dim > 0:
            self.gate_proj = nn.Linear(hidden_size, total_dim, bias=True)
            self._init_weights(bias_init)

    def _init_weights(self, bias_init: float):
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, bias_init)

    def _apply_activation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(logits)
        elif self.activation == "tanh":
            return (torch.tanh(logits) + 1) / 2
        elif self.activation == "silu":
            return F.silu(logits)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden_size]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute both gates in a single fused projection.

        Returns:
            g2: Value gate [batch, seq, num_kv_heads, head_dim] or None
            g1: Output gate [batch, seq, num_heads, head_dim] or None
        """
        batch_size, seq_len, _ = hidden_states.shape

        gate_all = self.gate_proj(hidden_states)  # [batch, seq, g2_dim + g1_dim]

        g2 = None
        g1 = None

        if self.use_value_gate and self.use_output_gate:
            g2_logits, g1_logits = gate_all.split([self.g2_dim, self.g1_dim], dim=-1)
            g2 = self._apply_activation(g2_logits).view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )
            g1 = self._apply_activation(g1_logits).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
        elif self.use_value_gate:
            g2 = self._apply_activation(gate_all).view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )
        elif self.use_output_gate:
            g1 = self._apply_activation(gate_all).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )

        return g2, g1


class DeepSeekGSA(nn.Module):
    """
    DeepSeek-style Gated Sparse Attention.

    This is the corrected implementation following DeepSeek's approach:
    1. Proper indexer scaling
    2. Inverse variance-k relationship
    3. Modular gate components
    4. Correct tensor shape handling
    """

    def __init__(self, config: DeepSeekGSAConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.layer_idx = config.layer_idx
        self.attention_dropout = config.attention_dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # QKV Projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Gated Lightning Indexer
        self.indexer = GatedLightningIndexer(
            hidden_size=self.hidden_size,
            indexer_dim=config.indexer_dim,
            num_indexer_heads=config.num_indexer_heads,
            activation=config.indexer_activation,
        )

        # Adaptive Top-K Selector
        self.topk_selector = AdaptiveTopKSelector(
            k_base=config.k_base,
            k_min=config.k_min,
            k_max=config.k_max,
            use_adaptive=config.use_adaptive_k,
            temperature=config.adaptive_k_temperature,
            method=config.adaptive_k_method,
        )

        # Fused gates (G1 + G2 in a single projection)
        self.use_value_gate = config.use_value_gate
        self.use_output_gate = config.use_output_gate
        if config.use_value_gate or config.use_output_gate:
            self.gates = FusedGates(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                bias_init=config.gate_bias_init,
                activation=config.gate_activation,
                use_value_gate=config.use_value_gate,
                use_output_gate=config.use_output_gate,
            )
        else:
            self.gates = None

        # Position Embedding: YaRN or standard RoPE
        self.use_yarn = config.use_yarn
        self.rotary_emb = self._create_position_embedding(config)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        # Triton kernel support
        self.use_triton_kernels = config.use_triton_kernels and HAS_TRITON
        if config.use_triton_kernels and not HAS_TRITON:
            import warnings

            warnings.warn(
                "use_triton_kernels=True but Triton is not installed. "
                "Falling back to PyTorch implementation. "
                "Install Triton with: pip install triton"
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize QKV weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(
            self.o_proj.weight, gain=1.0 / math.sqrt(2 * self.config.num_layers)
        )

    def _create_position_embedding(self, config: DeepSeekGSAConfig):
        """
        Create position embedding based on configuration.

        Returns YaRN embedding if use_yarn=True, otherwise standard RoPE.
        """
        if config.use_yarn:
            if not HAS_YARN:
                raise ImportError(
                    "YaRN embedding requested but yarn_embedding module not found. "
                    "Ensure components/embeddings/yarn_embedding.py exists."
                )

            if config.use_dynamic_yarn:
                # Dynamic YaRN: automatically scales based on input length
                return DynamicYaRNEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta,
                    original_max_position=config.yarn_original_max_position,
                    beta_fast=config.yarn_beta_fast,
                    beta_slow=config.yarn_beta_slow,
                    mscale=config.yarn_mscale,
                )
            else:
                # Static YaRN: fixed context extension
                return YaRNRotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta,
                    original_max_position=config.yarn_original_max_position,
                    scale=config.yarn_scale,
                    beta_fast=config.yarn_beta_fast,
                    beta_slow=config.yarn_beta_slow,
                    mscale=config.yarn_mscale,
                    mscale_all_dim=config.yarn_mscale_all_dim,
                )
        else:
            # Standard RoPE
            if not HAS_ROPE:
                return None
            return RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads for GQA: [B, S, H_kv, D] -> [B, S, H, D]"""
        batch, seq_len, num_kv_heads, head_dim = x.shape
        if self.num_kv_groups == 1:
            return x
        x = x.unsqueeze(3).expand(
            batch, seq_len, num_kv_heads, self.num_kv_groups, head_dim
        )
        return x.reshape(batch, seq_len, self.num_heads, head_dim)

    def _cache_to_internal_kv(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize cache tensors to internal [B, S, H_kv, D] format.

        Supports both:
        - [B, H_kv, S, D] (preferred cache format)
        - [B, S, H_kv, D] (legacy format)
        """
        if k_cache.dim() != 4 or v_cache.dim() != 4:
            raise ValueError(
                f"Expected 4D cache tensors, got k={tuple(k_cache.shape)}, v={tuple(v_cache.shape)}"
            )

        # Preferred: [B, H_kv, S, D]
        if (
            k_cache.size(1) == self.num_kv_heads
            and v_cache.size(1) == self.num_kv_heads
        ):
            return (
                k_cache.transpose(1, 2).contiguous(),
                v_cache.transpose(1, 2).contiguous(),
            )

        # Legacy: [B, S, H_kv, D]
        if (
            k_cache.size(2) == self.num_kv_heads
            and v_cache.size(2) == self.num_kv_heads
        ):
            return k_cache, v_cache

        raise ValueError(
            "Unrecognized KV cache layout. Expected [B, H_kv, S, D] or [B, S, H_kv, D], "
            f"got k={tuple(k_cache.shape)}, v={tuple(v_cache.shape)}"
        )

    def _internal_to_cache_kv(
        self,
        k: torch.Tensor,  # [B, S, H_kv, D]
        v: torch.Tensor,  # [B, S, H_kv, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert internal KV layout to cache layout [B, H_kv, S, D]."""
        return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    def _normalize_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_q: int,
        seq_kv: int,
        batch_size: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Normalize attention mask to [B, seq_q, seq_kv] additive format.
        """
        if attention_mask is None:
            return None

        if attention_mask.dim() == 4:
            mask = attention_mask[:, 0, :, :]
        elif attention_mask.dim() == 3:
            mask = attention_mask
        else:
            raise ValueError(f"Unsupported attention_mask rank: {attention_mask.dim()}")

        if mask.size(-2) != seq_q:
            mask = mask[:, -seq_q:, :]
        if mask.size(-1) != seq_kv:
            mask = mask[:, :, -seq_kv:]
        if batch_size is not None and mask.size(0) == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]
    ]:
        """
        Forward pass for DeepSeek GSA.

        Flow: h → [Q,K,V] → [G2] → [RoPE] → [Indexer] → [Top-k] → [SDPA] → [G1] → output
        """
        batch_size, seq_len, _ = hidden_states.shape

        # === Step 1: QKV Projections ===
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        # === Step 2: Compute fused gates (single projection for G1+G2) ===
        g2, g1 = (None, None)
        if self.gates is not None:
            g2, g1 = self.gates(hidden_states)

        # === Step 2b: Apply Value Gate (G2) ===
        if g2 is not None:
            v = v * g2

        # === Step 3: Apply RoPE ===
        if self.rotary_emb is not None:
            # Transpose for RoPE: [B, S, H, D] -> [B, H, S, D]
            q_rope = q.transpose(1, 2)
            k_rope = k.transpose(1, 2)
            rope_input = hidden_states
            if use_cache and position_ids is not None:
                # For decoding with KV cache, position_ids may exceed current seq_len.
                # Ensure rotary cache covers absolute positions.
                # Guarded behind use_cache so torch.compile never traces this path
                # during training (use_cache=False), avoiding the .item() graph break.
                max_pos = int(position_ids.max()) + 1
                if max_pos > hidden_states.shape[1]:
                    rope_input = hidden_states.new_empty(1, max_pos, 1)

            cos, sin = self.rotary_emb(rope_input, position_ids)
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
            q = q_rope.transpose(1, 2)  # Back to [B, S, H, D]
            k = k_rope.transpose(1, 2)

        # === Step 4: Handle KV Cache ===
        past_indexer_k = None
        if past_key_value is not None:
            if len(past_key_value) < 2:
                raise ValueError("past_key_value must contain at least (k, v)")

            past_k, past_v = self._cache_to_internal_kv(
                past_key_value[0], past_key_value[1]
            )
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

            if len(past_key_value) > 2:
                past_indexer_k = past_key_value[2]

        kv_seq_len = k.shape[1]

        # Build indexer-key cache for full (past + current) sparse selection.
        current_indexer_k = self.indexer.k_proj(
            hidden_states
        )  # [B, seq_len, d_indexer]
        if past_indexer_k is not None:
            if (
                past_indexer_k.size(0) != batch_size
                or past_indexer_k.size(-1) != self.indexer.indexer_dim
            ):
                raise ValueError(
                    "Invalid indexer-key cache shape. "
                    f"Expected [B, S_past, {self.indexer.indexer_dim}], got {tuple(past_indexer_k.shape)}"
                )
            kv_indexer_k = torch.cat([past_indexer_k, current_indexer_k], dim=1)
        elif past_key_value is not None:
            raise ValueError(
                "Received legacy cache without indexer keys. "
                "Re-run decoding from an empty cache so indexer cache can be built."
            )
        else:
            kv_indexer_k = current_indexer_k

        if kv_indexer_k.size(1) != kv_seq_len:
            raise RuntimeError(
                f"KV/indexer cache length mismatch: kv_seq_len={kv_seq_len}, indexer_seq_len={kv_indexer_k.size(1)}"
            )

        if use_cache:
            cache_k, cache_v = self._internal_to_cache_kv(k, v)
            present_key_value = (cache_k, cache_v, kv_indexer_k)
        else:
            present_key_value = None

        # === Step 5: Gated Lightning Indexer ===
        indexer_mask = self._normalize_attention_mask(
            attention_mask,
            seq_q=seq_len,
            seq_kv=kv_seq_len,
            batch_size=batch_size,
        )
        indexer_scores = self.indexer.forward_with_precomputed_k(
            hidden_states=hidden_states,
            k_idx=kv_indexer_k,
            attention_mask=indexer_mask,
        )

        # === Step 6: Adaptive Top-K Selection ===
        selected_indices, valid_mask, k_values = self.topk_selector(indexer_scores)
        # selected_indices: [batch, seq_q, k_effective]
        # valid_mask: [batch, seq_q, k_effective]

        # === Step 7: Sparse Attention ===
        attn_output, attn_weights = self._sparse_attention(
            q,
            k,
            v,
            selected_indices,
            valid_mask,
            attention_mask,
            kv_seq_len,
            output_attentions,
        )

        # === Step 8: Output Gate (G1) ===
        if g1 is not None:
            attn_output = attn_output * g1

        # === Step 9: Output Projection ===
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        output = self.o_proj(attn_output)

        return output, attn_weights, present_key_value

    def _sparse_attention(
        self,
        q: torch.Tensor,  # [batch, seq_q, n_heads, d_head]
        k: torch.Tensor,  # [batch, seq_kv, n_kv_heads, d_head]
        v: torch.Tensor,  # [batch, seq_kv, n_kv_heads, d_head]
        indices: torch.Tensor,  # [batch, seq_q, k_selected]
        mask: torch.Tensor,  # [batch, seq_q, k_selected]
        attention_mask: Optional[torch.Tensor],
        kv_seq_len: int,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute sparse attention using selected indices.

        Uses Triton kernel when available and enabled, otherwise falls back
        to memory-optimized PyTorch implementation.
        """
        seq_q = q.shape[1]

        # Repeat KV for GQA
        k = self._repeat_kv(k)  # [batch, seq_kv, n_heads, d_head]
        v = self._repeat_kv(v)

        # Apply causal mask to indices before attention
        # For each query position q, mask out gathered keys > q
        query_positions = torch.arange(seq_q, device=q.device).view(1, -1, 1)
        kv_offset = kv_seq_len - seq_q
        adjusted_query_pos = query_positions + kv_offset
        causal_invalid = indices > adjusted_query_pos
        mask = mask & ~causal_invalid

        # Use Triton kernel if available and enabled
        if self.use_triton_kernels and triton_sparse_attention is not None:
            # Triton kernel expects mask as bool tensor
            output, _ = triton_sparse_attention(
                q, k, v, indices, mask, scale=self.scale
            )
            # Apply dropout (Triton kernel doesn't include dropout)
            if self.training and self.attention_dropout > 0:
                pass  # Dropout skipped in Triton path for now
            return output, None
        else:
            # Fall back to Optimized PyTorch implementation (chunked) from kernels module
            if pytorch_sparse_attention is not None:
                output, _ = pytorch_sparse_attention(
                    q, k, v, indices, mask, scale=self.scale
                )
                return output, None
            else:
                # Fall back to naive implementation (legacy)
                return self._sparse_attention_pytorch(
                    q,
                    k,
                    v,
                    indices,
                    mask,
                    attention_mask,
                    kv_seq_len,
                    output_attentions,
                )

    def _sparse_attention_pytorch(
        self,
        q: torch.Tensor,  # [batch, seq_q, n_heads, d_head]
        k: torch.Tensor,  # [batch, seq_kv, n_heads, d_head] (already repeated for GQA)
        v: torch.Tensor,  # [batch, seq_kv, n_heads, d_head] (already repeated for GQA)
        indices: torch.Tensor,  # [batch, seq_q, k_selected]
        mask: torch.Tensor,  # [batch, seq_q, k_selected] (causal already applied)
        attention_mask: Optional[torch.Tensor],
        kv_seq_len: int,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        PyTorch implementation of sparse attention using F.scaled_dot_product_attention.

        After gathering top-k K,V tokens, uses SDPA which dispatches to Flash Attention
        / memory-efficient attention kernels for fused softmax+matmul.
        """
        batch_size = q.shape[0]
        seq_q = q.shape[1]

        # Memory-efficient gather: avoid expanding to [batch, seq_q, seq_kv, ...]
        k_gathered = self._gather_along_seq_efficient(
            k, indices
        )  # [batch, seq_q, k_selected, n_heads, d_head]
        v_gathered = self._gather_along_seq_efficient(v, indices)

        # Reshape for SDPA: need [batch * seq_q, n_heads, 1, d_head] for q
        # and [batch * seq_q, n_heads, k_selected, d_head] for k, v
        # This treats each query position as a separate "batch" element
        n_heads = q.shape[2]
        k_selected = indices.shape[2]

        # q: [batch, seq_q, n_heads, d_head] -> [batch*seq_q, n_heads, 1, d_head]
        q_sdpa = q.reshape(batch_size * seq_q, n_heads, 1, self.head_dim)

        # k_gathered: [batch, seq_q, k_selected, n_heads, d_head] -> [batch*seq_q, n_heads, k_selected, d_head]
        k_sdpa = k_gathered.permute(0, 1, 3, 2, 4).reshape(
            batch_size * seq_q, n_heads, k_selected, self.head_dim
        )
        v_sdpa = v_gathered.permute(0, 1, 3, 2, 4).reshape(
            batch_size * seq_q, n_heads, k_selected, self.head_dim
        )

        # Build attention mask for SDPA: [batch*seq_q, 1, 1, k_selected]
        # True = valid (opposite of the additive mask convention)
        attn_mask = mask.reshape(batch_size * seq_q, 1, 1, k_selected)

        # Merge additive padding mask if present
        sparse_mask = self._normalize_attention_mask(
            attention_mask,
            seq_q=seq_q,
            seq_kv=kv_seq_len,
            batch_size=batch_size,
        )
        if sparse_mask is not None:
            gathered_sparse_mask = torch.gather(sparse_mask, dim=-1, index=indices)
            # Convert additive mask to boolean (valid where mask > -inf threshold)
            padding_valid = gathered_sparse_mask > (
                torch.finfo(gathered_sparse_mask.dtype).min / 2
            )
            padding_valid = padding_valid.reshape(batch_size * seq_q, 1, 1, k_selected)
            attn_mask = attn_mask & padding_valid

        # Use SDPA — dispatches to Flash Attention / memory-efficient kernels
        dropout_p = self.attention_dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        # output: [batch*seq_q, n_heads, 1, d_head] -> [batch, seq_q, n_heads, d_head]
        output = output.squeeze(2).reshape(batch_size, seq_q, n_heads, self.head_dim)

        return output, None

    def _gather_along_seq_efficient(
        self,
        x: torch.Tensor,  # [batch, seq_kv, n_heads, d_head]
        indices: torch.Tensor,  # [batch, seq_q, k_selected]
    ) -> torch.Tensor:
        """
        Memory-efficient gather along sequence dimension.

        Instead of expanding x to [batch, seq_q, seq_kv, n_heads, d_head] which
        causes OOM for long sequences, we use index_select with flattening.

        Returns: [batch, seq_q, k_selected, n_heads, d_head]
        """
        batch, seq_kv, n_heads, d_head = x.shape
        _, seq_q, k_selected = indices.shape

        # Reshape x: [batch, seq_kv, n_heads * d_head]
        x_flat = x.view(batch, seq_kv, n_heads * d_head)

        # Flatten indices for batch gather: [batch, seq_q * k_selected]
        indices_flat = indices.view(batch, seq_q * k_selected)

        # Expand indices for the feature dimension: [batch, seq_q * k_selected, n_heads * d_head]
        indices_expanded = indices_flat.unsqueeze(-1).expand(-1, -1, n_heads * d_head)

        # Gather: [batch, seq_q * k_selected, n_heads * d_head]
        gathered_flat = torch.gather(x_flat, dim=1, index=indices_expanded)

        # Reshape back: [batch, seq_q, k_selected, n_heads, d_head]
        gathered = gathered_flat.view(batch, seq_q, k_selected, n_heads, d_head)

        return gathered


# =============================================================================
# Convenience factory function
# =============================================================================


def create_deepseek_gsa(
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    **kwargs,
) -> DeepSeekGSA:
    """Factory function to create DeepSeek GSA."""
    config = DeepSeekGSAConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        **kwargs,
    )
    return DeepSeekGSA(config)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing DeepSeek GSA Implementation")
    print("=" * 60)

    # Configuration
    config = DeepSeekGSAConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        indexer_dim=64,
        num_indexer_heads=4,
        k_base=2048,
        k_min=256,
        k_max=4096,
    )

    gsa = DeepSeekGSA(config)

    # Test forward pass
    batch_size = 2
    seq_len = 128

    x = torch.randn(batch_size, seq_len, config.hidden_size)
    output, attn_weights, _ = gsa(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test fused gates
    if gsa.gates is not None:
        g2, g1 = gsa.gates(x)
        if g2 is not None:
            print(f"Value gate (G2) shape: {g2.shape}")
        if g1 is not None:
            print(f"Output gate (G1) shape: {g1.shape}")

    # Check indexer scores
    indexer_scores = gsa.indexer(x)
    valid_scores = indexer_scores[indexer_scores != float("-inf")]
    print(
        f"\nIndexer scores range: [{valid_scores.min().item():.3f}, {valid_scores.max().item():.3f}]"
    )
    print(f"Indexer scores should be in (0, {config.num_indexer_heads}) for sigmoid")

    # Parameter count
    total_params = sum(p.numel() for p in gsa.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n✅ DeepSeek GSA implementation test passed!")

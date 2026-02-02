"""
Attention Module
================

Grouped-Query Attention (GQA) and Gated Sparse Attention (GSA) with RoPE.

GQA reduces KV cache memory by using fewer KV heads than query heads.
This is critical for efficient inference at scale.

Configuration across stages:
- 1B/3B: 16 query heads, 4 KV heads (4:1 ratio)
- 8B/70B: 32 query heads, 8 KV heads (4:1 ratio)

Head dimension is always 128 across all stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from model.config import MoEModelConfig, AttentionConfig


def _compute_inv_freq(base: float, dim: int, device: torch.device) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))


def _compute_yarn_inv_freq(
    base: float,
    dim: int,
    device: torch.device,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    old_context_len: int,
) -> Tuple[torch.Tensor, float]:
    inv_freq_extrapolation = _compute_inv_freq(base, dim, device)
    inv_freq_interpolation = inv_freq_extrapolation / factor

    half_dim = inv_freq_extrapolation.shape[0]
    idx = torch.arange(half_dim, device=device, dtype=torch.float32)

    def _dim_from_rot(n_rot: float) -> float:
        return (
            dim
            * math.log(old_context_len / (n_rot * 2.0 * math.pi))
            / (2.0 * math.log(base))
        )

    low = max(int(math.floor(_dim_from_rot(beta_fast))), 0)
    high = min(int(math.ceil(_dim_from_rot(beta_slow))), half_dim - 1)
    span = max(high - low, 1e-3)
    ramp = ((idx - low) / span).clamp_(0, 1)

    inv_freq = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1.0 - ramp)
    attention_rescale_factor = 0.1 * math.log(factor) + 1.0
    return inv_freq, attention_rescale_factor


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    RoPE encodes position information by rotating query and key vectors.
    This allows the attention mechanism to naturally attend to relative
    positions without explicit position embeddings.
    
    Reference: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling
        self.attention_rescale_factor = 1.0
        
        # Compute inverse frequencies
        inv_freq, attention_factor = self._get_inv_freq_and_scale(device)
        self.attention_rescale_factor = attention_factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.float32
        )
    
    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        """Build cos/sin cache for given sequence length."""
        self.max_seq_len_cached = seq_len
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        # Different from paper, but it uses a different permutation
        # to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        if self.attention_rescale_factor != 1.0:
            cos = cos * self.attention_rescale_factor
            sin = sin * self.attention_rescale_factor
        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)

    def _get_inv_freq_and_scale(self, device: Optional[torch.device]) -> Tuple[torch.Tensor, float]:
        if not self.rope_scaling:
            return _compute_inv_freq(self.base, self.dim, device), 1.0
        rope_type = self.rope_scaling.get("rope_type") or self.rope_scaling.get("type")
        if rope_type != "yarn":
            return _compute_inv_freq(self.base, self.dim, device), 1.0

        factor = float(self.rope_scaling.get("factor", 1.0))
        beta_fast = float(self.rope_scaling.get("beta_fast", 32))
        beta_slow = float(self.rope_scaling.get("beta_slow", 1))
        old_ctx = int(
            self.rope_scaling.get(
                "original_max_position_embeddings", self.max_position_embeddings
            )
        )
        inv_freq, attention_factor = _compute_yarn_inv_freq(
            self.base, self.dim, device, factor, beta_fast, beta_slow, old_ctx
        )
        if "attention_factor" in self.rope_scaling:
            attention_factor = float(self.rope_scaling["attention_factor"])
        return inv_freq, attention_factor
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for positions.
        
        Args:
            x: Input tensor (for device/dtype)
            seq_len: Sequence length
            
        Returns:
            cos, sin tensors for rotary embedding
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len,
                device=x.device,
                dtype=x.dtype
            )
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine tensor [seq_len, head_dim]
        sin: Sine tensor [seq_len, head_dim]
        position_ids: Optional position IDs
        
    Returns:
        Rotated query and key tensors
    """
    # Reshape cos/sin for broadcasting
    # [seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class GQAttention(nn.Module):
    """
    Grouped-Query Attention with RoPE.
    
    GQA uses fewer KV heads than query heads, reducing memory for KV cache
    while maintaining quality. Each KV head is shared across multiple query heads.
    
    Architecture:
        Q: [batch, seq, num_heads * head_dim]
        K: [batch, seq, num_kv_heads * head_dim]
        V: [batch, seq, num_kv_heads * head_dim]
        
        KV heads are repeated to match query heads before attention.
    
    Args:
        config: Model configuration
        layer_idx: Layer index for KV cache
    """
    
    def __init__(self, config: MoEModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.attention_config = config.attention
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = self.attention_config.num_attention_heads
        self.num_kv_heads = self.attention_config.num_kv_heads
        self.head_dim = self.attention_config.head_dim
        
        # Number of query heads per KV head
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Validate dimensions
        assert self.hidden_size == self.num_heads * self.head_dim, \
            f"hidden_size ({self.hidden_size}) != num_heads ({self.num_heads}) * head_dim ({self.head_dim})"
        
        # ============================================================
        # Projections
        # ============================================================
        
        # Query projection
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.attention_config.attention_bias
        )
        
        # Key projection (fewer heads)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.attention_config.attention_bias
        )
        
        # Value projection (fewer heads)
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.attention_config.attention_bias
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=self.attention_config.attention_bias
        )
        
        # ============================================================
        # RoPE
        # ============================================================
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=self.attention_config.rope_theta,
            rope_scaling=self.attention_config.rope_scaling,
        )
        
        # ============================================================
        # Dropout
        # ============================================================
        
        self.attention_dropout = nn.Dropout(self.attention_config.attention_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02)
        
        if self.attention_config.attention_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.o_proj.bias)
    
    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match query heads.
        
        This is the key operation for GQA - we expand the fewer KV heads
        to match the number of query heads before computing attention.
        
        Args:
            hidden_states: [batch, num_kv_heads, seq_len, head_dim]
            
        Returns:
            [batch, num_heads, seq_len, head_dim]
        """
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        
        if self.num_key_value_groups == 1:
            return hidden_states
        
        # Expand and reshape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim
        )
        
        return hidden_states.reshape(batch, self.num_heads, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for GQA.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] attention mask
            position_ids: [batch, seq_len] position IDs
            past_key_value: Cached KV for incremental decoding
            use_cache: Whether to return updated KV cache
            
        Returns:
            output: [batch, seq_len, hidden_size]
            past_key_value: Updated KV cache (if use_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # ============================================================
        # Projections
        # ============================================================
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        # ============================================================
        # RoPE
        # ============================================================
        
        cos, sin = self.rotary_emb(query_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # ============================================================
        # KV Cache
        # ============================================================
        
        if past_key_value is not None:
            # Concatenate with cached KV
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # ============================================================
        # Repeat KV for GQA
        # ============================================================
        
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)
        
        # ============================================================
        # Attention
        # ============================================================
        
        # Scale factor
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
        
        # Apply attention mask (causal)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Dropout
        attn_weights = self.attention_dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value


class GatedSparseAttention(nn.Module):
    """
    Gated Sparse Attention (GSA) from arXiv:2601.15305v1.

    Key components:
    - Value gate (G2): V' = V ⊙ σ(h Wg_V)
    - Gated lightning indexer: low-dim scoring + top-k selection
    - Output gate (G1): O_gated = O_sparse ⊙ σ(h Wg_O)
    """

    def __init__(self, config: MoEModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.attention_config = config.attention
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = self.attention_config.num_attention_heads
        self.num_kv_heads = self.attention_config.num_kv_heads
        self.head_dim = self.attention_config.head_dim

        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # ============================================================
        # Projections (Q, K, V, O)
        # ============================================================
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.attention_config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.attention_config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.attention_config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=self.attention_config.attention_bias
        )

        # ============================================================
        # Gates (G2: value, G1: output)
        # ============================================================
        self.v_gate_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True
        )
        self.o_gate_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=True
        )

        # ============================================================
        # Gated Lightning Indexer
        # ============================================================
        self.indexer_dim = self.attention_config.gsa_indexer_dim
        self.indexer_heads = self.attention_config.gsa_indexer_heads

        self.indexer_q_proj = nn.Linear(
            self.hidden_size,
            self.indexer_heads * self.indexer_dim,
            bias=False
        )
        self.indexer_k_proj = nn.Linear(
            self.hidden_size,
            self.indexer_heads * self.indexer_dim,
            bias=False
        )
        self.indexer_head_weight_proj = nn.Linear(
            self.hidden_size,
            self.indexer_heads,
            bias=True
        )
        self.indexer_head_bias = nn.Parameter(torch.zeros(self.indexer_heads))

        # EMA of indexer score variance (Equation 8)
        self.register_buffer("variance_ema", torch.tensor(1.0))

        # ============================================================
        # RoPE
        # ============================================================
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=self.attention_config.rope_theta,
            rope_scaling=self.attention_config.rope_scaling,
        )

        # Dropout
        self.attention_dropout = nn.Dropout(self.attention_config.attention_dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize attention, gate, and indexer weights."""
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02)

        nn.init.normal_(self.v_gate_proj.weight, std=0.02)
        nn.init.normal_(self.o_gate_proj.weight, std=0.02)
        nn.init.constant_(self.v_gate_proj.bias, self.attention_config.gsa_gate_bias_init)
        nn.init.constant_(self.o_gate_proj.bias, self.attention_config.gsa_gate_bias_init)

        nn.init.normal_(self.indexer_q_proj.weight, std=0.02)
        nn.init.normal_(self.indexer_k_proj.weight, std=0.02)
        nn.init.normal_(self.indexer_head_weight_proj.weight, std=0.02)
        nn.init.zeros_(self.indexer_head_weight_proj.bias)
        nn.init.zeros_(self.indexer_head_bias)

        if self.attention_config.attention_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.o_proj.bias)

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads (GQA)."""
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

        if self.num_key_value_groups == 1:
            return hidden_states

        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim
        )

        return hidden_states.reshape(batch, self.num_heads, seq_len, head_dim)

    def _compute_adaptive_k(self, scores: torch.Tensor, k_base: int, k_min: int, k_max: int) -> torch.Tensor:
        """
        Compute per-query adaptive k from score variance (Equation 8).
        k_t = clamp(k_base × Var(I_t) / V̄, k_min, k_max)
        """
        var = scores.var(dim=-1, unbiased=False)
        batch_var = var.mean()
        with torch.no_grad():
            self.variance_ema.copy_(
                self.attention_config.gsa_variance_ema_decay * self.variance_ema +
                (1 - self.attention_config.gsa_variance_ema_decay) * batch_var
            )
        ratio = var / (self.variance_ema + 1e-9)
        k_t = (k_base * ratio).clamp(k_min, k_max).long()
        return k_t

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        past_key_states = None
        past_value_states = None
        past_index_k = None
        if past_key_value is not None:
            if len(past_key_value) == 3:
                past_key_states, past_value_states, past_index_k = past_key_value
            else:
                raise ValueError(
                    "GSA cache expects (key, value, indexer_key) in past_key_value."
                )

        # ============================================================
        # Projections
        # ============================================================
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, heads, seq, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # ============================================================
        # RoPE
        # ============================================================
        cos, sin = self.rotary_emb(query_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # ============================================================
        # Value Gate (G2)
        # ============================================================
        v_gate = torch.sigmoid(self.v_gate_proj(hidden_states))
        v_gate = v_gate.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states * v_gate
        
        if past_key_states is not None:
            key_states = torch.cat([past_key_states, key_states], dim=2)
            value_states = torch.cat([past_value_states, value_states], dim=2)

        # ============================================================
        # Repeat KV for GQA
        # ============================================================
        kv_key_states = key_states
        kv_value_states = value_states
        key_states = self._repeat_kv(kv_key_states)
        value_states = self._repeat_kv(kv_value_states)

        # ============================================================
        # Gated Lightning Indexer
        # ============================================================
        index_q = self.indexer_q_proj(hidden_states).view(
            batch_size, seq_len, self.indexer_heads, self.indexer_dim
        )
        index_k = self.indexer_k_proj(hidden_states).view(
            batch_size, seq_len, self.indexer_heads, self.indexer_dim
        )
        if past_index_k is not None:
            index_k = torch.cat([past_index_k, index_k], dim=1)

        head_weights = torch.sigmoid(self.indexer_head_weight_proj(hidden_states))
        # scores: [batch, indexer_heads, seq, seq]
        scores = torch.einsum("bthd,bshd->bhts", index_q, index_k)
        scores = torch.sigmoid(scores + self.indexer_head_bias.view(1, self.indexer_heads, 1, 1))
        scores = scores * head_weights.permute(0, 2, 1).unsqueeze(-1)
        indexer_scores = scores.sum(dim=1)
        indexer_scores_raw = indexer_scores

        # ============================================================
        # Apply mask for causal/other constraints
        # ============================================================
        total_len = index_k.shape[1]
        past_len = total_len - seq_len

        mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                base_mask = attention_mask[:, 0]
            else:
                base_mask = attention_mask
            if base_mask.shape[0] == 1 and batch_size > 1:
                base_mask = base_mask.expand(batch_size, -1, -1)
            if base_mask.shape[-1] != total_len and past_len > 0 and base_mask.shape[-1] == seq_len:
                past_allowed = torch.ones(
                    (batch_size, seq_len, past_len),
                    dtype=torch.bool,
                    device=hidden_states.device
                )
                mask = torch.cat([past_allowed, base_mask == 0], dim=-1)
            else:
                mask = base_mask == 0
        else:
            if past_len > 0:
                causal = torch.tril(
                    torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool)
                )
                past_allowed = torch.ones((seq_len, past_len), device=hidden_states.device, dtype=torch.bool)
                mask = torch.cat([past_allowed, causal], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)

        if mask is not None:
            indexer_scores = indexer_scores.masked_fill(~mask, float("-inf"))

        # ============================================================
        # Adaptive Top-k Selection
        # ============================================================
        k_max = min(self.attention_config.gsa_k_max, total_len)
        k_min = min(self.attention_config.gsa_k_min, k_max)
        k_base = min(self.attention_config.gsa_k_base, k_max)
        k_min = max(1, k_min)
        k_base = max(k_min, k_base)

        k_t = self._compute_adaptive_k(indexer_scores_raw, k_base, k_min, k_max)

        topk_scores, topk_indices = torch.topk(indexer_scores, k=k_max, dim=-1)

        positions = torch.arange(k_max, device=hidden_states.device).view(1, 1, -1)
        k_mask = positions < k_t.unsqueeze(-1)

        if mask is not None:
            allowed = torch.gather(mask, dim=-1, index=topk_indices)
            valid = k_mask & allowed
        else:
            valid = k_mask

        # ============================================================
        # Gather selected K/V
        # ============================================================
        gather_index = topk_indices.unsqueeze(1).unsqueeze(-1)
        gather_index = gather_index.expand(batch_size, self.num_heads, seq_len, k_max, self.head_dim)

        key_selected = torch.gather(
            key_states.unsqueeze(2).expand(batch_size, self.num_heads, seq_len, total_len, self.head_dim),
            dim=3,
            index=gather_index
        )
        value_selected = torch.gather(
            value_states.unsqueeze(2).expand(batch_size, self.num_heads, seq_len, total_len, self.head_dim),
            dim=3,
            index=gather_index
        )

        # ============================================================
        # Sparse Attention over selected tokens
        # ============================================================
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.einsum("bhsd,bhskd->bhsk", query_states, key_selected) * scale
        attn_logits = attn_logits.masked_fill(~valid.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        attn_weights = attn_weights.masked_fill(~valid.unsqueeze(1), 0.0)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_output = torch.einsum("bhsk,bhskd->bhsd", attn_weights, value_selected)

        # ============================================================
        # Output Gate (G1)
        # ============================================================
        o_gate = torch.sigmoid(self.o_gate_proj(hidden_states))
        o_gate = o_gate.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = attn_output * o_gate

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        past_key_value = (kv_key_states, kv_value_states, index_k) if use_cache else None

        return attn_output, past_key_value


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is simpler and faster than LayerNorm while providing
    similar benefits. It normalizes by the RMS of the activations
    rather than mean and variance.
    
    Formula: y = x / RMS(x) * γ
    Where RMS(x) = √(mean(x²) + ε)
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        return self.weight * hidden_states.to(input_dtype)


def create_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create causal attention mask.
    
    Returns mask where position i can only attend to positions <= i.
    Mask value is -inf for masked positions, 0 for allowed positions.
    
    Returns: [1, 1, seq_len, seq_len]
    """
    mask = torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=dtype),
        diagonal=1
    )
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)

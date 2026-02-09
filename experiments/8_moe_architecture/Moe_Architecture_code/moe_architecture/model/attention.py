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


class MLAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) from DeepSeek V2/V3.
    
    MLA compresses KV into a low-rank latent space for massive parameter reduction
    while maintaining model quality. Key innovations:
    
    1. KV Compression: hidden → c_kv (small latent) → K, V (decompressed)
    2. Decoupled RoPE: K_rope comes directly from hidden, K_nope from latent
    3. Optional Q Compression: hidden → c_q → Q (if q_lora_rank > 0)
    4. Inference Optimization: Only cache c_kv instead of full K, V
    
    Reference: DeepSeek-V2 Technical Report
    
    Architecture:
        Down projections:
            W_DKV: hidden → c_kv (KV compression)
            W_DQ: hidden → c_q (Q compression, optional)
        
        Up projections:
            W_UK: c_kv → K_nope (content-based K)
            W_UV: c_kv → V
            W_UQ: c_q → Q (nope + rope)
            W_KR: hidden → K_rope (position-based K)
    """
    
    def __init__(self, config: MoEModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.attention_config = config.attention
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = self.attention_config.num_attention_heads
        
        # MLA dimensions
        self.kv_lora_rank = self.attention_config.kv_lora_rank  # c_kv
        self.q_lora_rank = self.attention_config.q_lora_rank    # c_q (0 = no compression)
        self.qk_rope_head_dim = self.attention_config.qk_rope_head_dim  # d_h^R
        self.qk_nope_head_dim = self.attention_config.qk_nope_head_dim  # d_h^C
        self.v_head_dim = self.attention_config.v_head_dim              # d_h^V
        
        # Total head dimensions
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        # ============================================================
        # Down Projections (Compression)
        # ============================================================
        
        # KV down-projection: hidden → c_kv
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank,
            bias=False
        )
        
        # Q down-projection (optional): hidden → c_q
        self.use_q_compression = self.q_lora_rank > 0
        if self.use_q_compression:
            self.q_down_proj = nn.Linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False
            )
        
        # ============================================================
        # Up Projections (Decompression)
        # ============================================================
        
        # K non-RoPE up-projection: c_kv → K_nope
        self.k_nope_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.qk_nope_head_dim,
            bias=False
        )
        
        # K RoPE projection (from hidden, not latent): hidden → K_rope
        self.k_rope_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.qk_rope_head_dim,
            bias=False
        )
        
        # V up-projection: c_kv → V
        self.v_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False
        )
        
        # Q up-projection: c_q → Q_nope + Q_rope (or hidden → Q if no compression)
        if self.use_q_compression:
            self.q_up_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False
            )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=self.attention_config.attention_bias
        )
        
        # ============================================================
        # RoPE (for decoupled rope dimensions only)
        # ============================================================
        
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim,  # Only for rope dimensions
            max_position_embeddings=config.max_position_embeddings,
            base=self.attention_config.rope_theta,
            rope_scaling=self.attention_config.rope_scaling,
        )
        
        # Dropout
        self.attention_dropout = nn.Dropout(self.attention_config.attention_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLA weights."""
        nn.init.normal_(self.kv_down_proj.weight, std=0.02)
        nn.init.normal_(self.k_nope_up_proj.weight, std=0.02)
        nn.init.normal_(self.k_rope_proj.weight, std=0.02)
        nn.init.normal_(self.v_up_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02)
        
        if self.use_q_compression:
            nn.init.normal_(self.q_down_proj.weight, std=0.02)
            nn.init.normal_(self.q_up_proj.weight, std=0.02)
        else:
            nn.init.normal_(self.q_proj.weight, std=0.02)
        
        if self.attention_config.attention_bias:
            nn.init.zeros_(self.o_proj.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for MLA.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] attention mask
            position_ids: [batch, seq_len] position IDs
            past_key_value: Cached (c_kv, k_rope) for incremental decoding
            use_cache: Whether to return updated cache
            
        Returns:
            output: [batch, seq_len, hidden_size]
            past_key_value: Updated cache (c_kv, k_rope) if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # ============================================================
        # Q Path
        # ============================================================
        
        if self.use_q_compression:
            # Compress Q: hidden → c_q → Q
            q_compressed = self.q_down_proj(hidden_states)
            query_states = self.q_up_proj(q_compressed)
        else:
            # Direct Q: hidden → Q
            query_states = self.q_proj(hidden_states)
        
        # Reshape Q to [batch, num_heads, seq_len, qk_head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.qk_head_dim
        ).transpose(1, 2)
        
        # Split Q into nope and rope components
        q_nope = query_states[..., :self.qk_nope_head_dim]
        q_rope = query_states[..., self.qk_nope_head_dim:]
        
        # ============================================================
        # KV Path
        # ============================================================
        
        # Compress KV: hidden → c_kv
        kv_compressed = self.kv_down_proj(hidden_states)
        
        # Decompress K_nope: c_kv → K_nope
        k_nope = self.k_nope_up_proj(kv_compressed).view(
            batch_size, seq_len, self.num_heads, self.qk_nope_head_dim
        ).transpose(1, 2)
        
        # Get K_rope directly from hidden (decoupled RoPE)
        k_rope = self.k_rope_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.qk_rope_head_dim
        ).transpose(1, 2)
        
        # Decompress V: c_kv → V
        value_states = self.v_up_proj(kv_compressed).view(
            batch_size, seq_len, self.num_heads, self.v_head_dim
        ).transpose(1, 2)
        
        # ============================================================
        # RoPE (only on rope components)
        # ============================================================
        
        cos, sin = self.rotary_emb(q_rope, seq_len=seq_len)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids)
        
        # ============================================================
        # KV Cache (store compressed latent + k_rope)
        # ============================================================
        
        if past_key_value is not None:
            # Unpack cached values
            past_kv_compressed, past_k_rope, past_value = past_key_value
            
            # Concatenate with current
            kv_compressed = torch.cat([past_kv_compressed, kv_compressed], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            
            # Re-decompress k_nope for full sequence
            k_nope = self.k_nope_up_proj(kv_compressed).view(
                batch_size, kv_compressed.shape[1], self.num_heads, self.qk_nope_head_dim
            ).transpose(1, 2)
        
        # Cache for next step
        if use_cache:
            past_key_value = (kv_compressed, k_rope, value_states)
        else:
            past_key_value = None
        
        # ============================================================
        # Combine K components
        # ============================================================
        
        # Concatenate k_nope and k_rope to get full K
        key_states = torch.cat([k_nope, k_rope], dim=-1)
        
        # Combine q_nope and q_rope to get full Q
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        
        # ============================================================
        # Attention
        # ============================================================
        
        # Scale factor
        scale = 1.0 / math.sqrt(self.qk_head_dim)
        
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
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value


class GatedRMSNorm(nn.Module):
    """
    Gated RMS Normalization.
    
    Applies RMSNorm and then gates the output with SiLU activation.
    Used in Qwen3 Next's linear attention for gating the output.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        
        # Gate with SiLU
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        
        return hidden_states.to(input_dtype)


class GatedDeltaNetAttention(nn.Module):
    """
    Gated Delta Rule Linear Attention from Qwen3 Next.
    
    Instead of storing full KV cache (O(seq_len)), uses a recurrent state
    of fixed size (O(key_dim × value_dim)). This enables:
    - O(1) memory for inference
    - Massive parameter reduction
    - Efficient long-context handling
    
    Key components:
    1. QKVZ projection: Combined query, key, value, and gate projection
    2. BA projection: Beta (update) and Alpha (decay) for recurrent dynamics
    3. Causal Conv1d: Temporal smoothing
    4. Delta rule: Recurrent update k_t × v_t - k_t × (k_t @ state)
    
    Training uses chunk-based processing for efficiency.
    Inference uses step-by-step recurrent updates.
    
    Reference: Qwen3 Next Technical Report, Flash Linear Attention library
    """
    
    def __init__(self, config: MoEModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.attention_config = config.attention
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        
        # Linear attention dimensions
        self.num_key_heads = self.attention_config.linear_num_key_heads
        self.num_value_heads = self.attention_config.linear_num_value_heads
        self.key_head_dim = self.attention_config.linear_key_head_dim
        self.value_head_dim = self.attention_config.linear_value_head_dim
        self.conv_kernel_size = self.attention_config.linear_conv_kernel_dim
        
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        
        # ============================================================
        # Projections
        # ============================================================
        
        # QKVZ combined projection
        # Q, K have key_dim each, V, Z (gate) have value_dim each
        self.in_proj_qkvz = nn.Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim * 2,
            bias=False
        )
        
        # Beta and Alpha projection for gating
        self.in_proj_ba = nn.Linear(
            self.hidden_size,
            self.num_value_heads * 2,
            bias=False
        )
        
        # Causal 1D convolution for temporal smoothing
        conv_dim = self.key_dim * 2 + self.value_dim  # Q, K, V (not Z)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False
        )
        
        # Output projection
        self.out_proj = nn.Linear(
            self.value_dim,
            self.hidden_size,
            bias=False
        )
        
        # Learnable decay parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        A = torch.empty(self.num_value_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Gated RMSNorm for output
        self.norm = GatedRMSNorm(self.value_head_dim, eps=1e-6)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize linear attention weights."""
        nn.init.normal_(self.in_proj_qkvz.weight, std=0.02)
        nn.init.normal_(self.in_proj_ba.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.conv1d.weight, std=0.02)
    
    def _l2norm(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """L2 normalize for feature normalization."""
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    
    def _chunk_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64
    ) -> torch.Tensor:
        """
        Chunk-based gated delta rule for training.
        
        This is a simplified PyTorch implementation of the algorithm.
        For production, use the FLA library's optimized CUDA kernels.
        
        Args:
            query: [batch, heads, seq, key_dim]
            key: [batch, heads, seq, key_dim]
            value: [batch, heads, seq, value_dim]
            g: [batch, heads, seq] - gate/decay
            beta: [batch, heads, seq] - update rate
            
        Returns:
            output: [batch, seq, heads, value_dim]
        """
        batch_size, num_heads, seq_len, k_dim = key.shape
        v_dim = value.shape[-1]
        
        # L2 normalize Q and K for stability
        query = self._l2norm(query, dim=-1)
        key = self._l2norm(key, dim=-1)
        
        # Scale query
        scale = 1.0 / math.sqrt(k_dim)
        query = query * scale
        
        # Initialize recurrent state: [batch, heads, key_dim, value_dim]
        state = torch.zeros(
            batch_size, num_heads, k_dim, v_dim,
            dtype=query.dtype, device=query.device
        )
        
        outputs = []
        
        # Process in chunks for memory efficiency
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            chunk_len = end_i - i
            
            q_chunk = query[:, :, i:end_i]  # [B, H, chunk, K]
            k_chunk = key[:, :, i:end_i]    # [B, H, chunk, K]
            v_chunk = value[:, :, i:end_i]  # [B, H, chunk, V]
            g_chunk = g[:, :, i:end_i]      # [B, H, chunk]
            beta_chunk = beta[:, :, i:end_i]  # [B, H, chunk]
            
            chunk_output = torch.zeros(
                batch_size, num_heads, chunk_len, v_dim,
                dtype=query.dtype, device=query.device
            )
            
            # Process each position in chunk
            for t in range(chunk_len):
                q_t = q_chunk[:, :, t]  # [B, H, K]
                k_t = k_chunk[:, :, t]  # [B, H, K]
                v_t = v_chunk[:, :, t]  # [B, H, V]
                g_t = g_chunk[:, :, t].unsqueeze(-1).unsqueeze(-1).exp()  # [B, H, 1, 1]
                beta_t = beta_chunk[:, :, t].unsqueeze(-1)  # [B, H, 1]
                
                # Decay state
                state = state * g_t
                
                # Compute delta: v_t - k_t @ state
                kv_mem = torch.einsum('bhk,bhkv->bhv', k_t, state)
                delta = (v_t - kv_mem) * beta_t
                
                # Update state: state + k_t ⊗ delta
                state = state + torch.einsum('bhk,bhv->bhkv', k_t, delta)
                
                # Output: q_t @ state
                chunk_output[:, :, t] = torch.einsum('bhk,bhkv->bhv', q_t, state)
            
            outputs.append(chunk_output)
        
        # Concatenate chunks
        output = torch.cat(outputs, dim=2)  # [B, H, S, V]
        output = output.transpose(1, 2).contiguous()  # [B, S, H, V]
        
        return output
    
    def _recurrent_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step recurrent gated delta rule for inference.
        
        Args:
            query: [batch, heads, 1, key_dim]
            key: [batch, heads, 1, key_dim]
            value: [batch, heads, 1, value_dim]
            g: [batch, heads, 1] - gate/decay
            beta: [batch, heads, 1] - update rate
            recurrent_state: [batch, heads, key_dim, value_dim]
            
        Returns:
            output: [batch, 1, heads, value_dim]
            new_state: [batch, heads, key_dim, value_dim]
        """
        batch_size, num_heads, _, k_dim = key.shape
        v_dim = value.shape[-1]
        
        # L2 normalize
        query = self._l2norm(query, dim=-1)
        key = self._l2norm(key, dim=-1)
        
        # Scale
        scale = 1.0 / math.sqrt(k_dim)
        query = query * scale
        
        # Squeeze seq dim
        q_t = query.squeeze(2)  # [B, H, K]
        k_t = key.squeeze(2)
        v_t = value.squeeze(2)
        g_t = g.squeeze(2).unsqueeze(-1).unsqueeze(-1).exp()  # [B, H, 1, 1]
        beta_t = beta.squeeze(2).unsqueeze(-1)  # [B, H, 1]
        
        # Initialize state if needed
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                batch_size, num_heads, k_dim, v_dim,
                dtype=query.dtype, device=query.device
            )
        
        # Decay
        recurrent_state = recurrent_state * g_t
        
        # Delta update
        kv_mem = torch.einsum('bhk,bhkv->bhv', k_t, recurrent_state)
        delta = (v_t - kv_mem) * beta_t
        recurrent_state = recurrent_state + torch.einsum('bhk,bhv->bhkv', k_t, delta)
        
        # Output
        output = torch.einsum('bhk,bhkv->bhv', q_t, recurrent_state)
        output = output.unsqueeze(1)  # [B, 1, H, V]
        
        return output, recurrent_state
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Gated DeltaNet linear attention.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] boolean mask (optional)
            position_ids: Not used in linear attention
            past_key_value: (conv_state, recurrent_state) for inference
            use_cache: Whether to return updated state
            
        Returns:
            output: [batch, seq_len, hidden_size]
            past_key_value: (conv_state, recurrent_state) if use_cache
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply mask to padding if provided
        if attention_mask is not None and attention_mask.dim() == 2:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        
        # Check if we're in incremental decoding mode
        use_recurrent = past_key_value is not None and seq_len == 1
        
        # ============================================================
        # Project to QKVZ and BA
        # ============================================================
        
        qkvz = self.in_proj_qkvz(hidden_states)  # [B, S, key_dim*2 + value_dim*2]
        ba = self.in_proj_ba(hidden_states)      # [B, S, num_value_heads*2]
        
        # Split QKVZ: reshape to [B, S, num_key_heads, head_dim*2 + value_dim/key_heads*2]
        q = qkvz[..., :self.key_dim]
        k = qkvz[..., self.key_dim:self.key_dim*2]
        v = qkvz[..., self.key_dim*2:self.key_dim*2+self.value_dim]
        z = qkvz[..., self.key_dim*2+self.value_dim:]  # Gate
        
        # Split BA
        beta = ba[..., :self.num_value_heads]
        alpha = ba[..., self.num_value_heads:]
        
        # ============================================================
        # Causal Conv1d
        # ============================================================
        
        # Combine Q, K, V for conv (not Z)
        qkv = torch.cat([q, k, v], dim=-1)  # [B, S, key_dim*2 + value_dim]
        qkv = qkv.transpose(1, 2)  # [B, C, S]
        
        if use_recurrent:
            # Incremental mode: update conv state
            conv_state = past_key_value[0] if past_key_value else None
            if conv_state is None:
                conv_state = torch.zeros(
                    batch_size, qkv.shape[1], self.conv_kernel_size - 1,
                    dtype=qkv.dtype, device=qkv.device
                )
            
            # Concatenate with history and apply conv
            qkv_with_history = torch.cat([conv_state, qkv], dim=-1)
            conv_state = qkv_with_history[..., -(self.conv_kernel_size - 1):]
            
            # Manual conv for single step
            qkv = F.conv1d(
                qkv_with_history,
                self.conv1d.weight.squeeze(1)[:, None, :],
                groups=qkv.shape[1],
                padding=0
            )
            qkv = F.silu(qkv[..., -seq_len:])
        else:
            # Training mode: full causal conv
            qkv = self.conv1d(qkv)[..., :seq_len]
            qkv = F.silu(qkv)
            conv_state = qkv[..., -(self.conv_kernel_size - 1):]  # Save for cache
        
        qkv = qkv.transpose(1, 2)  # [B, S, C]
        
        # Split back
        q = qkv[..., :self.key_dim]
        k = qkv[..., self.key_dim:self.key_dim*2]
        v = qkv[..., self.key_dim*2:]
        
        # ============================================================
        # Reshape for multi-head attention
        # ============================================================
        
        # Expand K heads to match V heads if needed
        heads_ratio = self.num_value_heads // self.num_key_heads
        
        q = q.view(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        k = k.view(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        v = v.view(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        z = z.view(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        
        # Expand Q and K to match V heads
        if heads_ratio > 1:
            q = q.repeat_interleave(heads_ratio, dim=2)
            k = k.repeat_interleave(heads_ratio, dim=2)
        
        # Transpose to [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # ============================================================
        # Compute decay gate g
        # ============================================================
        
        beta = beta.sigmoid()  # [B, S, num_value_heads]
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)
        g = g.transpose(1, 2)  # [B, H, S]
        beta = beta.transpose(1, 2)  # [B, H, S]
        
        # ============================================================
        # Apply Gated Delta Rule
        # ============================================================
        
        if use_recurrent:
            recurrent_state = past_key_value[1] if past_key_value else None
            output, recurrent_state = self._recurrent_gated_delta_rule(
                q, k, v, g, beta, recurrent_state
            )
        else:
            output = self._chunk_gated_delta_rule(q, k, v, g, beta)
            recurrent_state = None  # Not needed for training
        
        # ============================================================
        # Gated RMSNorm and Output
        # ============================================================
        
        # Reshape for norm: [B*S*H, V]
        output_flat = output.reshape(-1, self.value_head_dim)
        z_flat = z.reshape(-1, self.value_head_dim)
        
        output = self.norm(output_flat, z_flat)
        output = output.view(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        # Prepare cache
        if use_cache:
            past_key_value = (conv_state, recurrent_state)
        else:
            past_key_value = None
        
        return output, past_key_value


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

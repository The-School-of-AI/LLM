"""
Attention Module
================

Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE).

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
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
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
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
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
    # [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
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
            base=self.attention_config.rope_theta
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

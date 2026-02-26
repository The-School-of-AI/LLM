"""
Grouped Query Attention (GQA)
=============================

Efficient attention mechanism that shares key-value heads across query heads.
Standard attention mechanism for modern LLMs (Qwen3, LLaMA 3, Mistral).

GQA provides a balance between:
- Multi-Head Attention (MHA): Each query head has its own KV heads
- Multi-Query Attention (MQA): All query heads share a single KV head
- GQA: Groups of query heads share KV heads

Reference: "GQA: Training Generalized Multi-Query Transformer Models from 
            Multi-Head Checkpoints" (Ainslie et al., 2023)
"""

import math
import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../..")
from components.embeddings.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb


class BaseAttention(nn.Module, ABC):
    """
    Abstract base class for attention mechanisms.

    All attention implementations should inherit from this class.
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, 1, seq_len, seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_value: Cached KV states for inference
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        pass


class GroupedQueryAttention(BaseAttention):
    """
    Grouped Query Attention (GQA).

    Standard attention for modern LLMs with:
    - Grouped key-value heads for efficiency
    - Rotary position embeddings
    - Flash attention support (when available)
    - KV caching for inference
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.layer_idx = layer_idx
        self.attention_dropout = attention_dropout

        # Validate dimensions
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )

        # Projections
        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=attention_bias
        )

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        # Scaling factor
        self.scale = 1.0 / math.sqrt(head_dim)

        # Check for flash attention
        self._use_flash_attn = self._check_flash_attention()

    def _check_flash_attention(self) -> bool:
        """Check if flash attention is available."""
        try:
            import flash_attn

            return hasattr(flash_attn, "flash_attn_func")
        except ImportError:
            return False

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads to match the number of query heads.

        This is the key operation for GQA - it expands the KV heads
        to be processed with all query heads.
        """
        if n_rep == 1:
            return hidden_states

        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

        # Expand and reshape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, kv_seq_len]
            position_ids: [batch, seq_len]
            past_key_value: Cached (key, value) tensors
            output_attentions: Return attention weights
            use_cache: Return updated cache

        Returns:
            (output, attention_weights, past_key_value)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # Update cache
        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        # Repeat KV heads for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        if self._use_flash_attn and not output_attentions:
            attn_output = self._flash_attention(
                query_states, key_states, value_states, attention_mask
            )
            attn_weights = None
        else:
            attn_output, attn_weights = self._standard_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
            )

        # Reshape output: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard scaled dot-product attention."""
        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )

        # Dropout
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # Compute output
        attn_output = torch.matmul(attn_weights, value)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Flash attention (when available)."""
        from flash_attn import flash_attn_func

        # Flash attention expects [batch, seq, heads, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Determine if causal
        causal = attention_mask is None

        attn_output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=causal,
        )

        # Back to [batch, heads, seq, head_dim]
        return attn_output.transpose(1, 2)


def create_causal_mask(
    seq_length: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create causal attention mask.

    Returns mask where future positions are -inf.
    Shape: [1, 1, seq_length, seq_length]
    """
    mask = torch.full(
        (seq_length, seq_length), float("-inf"), device=device, dtype=dtype
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def create_attention_mask(
    attention_mask: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert binary attention mask to additive mask.

    Args:
        attention_mask: Binary mask [batch, seq_len] where 1 = attend, 0 = ignore

    Returns:
        Additive mask [batch, 1, 1, seq_len] where 0 = attend, -inf = ignore
    """
    # Invert: 1 -> 0, 0 -> -inf
    inverted_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min
    return inverted_mask.unsqueeze(1).unsqueeze(2)

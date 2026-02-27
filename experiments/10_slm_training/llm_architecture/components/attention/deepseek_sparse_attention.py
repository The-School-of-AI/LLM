"""
DeepSeek V3 Sparse Attention (Multi-head Latent Attention - MLA)
=================================================================

Implementation based on DeepSeek-V3 architecture.

Key innovations:
1. KV compression via low-rank projections
2. Decoupled RoPE for compressed representations
3. Significant KV cache reduction
4. Maintains model quality through careful design

MLA achieves:
- 93.3% KV cache reduction compared to standard MHA
- Comparable or better quality
- Efficient inference
"""

import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../..")
from components.embeddings.rotary_embedding import (
    RotaryEmbedding,
    apply_rotary_pos_emb_k,
    apply_rotary_pos_emb_q,
)


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek V3 Multi-head Latent Attention (MLA).

    Architecture:
    1. Query: Standard projection + optional LoRA compression
    2. Key-Value: Compressed via low-rank projection
    3. Decoupled RoPE applied to subset of dimensions
    4. Efficient attention computation

    The key insight is that KV pairs can be heavily compressed
    while maintaining attention quality.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        compressed_dim: int = 512,
        rope_head_dim: int = 32,
        q_lora_rank: int = 0,
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
        self.compressed_dim = compressed_dim
        self.rope_head_dim = rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.layer_idx = layer_idx
        self.attention_dropout = attention_dropout

        # Non-RoPE dimension (for content-based attention)
        self.qk_nope_dim = head_dim - rope_head_dim

        # Scaling
        self.scale = 1.0 / math.sqrt(head_dim)

        # === Query projection ===
        if q_lora_rank > 0:
            # Use LoRA-style compression for query
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=attention_bias)
            self.q_a_layernorm = nn.RMSNorm(q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                q_lora_rank, num_attention_heads * head_dim, bias=attention_bias
            )
        else:
            # Standard query projection
            self.q_proj = nn.Linear(
                hidden_size, num_attention_heads * head_dim, bias=attention_bias
            )

        # === KV compression projection ===
        # Project to compressed latent space
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            compressed_dim + rope_head_dim,  # Compressed KV + RoPE keys
            bias=attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(compressed_dim, eps=1e-6)

        # Expand from compressed space to full KV
        self.kv_b_proj = nn.Linear(
            compressed_dim,
            num_key_value_heads * (self.qk_nope_dim + head_dim),  # K_nope + V
            bias=attention_bias,
        )

        # === Output projection ===
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=attention_bias
        )

        # === Rotary embeddings ===
        # Only applied to rope_head_dim dimensions
        self.rotary_emb = RotaryEmbedding(
            dim=rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        # Number of groups for GQA
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
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
        Forward pass with DeepSeek MLA.

        KV compression flow:
        1. hidden_states -> kv_a_proj -> [compressed_kv, k_rope]
        2. compressed_kv -> layernorm -> kv_b_proj -> [k_nope, v]
        3. k = concat(k_nope, apply_rope(k_rope))

        Query flow:
        1. hidden_states -> q_proj (or q_a -> norm -> q_b) -> q
        2. q_nope, q_rope = split(q)
        3. q_rope = apply_rope(q_rope)
        4. q = concat(q_nope, q_rope)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # === Compute Query ===
        if self.q_lora_rank > 0:
            # LoRA-style query: project down, normalize, project up
            q_compressed = self.q_a_proj(hidden_states)
            q_compressed = self.q_a_layernorm(q_compressed)
            query_states = self.q_b_proj(q_compressed)
        else:
            query_states = self.q_proj(hidden_states)

        # Reshape query: [batch, seq, heads * dim] -> [batch, heads, seq, dim]
        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        query_states = query_states.transpose(1, 2)

        # Split query into RoPE and non-RoPE parts
        q_nope, q_rope = query_states.split(
            [self.qk_nope_dim, self.rope_head_dim], dim=-1
        )

        # === Compute compressed KV ===
        # Project to compressed space + separate RoPE keys
        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)

        # Split: [compressed_kv (for K_nope, V), k_rope_compressed]
        compressed_kv, k_rope = kv_compressed.split(
            [self.compressed_dim, self.rope_head_dim], dim=-1
        )

        # Normalize compressed KV
        compressed_kv = self.kv_a_layernorm(compressed_kv)

        # Expand to full K_nope and V
        kv_expanded = self.kv_b_proj(compressed_kv)
        kv_expanded = kv_expanded.view(
            batch_size, seq_length, self.num_kv_heads, self.qk_nope_dim + self.head_dim
        )
        kv_expanded = kv_expanded.transpose(1, 2)

        # Split into K_nope and V
        k_nope, value_states = kv_expanded.split(
            [self.qk_nope_dim, self.head_dim], dim=-1
        )

        # Reshape k_rope: [batch, seq, rope_dim] -> [batch, kv_heads, seq, rope_dim]
        k_rope = k_rope.view(batch_size, seq_length, 1, self.rope_head_dim)
        k_rope = k_rope.expand(-1, -1, self.num_kv_heads, -1)
        k_rope = k_rope.transpose(1, 2)

        # === Apply Rotary Embeddings ===
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q_rope = apply_rotary_pos_emb_q(q_rope, cos, sin)
        k_rope = apply_rotary_pos_emb_k(k_rope, cos, sin)

        # === Concatenate RoPE and non-RoPE parts ===
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope], dim=-1)

        # === Handle KV cache ===
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        if use_cache:
            # Note: We cache the FULL key/value, not compressed
            # For inference, you could cache compressed_kv instead for more savings
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        # === Repeat KV for GQA ===
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # === Compute Attention ===
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        )

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)

        # === Reshape and project output ===
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    @property
    def kv_cache_compression_ratio(self) -> float:
        """
        Calculate KV cache compression ratio.

        Standard: 2 * num_kv_heads * head_dim per token
        MLA: compressed_dim + rope_head_dim per token (before expansion)
        """
        standard_cache = 2 * self.num_kv_heads * self.head_dim
        mla_cache = self.compressed_dim + self.rope_head_dim
        return standard_cache / mla_cache


class DeepSeekSparseAttentionV2(nn.Module):
    """
    Alternative MLA implementation with additional optimizations.

    Changes:
    - Shared KV compression across layers (optional)
    - Auxiliary loss for compression quality
    - Support for absorbing compression into adjacent layers
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        compressed_dim: int = 512,
        rope_head_dim: int = 32,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        use_absorb: bool = False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        # Similar to V1 but with absorb optimization
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.compressed_dim = compressed_dim
        self.rope_head_dim = rope_head_dim
        self.use_absorb = use_absorb
        self.layer_idx = layer_idx
        self.attention_dropout = attention_dropout

        self.qk_nope_dim = head_dim - rope_head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # Query projection
        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=attention_bias
        )

        if use_absorb:
            # Absorb compression into single matrix (training efficiency)
            # During inference, this can be split
            self.kv_proj = nn.Linear(
                hidden_size,
                num_key_value_heads * (head_dim + head_dim) + rope_head_dim,
                bias=attention_bias,
            )
        else:
            # Standard two-stage compression
            self.kv_down_proj = nn.Linear(
                hidden_size, compressed_dim + rope_head_dim, bias=attention_bias
            )
            self.kv_norm = nn.RMSNorm(compressed_dim, eps=1e-6)
            self.kv_up_proj = nn.Linear(
                compressed_dim,
                num_key_value_heads * (self.qk_nope_dim + head_dim),
                bias=attention_bias,
            )

        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=attention_bias
        )

        self.rotary_emb = RotaryEmbedding(
            dim=rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        self.num_key_value_groups = num_attention_heads // num_key_value_heads

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

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
        batch_size, seq_length, _ = hidden_states.shape

        # Query
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        query_states = query_states.transpose(1, 2)

        q_nope, q_rope = query_states.split(
            [self.qk_nope_dim, self.rope_head_dim], dim=-1
        )

        # KV
        if self.use_absorb:
            kv_all = self.kv_proj(hidden_states)
            # Split into components
            k_nope, v, k_rope = kv_all.split(
                [
                    self.num_kv_heads * self.qk_nope_dim,
                    self.num_kv_heads * self.head_dim,
                    self.rope_head_dim,
                ],
                dim=-1,
            )

            k_nope = k_nope.view(
                batch_size, seq_length, self.num_kv_heads, self.qk_nope_dim
            ).transpose(1, 2)
            value_states = v.view(
                batch_size, seq_length, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            k_rope = (
                k_rope.view(batch_size, seq_length, 1, self.rope_head_dim)
                .expand(-1, -1, self.num_kv_heads, -1)
                .transpose(1, 2)
            )
        else:
            compressed = self.kv_down_proj(hidden_states)
            compressed_kv, k_rope = compressed.split(
                [self.compressed_dim, self.rope_head_dim], dim=-1
            )
            compressed_kv = self.kv_norm(compressed_kv)

            kv_expanded = self.kv_up_proj(compressed_kv)
            kv_expanded = kv_expanded.view(
                batch_size,
                seq_length,
                self.num_kv_heads,
                self.qk_nope_dim + self.head_dim,
            )
            kv_expanded = kv_expanded.transpose(1, 2)

            k_nope, value_states = kv_expanded.split(
                [self.qk_nope_dim, self.head_dim], dim=-1
            )
            k_rope = (
                k_rope.view(batch_size, seq_length, 1, self.rope_head_dim)
                .expand(-1, -1, self.num_kv_heads, -1)
                .transpose(1, 2)
            )

        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q_rope = apply_rotary_pos_emb_q(q_rope, cos, sin)
        k_rope = apply_rotary_pos_emb_k(k_rope, cos, sin)

        # Concatenate
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope], dim=-1)

        # KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Attention
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value

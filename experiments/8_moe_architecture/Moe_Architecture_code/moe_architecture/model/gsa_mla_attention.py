"""
GSA-MLA Hybrid Attention
========================

Combines Multi-head Latent Attention (MLA) with Gated Sparse Attention (GSA)
for parameter reduction with sparse attention benefits.

Key features:
- MLA: KV compression into low-rank latent space
- MLA: Optional Q compression
- MLA: Decoupled RoPE (K_rope from hidden, K_nope from latent)
- GSA: Value gate (G2) for collapse prevention
- GSA: Output gate (G1) for collapse prevention
- GSA: Lightning indexer for sparse selection
- GSA: Adaptive top-k based on score variance

Reference:
- DeepSeek-V2 Technical Report (MLA)
- arXiv:2601.15305v1 (GSA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from model.config import MoEModelConfig, AttentionConfig
from model.attention import RotaryEmbedding, apply_rotary_pos_emb


class GSALatentAttention(nn.Module):
    """
    GSA-Latent Sparse Attention combining MLA compression with GSA sparsity.
    
    Architecture:
        MLA compression:
            W_DKV: hidden → c_kv (KV compression)
            W_DQ: hidden → c_q (Q compression, optional)
            W_UK: c_kv → K_nope
            W_KR: hidden → K_rope (decoupled RoPE)
            W_UV: c_kv → V
            W_UQ: c_q → Q
        
        GSA components:
            V_gate: V' = V ⊙ σ(h W_gV)
            O_gate: O' = O ⊙ σ(h W_gO)
            Indexer: Low-dim scoring + top-k selection
    
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
        
        # MLA dimensions
        self.kv_lora_rank = self.attention_config.kv_lora_rank  # c_kv
        self.q_lora_rank = self.attention_config.q_lora_rank    # c_q (0 = no compression)
        self.qk_rope_head_dim = self.attention_config.qk_rope_head_dim  # d_h^R
        self.qk_nope_head_dim = self.attention_config.qk_nope_head_dim  # d_h^C
        self.v_head_dim = self.attention_config.v_head_dim              # d_h^V
        
        # Total head dimensions
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        # GSA dimensions
        self.indexer_dim = self.attention_config.gsa_indexer_dim
        self.indexer_heads = self.attention_config.gsa_indexer_heads
        
        # ============================================================
        # MLA: Down Projections (Compression)
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
        # MLA: Up Projections (Decompression)
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
        # GSA: Gates (G1 and G2)
        # ============================================================
        
        # Value gate (G2): V' = V ⊙ σ(h W_gV)
        self.v_gate_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=True
        )
        
        # Output gate (G1): O' = O ⊙ σ(h W_gO)
        self.o_gate_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=True
        )
        
        # ============================================================
        # GSA: Gated Lightning Indexer
        # ============================================================
        
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
        """Initialize all weights."""
        # MLA projections
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
        
        # GSA gates
        nn.init.normal_(self.v_gate_proj.weight, std=0.02)
        nn.init.normal_(self.o_gate_proj.weight, std=0.02)
        nn.init.constant_(self.v_gate_proj.bias, self.attention_config.gsa_gate_bias_init)
        nn.init.constant_(self.o_gate_proj.bias, self.attention_config.gsa_gate_bias_init)
        
        # GSA indexer
        nn.init.normal_(self.indexer_q_proj.weight, std=0.02)
        nn.init.normal_(self.indexer_k_proj.weight, std=0.02)
        nn.init.normal_(self.indexer_head_weight_proj.weight, std=0.02)
        nn.init.zeros_(self.indexer_head_weight_proj.bias)
        nn.init.zeros_(self.indexer_head_bias)
    
    def _compute_adaptive_k(
        self, 
        scores: torch.Tensor, 
        k_base: int, 
        k_min: int, 
        k_max: int
    ) -> torch.Tensor:
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
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Forward pass for GSA-Latent Sparse Attention.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] attention mask
            position_ids: [batch, seq_len] position IDs
            past_key_value: Cached (c_kv, k_rope, v_gated, index_k) for decoding
            use_cache: Whether to return updated cache
            
        Returns:
            output: [batch, seq_len, hidden_size]
            past_key_value: Updated cache if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # ============================================================
        # MLA: Q Path
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
        # MLA: KV Path with Compression
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
        # MLA: RoPE (only on rope components)
        # ============================================================
        
        cos, sin = self.rotary_emb(q_rope, seq_len=seq_len)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids)
        
        # ============================================================
        # GSA: Value Gate (G2)
        # ============================================================
        
        v_gate = torch.sigmoid(self.v_gate_proj(hidden_states))
        v_gate = v_gate.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        value_states = value_states * v_gate
        
        # ============================================================
        # MLA: KV Cache
        # ============================================================
        
        # Get cached values
        past_kv_compressed = None
        past_k_rope = None
        past_value = None
        past_index_k = None
        
        if past_key_value is not None:
            if len(past_key_value) == 4:
                past_kv_compressed, past_k_rope, past_value, past_index_k = past_key_value
            else:
                raise ValueError("GSA-MLA cache expects (kv_compressed, k_rope, value, index_k)")
        
        if past_kv_compressed is not None:
            kv_compressed = torch.cat([past_kv_compressed, kv_compressed], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            
            # Re-decompress k_nope for full sequence
            k_nope = self.k_nope_up_proj(kv_compressed).view(
                batch_size, kv_compressed.shape[1], self.num_heads, self.qk_nope_head_dim
            ).transpose(1, 2)
        
        # ============================================================
        # Combine K components
        # ============================================================
        
        # Concatenate k_nope and k_rope to get full K
        key_states = torch.cat([k_nope, k_rope], dim=-1)
        
        # Combine q_nope and q_rope to get full Q
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        
        # ============================================================
        # GSA: Gated Lightning Indexer
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
        # scores: [batch, indexer_heads, seq, total_seq]
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
            # Handle various mask formats - squeeze to [batch, seq, seq] or [seq, seq]
            base_mask = attention_mask
            while base_mask.dim() > 4:
                base_mask = base_mask.squeeze(0)
            while base_mask.dim() > 3 and base_mask.shape[0] == 1 and base_mask.shape[1] == 1:
                base_mask = base_mask.squeeze(0).squeeze(0)
            if base_mask.dim() == 4:
                # [batch, 1, seq, seq] -> [batch, seq, seq]
                base_mask = base_mask[:, 0]
            elif base_mask.dim() == 2:
                # [seq, seq] -> [1, seq, seq]
                base_mask = base_mask.unsqueeze(0)
            
            # Expand batch dimension if needed
            if base_mask.shape[0] == 1 and batch_size > 1:
                base_mask = base_mask.expand(batch_size, -1, -1)
            
            # Handle past KV cache extension
            if base_mask.shape[-1] != total_len and past_len > 0 and base_mask.shape[-1] == seq_len:
                past_allowed = torch.ones(
                    (batch_size, seq_len, past_len),
                    dtype=torch.bool,
                    device=hidden_states.device
                )
                # Create boolean mask: True where attention is allowed
                mask = torch.cat([past_allowed, base_mask == 0], dim=-1)
            else:
                mask = base_mask == 0
        else:
            # Generate causal mask if none provided
            causal = torch.tril(
                torch.ones((seq_len, total_len), device=hidden_states.device, dtype=torch.bool)
            )
            mask = causal.unsqueeze(0).expand(batch_size, -1, -1)
        
        if mask is not None:
            indexer_scores = indexer_scores.masked_fill(~mask, float("-inf"))
        
        # ============================================================
        # GSA: Adaptive Top-k Selection
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
        
        gather_index_k = topk_indices.unsqueeze(1).unsqueeze(-1)
        gather_index_k = gather_index_k.expand(batch_size, self.num_heads, seq_len, k_max, self.qk_head_dim)
        
        gather_index_v = topk_indices.unsqueeze(1).unsqueeze(-1)
        gather_index_v = gather_index_v.expand(batch_size, self.num_heads, seq_len, k_max, self.v_head_dim)
        
        key_selected = torch.gather(
            key_states.unsqueeze(2).expand(batch_size, self.num_heads, seq_len, total_len, self.qk_head_dim),
            dim=3,
            index=gather_index_k
        )
        value_selected = torch.gather(
            value_states.unsqueeze(2).expand(batch_size, self.num_heads, seq_len, total_len, self.v_head_dim),
            dim=3,
            index=gather_index_v
        )
        
        # ============================================================
        # Sparse Attention over selected tokens
        # ============================================================
        
        scale = 1.0 / math.sqrt(self.qk_head_dim)
        attn_logits = torch.einsum("bhsd,bhskd->bhsk", query_states, key_selected) * scale
        attn_logits = attn_logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
        
        attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        attn_weights = attn_weights.masked_fill(~valid.unsqueeze(1), 0.0)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        attn_output = torch.einsum("bhsk,bhskd->bhsd", attn_weights, value_selected)
        
        # ============================================================
        # GSA: Output Gate (G1)
        # ============================================================
        
        o_gate = torch.sigmoid(self.o_gate_proj(hidden_states))
        o_gate = o_gate.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        attn_output = attn_output * o_gate
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        # ============================================================
        # Cache for next step
        # ============================================================
        
        if use_cache:
            past_key_value = (kv_compressed, k_rope, value_states, index_k)
        else:
            past_key_value = None
        
        return attn_output, past_key_value

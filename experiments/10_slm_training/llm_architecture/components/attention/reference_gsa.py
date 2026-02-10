"""
Reference Gated Sparse Attention (GSA)
========================================

Implementation matching Test_Code/model_1b.py lines 714-898.

Key differences from existing GSA implementations:
- Full MHA: hidden_size // num_heads (no KV compression)
- Hardcoded d_idx = 32
- Attention sinks: first 4 tokens forced to float('inf') importance
- Reversible integration support (_saved_selection for deterministic recompute)
- Memory-efficient causal mask (broadcasting trick, no T×T matrix)
- Dual gating: W_gv (value gate) + W_go (output gate)
- Self-contained YARN RoPE

Reference: arXiv:2601.15305v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.attention.gated_deltanet import DeltaNetRotaryEmbedding


class ReferenceGSA(nn.Module):
    """
    Gated Sparse Attention (GSA) matching the Test_Code reference.

    Implements adaptive sparse attention with gating for quality.
    Used for 25% of layers to complement DeltaNet's efficiency.

    Forward signature: forward(x, attention_mask=None) -> Tensor
    Returns single tensor (B, T, hidden_size), NOT a tuple.
    """

    def __init__(self, hidden_size, num_heads, max_seq_len=262144, rope_base=10000,
                 k_base=512, k_min=32, k_max=1024, indexer_heads=4,
                 rope_original_max=8192, rope_scaling_factor=32.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # Full MHA
        self.max_seq_len = max_seq_len

        # Adaptive Sparsity Hyperparams
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.indexer_heads = indexer_heads

        # Lightning Indexer (d_idx = 32, hardcoded)
        self.d_idx = 32
        self.W_Iq = nn.Linear(hidden_size, indexer_heads * self.d_idx, bias=False)
        self.W_Ik = nn.Linear(hidden_size, self.d_idx, bias=False)
        self.W_Iw = nn.Linear(hidden_size, indexer_heads, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(indexer_heads))

        self.register_buffer("variance_ema", torch.tensor(1.0))
        self.variance_alpha = 0.01

        # Attention Projections (Full MHA - no KV compression)
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dual Gating
        self.W_gv = nn.Linear(hidden_size, hidden_size, bias=False)  # Value gate
        self.W_go = nn.Linear(hidden_size, hidden_size, bias=False)  # Output gate

        # Rotary embeddings with YARN scaling
        self.rotary_emb = DeltaNetRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_base,
            original_max_position_embeddings=rope_original_max,
            scaling_factor=rope_scaling_factor
        )

        # Reversible integration support
        self._saved_selection = None

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_Iq, self.W_Ik, self.W_Iw, self.W_q, self.W_k, self.W_v,
                  self.o_proj, self.W_gv, self.W_go]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_bias)

    def forward(self, x, attention_mask=None):
        """
        Forward pass implementing Gated Sparse Attention.

        Args:
            x: Input tensor (B, T, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Output tensor (B, T, hidden_size)
        """
        B, T, C = x.shape
        device = x.device

        # Lightning Indexer
        q_I = self.W_Iq(x).view(B, T, self.indexer_heads, self.d_idx)
        k_I = self.W_Ik(x)
        w = torch.sigmoid(self.W_Iw(x))

        q_I_p = q_I.permute(0, 2, 1, 3)  # (B, indexer_heads, T, d_idx)
        k_I_p = k_I.permute(0, 2, 1).unsqueeze(1)  # (B, 1, d_idx, T)

        match_logits = torch.matmul(q_I_p, k_I_p)  # (B, indexer_heads, T, T)
        match_logits = match_logits + self.gate_bias.view(1, self.indexer_heads, 1, 1)
        match_gate = torch.sigmoid(match_logits)

        w_exp = w.permute(0, 2, 1).unsqueeze(-1)  # (B, indexer_heads, T, 1)
        importance_score = (w_exp * match_gate).sum(dim=1)  # (B, T, T)

        # Causal masking (memory-efficient: broadcasting trick, no T×T matrix allocation)
        if T > 1:
            positions = torch.arange(T, device=device)
            causal_mask_broadcast = positions.view(1, -1, 1) >= positions.view(1, 1, -1)
            importance_score_masked = importance_score.masked_fill(~causal_mask_broadcast, 0.0)
            causal_mask = causal_mask_broadcast
        else:
            importance_score_masked = importance_score
            causal_mask = None

        # Adaptive Sparsity
        var_t = importance_score_masked.var(dim=-1, unbiased=False)

        is_reversible_forward = self.training and (not torch.is_grad_enabled())
        is_reversible_reconstruct = (self.training and torch.is_grad_enabled()
                                     and getattr(self, "_saved_selection", None) is not None)

        if is_reversible_forward:
            var_t_mean = var_t.mean().detach()
            self.variance_ema.mul_(0.99).add_(var_t_mean, alpha=0.01)

        if is_reversible_reconstruct:
            k_t, top_indices = self._saved_selection
            self._saved_selection = None
            avg_V = self.variance_ema.clamp(min=1e-6)
        else:
            avg_V = self.variance_ema.clamp(min=1e-6)
            k_t_float = self.k_base * var_t / avg_V
            k_t = k_t_float.floor().clamp(min=self.k_min, max=self.k_max).long()

            if T > 1:
                importance_for_selection = importance_score.masked_fill(~causal_mask, -float('inf'))
            else:
                importance_for_selection = importance_score

            # Attention sinks: first 4 tokens always selected
            sink_size = 4
            if T > sink_size:
                sink_mask = torch.zeros_like(importance_for_selection, dtype=torch.bool)
                sink_mask[:, :, :sink_size] = True
                importance_for_selection = importance_for_selection.masked_fill(sink_mask, float('inf'))

            k_limit = min(T, max(k_t.max().item(), sink_size))
            _, top_indices = importance_for_selection.topk(k_limit, dim=-1)

            if is_reversible_forward:
                self._saved_selection = (k_t, top_indices)

        # Construct boolean mask
        k_limit = top_indices.size(-1)
        range_k = torch.arange(k_limit, device=device).unsqueeze(0).unsqueeze(0)
        keep_in_topk = range_k < k_t.unsqueeze(-1)

        selection_mask = torch.zeros_like(importance_score, dtype=torch.bool)
        selection_mask.scatter_(dim=-1, index=top_indices, src=keep_in_topk)

        if T > 1:
            selection_mask = selection_mask & causal_mask

        # Dual Gating & Attention
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        g_v = torch.sigmoid(self.W_gv(x))
        v = v * g_v

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Rotary embeddings
        if T > self.rotary_emb.cos_cached.size(0):
            self.rotary_emb._set_cos_sin_cache(T)
        cos = self.rotary_emb.cos_cached[:T].unsqueeze(0).unsqueeze(2)
        sin = self.rotary_emb.sin_cached[:T].unsqueeze(0).unsqueeze(2)
        q = self.rotary_emb._apply_rotary(q, cos, sin)
        k = self.rotary_emb._apply_rotary(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Masked attention
        min_val = torch.finfo(q.dtype).min
        bias_mask = torch.zeros_like(selection_mask, dtype=q.dtype)
        bias_mask = bias_mask.masked_fill(~selection_mask, min_val)

        if attention_mask is not None:
            bias_mask = bias_mask + attention_mask

        o_sparse = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias_mask.unsqueeze(1),
            dropout_p=0.0,
            is_causal=False
        )

        o_sparse = o_sparse.transpose(1, 2).contiguous().view(B, T, self.hidden_size)

        # Output gate
        g_o = torch.sigmoid(self.W_go(x))

        return self.o_proj(o_sparse * g_o)

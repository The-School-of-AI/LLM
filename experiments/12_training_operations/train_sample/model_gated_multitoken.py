"""
Model Architecture Module with Gated Sparse Attention (GSA) and Multi-Token Prediction (MTP)

This module contains all the neural network architecture components for the SmolLM model,
including:
- Embedding layers (Fourier-based phonetic embeddings)
- Core transformer components (RMSNorm, RotaryEmbedding, GatedSparseAttention)
- Gated Sparse Attention (GSA) - arXiv:2601.15305v1 - Replaces MultiheadLatentAttention
- Multi-Token Prediction (MTP) - DeepSeek-V3 style dual-head prediction
- Mixture-of-Experts (MoE) with null expert routing for data sparsity
- Multi-Head Composition (mHC) for multi-stream residual routing
- Complete SmolLM architecture with reversible midpoint integration

Based on model_gated.py with Multi-Token Prediction from fourier_proper_mhc_multiToken.py
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fourier_se_decoder import PFCodec
from torch.nn.functional import scaled_dot_product_attention

# ============================================================================
# Embedding Layer
# ============================================================================


class PureHybridEmbeddingTorch(nn.Module):
    """
    Pure Phonetic Embedding:
    - Fetches PF(word) for each token (precomputed).
    - Normalizes per-token (mean/std).
    - Returns normalized PFn directly.
    - NO Semantic Enrichment (SE), NO lambda_se.
    """

    def __init__(self, vocab_words: List[str], pf_codec: PFCodec):
        super().__init__()
        PF_table = pf_codec.encode_batch(vocab_words)  # (V, D_pf)
        PF_np = PF_table.astype(np.float32)  # (V, D_pf)

        # Register as buffer (not a parameter)
        pf_tensor = torch.from_numpy(PF_np).to(torch.bfloat16)
        self.register_buffer("PF_table", pf_tensor, persistent=True)

    def forward(self, token_ids):
        # Convert to float32 for numerical stability in normalization
        PF = self.PF_table[token_ids].to(dtype=torch.float32)  # (B,T,D_pf)

        # Normalize PF per token (zero mean, unit std along D)
        PF_centered = PF - PF.mean(dim=-1, keepdim=True)
        PF_std = PF_centered.std(dim=-1, keepdim=True) + 1e-6
        PFn = PF_centered / PF_std

        # Return normalized phonetic vector directly
        return PFn

    def module(self):
        return self


# ============================================================================
# Core Transformer Components
# ============================================================================


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    (used by LLaMA, DeepSeek, etc.)
    Matching deepscreen implementation with rsqrt for efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """
    LLaMA-style Rotary Positional Embedding
    Applied to Q/K only.
    Matching deepscreen implementation with caching for efficiency.
    """

    def __init__(
        self, dim: int, max_position_embeddings: int = 8192, base: int = 10000
    ):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_position_embeddings = max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat(
            (
                x1 * cos[..., ::2] - x2 * sin[..., ::2],
                x1 * sin[..., ::2] + x2 * cos[..., ::2],
            ),
            dim=-1,
        )


class GatedSparseAttention(nn.Module):
    """
    Gated Sparse Attention (GSA) - arXiv:2601.15305v1

    Implements:
    1. Gated Lightning Indexer (Sec 3.3): Efficient token selection
    2. Adaptive Sparsity (Sec 3.4): Variance-based dynamic budget
    3. Dual Gating (Sec 3.5): Value and Output gating

    Replaces MultiheadLatentAttention.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        max_seq_len=512,
        rope_base=10000,
        k_base=256,
        k_min=16,
        k_max=512,
        indexer_heads=4,
    ):
        # NOTE: Reduced k_base/max defaults for seq_len=512 context (vs paper's long context defaults)
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len

        # Adaptive Sparsity Hyperparams
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.indexer_heads = indexer_heads

        # --- 1. Lightning Indexer Components ---
        self.d_idx = 32  # Dimension for indexer projections (lightweight)

        # Query projection: h_t -> q^I (B, T, H^I * D_idx)
        self.W_Iq = nn.Linear(hidden_size, indexer_heads * self.d_idx, bias=False)
        # Key projection: h_s -> k^I (B, T, D_idx) - Shared across indexer heads
        self.W_Ik = nn.Linear(hidden_size, self.d_idx, bias=False)
        # Head weights: h_t -> w (B, T, H^I)
        self.W_Iw = nn.Linear(hidden_size, indexer_heads, bias=False)
        # Learnable bias per indexer head: b^I_j
        self.gate_bias = nn.Parameter(torch.zeros(indexer_heads))

        # Variance EMA buffer (non-parameter)
        self.register_buffer("variance_ema", torch.tensor(1.0))
        self.variance_alpha = 0.01

        # --- 2. Attention Projections (Standard) ---
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # --- 3. Dual Gating Components ---
        # Value Gate: acts on Values before aggregation. Input: Source hidden user state h_s
        self.W_gv = nn.Linear(hidden_size, hidden_size, bias=False)
        # Output Gate: acts on Attention Output. Input: Query hidden state h_t
        self.W_go = nn.Linear(hidden_size, hidden_size, bias=False)

        # Rotary Embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, max_position_embeddings=max_seq_len, base=rope_base
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize projections with std=0.02
        for m in [
            self.W_Iq,
            self.W_Ik,
            self.W_Iw,
            self.W_q,
            self.W_k,
            self.W_v,
            self.o_proj,
            self.W_gv,
            self.W_go,
        ]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Gate bias: 0.0 is safe (sigmoid(0)=0.5). Paper implies "learnable thresholds".
        nn.init.zeros_(self.gate_bias)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        device = x.device

        # ========================================================================
        # 1. Gated Lightning Indexer (Importance Score I_{t,s})
        # ========================================================================

        # q^I: (B, T, H_I, D_idx)
        q_I = self.W_Iq(x).view(B, T, self.indexer_heads, self.d_idx)
        # k^I: (B, T, D_idx)
        k_I = self.W_Ik(x)

        # Weights w: (B, T, H_I) -> sigmoid
        w = torch.sigmoid(self.W_Iw(x))

        # Compute match score: q^I_t . k^I_s + b^I
        q_I_p = q_I.permute(0, 2, 1, 3)
        k_I_p = k_I.permute(0, 2, 1).unsqueeze(1)

        # match_logits: (B, H_I, T, T)
        match_logits = torch.matmul(q_I_p, k_I_p)
        match_logits = match_logits + self.gate_bias.view(1, self.indexer_heads, 1, 1)

        match_gate = torch.sigmoid(match_logits)

        # Importance Score I_{t,s}
        w_exp = w.permute(0, 2, 1).unsqueeze(-1)
        importance_score = (w_exp * match_gate).sum(dim=1)  # (B, T, T)

        # Causal Masking
        # For variance calc, we set future to 0.0 (as in original paper/code)
        # For selection, we will set to -inf later.
        if T > 1:
            causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
            # Mask future with 0 for now
            importance_score_masked = importance_score.masked_fill(
                ~causal_mask.unsqueeze(0), 0.0
            )
        else:
            importance_score_masked = importance_score
            causal_mask = None  # Handle T=1 case

        # ========================================================================
        # 2. Adaptive Sparsity (Budget k_t)
        # ========================================================================

        # Var(I_{t,:}) per query - using 0-masked scores
        var_t = importance_score_masked.var(dim=-1, unbiased=False)

        # REVERSIBILITY LOGIC:
        # 1. Update EMA only during true forward (no_grad)
        # 2. Cache k_t and top_indices during true forward
        # 3. Reuse cached values during reconstruction (enable_grad) to ensure identical mask

        is_reversible_forward = self.training and (not torch.is_grad_enabled())
        # We check for cached selection to identify reconstruction pass
        is_reversible_reconstruct = (
            self.training
            and torch.is_grad_enabled()
            and getattr(self, "_saved_selection", None) is not None
        )

        if is_reversible_forward:
            # DEBUG LOGGING START
            should_log = torch.rand(1).item() < 0.01  # 1% chance

            var_t_mean = var_t.mean().detach()
            old_ema = self.variance_ema.item()

            # GATED UPDATE: Only update in no-grad forward
            # In-Place update to ensure persistence
            self.variance_ema.mul_(0.99).add_(var_t_mean, alpha=0.01)

            if should_log:
                print(
                    f"[GSA DEBUG] ID: {id(self)} | BufID: {id(self.variance_ema)} | var_t: {var_t_mean:.4f} | EMA: {old_ema:.4f} -> {self.variance_ema.item():.4f}"
                )
            # DEBUG LOGGING END

        # Calculate or Restore k_t and top_indices
        if is_reversible_reconstruct:
            # REUSE (Reconstruction Phase)
            k_t, top_indices = self._saved_selection
            # Clear cache to keep state clean (though we overwrite it next forward anyway)
            self._saved_selection = None
            avg_V = self.variance_ema.clamp(min=1e-6)  # Just for stats if needed
        else:
            # COMPUTE (Forward Phase or Standard Training)
            avg_V = self.variance_ema.clamp(min=1e-6)
            k_t_float = self.k_base * var_t / avg_V
            k_t = (
                k_t_float.floor().clamp(min=self.k_min, max=self.k_max).long()
            )  # (B, T)

            # --- Sparse Selection (Top-K Masking) ---

            # Prepare scores: Mask future with very negative number (not -inf for safety)
            # Safe low value for selection (softmax/topk invariant shift doesn't apply here, but order matters)
            # Actually for topk -inf is fine, but we avoid in-place

            if T > 1:
                # Out-of-place mask filling
                importance_for_selection = importance_score.masked_fill(
                    ~causal_mask.unsqueeze(0), -float("inf")
                )
            else:
                importance_for_selection = importance_score

            # Ensure Attention Sinks
            sink_size = 4
            if T > sink_size:
                # Out-of-place sink forcing
                # Create a mask for sink positions
                sink_mask = torch.zeros_like(importance_for_selection, dtype=torch.bool)
                sink_mask[:, :, :sink_size] = True
                # Set sink positions to infinity out-of-place
                importance_for_selection = importance_for_selection.masked_fill(
                    sink_mask, float("inf")
                )

            # Determine limit
            k_max_needed = k_t.max().item()
            k_limit = min(T, max(k_max_needed, sink_size))

            # Select Top-K
            _, top_indices = importance_for_selection.topk(k_limit, dim=-1)

            # SAVE STATE if this is the reversible forward pass
            if is_reversible_forward:
                self._saved_selection = (k_t, top_indices)

        avg_k = k_t.float().mean().detach()

        # ========================================================================
        # 3. Construct Boolean Mask (Reused logic)
        # ========================================================================

        # We need to rebuild the mask from k_t and top_indices (whether cached or fresh)
        k_limit = top_indices.size(-1)

        # Mask for top_indices dimension: keep index j if j < k_t
        range_k = (
            torch.arange(k_limit, device=device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, k_limit)
        keep_in_topk = range_k < k_t.unsqueeze(-1)  # (B, T, k_limit)

        # Scatter this into the full (B, T, T) mask
        # Initialize with False
        selection_mask = torch.zeros_like(importance_score, dtype=torch.bool)
        # Scatter 'keep_in_topk' booleans into the positions 'top_indices'
        selection_mask.scatter_(dim=-1, index=top_indices, src=keep_in_topk)

        # Causal Masking (strict enforcement)
        if T > 1:
            selection_mask = selection_mask & causal_mask.unsqueeze(0)

        # ========================================================================
        # 4. Dual Gating & Attention (Masked)
        # ========================================================================

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Value Gate
        g_v = torch.sigmoid(self.W_gv(x))
        avg_gv = g_v.mean().detach()
        v = v * g_v

        # Reshape & Rotary
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Rotary Embedding (Applied on B, T, H, D)
        if T > self.rotary_emb.cos_cached.size(0):
            self.rotary_emb._set_cos_sin_cache(T)
        cos = self.rotary_emb.cos_cached[:T].unsqueeze(0).unsqueeze(2)
        sin = self.rotary_emb.sin_cached[:T].unsqueeze(0).unsqueeze(2)
        q = self.rotary_emb._apply_rotary(q, cos, sin)
        k = self.rotary_emb._apply_rotary(k, cos, sin)

        # Transpose to (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot Product Attention with GSA Mask

        # Prepare Mask for SDPA
        # selection_mask is boolean (B, T, T) where True=Keep
        # Construct additive mask: 0 for keep, MIN_VAL for drop
        # Use safe minimum for MPS/float16
        min_val = torch.finfo(q.dtype).min

        # Out-of-place mask creation
        bias_mask = torch.zeros_like(selection_mask, dtype=q.dtype)
        bias_mask = bias_mask.masked_fill(~selection_mask, min_val)

        # Apply external attention_mask if provided (assumed additive)
        if attention_mask is not None:
            bias_mask = bias_mask + attention_mask

        # F.scaled_dot_product_attention handles the fusion
        # q, k, v are (B, H, T, D) -> Output (B, H, T, D)
        o_sparse = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias_mask.unsqueeze(1),  # Broadcast over heads
            dropout_p=0.0,
            is_causal=False,  # Causal masking is already baked into selection_mask
        )

        o_sparse = o_sparse.transpose(1, 2).contiguous().view(B, T, self.hidden_size)

        # Output Gate
        g_o = torch.sigmoid(self.W_go(x))
        avg_go = g_o.mean().detach()

        # Stats
        if self.training:
            self.last_stats = {
                "gsa/k_avg": avg_k,
                "gsa/var_score": avg_V,
                "gsa/gate_v": avg_gv,
                "gsa/gate_o": avg_go,
            }

        return self.o_proj(o_sparse * g_o)


# ============================================================================
# MultiheadLatentAttention (REPLACED BY GatedSparseAttention)
# ============================================================================
# Kept for reference only - not used in model_gated.py


class MultiheadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        compression_ratio=8,
        max_seq_len=512,
        rope_base=10000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_dim = hidden_size // compression_ratio
        self.compressed_dim = self.head_dim // 2

        self.W_DKV = nn.Linear(hidden_size, self.latent_dim, bias=False)
        self.W_UK = nn.Linear(self.latent_dim, hidden_size // 2, bias=False)
        self.W_UV = nn.Linear(self.latent_dim, hidden_size, bias=False)
        self.W_KR = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        self.W_DQ = nn.Linear(hidden_size, self.latent_dim, bias=False)
        self.W_UQ = nn.Linear(self.latent_dim, hidden_size // 2, bias=False)
        self.W_QR = nn.Linear(self.latent_dim, hidden_size // 2, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim // 2, max_position_embeddings=max_seq_len, base=rope_base
        )

        # DeepScreen initialization (std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in [
            self.W_DKV,
            self.W_DQ,
            self.W_UK,
            self.W_UV,
            self.W_UQ,
            self.W_KR,
            self.W_QR,
            self.o_proj,
        ]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        c_kv = self.W_DKV(x)
        c_q = self.W_DQ(x)

        k_c = self.W_UK(c_kv)
        v = self.W_UV(c_kv)
        q_c = self.W_UQ(c_q)

        k_r = self.W_KR(x)
        q_r = self.W_QR(c_q)

        k_c = k_c.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        q_c = q_c.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        k_r = k_r.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        q_r = q_r.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)

        # Use cached rotary embeddings (matching deepscreen)
        if seq_len > self.rotary_emb.cos_cached.size(0):
            self.rotary_emb._set_cos_sin_cache(seq_len)

        cos = (
            self.rotary_emb.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        )  # (1, T, 1, H/2)
        sin = self.rotary_emb.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        q_r = self.rotary_emb._apply_rotary(q_r, cos, sin)
        k_r = self.rotary_emb._apply_rotary(k_r, cos, sin)

        q = torch.cat([q_c, q_r], dim=-1)
        k = torch.cat([k_c, k_r], dim=-1)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=True
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        return self.o_proj(attn_output)


# ============================================================================
# MoE Implementation - EXACT COPY from Deepscreen (batched tensor style)
# ============================================================================
# This matches deepscreen/model/moe_ffn.py exactly for identical gradient flow


class MoEGate(nn.Module):
    """Router gate for MoE with null experts for data sparsity."""

    def __init__(
        self, d_model: int, num_experts: int, top_k: int, data_sparsity: float = 0.5
    ):
        super().__init__()
        self.num_experts = num_experts  # N real experts
        self.top_k = top_k
        self.data_sparsity = data_sparsity  # ρ (target data sparsity)

        # Calculate number of null expert copies: M = N · (1-ρ)/ρ
        # For ρ=0.5, N=8: M = 8 · 0.5/0.5 = 8 null copies
        self.num_null_copies = int(num_experts * (1 - data_sparsity) / data_sparsity)
        self.total_slots = num_experts + self.num_null_copies  # N + M

        # Gate for REAL experts only
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(num_experts))

        # Single NULL expert logit (will be duplicated M times)
        # Initialize to 0 to start balanced
        self.null_logit = nn.Parameter(torch.tensor(0.0))

        # Init gate to small values (matches Deepscreen exactly)
        self.gate.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, D)
        Returns:
            topk_idx: (B, T, K) - indices in range [0, N+M)
            topk_weight: (B, T, K) - renormalized weights (sum to 1)
            is_null: (B, T, K) - boolean mask indicating null expert selection
        """
        B, T, D = x.shape

        # 1. Compute logits for real experts: (B, T, N)
        real_logits = self.gate(x) + self.logit_bias

        # 2. Duplicate null logit M times: (B, T, M)
        null_logits = (
            self.null_logit.unsqueeze(0).unsqueeze(0).expand(B, T, self.num_null_copies)
        )

        # 3. Concatenate: (B, T, N+M)
        logits = torch.cat([real_logits, null_logits], dim=-1)

        # 4. Softmax routing (Paper Requirement)
        probs = F.softmax(logits, dim=-1)

        # 5. Select top-K from N+M slots
        topk_weight, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        # 6. Identify null expert selections (indices >= N)
        is_null = topk_idx >= self.num_experts

        # 7. Renormalize weights over ONLY real experts
        # Zero out null weights, then renormalize
        real_weights = topk_weight * (~is_null).float()
        weight_sum = real_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        topk_weight = real_weights / weight_sum

        # 8. Compute Auxiliary Losses (Paper Eq 6 & 7)
        # Global Load Balancing Loss (Eq 6)
        # P_i: average routing probability for slot i
        P = probs.mean(dim=(0, 1))  # (N+M,)

        # f_i: fraction of tokens routed to slot i
        # Flatten topk_idx to count selections
        idx_flat = topk_idx.view(-1)
        counts = torch.bincount(idx_flat, minlength=self.total_slots).float()
        # f_i = count_i / (total_tokens * top_k) ? No, usually f_i sums to k (average selections per token)
        # Paper says "f_i is the fraction of tokens routed to slot i".
        # If every token picks k slots, sum(f_i) = k.
        # Let's normalize by total tokens B*T.
        f = counts / (B * T)

        L_bal = self.total_slots * torch.sum(f * P)

        # Z-Loss (Eq 7)
        # "log^2( sum(exp(logits)) )" -> (log_sum_exp(logits))^2
        lse = torch.logsumexp(logits, dim=-1)
        L_z = (lse**2).mean()

        # Combine losses
        # Weights from paper: 2e-2 for Bal, 1e-3 for Z-Loss
        aux_loss = 2e-2 * L_bal + 1e-3 * L_z

        return topk_idx, topk_weight, is_null, aux_loss


class MoEFFN(nn.Module):
    """
    MoE FFN with null experts for data sparsity (batched tensor implementation).

    Key features:
    - Expert weights stored as batched 3D tensors (not separate nn.Linear modules)
    - Direct matrix multiplication: chunk_x @ self.W_gate[e]
    - Null experts: zero-compute slots that skip processing entirely
    - Identical gradient flow and numerical characteristics as Deepscreen
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        data_sparsity: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts  # Only REAL experts
        self.top_k = top_k
        self.dropout = dropout

        # Gate with null experts
        self.gate = MoEGate(d_model, num_experts, top_k, data_sparsity=data_sparsity)

        # Expert weights for REAL experts only (no weights for null)
        # This is the key difference from ModuleList approach!
        self.W_gate = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_up = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_down = nn.Parameter(torch.randn(num_experts, d_hidden, d_model) * 0.02)

        # Shared Expert (1 shared expert, always active)
        self.shared_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_up = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_down = nn.Linear(d_hidden, d_model, bias=False)
        self._init_shared_weights()

        self.last_indices = None  # For balancing

    def _init_shared_weights(self):
        """Initialize shared expert weights to std=0.02 (matches Deepscreen)."""
        for module in [self.shared_gate, self.shared_up, self.shared_down]:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.num_experts  # Only real experts
        device, dtype = x.device, x.dtype

        # 1. Shared Expert Path (always active for all tokens)
        shared_h = F.silu(self.shared_gate(x)) * self.shared_up(x)
        if self.training and self.dropout > 0:
            shared_h = F.dropout(shared_h, p=self.dropout)
        shared_out = self.shared_down(shared_h)

        # 2. Routed Experts Path with NULL expert handling
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        self.last_indices = topk_idx.detach().clone()  # Cache for balancer

        flat_x = x.view(N, D)
        flat_idx = topk_idx.view(N, K)
        flat_weight = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)

        # 3. Filter out null expert assignments
        # Create mask for real expert assignments
        real_mask = ~flat_is_null  # (N, K)

        # Flatten and filter
        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

        # Only keep real expert assignments
        real_token_indices = token_indices[real_mask]  # (num_real_assignments,)
        real_expert_indices = flat_idx[real_mask]  # (num_real_assignments,)
        real_weights = flat_weight[real_mask]  # (num_real_assignments,)

        # 4. Sort by expert for vectorized computation
        sort_idx = real_expert_indices.argsort()
        sorted_token_indices = real_token_indices[sort_idx]
        sorted_weights = real_weights[sort_idx]
        sorted_x = flat_x[sorted_token_indices]

        expert_counts = torch.bincount(real_expert_indices, minlength=E)
        offsets = expert_counts.cumsum(0)

        # 5. Process each REAL expert's chunk
        num_real_assignments = sorted_token_indices.size(0)
        sorted_out = torch.empty(num_real_assignments, D, device=device, dtype=dtype)

        start = 0
        for e in range(E):
            end = offsets[e].item()
            if end > start:
                chunk_x = sorted_x[start:end]
                # Expert SwiGLU with DIRECT MATMUL (matches Deepscreen exactly)
                h = F.silu(chunk_x @ self.W_gate[e]) * (chunk_x @ self.W_up[e])
                if self.training and self.dropout > 0:
                    h = F.dropout(h, p=self.dropout)
                sorted_out[start:end] = h @ self.W_down[e]
            start = end

        # 6. Scatter back (only real expert outputs, null contributes 0)
        weighted_out = sorted_out * sorted_weights.unsqueeze(-1)
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)
        routed_out.scatter_add_(
            0, sorted_token_indices.unsqueeze(-1).expand(-1, D), weighted_out
        )

        y = shared_out + routed_out.view(B, T, D)
        return y, aux_loss

    # Removed update_balancing_model_a (Loss-Free Balancing) as it violates the Paper's approach.
    # The paper mandates using L_bal and L_z auxiliary losses instead.


class LlamaMLP(nn.Module):
    """MLP wrapper using MoEFFN with null experts for data sparsity."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_experts,
        num_shared_experts,
        top_k,
        data_sparsity=0.5,
    ):
        super().__init__()
        # Note: num_shared_experts is always 1 in Deepscreen, handled internally by MoEFFN
        self.moe = MoEFFN(
            d_model=hidden_size,
            d_hidden=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            dropout=0.0,
            data_sparsity=data_sparsity,
        )

    def forward(self, x):
        return self.moe(x)


# ============================================================================
# mHC (Multi-Head Composition) Implementation
# ============================================================================


@torch.jit.script
def sinkhorn_knopp(
    logits: torch.Tensor, iters: int = 20, eps: float = 1e-6
) -> torch.Tensor:
    """
    logits: (..., n, n)
    returns: doubly-stochastic matrix (..., n, n)
    Implements the paper's exp + alternating row/col normalization with numerical stability.
    """
    # CRITICAL FIX: Log-sum-exp trick prevents overflow when logits are large
    logits = logits - logits.amax(dim=-1, keepdim=True)
    # Make positive (entropic projection start)
    M = torch.exp(logits).clamp_min(eps)

    for _ in range(iters):
        # Row normalize
        M = M / (M.sum(dim=-1, keepdim=True).clamp_min(eps))
        # Col normalize
        M = M / (M.sum(dim=-2, keepdim=True).clamp_min(eps))
    return M


class MHCCoeffs(nn.Module):
    """
    Produces H_pre, H_post, H_res from the n-stream residual state.
    Uses per-token dynamic mappings + static biases + small gated scaling (alpha_*).
    """

    def __init__(self, d_model: int, n_streams: int = 4, iters: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.iters = iters

        d_in = self.n * d_model

        # Dynamic projections (phi_* in paper)
        self.phi_pre = nn.Linear(d_in, self.n, bias=False)
        self.phi_post = nn.Linear(d_in, self.n, bias=False)
        self.phi_res = nn.Linear(d_in, self.n * self.n, bias=False)

        # Static biases (b_* in paper) (Initialized to 0)
        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))

        # Small gated scaling (alpha_* in paper; initialize small/zero)
        # Starting at 0.0 allows the model to start like the baseline (if biases allow passage)
        # but here the "baseline" is multi-stream, so 0.0 means purely bias-driven routing initially.
        # 0.1 This provides the "spark" needed for gradient flow through the routing layers. REQUIRED for mHC
        self.alpha_pre = nn.Parameter(torch.tensor(0.1))
        self.alpha_post = nn.Parameter(torch.tensor(0.1))
        self.alpha_res = nn.Parameter(torch.tensor(0.1))

        # RMSNorm over flattened n*C vector
        self.rms = RMSNorm(d_in)

        # Init like DeepScreen convention
        for m in [self.phi_pre, self.phi_post, self.phi_res]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x_stream: torch.Tensor):
        """
        x_stream: (B, T, n, d_model)
        returns:
          H_pre  : (B, T, n)      (nonneg, sums NOT forced)
          H_post : (B, T, n)      (nonneg, ~[0,2])
          H_res  : (B, T, n, n)   (doubly stochastic)
        """
        B, T, n, D = x_stream.shape
        # assert n == self.n and D == self.d_model

        x_flat = x_stream.reshape(B, T, n * D)
        x_flat = self.rms(x_flat)

        pre_logits = self.alpha_pre * self.phi_pre(x_flat) + self.b_pre
        post_logits = self.alpha_post * self.phi_post(x_flat) + self.b_post

        res_logits = self.alpha_res * self.phi_res(x_flat)  # (B,T,n*n)
        res_logits = res_logits.view(B, T, n, n) + self.b_res

        # Manifold constraints (paper Eq. 8)
        H_pre = torch.sigmoid(pre_logits)  # nonnegative
        H_post = 2.0 * torch.sigmoid(post_logits)  # nonnegative, helps magnitude
        H_res = sinkhorn_knopp(res_logits, iters=self.iters)

        return H_pre, H_post, H_res


class MHCSublayer(nn.Module):
    """
    Wrap any sublayer F: (B,T,D)->(B,T,D) with mHC residual routing.
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int,
        sublayer: nn.Module,
        norm: nn.Module,
        iters: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.sublayer = sublayer
        self.norm = norm  # Norm applied to the aggregated input x_in
        self.coeffs = MHCCoeffs(d_model=d_model, n_streams=n_streams, iters=iters)

    def forward(self, x_stream: torch.Tensor, attention_mask=None):
        """
        x_stream: (B,T,n,D)
        """
        H_pre, H_post, H_res = self.coeffs(x_stream)

        # ----- PRE READ (H_pre x_stream) -----
        # weighted sum across streams -> (B,T,D)
        # x_in = sum_i (H_pre[i] * x_stream[i])
        # H_pre: (B,T,n), x_stream: (B,T,n,D)
        x_in = (x_stream * H_pre.unsqueeze(-1)).sum(dim=2)

        # ----- LAYER FUNCTION -----
        x_in = self.norm(x_in)
        # Check if sublayer returned aux_loss (tuple)
        aux_loss = None
        if attention_mask is None:
            out = self.sublayer(x_in)
        else:
            out = self.sublayer(x_in, attention_mask)

        if isinstance(out, tuple):
            y, aux_loss = out
        else:
            y = out

        # ----- POST WRITE (H_post^T y) -----
        # y_stream[i] = H_post[i] * y
        y_stream = y.unsqueeze(2) * H_post.unsqueeze(-1)  # (B,T,n,D)

        # ----- RES MIX (H_res x_stream) -----
        # einsum: (B,T,n,n) x (B,T,n,D) -> (B,T,n,D)
        # H_res[b,t,i,j] is weight from stream j to stream i
        x_res = torch.einsum("btij,btjd->btid", H_res, x_stream)

        return x_res + y_stream, aux_loss


# ============================================================================
# Decoder Layer
# ============================================================================


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        compression_ratio,
        num_experts,
        num_shared_experts,
        top_k,
        n_streams=4,
        sinkhorn_iters=20,
        max_seq_len=512,
        rope_base=10000,
        data_sparsity=0.5,
    ):
        super().__init__()
        self.n_streams = n_streams

        # Core sublayers
        # Gated Sparse Attention (replaces MLA)
        attn = GatedSparseAttention(
            hidden_size, num_heads, max_seq_len=max_seq_len, rope_base=rope_base
        )
        mlp = LlamaMLP(
            hidden_size,
            intermediate_size,
            num_experts,
            num_shared_experts,
            top_k,
            data_sparsity=data_sparsity,
        )

        # mHC Wrappers
        self.attn_block = MHCSublayer(
            d_model=hidden_size,
            n_streams=n_streams,
            sublayer=attn,
            norm=RMSNorm(hidden_size),
            iters=sinkhorn_iters,
        )

        self.mlp_block = MHCSublayer(
            d_model=hidden_size,
            n_streams=n_streams,
            sublayer=mlp,
            norm=RMSNorm(hidden_size),
            iters=sinkhorn_iters,
        )

    def force(self, x):
        """
        Computes the residual delta (attn + mlp) and auxiliary loss.
        Returns: (delta, aux_loss)
        """
        # 1. Run Attention Block
        # Expects returns like: (hidden_states, aux_loss) or (hidden_states, None)
        h, aux1 = self.attn_block(x, attention_mask=None)

        # 2. Run MLP Block
        out, aux2 = self.mlp_block(h, attention_mask=None)

        # 3. Calculate Delta
        delta = out - x

        # 4. Hardening Tweak 1: Safe Aux Accumulation
        # Initialize as None so we don't accidentally cast to the wrong dtype/device
        aux = None

        if aux1 is not None:
            aux = aux1

        if aux2 is not None:
            if aux is None:
                aux = aux2
            else:
                aux = aux + aux2

        # If still None (e.g. Pure Fourier Layer), return a safe scalar zero
        # Note: We must ensure it's on the same device and float32
        if aux is None:
            aux = x.new_zeros((), dtype=torch.float32)

        return delta, aux

    def forward(self, x_stream, attention_mask=None):
        # Standard forward kept for compatibility or pre-fill if needed
        # x_stream is (B,T,n,D)
        # attn_block (MHCSublayer) returns (x, aux_loss) but MultiheadLatentAttention doesn't produce loss, so aux is None
        x_stream, aux1 = self.attn_block(x_stream, attention_mask=attention_mask)

        # mlp_block (MHCSublayer) returns (x, aux_loss) from MoE
        x_stream, aux2 = self.mlp_block(x_stream, attention_mask=None)

        total_aux = None
        if aux1 is not None or aux2 is not None:
            total_aux = (aux1 if aux1 is not None else 0) + (
                aux2 if aux2 is not None else 0
            )

        return x_stream, total_aux


# ============================================================================
# Multi-Token Prediction (MTP) Block
# ============================================================================


class MTPTransformerBlock(nn.Module):
    """
    DeepSeek-V3 Style MTP Module: A full Transformer Block WITH mHC Residuals.

    Purpose: Predicts the token AFTER next (t+2) by fusing:
    - Hidden state h_t from main backbone
    - Embedding of the next token (t+1)

    Architecture:
    1. Fusion: Projects concatenated [State_t; Emb_{t+1}] -> hidden_size
    2. Expansion: Expands to n_streams for mHC processing
    3. Processing: Passes through mHC-wrapped Attention (GSA) and MoE/MLP blocks
    4. Collapse: Collapses back to single stream for prediction
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        num_experts,
        num_shared_experts,
        top_k,
        max_seq_len,
        rope_base,
        n_streams=4,
        sinkhorn_iters=20,
        data_sparsity=0.5,
    ):
        super().__init__()

        self.n_streams = n_streams
        self.hidden_size = hidden_size

        # 1. Fusion Layer: Reduce 2*D (State + Emb) -> D
        self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # 2. Core Sublayers - Using GatedSparseAttention
        self.attn = GatedSparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )

        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            data_sparsity=data_sparsity,
        )

        # 3. mHC Wrappers (The Upgrade!)
        self.attn_block = MHCSublayer(
            d_model=hidden_size,
            n_streams=n_streams,
            sublayer=self.attn,
            norm=RMSNorm(hidden_size),
            iters=sinkhorn_iters,
        )

        self.mlp_block = MHCSublayer(
            d_model=hidden_size,
            n_streams=n_streams,
            sublayer=self.mlp,
            norm=RMSNorm(hidden_size),
            iters=sinkhorn_iters,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Skip mHC and MoE internal inits
        if isinstance(module, (MoEFFN, MoEGate, MHCCoeffs)):
            return

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_t, next_emb, attention_mask=None):
        """
        Args:
            h_t:      [B, T, D] (Hidden state from main backbone)
            next_emb: [B, T, D] (Embedding of t+1 token)
            attention_mask: Optional attention mask

        Returns:
            x_out: [B, T, D] (Fused hidden state for MTP prediction)

        NOTE: Unlike backbone layers, MTP block does NOT return aux_loss.
        This matches the original MTP design where MTP is a pure predictor
        without contributing to MoE load balancing losses.
        """
        batch_size, seq_len, _ = h_t.shape

        # 1. Fuse Inputs -> (B, T, D)
        x = torch.cat([h_t, next_emb], dim=-1)  # [B, T, 2D]
        x = self.fusion_proj(x)  # [B, T, D]

        # 2. Expand to Streams (B, T, n, D)
        # We start with x in stream 0, zeros elsewhere (like backbone input)
        x_stream = torch.zeros(
            batch_size,
            seq_len,
            self.n_streams,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        x_stream[:, :, 0, :] = x

        # 3. mHC Blocks (ignore aux_loss - matches original MTP design)
        x_stream, _ = self.attn_block(x_stream, attention_mask=attention_mask)
        x_stream, _ = self.mlp_block(x_stream, attention_mask=None)

        # 4. Collapse Streams -> (B, T, D)
        x_out = x_stream.mean(dim=2)

        return x_out


# ============================================================================
# Complete Model Architecture
# ============================================================================


class SmolLM(nn.Module):
    """
    SmolLM architecture with configurable embeddings (Fourier or standard).
    """

    def __init__(
        self,
        vocab_size,
        embedding_type="fourier",
        bpe_vocab=None,
        pf_codec=None,
        hidden_size=576,
        num_hidden_layers=10,
        num_heads=9,
        intermediate_size=1536,
        max_seq_len=512,
        compression_ratio=8,
        num_experts=8,
        num_shared_experts=1,
        top_k=2,
        K=1152,
        n_streams=4,
        sinkhorn_iters=20,
        data_sparsity=0.5,
    ):  # Added mHC args + data_sparsity
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_type = embedding_type.lower()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.data_sparsity = data_sparsity

        if self.embedding_type == "fourier":
            if bpe_vocab is None or pf_codec is None:
                raise ValueError(
                    "bpe_vocab and pf_codec required for Fourier embeddings"
                )
            # - gate_dim=384: head_dim=32 (standard), less bottleneck than 252
            # - hidden=1536: 4× expansion ratio (standard transformer FF ratio)
            # These changes reduce information bottleneck and improve expressiveness
            self.fourier_embeddings = PureHybridEmbeddingTorch(
                bpe_vocab, pf_codec
            ).module()
            D_pf = pf_codec.D  # 2048
            self.pf_to_model = nn.Linear(D_pf, hidden_size, bias=False)

            # ============================================================================
            # CRITICAL: RMSNorm after Fourier projection
            # ============================================================================
            # Problem: proj_std grows during training (from 0.02 to 0.63+ by step 9000)
            # This scale drift causes late-stage convergence issues vs baseline.
            # https://chatgpt.com/s/t_694de30914c08191bbbe6646f34c1e3d
            # Solution: Add RMSNorm to force stable scale at transformer input.
            # This matches how transformer layers work (norm before each block).
            # The norm will keep proj_std stable regardless of pf_to_model weight growth.

            # ============================================================================
            self.embed_norm = RMSNorm(hidden_size)

            self.token_embed = None
            self.use_fourier = True
            # ------------------------------------------------------------
            # NEW: learnable lambda for injecting previous lm_head-input state (e_prev)
            # We keep this separate from lambda_se (which lives inside Fourier gating module).
            # softplus(-4.0) ~ 0.018 (small start, lets model opt in)
            # ------------------------------------------------------------
            # softplus(-2.25) ≈ 0.105.. 0.04 went to 0.1.. performed better than 0.01.. now trying 0.1, same as se
            # moving from -2.25 to 0.54. loss after 3000 steps is 3.82 (average loss of last 10 steps)
            # moving back to -2.25. loss after 3000 steps is 3.7852 (average loss of last 10 steps) lambda_e goes to 0.14
            self.lambda_e_raw = nn.Parameter(
                torch.tensor(-1.9)
            )  # starting at 0.14.. it liked to settle there
            self.e_inj_ln = nn.LayerNorm(hidden_size)  # stabilizes injected state
            self._D_pf = D_pf  # Store for later initialization
        else:  # standard
            self.token_embed = nn.Embedding(vocab_size, hidden_size)
            self.fourier_embeddings = None
            self.pf_to_model = None
            self.embed_norm = None  # No norm needed for baseline (already stable scale)
            self.use_fourier = False

        # SmolLM Transformer layers
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    compression_ratio=compression_ratio,
                    num_experts=num_experts,
                    num_shared_experts=num_shared_experts,
                    top_k=top_k,
                    max_seq_len=max_seq_len,
                    rope_base=10000,
                    n_streams=n_streams,  # Pass mHC args
                    sinkhorn_iters=sinkhorn_iters,
                    data_sparsity=data_sparsity,  # Pass null expert data sparsity
                )
                for _ in range(num_hidden_layers)
            ]
        )

        # ============================================================================
        # REVERSIBLE MIDPOINT INTEGRATION (Hardened)
        # ============================================================================
        from reversible_ops_midpoint import ReversibleMidpointStack

        # step_size=0.1, noise_eps=0.0 as per instructions ORIGINAL
        # step_size=0.25, noise_eps=1e-4 as per instructions
        # 0.01 is the new recommendation from Gemini. Captured by verify_gradienst on v2 version.
        # self.stack = ReversibleMidpointStack(
        #     self.layers,
        #     step_size=0.25,   # start here
        #     a=0.5,           # stabilizer, still reversible for a!=0 (lower a = more damping toward p_cur)
        #     noise_eps=0.0,    # turn off while debugging convergence
        #     bootstrap="no_kick", # we tested kick_start already, and its 2-3% off from baseline by 10800 steps
        # )
        # In model_gated_multitoken.py

        self.stack = ReversibleMidpointStack(
            self.layers,
            # CRITICAL CHANGE 1
            step_size=0.25,  # CHANGED 0.25 to 0.5 didn't work, back to 0.25
            a=0.5,
            noise_eps=0.0,
            # CRITICAL CHANGE 3: Stop wasting Layer 1!
            # "euler" uses the first layer to kickstart the momentum.
            bootstrap="euler",
        )

        # Note: dropout=0.0 should be set in LlamaDecoderLayer/MoEFFN config.
        # Ensure it is 0.0 ideally by checking or passing arg.
        # But for now we assume config sets it to 0.0 or we rely on user settings.
        # The user instructions said "Set dropout=0.0".
        # We assume the caller provides dropout=0.0 in init or we force it here?
        # The provided code doesn't force it here, but relies on Config.
        # We will trust the constructor args passed in.

        self.norm = RMSNorm(hidden_size)

        # Multi-Token Prediction (MTP) Block - DeepSeek-V3 style
        self.mtp_block = MTPTransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            max_seq_len=max_seq_len,
            rope_base=10000,
            n_streams=n_streams,
            sinkhorn_iters=sinkhorn_iters,
            data_sparsity=data_sparsity,
        )

        # Output projection (shared for both NTP and MTP predictions)
        self.lm_head = nn.Linear(hidden_size, self.vocab_size, bias=False)

        # DeepScreen initialization (std=0.02) - applied to all Linear and Embedding layers
        self.apply(self._init_weights)

        # ============================================================================
        # CRITICAL: Fourier pf_to_model Scale Matching (MUST be AFTER apply!)
        # ============================================================================
        # BUG FIX: self.apply(_init_weights) overwrites pf_to_model with std=0.02!
        # pf_to_model is NOT inside fourier_embeddings, so the skip logic doesn't catch it.
        # We must re-initialize it AFTER apply() to ensure our custom init sticks.
        #
        # Problem: Fourier embedding output has different scale than baseline embedding.
        #
        # For baseline embedding (nn.Embedding with std=0.02):
        #   - Output is just selecting rows from weight matrix
        #   - Output std ≈ 0.02 per element
        #
        # For Fourier embedding (pf_to_model with std=0.02):
        #   - Input: EMB with std ≈ 1 (normalized PFn + SE)
        #   - Linear output variance = fan_in × weight_var × input_var
        #   - Output std = sqrt(2048) × 0.02 × 1 ≈ 0.9
        #
        # Scale mismatch: 0.9 / 0.02 = 45x larger!
        # The transformer layers (initialized with std=0.02) expect baseline-scale inputs.
        #
        # Fix: Initialize pf_to_model with scaled std to produce matching output scale.
        #   target_output_std = 0.02 (match baseline)
        #   input_std = 1.0 (normalized EMB)
        #   required_weight_std = target_output_std / (sqrt(fan_in) × input_std)
        #                       = 0.02 / (sqrt(2048) × 1.0)
        #                       = 0.02 / 45.25
        #                       ≈ 0.00044
        #
        # See: FiarComparison/FOURIER_INITIALIZATION.md for full derivation
        # ============================================================================
        if self.use_fourier and self.pf_to_model is not None:
            pf_to_model_std = 0.02 / math.sqrt(self._D_pf)  # ≈ 0.00044 for D_pf=2048
            self.pf_to_model.weight.data.normal_(mean=0.0, std=pf_to_model_std)
            print(
                f"   🔧 pf_to_model RE-initialized with std={pf_to_model_std:.6f} (scale-matched to baseline)"
            )

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        embedding_params = sum(
            p.numel()
            for p in (
                self.fourier_embeddings.parameters()
                if self.use_fourier
                else self.token_embed.parameters()
            )
        )
        if self.use_fourier:
            embedding_params += sum(p.numel() for p in self.pf_to_model.parameters())

        print(f"🤖 SMOLLM ({embedding_type.upper()}):")
        if self.use_fourier:
            print(f"   K (semantic anchors): {K}")
            print(f"   PF dim: {pf_codec.D} -> model dim: {hidden_size}")
        print(f"   Transformer layers: {num_hidden_layers}, heads: {num_heads}")
        print("   Attention: Gated Sparse Attention (GSA) - arXiv:2601.15305v1")
        print(
            f"   MoE: {num_experts} experts, {num_shared_experts} shared, top-{top_k}"
        )
        print(
            "   Prediction: Multi-Token Prediction (MTP) - DeepSeek-V3 style dual-head"
        )

        # Calculate null expert info
        num_null_copies = int(num_experts * (1 - data_sparsity) / data_sparsity)
        total_slots = num_experts + num_null_copies
        print(
            f"   🔀 Null Experts: {num_null_copies} null copies (ρ={data_sparsity:.1f}, {total_slots} total slots)"
        )
        print(
            f"   🔀 Target data sparsity: {data_sparsity*100:.0f}% tokens use real experts"
        )

        print(f"   📊 Vocabulary size: {self.vocab_size}")
        print(f"   📊 Embedding params: {embedding_params:,}")
        print(f"   📊 Total parameters: {total_params:,}")

    def lambda_e(self):
        return F.softplus(self.lambda_e_raw)

    def _init_weights(self, module):
        """
        DeepScreen initialization: Force 0.02 std initialization for all Linear layers.
        Matches DeepSeek/GPT-2/LLaMA conventions.

        IMPORTANT: Skips modules that have their own initialization:
        - Fourier embedding layers (custom init)
        - MoEFFN (batched tensor init with std=0.02 done in __init__)
        - MoEGate (gate Linear init with std=0.02 done in __init__)
        - MHCCoeffs (custom init in __init__)
        """
        # Skip Fourier embedding layers - they have their own initialization
        if self.use_fourier and self.fourier_embeddings is not None:
            for name, param in self.fourier_embeddings.named_modules():
                if module is param:
                    return  # Skip initialization for Fourier embedding layers

        # Skip MoE layers - they have their own initialization (batched tensors + gate)
        # MoEFFN initializes W_gate, W_up, W_down with randn*0.02 in __init__
        # MoEGate initializes gate.weight with normal_(std=0.02) in __init__
        if isinstance(module, (MoEFFN, MoEGate, MHCCoeffs)):
            return  # Skip - MoE and mHC have their own init

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids,
        next_token_ids=None,
        attention_mask=None,
        prev_lm_in=None,
        return_state=False,
        return_emb_stats=False,
        return_loss=False,
    ):
        """
        Universal Forward Pass with Multi-Token Prediction (MTP).

        Args:
            input_ids: [B, T] - Input token IDs
            next_token_ids: [B, T] - Optional next token IDs for MTP (t+1 tokens)
            attention_mask: Optional attention mask
            prev_lm_in: Previous LM head input for recurrence
            return_state: Whether to return intermediate state
            return_emb_stats: Whether to return embedding statistics
            return_loss: Whether to return auxiliary loss

        Returns:
            - logits_ntp: [B, T, vocab_size] - Next Token Prediction (t+1)
            - logits_mtp: [B, T, vocab_size] or None - Multi-Token Prediction (t+2)
            - Additional returns based on flags (aux_loss, emb_stats)

        Prediction Heads:
            1. NTP (Next Token Prediction): Predicts t+1 for all positions
            2. MTP (Multi-Token Prediction): Predicts t+2 when next_token_ids provided
        """
        batch_size, seq_len = input_ids.size()

        emb_stats = None
        if self.use_fourier:
            # Only get EMB (default), not the unused PF, SE, ap, am (saves memory)
            EMB = self.fourier_embeddings(input_ids)  # [B,T,2048]

            # Compute embedding stats if requested (for initialization validation)
            # This is cheap: just computing std of the embedding, no gradients needed
            if return_emb_stats:
                with torch.no_grad():
                    emb_stats = {
                        "emb_std": EMB.std().item(),  # Overall std of EMB (should be ~1.0)
                        "emb_mean": EMB.mean().item(),  # Overall mean (should be ~0.0)
                        "emb_norm": EMB.norm(dim=-1)
                        .mean()
                        .item(),  # Avg L2 norm per token
                    }

            # Explicit cast to model dtype (bf16) right before projection
            # This implementation satisfies "Fourier computed in FP32, then cast once"
            dtype_target = self.pf_to_model.weight.dtype

            # Ensure EMB is float32 (it should be from fourier_embeddings if coded right, but explicit is safer)
            # Then cast to target dtype.
            x = self.pf_to_model(EMB.to(dtype=dtype_target))  # [B,T,hidden_size]
            # ------------------------------------------------------------
            # NEW: inject previous lm_head input vector into position 0 only
            # This avoids circular dependence inside a single parallel forward pass.
            # prev_lm_in: (B, d_model)
            # ------------------------------------------------------------
            if (prev_lm_in is not None) and self.use_fourier:
                # Only inject at the first position (t=0) of the current chunk
                inj = torch.zeros_like(x)
                inj[:, 0, :] = prev_lm_in
                # disabling recurrance to see what happens to norms and loss
                x = x + self.lambda_e() * self.e_inj_ln(inj)

            # Apply RMSNorm to stabilize scale (prevents proj_std from growing during training)
            x = self.embed_norm(x)

            # Also log post-projection stats (should match baseline embedding scale)
            # After RMSNorm, these should stay stable throughout training
            if return_emb_stats:
                with torch.no_grad():
                    emb_stats["proj_std"] = (
                        x.std().item()
                    )  # Should be stable ~1.0 after norm
                    emb_stats["proj_mean"] = x.mean().item()  # Should stay ~0.0
        else:
            x = self.token_embed(input_ids)  # [B,T,hidden_size]

            # Log baseline embedding stats for comparison
            if return_emb_stats:
                with torch.no_grad():
                    emb_stats = {
                        "emb_std": x.std().item(),
                        "emb_mean": x.mean().item(),
                        "emb_norm": x.norm(dim=-1).mean().item(),
                        "proj_std": x.std().item(),  # Same as emb_std for baseline
                        "proj_mean": x.mean().item(),
                    }

        # ----------------------------------------------------------------------------
        # mHC Expansion: Expand single stream 'x' into n-streams 'x_stream'
        # ----------------------------------------------------------------------------
        # x: (B, T, D)
        # x_stream: (B, T, n, D)
        # Stream 0 gets x (identity-ish init), others get 0
        B, T, D = x.shape
        x_stream = torch.zeros(B, T, self.n_streams, D, device=x.device, dtype=x.dtype)
        x_stream[:, :, 0, :] = x

        # Pass through mHC layers (state is now n-stream)
        # Pass through mHC layers (state is now n-stream)
        # Pass through mHC layers (state is now n-stream)
        # Reversible Leapfrog with Safety Casts

        # Critical: Explicit FP32 cast of Fourier features, THEN cast to model dtype
        # This prevents accidental fp16/bf16 loss of precision in Fourier steps before projection.
        # If use_fourier=True, 'x' is already projected.
        # We need to make sure the projection input was handled safely.
        # Wait, 'x' here IS the projected tensor.
        # In forward():
        # EMB = self.fourier_embeddings(input_ids) # FP32 usually if config says so, but let's be sure.
        # x = self.pf_to_model(EMB)
        # The chatgpt safety instruction was:
        # "Fourier codec in FP32 ... cast once to model dtype BEFORE the projection layer"
        # My code above:
        # if self.use_fourier:
        #    EMB = self.fourier_embeddings(input_ids)
        #    ...
        #    x = self.pf_to_model(EMB)

        # I need to modify the block *above* this replacement to ensure casting.
        # But this replacement chunk targets the layer loop.
        # I will handle the layer loop replacement first.

        # Replace strictly sequential layer loop with Reversible Stack
        x_stream, total_aux_loss = self.stack(x_stream)

        # total_aux_loss is a scalar tensor containing sum of all aux losses

        # ----------------------------------------------------------------------------
        # mHC Readout: Collapse n-streams back to single stream 'h_main'
        # ----------------------------------------------------------------------------
        # Simple mean readout (stable)
        h_main = x_stream.mean(dim=2)  # (B, T, D)
        h_main = self.norm(h_main)  # Normalized hidden state

        # ----------------------------------------------------------------------------
        # Head 1: NTP Prediction (Next Token Prediction)
        # ----------------------------------------------------------------------------
        # Predicts t+1 for all positions
        logits_ntp = self.lm_head(h_main)

        # ----------------------------------------------------------------------------
        # Head 2: MTP Prediction (Multi-Token Prediction)
        # ----------------------------------------------------------------------------
        # Predicts t+2 when next_token_ids provided
        logits_mtp = None
        if next_token_ids is not None:
            # Check shapes: backbone output must match teacher forcing tokens
            # If lengths mismatch, truncate to shortest
            min_len = min(h_main.size(1), next_token_ids.size(1))
            h_use = h_main[:, :min_len, :]
            next_ids_use = next_token_ids[:, :min_len]

            # A. Get embeddings of the "next" tokens (t+1)
            if self.use_fourier:
                next_emb = self.fourier_embeddings(next_ids_use)
                next_emb = self.pf_to_model(
                    next_emb.to(dtype=self.pf_to_model.weight.dtype)
                )
                next_emb = self.embed_norm(next_emb)
            else:
                next_emb = self.token_embed(next_ids_use)

            # B. MTP Block: Fuses h_main and next_emb to predict t+2
            # The block handles Fusion -> mHC(Attn + MLP) internally
            h_mtp = self.mtp_block(h_use, next_emb, attention_mask=None)

            # NO MTP aux_loss accumulation - MTP block is pure predictor (matches original)

            # C. Output Head
            logits_mtp = self.lm_head(self.norm(h_mtp))

        # ----------------------------------------------------------------------------
        # Return based on flags
        # ----------------------------------------------------------------------------
        # IMPORTANT: always return something (avoid None fallthrough)
        if return_emb_stats:
            return logits_ntp, logits_mtp, total_aux_loss, emb_stats
        if return_loss:
            return logits_ntp, logits_mtp, total_aux_loss
        return logits_ntp, logits_mtp

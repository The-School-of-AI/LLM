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

import warnings
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from components.attention.gated_deltanet import DeltaNetRotaryEmbedding

try:
    from components.kernels import (
        HAS_TRITON,
        pytorch_sparse_attention,
        triton_gated_indexer,
        triton_sparse_attention,
    )
except ImportError:
    HAS_TRITON = False
    triton_sparse_attention = None
    pytorch_sparse_attention = None
    triton_gated_indexer = None

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    HAS_SDPA_BACKEND = True
except Exception:
    sdpa_kernel = None
    SDPBackend = None
    HAS_SDPA_BACKEND = False


class ReferenceGSA(nn.Module):
    """
    Gated Sparse Attention (GSA) matching the Test_Code reference.

    Implements adaptive sparse attention with gating for quality.
    Used for 25% of layers to complement DeltaNet's efficiency.

    Forward signature: forward(x, attention_mask=None) -> Tensor
    Returns single tensor (B, T, hidden_size), NOT a tuple.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        max_seq_len=262144,
        rope_base=10000,
        k_base=512,
        k_min=32,
        k_max=1024,
        indexer_heads=4,
        rope_original_max=8192,
        rope_scaling_factor=32.0,
        use_triton_kernels=True,
        sparse_backend="auto",
        triton_min_seq_len=512,
        prefer_flash=True,
        sdpa_chunk_size=16,
    ):
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
            scaling_factor=rope_scaling_factor,
        )

        # Reversible integration support
        self._saved_selection = None

        # Sparse attention backend selection
        self.use_triton_kernels = bool(use_triton_kernels and HAS_TRITON)
        self.sparse_backend = str(sparse_backend).lower()
        self.triton_min_seq_len = int(triton_min_seq_len)
        self.prefer_flash = bool(prefer_flash)
        self.sdpa_chunk_size = max(1, int(sdpa_chunk_size))

        valid_backends = {"auto", "triton", "pytorch", "flash", "dense"}
        if self.sparse_backend not in valid_backends:
            raise ValueError(
                f"Invalid sparse_backend='{self.sparse_backend}'. "
                f"Expected one of {sorted(valid_backends)}."
            )

        if use_triton_kernels and not HAS_TRITON:
            warnings.warn(
                "ReferenceGSA: use_triton_kernels=True but Triton is not installed. "
                "Falling back to PyTorch/SDPA sparse attention."
            )

        self._warned_triton_unavailable = False
        self._warned_triton_failure = False

        # Backends for SDPA kernel preference (FlashAttention first on CUDA).
        if HAS_SDPA_BACKEND:
            self._sdpa_backends = []
            if hasattr(SDPBackend, "FLASH_ATTENTION"):
                self._sdpa_backends.append(SDPBackend.FLASH_ATTENTION)
            if hasattr(SDPBackend, "EFFICIENT_ATTENTION"):
                self._sdpa_backends.append(SDPBackend.EFFICIENT_ATTENTION)
            if hasattr(SDPBackend, "MATH"):
                self._sdpa_backends.append(SDPBackend.MATH)
        else:
            self._sdpa_backends = []

        self._init_weights()

    def _init_weights(self):
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
        nn.init.zeros_(self.gate_bias)

    @staticmethod
    def _is_compiling() -> bool:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            return bool(torch.compiler.is_compiling())
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "is_compiling"):
            return bool(torch._dynamo.is_compiling())
        return False

    def _normalize_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Normalize additive attention masks to [B, T, T] for self-attention.
        """
        if attention_mask is None:
            return None

        if attention_mask.dim() == 4:
            mask = attention_mask[:, 0, :, :]
        elif attention_mask.dim() == 3:
            mask = attention_mask
        else:
            raise ValueError(f"Unsupported attention_mask rank: {attention_mask.dim()}")

        if mask.size(-2) != seq_len:
            mask = mask[:, -seq_len:, :]
        if mask.size(-1) != seq_len:
            mask = mask[:, :, -seq_len:]
        if mask.size(0) == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        if mask.dtype == torch.bool:
            additive = torch.zeros_like(mask, dtype=dtype, device=device)
            additive = additive.masked_fill(~mask, torch.finfo(dtype).min)
            return additive
        return mask.to(device=device, dtype=dtype)

    def _resolve_backend(self, seq_len: int, device: torch.device) -> str:
        if self.sparse_backend != "auto":
            return self.sparse_backend

        if device.type == "cuda":
            if (
                self.use_triton_kernels
                and triton_sparse_attention is not None
                and seq_len >= self.triton_min_seq_len
                and not self._is_compiling()
            ):
                return "triton"
            return "flash"

        return "flash"

    @staticmethod
    def _gather_along_seq_efficient(
        x: torch.Tensor, indices: torch.Tensor  # [B, T, H, D]  # [B, T, K]
    ) -> torch.Tensor:
        """
        Gather K/V along sequence dimension without expanding to [B, T, T, ...].
        """
        batch_size = x.shape[0]
        batch_idx = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1)
        return x[batch_idx, indices]  # [B, T, K, H, D]

    def _sparse_attention_sdpa(
        self,
        q: torch.Tensor,  # [B, T, H, D]
        k: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, H, D]
        indices: torch.Tensor,  # [B, T, K]
        mask: torch.Tensor,  # [B, T, K] bool
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Sparse attention via gathered K/V + SDPA.
        On CUDA, SDPA will prefer FlashAttention kernels when available.
        """
        batch_size, seq_len, n_heads, _ = q.shape
        k_selected = indices.size(-1)
        additive_mask = self._normalize_attention_mask(
            attention_mask,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=q.dtype,
            device=q.device,
        )

        sdpa_context = nullcontext()
        if (
            self.prefer_flash
            and q.device.type == "cuda"
            and self._sdpa_backends
            and not self._is_compiling()
        ):
            sdpa_context = sdpa_kernel(self._sdpa_backends)

        chunk = min(self.sdpa_chunk_size, seq_len)
        while True:
            try:
                out_full = torch.empty_like(q)
                with sdpa_context:
                    for start in range(0, seq_len, chunk):
                        end = min(start + chunk, seq_len)
                        q_chunk = q[:, start:end]  # [B, q_chunk, H, D]
                        idx_chunk = indices[:, start:end]  # [B, q_chunk, K]
                        mask_chunk = mask[:, start:end]  # [B, q_chunk, K]
                        q_chunk_len = end - start

                        k_gathered = self._gather_along_seq_efficient(k, idx_chunk)
                        v_gathered = self._gather_along_seq_efficient(v, idx_chunk)

                        q_sdpa = q_chunk.reshape(
                            batch_size * q_chunk_len, n_heads, 1, self.head_dim
                        )
                        k_sdpa = k_gathered.permute(0, 1, 3, 2, 4).reshape(
                            batch_size * q_chunk_len, n_heads, k_selected, self.head_dim
                        )
                        v_sdpa = v_gathered.permute(0, 1, 3, 2, 4).reshape(
                            batch_size * q_chunk_len, n_heads, k_selected, self.head_dim
                        )

                        attn_mask_bool = mask_chunk.reshape(
                            batch_size * q_chunk_len, 1, 1, k_selected
                        )
                        if additive_mask is not None:
                            additive_chunk = additive_mask[:, start:end]
                            gathered_additive = torch.gather(
                                additive_chunk, dim=-1, index=idx_chunk
                            )
                            valid_from_additive = gathered_additive > (
                                torch.finfo(gathered_additive.dtype).min / 2
                            )
                            valid_from_additive = valid_from_additive.reshape(
                                batch_size * q_chunk_len, 1, 1, k_selected
                            )
                            attn_mask_bool = attn_mask_bool & valid_from_additive

                        out_chunk = F.scaled_dot_product_attention(
                            q_sdpa,
                            k_sdpa,
                            v_sdpa,
                            attn_mask=attn_mask_bool,
                            dropout_p=0.0,
                            is_causal=False,
                        )
                        out_full[:, start:end] = out_chunk.squeeze(2).reshape(
                            batch_size, q_chunk_len, n_heads, self.head_dim
                        )
                return out_full
            except torch.OutOfMemoryError:
                if q.device.type != "cuda":
                    raise
                if chunk <= 1:
                    raise
                torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)

    def _sparse_attention_dense(
        self,
        q: torch.Tensor,  # [B, T, H, D]
        k: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, H, D]
        indices: torch.Tensor,  # [B, T, K]
        mask: torch.Tensor,  # [B, T, K] bool
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Dense fallback path: materialize [B, T, T] sparse bias and call SDPA.
        """
        batch_size, seq_len = q.shape[:2]

        selection_mask = torch.zeros(
            batch_size,
            seq_len,
            seq_len,
            device=q.device,
            dtype=torch.bool,
        )
        selection_mask.scatter_(dim=-1, index=indices, src=mask)

        q_bhtd = q.transpose(1, 2)
        k_bhtd = k.transpose(1, 2)
        v_bhtd = v.transpose(1, 2)

        min_val = torch.finfo(q.dtype).min
        bias_mask = torch.zeros_like(selection_mask, dtype=q.dtype)
        bias_mask = bias_mask.masked_fill(~selection_mask, min_val)

        additive_mask = self._normalize_attention_mask(
            attention_mask,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=q.dtype,
            device=q.device,
        )
        if additive_mask is not None:
            bias_mask = bias_mask + additive_mask

        out = F.scaled_dot_product_attention(
            q_bhtd,
            k_bhtd,
            v_bhtd,
            attn_mask=bias_mask.unsqueeze(1),
            dropout_p=0.0,
            is_causal=False,
        )
        return out.transpose(1, 2).contiguous()

    def _run_sparse_attention(
        self,
        q: torch.Tensor,  # [B, T, H, D]
        k: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, H, D]
        indices: torch.Tensor,  # [B, T, K]
        mask: torch.Tensor,  # [B, T, K] bool
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        backend = self._resolve_backend(seq_len=q.size(1), device=q.device)

        if backend == "triton":
            if (
                self.use_triton_kernels
                and triton_sparse_attention is not None
                and q.device.type == "cuda"
            ):
                try:
                    out, _ = triton_sparse_attention(q, k, v, indices, mask)
                    return out
                except Exception as exc:
                    if not self._warned_triton_failure:
                        warnings.warn(
                            f"ReferenceGSA Triton kernel failed ({exc}); "
                            "falling back to PyTorch sparse attention."
                        )
                        self._warned_triton_failure = True
            elif not self._warned_triton_unavailable:
                warnings.warn(
                    "ReferenceGSA sparse_backend='triton' requested but Triton path "
                    "is unavailable on this device/runtime. Falling back."
                )
                self._warned_triton_unavailable = True

            if pytorch_sparse_attention is not None:
                out, _ = pytorch_sparse_attention(q, k, v, indices, mask)
                return out
            return self._sparse_attention_sdpa(q, k, v, indices, mask, attention_mask)

        if backend == "pytorch":
            if pytorch_sparse_attention is not None:
                out, _ = pytorch_sparse_attention(q, k, v, indices, mask)
                return out
            return self._sparse_attention_sdpa(q, k, v, indices, mask, attention_mask)

        if backend == "flash":
            return self._sparse_attention_sdpa(q, k, v, indices, mask, attention_mask)

        if backend == "dense":
            return self._sparse_attention_dense(q, k, v, indices, mask, attention_mask)

        raise ValueError(f"Unsupported sparse backend: {backend}")

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
        w_raw = self.W_Iw(x)  # raw logits (B, T, indexer_heads)

        # Use Triton gated indexer on CUDA, PyTorch inline on CPU
        _use_triton_indexer = (
            triton_gated_indexer is not None
            and device.type == "cuda"
            and not self._is_compiling()
        )

        if _use_triton_indexer:
            try:
                # Triton kernel: applies sigmoid(w), sigmoid(qk*scale + bias), weighted sum
                # Returns (B, T, T) with causal mask (future = -inf)
                scale = 1.0 / (self.d_idx**0.5)
                importance_score = triton_gated_indexer(
                    q_I,
                    k_I,
                    w_raw,
                    self.gate_bias,
                    scale=scale,
                    causal=True,
                )
                # Triton kernel already applied causal mask (-inf for future)
                # Convert -inf positions to 0 for importance_score_masked
                importance_score_masked = importance_score.clamp(min=0.0)
                _triton_indexer_ok = True
            except Exception:
                _triton_indexer_ok = False
        else:
            _triton_indexer_ok = False

        if not _triton_indexer_ok:
            # PyTorch fallback: inline indexer computation
            w = torch.sigmoid(w_raw)

            q_I_p = q_I.permute(0, 2, 1, 3)  # (B, indexer_heads, T, d_idx)
            k_I_p = k_I.permute(0, 2, 1).unsqueeze(1)  # (B, 1, d_idx, T)

            match_logits = torch.matmul(q_I_p, k_I_p)  # (B, indexer_heads, T, T)
            match_logits = match_logits + self.gate_bias.view(
                1, self.indexer_heads, 1, 1
            )
            match_gate = torch.sigmoid(match_logits)

            w_exp = w.permute(0, 2, 1).unsqueeze(-1)  # (B, indexer_heads, T, 1)
            importance_score = (w_exp * match_gate).sum(dim=1)  # (B, T, T)

            # Causal masking (memory-efficient: broadcasting trick, no T×T matrix allocation)
            if T > 1:
                positions = torch.arange(T, device=device)
                causal_mask_broadcast = positions.view(1, -1, 1) >= positions.view(
                    1, 1, -1
                )
                importance_score_masked = importance_score.masked_fill(
                    ~causal_mask_broadcast, 0.0
                )
            else:
                importance_score_masked = importance_score

        # Adaptive Sparsity
        var_t = importance_score_masked.var(dim=-1, unbiased=False)

        is_reversible_forward = self.training and (not torch.is_grad_enabled())
        is_reversible_reconstruct = (
            self.training
            and torch.is_grad_enabled()
            and getattr(self, "_saved_selection", None) is not None
        )

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

            if _triton_indexer_ok:
                # Triton path already has -inf for future positions
                importance_for_selection = importance_score
            elif T > 1:
                importance_for_selection = importance_score.masked_fill(
                    ~causal_mask_broadcast, -float("inf")
                )
            else:
                importance_for_selection = importance_score

            # Attention sinks: first 4 tokens always selected
            sink_size = min(4, T)
            if T > sink_size:
                sink_mask = torch.zeros_like(importance_for_selection, dtype=torch.bool)
                sink_mask[:, :, :sink_size] = True
                importance_for_selection = importance_for_selection.masked_fill(
                    sink_mask, float("inf")
                )

            # Use static k_max upper bound instead of data-dependent k_t.max().item().
            # This avoids GPU->CPU sync and torch.compile graph breaks.
            k_limit = min(T, max(self.k_max, sink_size))
            _, top_indices = importance_for_selection.topk(k_limit, dim=-1)

            if is_reversible_forward:
                self._saved_selection = (k_t, top_indices)

        # Construct sparse keep mask aligned with top_indices.
        k_limit = top_indices.size(-1)
        range_k = torch.arange(k_limit, device=device).view(1, 1, -1)
        keep_in_topk = range_k < k_t.unsqueeze(-1)

        if T > 1:
            query_positions = torch.arange(T, device=device).view(1, T, 1)
            causal_valid = top_indices <= query_positions
            sparse_mask = keep_in_topk & causal_valid
        else:
            sparse_mask = keep_in_topk

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

        o_sparse = self._run_sparse_attention(
            q,
            k,
            v,
            top_indices,
            sparse_mask,
            attention_mask=attention_mask,
        )
        o_sparse = o_sparse.contiguous().view(B, T, self.hidden_size)

        # Output gate
        g_o = torch.sigmoid(self.W_go(x))

        return self.o_proj(o_sparse * g_o)

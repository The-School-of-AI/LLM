"""
TurboQuant Linear with TQP low-rank accumulator (Phase 3d)

Base weights stored as 4-bit TQ indices (frozen between flushes).
TQP adapter in fp32 provides the training signal.
Periodic flush absorbs the adapter into the quantized base weights.

Key difference from standard TQP:
  - Base weights EVOLVE over time (via periodic flush)
  - After N flushes, effective rank of total weight change is N × r
  - Base weights are TQ-compressed (4-bit) instead of fp32/bf16 frozen

Key difference from shadowless / bf16 accumulator:
  - Adapter trains via standard autograd (no stochastic rounding)
  - No per-weight scalar accumulator — low-rank A@B product instead
  - Standard torch.optim.AdamW works on tqp_A, tqp_B directly
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from codebook import get_codebook
    from stochastic_round import dequantize, nearest_round
except ImportError:
    from lightninglm.tqp.codebook import get_codebook
    from lightninglm.tqp.stochastic_round import dequantize, nearest_round


def is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def get_rotation_matrix(n: int, seed: int = 42) -> np.ndarray:
    if is_power_of_2(n):
        H = np.array([[1.0]])
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        H = H / math.sqrt(n)
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        return (H * signs[np.newaxis, :]).astype(np.float32)
    else:
        rng = np.random.default_rng(seed)
        G = rng.standard_normal((n, n)).astype(np.float64)
        Q, R = np.linalg.qr(G)
        d = np.sign(np.diag(R))
        d[d == 0] = 1
        Q = Q * d[np.newaxis, :]
        return Q.astype(np.float32)


# ============================================================================
# TurboQuantPretrainingLinear — standalone linear layer
# ============================================================================


# ============================================================================
# USE_BF16_BASE MODE — frozen bf16 base + TQP-adapter-only training
# ============================================================================
import os as _os_for_bf16mode


def _bf16_base_mode_enabled():
    return _os_for_bf16mode.environ.get("USE_BF16_BASE", "0") == "1"


class TurboQuantPretrainingLinear(nn.Module):
    """
    Drop-in nn.Linear replacement with:
      - 4-bit TQ-quantized base weights (frozen between flushes)
      - fp32 TQP adapter (trainable via standard autograd)
      - Periodic flush absorbs adapter into base weights

    Forward (pre-rotate-input trick):
      x_rot = x @ R.T
      w_rot = codebook[weight_indices] * row_norms  (frozen dequant)
      delta_rot = tqp_A @ tqp_B                   (fp32 adapter)
      output = x_rot @ (w_rot + delta_rot).T
             = base_output + tqp_output
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_bits: int = 4,
        rank: int = 32,
        rotation_seed: int = 42,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.rank = rank

        # Frozen rotation matrix
        R_np = get_rotation_matrix(in_features, seed=rotation_seed)
        self.register_buffer("rotation_matrix", torch.from_numpy(R_np))

        # Frozen 4-bit Lloyd-Max codebook for Beta distribution
        levels_np, _ = get_codebook(in_features, weight_bits)
        self.register_buffer("weight_codebook", torch.from_numpy(levels_np))

        # Weight indices (frozen between flushes — updated only at flush time)
        self.register_buffer(
            "weight_indices", torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Row norms (fp32, NOT a Parameter — updated at flush time)
        self.register_buffer("row_norms", torch.ones(out_features))

        # TQP adapter — the ONLY trainable weight state
        # A: [out, rank], B: [rank, in]
        # Both operate in ROTATED space (since input is pre-rotated)
        self.tqp_A = nn.Parameter(torch.zeros(out_features, rank))
        self.tqp_B = nn.Parameter(torch.zeros(rank, in_features))

        # TQP adapter init: both A and B get small random values so the TQP adapter output
        # is non-zero from step 1.  This is critical when base_out is detached:
        # with A=0 the SwiGLU chain (gate*up → down) becomes fully detached,
        # blocking gradients to gate/up TQP params entirely.
        nn.init.normal_(self.tqp_A, std=1e-4)
        nn.init.normal_(self.tqp_B, std=1.0 / math.sqrt(rank))

        # Bias (optional)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Flush tracking
        self._flush_count = 0
        # Cache: dequantized + rotated-back base weight in rotated space [out, in].
        # Valid between flushes. We actually cache w_rot_base (scaled by norms)
        # so forward just needs to add tqp update and rotate-back happens per forward.
        # Simpler: cache the FULL base matmul result against any input. But we
        # can't cache across different x, so just cache w_rot_base.
        self._cached_w_rot_base = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_bits: int = 4,
        rank: int = 32,
        rotation_seed: int = 42,
    ) -> "TurboQuantPretrainingLinear":
        tq = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            weight_bits=weight_bits,
            rank=rank,
            rotation_seed=rotation_seed,
        )

        with torch.no_grad():
            W = linear.weight.data.float()  # [out, in]
            norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
            W_normalized = W / norms
            tq.row_norms.copy_(norms.squeeze())
            R = tq.rotation_matrix.float()
            W_rot = W_normalized @ R.t()
            # Nearest (deterministic) quantize — we're snapshotting pretrained
            tq.weight_indices.copy_(nearest_round(W_rot, tq.weight_codebook.float()))

            if linear.bias is not None and tq.bias is not None:
                tq.bias.data.copy_(linear.bias.data)

        return tq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        R = self.rotation_matrix.to(dtype)
        x_rot = x @ R.t()  # [*, in]

        # Base path: dequantized quantized weights (frozen, no grad)
        # Cache between flushes — weight_indices doesn't change
        if self._cached_w_rot_base is None or self._cached_w_rot_base.dtype != dtype:
            with torch.no_grad():
                w_rot_base = self.weight_codebook.to(dtype)[self.weight_indices.long()]
                w_rot_base = w_rot_base * self.row_norms.to(dtype).unsqueeze(1)
                self._cached_w_rot_base = w_rot_base

        # Base output (no autograd — indices are frozen)
        base_output = F.linear(x_rot, self._cached_w_rot_base)

        # TQP adapter path: (x_rot @ tqp_B.T) @ tqp_A.T — cheap rank-r matmul
        tqp_output = F.linear(
            F.linear(x_rot, self.tqp_B.to(dtype)), self.tqp_A.to(dtype)
        )

        output = base_output + tqp_output

        if self.bias is not None:
            output = output + self.bias.to(dtype)

        return output

    @torch.no_grad()
    def flush(self) -> dict:
        """
        Absorb the TQP adapter into the base weights.
        Reset tqp_A (PEFT-style init) and tqp_B (zero).
        Re-normalize rows.
        Return stats.
        """
        # Current adapter magnitude
        AB = self.tqp_A @ self.tqp_B  # [out, in]
        ab_norm = AB.norm().item()

        # Current base weights in rotated space
        w_rot_base = dequantize(self.weight_indices, self.weight_codebook.float())
        w_rot_base = w_rot_base * self.row_norms.unsqueeze(1)  # scale by current norms

        # Apply update
        w_new = w_rot_base + AB

        # Re-normalize rows
        new_norms = w_new.norm(dim=1).clamp(min=1e-8)
        w_normed = w_new / new_norms.unsqueeze(1)

        # Re-quantize deterministically (nearest)
        old_indices = self.weight_indices.clone()
        new_indices = nearest_round(w_normed, self.weight_codebook.float())
        self.weight_indices.copy_(new_indices)
        self.row_norms.copy_(new_norms)

        # Reset TQP — both A and B get small random init so gradients flow
        nn.init.normal_(self.tqp_A, std=1e-4)
        nn.init.normal_(self.tqp_B, std=1.0 / math.sqrt(self.rank))

        # Invalidate cache — weight_indices changed
        self._cached_w_rot_base = None

        # Stats
        w_changed = (new_indices != old_indices).float().mean().item()
        self._flush_count += 1

        return {
            "AB_norm": ab_norm,
            "w_change_frac": w_changed,
            "flush_count": self._flush_count,
        }

    def memory_bytes(self) -> dict:
        return {
            "weight_indices": self.weight_indices.numel(),  # int8
            "tqp_A": self.tqp_A.numel() * 4,  # fp32
            "tqp_B": self.tqp_B.numel() * 4,  # fp32
            "row_norms": self.row_norms.numel() * 4,
            "rotation_matrix": self.rotation_matrix.numel() * 4,
            "weight_codebook": self.weight_codebook.numel() * 4,
            "total": (
                self.weight_indices.numel()
                + self.tqp_A.numel() * 4
                + self.tqp_B.numel() * 4
                + self.row_norms.numel() * 4
                + self.rotation_matrix.numel() * 4
                + self.weight_codebook.numel() * 4
            ),
        }

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.weight_bits}, rank={self.rank}, "
            f"bias={self.bias is not None}"
        )


# ============================================================================
# TurboQuantPretrainingExpertWeights — for MoE expert tensors [E, d_in, d_out]
# ============================================================================


class TurboQuantPretrainingExpertWeights(nn.Module):
    """
    TQP accumulator for MoE expert weight tensors.
    Per-expert adapters (Option A): each expert has its own rank-r adapter.

    Storage (E = num_experts):
      weight_indices: [E, d_out, d_in] int8
      tqp_A: [E, d_out, rank] fp32  (per-expert)
      tqp_B: [E, rank, d_in] fp32   (per-expert)
      row_norms: [E, d_out] fp32
      rotation_matrix: [d_in, d_in] fp32 (shared across experts)
    """

    def __init__(
        self,
        num_experts: int,
        d_in: int,
        d_out: int,
        weight_bits: int = 4,
        rank: int = 32,
        rotation_seed: int = 42,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_in = d_in
        self.d_out = d_out
        self.weight_bits = weight_bits
        self.rank = rank

        # Rotation (shared across experts, applied on d_in)
        R_np = get_rotation_matrix(d_in, seed=rotation_seed)
        self.register_buffer("rotation_matrix", torch.from_numpy(R_np))

        # Codebook
        levels_np, _ = get_codebook(d_in, weight_bits)
        self.register_buffer("weight_codebook", torch.from_numpy(levels_np))

        # Storage: [E, d_out, d_in] — each "row" is d_in-dimensional
        self.register_buffer(
            "weight_indices", torch.zeros(num_experts, d_out, d_in, dtype=torch.int8)
        )
        self.register_buffer("row_norms", torch.ones(num_experts, d_out))

        # USE_BF16_BASE MODE: allocate bf16 base_weight as a Parameter (frozen).
        # _dequant_single_expert returns base_weight[e] directly in this mode (no codebook).
        # The codebook/indices/row_norms above remain allocated but are dead weight, kept
        # for state_dict compatibility so the SAME checkpoint loads in either mode.
        if _bf16_base_mode_enabled():
            self.base_weight = nn.Parameter(
                torch.zeros(num_experts, d_out, d_in, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self._use_bf16_base = True
        else:
            self._use_bf16_base = False

        # Per-expert TQP adapters
        self.tqp_A = nn.Parameter(torch.zeros(num_experts, d_out, rank))
        self.tqp_B = nn.Parameter(torch.zeros(num_experts, rank, d_in))

        # TQP adapter init: A gets tiny random, B gets normal — so A×B is small but non-zero
        nn.init.normal_(self.tqp_A, std=1e-4)
        nn.init.normal_(self.tqp_B, std=1.0 / math.sqrt(rank))

        self._flush_count = 0
        # Cache: dequantized base weight in ROTATED space [E, d_out, d_in] (fp32).
        # Valid between flushes. Invalidated on flush().
        # Used by compute_selected_expert_outputs() for fast per-expert forward.
        self._cached_w_rot_base = None

    @classmethod
    def from_weight(
        cls,
        W: torch.Tensor,  # [E, d_in, d_out]
        weight_bits: int = 4,
        rank: int = 32,
        rotation_seed: int = 42,
    ) -> "TurboQuantPretrainingExpertWeights":
        num_experts, d_in, d_out = W.shape
        tq = cls(num_experts, d_in, d_out, weight_bits, rank, rotation_seed)

        with torch.no_grad():
            # Transpose to [E, d_out, d_in]
            W_t = W.float().transpose(1, 2)
            norms = W_t.norm(dim=2, keepdim=True).clamp(min=1e-8)
            W_normalized = W_t / norms
            tq.row_norms.copy_(norms.squeeze(2))
            R = tq.rotation_matrix.float()
            W_rot = torch.matmul(W_normalized, R.t())
            tq.weight_indices.copy_(nearest_round(W_rot, tq.weight_codebook.float()))

        return tq

    @classmethod
    def from_weight_lazy(
        cls,
        W: torch.Tensor,  # [E, d_in, d_out]
        weight_bits: int = 4,
        rank: int = 32,
        rotation_seed: int = 42,
    ) -> "TurboQuantPretrainingExpertWeights":
        """
        Create a TQP wrapper WITHOUT doing rotation/quantization.
        Stores raw weights as a buffer. Quantization happens on first forward
        (on GPU), then the raw weights are deleted.

        This makes model init ~100x faster (seconds vs 30+ minutes).
        """
        num_experts, d_in, d_out = W.shape
        # Minimal init — skip rotation matrix and codebook computation
        tq = object.__new__(cls)
        nn.Module.__init__(tq)
        tq.num_experts = num_experts
        tq.d_in = d_in
        tq.d_out = d_out
        tq.weight_bits = weight_bits
        tq.rank = rank
        tq._rotation_seed = rotation_seed
        tq._flush_count = 0
        tq._cached_w_rot_base = None

        # Placeholder buffers (will be filled on materialize)
        tq.register_buffer("rotation_matrix", torch.empty(0, dtype=torch.bfloat16))
        tq.register_buffer("weight_codebook", torch.empty(0))
        tq.register_buffer(
            "weight_indices", torch.zeros(num_experts, d_out, d_in, dtype=torch.int8)
        )
        tq.register_buffer(
            "row_norms", torch.ones(num_experts, d_out, dtype=torch.bfloat16)
        )

        # TQP adapters as nn.Parameter for DeepSpeed compatibility.
        # They ARE registered on the module, but we EXCLUDE them from the
        # reversible stack's param_keys in rebuild_reversible_cached_keys().
        tq.tqp_A = nn.Parameter(
            torch.randn(num_experts, d_out, rank, dtype=torch.bfloat16) * 1e-4
        )
        tq.tqp_B = nn.Parameter(
            torch.randn(num_experts, rank, d_in, dtype=torch.bfloat16)
            * (1.0 / math.sqrt(rank))
        )

        # Store raw weight as plain attribute (NOT a buffer) so DeepSpeed
        # doesn't move it to GPU. We materialize on CPU before DeepSpeed init.
        tq._pending_weight = W.detach()
        tq._lazy_init_done = False
        return tq

    def _materialize_lazy(self):
        """Run rotation + quantization on current device (GPU). Called once on first forward."""
        if self._lazy_init_done:
            return
        import time as _time

        _t0 = _time.time()
        device = self._pending_weight.device

        # Compute rotation matrix and codebook, register as proper buffers
        R_np = get_rotation_matrix(self.d_in, seed=self._rotation_seed)
        self.register_buffer("rotation_matrix", torch.from_numpy(R_np).to(device))
        levels_np, _ = get_codebook(self.d_in, self.weight_bits)
        self.register_buffer("weight_codebook", torch.from_numpy(levels_np).to(device))

        W = self._pending_weight  # [E, d_in, d_out]
        with torch.no_grad():
            W_t = W.float().transpose(1, 2)  # [E, d_out, d_in]
            norms = W_t.norm(dim=2, keepdim=True).clamp(min=1e-8)
            W_normalized = W_t / norms
            self.row_norms.copy_(norms.squeeze(2))
            R = self.rotation_matrix.float()
            W_rot = torch.matmul(W_normalized, R.t())
            self.weight_indices.copy_(
                nearest_round(W_rot, self.weight_codebook.float())
            )

        # Free raw weights
        self._pending_weight = None
        self._lazy_init_done = True
        print(
            f"  [TQP] Materialized {self.num_experts} experts on {device} in {_time.time()-_t0:.1f}s"
        )

    def _materialize_lazy_gpu(self, gpu_device):
        """Run rotation + quantization on GPU for speed. Results stored on CPU."""
        if self._lazy_init_done:
            return
        import time as _time

        _t0 = _time.time()

        # Compute rotation matrix and codebook, register as proper buffers
        R_np = get_rotation_matrix(self.d_in, seed=self._rotation_seed)
        self.register_buffer("rotation_matrix", torch.from_numpy(R_np))
        levels_np, _ = get_codebook(self.d_in, self.weight_bits)
        self.register_buffer("weight_codebook", torch.from_numpy(levels_np))

        W = self._pending_weight  # [E, d_in, d_out] on CPU
        with torch.no_grad():
            # Sanitize: torch.empty produces garbage → NaN in SwiGLU.
            # Generate clean randn on CPU (safe for ZeRO-3 GPU memory),
            # then move to GPU for fast quantization matmul.
            _scale = 1.0 / (self.d_in**0.5)
            _needs_sanitize = True
            try:
                W_f = W.float()
                if torch.isfinite(W_f).all() and W_f.abs().max().item() < 10.0:
                    _needs_sanitize = False
            except Exception:
                pass
            if _needs_sanitize:
                W = torch.randn(W.shape, dtype=torch.float32) * _scale  # CPU
            W_gpu = W.to(gpu_device).float()
            R_gpu = self.rotation_matrix.to(gpu_device).float()
            cb_gpu = self.weight_codebook.to(gpu_device).float()

            W_t = W_gpu.transpose(1, 2)  # [E, d_out, d_in]
            norms = W_t.norm(dim=2, keepdim=True).clamp(min=1e-8)
            # Clamp row norms to prevent bf16 overflow in forward
            # bf16 max ~65504; with codebook values ~0.5 and hidden dim 1024,
            # output magnitude ~ norm * 0.5 * sqrt(1024) ~ norm * 16
            # Keep norm < 4000 to stay well within bf16 range after matmul
            _max_norm = 4000.0
            _clamped = (norms > _max_norm).sum().item()
            if _clamped > 0:
                print(
                    f"  [TQP] WARNING: {_clamped} row norms > {_max_norm}, "
                    f"clamping (max was {norms.max().item():.1f})",
                    flush=True,
                )
                norms = norms.clamp(max=_max_norm)
            W_normalized = W_t / norms
            self.row_norms.copy_(norms.squeeze(2).cpu())

            # Big matmul on GPU — this is 100x faster than CPU
            W_rot = torch.matmul(W_normalized, R_gpu.t())

            # Quantize on GPU
            new_indices = nearest_round(W_rot, cb_gpu)
            self.weight_indices.copy_(new_indices.cpu())

            # Free GPU memory immediately
            del W_gpu, W_t, norms, W_normalized, W_rot, R_gpu, cb_gpu, new_indices
            torch.cuda.empty_cache()

        self._pending_weight = None
        self._lazy_init_done = True
        import gc

        gc.collect()
        print(
            f"  [TQP] GPU-materialized {self.num_experts} experts in {_time.time()-_t0:.1f}s"
        )

    def _ensure_cache_rotated(self):
        """
        Cache the base weight in rotated space as [E, d_out, d_in] float32.
        This avoids the rotate-back step at every forward. The cache is valid
        between flushes.
        """
        if hasattr(self, "_lazy_init_done") and not self._lazy_init_done:
            self._materialize_lazy()
        if self._cached_w_rot_base is None:
            with torch.no_grad():
                w_rot = dequantize(self.weight_indices, self.weight_codebook.float())
                norms = self.row_norms.float().unsqueeze(2)  # [E, d_out, 1]
                w_rot = w_rot * norms  # scale by norms
                # Store as float32 contiguous: [E, d_out, d_in]
                self._cached_w_rot_base = w_rot.contiguous()

    def _dequant_single_expert(self, expert_idx, device=None):
        """Dequantize one expert's weight using a reusable buffer. Returns [d_out, d_in] view."""
        # USE_BF16_BASE MODE: return base_weight[expert_idx] directly (no codebook lookup).
        # base_weight is stored in the SAME rotated form as the dequantized result, so the
        # caller's downstream forward path (rotation matrix application + matmul) works
        # unchanged.
        if getattr(self, "_use_bf16_base", False):
            with torch.no_grad():
                bw = self.base_weight[expert_idx]
                if device is not None:
                    bw = bw.to(device)
            return bw

        with torch.no_grad():
            indices = self.weight_indices[expert_idx]  # [d_out, d_in] int8
            if device is not None:
                indices = indices.to(device)
            cb = self.weight_codebook.to(device=indices.device).float()

            # Pre-allocate a reusable buffer (same shape every call = allocator reuses block)
            if not hasattr(self, "_w_buf") or self._w_buf.device != indices.device:
                self._w_buf = torch.empty(
                    self.d_out, self.d_in, dtype=torch.float32, device=indices.device
                )
                self._norms_buf = torch.empty(
                    self.d_out, dtype=torch.float32, device=indices.device
                )

            # Dequantize into buffer: codebook lookup + norm scaling
            # Use .copy_ to write into pre-allocated memory instead of creating new tensor
            self._w_buf.copy_(cb[indices.long()])
            self._norms_buf.copy_(
                self.row_norms[expert_idx].to(device=indices.device).float()
            )
            self._w_buf.mul_(self._norms_buf.unsqueeze(1))
        return self._w_buf

    def compute_expert_chunk(
        self,
        x_chunk: torch.Tensor,  # [M, d_in] — tokens assigned to expert e (NOT pre-rotated)
        expert_idx: int,  # which expert
        skip_tqp: bool = False,  # skip TQP (for reversible stack bypass)
    ) -> torch.Tensor:
        """
        Compute output for a chunk of tokens ALL assigned to expert_idx.
        Uses reusable dequant buffer to avoid per-call GPU memory allocation.
        """
        if not getattr(type(self), "_ck_diag_done", False):
            print(
                f"[CK-DIAG] compute_expert_chunk called: e={expert_idx} skip_tqp={skip_tqp} x_chunk.requires_grad={x_chunk.requires_grad} grad_enabled={torch.is_grad_enabled()} tqp_A.requires_grad={self.tqp_A.requires_grad}",
                flush=True,
            )
            type(self)._ck_diag_done = True
        if hasattr(self, "_lazy_init_done") and not self._lazy_init_done:
            self._materialize_lazy()

        dtype = x_chunk.dtype
        device = x_chunk.device

        R = self.rotation_matrix.to(device=device, dtype=dtype)
        x_rot = x_chunk @ R.t()

        # Base path: dequantized TQ weight (frozen, no grad needed)
        # _dequant_single_expert returns a SHARED buffer — must consume before next call
        w_base = self._dequant_single_expert(expert_idx, device=device).to(dtype)
        base_out = F.linear(x_rot, w_base)
        # Detach + sanitize immediately, then the buffer can be reused
        base_out = base_out.detach()
        base_out = torch.nan_to_num(base_out, nan=0.0, posinf=60000.0, neginf=-60000.0)
        if base_out.dtype == torch.bfloat16:
            base_out = base_out.clamp(-60000.0, 60000.0)

        # TQP adapter path: (x_rot @ B.T) @ A.T — cheap rank-r matmul (differentiable)
        if skip_tqp:
            return base_out
        A_e = self.tqp_A[expert_idx]
        B_e = self.tqp_B[expert_idx]
        xB = F.linear(x_rot, B_e)
        tqp_out = F.linear(xB, A_e)

        # DIAGNOSTIC (one-shot): print whether gradient reaches tqp_out during recompute.
        if not getattr(self, "_tqp_grad_diag_done", False):
            if tqp_out.requires_grad and torch.is_grad_enabled():

                def _dbg_hook(g, _expert_idx=expert_idx, _self=self):
                    if g is None:
                        print(f"[GRAD-DIAG] expert={_expert_idx} grad=None", flush=True)
                    else:
                        print(
                            f"[GRAD-DIAG] expert={_expert_idx} grad.norm={g.norm().item():.6e} shape={tuple(g.shape)}",
                            flush=True,
                        )
                    _self._tqp_grad_diag_done = True

                tqp_out.register_hook(_dbg_hook)

        return base_out + tqp_out

    # Legacy (slow) path — keep for compatibility/debugging
    def get_dequantized_weight_original_space(self) -> torch.Tensor:
        """
        DEPRECATED: slow path that materializes [E, d_in, d_out].
        Use compute_selected_expert_outputs() instead.
        """
        dtype = self.tqp_A.dtype
        R = self.rotation_matrix.to(dtype)
        self._ensure_cache_rotated()
        base = self._cached_w_rot_base  # [E, d_out, d_in]
        delta_rot = torch.matmul(self.tqp_A, self.tqp_B)  # [E, d_out, d_in]
        w_rot_combined = base + delta_rot
        W_orig = torch.matmul(w_rot_combined, R)  # [E, d_out, d_in]
        return W_orig.transpose(1, 2)

    @torch.no_grad()
    def flush(self) -> dict:
        """Absorb TQP adapter into weight_indices. Per-expert to avoid OOM."""
        # USE_BF16_BASE MODE: flush is a NO-OP. Base is frozen; TQP never absorbed.
        # We still increment a counter and return stats for compatibility.
        if getattr(self, "_use_bf16_base", False):
            ab_norm = (
                float((self.tqp_A @ self.tqp_B).norm().item())
                if self.num_experts > 0
                else 0.0
            )
            self._flush_count = int(getattr(self, "_flush_count", 0)) + 1
            return {
                "AB_norm": ab_norm,
                "w_change_frac": 0.0,
                "flush_count": self._flush_count,
                "mode": "bf16_base_tqp_noop",
            }

        print(
            f"  [FLUSH DEBUG] tqp_A norm={self.tqp_A.norm().item():.6f} "
            f"tqp_B norm={self.tqp_B.norm().item():.6f} "
            f"tqp_A.device={self.tqp_A.device} "
            f"tqp_A.requires_grad={self.tqp_A.requires_grad}"
        )
        cb = self.weight_codebook.to(device=self.weight_indices.device).float()
        ab_norm_sum = 0.0
        w_changed_sum = 0.0

        for e in range(self.num_experts):
            # Dequantize single expert
            w_rot = cb[self.weight_indices[e].long()]  # [d_out, d_in]
            w_rot = w_rot * self.row_norms[e].float().unsqueeze(1)

            # Adapter update for this expert
            ab = self.tqp_A[e] @ self.tqp_B[e]  # [d_out, d_in]
            ab_norm_sum += ab.norm().item()

            w_new = w_rot + ab

            # Re-normalize
            new_norm = w_new.norm(dim=1).clamp(min=1e-8)
            w_normed = w_new / new_norm.unsqueeze(1)

            # Re-quantize
            old_idx = self.weight_indices[e].clone()
            new_idx = nearest_round(w_normed.unsqueeze(0), cb).squeeze(0)
            self.weight_indices[e].copy_(new_idx)
            self.row_norms[e].copy_(new_norm)

            w_changed_sum += (new_idx != old_idx).float().mean().item()

        # Reset TQP — A tiny random, B normal, so A×B is small but non-zero
        nn.init.normal_(self.tqp_A, std=1e-4)
        nn.init.normal_(self.tqp_B, std=1.0 / math.sqrt(self.rank))

        self._cached_w_rot_base = None
        self._flush_count += 1

        return {
            "AB_norm": ab_norm_sum / self.num_experts,
            "w_change_frac": w_changed_sum / self.num_experts,
            "flush_count": self._flush_count,
        }

    def memory_bytes(self) -> dict:
        return {
            "weight_indices": self.weight_indices.numel(),  # int8
            "tqp_A": self.tqp_A.numel() * 4,
            "tqp_B": self.tqp_B.numel() * 4,
            "row_norms": self.row_norms.numel() * 4,
            "total": (
                self.weight_indices.numel()
                + self.tqp_A.numel() * 4
                + self.tqp_B.numel() * 4
                + self.row_norms.numel() * 4
                + self.rotation_matrix.numel() * 4
                + self.weight_codebook.numel() * 4
            ),
        }


# ============================================================================
# MoE wrapper with TQP adapters
# ============================================================================


class TurboQuantPretrainingMoEWrapper(nn.Module):
    """
    Wraps an MoEFFN with TQP modules.
    Expert weights: TurboQuantPretrainingExpertWeights
    Shared expert linears: TurboQuantPretrainingLinear
    Exposes same attributes as original for path-based resolution.
    """

    def __init__(
        self,
        moe_ffn,
        weight_bits=4,
        rank=32,
        rotation_seed=42,
        quantize_shared=True,
        lazy=False,
    ):
        super().__init__()
        self.weight_bits = weight_bits
        self.rank = rank
        self.quantize_shared = quantize_shared

        # Preserve attributes
        self.gate = moe_ffn.gate
        self.num_experts = moe_ffn.num_experts
        self.top_k = moe_ffn.top_k
        self.dropout = moe_ffn.dropout

        _from = (
            TurboQuantPretrainingExpertWeights.from_weight_lazy
            if lazy
            else TurboQuantPretrainingExpertWeights.from_weight
        )
        self.tq_gate_weights = _from(
            moe_ffn.W_gate.data, weight_bits, rank, rotation_seed
        )
        self.tq_up = _from(moe_ffn.W_up.data, weight_bits, rank, rotation_seed + 100)
        self.tq_down = _from(
            moe_ffn.W_down.data, weight_bits, rank, rotation_seed + 200
        )

        if quantize_shared:
            self.shared_gate = TurboQuantPretrainingLinear.from_linear(
                moe_ffn.shared_gate, weight_bits, rank, rotation_seed + 300
            )
            self.shared_up = TurboQuantPretrainingLinear.from_linear(
                moe_ffn.shared_up, weight_bits, rank, rotation_seed + 400
            )
            self.shared_down = TurboQuantPretrainingLinear.from_linear(
                moe_ffn.shared_down, weight_bits, rank, rotation_seed + 500
            )
        else:
            self.shared_gate = moe_ffn.shared_gate
            self.shared_up = moe_ffn.shared_up
            self.shared_down = moe_ffn.shared_down

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast forward path — matches the original MoEFFN's sort-by-expert approach.

        For each real expert e with assigned tokens: process its chunk with a
        dense matmul (NOT bmm with gather). This is much more memory-friendly
        than per-token gather and matches the pattern the original code uses.
        """
        # ── EP dispatch: if expert parallelism is enabled, use all-to-all ──
        if getattr(self, "_ep_enabled", False):
            return self._forward_ep(x)

        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.num_experts
        device, dtype = x.device, x.dtype

        # 1. Shared expert (dense)
        shared_out = self.shared_down(F.silu(self.shared_gate(x)) * self.shared_up(x))

        # 2. Routed experts
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        flat_x = x.view(N, D)
        flat_idx = topk_idx.view(N, K)
        flat_weight = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)
        real_mask = ~flat_is_null  # [N, K]

        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)
        real_token_indices = token_indices[real_mask]  # [M]
        real_expert_indices = flat_idx[real_mask]  # [M]
        real_weights = flat_weight[real_mask]  # [M]

        # 3. Sort by expert
        sort_idx = real_expert_indices.argsort()
        sorted_token_indices = real_token_indices[sort_idx]
        sorted_weights = real_weights[sort_idx]
        sorted_x = flat_x[sorted_token_indices]  # [M, D]

        expert_counts = torch.bincount(real_expert_indices, minlength=E)
        offsets = expert_counts.cumsum(0)

        # 4. Process experts — use fused Triton kernel if available
        M = sorted_token_indices.size(0)

        # Build [E+1] offsets with 0 prepended
        expert_offsets = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.int64), offsets.to(torch.int64)]
        )

        try:
            # Fused kernel requires indices on GPU — skip if indices are on CPU
            # (CPU indices = memory-optimized mode for large expert counts)
            if self.tq_gate_weights.weight_indices.device.type == "cpu":
                raise ImportError("indices on CPU, use per-expert loop")
            from lightninglm.tqp.triton_tqp_grouped import (
                grouped_dequant_matmul,
                grouped_rotate,
                grouped_tqp,
            )

            # Fused path: Triton for base dequant+matmul (frozen, no grad),
            # standard PyTorch for TQP (needs grad).
            def _fused_expert(x_in, offsets, tq_w):
                R = tq_w.rotation_matrix.to(dtype=x_in.dtype)
                x_rot = x_in @ R.t()
                # Base: fused Triton kernel (no grad — frozen quantized weights)
                with torch.no_grad():
                    nf = tq_w.row_norms.float()
                    cb = tq_w.weight_codebook.float()
                    y_base = grouped_dequant_matmul(
                        x_rot, offsets, tq_w.weight_indices, nf, cb
                    )
                    y_base = torch.nan_to_num(
                        y_base, nan=0.0, posinf=60000.0, neginf=-60000.0
                    )
                    y_base = y_base.clamp(-60000.0, 60000.0)
                # TQP: standard PyTorch ops (autograd tracks through tqp_A/B)
                y_tqp = grouped_tqp(x_rot, offsets, tq_w.tqp_A, tq_w.tqp_B)
                return y_base + y_tqp

            h_gate = _fused_expert(sorted_x, expert_offsets, self.tq_gate_weights).to(
                dtype
            )
            h_up = _fused_expert(sorted_x, expert_offsets, self.tq_up).to(dtype)
            hidden = F.silu(h_gate) * h_up
            if self.dropout > 0 and self.training:
                hidden = F.dropout(hidden, p=self.dropout, training=True)
            sorted_out = _fused_expert(hidden, expert_offsets, self.tq_down).to(dtype)
        except (ImportError, Exception):
            # Fallback: per-expert Python loop
            sorted_out = torch.empty(M, D, device=device, dtype=dtype)
            start = 0
            for e in range(E):
                end = offsets[e].item()
                if end > start:
                    chunk_x = sorted_x[start:end]
                    h_gate = self.tq_gate_weights.compute_expert_chunk(chunk_x, e).to(
                        dtype
                    )
                    h_up = self.tq_up.compute_expert_chunk(chunk_x, e).to(dtype)
                    hidden = F.silu(h_gate) * h_up
                    if self.dropout > 0 and self.training:
                        hidden = F.dropout(hidden, p=self.dropout, training=True)
                    sorted_out[start:end] = self.tq_down.compute_expert_chunk(
                        hidden, e
                    ).to(dtype)
                start = end

        # 5. Scatter back
        weighted_out = sorted_out * sorted_weights.to(dtype).unsqueeze(-1)
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)
        routed_out.scatter_add_(
            0,
            sorted_token_indices.unsqueeze(-1).expand(-1, D),
            weighted_out.to(dtype),
        )
        routed_out = routed_out.view(B, T, D)

        output = shared_out + routed_out

        if self.dropout > 0 and self.training:
            output = F.dropout(output, p=self.dropout, training=True)

        return output, aux_loss

    def _forward_ep(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """EP-aware forward with memory-optimized dispatch/combine (OOM fix v2)."""
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.num_experts  # global expert count (gate still uses this)
        device, dtype = x.device, x.dtype
        ep_ctx = self._ep_ctx

        # 1. Shared expert (dense, replicated)
        shared_out = self.shared_down(F.silu(self.shared_gate(x)) * self.shared_up(x))

        # 2. Gate (replicated)
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        flat_x = x.view(N, D)
        flat_idx = topk_idx.view(N, K)
        flat_weight = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)
        real_mask = ~flat_is_null

        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)
        real_token_indices = token_indices[real_mask]
        real_expert_indices = flat_idx[real_mask]
        real_weights = flat_weight[real_mask]

        M = real_token_indices.size(0)
        if M == 0:
            return shared_out, aux_loss

        tokens_to_send = flat_x[real_token_indices]

        # 3. All-to-all dispatch
        recv_tokens, recv_local_eids, recv_weights, meta = ep_ctx.dispatch(
            tokens_to_send, real_expert_indices, real_weights
        )
        del tokens_to_send  # free dispatch input

        # 4. Local expert compute
        M_local = recv_tokens.size(0)
        local_E = self.tq_gate_weights.num_experts

        if M_local > 0:
            sort_idx = recv_local_eids.argsort(stable=True)
            sorted_tokens = recv_tokens[sort_idx]
            sorted_eids = recv_local_eids[sort_idx]
            del recv_tokens  # free dispatch buffer

            expert_counts = torch.bincount(sorted_eids.long(), minlength=local_E)
            offsets = expert_counts.cumsum(0)

            # Collect expert outputs in a list and cat — NOT in-place assignment.
            # In-place into torch.empty() breaks autograd for TQP params.
            _skip_tqp = getattr(self, "_skip_tqp_in_force", False)
            _expert_outputs = []
            start = 0
            for e in range(local_E):
                end = offsets[e].item()
                if end > start:
                    chunk = sorted_tokens[start:end]
                    h_g = self.tq_gate_weights.compute_expert_chunk(
                        chunk, e, skip_tqp=_skip_tqp
                    ).to(dtype)
                    h_u = self.tq_up.compute_expert_chunk(
                        chunk, e, skip_tqp=_skip_tqp
                    ).to(dtype)
                    h = F.silu(h_g) * h_u
                    del h_g, h_u
                    if self.dropout > 0 and self.training:
                        h = F.dropout(h, p=self.dropout, training=True)
                    _expert_outputs.append(
                        self.tq_down.compute_expert_chunk(h, e, skip_tqp=_skip_tqp).to(
                            dtype
                        )
                    )
                    del h
                start = end
            del sorted_tokens
            sorted_out = (
                torch.cat(_expert_outputs, dim=0)
                if _expert_outputs
                else torch.empty(0, D, device=device, dtype=dtype)
            )
            del _expert_outputs

            # Unsort back to dispatch order — autograd-safe via inverse permutation.
            # The previous "local_output[sort_idx] = sorted_out" pattern breaks the
            # autograd graph (in-place assign to torch.empty), preventing TQP grads.
            inv_sort = sort_idx.argsort()
            local_output = sorted_out[inv_sort]
            del sorted_out, sort_idx, inv_sort
        else:
            del recv_tokens
            local_output = torch.empty(0, D, device=device, dtype=dtype)

        # 5. All-to-all combine
        combined_output = ep_ctx.combine(local_output, meta)
        del local_output, meta  # free combine inputs

        # 6. Weighted sum + scatter back
        weighted_out = combined_output * real_weights.to(dtype).unsqueeze(-1)
        del combined_output
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)
        routed_out.scatter_add_(
            0,
            real_token_indices.unsqueeze(-1).expand(-1, D),
            weighted_out.to(dtype),
        )
        del weighted_out
        routed_out = routed_out.view(B, T, D)

        output = shared_out + routed_out
        if self.dropout > 0 and self.training:
            output = F.dropout(output, p=self.dropout, training=True)

        # In-expert TQP: no routing cache needed — TQP already applied in compute_expert_chunk.
        return output, aux_loss

    def tqp_residual_ep(self, dummy_x: torch.Tensor) -> torch.Tensor:
        """
        Compute TQP-adapter-only residual using cached routing + input.
        LOCAL ONLY — no EP dispatch/combine (those use @torch.no_grad which
        kills the autograd graph). Each GPU computes TQP for its local experts
        on the tokens that were routed to them.

        Returns [B, T, D] TQP delta.
        """
        if not hasattr(self, "_cached_routing") or self._cached_routing is None:
            return (
                torch.zeros(1, device=dummy_x.device, dtype=dummy_x.dtype).expand(
                    dummy_x.shape[0], dummy_x.shape[1], dummy_x.shape[-1]
                )
                * 0
            )

        routing = self._cached_routing
        self._cached_routing = None  # consume once

        x = routing["input_x"]  # [B, T, D] — cached MoE input
        B, T, D = x.shape
        N = B * T
        device, dtype = x.device, x.dtype
        ep_ctx = self._ep_ctx

        real_token_indices = routing["real_token_indices"].to(device)
        real_expert_indices = routing["real_expert_indices"].to(device)
        real_weights = routing["real_weights"].to(device)

        M = real_token_indices.size(0)
        if M == 0:
            return torch.zeros(B, T, D, device=device, dtype=dtype)

        # Filter to LOCAL experts only (no all-to-all needed)
        local_start = ep_ctx.local_expert_start
        local_end = local_start + ep_ctx.local_num_experts
        local_mask = (real_expert_indices >= local_start) & (
            real_expert_indices < local_end
        )

        if not local_mask.any():
            return torch.zeros(B, T, D, device=device, dtype=dtype)

        local_token_indices = real_token_indices[local_mask]
        local_expert_ids = (
            real_expert_indices[local_mask] - local_start
        )  # 0-based local
        local_weights = real_weights[local_mask]

        flat_x = x.view(N, D)
        local_tokens = flat_x[local_token_indices]  # [M_local, D]

        # Sort by expert for efficient processing
        sort_idx = local_expert_ids.argsort(stable=True)
        sorted_tokens = local_tokens[sort_idx]
        sorted_eids = local_expert_ids[sort_idx]
        sorted_weights = local_weights[sort_idx]
        sorted_token_indices = local_token_indices[sort_idx]

        local_E = self.tq_gate_weights.num_experts
        expert_counts = torch.bincount(sorted_eids.long(), minlength=local_E)
        offsets = expert_counts.cumsum(0)

        # Hoist rotation matrices outside per-expert loop (one .to() per layer, not per expert)
        R_g_t = (
            self.tq_gate_weights.rotation_matrix.to(device=device, dtype=dtype)
            .t()
            .contiguous()
        )
        R_u_t = (
            self.tq_up.rotation_matrix.to(device=device, dtype=dtype).t().contiguous()
        )
        R_d_t = (
            self.tq_down.rotation_matrix.to(device=device, dtype=dtype).t().contiguous()
        )

        # TQP-adapter-only expert compute — DIFFERENTIABLE (no @torch.no_grad)
        tqp_outputs = []
        start = 0
        for e in range(local_E):
            end = offsets[e].item()
            if end > start:
                chunk = sorted_tokens[start:end]
                # Gate TQP
                x_rot_g = chunk @ R_g_t
                h_g = F.linear(
                    F.linear(x_rot_g, self.tq_gate_weights.tqp_B[e]),
                    self.tq_gate_weights.tqp_A[e],
                ).to(dtype)
                # Up TQP
                x_rot_u = chunk @ R_u_t
                h_u = F.linear(
                    F.linear(x_rot_u, self.tq_up.tqp_B[e]), self.tq_up.tqp_A[e]
                ).to(dtype)
                # SwiGLU
                h = F.silu(h_g) * h_u
                # Down TQP
                h_rot = h @ R_d_t
                tqp_out = F.linear(
                    F.linear(h_rot, self.tq_down.tqp_B[e]), self.tq_down.tqp_A[e]
                ).to(dtype)
                tqp_outputs.append(tqp_out)
            start = end

        if not tqp_outputs:
            return torch.zeros(B, T, D, device=device, dtype=dtype)

        sorted_tqp = torch.cat(tqp_outputs, dim=0)

        # Weighted scatter back to [N, D]
        weighted = sorted_tqp * sorted_weights.to(dtype).unsqueeze(-1)
        tqp_out = torch.zeros(N, D, device=device, dtype=dtype)
        tqp_out.scatter_add_(
            0,
            sorted_token_indices.unsqueeze(-1).expand(-1, D),
            weighted,
        )

        return tqp_out.view(B, T, D)

    def flush_all(self) -> dict:
        """Flush all TQP adapters in this MoE layer. Returns aggregated stats."""
        stats_list = []
        for name, mod in [
            ("tq_gate", self.tq_gate_weights),
            ("tq_up", self.tq_up),
            ("tq_down", self.tq_down),
        ]:
            s = mod.flush()
            s["module"] = name
            stats_list.append(s)

        if self.quantize_shared:
            for name, mod in [
                ("shared_gate", self.shared_gate),
                ("shared_up", self.shared_up),
                ("shared_down", self.shared_down),
            ]:
                s = mod.flush()
                s["module"] = name
                stats_list.append(s)

        # Aggregate
        return {
            "AB_norm_mean": sum(s["AB_norm"] for s in stats_list) / len(stats_list),
            "AB_norm_max": max(s["AB_norm"] for s in stats_list),
            "w_change_mean": sum(s["w_change_frac"] for s in stats_list)
            / len(stats_list),
            "w_change_max": max(s["w_change_frac"] for s in stats_list),
            "per_module": stats_list,
        }

    def total_memory_bytes(self) -> int:
        total = 0
        for mod in [self.tq_gate_weights, self.tq_up, self.tq_down]:
            total += mod.memory_bytes()["total"]
        if self.quantize_shared:
            for mod in [self.shared_gate, self.shared_up, self.shared_down]:
                if hasattr(mod, "memory_bytes"):
                    total += mod.memory_bytes()["total"]
        return total


def flush_all_tqp(model: nn.Module) -> dict:
    """Walk model tree, flush all TQP MoE wrappers."""
    all_stats = []
    for name, mod in model.named_modules():
        if isinstance(mod, TurboQuantPretrainingMoEWrapper):
            s = mod.flush_all()
            s["layer"] = name
            all_stats.append(s)
    if not all_stats:
        return {}
    return {
        "AB_norm_mean": sum(s["AB_norm_mean"] for s in all_stats) / len(all_stats),
        "AB_norm_max": max(s["AB_norm_max"] for s in all_stats),
        "w_change_mean": sum(s["w_change_mean"] for s in all_stats) / len(all_stats),
        "w_change_max": max(s["w_change_max"] for s in all_stats),
        "num_layers_flushed": len(all_stats),
    }


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    print("=" * 70)
    print("TurboQuant TQP Tests")
    print("=" * 70)

    # Test 1: standalone TurboQuantPretrainingLinear
    print("\n--- Test 1: TurboQuantPretrainingLinear forward/backward ---")
    orig = nn.Linear(512, 256, bias=False)
    tq = TurboQuantPretrainingLinear.from_linear(orig, weight_bits=4, rank=16)
    print(f"  Shape: [{tq.out_features}, {tq.in_features}], rank={tq.rank}")
    print(f"  tqp_A init norm: {tq.tqp_A.norm():.4f} (nonzero)")
    print(f"  tqp_B init norm: {tq.tqp_B.norm():.4f} (zero)")
    print(
        f"  weight_indices range: [{tq.weight_indices.min()}, {tq.weight_indices.max()}]"
    )

    # Forward at init should match orig closely (TQP contribution is zero)
    x = torch.randn(2, 8, 512)
    with torch.no_grad():
        out_orig = orig(x)
        out_tq = tq(x)
        rel_error = (out_orig - out_tq).norm() / out_orig.norm()
        print(f"  At init: relative error vs orig: {rel_error:.4f} (4-bit quant noise)")

    # Backward — check gradients flow to tqp_A and tqp_B
    out_tq = tq(x)
    loss = out_tq.sum()
    loss.backward()
    print(f"  tqp_A grad norm: {tq.tqp_A.grad.norm():.6f}")
    print(f"  tqp_B grad norm: {tq.tqp_B.grad.norm():.6f}")
    print(
        f"  Non-zero grads: A={tq.tqp_A.grad.norm() > 0}, B={tq.tqp_B.grad.norm() > 0}"
    )

    # Test 2: Flush
    print("\n--- Test 2: Flush ---")
    # Simulate training: perturb tqp params
    with torch.no_grad():
        tq.tqp_A.add_(torch.randn_like(tq.tqp_A) * 0.01)
        tq.tqp_B.add_(torch.randn_like(tq.tqp_B) * 0.01)

    old_indices = tq.weight_indices.clone()
    flush_stats = tq.flush()
    print(f"  AB norm before flush: {flush_stats['AB_norm']:.6f}")
    print(f"  w_change after flush: {flush_stats['w_change_frac']:.4f}")
    print(f"  tqp_A after reset: {tq.tqp_A.norm():.4f} (should be nonzero)")
    print(f"  tqp_B after reset: {tq.tqp_B.norm():.4f} (should be zero)")

    # Test 3: Expert weights
    print("\n--- Test 3: TurboQuantPretrainingExpertWeights ---")
    W = torch.randn(4, 512, 128) * 0.02
    tq_exp = TurboQuantPretrainingExpertWeights.from_weight(W, weight_bits=4, rank=16)
    print(f"  Shape: [{tq_exp.num_experts}, {tq_exp.d_in}, {tq_exp.d_out}]")
    print(f"  tqp_A: {tq_exp.tqp_A.shape} (per-expert)")
    print(f"  tqp_B: {tq_exp.tqp_B.shape} (per-expert)")

    # Test dequantize_to_original_space produces valid gradient path
    W_deq = tq_exp.get_dequantized_weight_original_space()
    print(f"  Dequantized shape: {W_deq.shape}")
    print(f"  Has grad: {W_deq.requires_grad}")

    # Backward through it
    loss = W_deq.sum()
    loss.backward()
    print(f"  tqp_A grad: {tq_exp.tqp_A.grad.norm():.4f}")
    print(f"  tqp_B grad: {tq_exp.tqp_B.grad.norm():.4f}")

    # Test 4: Memory
    print("\n--- Test 4: Memory footprint ---")
    mem_lin = tq.memory_bytes()
    print("  TurboQuantPretrainingLinear [512, 256]:")
    for k, v in mem_lin.items():
        print(f"    {k}: {v:,} bytes")

    mem_exp = tq_exp.memory_bytes()
    print("  TurboQuantPretrainingExpertWeights [4, 512, 128]:")
    for k, v in mem_exp.items():
        print(f"    {k}: {v:,} bytes")

    print("\nAll TQP tests passed!")

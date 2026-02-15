"""
Persistent Triton Kernel for MoEFFN — Phases 2.2-3 of IMPLEMENTATION_PLAN.md
=============================================================================

Two persistent kernel launches replace the entire 3×E expert loop:

  Kernel 1  persistent_swiglu_kernel
            h = silu(x @ W_gate[e]) * (x @ W_up[e])   for ALL experts
            Fused gate + up + SiLU activation in registers.

  Kernel 2  persistent_down_kernel
            out = h @ W_down[e]                        for ALL experts
            Standard grouped GEMM.

Backward:  Batched GEMM via torch.bmm (correct gradients, no custom kernels).

Key design:
  - Column-major tile schedule → weight columns stay hot in L2 cache
  - Round-robin work distribution across all SMs
  - Per-tile expert-aware bounds checking (no cross-expert contamination)
  - Autograd Function wraps both kernels; backward recomputes activations

Expected: 1.8-2.4x speedup on H100 vs baseline loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, List


# ============================================================================
# Tile Schedule Precomputation  (CPU-side, O(E) — microseconds)
# ============================================================================

def _precompute_tiles(
    expert_counts: List[int],
    expert_offsets: List[int],
    N_cols: int,
    BLOCK_M: int,
    BLOCK_N: int,
    device: torch.device,
):
    """
    Build column-major tile schedule for persistent grouped GEMM.

    Column-major = n-tiles (output columns) outer, m-tiles (rows) inner.
    Consecutive tiles share the same weight column block → L2 cache reuse.

    Returns
    -------
    tile_expert_ids  (T,) int64 — expert owning each tile
    tile_m_offsets   (T,) int64 — global row start in sorted_x / h
    tile_n_offsets   (T,) int64 — column start in output
    tile_m_bounds    (T,) int64 — row upper-bound (exclusive) for masking
    total_tiles      int
    """
    t_eids, t_moffs, t_noffs, t_mbounds = [], [], [], []

    num_n = (N_cols + BLOCK_N - 1) // BLOCK_N

    for e, (count, offset) in enumerate(zip(expert_counts, expert_offsets)):
        if count == 0:
            continue
        num_m = (count + BLOCK_M - 1) // BLOCK_M
        bound = offset + count
        for n in range(num_n):          # column-major outer
            for m in range(num_m):      # row inner
                t_eids.append(e)
                t_moffs.append(offset + m * BLOCK_M)
                t_noffs.append(n * BLOCK_N)
                t_mbounds.append(bound)

    total = len(t_eids)
    if total == 0:
        z = torch.zeros(1, dtype=torch.int64, device=device)
        return z, z, z, z, 0

    return (
        torch.tensor(t_eids, dtype=torch.int64, device=device),
        torch.tensor(t_moffs, dtype=torch.int64, device=device),
        torch.tensor(t_noffs, dtype=torch.int64, device=device),
        torch.tensor(t_mbounds, dtype=torch.int64, device=device),
        total,
    )


# ============================================================================
# Triton Kernel 1 — Persistent Fused SwiGLU  (all experts, 1 launch)
# ============================================================================

@triton.jit
def persistent_swiglu_kernel(
    # ── data pointers ──
    x_ptr,          # (A, K)   sorted tokens
    W_gate_ptr,     # (E, K, N)
    W_up_ptr,       # (E, K, N)
    h_ptr,          # (A, N)   output
    # ── tile schedule ──
    tile_eids_ptr, tile_moffs_ptr, tile_noffs_ptr, tile_mbounds_ptr,
    # ── dimensions ──
    K: tl.constexpr,          # d_model
    N: tl.constexpr,          # d_hidden
    total_tiles,
    # ── strides for x (A, K) ──
    stride_xm, stride_xk,
    # ── strides for W (E, K, N) — same layout for gate & up ──
    stride_we, stride_wk, stride_wn,
    # ── strides for h (A, N) ──
    stride_hm, stride_hn,
    # ── block sizes ──
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Persistent CTA kernel.  Each program loops over tiles in round-robin.
    Computes  h[rows] = silu(x[rows] @ W_gate[e]) * (x[rows] @ W_up[e])
    where `rows` and `e` are decoded from the precomputed tile schedule.
    """
    pid = tl.program_id(0)
    num_sms = tl.num_programs(0)

    tile_id = pid
    while tile_id < total_tiles:
        # ── decode tile ──
        expert_id = tl.load(tile_eids_ptr + tile_id)
        m_off     = tl.load(tile_moffs_ptr + tile_id)
        n_off     = tl.load(tile_noffs_ptr + tile_id)
        m_bound   = tl.load(tile_mbounds_ptr + tile_id)

        offs_m = m_off + tl.arange(0, BLOCK_M)
        offs_n = n_off + tl.arange(0, BLOCK_N)

        # ── accumulators in registers ──
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        wg_base = W_gate_ptr + expert_id * stride_we
        wu_base = W_up_ptr   + expert_id * stride_we

        # ── tiled K-loop ──
        for kb in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)

            x_mask = (offs_m[:, None] < m_bound) & (offs_k[None, :] < K)
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=x_mask, other=0.0,
            )

            w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            wg_tile = tl.load(
                wg_base + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask=w_mask, other=0.0,
            )
            wu_tile = tl.load(
                wu_base + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask=w_mask, other=0.0,
            )

            acc_gate += tl.dot(x_tile, wg_tile, allow_tf32=ALLOW_TF32)
            acc_up   += tl.dot(x_tile, wu_tile, allow_tf32=ALLOW_TF32)

        # ── fused SwiGLU in registers ──
        h_val = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

        # ── store output tile ──
        h_mask = (offs_m[:, None] < m_bound) & (offs_n[None, :] < N)
        tl.store(
            h_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn,
            h_val, mask=h_mask,
        )

        tile_id += num_sms  # round-robin


# ============================================================================
# Triton Kernel 2 — Persistent Down Projection  (all experts, 1 launch)
# ============================================================================

@triton.jit
def persistent_down_kernel(
    h_ptr,          # (A, Kin)
    W_down_ptr,     # (E, Kin, Nout)
    out_ptr,        # (A, Nout)
    tile_eids_ptr, tile_moffs_ptr, tile_noffs_ptr, tile_mbounds_ptr,
    Kin: tl.constexpr,        # d_hidden
    Nout: tl.constexpr,       # d_model
    total_tiles,
    stride_hm, stride_hk,
    stride_we, stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Persistent grouped GEMM:  out[rows] = h[rows] @ W_down[e]
    Same round-robin pattern as the SwiGLU kernel.
    """
    pid = tl.program_id(0)
    num_sms = tl.num_programs(0)

    tile_id = pid
    while tile_id < total_tiles:
        expert_id = tl.load(tile_eids_ptr + tile_id)
        m_off     = tl.load(tile_moffs_ptr + tile_id)
        n_off     = tl.load(tile_noffs_ptr + tile_id)
        m_bound   = tl.load(tile_mbounds_ptr + tile_id)

        offs_m = m_off + tl.arange(0, BLOCK_M)
        offs_n = n_off + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        wd_base = W_down_ptr + expert_id * stride_we

        for kb in range(0, tl.cdiv(Kin, BLOCK_K)):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)

            h_mask = (offs_m[:, None] < m_bound) & (offs_k[None, :] < Kin)
            h_tile = tl.load(
                h_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
                mask=h_mask, other=0.0,
            )

            w_mask = (offs_k[:, None] < Kin) & (offs_n[None, :] < Nout)
            wd_tile = tl.load(
                wd_base + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask=w_mask, other=0.0,
            )

            acc += tl.dot(h_tile, wd_tile, allow_tf32=ALLOW_TF32)

        out_mask = (offs_m[:, None] < m_bound) & (offs_n[None, :] < Nout)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc, mask=out_mask,
        )

        tile_id += num_sms


# ============================================================================
# Pad / Unpad helpers  (used only inside backward — no autograd needed)
# ============================================================================

def _pad_for_bmm(flat, counts, max_c, E, cols):
    """(A, cols) → (E, max_c, cols) with zero-padding per expert."""
    chunks = []
    off = 0
    for e in range(E):
        c = counts[e]
        if c > 0:
            ch = flat[off:off + c]
            if c < max_c:
                ch = torch.cat([ch, ch.new_zeros(max_c - c, cols)], dim=0)
            chunks.append(ch)
            off += c
        else:
            chunks.append(flat.new_zeros(max_c, cols))
    return torch.stack(chunks, dim=0)


def _unpad_from_bmm(padded, counts, total_rows, cols):
    """(E, max_c, cols) → (A, cols), extracting valid rows per expert."""
    out = padded.new_empty(total_rows, cols)
    off = 0
    for e, c in enumerate(counts):
        if c > 0:
            out[off:off + c] = padded[e, :c]
            off += c
    return out


# ============================================================================
# Autograd Function — Persistent forward + Batched backward
# ============================================================================

# ── Default block config (overridable via set_block_config) ────────────────
_BLOCK_CONFIG = {
    "BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_N": 64,
    "num_warps": 4, "num_stages": 3,
}

def set_block_config(BLOCK_M=64, BLOCK_K=64, BLOCK_N=64,
                     num_warps=4, num_stages=3):
    """Set global block config for the persistent kernels."""
    _BLOCK_CONFIG["BLOCK_M"] = BLOCK_M
    _BLOCK_CONFIG["BLOCK_K"] = BLOCK_K
    _BLOCK_CONFIG["BLOCK_N"] = BLOCK_N
    _BLOCK_CONFIG["num_warps"] = num_warps
    _BLOCK_CONFIG["num_stages"] = num_stages

def get_block_config():
    """Return current block config dict."""
    return dict(_BLOCK_CONFIG)


class PersistentMoEFunction(torch.autograd.Function):
    """
    forward  → 2 persistent Triton kernel launches  (fast)
    backward → batched GEMM via torch.bmm           (correct)
    """

    @staticmethod
    def forward(ctx, sorted_x, W_gate, W_up, W_down, expert_counts):
        """
        Parameters
        ----------
        sorted_x      : (A, D)  tokens packed by expert
        W_gate, W_up  : (E, D, H)
        W_down        : (E, H, D)
        expert_counts : (E,) int tensor — tokens per expert

        Returns
        -------
        sorted_out : (A, D)  expert outputs in same order as sorted_x
        """
        E, D, H = W_gate.shape
        A = sorted_x.shape[0]
        device = sorted_x.device
        dtype = sorted_x.dtype

        # ── read block config ──
        BLOCK_M = _BLOCK_CONFIG["BLOCK_M"]
        BLOCK_K = _BLOCK_CONFIG["BLOCK_K"]
        BLOCK_N = _BLOCK_CONFIG["BLOCK_N"]
        nw      = _BLOCK_CONFIG["num_warps"]
        ns      = _BLOCK_CONFIG["num_stages"]

        # ── build tile schedule on CPU (microseconds) ──
        ec = expert_counts.cpu().tolist()
        eo = [0]
        for c in ec:
            eo.append(eo[-1] + c)

        num_sms = torch.cuda.get_device_properties(device).multi_processor_count

        # Respect PyTorch's global TF32 setting in Triton kernels
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32

        # ── Kernel 1: persistent fused SwiGLU ──
        t1 = _precompute_tiles(ec, eo, H, BLOCK_M, BLOCK_N, device)
        te1, tm1, tn1, tb1, ntiles1 = t1

        h = torch.empty(A, H, device=device, dtype=dtype)
        if ntiles1 > 0:
            persistent_swiglu_kernel[(min(ntiles1, num_sms),)](
                sorted_x, W_gate, W_up, h,
                te1, tm1, tn1, tb1,
                D, H, ntiles1,
                sorted_x.stride(0), sorted_x.stride(1),
                W_gate.stride(0), W_gate.stride(1), W_gate.stride(2),
                h.stride(0), h.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
                ALLOW_TF32=allow_tf32,
                num_warps=nw, num_stages=ns,
            )

        # ── Kernel 2: persistent down projection ──
        t2 = _precompute_tiles(ec, eo, D, BLOCK_M, BLOCK_N, device)
        te2, tm2, tn2, tb2, ntiles2 = t2

        out = torch.empty(A, D, device=device, dtype=dtype)
        if ntiles2 > 0:
            persistent_down_kernel[(min(ntiles2, num_sms),)](
                h, W_down, out,
                te2, tm2, tn2, tb2,
                H, D, ntiles2,
                h.stride(0), h.stride(1),
                W_down.stride(0), W_down.stride(1), W_down.stride(2),
                out.stride(0), out.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
                ALLOW_TF32=allow_tf32,
                num_warps=nw, num_stages=ns,
            )

        ctx.save_for_backward(sorted_x, W_gate, W_up, W_down, expert_counts)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        sorted_x, W_gate, W_up, W_down, expert_counts = ctx.saved_tensors
        E, D, H = W_gate.shape
        A = sorted_x.shape[0]

        ec = expert_counts.cpu().tolist()
        max_c = max(ec) if ec else 0

        if A == 0 or max_c == 0:
            return (torch.zeros_like(sorted_x),
                    torch.zeros_like(W_gate), torch.zeros_like(W_up),
                    torch.zeros_like(W_down), None)

        # ── pad for batched GEMM ──
        px   = _pad_for_bmm(sorted_x, ec, max_c, E, D)   # (E, mc, D)
        pg   = _pad_for_bmm(grad_out, ec, max_c, E, D)    # (E, mc, D)

        # ── recompute intermediates (activation recomputation) ──
        h_gate = torch.bmm(px, W_gate)                     # (E, mc, H)
        h_up   = torch.bmm(px, W_up)                       # (E, mc, H)
        sig    = torch.sigmoid(h_gate)
        silu_g = h_gate * sig                               # silu(h_gate)
        h_full = silu_g * h_up                              # SwiGLU output

        # ── grad through down: out = h_full @ W_down ──
        grad_h      = torch.bmm(pg, W_down.transpose(1, 2))           # (E, mc, H)
        grad_W_down = torch.bmm(h_full.transpose(1, 2), pg)           # (E, H, D)

        # ── grad through SwiGLU: h = silu(h_gate) * h_up ──
        dsilu       = sig * (1.0 + h_gate * (1.0 - sig))
        grad_hgate  = grad_h * h_up * dsilu                            # (E, mc, H)
        grad_hup    = grad_h * silu_g                                  # (E, mc, H)

        # ── grad through matmuls ──
        grad_px     = (torch.bmm(grad_hgate, W_gate.transpose(1, 2)) +
                       torch.bmm(grad_hup,   W_up.transpose(1, 2)))   # (E, mc, D)
        grad_W_gate = torch.bmm(px.transpose(1, 2), grad_hgate)       # (E, D, H)
        grad_W_up   = torch.bmm(px.transpose(1, 2), grad_hup)         # (E, D, H)

        # ── unpad ──
        grad_sorted_x = _unpad_from_bmm(grad_px, ec, A, D)

        return grad_sorted_x, grad_W_gate, grad_W_up, grad_W_down, None


# ============================================================================
# Public API
# ============================================================================

def persistent_moe_experts(sorted_x, W_gate, W_up, W_down, expert_counts):
    """
    Process ALL experts with 2 persistent kernel launches.

    Block sizes controlled via set_block_config() / get_block_config().

    Parameters
    ----------
    sorted_x      : (A, D) tokens packed by expert
    W_gate, W_up  : (E, D, H) expert weights
    W_down        : (E, H, D) expert weights
    expert_counts : (E,) int tensor

    Returns
    -------
    sorted_out : (A, D) expert outputs (same order as sorted_x)
    """
    return PersistentMoEFunction.apply(sorted_x, W_gate, W_up, W_down, expert_counts)


# ============================================================================
# MoEFFN_Persistent  — drop-in replacement for MoEFFN
# ============================================================================

class MoEFFN_Persistent(nn.Module):
    """
    MoEFFN with persistent Triton kernels.

    Architecture unchanged:
      - Null expert support via data_sparsity
      - Shared expert (always active)
      - Same routing, same weight layout

    Optimization:
      - 2 persistent kernel launches replace 3×E individual matmuls
      - Column-major tile schedule for L2 cache locality
      - Round-robin work distribution across all SMs
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
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout

        from moe_standalone_kaggle import MoEGate
        self.gate = MoEGate(d_model, num_experts, top_k, data_sparsity=data_sparsity)

        self.W_gate = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_up   = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_down = nn.Parameter(torch.randn(num_experts, d_hidden, d_model) * 0.02)

        self.shared_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_up   = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_down = nn.Linear(d_hidden, d_model, bias=False)
        self._init_shared_weights()

        self.last_indices = None

    def _init_shared_weights(self):
        for m in [self.shared_gate, self.shared_up, self.shared_down]:
            m.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.num_experts
        device, dtype = x.device, x.dtype

        # ── shared expert (unchanged) ──
        shared_h = F.silu(self.shared_gate(x)) * self.shared_up(x)
        if self.training and self.dropout > 0:
            shared_h = F.dropout(shared_h, p=self.dropout)
        shared_out = self.shared_down(shared_h)

        # ── routing (unchanged) ──
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        self.last_indices = topk_idx.detach().clone()

        flat_x       = x.view(N, D)
        flat_idx     = topk_idx.view(N, K)
        flat_weight  = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)

        # ── filter nulls (unchanged) ──
        real_mask = ~flat_is_null
        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

        real_token_indices  = token_indices[real_mask]
        real_expert_indices = flat_idx[real_mask]
        real_weights        = flat_weight[real_mask]

        # ── sort by expert (unchanged) ──
        sort_idx              = real_expert_indices.argsort()
        sorted_token_indices  = real_token_indices[sort_idx]
        sorted_weights        = real_weights[sort_idx]
        sorted_x              = flat_x[sorted_token_indices]
        sorted_expert_indices = real_expert_indices[sort_idx]

        expert_counts = torch.bincount(sorted_expert_indices, minlength=E)

        # ⚡ PERSISTENT KERNEL — 2 launches for ALL experts ⚡
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)

        if sorted_token_indices.numel() > 0:
            sorted_out = persistent_moe_experts(
                sorted_x, self.W_gate, self.W_up, self.W_down, expert_counts,
            )

            # weighted scatter (unchanged from baseline)
            weighted_out = sorted_out * sorted_weights.unsqueeze(-1)
            routed_out.scatter_add_(
                0,
                sorted_token_indices.unsqueeze(-1).expand(-1, D),
                weighted_out,
            )

        routed_out = routed_out.view(B, T, D)

        # ── combine (unchanged) ──
        y = shared_out + routed_out
        return y, aux_loss


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Persistent Triton Kernel — Self-Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available — requires GPU")
        exit(0)

    gpu = torch.cuda.get_device_name(0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\nGPU: {gpu}  ({sms} SMs)")

    # ── import baseline for comparison ──
    from moe_standalone_kaggle import MoEFFN as MoEFFN_Baseline

    B, T, D, E, H = 2, 128, 576, 8, 1536
    print(f"Config: B={B}, T={T}, D={D}, experts={E}, hidden={H}\n")

    # ── build models with identical weights ──
    torch.manual_seed(42)
    baseline = MoEFFN_Baseline(D, H, E, top_k=2, data_sparsity=0.5).to(device)

    torch.manual_seed(42)
    persistent = MoEFFN_Persistent(D, H, E, top_k=2, data_sparsity=0.5).to(device)

    with torch.no_grad():
        persistent.W_gate.copy_(baseline.W_gate)
        persistent.W_up.copy_(baseline.W_up)
        persistent.W_down.copy_(baseline.W_down)
        persistent.shared_gate.weight.copy_(baseline.shared_gate.weight)
        persistent.shared_up.weight.copy_(baseline.shared_up.weight)
        persistent.shared_down.weight.copy_(baseline.shared_down.weight)
        persistent.gate.gate.weight.copy_(baseline.gate.gate.weight)
        persistent.gate.logit_bias.copy_(baseline.gate.logit_bias)
        persistent.gate.null_logit.copy_(baseline.gate.null_logit)

    # ── forward ──
    x_base = torch.randn(B, T, D, device=device, requires_grad=True)
    x_pers = x_base.clone().detach().requires_grad_(True)

    out_b, aux_b = baseline(x_base)
    out_p, aux_p = persistent(x_pers)

    out_abs  = (out_b - out_p).abs().max().item()
    out_rng  = out_b.abs().max().item() + 1e-8
    out_rel  = out_abs / out_rng

    print(f"Forward  abs diff: {out_abs:.2e}   rel diff: {out_rel:.2e}")

    # ── backward ──
    (out_b.sum() + aux_b).backward()
    (out_p.sum() + aux_p).backward()

    grad_abs = (x_base.grad - x_pers.grad).abs().max().item()
    grad_rng = x_base.grad.abs().max().item() + 1e-8
    grad_rel = grad_abs / grad_rng

    print(f"Gradient abs diff: {grad_abs:.2e}   rel diff: {grad_rel:.2e}")

    ok = out_rel < 1e-3 and grad_rel < 1e-3
    print(f"\n{'✅ PASS — persistent kernel matches baseline!' if ok else '❌ FAIL'}")

    if ok:
        # ── quick benchmark ──
        import time
        warmup, iters = 3, 20

        for model, label in [(baseline, "Baseline"), (persistent, "Persistent")]:
            model.eval()
            x_bench = torch.randn(B, T, D, device=device)
            with torch.no_grad():
                for _ in range(warmup):
                    model(x_bench)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    model(x_bench)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
            print(f"  {label:12s}: {elapsed/iters*1e3:.2f} ms/iter")

        print()

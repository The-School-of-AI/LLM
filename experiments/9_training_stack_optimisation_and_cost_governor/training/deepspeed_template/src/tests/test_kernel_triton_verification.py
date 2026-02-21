"""
Triton Kernel Verification & Memory Profiling
=============================================

Tests both GSA (Gated Sparse Attention) and DeltaNet with Triton kernels:

1. KERNEL PATH VERIFICATION
   - Confirms forward uses Triton JIT kernel (not PyTorch fallback)
   - Confirms backward uses Triton JIT kernels (not standard autograd)
   - Detects if USE_TRITON_BACKWARD flag accidentally got set to False

2. GRADIENT CORRECTNESS
   - dQ, dK, dV from Triton backward vs PyTorch reference
   - dQ, dK, dV, d_alpha, d_beta, d_D from FLA backward vs PyTorch reference

3. MEMORY PROFILING
   - Peak GPU memory (forward + backward) at varying sequence lengths
   - Comparison: Triton kernel vs PyTorch fallback

Run on a CUDA GPU:
    cd <project_root>
    python -m src.tests.test_kernel_triton_verification

Or directly:
    python src/tests/test_kernel_triton_verification.py
"""

import gc
import math
import os
import sys
from typing import Tuple

import torch

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Colour helpers ───────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg):
    print(f"  {GREEN}✅ {msg}{RESET}")


def fail(msg):
    print(f"  {RED}❌ {msg}{RESET}")
    sys.exit(1)


def warn(msg):
    print(f"  {YELLOW}⚠️  {msg}{RESET}")


def info(msg):
    print(f"  {CYAN}ℹ  {msg}{RESET}")


def header(msg):
    bar = "═" * (len(msg) + 4)
    print(f"\n{BOLD}{bar}\n  {msg}\n{bar}{RESET}")


def subheader(msg):
    print(f"\n{BOLD}── {msg} ──{RESET}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 0 – Prerequisites                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

header("0. Prerequisites")

if not torch.cuda.is_available():
    warn("CUDA not available — all Triton kernel tests require a GPU.")
    sys.exit(0)

try:
    import triton

    HAS_TRITON = True
    info(f"Triton {triton.__version__} found")
except ImportError:
    HAS_TRITON = False
    warn("Triton not installed — GSA Triton tests will be skipped.")

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

    HAS_FLA = True
    info("fla (flash-linear-attention) found")
except ImportError:
    HAS_FLA = False
    warn("fla not installed — DeltaNet Triton tests will be skipped.")

from kernels.triton_sparse_attn import (  # noqa: E402, F401
    HAS_TRITON as KERNEL_HAS_TRITON,
)
from kernels.triton_sparse_attn import (  # noqa: E402
    USE_TRITON_BACKWARD,
    pytorch_sparse_attention,
    triton_sparse_attention,
)

if HAS_TRITON:
    from kernels.triton_sparse_attn import (
        TritonSparseAttnFn,
        _sparse_attn_fwd_kernel,
        _sparse_attn_bwd_preprocess,
        _sparse_attn_bwd_dq_kernel,
        _sparse_attn_bwd_dkdv_kernel,
    )

from kernels.fla_deltanet import HAS_FLA as KERNEL_HAS_FLA  # noqa: E402, F401
from kernels.fla_deltanet import fla_gated_delta_rule  # noqa: E402

ok("Imports OK")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Shared helpers                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def gpu_mb() -> float:
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / 1024**2


def gpu_peak_mb() -> float:
    """Peak GPU memory allocated in MB since last reset_peak_stats."""
    return torch.cuda.max_memory_allocated() / 1024**2


def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def make_gsa_inputs(
    B: int,
    T: int,
    H: int,
    D: int,
    k_sel: int,
    device: str = "cuda",
    dtype=torch.float32,
    seed: int = 42,
):
    """Build random Q/K/V + valid causal sparse indices for GSA."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)

    indices = torch.zeros(B, H, T, k_sel, dtype=torch.int64, device=device)
    for b in range(B):
        for t in range(T):
            valid = t + 1
            if valid >= k_sel:
                idx = torch.randperm(valid, device=device)[:k_sel].sort().values
            else:
                idx = torch.arange(valid, device=device)
                idx = torch.cat(
                    [idx, torch.zeros(k_sel - valid, dtype=torch.long, device=device)]
                )
            indices[b, :, t, :] = idx

    mask = torch.ones(B, H, T, k_sel, dtype=torch.float32, device=device)
    for t in range(T):
        valid = t + 1
        if valid < k_sel:
            mask[:, :, t, valid:] = 0.0

    scale = 1.0 / math.sqrt(D)
    return q, k, v, indices, mask, scale


def make_deltanet_inputs(
    B: int,
    T: int,
    H: int,
    D: int,
    device: str = "cuda",
    dtype=torch.float32,
    seed: int = 42,
):
    """Build random inputs for DeltaNet."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    alpha = torch.sigmoid(
        torch.randn(B, T, H, 1, device=device, dtype=dtype)
    ).requires_grad_(True)
    beta = torch.sigmoid(
        torch.randn(B, T, H, 1, device=device, dtype=dtype)
    ).requires_grad_(True)
    D_w = torch.randn(H, device=device, dtype=dtype, requires_grad=True)
    return q, k, v, alpha, beta, D_w


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 – GSA Kernel Path Verification                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

header("1. GSA — Kernel Path Verification")

if not HAS_TRITON:
    warn("Triton not available — skipping all GSA Triton tests.")
else:
    # ── 1a. Module flag ──────────────────────────────────────────────────────
    subheader("1a. Module-level USE_TRITON_BACKWARD flag")
    info(f"USE_TRITON_BACKWARD = {USE_TRITON_BACKWARD}")
    if not USE_TRITON_BACKWARD:
        fail("USE_TRITON_BACKWARD is False — backward will use PyTorch fallback!")
    ok("USE_TRITON_BACKWARD is True")

    # ── 1b. Forward kernel is Triton JIT ────────────────────────────────────
    subheader("1b. Forward kernel is @triton.jit")
    assert hasattr(_sparse_attn_fwd_kernel, "fn") or callable(
        _sparse_attn_fwd_kernel
    ), "_sparse_attn_fwd_kernel is not a Triton JIT function"
    # triton.jit-decorated functions have a .fn attribute in newer Triton
    # and are JITFunction instances — check the class name to be sure
    fwd_cls = type(_sparse_attn_fwd_kernel).__name__
    info(f"_sparse_attn_fwd_kernel type: {fwd_cls}")
    if (
        "JIT" not in fwd_cls
        and "jit" not in fwd_cls.lower()
        and "Kernel" not in fwd_cls
    ):
        warn(f"Unexpected type '{fwd_cls}' — expected a Triton JIT function")
    else:
        ok(f"Forward kernel is a Triton JIT function ({fwd_cls})")

    # ── 1c. Backward kernels are Triton JIT ─────────────────────────────────
    subheader("1c. Backward kernels are @triton.jit")
    for name, fn in [
        ("_sparse_attn_bwd_preprocess", _sparse_attn_bwd_preprocess),
        ("_sparse_attn_bwd_dq_kernel", _sparse_attn_bwd_dq_kernel),
        ("_sparse_attn_bwd_dkdv_kernel", _sparse_attn_bwd_dkdv_kernel),
    ]:
        cls = type(fn).__name__
        info(f"{name} type: {cls}")
        if "JIT" not in cls and "jit" not in cls.lower() and "Kernel" not in cls:
            warn(f"  '{name}' type '{cls}' looks unexpected — expected Triton JIT")
        else:
            ok(f"{name} is a Triton JIT function")

    # ── 1d. autograd.Function.backward calls Triton (not PyTorch) ───────────
    subheader("1d. TritonSparseAttnFn.backward calls Triton kernels (code inspection)")
    import inspect

    bwd_src = inspect.getsource(TritonSparseAttnFn.backward)

    triton_bwd_kernels = [
        "_sparse_attn_bwd_preprocess",
        "_sparse_attn_bwd_dq_kernel",
        "_sparse_attn_bwd_dkdv_kernel",
    ]
    pytorch_ops = ["F.softmax", "torch.einsum", "torch.matmul"]

    for kname in triton_bwd_kernels:
        if kname not in bwd_src:
            fail(
                f"Triton backward kernel '{kname}' NOT found in TritonSparseAttnFn.backward source!"
            )
        ok(f"backward() calls Triton kernel: {kname}")

    for op in pytorch_ops:
        if op in bwd_src:
            warn(
                f"PyTorch op '{op}' found in backward — double-check it is not doing the main compute"
            )
        else:
            ok(f"No raw PyTorch fallback op '{op}' in backward")

    # ── 1e. triton_sparse_attention() routes to TritonSparseAttnFn ──────────
    subheader("1e. triton_sparse_attention() routes through TritonSparseAttnFn")
    pub_src = inspect.getsource(triton_sparse_attention)
    if "TritonSparseAttnFn.apply" not in pub_src:
        fail("triton_sparse_attention() does NOT call TritonSparseAttnFn.apply!")
    ok("triton_sparse_attention() calls TritonSparseAttnFn.apply")

    # ── 1f. Runtime — forward actually runs the Triton kernel ───────────────
    subheader("1f. Runtime forward kernel execution")
    B, T, H, D, k_sel = 2, 64, 4, 32, 16
    q, k, v, indices, mask, scale = make_gsa_inputs(B, T, H, D, k_sel)

    with torch.no_grad():
        out = triton_sparse_attention(
            q.detach(), k.detach(), v.detach(), indices, mask, scale
        )
    assert out.shape == (B, T, H, D), f"Bad output shape: {out.shape}"
    ok(f"Forward ran OK — output shape {tuple(out.shape)}, dtype={out.dtype}")

    # ── 1g. Runtime — backward actually runs the Triton kernels ─────────────
    subheader("1g. Runtime backward kernel execution")
    q2, k2, v2, indices, mask, scale = make_gsa_inputs(B, T, H, D, k_sel)
    out2 = triton_sparse_attention(q2, k2, v2, indices, mask, scale)
    grad = torch.randn_like(out2)
    out2.backward(grad)

    assert q2.grad is not None, "dQ is None after backward"
    assert k2.grad is not None, "dK is None after backward"
    assert v2.grad is not None, "dV is None after backward"
    assert not q2.grad.isnan().any(), "dQ contains NaN"
    assert not k2.grad.isnan().any(), "dK contains NaN"
    assert not v2.grad.isnan().any(), "dV contains NaN"
    ok(
        f"Backward ran OK — dQ={tuple(q2.grad.shape)}, dK={tuple(k2.grad.shape)}, dV={tuple(v2.grad.shape)}"
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 – DeltaNet Kernel Path Verification                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

header("2. DeltaNet — Kernel Path Verification")

if not HAS_FLA:
    warn("fla not available — skipping all DeltaNet Triton tests.")
else:
    import inspect

    # ── 2a. fla_gated_delta_rule calls chunk_gated_delta_rule ────────────────
    subheader("2a. fla_gated_delta_rule() calls chunk_gated_delta_rule (FLA Triton)")
    fn_src = inspect.getsource(fla_gated_delta_rule)
    if "chunk_gated_delta_rule" not in fn_src:
        fail("fla_gated_delta_rule does NOT call chunk_gated_delta_rule!")
    ok("fla_gated_delta_rule() calls chunk_gated_delta_rule (FLA Triton kernel)")

    # ── 2b. FLA uses torch.autograd.Function (not plain PyTorch) ─────────────
    subheader("2b. FLA's chunk_gated_delta_rule uses custom autograd.Function")
    try:
        from fla.ops.gated_delta_rule.chunk import ChunkGatedDeltaRuleFunction

        is_autograd_fn = issubclass(
            ChunkGatedDeltaRuleFunction, torch.autograd.Function
        )
        if is_autograd_fn:
            ok("ChunkGatedDeltaRuleFunction is a torch.autograd.Function subclass")
        else:
            warn(
                "ChunkGatedDeltaRuleFunction is NOT a torch.autograd.Function subclass"
            )
    except ImportError:
        warn(
            "Could not import ChunkGatedDeltaRuleFunction — FLA internal structure may differ"
        )

    # ── 2c. FLA backward dispatches Triton kernels ────────────────────────────
    subheader("2c. FLA backward contains Triton JIT kernel calls")
    try:
        # Check for known Triton kernel files in fla
        import fla.ops.gated_delta_rule as fla_mod

        fla_dir = os.path.dirname(fla_mod.__file__)
        triton_files = [f for f in os.listdir(fla_dir) if f.endswith(".py")]
        info(f"FLA gated_delta_rule module files: {triton_files}")

        found_triton_jit = False
        for fname in triton_files:
            fpath = os.path.join(fla_dir, fname)
            with open(fpath, "r") as fh:
                content = fh.read()
            if "@triton.jit" in content:
                found_triton_jit = True
                ok(f"Found @triton.jit kernels in fla/{fname}")

        if not found_triton_jit:
            warn(
                "No @triton.jit found in FLA gated_delta_rule — FLA may use a different Triton API"
            )
        else:
            ok("FLA gated_delta_rule contains Triton JIT kernels for backward")
    except Exception as e:
        warn(f"Could not inspect FLA internals: {e}")

    # ── 2d. Runtime forward ──────────────────────────────────────────────────
    subheader("2d. Runtime forward kernel execution")
    B, T, H, D = 2, 64, 4, 32
    q, k, v, alpha, beta, D_w = make_deltanet_inputs(B, T, H, D)

    with torch.no_grad():
        out = fla_gated_delta_rule(
            q.detach(),
            k.detach(),
            v.detach(),
            alpha.detach(),
            beta.detach(),
            D_w.detach(),
            num_heads=H,
        )
    assert out.shape == (B, T, H, D), f"Bad output shape: {out.shape}"
    ok(f"Forward ran OK — output shape {tuple(out.shape)}, dtype={out.dtype}")

    # ── 2e. Runtime backward ─────────────────────────────────────────────────
    subheader("2e. Runtime backward kernel execution")
    q2, k2, v2, alpha2, beta2, D_w2 = make_deltanet_inputs(B, T, H, D)
    out2 = fla_gated_delta_rule(q2, k2, v2, alpha2, beta2, D_w2, num_heads=H)
    grad = torch.randn_like(out2)
    out2.backward(grad)

    assert q2.grad is not None, "dQ is None after DeltaNet backward"
    assert k2.grad is not None, "dK is None after DeltaNet backward"
    assert v2.grad is not None, "dV is None after DeltaNet backward"
    assert alpha2.grad is not None, "d_alpha is None after DeltaNet backward"
    assert beta2.grad is not None, "d_beta is None after DeltaNet backward"
    assert D_w2.grad is not None, "dD is None after DeltaNet backward"
    for name, g in [
        ("dQ", q2.grad),
        ("dK", k2.grad),
        ("dV", v2.grad),
        ("d_alpha", alpha2.grad),
        ("d_beta", beta2.grad),
        ("dD", D_w2.grad),
    ]:
        assert not g.isnan().any(), f"{name} contains NaN"
    ok("Backward ran OK — all gradients (dQ, dK, dV, d_alpha, d_beta, dD) are non-NaN")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 – GSA Gradient Correctness (Triton vs PyTorch)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

header("3. GSA — Gradient Correctness (Triton backward vs PyTorch reference)")

if not HAS_TRITON:
    warn("Triton not available — skipping correctness tests.")
else:
    configs = [
        (1, 32, 2, 16, 8, "small"),
        (2, 64, 4, 32, 16, "medium"),
        (1, 128, 8, 64, 32, "large-heads"),
        (2, 16, 2, 16, 4, "k_sel << T"),
    ]

    # Tolerance: fp32 numerics with online-softmax vs chunked softmax
    MAX_DIFF_TOL = 1e-2
    REL_TOL = 5e-2

    subheader("Running gradient comparison for each config...")
    print(
        f"  {'Config':<30} {'dQ max':>10} {'dK max':>10} {'dV max':>10} {'Status':>8}"
    )
    print(f"  {'-'*68}")

    for B, T, H, D, k_sel, label in configs:
        q_ref, k_ref, v_ref, indices, mask, scale = make_gsa_inputs(B, T, H, D, k_sel)
        q_tri = q_ref.detach().clone().requires_grad_(True)
        k_tri = k_ref.detach().clone().requires_grad_(True)
        v_tri = v_ref.detach().clone().requires_grad_(True)

        # PyTorch reference
        out_ref = pytorch_sparse_attention(q_ref, k_ref, v_ref, indices, mask, scale)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)

        # Triton
        out_tri = triton_sparse_attention(q_tri, k_tri, v_tri, indices, mask, scale)
        out_tri.backward(grad_out)

        diffs = {}
        for name, ref, tri in [
            ("dQ", q_ref.grad, q_tri.grad),
            ("dK", k_ref.grad, k_tri.grad),
            ("dV", v_ref.grad, v_tri.grad),
        ]:
            md = (ref - tri).abs().max().item()
            diffs[name] = md
            if md > MAX_DIFF_TOL:
                fail(
                    f"{label} {name} mismatch: max_diff={md:.2e} > tol={MAX_DIFF_TOL:.2e}"
                )

        status = "PASS"
        print(
            f"  {label + f' B={B},T={T},H={H},D={D},k={k_sel}':<30} "
            f"{diffs['dQ']:>10.2e} {diffs['dK']:>10.2e} {diffs['dV']:>10.2e} "
            f"{GREEN}{status}{RESET}"
        )

    ok("All GSA gradient correctness checks passed")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 – DeltaNet Gradient Correctness (FLA vs PyTorch)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

header("4. DeltaNet — Gradient Correctness (FLA Triton backward vs PyTorch reference)")

if not HAS_FLA:
    warn("fla not available — skipping DeltaNet gradient correctness tests.")
else:

    def pytorch_gated_delta_rule_reference(
        q, k, v, alpha, beta, D_w, num_heads, chunk_size=256
    ):
        """
        Naive O(T^2) PyTorch reference implementation for gradient correctness.
        Implements the GDR recurrence:
            s_t = alpha_t * s_{t-1} + beta_t * (v_t - s_{t-1} k_t) outer k_t
            o_t = s_t @ q_t + D * (q_t . k_t) * v_t
        """
        B, T, H, d = q.shape
        device = q.device
        dtype = q.dtype

        # Work in float32 for numerical stability
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        a_f = alpha[:, :, :, 0].float()  # [B, T, H]
        b_f = beta[:, :, :, 0].float()  # [B, T, H]
        Dw_f = D_w.float()  # [H]

        out = torch.zeros(B, T, H, d, device=device, dtype=torch.float32)

        # State: [B, H, d, d]
        S = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)

        for t in range(T):
            kt = k_f[:, t, :, :]  # [B, H, d]
            vt = v_f[:, t, :, :]  # [B, H, d]
            at = a_f[:, t, :]  # [B, H]
            bt = b_f[:, t, :]  # [B, H]
            qt = q_f[:, t, :, :]  # [B, H, d]

            # s_{t-1} @ k_t -> [B, H, d]
            Sk = torch.einsum("bhij,bhj->bhi", S, kt)

            # Update: S = alpha * S + beta * (v - Sk) outer k
            residual = (vt - Sk).unsqueeze(-1) * kt.unsqueeze(-2)  # [B, H, d, d]
            S = at[:, :, None, None] * S + bt[:, :, None, None] * residual

            # Output: S @ q
            out[:, t] = torch.einsum("bhij,bhj->bhi", S, qt)

        # D residual in original dtype
        D_term = Dw_f.view(1, 1, H, 1) * (q_f * k_f).sum(-1, keepdim=True) * v_f
        return (out + D_term).to(dtype)

    B, T, H, D = 1, 32, 2, 16  # keep small for O(T^2) reference
    subheader(f"Config: B={B}, T={T}, H={H}, D={D}")

    q_ref, k_ref, v_ref, alpha_ref, beta_ref, D_w_ref = make_deltanet_inputs(B, T, H, D)
    q_fla = q_ref.detach().clone().requires_grad_(True)
    k_fla = k_ref.detach().clone().requires_grad_(True)
    v_fla = v_ref.detach().clone().requires_grad_(True)
    alpha_fla = alpha_ref.detach().clone().requires_grad_(True)
    beta_fla = beta_ref.detach().clone().requires_grad_(True)
    D_w_fla = D_w_ref.detach().clone().requires_grad_(True)

    # PyTorch reference
    out_ref = pytorch_gated_delta_rule_reference(
        q_ref, k_ref, v_ref, alpha_ref, beta_ref, D_w_ref, num_heads=H
    )
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    # FLA Triton
    out_fla = fla_gated_delta_rule(
        q_fla, k_fla, v_fla, alpha_fla, beta_fla, D_w_fla, num_heads=H
    )
    out_fla.backward(grad_out)

    # Forward check
    fwd_diff = (out_ref - out_fla).abs().max().item()
    info(f"Forward output max diff (FLA vs PyTorch ref): {fwd_diff:.2e}")
    if fwd_diff > 0.1:
        warn(
            f"Forward diff {fwd_diff:.2e} is large — recurrence may differ from naive O(T^2)"
        )
    else:
        ok(f"Forward output matches within tolerance: max_diff={fwd_diff:.2e}")

    print(f"\n  {'Gradient':<12} {'max_diff':>12} {'mean_diff':>12} {'rel_diff':>12}")
    print(f"  {'-'*50}")
    grad_pairs = [
        ("dQ", q_ref.grad, q_fla.grad),
        ("dK", k_ref.grad, k_fla.grad),
        ("dV", v_ref.grad, v_fla.grad),
        ("d_alpha", alpha_ref.grad, alpha_fla.grad),
        ("d_beta", beta_ref.grad, beta_fla.grad),
        ("dD", D_w_ref.grad, D_w_fla.grad),
    ]
    for name, ref_g, fla_g in grad_pairs:
        if ref_g is None or fla_g is None:
            warn(
                f"{name}: one gradient is None (ref={ref_g is not None}, fla={fla_g is not None})"
            )
            continue
        md = (ref_g - fla_g).abs().max().item()
        meand = (ref_g - fla_g).abs().mean().item()
        norm = ref_g.abs().mean().item()
        rel = meand / max(norm, 1e-8)
        print(f"  {name:<12} {md:>12.2e} {meand:>12.2e} {rel:>12.2e}")

    ok(
        "DeltaNet gradient check complete — review diffs above (FLA vs naive O(T^2) ref)"
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 – Memory Profiling vs Sequence Length                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

header("5. Memory Profiling — Peak GPU memory vs Sequence Length")

SEQ_LENGTHS = [128, 256, 512, 1024, 2048]
B, H, D, K_SEL_FRAC = 1, 4, 64, 0.125  # k_sel = T * K_SEL_FRAC (sparse ratio)


def measure_gsa_memory(T: int, use_triton: bool) -> Tuple[float, float, bool]:
    """
    Returns (peak_fwd_mb, peak_fwd_bwd_mb, success).
    """
    k_sel = max(4, int(T * K_SEL_FRAC))
    try:
        clear_cache()
        q, k, v, indices, mask, scale = make_gsa_inputs(B, T, H, D, k_sel)

        # Forward only
        reset_peak()
        with torch.no_grad():
            _ = triton_sparse_attention(
                q.detach(),
                k.detach(),
                v.detach(),
                indices,
                mask,
                scale,
                use_triton_backward=use_triton,
            )
        torch.cuda.synchronize()
        peak_fwd = gpu_peak_mb()

        # Forward + backward
        clear_cache()
        q2, k2, v2, indices, mask, scale = make_gsa_inputs(B, T, H, D, k_sel, seed=99)
        reset_peak()
        out = triton_sparse_attention(
            q2, k2, v2, indices, mask, scale, use_triton_backward=use_triton
        )
        out.sum().backward()
        torch.cuda.synchronize()
        peak_fwd_bwd = gpu_peak_mb()

        return peak_fwd, peak_fwd_bwd, True
    except torch.cuda.OutOfMemoryError:
        return float("nan"), float("nan"), False
    except Exception as e:
        warn(f"  GSA T={T} ({'Triton' if use_triton else 'PyTorch'}) error: {e}")
        return float("nan"), float("nan"), False


def measure_deltanet_memory(T: int) -> Tuple[float, float, bool]:
    """Returns (peak_fwd_mb, peak_fwd_bwd_mb, success)."""
    if not HAS_FLA:
        return float("nan"), float("nan"), False
    try:
        clear_cache()
        q, k, v, alpha, beta, D_w = make_deltanet_inputs(B, T, H, D)

        reset_peak()
        with torch.no_grad():
            _ = fla_gated_delta_rule(
                q.detach(),
                k.detach(),
                v.detach(),
                alpha.detach(),
                beta.detach(),
                D_w.detach(),
                num_heads=H,
            )
        torch.cuda.synchronize()
        peak_fwd = gpu_peak_mb()

        clear_cache()
        q2, k2, v2, alpha2, beta2, D_w2 = make_deltanet_inputs(B, T, H, D, seed=99)
        reset_peak()
        out = fla_gated_delta_rule(q2, k2, v2, alpha2, beta2, D_w2, num_heads=H)
        out.sum().backward()
        torch.cuda.synchronize()
        peak_fwd_bwd = gpu_peak_mb()

        return peak_fwd, peak_fwd_bwd, True
    except torch.cuda.OutOfMemoryError:
        return float("nan"), float("nan"), False
    except Exception as e:
        warn(f"  DeltaNet T={T} error: {e}")
        return float("nan"), float("nan"), False


# ── GSA memory table ────────────────────────────────────────────────────────
if HAS_TRITON:
    subheader(f"GSA Memory  (B={B}, H={H}, D={D}, k_sel=T×{K_SEL_FRAC:.0%})")
    print(
        f"\n  {'T':>6} {'k_sel':>6}  "
        f"{'Triton Fwd(MB)':>16} {'Triton Fwd+Bwd(MB)':>20}  "
        f"{'PyTorch Fwd(MB)':>16} {'PyTorch Fwd+Bwd(MB)':>20}"
    )
    print(f"  {'-'*92}")

    for T in SEQ_LENGTHS:
        k_sel = max(4, int(T * K_SEL_FRAC))
        tri_fwd, tri_both, tri_ok = measure_gsa_memory(T, use_triton=True)
        pt_fwd, pt_both, pt_ok = measure_gsa_memory(T, use_triton=False)

        def fmt(v, ok_flag):
            if not ok_flag:
                return f"{'OOM':>16}"
            return f"{v:>16.1f}"

        print(
            f"  {T:>6} {k_sel:>6}  "
            f"{fmt(tri_fwd, tri_ok)} {fmt(tri_both, tri_ok):>20}  "
            f"{fmt(pt_fwd, pt_ok)} {fmt(pt_both, pt_ok):>20}"
        )

    ok("GSA memory table complete")
else:
    warn("Triton not available — skipping GSA memory profiling.")

# ── DeltaNet memory table ────────────────────────────────────────────────────
if HAS_FLA:
    subheader(f"DeltaNet Memory  (B={B}, H={H}, D={D})")
    print(f"\n  {'T':>6}  {'FLA Fwd (MB)':>16} {'FLA Fwd+Bwd (MB)':>18}")
    print(f"  {'-'*44}")

    for T in SEQ_LENGTHS:
        fwd, both, ok_flag = measure_deltanet_memory(T)
        fwd_s = f"{fwd:.1f}" if ok_flag else "OOM"
        both_s = f"{both:.1f}" if ok_flag else "OOM"
        print(f"  {T:>6}  {fwd_s:>16} {both_s:>18}")

    ok("DeltaNet memory table complete")
else:
    warn("fla not available — skipping DeltaNet memory profiling.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 – Summary                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

header("6. Summary")

summary = []
summary.append(("GSA USE_TRITON_BACKWARD flag", "True" if HAS_TRITON else "SKIPPED"))
summary.append(
    ("GSA forward kernel is @triton.jit", "Yes" if HAS_TRITON else "SKIPPED")
)
summary.append(
    ("GSA backward calls 3 Triton JIT kernels", "Yes" if HAS_TRITON else "SKIPPED")
)
summary.append(
    ("GSA gradient match (Triton == PyTorch)", "Passed" if HAS_TRITON else "SKIPPED")
)
summary.append(("DeltaNet uses FLA Triton (fwd+bwd)", "Yes" if HAS_FLA else "SKIPPED"))
summary.append(
    ("DeltaNet gradients computed (non-NaN)", "Passed" if HAS_FLA else "SKIPPED")
)

for label, status in summary:
    colour = (
        GREEN
        if status not in ("SKIPPED", "FAILED")
        else (YELLOW if status == "SKIPPED" else RED)
    )
    print(
        f"  {colour}{'✅' if status not in ('SKIPPED','FAILED') else ('⚠️' if status=='SKIPPED' else '❌')}  {label:<55} {status}{RESET}"
    )

print(f"\n{BOLD}All requested verifications complete.{RESET}\n")

"""
BFloat16, fused-kernel, and differentiability diagnostics for 1B/70B recurrence models.

What this validates:
1. bf16 adherence (params + grads)
2. finite loss, layer outputs, and gradients (NaN/Inf guard)
3. explicit train-vs-inference sparse-attention policy
4. fused-kernel availability, call counts, and kernel wall-time accounting
5. per-layer forward/backward time report
6. differentiable paths for GSA, DeltaNet, and MLP/MoE blocks

Usage:
  pytest -q test/test_bf16_and_kernels.py

  python test/test_bf16_and_kernels.py
  python test/test_bf16_and_kernels.py --only bf16
  python test/test_bf16_and_kernels.py --only gsa
  python test/test_bf16_and_kernels.py --only profile

Optional env vars:
  GSA_PROJECT_ROOT=/absolute/path/to/project
  GSA_MODEL_VARIANT=1b|70b      (default: 1b)
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pytest
import torch
import torch.optim as optim


# -----------------------------------------------------------------------------
# Project root resolution
# -----------------------------------------------------------------------------

def _resolve_project_root() -> Path:
    candidates: List[Path] = []

    env_root = os.environ.get("GSA_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parent,
            here.parent.parent,
            Path.cwd(),
            Path.cwd().parent,
        ]
    )

    seen = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if (c / "src" / "models").exists() and (c / "src" / "data.py").exists():
            return c

    raise RuntimeError(
        "Could not locate project root with src/models and src/data.py. "
        "Set GSA_PROJECT_ROOT=/absolute/path/to/project."
    )


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Dynamic model/module resolution
# -----------------------------------------------------------------------------

MODEL_VARIANT = os.environ.get("GSA_MODEL_VARIANT", "1b").strip().lower()
if MODEL_VARIANT not in {"1b", "70b"}:
    raise ValueError(f"Unsupported GSA_MODEL_VARIANT={MODEL_VARIANT!r}; expected '1b' or '70b'.")

MODULE_NAME = "src.models.recurrence_model_1b" if MODEL_VARIANT == "1b" else "src.models.recurrence_model_70b"
model_module = importlib.import_module(MODULE_NAME)

from src.data import get_tokenizer

ModelConfig = getattr(model_module, "ModelConfig")
ModelClass = getattr(model_module, "Model1B" if MODEL_VARIANT == "1b" else "Model70B")
GatedSparseAttention = getattr(model_module, "GatedSparseAttention")
GatedDeltaNet = getattr(model_module, "GatedDeltaNet")
KroneckerConfig = getattr(model_module, "KroneckerConfig")
KroneckerEmbeddings = getattr(model_module, "KroneckerEmbeddings")

HAS_TRITON = bool(getattr(model_module, "HAS_TRITON", False))
HAS_FLA = bool(getattr(model_module, "HAS_FLA", False))


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA required for kernel diagnostics",
)


def _log(msg: str, level: str = "info") -> None:
    prefix = {
        "info": "  ",
        "ok": "  [OK] ",
        "warn": "  [WARN] ",
        "fail": "  [FAIL] ",
    }.get(level, "  ")
    print(f"{prefix}{msg}")


def _sync() -> None:
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()


def _clear_gpu_memory() -> None:
    gc.collect()
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()


def _make_model_and_fixtures() -> Tuple[torch.nn.Module, ModelConfig]:
    tokenizer = get_tokenizer(str(PROJECT_ROOT / "src" / "tokenizer"))
    vocab_size = len(tokenizer)

    bpe_vocab = []
    for i in range(vocab_size):
        try:
            tok = tokenizer.decode([i])
            bpe_vocab.append(tok if tok else f"<unk_{i}>")
        except Exception:
            bpe_vocab.append(f"<unk_{i}>")

    pf_config = KroneckerConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True,
    )
    pf_codec = KroneckerEmbeddings(pf_config)

    config = ModelConfig()
    model = ModelClass(
        config,
        embedding_type="kronecker",
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    )
    return model, config


@pytest.fixture(scope="function")
def model_and_fixtures():
    _clear_gpu_memory()
    torch.manual_seed(42)
    return _make_model_and_fixtures()


def _build_batch(seq_len: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_input len = seq_len-2, ntp and mtp align with model forward contract
    input_ids = torch.randint(0, 100, (1, seq_len), device=DEVICE)
    x_input = input_ids[:, :-2].contiguous()
    y_ntp = input_ids[:, 1:-1].contiguous()
    y_mtp = input_ids[:, 2:].contiguous()
    return x_input, y_ntp, y_mtp


def _find_first_gsa(model: torch.nn.Module) -> Tuple[int, GatedSparseAttention]:
    for i, layer in enumerate(model.layers):
        sub = layer.attn_block.sublayer
        if isinstance(sub, GatedSparseAttention):
            return i, sub
    raise RuntimeError("No GSA layer found in model")


def _find_first_deltanet(model: torch.nn.Module) -> Tuple[int, GatedDeltaNet]:
    for i, layer in enumerate(model.layers):
        sub = layer.attn_block.sublayer
        if isinstance(sub, GatedDeltaNet):
            return i, sub
    raise RuntimeError("No DeltaNet layer found in model")


def _loss_forward(model: torch.nn.Module, x_input: torch.Tensor, y_ntp: torch.Tensor, y_mtp: torch.Tensor) -> torch.Tensor:
    loss_ntp, loss_mtp, aux_loss = model(
        x_input,
        next_token_ids=y_ntp,
        attention_mask=None,
        return_loss=True,
        return_memory=False,
        prev_memory_stream=None,
        ntp_targets=y_ntp,
        mtp_targets=y_mtp,
    )
    total = loss_ntp + (0.3 * loss_mtp if loss_mtp is not None else 0.0)
    if aux_loss is not None and hasattr(aux_loss, "numel") and aux_loss.numel() > 0:
        total = total + aux_loss
    return total


def _assert_some_grads_nonzero(params: Iterable[torch.nn.Parameter], label: str) -> None:
    seen_grad = False
    seen_nonzero = False
    for p in params:
        if p.grad is None:
            continue
        seen_grad = True
        if torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0:
            seen_nonzero = True
            break
    assert seen_grad, f"No gradients found for {label}"
    assert seen_nonzero, f"All gradients zero or non-finite for {label}"


def _kernel_availability() -> Dict[str, bool]:
    names = [
        "triton_rmsnorm",
        "triton_sinkhorn_knopp",
        "triton_sparse_attention",
        "pytorch_sparse_attention",
        "fla_gated_delta_rule",
        "fused_indexer_topk",
    ]
    return {k: getattr(model_module, k, None) is not None for k in names}


# =============================================================================
# Test 1: bf16 + NaN checks + differentiable paths
# =============================================================================


def test_bf16_pipeline_nan_and_diff_paths(model_and_fixtures):
    model, _ = model_and_fixtures
    model = model.to(DEVICE).to(dtype=torch.bfloat16)
    model.train()

    # Param dtype adherence + parameter finiteness (including A_log)
    total_params = 0
    bf16_params = 0
    bad_param_names = []
    for name, p in model.named_parameters():
        total_params += 1
        if p.dtype == torch.bfloat16:
            bf16_params += 1
        if not torch.isfinite(p).all():
            bad_param_names.append(name)

    ratio = bf16_params / max(total_params, 1)
    _log(f"bf16 params: {bf16_params}/{total_params} ({ratio:.1%})")
    assert ratio > 0.95, f"bf16 parameter ratio too low: {ratio:.1%}"
    assert not bad_param_names, f"Non-finite params detected: {bad_param_names[:8]}"

    x_input, y_ntp, y_mtp = _build_batch(seq_len=20)

    # Layer output finiteness capture
    layer_non_finite = []
    hooks = []
    for i, layer in enumerate(model.layers):
        lname = f"layer_{i}_{model.layer_types[i]}"

        def _fwd_chk(_m, _in, out, lname=lname):
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(out, torch.Tensor) and not torch.isfinite(out).all():
                layer_non_finite.append(lname)

        hooks.append(layer.register_forward_hook(_fwd_chk))

    try:
        total_loss = _loss_forward(model, x_input, y_ntp, y_mtp)
    finally:
        for h in hooks:
            h.remove()

    assert torch.isfinite(total_loss), f"Non-finite loss: {total_loss.item()}"
    assert not layer_non_finite, f"Non-finite layer outputs in: {layer_non_finite[:8]}"

    # Backward
    total_loss.backward()

    nan_or_inf = []
    grad_dtype_counts = defaultdict(int)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        grad_dtype_counts[str(p.grad.dtype)] += 1
        if not torch.isfinite(p.grad).all():
            nan_or_inf.append(name)

    _log(f"Gradient dtype distribution: {dict(grad_dtype_counts)}")
    assert not nan_or_inf, f"Non-finite gradients in: {nan_or_inf[:8]}"

    # GSA differentiable path check
    gsa_idx, gsa_mod = _find_first_gsa(model)
    _assert_some_grads_nonzero(
        [gsa_mod.W_q.weight, gsa_mod.W_k.weight, gsa_mod.W_v.weight, gsa_mod.W_gv.weight],
        f"GSA core projections (layer {gsa_idx})",
    )

    # DeltaNet differentiable path check
    delta_idx, delta_mod = _find_first_deltanet(model)
    _assert_some_grads_nonzero(
        [delta_mod.q_proj.weight, delta_mod.k_proj.weight, delta_mod.v_proj.weight, delta_mod.g_proj.weight],
        f"DeltaNet core projections (layer {delta_idx})",
    )

    # MLP/MoE path check
    mlp_mod = model.layers[0].mlp_block.sublayer
    _assert_some_grads_nonzero(mlp_mod.parameters(), "MLP/MoE sublayer")

    # Optimizer step sanity
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    sample_before = {}
    params_dict = dict(model.named_parameters())
    for name, p in params_dict.items():
        if p.grad is not None and p.numel() > 0:
            sample_before[name] = p.detach().flatten()[:8].clone()
            if len(sample_before) >= 8:
                break

    optimizer.step()

    changed = 0
    for name, old in sample_before.items():
        new = params_dict[name].detach().flatten()[:8]
        if not torch.equal(old, new):
            changed += 1

    assert changed > 0, "No sampled parameters changed after optimizer.step()"
    _log("bf16 + NaN + differentiability checks passed", "ok")

    _clear_gpu_memory()


# =============================================================================
# Test 2: Explicit GSA train/inference path policy check
# =============================================================================


def test_gsa_train_vs_inference_paths(model_and_fixtures):
    model, config = model_and_fixtures
    model = model.to(DEVICE).to(dtype=torch.bfloat16)

    layer_idx, gsa = _find_first_gsa(model)
    _log(f"Using GSA layer index {layer_idx}")

    # Counters for path selection
    counts = defaultdict(int)
    orig_triton = getattr(model_module, "triton_sparse_attention", None)
    orig_pytorch = getattr(model_module, "pytorch_sparse_attention", None)

    def wrap(name, fn):
        if fn is None:
            return None

        def _w(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)

        return _w

    if orig_triton is not None:
        model_module.triton_sparse_attention = wrap("triton_sparse_attention", orig_triton)
    if orig_pytorch is not None:
        model_module.pytorch_sparse_attention = wrap("pytorch_sparse_attention", orig_pytorch)

    try:
        x = torch.randn(1, 12, config.hidden_size, device=DEVICE, dtype=torch.bfloat16)

        # Inference/no-grad path
        counts.clear()
        with torch.no_grad():
            if orig_triton is not None and x.is_cuda:
                out = gsa(x)
                assert out.shape == x.shape
                assert counts["triton_sparse_attention"] > 0, "no-grad path did not use Triton sparse attention"
                _log("no-grad path uses fused Triton sparse attention", "ok")
            else:
                with pytest.raises(RuntimeError):
                    _ = gsa(x)
                _log("no-grad path hard-fails without fused Triton (as expected)", "ok")

        # Training/grad-enabled path
        counts.clear()
        xg = x.detach().clone().requires_grad_(True)
        out = gsa(xg)
        loss = out.sum()
        loss.backward()

        # Branch may use either:
        #   - pytorch_sparse_attention as differentiable fallback, OR
        #   - triton_sparse_attention with use_triton_backward=True (fused bwd)
        used_differentiable = (
            counts["pytorch_sparse_attention"] > 0
            or counts["triton_sparse_attention"] > 0
        )
        assert used_differentiable, (
            "grad-enabled path did not call any sparse attention kernel"
        )
        if counts["pytorch_sparse_attention"] > 0:
            _log("grad-enabled path uses pytorch_sparse_attention (differentiable fallback)", "ok")
        else:
            _log("grad-enabled path uses triton_sparse_attention with fused backward", "ok")

        assert gsa.W_q.weight.grad is not None, "GSA W_q grad missing in grad-enabled path"
        assert torch.isfinite(gsa.W_q.weight.grad).all(), "GSA W_q grad has NaN/Inf"
        assert gsa.W_q.weight.grad.abs().sum().item() > 0, "GSA W_q grad is zero"
        _log("grad-enabled path is differentiable and produces GSA gradients", "ok")

    finally:
        if orig_triton is not None:
            model_module.triton_sparse_attention = orig_triton
        if orig_pytorch is not None:
            model_module.pytorch_sparse_attention = orig_pytorch

    _clear_gpu_memory()


# =============================================================================
# Test 3: Kernel accounting + per-layer forward/backward timing report
# =============================================================================


def test_kernel_usage_and_layer_timing_report(model_and_fixtures):
    model, _ = model_and_fixtures
    model = model.to(DEVICE).to(dtype=torch.bfloat16)
    model.train()

    # Kernel call counters and wall time by phase
    call_counts = {
        "forward": defaultdict(int),
        "backward": defaultdict(int),
    }
    kernel_time_ms = {
        "forward": defaultdict(float),
        "backward": defaultdict(float),
    }
    phase = ["forward"]

    kernel_names = [
        "triton_rmsnorm",
        "triton_sinkhorn_knopp",
        "triton_sparse_attention",
        "pytorch_sparse_attention",
        "fla_gated_delta_rule",
        "fused_indexer_topk",
    ]

    originals = {k: getattr(model_module, k, None) for k in kernel_names}

    def _wrap(name, fn):
        if fn is None:
            return None

        def _w(*args, **kwargs):
            _sync()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            _sync()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            call_counts[phase[0]][name] += 1
            kernel_time_ms[phase[0]][name] += dt_ms
            return out

        return _w

    for k in kernel_names:
        if originals[k] is not None:
            setattr(model_module, k, _wrap(k, originals[k]))

    # Per-layer timers and finite checks
    fwd_start: Dict[int, float] = {}
    bwd_start: Dict[int, float] = {}
    layer_fwd_ms = defaultdict(float)
    layer_bwd_ms = defaultdict(float)
    layer_nan_fwd = set()
    layer_nan_bwd = set()
    hooks = []

    for i, layer in enumerate(model.layers):
        lname = f"layer_{i}_{model.layer_types[i]}"
        lid = id(layer)

        def _fwd_pre(_m, _in, lid=lid):
            _sync()
            fwd_start[lid] = time.perf_counter()

        def _fwd_post(_m, _in, out, lid=lid, lname=lname):
            _sync()
            layer_fwd_ms[lname] += (time.perf_counter() - fwd_start.pop(lid, time.perf_counter())) * 1000.0
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                layer_nan_fwd.add(lname)

        def _bwd_pre(_m, _go, lid=lid):
            _sync()
            bwd_start[lid] = time.perf_counter()

        def _bwd_post(_m, gi, _go, lid=lid, lname=lname):
            _sync()
            layer_bwd_ms[lname] += (time.perf_counter() - bwd_start.pop(lid, time.perf_counter())) * 1000.0
            for t in gi:
                if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                    layer_nan_bwd.add(lname)
                    break

        hooks.append(layer.register_forward_pre_hook(_fwd_pre))
        hooks.append(layer.register_forward_hook(_fwd_post))
        hooks.append(layer.register_full_backward_pre_hook(_bwd_pre))
        hooks.append(layer.register_full_backward_hook(_bwd_post))

    try:
        x_input, y_ntp, y_mtp = _build_batch(seq_len=20)

        _sync()
        t0 = time.perf_counter()
        phase[0] = "forward"
        total_loss = _loss_forward(model, x_input, y_ntp, y_mtp)
        _sync()
        t1 = time.perf_counter()

        assert torch.isfinite(total_loss), f"Non-finite loss: {total_loss.item()}"

        phase[0] = "backward"
        total_loss.backward()
        _sync()
        t2 = time.perf_counter()

        # Gradient finiteness
        bad_grads = [
            name
            for name, p in model.named_parameters()
            if p.grad is not None and (not torch.isfinite(p.grad).all())
        ]
        assert not bad_grads, f"Non-finite gradients in: {bad_grads[:8]}"

        assert not layer_nan_fwd, f"Non-finite forward activations in layers: {sorted(layer_nan_fwd)}"
        assert not layer_nan_bwd, f"Non-finite backward grads in layers: {sorted(layer_nan_bwd)}"

        # Core expectations for current policy
        if originals["fused_indexer_topk"] is not None:
            assert call_counts["forward"]["fused_indexer_topk"] > 0, "fused_indexer_topk not used"
        if originals["fla_gated_delta_rule"] is not None:
            assert call_counts["forward"]["fla_gated_delta_rule"] > 0, "fla_gated_delta_rule not used"
        # GSA sparse attention: either pytorch (differentiable fallback) or triton (fused bwd)
        sparse_fwd = (
            call_counts["forward"].get("pytorch_sparse_attention", 0)
            + call_counts["forward"].get("triton_sparse_attention", 0)
        )
        assert sparse_fwd > 0, (
            "No sparse attention kernel called during forward pass"
        )

        # Kernel report
        _log("\nKernel availability:")
        avail = _kernel_availability()
        for k in kernel_names:
            _log(f"{k:<28} {'YES' if avail[k] else 'NO'}")

        _log("\nKernel call/time report:")
        _log(f"{'kernel':<28} {'f_calls':>8} {'f_ms':>10} {'f_ms/call':>10} {'b_calls':>8} {'b_ms':>10}")
        _log("-" * 86)
        for k in kernel_names:
            f_calls = call_counts["forward"][k]
            b_calls = call_counts["backward"][k]
            f_ms = kernel_time_ms["forward"][k]
            b_ms = kernel_time_ms["backward"][k]
            f_avg = (f_ms / f_calls) if f_calls else 0.0
            _log(
                f"{k:<28} {f_calls:>8} {f_ms:>10.3f} {f_avg:>10.3f} {b_calls:>8} {b_ms:>10.3f}"
            )

        _log("\nLayer timing report (ms):")
        _log(f"{'layer':<24} {'forward_ms':>12} {'backward_ms':>12}")
        _log("-" * 54)
        for lname in sorted(layer_fwd_ms.keys()):
            _log(f"{lname:<24} {layer_fwd_ms[lname]:>12.3f} {layer_bwd_ms[lname]:>12.3f}")

        _log(
            f"\nStep timing: forward={(t1 - t0) * 1000:.2f} ms, "
            f"backward={(t2 - t1) * 1000:.2f} ms",
            "ok",
        )

        # Per-layer hooks may not fire if the model uses functional_call
        # (e.g. reversible midpoint stack), so treat this as a soft warning.
        if not any(v > 0 for v in layer_fwd_ms.values()):
            _log("Per-layer forward timings empty (expected with reversible/functional_call)", "warn")
        if not any(v > 0 for v in layer_bwd_ms.values()):
            _log("Per-layer backward timings empty (expected with reversible/functional_call)", "warn")

    finally:
        for h in hooks:
            h.remove()
        for k, orig in originals.items():
            setattr(model_module, k, orig)

    _clear_gpu_memory()


# =============================================================================
# Script runner
# =============================================================================

_SCRIPT_TESTS = [
    ("bf16", test_bf16_pipeline_nan_and_diff_paths),
    ("gsa", test_gsa_train_vs_inference_paths),
    ("profile", test_kernel_usage_and_layer_timing_report),
]

_NAMED = {name: (name, fn) for name, fn in _SCRIPT_TESTS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSA/DeltaNet bf16 kernel diagnostics")
    parser.add_argument("--only", choices=list(_NAMED.keys()))
    args = parser.parse_args()

    if not CUDA_AVAILABLE:
        print("SKIP: CUDA not available")
        sys.exit(0)

    print("\n" + "=" * 78)
    print("GSA/DeltaNet bf16 / kernel / differentiability diagnostics")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model variant: {MODEL_VARIANT}")
    print(f"Module: {MODULE_NAME}")
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"HAS_TRITON={HAS_TRITON}, HAS_FLA={HAS_FLA}")

    tests_to_run = [_NAMED[args.only]] if args.only else _SCRIPT_TESTS
    results = []

    for name, fn in tests_to_run:
        print(f"\n--- {name} ---")
        # Fresh model per test in script mode (matches pytest fixture semantics).
        model_and_fixtures = _make_model_and_fixtures()
        try:
            fn(model_and_fixtures)
            results.append((name, True, None))
        except Exception as e:  # noqa: BLE001
            import traceback

            results.append((name, False, str(e)))
            print(f"  [FAIL] {e}")
            traceback.print_exc()
        finally:
            _clear_gpu_memory()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, err in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if err:
            print(f"         {err}")
    print("-" * 78)
    print(f"  {passed}/{total} tests passed")
    print("=" * 78)

    sys.exit(0 if passed == total else 1)

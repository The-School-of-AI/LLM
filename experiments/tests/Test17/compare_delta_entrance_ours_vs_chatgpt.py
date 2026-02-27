import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Tuple

import torch


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _dtype_from_name(name: str) -> torch.dtype:
    m = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in m:
        raise ValueError(f"Unsupported dtype: {name}")
    return m[name]


def _make_mask(kind: str, b: int, t: int, device: torch.device) -> torch.Tensor:
    if kind == "ones":
        return torch.ones((b, t), device=device, dtype=torch.uint8)
    if kind == "zeros":
        return torch.zeros((b, t), device=device, dtype=torch.uint8)
    if kind == "random":
        return (torch.rand((b, t), device=device) > 0.25).to(torch.uint8)
    raise ValueError(f"Unknown mask kind: {kind}")


def _make_qkv(
    b: int,
    t: int,
    c: int,
    device: torch.device,
    dtype: torch.dtype,
    scale: float,
    non_contiguous: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if non_contiguous:
        q = (torch.randn((b, t, c * 2), device=device, dtype=dtype) * scale)[..., ::2]
        k = (torch.randn((b, t, c * 2), device=device, dtype=dtype) * scale)[..., ::2]
        v = (torch.randn((b, t, c * 2), device=device, dtype=dtype) * scale)[..., ::2]
        return q, k, v
    q = torch.randn((b, t, c), device=device, dtype=dtype) * scale
    k = torch.randn((b, t, c), device=device, dtype=dtype) * scale
    v = torch.randn((b, t, c), device=device, dtype=dtype) * scale
    return q, k, v


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    d = (a.float() - b.float()).abs()
    rel = d / (b.float().abs() + 1e-6)
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "mean_rel": float(rel.mean().item()),
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return float("nan")
    return float(torch.dot(af, bf).item() / denom.item())


def _run_one_kernel(
    mod,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    bq: torch.Tensor,
    bk: torch.Tensor,
    bv: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> Dict[str, object]:
    with torch.no_grad():
        qo_ref, ko_ref, vo_ref = mod.pytorch_unfused_exact(
            q, k, v, wq, wk, wv, bq, bk, bv, cos_half, sin_half, mask, eps=eps
        )
        qo_tri, ko_tri, vo_tri = mod.fused_delta_entrance(
            q, k, v, wq, wk, wv, bq, bk, bv, cos_half, sin_half, mask, eps=eps
        )

    forward = {
        "q": _tensor_metrics(qo_tri, qo_ref),
        "k": _tensor_metrics(ko_tri, ko_ref),
        "v": _tensor_metrics(vo_tri, vo_ref),
    }

    # Backward parity
    q1 = q.detach().clone().requires_grad_(True)
    k1 = k.detach().clone().requires_grad_(True)
    v1 = v.detach().clone().requires_grad_(True)
    wq1 = wq.detach().clone().requires_grad_(True)
    wk1 = wk.detach().clone().requires_grad_(True)
    wv1 = wv.detach().clone().requires_grad_(True)
    bq1 = bq.detach().clone().requires_grad_(True)
    bk1 = bk.detach().clone().requires_grad_(True)
    bv1 = bv.detach().clone().requires_grad_(True)

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    wq2 = wq.detach().clone().requires_grad_(True)
    wk2 = wk.detach().clone().requires_grad_(True)
    wv2 = wv.detach().clone().requires_grad_(True)
    bq2 = bq.detach().clone().requires_grad_(True)
    bk2 = bk.detach().clone().requires_grad_(True)
    bv2 = bv.detach().clone().requires_grad_(True)

    qo_ref, ko_ref, vo_ref = mod.pytorch_unfused_exact(
        q1, k1, v1, wq1, wk1, wv1, bq1, bk1, bv1, cos_half, sin_half, mask, eps=eps
    )
    loss_ref = qo_ref.float().square().mean() + ko_ref.float().square().mean() + vo_ref.float().square().mean()
    loss_ref.backward()

    qo_tri, ko_tri, vo_tri = mod.fused_delta_entrance(
        q2, k2, v2, wq2, wk2, wv2, bq2, bk2, bv2, cos_half, sin_half, mask, eps=eps
    )
    loss_tri = qo_tri.float().square().mean() + ko_tri.float().square().mean() + vo_tri.float().square().mean()
    loss_tri.backward()

    grads = {
        "dq": (q2.grad, q1.grad),
        "dk": (k2.grad, k1.grad),
        "dv": (v2.grad, v1.grad),
        "dwq": (wq2.grad, wq1.grad),
        "dwk": (wk2.grad, wk1.grad),
        "dwv": (wv2.grad, wv1.grad),
        "dbq": (bq2.grad, bq1.grad),
        "dbk": (bk2.grad, bk1.grad),
        "dbv": (bv2.grad, bv1.grad),
    }
    backward = {}
    for name, (g_tri, g_ref) in grads.items():
        backward[name] = {
            **_tensor_metrics(g_tri, g_ref),
            "cosine": _cosine(g_tri, g_ref),
        }

    return {"forward": forward, "backward": backward}


def _run_cross_kernel_diff(
    ours_mod,
    chatgpt_mod,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    bq: torch.Tensor,
    bk: torch.Tensor,
    bv: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> Dict[str, object]:
    with torch.no_grad():
        qo_ours, ko_ours, vo_ours = ours_mod.fused_delta_entrance(
            q, k, v, wq, wk, wv, bq, bk, bv, cos_half, sin_half, mask, eps=eps
        )
        qo_gpt, ko_gpt, vo_gpt = chatgpt_mod.fused_delta_entrance(
            q, k, v, wq, wk, wv, bq, bk, bv, cos_half, sin_half, mask, eps=eps
        )
    return {
        "forward_q": _tensor_metrics(qo_ours, qo_gpt),
        "forward_k": _tensor_metrics(ko_ours, ko_gpt),
        "forward_v": _tensor_metrics(vo_ours, vo_gpt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run Triton parity tests.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = _dtype_from_name(args.dtype)

    root = Path(__file__).resolve().parent
    ours_path = root / "code" / "src" / "kernels" / "triton_delta_entrance.py"
    gpt_path = root / "code" / "src" / "kernels" / "triton_delta_entrance_chatgpt.py"
    ours_mod = _load_module(ours_path, "delta_entrance_ours")
    gpt_mod = _load_module(gpt_path, "delta_entrance_chatgpt")

    b = args.batch_size
    t = args.seq_len
    h = args.num_heads
    d = args.head_dim
    c = h * d

    # Compact RoPE tables by contract: (T, D//2).
    cos_half = torch.randn((t, d // 2), device=device, dtype=dtype)
    sin_half = torch.randn((t, d // 2), device=device, dtype=dtype)

    wq = torch.randn((c, 1, 4), device=device, dtype=dtype)
    wk = torch.randn((c, 1, 4), device=device, dtype=dtype)
    wv = torch.randn((c, 1, 4), device=device, dtype=dtype)
    bq = torch.randn((c,), device=device, dtype=dtype)
    bk = torch.randn((c,), device=device, dtype=dtype)
    bv = torch.randn((c,), device=device, dtype=dtype)

    cases = [
        {"name": "all_ones_mask", "mask": "ones", "scale": 1.0, "non_contiguous": False},
        {"name": "random_mask", "mask": "random", "scale": 1.0, "non_contiguous": False},
        {"name": "all_zero_mask", "mask": "zeros", "scale": 1.0, "non_contiguous": False},
        {"name": "tiny_inputs_1e-4", "mask": "random", "scale": 1e-4, "non_contiguous": False},
        {"name": "non_contiguous_views", "mask": "random", "scale": 1.0, "non_contiguous": True},
    ]

    out = {
        "config": {
            "batch_size": b,
            "seq_len": t,
            "num_heads": h,
            "head_dim": d,
            "dtype": args.dtype,
            "eps": args.eps,
            "seed": args.seed,
        },
        "cases": [],
    }

    for case in cases:
        q, k, v = _make_qkv(b, t, c, device, dtype, case["scale"], case["non_contiguous"])
        mask = _make_mask(case["mask"], b, t, device)
        ours = _run_one_kernel(
            ours_mod,
            q, k, v, wq, wk, wv, bq, bk, bv, cos_half, sin_half, mask, args.eps
        )
        gpt = _run_one_kernel(
            gpt_mod,
            q, k, v, wq, wk, wv, bq, bk, bv, cos_half, sin_half, mask, args.eps
        )
        cross = _run_cross_kernel_diff(
            ours_mod, gpt_mod, q, k, v, wq, wk, wv, bq, bk, bv, cos_half, sin_half, mask, args.eps
        )
        out["cases"].append({
            "name": case["name"],
            "mask_kind": case["mask"],
            "scale": case["scale"],
            "non_contiguous": case["non_contiguous"],
            "ours_vs_reference": ours,
            "chatgpt_vs_reference": gpt,
            "ours_vs_chatgpt": cross,
        })

    text = json.dumps(out, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
        print(f"Wrote: {args.json_out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

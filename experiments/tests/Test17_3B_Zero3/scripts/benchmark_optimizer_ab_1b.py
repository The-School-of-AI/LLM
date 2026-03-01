#!/usr/bin/env python3
"""
Optimizer A/B benchmark for Test17 1B model.

Runs two training slices sequentially on identical synthetic-token batches:
1) Baseline AdamW
2) NAMO-D

Each run:
- same model config
- same batch stream
- same number of steps
- reports final/avg loss, throughput, and peak GPU memory

Between runs, GPU memory is explicitly cleared so run #2 is not affected by run #1.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml


@dataclass
class RunResult:
    name: str
    steps: int
    batch_size: int
    seq_len: int
    dtype: str
    avg_loss: float
    final_loss: float
    avg_last_50_loss: float
    tokens_per_s: float
    avg_step_ms: float
    peak_alloc_gb: float
    peak_reserved_gb: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1B optimizer A/B benchmark (AdamW vs NAMO-D)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "optimizer_ab_1b_500steps.yaml",
    )
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def load_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()


def build_model(device: torch.device, dtype: torch.dtype):
    repo_root = Path(__file__).resolve().parents[1]
    code_root = repo_root / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from src.models.recurrence_model_1b import Model1B, ModelConfig

    cfg = ModelConfig()
    model = Model1B(cfg, embedding_type="standard", bpe_vocab=None, pf_codec=None)
    model = model.to(device=device, dtype=dtype)
    model.train()
    return model, cfg


def split_params_for_muon(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    nodecay_keys = ("lm_head", "wte", "wpe", "embedding", "norm", "bias")
    decay_params: List[torch.nn.Parameter] = []
    nodecay_params: List[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or any(k in n.lower() for k in nodecay_keys):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    muon_params: List[torch.nn.Parameter] = []
    adamw_params: List[torch.nn.Parameter] = list(nodecay_params)
    for p in decay_params:
        if p.dim() in (2, 4):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def build_optimizer(name: str, model: torch.nn.Module, cfg: Dict[str, Any]):
    lr = float(cfg["learning_rate"])
    wd = float(cfg["weight_decay"])

    if name == "adamw":
        groups = [
            {"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": wd},
        ]
        fused_ok = torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        extra = {"fused": True} if fused_ok else {}
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, **extra)

    if name == "namo_d":
        try:
            from namo import NAMO_D
        except Exception:
            repo_root = Path(__file__).resolve().parents[1]
            local_namo_src = repo_root / "third_party" / "namo" / "src"
            if local_namo_src.exists():
                if str(local_namo_src) not in sys.path:
                    sys.path.insert(0, str(local_namo_src))
                try:
                    from namo import NAMO_D  # type: ignore[no-redef]
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(
                        f"Failed to import NAMO_D from local checkout: {local_namo_src}"
                    ) from e
            else:  # pragma: no cover
                raise RuntimeError(
                    "Failed to import NAMO_D. Expected local checkout at Test17/third_party/namo/src."
                )

        muon_params, adamw_params = split_params_for_muon(model)
        return NAMO_D(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=0.95,
            mu2=0.99,
            adamnorm_eps=1e-8,
            nesterov=True,
            ns_steps=5,
            use_exact_svd=False,
            scale_coeff=float(cfg.get("namo_d_scale_coeff", 0.2)),
            col_state_clamp_c=float(cfg.get("namo_d_col_state_clamp_c", 0.7)),
            adamw_betas=(0.9, 0.95),
            adamw_eps=1e-8,
            adamw_wd=0.0,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def make_batches(vocab_size: int, steps: int, batch_size: int, seq_len: int, seed: int) -> List[torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return [
        torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, dtype=torch.long)
        for _ in range(steps)
    ]


def train_slice(
    name: str,
    model: torch.nn.Module,
    optimizer,
    batches_cpu: List[torch.Tensor],
    device: torch.device,
    amp_dtype: torch.dtype,
    log_interval: int = 50,
) -> RunResult:
    losses: List[float] = []
    step_ms: List[float] = []
    total_tokens = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t_run0 = time.perf_counter()

    for i, batch_cpu in enumerate(batches_cpu, start=1):
        x = batch_cpu.to(device, non_blocking=True)
        total_tokens += x.numel()

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits_ntp, logits_mtp, aux_loss = model(
                x,
                attention_mask=None,
                return_memory=False,
                return_loss=True,
                return_hidden=False,
            )

            ntp_logits = logits_ntp[:, :-1, :].contiguous()
            ntp_targets = x[:, 1:].contiguous()
            loss_ntp = torch.nn.functional.cross_entropy(
                ntp_logits.view(-1, ntp_logits.size(-1)),
                ntp_targets.view(-1),
            )

            if logits_mtp is not None and x.size(1) > 2:
                mtp_logits = logits_mtp[:, :-2, :].contiguous()
                mtp_targets = x[:, 2:].contiguous()
                loss_mtp = torch.nn.functional.cross_entropy(
                    mtp_logits.view(-1, mtp_logits.size(-1)),
                    mtp_targets.view(-1),
                )
            else:
                loss_mtp = loss_ntp.new_zeros(())

            aux = aux_loss.mean() if (aux_loss is not None and isinstance(aux_loss, torch.Tensor)) else loss_ntp.new_zeros(())
            loss = loss_ntp + 0.3 * loss_mtp + aux

        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ms = (time.perf_counter() - t0) * 1000.0

        lv = float(loss.detach().float().item())
        losses.append(lv)
        step_ms.append(ms)

        if (i % log_interval == 0) or (i == len(batches_cpu)):
            print(f"[{name}] step={i}/{len(batches_cpu)} loss={lv:.4f} step_ms={ms:.2f}")

    total_s = time.perf_counter() - t_run0
    peak_alloc = 0.0
    peak_reserved = 0.0
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    tail = losses[-50:] if len(losses) >= 50 else losses
    return RunResult(
        name=name,
        steps=len(batches_cpu),
        batch_size=batches_cpu[0].size(0),
        seq_len=batches_cpu[0].size(1),
        dtype=str(amp_dtype).replace("torch.", ""),
        avg_loss=float(sum(losses) / len(losses)),
        final_loss=float(losses[-1]),
        avg_last_50_loss=float(sum(tail) / len(tail)),
        tokens_per_s=float(total_tokens / total_s),
        avg_step_ms=float(sum(step_ms) / len(step_ms)),
        peak_alloc_gb=float(peak_alloc),
        peak_reserved_gb=float(peak_reserved),
    )


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    bench = cfg["benchmark"]

    set_seed(int(bench.get("seed", 42)))
    device = torch.device(bench.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_str = str(bench.get("dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    steps = int(bench["steps"])
    bs = int(bench["batch_size"])
    seqlen = int(bench["seq_len"])
    lr = float(bench["learning_rate"])
    wd = float(bench["weight_decay"])

    print("=" * 80)
    print("1B Optimizer A/B Benchmark (AdamW vs NAMO-D)")
    print("=" * 80)
    print(f"device={device} dtype={dtype_str} steps={steps} bs={bs} seq={seqlen} lr={lr} wd={wd}")

    # Build deterministic synthetic batches once and reuse for both optimizers.
    # This keeps data identical across runs.
    vocab_size = 131072
    batches_cpu = make_batches(vocab_size=vocab_size, steps=steps, batch_size=bs, seq_len=seqlen, seed=int(bench.get("data_seed", 1234)))

    results: List[RunResult] = []
    for optim_name in ["adamw", "namo_d"]:
        clear_cuda()
        gc.collect()

        model, _ = build_model(device, amp_dtype)
        optim_cfg = {
            "learning_rate": lr,
            "weight_decay": wd,
            "namo_d_col_state_clamp_c": float(bench.get("namo_d_col_state_clamp_c", 0.7)),
            "namo_d_scale_coeff": float(bench.get("namo_d_scale_coeff", 0.2)),
        }
        optimizer = build_optimizer(optim_name, model, optim_cfg)

        out = train_slice(
            name=optim_name,
            model=model,
            optimizer=optimizer,
            batches_cpu=batches_cpu,
            device=device,
            amp_dtype=amp_dtype,
            log_interval=int(bench.get("log_interval", 50)),
        )
        results.append(out)

        del optimizer
        del model
        clear_cuda()
        gc.collect()

    base = results[0]
    test = results[1]
    comparison = {
        "loss_final_delta": test.final_loss - base.final_loss,
        "loss_last50_delta": test.avg_last_50_loss - base.avg_last_50_loss,
        "throughput_delta_toks": test.tokens_per_s - base.tokens_per_s,
        "throughput_delta_pct": ((test.tokens_per_s / base.tokens_per_s) - 1.0) * 100.0 if base.tokens_per_s > 0 else None,
        "peak_reserved_delta_gb": test.peak_reserved_gb - base.peak_reserved_gb,
        "peak_alloc_delta_gb": test.peak_alloc_gb - base.peak_alloc_gb,
    }

    payload = {
        "config_path": str(args.config),
        "benchmark": bench,
        "runs": [asdict(r) for r in results],
        "comparison_namo_d_vs_adamw": comparison,
    }

    print("\n-- Summary --")
    for r in results:
        print(
            f"{r.name:7s} | final_loss={r.final_loss:.4f} | last50={r.avg_last_50_loss:.4f} | "
            f"toks/s={r.tokens_per_s:.2f} | avg_step_ms={r.avg_step_ms:.2f} | "
            f"peak_alloc_gb={r.peak_alloc_gb:.2f} | peak_reserved_gb={r.peak_reserved_gb:.2f}"
        )
    print(
        f"delta(namo_d-adamw) | final_loss={comparison['loss_final_delta']:+.4f} | "
        f"last50={comparison['loss_last50_delta']:+.4f} | "
        f"throughput={comparison['throughput_delta_toks']:+.2f} tok/s ({comparison['throughput_delta_pct']:+.2f}%) | "
        f"peak_reserved={comparison['peak_reserved_delta_gb']:+.2f} GB"
    )

    out_path = args.json_out or (Path(__file__).resolve().parents[1] / "results" / "optimizer_ab_1b_500steps.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

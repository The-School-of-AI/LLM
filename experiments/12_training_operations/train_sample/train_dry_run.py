#!/usr/bin/env python3
"""
Dry-run training script using the miniaturized 70B architecture.

Exercises the full architecture (DeltaNet, GSA, MoE, mHC, MTP, memory
stream recurrence, reversible midpoint) at ~20M params for fast testing
of the P12 observability pipeline.

Usage:
    python train_dry_run.py

No tokenizer or dataset required — uses synthetic random data.
"""

import gc
import os
import sys
import time

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure components are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# P12 Observability
from components import TrainingOps
from config_mini import apply_mini_config
from recurrence_model_70b import Model70B, ModelConfig

# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _load_env_file_if_present(path: str) -> bool:
    """Load KEY=VALUE pairs from a simple env file if it exists."""
    if not os.path.isfile(path):
        return False

    loaded_any = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded_any = True
    return loaded_any


# ---------------------------------------------------------------------------
# Synthetic dataset (no tokenizer needed)
# ---------------------------------------------------------------------------


class SyntheticTokenDataset(Dataset):
    """Generates random token sequences for dry-run testing."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # +2 for NTP and MTP targets
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 2,))
        return {"input_ids": tokens}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(model, loader, device, num_steps=50, ops=None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    data_iter = iter(loader)
    tokens_processed_total = 0

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        x = input_ids[:, :-2]
        y_ntp = input_ids[:, 1:-1]
        y_mtp = input_ids[:, 2:]

        t0 = time.time()

        logits_ntp, logits_mtp, aux_loss = model(
            x,
            next_token_ids=y_ntp,
            return_loss=True,
            return_memory=False,
            prev_memory_stream=None,
        )

        V = logits_ntp.size(-1)
        loss_ntp = criterion(logits_ntp.reshape(-1, V), y_ntp.reshape(-1))
        loss_mtp = (
            criterion(logits_mtp.reshape(-1, V), y_mtp.reshape(-1))
            if logits_mtp is not None
            else torch.tensor(0.0)
        )
        loss = loss_ntp + 0.3 * loss_mtp + aux_loss

        # Lightweight periodic validation probe on synthetic data.
        loss_val = None
        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_tokens = torch.randint(
                    0, V, (x.size(0), x.size(1) + 2), device=device
                )
                x_val = val_tokens[:, :-2]
                y_val = val_tokens[:, 1:-1]
                logits_val, _, _ = model(
                    x_val,
                    next_token_ids=y_val,
                    return_loss=True,
                    return_memory=False,
                    prev_memory_stream=None,
                )
                loss_val = criterion(
                    logits_val.reshape(-1, V), y_val.reshape(-1)
                ).item()
            model.train()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dt = (time.time() - t0) * 1000.0
        tok_sec = x.numel() / max(dt / 1000.0, 1e-9)
        batch_sec = 1000.0 / max(dt, 1e-9)
        tokens_processed_total += int(x.numel())
        cpu_idle_percent = float(
            getattr(psutil.cpu_times_percent(interval=None), "idle", 0.0)
        )

        print(
            f"step {step:3d} | loss {loss.item():.4f} | "
            f"ntp {loss_ntp.item():.4f} | mtp {loss_mtp.item():.4f} | "
            f"aux {aux_loss.item():.4f} | {dt:.0f}ms | {tok_sec:.0f} tok/s"
        )

        if ops is not None:
            metrics = {
                "loss": loss.item(),
                "loss/train": loss.item(),
                "loss/train_t_plus_1": loss_ntp.item(),
                "loss/train_t_plus_2": loss_mtp.item(),
                "loss/router_moe": aux_loss.item(),
                "loss/router_null": 0.0,
                "lr": optimizer.param_groups[0]["lr"],
                "throughput/tokens_per_sec": tok_sec,
                "throughput/batches_per_sec": batch_sec,
                "tokens/processed_total": float(tokens_processed_total),
                "router/null_ratio": 0.5,
                "cpu/idle_percent": cpu_idle_percent,
                "step_time_ms": dt,
            }
            if loss_val is not None:
                metrics["loss/val"] = loss_val
            ops.log_step(step=step, metrics=metrics)

            # Array metrics examples
            k = min(8, input_ids.size(-1))
            top_vals, top_idx = torch.topk(
                torch.bincount(input_ids.reshape(-1), minlength=V).float(), k=k
            )
            ops.log_metric_array(
                step=step,
                metric="moe/favorite_tokens_topk",
                keys=[str(int(i.item())) for i in top_idx],
                values=[float(v.item()) for v in top_vals],
                unit="count",
                tags={"source": "synthetic_batch"},
            )

            # Approximate routing distribution proxy for dry-run observability.
            n_bins = 8
            x_norm = torch.softmax(
                torch.arange(n_bins, device=x.device, dtype=torch.float32), dim=0
            )
            ops.log_metric_array(
                step=step,
                metric="moe/routing_dist_mean",
                keys=[f"expert_{i}" for i in range(n_bins)],
                values=[float(v.item()) for v in x_norm],
                unit="ratio",
            )

            fft_src = x[0].float()
            fft = torch.fft.rfft(fft_src)
            energy = fft.real * fft.real + fft.imag * fft.imag
            max_buckets = min(8, energy.numel())
            ops.log_metric_array(
                step=step,
                metric="moe/fourier_bucket_energy",
                keys=[f"bucket_{i}" for i in range(max_buckets)],
                values=[float(energy[i].item()) for i in range(max_buckets)],
                unit="energy",
            )

            # GPU utilization is emitted by SystemMetricsCollector (sys.gpu.*),
            # so we intentionally avoid duplicate per-step gpu/utilization logs here.

            # Periodically emit checkpoint events to validate checkpoint pipeline.
            if step > 0 and step % 25 == 0:
                ops.log_checkpoint(
                    step=step,
                    path=f"/tmp/checkpoints/dry_run_step_{step}.pt",
                    s3_key=f"s3://dry-run/{ops.run_id}/step_{step}.pt",
                    loss=float(loss.item()),
                    tag="temporary",
                    duration_s=0.0,
                    size_bytes=0,
                    metadata={"dry_run": True},
                )
                ops.log_event(
                    step=step,
                    event_type="checkpoint_uploaded",
                    message=f"Checkpoint uploaded to s3://dry-run/{ops.run_id}/step_{step}.pt",
                    payload={"step": step},
                )
                ops.log_event(
                    step=step,
                    event_type="checkpoint_benchmarked",
                    message=f"Checkpoint benchmark completed for step {step}",
                    payload={"step": step, "latency_ms": float(dt)},
                )

            if step % 50 == 0:
                ops.log_event(
                    step=step,
                    event_type="sample_generated",
                    message=f"Generated synthetic sample at step {step}",
                    payload={
                        "token_preview": [int(t) for t in input_ids[0, :8].tolist()]
                    },
                )

        del logits_ntp, logits_mtp, x, y_ntp, y_mtp, loss
        if step % 10 == 0:
            gc.collect()

    print("Training complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  DRY RUN — Mini 70B Architecture (~20M params)")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Best-effort env bootstrap for local dry-runs.
    _load_env_file_if_present(os.path.expanduser("~/.p12.env"))
    _load_env_file_if_present(os.path.expanduser("~/temp/training-instance.env"))

    # Build mini config
    config = ModelConfig()
    apply_mini_config(config)

    # Create model with standard embeddings (no tokenizer needed)
    model = Model70B(config, embedding_type="standard")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Synthetic data (high-util defaults for A10 24GB)
    seq_len = int(os.environ.get("DRYRUN_SEQ_LEN", "256"))
    batch_size = int(os.environ.get("DRYRUN_BATCH_SIZE", "16"))
    num_steps = int(os.environ.get("DRYRUN_STEPS", "200"))
    dataset = SyntheticTokenDataset(config.vocab_size, seq_len, num_samples=500)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(
        f"Dry-run workload: batch_size={batch_size}, seq_len={seq_len}, steps={num_steps}"
    )

    # P12 Observability
    run_id = f"dry_run_{int(time.time())}"
    clickhouse_url = (
        os.environ.get("CLICKHOUSE_ENDPOINT")
        or os.environ.get("CLICKHOUSE_HTTPS_ENDPOINT")
        or os.environ.get("CLICKHOUSE_HTTP_ENDPOINT")
    )
    vector_service_name = (
        os.environ.get("VECTOR_SERVICE_NAME", "p12-vector.service").strip() or None
    )
    if clickhouse_url is not None:
        print(f"ClickHouse endpoint: {clickhouse_url}")
    if vector_service_name is not None:
        print(f"Vector service preflight: {vector_service_name}")
    ops = TrainingOps(
        run_id=run_id,
        rank=int(os.environ.get("RANK", 0)),
        clickhouse_url=clickhouse_url,
        default_context={"model": "mini_70b_arch", "test": "dry_run"},
        skip_vector_check=False,
        vector_service_name=vector_service_name,
    )

    ops.log_event(
        step=0,
        event_type="stage_transition",
        message="dry_run_started",
        payload={"stage": "warmup"},
    )

    train(model, loader, device, num_steps=num_steps, ops=ops)
    ops.log_event(
        step=num_steps,
        event_type="stage_transition",
        message="dry_run_completed",
        payload={"stage": "done"},
    )
    ops.shutdown()

    print("\n" + "=" * 60)
    print("  DRY RUN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

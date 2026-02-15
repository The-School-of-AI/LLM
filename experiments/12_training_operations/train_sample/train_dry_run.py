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

import os
import sys
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure components are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recurrence_model_70b import ModelConfig, Model70B
from config_mini import apply_mini_config

# P12 Observability
from components import TrainingOps


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
        loss_mtp = criterion(logits_mtp.reshape(-1, V), y_mtp.reshape(-1)) if logits_mtp is not None else torch.tensor(0.0)
        loss = loss_ntp + 0.3 * loss_mtp + aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dt = (time.time() - t0) * 1000.0
        tok_sec = x.numel() / max(dt / 1000.0, 1e-9)

        print(
            f"step {step:3d} | loss {loss.item():.4f} | "
            f"ntp {loss_ntp.item():.4f} | mtp {loss_mtp.item():.4f} | "
            f"aux {aux_loss.item():.4f} | {dt:.0f}ms | {tok_sec:.0f} tok/s"
        )

        if ops is not None:
            ops.log_step(step=step, metrics={
                "loss": loss.item(),
                "loss_ntp": loss_ntp.item(),
                "loss_mtp": loss_mtp.item(),
                "aux_loss": aux_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "tokens_per_second": tok_sec,
                "step_time_ms": dt,
            })

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
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
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
    num_steps = int(os.environ.get("DRYRUN_STEPS", "100"))
    dataset = SyntheticTokenDataset(config.vocab_size, seq_len, num_samples=500)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dry-run workload: batch_size={batch_size}, seq_len={seq_len}, steps={num_steps}")

    # P12 Observability
    run_id = f"dry_run_{int(time.time())}"
    clickhouse_url = (
        os.environ.get("CLICKHOUSE_ENDPOINT")
        or os.environ.get("CLICKHOUSE_HTTPS_ENDPOINT")
        or os.environ.get("CLICKHOUSE_HTTP_ENDPOINT")
    )
    if clickhouse_url is not None:
        print(f"ClickHouse endpoint: {clickhouse_url}")
    ops = TrainingOps(
        run_id=run_id,
        rank=int(os.environ.get("RANK", 0)),
        clickhouse_url=clickhouse_url,
        default_context={"model": "mini_70b_arch", "test": "dry_run"},
        skip_vector_check=False,
    )

    train(model, loader, device, num_steps=num_steps, ops=ops)
    ops.shutdown()

    print("\n" + "=" * 60)
    print("  DRY RUN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

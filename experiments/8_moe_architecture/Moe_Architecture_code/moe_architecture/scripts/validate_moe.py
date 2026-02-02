#!/usr/bin/env python3
"""
Lightweight MoE validation loop.

Checks:
1) Loss vs baseline (optional)
2) Routing health gates
3) Null routing targets (junk vs signal)

This is a sanity check, not a full evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import get_config
from model.transformer import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="3b_moe", choices=["1b_dense", "3b_moe", "8b_moe", "70b_moe"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--junk-rate", type=float, default=0.2)
    parser.add_argument("--baseline-loss", type=float, default=None)
    parser.add_argument("--loss-tol", type=float, default=0.1, help="Allowed relative loss delta vs baseline")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON report path")
    return parser.parse_args()


def build_batch(vocab_size: int, batch_size: int, seq_len: int, junk_rate: float, junk_id: int = 0):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.rand(batch_size, seq_len) < junk_rate
    tokens[mask] = junk_id
    labels = tokens.clone()
    return tokens, labels


def aggregate_router_info(router_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    junk_rates = []
    signal_rates = []
    healthy_layers = 0
    total_layers = 0
    alerts = []
    for info in router_info:
        total_layers += 1
        if "junk_null_rate" in info:
            junk_rates.append(info["junk_null_rate"])
        if "signal_null_rate" in info:
            signal_rates.append(info["signal_null_rate"])
        health = info.get("health")
        if isinstance(health, dict):
            if health.get("is_healthy", False):
                healthy_layers += 1
            else:
                alerts.extend(health.get("alerts", []))
    return {
        "avg_junk_null_rate": sum(junk_rates) / max(1, len(junk_rates)),
        "avg_signal_null_rate": sum(signal_rates) / max(1, len(signal_rates)),
        "healthy_layers": healthy_layers,
        "total_layers": total_layers,
        "alerts": alerts,
    }


def main() -> None:
    args = parse_args()
    config = get_config(args.config)
    model = create_model(config).to(args.device)
    model.train()  # enable routing counts without doing backward

    losses = []
    router_summaries = []

    for _ in range(args.steps):
        input_ids, labels = build_batch(
            config.tokenizer.vocab_size,
            args.batch_size,
            args.seq_len,
            args.junk_rate,
            junk_id=config.tokenizer.pad_token_id,
        )
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=labels, return_router_info=True)

        if "loss" in outputs:
            losses.append(outputs["loss"].item())

        if "router_info" in outputs:
            router_summaries.append(aggregate_router_info(outputs["router_info"]))

    avg_loss = sum(losses) / max(1, len(losses))
    avg_junk = sum(r["avg_junk_null_rate"] for r in router_summaries) / max(1, len(router_summaries))
    avg_signal = sum(r["avg_signal_null_rate"] for r in router_summaries) / max(1, len(router_summaries))
    healthy_layers = max(r["healthy_layers"] for r in router_summaries) if router_summaries else 0
    total_layers = max(r["total_layers"] for r in router_summaries) if router_summaries else 0

    report = {
        "avg_loss": avg_loss,
        "loss_vs_baseline_ok": None,
        "baseline_loss": args.baseline_loss,
        "avg_junk_null_rate": avg_junk,
        "avg_signal_null_rate": avg_signal,
        "healthy_layers": healthy_layers,
        "total_layers": total_layers,
        "health_ok": healthy_layers == total_layers if total_layers else None,
    }

    if args.baseline_loss is not None:
        rel_delta = abs(avg_loss - args.baseline_loss) / max(args.baseline_loss, 1e-9)
        report["loss_vs_baseline_ok"] = rel_delta <= args.loss_tol
        report["loss_relative_delta"] = rel_delta

    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tiny training loop with dashboard integration.

Runs a few steps on random data, logs routing metrics to Redis,
and updates the Team 7 dashboard in real time.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Add moe_tools to path
MOE_TOOLS = ROOT / "moe_deliverables" / "MOE_tools"
sys.path.insert(0, str(MOE_TOOLS))

from configs import get_config
from model.transformer import create_model
from utils.dashboard_logger import DashboardLogger
from utils.model_utils import build_optimizer
from moe_tools.diagnostics.routing_diagnostics import RoutingDiagnostics, RoutingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="3b_moe", choices=["1b_dense", "3b_moe", "8b_moe", "70b_moe"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--junk-rate", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--redis-key", default="moe_metrics")
    parser.add_argument("--metrics-out", default="", help="Write JSON summary metrics to this path")
    return parser.parse_args()


def build_batch(vocab_size: int, batch_size: int, seq_len: int, junk_rate: float, junk_id: int):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.rand(batch_size, seq_len) < junk_rate
    tokens[mask] = junk_id
    labels = tokens.clone()
    return tokens, labels


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    config = get_config(args.config)
    model = create_model(config).to(args.device)
    model.train()

    optimizer = build_optimizer(model, lr=args.lr)

    diagnostics = RoutingDiagnostics(RoutingConfig(
        num_routed_experts=config.num_routed_experts,
        num_shared_experts=config.num_shared_experts,
        num_null_experts=config.num_null_experts,
        top_k=config.router.top_k,
        num_layers=config.num_layers,
        null_expert_start_idx=config.num_routed_experts,
    ))

    logger = DashboardLogger(
        enabled=True,
        redis_url=args.redis_url,
        redis_key=args.redis_key,
    )

    losses = []
    throughputs = []
    last_metrics = None

    def _parse_percent(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            raw = value.strip()
            if raw.endswith("%"):
                raw = raw[:-1]
            try:
                return float(raw)
            except ValueError:
                return None
        return None

    for step in range(1, args.steps + 1):
        step_start = time.time()
        input_ids, labels = build_batch(
            config.tokenizer.vocab_size,
            args.batch_size,
            args.seq_len,
            args.junk_rate,
            junk_id=config.tokenizer.pad_token_id,
        )
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels, return_router_info=True)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        step_time = max(time.time() - step_start, 1e-6)

        if step % args.log_every == 0 and "router_info" in outputs:
            for layer_info in outputs["router_info"]:
                idx = layer_info["expert_indices"].view(-1, layer_info["expert_indices"].shape[-1]).detach().cpu()
                w = layer_info["gating_weights"].view(-1, layer_info["gating_weights"].shape[-1]).detach().cpu()
                tok = input_ids.view(-1).detach().cpu()
                diagnostics.log_batch(
                    layer_idx=layer_info["layer_idx"],
                    expert_indices=idx.tolist(),
                    expert_weights=w.tolist(),
                    token_ids=tok.tolist(),
                )

            diagnostics.step()
            metrics = diagnostics.get_dashboard_metrics()
            metrics.setdefault("training", {})
            metrics["training"]["loss"] = float(loss.item())
            metrics["training"]["throughput"] = float((args.batch_size * args.seq_len) / step_time)
            logger.log(metrics)
            last_metrics = metrics

        if step % args.log_every == 0:
            print(f"step={step} loss={loss.item():.4f}")
        losses.append(loss.item())
        throughputs.append((args.batch_size * args.seq_len) / step_time)

    logger.close()

    if args.metrics_out:
        summary = {
            "config": args.config,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "avg_loss": sum(losses) / len(losses) if losses else None,
            "final_loss": losses[-1] if losses else None,
            "avg_tokens_per_sec": sum(throughputs) / len(throughputs) if throughputs else None,
            "final_tokens_per_sec": throughputs[-1] if throughputs else None,
        }
        if last_metrics:
            summary["routing_health_gates"] = last_metrics.get("health_gates", {})
            summary["all_gates_pass"] = last_metrics.get("all_gates_pass")
            summary["routing_health"] = last_metrics.get("routing_health", {})
            summary["null_expert"] = last_metrics.get("null_expert", {})
            summary["alerts"] = last_metrics.get("alerts", [])
            summary["null_on_junk_pct"] = _parse_percent(
                last_metrics.get("null_expert", {}).get("junk_to_null_rate")
            )
            summary["null_on_signal_pct"] = _parse_percent(
                last_metrics.get("null_expert", {}).get("signal_to_null_rate")
            )

        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

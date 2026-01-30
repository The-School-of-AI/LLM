#!/usr/bin/env python3
"""
Tiny training loop with dashboard integration.

Runs a few steps on random data, logs routing metrics to Redis,
and updates the Team 7 dashboard in real time.
"""

import argparse
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

        if step % args.log_every == 0:
            print(f"step={step} loss={loss.item():.4f}")

    logger.close()


if __name__ == "__main__":
    main()

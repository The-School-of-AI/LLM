#!/usr/bin/env python3
"""
Expand a dense (1B) checkpoint into a MoE (3B) checkpoint.

This copies FFN weights into routed/shared experts for MoE layers and
keeps other weights (embeddings/attention/norms) unchanged. Router and
gating weights remain randomly initialized in the target MoE model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import config_3b_moe  # noqa: E402
from model.transformer import create_model  # noqa: E402
from utils.model_utils import expand_dense_to_moe, load_checkpoint, save_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand a 1B dense checkpoint into a 3B MoE checkpoint."
    )
    parser.add_argument("--dense-ckpt", required=True, help="Path to 1B dense checkpoint")
    parser.add_argument("--out-ckpt", required=True, help="Path to save expanded 3B MoE checkpoint")
    parser.add_argument("--device", default="cpu", help="Device for loading (default: cpu)")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="Noise std for expert symmetry breaking (defaults to 3B config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dense_ckpt = load_checkpoint(args.dense_ckpt, model=None, device=args.device, strict=True)
    dense_state = dense_ckpt["model_state_dict"]

    moe_config = config_3b_moe.get_config()
    noise_std = (
        args.noise_std
        if args.noise_std is not None
        else moe_config.expert.noise_std_for_expansion
    )

    moe_layer_indices = [
        idx for idx in range(moe_config.num_layers) if moe_config.is_moe_layer(idx)
    ]

    expanded_state = expand_dense_to_moe(
        dense_state_dict=dense_state,
        num_experts=moe_config.num_routed_experts,
        num_shared_experts=moe_config.num_shared_experts,
        noise_std=noise_std,
        moe_layer_indices=moe_layer_indices,
    )

    moe_model = create_model(moe_config).to(args.device)
    missing, unexpected = moe_model.load_state_dict(expanded_state, strict=False)
    if missing:
        print(f"[expand] Missing keys (kept init): {len(missing)}")
    if unexpected:
        print(f"[expand] Unexpected keys (ignored): {len(unexpected)}")

    step = int(dense_ckpt.get("step", 0))
    save_checkpoint(
        model=moe_model,
        optimizer=None,
        config=moe_config,
        step=step,
        path=args.out_ckpt,
        additional_state={
            "source_checkpoint": args.dense_ckpt,
            "expanded_from_dense": True,
        },
    )

    print(f"Saved expanded checkpoint to {args.out_ckpt}")


if __name__ == "__main__":
    main()

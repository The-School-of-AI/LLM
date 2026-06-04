#!/usr/bin/env python3
"""Upcycle the 2B dense checkpoint into the 5B MoE topology.

The 2B and 5B stages both use eight backbone layers. The growth step is not a
depth expansion; it copies the shared model weights and initializes the MoE FFN
from the trained dense FFN in each layer.
"""

from __future__ import annotations

import argparse
import glob
import logging
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from lightninglm.data.data import get_tokenizer
from lightninglm.models.recurrence_model_1b_non_rev import (
    KroneckerConfig as DenseKroneckerConfig,
)
from lightninglm.models.recurrence_model_1b_non_rev import (
    KroneckerEmbeddings as DenseKroneckerEmbeddings,
)
from lightninglm.models.recurrence_model_1b_non_rev import Model1B
from lightninglm.models.recurrence_model_1b_non_rev import ModelConfig as DenseConfig
from lightninglm.models.recurrence_model_3b_moe import Model3B
from lightninglm.models.recurrence_model_3b_moe import ModelConfig as MoEConfig

log = logging.getLogger("dense_to_moe")


def load_state_dict(path: Path) -> dict:
    if path.is_dir():
        candidate = path / "mp_rank_00_model_states.pt"
        if candidate.exists():
            path = candidate
        else:
            matches = sorted(glob.glob(str(path / "*.pt")))
            if not matches:
                raise FileNotFoundError(f"no .pt checkpoint found under {path}")
            path = Path(matches[0])

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "module" in payload:
        state = payload["module"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint at {path} did not contain a state dict")
    return {
        key: (
            value.to(torch.bfloat16)
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32
            else value
        )
        for key, value in state.items()
    }


def build_vocab_and_codec(tokenizer_dir: Path):
    tokenizer = get_tokenizer(str(tokenizer_dir))
    bpe_vocab = []
    for idx in range(len(tokenizer)):
        try:
            token = tokenizer.decode([idx])
            bpe_vocab.append(token if token else f"<unk_{idx}>")
        except Exception:
            bpe_vocab.append(f"<unk_{idx}>")

    pf_config = DenseKroneckerConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True,
    )
    pf_codec = DenseKroneckerEmbeddings(pf_config)
    return tokenizer, bpe_vocab, pf_codec


def build_dense_model(tokenizer_dir: Path) -> tuple[nn.Module, list[str], nn.Module]:
    tokenizer, bpe_vocab, pf_codec = build_vocab_and_codec(tokenizer_dir)
    config = DenseConfig()
    config.vocab_size = len(tokenizer)
    model = Model1B(
        config=config,
        embedding_type="kronecker",
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    )
    return model.to(dtype=torch.bfloat16), bpe_vocab, pf_codec


def build_moe_model(
    vocab_size: int, bpe_vocab: list[str], pf_codec: nn.Module
) -> nn.Module:
    config = MoEConfig()
    config.vocab_size = vocab_size
    config.moe_backend = "auto"
    config.require_fused_moe_kernel = False
    config.allow_moe_vectorized_fallback = True
    model = Model3B(
        config=config,
        embedding_type="kronecker",
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    )
    return model.to(dtype=torch.bfloat16)


def extract_dense_ffn_weights(layer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense_mlp = layer.mlp_block.sublayer.mlp
    liger_mlp = dense_mlp.mlp
    return (
        liger_mlp.gate_proj.weight.data.clone(),
        liger_mlp.up_proj.weight.data.clone(),
        liger_mlp.down_proj.weight.data.clone(),
    )


def extract_mtp_ffn_weights(
    mtp_block,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense_mlp = mtp_block.mlp.mlp
    liger_mlp = dense_mlp.mlp
    return (
        liger_mlp.gate_proj.weight.data.clone(),
        liger_mlp.up_proj.weight.data.clone(),
        liger_mlp.down_proj.weight.data.clone(),
    )


def random_partition_experts(
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    num_experts: int,
    target_dim: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    source_dim = w_gate.shape[0]
    experts = []
    for expert_idx in range(num_experts):
        rng = torch.Generator(device="cpu").manual_seed(seed + expert_idx * 1000)
        indices = torch.randperm(source_dim, generator=rng)[:target_dim]
        experts.append(
            (
                w_gate[indices, :].T.contiguous(),
                w_up[indices, :].T.contiguous(),
                w_down[:, indices].T.contiguous(),
            )
        )
    return experts


def drop_upcycle_experts(
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    num_experts: int,
    target_dim: int,
    seed: int,
    drop_fraction: float = 0.5,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    source_dim = w_gate.shape[0]
    num_drop = int(target_dim * drop_fraction)
    experts = []
    stats = (
        (w_gate.mean().item(), w_gate.std().item()),
        (w_up.mean().item(), w_up.std().item()),
        (w_down.mean().item(), w_down.std().item()),
    )
    for expert_idx in range(num_experts):
        rng = torch.Generator(device="cpu").manual_seed(seed + expert_idx * 1000)
        indices = torch.randperm(source_dim, generator=rng)[:target_dim]
        drop_indices = torch.randperm(target_dim, generator=rng)[:num_drop]

        wg_e = w_gate[indices, :].clone()
        wu_e = w_up[indices, :].clone()
        wd_e = w_down[:, indices].clone()

        wg_e[drop_indices, :] = torch.normal(
            mean=stats[0][0],
            std=stats[0][1],
            size=(num_drop, w_gate.shape[1]),
            dtype=w_gate.dtype,
        )
        wu_e[drop_indices, :] = torch.normal(
            mean=stats[1][0],
            std=stats[1][1],
            size=(num_drop, w_up.shape[1]),
            dtype=w_up.dtype,
        )
        wd_e[:, drop_indices] = torch.normal(
            mean=stats[2][0],
            std=stats[2][1],
            size=(w_down.shape[0], num_drop),
            dtype=w_down.dtype,
        )

        experts.append((wg_e.T.contiguous(), wu_e.T.contiguous(), wd_e.T.contiguous()))
    return experts


def assign_moe_weights(
    moe_ffn,
    shared_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expert_weights: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    w_gate, w_up, w_down = shared_weights
    moe_ffn.shared_gate.weight.data.copy_(w_gate)
    moe_ffn.shared_up.weight.data.copy_(w_up)
    moe_ffn.shared_down.weight.data.copy_(w_down)
    moe_ffn.W_gate.data.copy_(torch.stack([item[0] for item in expert_weights]))
    moe_ffn.W_up.data.copy_(torch.stack([item[1] for item in expert_weights]))
    moe_ffn.W_down.data.copy_(torch.stack([item[2] for item in expert_weights]))
    if getattr(moe_ffn, "gate", None) is not None:
        moe_ffn.gate.gate.weight.data.normal_(mean=0.0, std=0.02)
        moe_ffn.gate.logit_bias.data.zero_()
        moe_ffn.gate.null_logit.data.zero_()


def copy_shared_weights(dense_model: nn.Module, moe_model: nn.Module) -> None:
    if getattr(dense_model, "kronecker_embeddings", None) is not None:
        moe_model.kronecker_embeddings.load_state_dict(
            dense_model.kronecker_embeddings.state_dict()
        )
    if getattr(dense_model, "pf_to_model", None) is not None:
        moe_model.pf_to_model.load_state_dict(dense_model.pf_to_model.state_dict())
    if getattr(dense_model, "embed_norm", None) is not None:
        moe_model.embed_norm.load_state_dict(dense_model.embed_norm.state_dict())

    moe_model.norm.load_state_dict(dense_model.norm.state_dict())
    moe_model.lm_head.load_state_dict(dense_model.lm_head.state_dict())
    moe_model.lambda_r_raw.data.copy_(dense_model.lambda_r_raw.data)
    moe_model.memory_ln.load_state_dict(dense_model.memory_ln.state_dict())
    moe_model.memory_gate_proj.load_state_dict(
        dense_model.memory_gate_proj.state_dict()
    )

    for dense_layer, moe_layer in zip(dense_model.layers, moe_model.layers):
        moe_layer.attn_block.load_state_dict(dense_layer.attn_block.state_dict())
        moe_layer.mlp_block.coeffs.load_state_dict(
            dense_layer.mlp_block.coeffs.state_dict()
        )
        moe_layer.mlp_block.norm.load_state_dict(
            dense_layer.mlp_block.norm.state_dict()
        )

    dense_mtp = getattr(dense_model, "mtp_block", None)
    moe_mtp = getattr(moe_model, "mtp_block", None)
    if dense_mtp is not None and moe_mtp is not None:
        moe_mtp.fusion_proj.load_state_dict(dense_mtp.fusion_proj.state_dict())
        moe_mtp.attn.load_state_dict(dense_mtp.attn.state_dict())
        moe_mtp.attn_block.coeffs.load_state_dict(
            dense_mtp.attn_block.coeffs.state_dict()
        )
        moe_mtp.attn_block.norm.load_state_dict(dense_mtp.attn_block.norm.state_dict())
        moe_mtp.mlp_block.coeffs.load_state_dict(
            dense_mtp.mlp_block.coeffs.state_dict()
        )
        moe_mtp.mlp_block.norm.load_state_dict(dense_mtp.mlp_block.norm.state_dict())


def upcycle_dense_to_moe(
    dense_model: nn.Module,
    moe_model: nn.Module,
    strategy: Callable,
    seed: int,
) -> nn.Module:
    copy_shared_weights(dense_model, moe_model)
    num_experts = moe_model.config.num_real_experts
    target_dim = moe_model.config.expert_intermediate_size

    for layer_idx, (dense_layer, moe_layer) in enumerate(
        zip(dense_model.layers, moe_model.layers)
    ):
        log.info("initializing MoE experts for layer %s", layer_idx)
        dense_weights = extract_dense_ffn_weights(dense_layer)
        expert_weights = strategy(
            *dense_weights,
            num_experts=num_experts,
            target_dim=target_dim,
            seed=seed + layer_idx,
        )
        assign_moe_weights(
            moe_layer.mlp_block.sublayer.moe, dense_weights, expert_weights
        )

    dense_mtp = getattr(dense_model, "mtp_block", None)
    moe_mtp = getattr(moe_model, "mtp_block", None)
    if dense_mtp is not None and moe_mtp is not None:
        dense_weights = extract_mtp_ffn_weights(dense_mtp)
        expert_weights = strategy(
            *dense_weights,
            num_experts=num_experts,
            target_dim=target_dim,
            seed=seed + 10000,
        )
        assign_moe_weights(moe_mtp.mlp.moe, dense_weights, expert_weights)

    return moe_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="2B dense checkpoint .pt or ZeRO checkpoint directory",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="output 5B MoE initialization checkpoint",
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument(
        "--strategy", choices=["partition", "drop"], default="partition"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.manual_seed(args.seed)

    started = time.time()
    dense_model, bpe_vocab, pf_codec = build_dense_model(args.tokenizer_dir)
    dense_state = load_state_dict(args.src)
    dense_model.load_state_dict(dense_state, strict=True)
    dense_model.eval()

    moe_model = build_moe_model(dense_model.config.vocab_size, bpe_vocab, pf_codec)
    strategy_fn = (
        random_partition_experts
        if args.strategy == "partition"
        else drop_upcycle_experts
    )
    moe_model = upcycle_dense_to_moe(dense_model, moe_model, strategy_fn, args.seed)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": moe_model.state_dict(),
            "metadata": {
                "source": str(args.src),
                "target_stage": "5B",
                "growth": "2B dense to 5B MoE upcycle",
                "strategy": args.strategy,
                "seed": args.seed,
                "elapsed_seconds": time.time() - started,
            },
        },
        args.dst,
    )
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Save deterministic init model for Test 10")
    parser.add_argument("--config", required=True, help="Path to test YAML config")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--meta", required=True, help="Output metadata .json path")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    out_path = Path(args.output).resolve()
    meta_path = Path(args.meta).resolve()

    test_root = cfg_path.parents[1]
    code_dir = test_root / "code"
    if not code_dir.exists():
        raise FileNotFoundError(f"Missing code directory: {code_dir}")

    sys.path.insert(0, str(code_dir))

    from src.data import get_tokenizer
    from src.kernels import (
        HAS_FLA,
        HAS_TRITON,
        fla_gated_delta_rule,
        fused_indexer_topk,
        triton_sparse_attention,
    )
    from src.models.liger_ops import LigerFusedLinearCrossEntropyLoss, LigerSwiGLUMLP
    from src.models.recurrence_model_3b_moe import KroneckerConfig, KroneckerEmbeddings
    from src.models.recurrence_model_3b_moe import Model1B, ModelConfig, MoEFFN
    from src.utils import set_seed

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    tokenizer_name = cfg["model"]["tokenizer_name"]
    embedding_type = cfg["model"].get("embedding_type", "standard")
    model_overrides = cfg["model"].get("overrides", {})
    require_gsa_triton = bool(cfg["training"].get("require_gsa_triton", False))
    require_deltanet_fused = bool(cfg["training"].get("require_deltanet_fused", False))
    require_additional_fused_kernels = bool(
        cfg["training"].get("require_additional_fused_kernels", False)
    )

    tokenizer = get_tokenizer(tokenizer_name)
    model_cfg = ModelConfig()
    model_cfg.vocab_size = len(tokenizer)
    model_cfg.require_fused_deltanet_kernel = bool(
        cfg["model"].get("require_fused_deltanet_kernel", False)
    )

    for key, value in model_overrides.items():
        if not hasattr(model_cfg, key):
            raise ValueError(f"Unknown model override key: {key}")
        setattr(model_cfg, key, value)

    if model_cfg.data_sparsity > 0.0 and model_cfg.num_real_experts > 0:
        model_cfg.num_null_experts = int(
            model_cfg.num_real_experts * (1.0 - model_cfg.data_sparsity) / model_cfg.data_sparsity
        )
        model_cfg.total_expert_slots = model_cfg.num_real_experts + model_cfg.num_null_experts
    else:
        model_cfg.num_null_experts = 0
        model_cfg.total_expert_slots = 0

    if (model_cfg.num_deltanet_layers + model_cfg.num_gsa_layers) != model_cfg.num_layers:
        raise RuntimeError(
            "Invalid layer mix: num_deltanet_layers + num_gsa_layers must equal num_layers"
        )

    bpe_vocab = None
    pf_codec = None
    if embedding_type == "kronecker":
        bpe_vocab = []
        for i in range(len(tokenizer)):
            try:
                token = tokenizer.decode([i])
                bpe_vocab.append(token if token else f"<unk_{i}>")
            except Exception:
                bpe_vocab.append(f"<unk_{i}>")
        pf_cfg = KroneckerConfig(
            CHAR_DIM=256,
            POS_DIM=32,
            D=8192,
            length_normalize=True,
            truncate_long_words=True,
        )
        pf_codec = KroneckerEmbeddings(pf_cfg)

    model = Model1B(
        config=model_cfg,
        embedding_type=embedding_type,
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    ).to(dtype=torch.bfloat16)

    if type(getattr(model, "stack", None)).__name__ != "ReversibleMidpointStack":
        raise RuntimeError(
            f"Expected ReversibleMidpointStack, got: {type(getattr(model, 'stack', None)).__name__}"
        )

    # Confirm DDDGDDDG layer assignment for 8 layers.
    expected_pattern = ["deltanet", "deltanet", "deltanet", "gsa", "deltanet", "deltanet", "deltanet", "gsa"]
    actual_pattern = list(getattr(model, "layer_types", []))
    if actual_pattern != expected_pattern:
        raise RuntimeError(f"Expected layer pattern {expected_pattern}, got {actual_pattern}")

    if require_gsa_triton:
        if not HAS_TRITON:
            raise RuntimeError("Triton is required for Test 10, but HAS_TRITON=False")
        if fused_indexer_topk is None:
            raise RuntimeError("fused_indexer_topk is required for Test 10 but unavailable")
        if triton_sparse_attention is None:
            raise RuntimeError("triton_sparse_attention is required for Test 10 but unavailable")

    if require_deltanet_fused:
        if not HAS_FLA:
            raise RuntimeError("FLA is required for Test 10 DeltaNet fused path, but HAS_FLA=False")
        if fla_gated_delta_rule is None:
            raise RuntimeError("fla_gated_delta_rule is required for Test 10 but unavailable")

    if require_additional_fused_kernels:
        fused_ce = getattr(model, "liger_fused_ce", None)
        if fused_ce is None or not isinstance(fused_ce, LigerFusedLinearCrossEntropyLoss):
            raise RuntimeError("Expected model.liger_fused_ce to be LigerFusedLinearCrossEntropyLoss")
        if not hasattr(model, "_fused_linear_cross_entropy"):
            raise RuntimeError("Expected model._fused_linear_cross_entropy to exist")
        first_mlp = getattr(model.layers[0].ffn_block.sublayer, "mlp", None)
        if first_mlp is None or not isinstance(first_mlp, LigerSwiGLUMLP):
            raise RuntimeError("Expected first layer MLP to use LigerSwiGLUMLP")

    # 3B-class MoE assertions.
    if model_cfg.num_real_experts != 20:
        raise RuntimeError(f"Expected num_real_experts=20, got {model_cfg.num_real_experts}")
    if model_cfg.num_null_experts != 20:
        raise RuntimeError(f"Expected num_null_experts=20, got {model_cfg.num_null_experts}")
    if model_cfg.total_expert_slots != 40:
        raise RuntimeError(f"Expected total_expert_slots=40, got {model_cfg.total_expert_slots}")
    if model_cfg.top_k != 2:
        raise RuntimeError(f"Expected top_k=2, got {model_cfg.top_k}")
    if model_cfg.expert_intermediate_size != 1024:
        raise RuntimeError(
            f"Expected expert_intermediate_size=1024, got {model_cfg.expert_intermediate_size}"
        )
    if model_cfg.shared_expert_intermediate_size != 2048:
        raise RuntimeError(
            "Expected shared_expert_intermediate_size=2048, "
            f"got {model_cfg.shared_expert_intermediate_size}"
        )

    layer0_mlp = getattr(model.layers[0].ffn_block.sublayer, "mlp", None)
    if layer0_mlp is None or not isinstance(layer0_mlp, MoEFFN):
        raise RuntimeError("Expected MoEFFN in Test 10 (MoE must be enabled)")
    gate_slots = int(getattr(layer0_mlp.gate, "total_slots", -1))
    if gate_slots != 40:
        raise RuntimeError(f"Expected layer0 MoE gate total_slots=40, got {gate_slots}")

    if embedding_type == "kronecker":
        if not getattr(model, "use_kronecker", False):
            raise RuntimeError("Expected model.use_kronecker=True for kronecker embedding test")
        if getattr(model, "token_embed", None) is not None:
            raise RuntimeError("Expected model.token_embed=None when using kronecker embeddings")
        if getattr(model, "pf_to_model", None) is None:
            raise RuntimeError("Expected model.pf_to_model to exist in kronecker mode")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "seed": seed,
        "model_variant": "reversible",
        "embedding_type": embedding_type,
        "tokenizer_name": tokenizer_name,
        "vocab_size": len(tokenizer),
        "model_overrides": model_overrides,
    }
    torch.save(payload, out_path)

    meta = {
        "seed": seed,
        "model_variant": "reversible",
        "embedding_type": embedding_type,
        "tokenizer_name": tokenizer_name,
        "vocab_size": len(tokenizer),
        "model_overrides": model_overrides,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "init_checkpoint_path": str(out_path),
        "init_checkpoint_sha256": sha256_file(out_path),
        "config_path": str(cfg_path),
        "reversibility_assertion_passed": True,
        "dddgdddg_assertion_passed": True,
        "moe_assertions_passed": True,
        "require_gsa_triton": bool(require_gsa_triton),
        "gsa_triton_assertion_passed": bool(require_gsa_triton),
        "require_deltanet_fused": bool(require_deltanet_fused),
        "deltanet_fused_assertion_passed": bool(require_deltanet_fused),
        "require_additional_fused_kernels": bool(require_additional_fused_kernels),
        "additional_fused_assertion_passed": bool(require_additional_fused_kernels),
        "kronecker_assertions_passed": bool(embedding_type == "kronecker"),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved init model: {out_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

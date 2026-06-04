#!/usr/bin/env python3
"""Save deterministic init model for 1B Non-Reversible or 3B MoE Reversible training."""

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
    parser = argparse.ArgumentParser(
        description="Save deterministic init model for 1B Non-Rev"
    )
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

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from lightninglm.data.data import get_tokenizer
    from lightninglm.kernels import (
        HAS_TRITON,
        fused_indexer_topk,
        triton_sparse_attention,
    )
    from lightninglm.utils.utils import set_seed

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    model_name = cfg.get("model", {}).get("model_name", "1b")
    embedding_type = cfg["model"].get("embedding_type", "kronecker")
    require_gsa_triton = bool(cfg["training"].get("require_gsa_triton", False))
    is_3b = model_name == "3bmoe"
    is_8b = model_name == "8bmoe"

    if is_8b:
        from lightninglm.models.recurrence_model_8b_moe import (
            KroneckerConfig,
            KroneckerEmbeddings,
        )
        from lightninglm.models.recurrence_model_8b_moe import Model8B as ModelClass
        from lightninglm.models.recurrence_model_8b_moe import ModelConfig
    elif is_3b:
        from lightninglm.models.recurrence_model_3b_moe import (
            KroneckerConfig,
            KroneckerEmbeddings,
        )
        from lightninglm.models.recurrence_model_3b_moe import Model3B as ModelClass
        from lightninglm.models.recurrence_model_3b_moe import ModelConfig
    else:
        from lightninglm.models.recurrence_model_1b_non_rev import (
            KroneckerConfig,
            KroneckerEmbeddings,
        )
        from lightninglm.models.recurrence_model_1b_non_rev import Model1B as ModelClass
        from lightninglm.models.recurrence_model_1b_non_rev import ModelConfig

    tokenizer = get_tokenizer()
    model_cfg = ModelConfig()
    model_cfg.moe_backend = "auto"
    model_cfg.require_fused_moe_kernel = False
    model_cfg.allow_moe_vectorized_fallback = True
    model_cfg.vocab_size = len(tokenizer)

    bpe_vocab = None
    pf_codec = None
    if embedding_type == "kronecker":
        print(f"Building Kronecker vocabulary for {len(tokenizer)} tokens...")
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

    variant_label = (
        "8B MoE Reversible"
        if is_8b
        else ("3B MoE Reversible" if is_3b else "1B Non-Reversible")
    )
    print(
        f"Building {variant_label} model (vocab={len(tokenizer)}, embed={embedding_type})..."
    )
    model = ModelClass(
        config=model_cfg,
        embedding_type=embedding_type,
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    ).to(dtype=torch.bfloat16)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e9:.3f}B)")

    if require_gsa_triton:
        if not HAS_TRITON:
            raise RuntimeError("Triton is required but HAS_TRITON=False")
        if fused_indexer_topk is None:
            raise RuntimeError("fused_indexer_topk is required but unavailable")
        if triton_sparse_attention is None:
            raise RuntimeError("triton_sparse_attention is required but unavailable")

    if embedding_type == "kronecker":
        if not getattr(model, "use_kronecker", False):
            raise RuntimeError(
                "Expected model.use_kronecker=True for kronecker embedding"
            )
        if getattr(model, "token_embed", None) is not None:
            raise RuntimeError(
                "Expected model.token_embed=None when using kronecker embeddings"
            )
        if getattr(model, "pf_to_model", None) is None:
            raise RuntimeError("Expected model.pf_to_model to exist in kronecker mode")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_variant = (
        "reversible-8b-moe"
        if is_8b
        else ("reversible-3b-moe" if is_3b else "non-reversible")
    )
    payload = {
        "state_dict": model.state_dict(),
        "seed": seed,
        "model_variant": model_variant,
        "embedding_type": embedding_type,
        "vocab_size": len(tokenizer),
    }
    torch.save(payload, out_path)

    meta = {
        "seed": seed,
        "model_variant": model_variant,
        "embedding_type": embedding_type,
        "vocab_size": len(tokenizer),
        "parameter_count": int(n_params),
        "init_checkpoint_path": str(out_path),
        "init_checkpoint_sha256": sha256_file(out_path),
        "config_path": str(cfg_path),
        "require_gsa_triton": bool(require_gsa_triton),
        "kronecker_assertions_passed": bool(embedding_type == "kronecker"),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved init model: {out_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

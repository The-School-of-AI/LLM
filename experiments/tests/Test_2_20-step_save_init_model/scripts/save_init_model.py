#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
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
    parser = argparse.ArgumentParser(description="Save deterministic init model for Test 2")
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
    from src.models.different_recurrence_model_1b_wo_rev import KroneckerConfig, KroneckerEmbeddings
    from src.models.different_recurrence_model_1b_wo_rev import Model1B, ModelConfig
    from src.utils import set_seed

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    tokenizer_name = cfg["model"]["tokenizer_name"]
    embedding_type = cfg["model"].get("embedding_type", "standard")

    tokenizer = get_tokenizer(tokenizer_name)
    model_cfg = ModelConfig()
    model_cfg.vocab_size = len(tokenizer)
    model_cfg.require_fused_deltanet_kernel = bool(cfg["model"].get("require_fused_deltanet_kernel", False))

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
        pf_cfg = KroneckerConfig(CHAR_DIM=256, POS_DIM=32, D=8192, length_normalize=True, truncate_long_words=True)
        pf_codec = KroneckerEmbeddings(pf_cfg)

    model = Model1B(
        config=model_cfg,
        embedding_type=embedding_type,
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    ).to(dtype=torch.bfloat16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "seed": seed,
        "model_variant": "diff_rec",
        "embedding_type": embedding_type,
        "tokenizer_name": tokenizer_name,
        "vocab_size": len(tokenizer),
    }
    torch.save(payload, out_path)

    meta = {
        "seed": seed,
        "model_variant": "diff_rec",
        "embedding_type": embedding_type,
        "tokenizer_name": tokenizer_name,
        "vocab_size": len(tokenizer),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "init_checkpoint_path": str(out_path),
        "init_checkpoint_sha256": sha256_file(out_path),
        "config_path": str(cfg_path),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved init model: {out_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

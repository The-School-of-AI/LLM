#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml

from llm.data import get_tokenizer
from llm.kernels import HAS_TRITON, fused_indexer_topk, triton_sparse_attention
from llm.models.recurrence_model_1b import (
    KroneckerConfig,
    KroneckerEmbeddings,
    Model1B,
    ModelConfig,
)
from llm.utils import set_seed


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save deterministic init model for Test 13"
    )
    parser.add_argument("--config", required=True, help="Path to test YAML config")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--meta", required=True, help="Output metadata .json path")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    out_path = Path(args.output).resolve()
    meta_path = Path(args.meta).resolve()

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    tokenizer_name = _resolve_path(cfg["model"]["tokenizer_name"], cfg_path)
    embedding_type = cfg["model"].get("embedding_type", "standard")
    require_gsa_triton = bool(cfg["training"].get("require_gsa_triton", False))

    tokenizer = get_tokenizer(tokenizer_name)
    model_cfg = ModelConfig()
    model_cfg.vocab_size = len(tokenizer)

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

    # Critical correctness check: this test must run reversible stack.
    if type(getattr(model, "stack", None)).__name__ != "ReversibleMidpointStack":
        raise RuntimeError(
            f"Expected ReversibleMidpointStack, got: {type(getattr(model, 'stack', None)).__name__}"
        )

    if require_gsa_triton:
        if not HAS_TRITON:
            raise RuntimeError("Triton is required for Test 6, but HAS_TRITON=False")
        if fused_indexer_topk is None:
            raise RuntimeError(
                "fused_indexer_topk is required for Test 6 but unavailable"
            )
        if triton_sparse_attention is None:
            raise RuntimeError(
                "triton_sparse_attention is required for Test 6 but unavailable"
            )

    if embedding_type == "kronecker":
        # Critical correctness check: Kronecker path must replace standard token embedding.
        if not getattr(model, "use_kronecker", False):
            raise RuntimeError(
                "Expected model.use_kronecker=True for kronecker embedding test"
            )
        if getattr(model, "token_embed", None) is not None:
            raise RuntimeError(
                "Expected model.token_embed=None when using kronecker embeddings"
            )
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
    }
    torch.save(payload, out_path)

    meta = {
        "seed": seed,
        "model_variant": "reversible",
        "embedding_type": embedding_type,
        "tokenizer_name": tokenizer_name,
        "vocab_size": len(tokenizer),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "init_checkpoint_path": str(out_path),
        "init_checkpoint_sha256": sha256_file(out_path),
        "config_path": str(cfg_path),
        "reversibility_assertion_passed": True,
        "require_gsa_triton": bool(require_gsa_triton),
        "gsa_triton_assertion_passed": bool(require_gsa_triton),
        "kronecker_assertions_passed": bool(embedding_type == "kronecker"),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved init model: {out_path}")
    print(f"Saved metadata: {meta_path}")


def _resolve_path(path, config_path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    if config_path:
        base = os.path.dirname(os.path.abspath(config_path))
        return os.path.abspath(os.path.join(base, path))
    return os.path.abspath(path)


if __name__ == "__main__":
    main()

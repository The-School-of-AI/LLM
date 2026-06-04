#!/usr/bin/env python3
"""Write deterministic tensor hashes for a PyTorch checkpoint.

The manifest records each tensor name, dtype, shape, and SHA-256 digest of its
CPU-contiguous bytes. It is intended for release reproducibility checks between
stage outputs, grown checkpoints, and consolidated 120B artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def unwrap_state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        for key in ("state_dict", "module", "model"):
            if key in obj and isinstance(obj[key], dict):
                return unwrap_state_dict(obj[key])
    if not isinstance(obj, dict):
        raise TypeError("checkpoint does not contain a dictionary state")
    return obj


def tensor_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = unwrap_state_dict(checkpoint)
    entries = []

    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        entries.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": tensor_digest(value),
            }
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "tensor_count": len(entries),
        "tensors": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} with {len(entries)} tensor hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a depth-grown checkpoint by re-entering source layers.

This utility is for the 5B -> 9B depth-growth transition. It expands an
8-layer MoE checkpoint into the 20-layer 9B topology with the public
`lightninglm_5b_to_9b` mapping:

    1-8, 1-4, 1-8

or, zero-indexed:

    [0,1,2,3,4,5,6,7, 0,1,2,3, 0,1,2,3,4,5,6,7]

The rule preserves the source terminal layer as the terminal layer of the grown
model, so the language-model head continues to receive hidden states from the
layer it was trained to read.

The script is deliberately conservative: it only rewrites layer-indexed keys
whose prefix matches the supplied layer prefix. Non-layer tensors are copied
once. Target-only tensors can be provided by loading the output into an
instantiated target model and letting that model validate any missing keys.
"""
import argparse
import re
from collections import OrderedDict
from pathlib import Path

import torch

MAPPINGS = {
    "lightninglm_5b_to_9b": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ],
}


def parse_mapping(name_or_csv: str) -> list[int]:
    if name_or_csv in MAPPINGS:
        return MAPPINGS[name_or_csv]
    return [int(x.strip()) for x in name_or_csv.split(",") if x.strip()]


def load_state_dict(path: Path) -> OrderedDict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "module" in obj:
        obj = obj["module"]
    if not isinstance(obj, dict):
        raise TypeError(f"expected a checkpoint dict at {path}")
    return OrderedDict(obj)


def depth_map_state_dict(
    state: OrderedDict, mapping: list[int], layer_prefix: str
) -> OrderedDict:
    pattern = re.compile(rf"^{re.escape(layer_prefix)}\.(\d+)\.(.+)$")
    source_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
    non_layer = OrderedDict()

    for key, value in state.items():
        match = pattern.match(key)
        if match:
            layer_idx = int(match.group(1))
            suffix = match.group(2)
            source_by_layer.setdefault(layer_idx, []).append((suffix, value))
        else:
            non_layer[key] = value

    missing = sorted(set(mapping) - set(source_by_layer))
    if missing:
        raise KeyError(
            f"source checkpoint is missing layers referenced by mapping: {missing}"
        )

    out = OrderedDict(non_layer)
    for target_idx, source_idx in enumerate(mapping):
        for suffix, value in source_by_layer[source_idx]:
            out[f"{layer_prefix}.{target_idx}.{suffix}"] = value.clone()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--mapping", default="lightninglm_5b_to_9b")
    parser.add_argument("--layer-prefix", default="layers")
    parser.add_argument("--metadata-key", default="_depth_map_metadata")
    args = parser.parse_args()

    mapping = parse_mapping(args.mapping)
    state = load_state_dict(args.src)
    grown = depth_map_state_dict(state, mapping, args.layer_prefix)
    grown[args.metadata_key] = {
        "source": str(args.src),
        "mapping_name": args.mapping,
        "mapping": mapping,
        "layer_prefix": args.layer_prefix,
        "note": "LightningLM 5B -> 9B depth growth; terminal source layer preserved.",
    }

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grown, args.dst)
    print(f"wrote {args.dst} with {len(mapping)} target layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

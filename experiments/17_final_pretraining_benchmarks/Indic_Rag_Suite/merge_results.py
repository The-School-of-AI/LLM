"""
Merge sharded result JSONs (e.g. from --shard-index i --shard-total N).
Aggregates retrieval (Hit@1, MRR, MRR@K, Recall@k, NDCG@k) and generation (EM, token_f1) by n-weighted average.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def merge_retrieval_results(shard_results: list[dict]) -> dict:
    """Merge retrieval: n-weighted average for all numeric metrics (hit_at_1, mrr, mrr_at_10, recall_at_*, ndcg_at_*)."""
    merged: dict[str, dict] = {}
    for shard in shard_results:
        for lang, m in shard.items():
            if lang not in merged:
                merged[lang] = {"n": 0}
            n = m.get("n", 0)
            merged[lang]["n"] += n
            for k, v in m.items():
                if k == "n" or not isinstance(v, (int, float)):
                    continue
                key_sum = f"{k}_sum"
                merged[lang][key_sum] = merged[lang].get(key_sum, 0) + v * n
    out: dict[str, dict] = {}
    for lang, v in merged.items():
        n = v["n"]
        out[lang] = {"n": n}
        for k, val in list(v.items()):
            if k.endswith("_sum"):
                base_k = k[:-4]
                out[lang][base_k] = val / n if n else 0.0
    return out


def merge_generation_results(shard_results: list[dict]) -> dict:
    """Merge generation: n-weighted average for exact_match and token_f1."""
    merged: dict[str, dict] = {}
    for shard in shard_results:
        for lang, m in shard.items():
            if lang not in merged:
                merged[lang] = {"n": 0, "em_sum": 0.0, "f1_sum": 0.0}
            n = int(m.get("n", 0))
            merged[lang]["n"] += n
            merged[lang]["em_sum"] += m.get("exact_match", 0) * n
            merged[lang]["f1_sum"] += m.get("token_f1", 0) * n
    out: dict[str, dict] = {}
    for lang, v in merged.items():
        n = v["n"]
        if n == 0:
            out[lang] = {"exact_match": 0.0, "n": 0}
        else:
            out[lang] = {
                "exact_match": v["em_sum"] / n,
                "n": n,
            }
            if v["f1_sum"] != 0 or "token_f1" in str(shard_results):
                out[lang]["token_f1"] = v["f1_sum"] / n
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sharded benchmark result JSONs")
    parser.add_argument("results", nargs="+", help="Paths to result JSON files")
    parser.add_argument("-o", "--output", required=True, help="Merged output JSON path")
    args = parser.parse_args()

    all_data = [load_json(Path(p)) for p in args.results]
    if not all_data:
        print("No result files", file=sys.stderr)
        return 1

    base = all_data[0]
    merged = {
        "config": {**base.get("config", {}), "merged_from": len(all_data), "sharded": True},
        "tasks": {},
    }

    if "retrieval" in base.get("tasks", {}):
        retrieval_shards = [d["tasks"]["retrieval"] for d in all_data if "retrieval" in d.get("tasks", {})]
        merged["tasks"]["retrieval"] = merge_retrieval_results(retrieval_shards)

    if "generation" in base.get("tasks", {}):
        gen_shards = [d["tasks"]["generation"] for d in all_data if "generation" in d.get("tasks", {})]
        merged["tasks"]["generation"] = merge_generation_results(gen_shards)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(all_data)} files -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

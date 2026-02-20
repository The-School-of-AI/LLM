"""
Check selected_indices.jsonl across output/coresets stages for overlapping chunk ids.
Usage: run from repo root or from coreset_engine; script locates `coreset_engine/output/coresets`.
"""

import json
import os
from collections import Counter

ROOT = os.path.join(
    os.path.dirname(__file__), "..", "coreset_engine", "output", "coresets"
)
ROOT = os.path.normpath(ROOT)


def read_ids(path):
    ids = []
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # try common keys
                if isinstance(obj, dict):
                    if "chunk_id" in obj:
                        ids.append(obj["chunk_id"])
                    elif "id" in obj:
                        ids.append(obj["id"])
                    else:
                        # take first value if single-key dict
                        vals = list(obj.values())
                        if vals:
                            ids.append(str(vals[0]))
                else:
                    ids.append(str(obj))
            except Exception:
                # plain-line id
                ids.append(line)
    return ids


def main():
    if not os.path.isdir(ROOT):
        print(f"Coreset output folder not found: {ROOT}")
        return 1

    stage_sets = {}
    stage_counts = {}
    for name in sorted(os.listdir(ROOT)):
        stage_dir = os.path.join(ROOT, name)
        if not os.path.isdir(stage_dir):
            continue
        sel_file = os.path.join(stage_dir, "selected_indices.jsonl")
        ids = read_ids(sel_file)
        stage_sets[name] = set(ids)
        stage_counts[name] = len(ids)

    print("Stage counts:")
    for s, c in stage_counts.items():
        print(f" - {s}: {c}")

    # total unique
    all_ids = Counter()
    for s, st in stage_sets.items():
        for i in st:
            all_ids[i] += 1
    total_unique = len(all_ids)
    total_ids_across = sum(stage_counts.values())
    print(f"\nTotal IDs across stages (sum of counts): {total_ids_across}")
    print(f"Total unique IDs across stages (union): {total_unique}")

    # overlaps
    multi = {i: cnt for i, cnt in all_ids.items() if cnt > 1}
    print(f"Chunks appearing in >1 stage: {len(multi)}")

    if multi:
        # pairwise overlaps
        names = list(stage_sets.keys())
        print("\nPairwise overlaps:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = names[i]
                b = names[j]
                ov = len(stage_sets[a].intersection(stage_sets[b]))
                if ov > 0:
                    print(f" - {a} ∩ {b}: {ov}")

        # top duplicates
        print("\nSample duplicated IDs (up to 20) with counts and stages:")
        shown = 0
        for cid, cnt in sorted(multi.items(), key=lambda x: -x[1]):
            if shown >= 20:
                break
            stages = [s for s, st in stage_sets.items() if cid in st]
            print(f" - {cid}: count={cnt}, stages={stages}")
            shown += 1
    else:
        print("No overlaps detected. Good.")

    # return non-zero if any overlap
    return 2 if multi else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

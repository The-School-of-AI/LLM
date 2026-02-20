import glob
import json
from collections import defaultdict
from itertools import combinations

base = "output/coresets"
files = glob.glob(base + "/*/selected_indices.jsonl")
if not files:
    print("No selected_indices.jsonl files found under", base)
    raise SystemExit(1)

stage_ids = {}
for f in files:
    stage = f.split("/")[-2]
    ids = []
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                # assume each line is either a chunk id or a JSON object with 'chunk_id'
                if line.startswith("{"):
                    obj = json.loads(line)
                    cid = obj.get("chunk_id") or obj.get("id") or obj.get("chunk")
                else:
                    cid = line
                if cid is None:
                    continue
                ids.append(str(cid))
            except Exception:
                continue
    stage_ids[stage] = ids

# counts
for s, ids in stage_ids.items():
    print(f"Stage {s}: {len(ids)} selected chunks")

# compute overlaps
id_to_stages = defaultdict(list)
for s, ids in stage_ids.items():
    for cid in ids:
        id_to_stages[cid].append(s)

multi = {cid: stages for cid, stages in id_to_stages.items() if len(stages) > 1}
print("\nTotal unique selected chunk ids across all stages:", len(id_to_stages))
print("Total chunk ids appearing in >1 stage:", len(multi))

# per-pair overlap counts


pair_counts = defaultdict(int)
for cid, stages in multi.items():
    for a, b in combinations(sorted(stages), 2):
        pair_counts[(a, b)] += 1

if pair_counts:
    print("\nOverlap counts between stage pairs:")
    for (a, b), cnt in sorted(pair_counts.items(), key=lambda x: -x[1]):
        print(f"  {a} <-> {b}: {cnt}")

# sample duplicate ids
sample = list(multi.items())[:10]
if sample:
    print("\nSample overlapping IDs and stages:")
    for cid, stages in sample:
        print(f"  {cid}: {stages}")
else:
    print("\nNo overlapping IDs found.")


import json, sys, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curriculum_tools import map_doc_to_band, compute_quantile_edges, reduce_band_distribution
from curriculum_yaml_generator import build_curriculum_yaml, dump_yaml

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    data_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    for row in load_jsonl(data_path):
        out = map_doc_to_band(row, text_key="text", edges=None)
        scores.append(out["score"])

    edges = compute_quantile_edges(scores)
    (out_dir / "band_edges.json").write_text(json.dumps({"edges": edges}, indent=2), encoding="utf-8")

    band_rows = []
    for row in load_jsonl(data_path):
        out = map_doc_to_band(row, text_key="text", edges=edges)
        out["proxy_domain"] = f'{row.get("dataset_id","")}/{row.get("segment_id","")}'
        band_rows.append(out)

    base_dist = reduce_band_distribution(band_rows)
    (out_dir / "base_distribution.json").write_text(json.dumps(base_dist, indent=2), encoding="utf-8")

    curr = build_curriculum_yaml(
        frozen_on=datetime.date.today().isoformat(),
        compute_profiles_from_base=True,
        base_distribution=base_dist,
    )
    dump_yaml(curr, str(out_dir / "curriculum.yaml"))

    band_doc_counts = {}
    band_token_counts = {}
    for r in band_rows:
        b = r["band"]
        band_doc_counts[b] = band_doc_counts.get(b, 0) + 1
        band_token_counts[b] = band_token_counts.get(b, 0) + int(r.get("tokens", 0))

    (out_dir / "band_doc_counts.json").write_text(json.dumps(band_doc_counts, indent=2), encoding="utf-8")
    (out_dir / "band_token_counts.json").write_text(json.dumps(band_token_counts, indent=2), encoding="utf-8")

    print("DONE")
    print("Edges:", edges)
    print("Base distribution:", base_dist)

if __name__ == "__main__":
    main()

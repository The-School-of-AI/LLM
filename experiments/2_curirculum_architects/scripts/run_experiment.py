
import json, sys, datetime
from pathlib import Path
try:
    import pandas as pd
except ImportError:
    pd = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curriculum_tools import map_doc_to_band, compute_quantile_edges, reduce_band_distribution
from curriculum_yaml_generator import build_curriculum_yaml, dump_yaml

TEXT_KEYS = ["text","content","raw","document","body"]

def load_data(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif path.suffix == ".parquet":
        if pd is None:
            raise ImportError("pandas is required to read parquet files. Please pip install pandas pyarrow.")
        df = pd.read_parquet(path, engine='pyarrow')
        # Yield records one by one
        for _, row in df.iterrows():
            yield row.to_dict()
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .jsonl or .parquet")

def pick_text_key(row):
    for k in TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return k
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_experiment.py <input_file> <output_dir>")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    text_key_used = None
    n_used = 0
    n_skipped = 0

    print(f"Reading {data_path}...")

    # Pass 1: Sampling for edges
    for row in load_data(data_path):
        tk = pick_text_key(row)
        if tk is None:
            n_skipped += 1
            continue
        if text_key_used is None:
            text_key_used = tk
        
        out = map_doc_to_band(row, text_key=tk, edges=None)
        scores.append(out["score"])
        n_used += 1

    if n_used == 0:
        raise RuntimeError("No usable text field found. Tried keys: " + ", ".join(TEXT_KEYS))

    edges = compute_quantile_edges(scores)
    (out_dir / "band_edges.json").write_text(json.dumps(
        {"edges": edges, "text_key": text_key_used, "n_used": n_used, "n_skipped": n_skipped},
        indent=2
    ), encoding="utf-8")

    # Pass 2: Banding
    band_rows = []
    for row in load_data(data_path):
        tk = pick_text_key(row)
        if tk is None:
            continue
        out = map_doc_to_band(row, text_key=tk, edges=edges)

        ds = row.get("dataset_id") or row.get("source") or row.get("dataset") or "generic_dataset"
        seg = row.get("segment_id") or row.get("subset") or row.get("split") or "default"
        out["proxy_domain"] = f"{ds}/{seg}"
        out["dataset_id"] = ds
        out["segment_id"] = seg
        band_rows.append(out)

    base_dist = reduce_band_distribution(band_rows)
    (out_dir / "base_distribution.json").write_text(json.dumps(base_dist, indent=2), encoding="utf-8")

    # Generate Curriculum
    curr = build_curriculum_yaml(
        frozen_on=datetime.date.today().isoformat(),
        compute_profiles_from_base=True,
        base_distribution=base_dist
    )
    dump_yaml(curr, str(out_dir / "curriculum.yaml"))

    # Compute Stats
    band_doc_counts = {}
    band_token_counts = {}
    modality_token_counts = {}
    for r in band_rows:
        b = r["band"]
        band_doc_counts[b] = band_doc_counts.get(b, 0) + 1
        band_token_counts[b] = band_token_counts.get(b, 0) + int(r.get("tokens", 0))
        
        mods = r.get("modalities", {})
        for m, flag in mods.items():
            if flag:
                modality_token_counts[m] = modality_token_counts.get(m, 0) + int(r.get("tokens", 0))

    (out_dir / "band_doc_counts.json").write_text(json.dumps(band_doc_counts, indent=2), encoding="utf-8")
    (out_dir / "band_token_counts.json").write_text(json.dumps(band_token_counts, indent=2), encoding="utf-8")
    (out_dir / "modality_token_counts.json").write_text(json.dumps(modality_token_counts, indent=2), encoding="utf-8")

    print("DONE")
    print("text_key_used:", text_key_used)
    print("n_used:", n_used, "n_skipped:", n_skipped)
    print("edges:", edges)
    print("base_dist:", base_dist)

if __name__ == "__main__":
    main()

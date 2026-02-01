import json
from pathlib import Path

from tomlkit import parse

CONFIG_PATH = Path("configs/sample.toml")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    cfg = parse(CONFIG_PATH.read_text(encoding="utf-8"))
    out_root = Path(cfg["run"]["out_dir"])
    manifest_path = out_root / "sample_manifest.json"

    if not manifest_path.exists():
        raise RuntimeError("out/sample_manifest.json not found. Run sample_download.py first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dolma_shards_cfg = int(cfg["sample"].get("dolma_shards", 0))
    failures = 0

    print("Verifying outputs based on manifest\n")

    for o in manifest.get("outputs", []):
        ds = o.get("dataset")
        mode = o.get("mode")

        # Dolma shard verification (only if configured)
        if ds == "dolma" and mode == "shards":
            if dolma_shards_cfg == 0:
                # If manifest has dolma but config says 0, just ignore it.
                print("SKIP: dolma present in manifest but dolma_shards=0 in config")
                continue

            shards = o.get("shards_downloaded", [])
            requested = int(o.get("shards_requested", 0))

            if requested != dolma_shards_cfg:
                print(f"FAIL: dolma config mismatch (config={dolma_shards_cfg}, manifest={requested})")
                failures += 1
                continue

            if len(shards) != requested:
                print(f"FAIL: dolma shard count mismatch (expected={requested}, got={len(shards)})")
                failures += 1
                continue

            ok = True
            for s in shards:
                p = Path(s["path"])
                if not p.exists():
                    print(f"FAIL: missing dolma shard file {p}")
                    failures += 1
                    ok = False
                    continue
                size = p.stat().st_size
                if size <= 0:
                    print(f"FAIL: empty dolma shard file {p}")
                    failures += 1
                    ok = False
            if ok:
                print(f"OK: dolma shards verified ({requested})")
            continue

        # Rows (JSONL) verification
        p_str = o.get("path")
        if not p_str:
            print(f"FAIL: missing path in manifest entry: {o}")
            failures += 1
            continue

        p = Path(p_str)
        expected = int(o.get("written_n", -1))
        label = f"{ds}"

        # make label more informative
        if ds == "sangraha":
            label = f"sangraha/{o.get('config')}/{o.get('split')}"
        elif ds == "indiccorp_v2":
            label = f"indiccorp_v2/{o.get('split')}"

        if not p.exists():
            print(f"FAIL: missing file {label}: {p}")
            failures += 1
            continue

        # parse + count
        count = 0
        try:
            for _ in read_jsonl(p):
                count += 1
        except Exception as e:
            print(f"FAIL: JSONL parse error {label}: {p} ({e})")
            failures += 1
            continue

        if expected >= 0 and count != expected:
            print(f"FAIL: count mismatch {label}: expected={expected} got={count}")
            failures += 1
        else:
            print(f"OK: {label}: {count} records")

    if failures:
        raise RuntimeError(f"Verification failed with {failures} error(s).")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

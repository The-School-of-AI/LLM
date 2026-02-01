import json
import random
import time
from itertools import islice
from pathlib import Path

import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tomlkit import parse

CONFIG_PATH = Path("configs/sample.toml")


def write_jsonl(path: Path, rows_iter):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    first_keys = []
    with path.open("w", encoding="utf-8") as f:
        for r in rows_iter:
            if count == 0:
                first_keys = list(r.keys())
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1
    return count, first_keys


def stream_rows_to_jsonl(repo: str, name: str, split: str, n: int, out_path: Path):
    ds = load_dataset(repo, name if name else None, split=split, streaming=True)
    return write_jsonl(out_path, islice(iter(ds), n))


def load_dolma_urls(repo_id: str, version: str):
    url_file = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"urls/{version}.txt",
    )
    urls = Path(url_file).read_text(encoding="utf-8").splitlines()
    return [u.strip() for u in urls if u.strip()]


def download_file(url: str, out_path: Path, timeout=1200):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main():
    cfg = parse(CONFIG_PATH.read_text(encoding="utf-8"))
    out_root = Path(cfg["run"]["out_dir"]) / "raw"
    seed = int(cfg["run"].get("seed", 42))
    random.seed(seed)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "outputs": [],
    }

    # ---- Dolma: download first N shard files ----
    dolma_cfg = cfg["datasets"]["dolma"]
    dolma_repo = str(dolma_cfg["repo_id"])
    dolma_ver = str(dolma_cfg["version"])
    dolma_shards = int(cfg["sample"].get("dolma_shards", 0))

    if dolma_shards > 0:
        print(f"Sampling dolma (shards) | repo={dolma_repo} version={dolma_ver} shards={dolma_shards}")
        urls = load_dolma_urls(dolma_repo, dolma_ver)
        if len(urls) < dolma_shards:
            raise RuntimeError(f"Requested {dolma_shards} shards but only found {len(urls)} urls")

        dolma_dir = out_root / "dolma" / dolma_ver / "shards"
        downloaded = []

        for i, url in enumerate(urls[:dolma_shards]):
            filename = url.split("/")[-1]
            out_path = dolma_dir / filename

            if out_path.exists() and out_path.stat().st_size > 0:
                size = out_path.stat().st_size
                print(f"  skipping (exists) {i+1}/{dolma_shards}: {filename} ({size} bytes)")
                downloaded.append(
                    {"url": url, "path": str(out_path), "bytes": size, "filename": filename, "skipped": True}
                )
                continue

            print(f"  downloading shard {i+1}/{dolma_shards}: {filename}")
            download_file(url, out_path)
            size = out_path.stat().st_size
            downloaded.append({"url": url, "path": str(out_path), "bytes": size, "filename": filename})
            print(f"  saved {size} bytes -> {out_path}")

        manifest["outputs"].append(
            {
                "dataset": "dolma",
                "mode": "shards",
                "repo_id": dolma_repo,
                "version": dolma_ver,
                "shards_requested": dolma_shards,
                "shards_downloaded": downloaded,
            }
        )
    else:
        print("Skipping dolma (shards=0)")

    # ---- Sangraha: streaming row sample ----
    sang_cfg = cfg["datasets"]["sangraha"]
    sang_repo = str(sang_cfg["repo"])
    sang_config = str(sang_cfg.get("config", "verified"))
    sang_n = int(cfg["sample"]["sangraha_n"])
    langs = [str(x) for x in cfg["sample"]["sangraha_langs"]]

    for lang in langs:
        print(f"Sampling sangraha | repo={sang_repo} config={sang_config} split={lang} n={sang_n}")
        out_path = out_root / "sangraha" / sang_config / lang / "sample.jsonl"
        written_n, first_keys = stream_rows_to_jsonl(sang_repo, sang_config, lang, sang_n, out_path)

        manifest["outputs"].append(
            {
                "dataset": "sangraha",
                "mode": "rows",
                "repo": sang_repo,
                "config": sang_config,
                "split": lang,
                "requested_n": sang_n,
                "written_n": written_n,
                "path": str(out_path),
                "first_keys": first_keys[:25],
            }
        )
        print(f"  wrote {written_n} -> {out_path}")

    # ---- IndicCorpV2: streaming row sample for multiple splits ----
    ind_cfg = cfg["datasets"]["indiccorp_v2"]
    ind_repo = str(ind_cfg["repo"])
    ind_n = int(cfg["sample"]["indic_n"])
    ind_splits = [str(x) for x in cfg["sample"]["indic_splits"]]

    for sp in ind_splits:
        print(f"Sampling indiccorp_v2 | repo={ind_repo} split={sp} n={ind_n}")
        out_path = out_root / "indiccorp_v2" / sp.replace("/", "_") / "sample.jsonl"
        written_n, first_keys = stream_rows_to_jsonl(ind_repo, "", sp, ind_n, out_path)

        manifest["outputs"].append(
            {
                "dataset": "indiccorp_v2",
                "mode": "rows",
                "repo": ind_repo,
                "split": sp,
                "requested_n": ind_n,
                "written_n": written_n,
                "path": str(out_path),
                "first_keys": first_keys[:25],
            }
        )
        print(f"  wrote {written_n} -> {out_path}")

    manifest_path = out_root.parent / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

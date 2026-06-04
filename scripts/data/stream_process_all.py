#!/usr/bin/env python3
"""
Streaming batch pipeline: download → process → upload shards → delete local → next batch.

Processes the full ~4.5 TB pretraining corpus in ~6 batches, never using more than
~1 TB of local NVMe at a time.

Usage:
    python3 stream_process_all.py --batch 0          # run batch 0
    python3 stream_process_all.py --batch all         # run all batches sequentially
    python3 stream_process_all.py --batch 0 --dry-run # preview batch 0 without downloading
    python3 stream_process_all.py --list-batches      # show batch plan

Paths (hardcoded for this cluster):
    Source S3:    s3://t1-dataacquisition-datasets/
    Upload S3:    s3://t1-dataacquisition-datasets-2/shards/
    Local work:   /mnt/nvme1/batch_work/
    Tokenizer:    /mnt/nvme1/FINAL_TOKENIZER/
    process.py:   /mnt/nvme1/process.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_BUCKET = "s3://t1-dataacquisition-datasets"
UPLOAD_BUCKET = "s3://t1-dataacquisition-datasets-2/shards"
LOCAL_WORK_DIR = "/mnt/nvme0/pipeline"
LOCAL_INPUT_DIR = "/mnt/nvme0/pipeline/input"
LOCAL_OUTPUT_DIR = "/mnt/nvme0/pipeline/output"
TOKENIZER_DIR = "/mnt/nvme0/FINAL_TOKENIZER"
PROCESS_PY = "/mnt/nvme0/process.py"
BAND_MAP_PATH = "/mnt/nvme0/full_band_map.json"
WORKERS = 127  # leave 1 CPU free on 128-core machine
SHARD_MAX_BLOCKS = 8192
SHARD_OFFSET_FILE = "/mnt/nvme0/shard_offset.json"  # persists across batches

# ═══════════════════════════════════════════════════════════════════════════
# SOURCE MANIFEST — maps every S3 path to the BUILTIN_BAND_MAP source name
# ═══════════════════════════════════════════════════════════════════════════
#
# Format: (s3_prefix, target_source_name, approx_size_gb)
# s3_prefix is relative to SOURCE_BUCKET
#
# Sources come from two locations:
#   1. processed_dataset/normalized_data/source=<name>/  (already has source= prefix)
#   2. huggingface_golden/<Name>/  (needs renaming to source=<mapped_name>)
#
# B5 benchmark sources are EXCLUDED (already processed separately).


@dataclass
class SourceEntry:
    s3_prefix: str  # full S3 prefix under SOURCE_BUCKET
    target_name: str  # maps to BUILTIN_BAND_MAP key
    size_gb: float  # approximate compressed parquet size
    path_type: str  # "normalized" or "golden"


# ── normalized_data sources (already have source= prefix) ────────────────
# These download directly into source=<name>/ format
NORMALIZED_SOURCES = [
    # English Web (B0-B2)
    SourceEntry(
        "processed_dataset/normalized_data/source=cc_head/",
        "cc_head",
        700.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=cc_middle/",
        "cc_middle",
        903.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=cc_tail/",
        "cc_tail",
        819.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=cc_news/",
        "cc_news",
        17.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=refinedweb/",
        "refinedweb",
        828.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=C4/", "C4", 252.0, "normalized"
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=reddit/",
        "reddit",
        175.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=stackexchange/",
        "stackexchange",
        24.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=megawika/",
        "megawika",
        32.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=books/", "books", 7.0, "normalized"
    ),
    # English Academic/STEM (B2-B5)
    SourceEntry(
        "processed_dataset/normalized_data/source=pes2o/", "pes2o", 111.0, "normalized"
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=redpajama-arxiv/",
        "redpajama-arxiv",
        24.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=proof_pile_2-algebraic_stack/",
        "proof_pile_2-algebraic_stack",
        10.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=proof_pile_2-open_web_math/",
        "proof_pile_2-open_web_math",
        12.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=flan/", "flan", 26.0, "normalized"
    ),
    # Code (B3)
    SourceEntry(
        "processed_dataset/normalized_data/source=Starcoder/",
        "Starcoder",
        199.0,
        "normalized",
    ),
    # Indic: Sangraha
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_hi/",
        "sangraha_hi",
        22.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_bn/",
        "sangraha_bn",
        18.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_ta/",
        "sangraha_ta",
        11.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_te/",
        "sangraha_te",
        9.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_kn/",
        "sangraha_kn",
        5.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_ml/",
        "sangraha_ml",
        7.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_gu/",
        "sangraha_gu",
        6.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_mr/",
        "sangraha_mr",
        8.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_pa/",
        "sangraha_pa",
        3.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_or/",
        "sangraha_or",
        2.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sangraha_as/",
        "sangraha_as",
        1.0,
        "normalized",
    ),
    # Indic: AI4Bharat
    SourceEntry(
        "processed_dataset/normalized_data/source=ai-bharath-BPCC_seed/",
        "ai-bharath-BPCC_seed",
        0.05,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=ai-bharath-comparable/",
        "ai-bharath-comparable",
        0.8,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=ai-bharath-daily/",
        "ai-bharath-daily",
        0.01,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=ai-bharath-ilci/",
        "ai-bharath-ilci",
        0.2,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=ai-bharath-massive/",
        "ai-bharath-massive",
        0.01,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=ai-bharath-nllb_filtered/",
        "ai-bharath-nllb_filtered",
        9.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=ai-bharath-samanantar/",
        "ai-bharath-samanantar",
        14.0,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=ai-bharath-wiki/",
        "ai-bharath-wiki",
        0.1,
        "normalized",
    ),
    # Indic: ERAV4
    SourceEntry(
        "processed_dataset/normalized_data/source=erav4_lang_hi/",
        "erav4_lang_hi",
        0.01,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=erav4_lang_as/",
        "erav4_lang_as",
        0.01,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=erav4_lang_kn/",
        "erav4_lang_kn",
        0.01,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=erav4_lang_mr/",
        "erav4_lang_mr",
        0.01,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=erav4_lang_pa/",
        "erav4_lang_pa",
        0.01,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=erav4_lang_te/",
        "erav4_lang_te",
        0.01,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=erav4_math/",
        "erav4_math",
        0.01,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=erav4_pattern/",
        "erav4_pattern",
        0.01,
        "normalized",
    ),
    # Indic: Other
    SourceEntry(
        "processed_dataset/normalized_data/source=samvaad_hi/",
        "samvaad_hi",
        0.1,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=sarvamai_mmlu/",
        "sarvamai_mmlu",
        0.1,
        "normalized",
    ),
    SourceEntry(
        "processed_dataset/normalized_data/source=ncert/", "ncert", 0.03, "normalized"
    ),
]

# ── huggingface_golden sources (need renaming to source=<mapped_name>) ───
GOLDEN_SOURCES = [
    # SFT: Math & Reasoning
    SourceEntry(
        "huggingface_golden/Nemotron-Math-v2/", "nemotron_math_v2", 47.0, "golden"
    ),
    SourceEntry("huggingface_golden/HardGen/", "hardgen", 0.03, "golden"),
    SourceEntry(
        "huggingface_golden/Claude-4.5-High-Reasoning-250x/",
        "claude_high_reasoning",
        0.002,
        "golden",
    ),
    SourceEntry("huggingface_golden/GSM8K/", "gsm8k", 0.002, "golden"),
    # SFT: Instruction & Science
    SourceEntry(
        "huggingface_golden/HelpSteer3-Feedback/", "helpsteer3", 0.12, "golden"
    ),
    SourceEntry(
        "huggingface_golden/Nemotron-Post-Training/",
        "nemotron_post_training",
        29.0,
        "golden",
    ),
    SourceEntry("huggingface_golden/MegaScience/", "megascience", 1.2, "golden"),
    # SFT: Code
    SourceEntry("huggingface_golden/Ling-Coder-SFT/", "ling_coder_sft", 5.8, "golden"),
    # SFT: General Conversation
    SourceEntry("huggingface_golden/SmolTalk2-SFT/", "smoltalk2_sft", 21.0, "golden"),
    # SFT: Structured Knowledge
    SourceEntry("huggingface_golden/SmolTalk2-Mid/", "smoltalk2_mid", 35.0, "golden"),
    SourceEntry(
        "huggingface_golden/Open-PerfectBlend/", "open_perfectblend", 1.0, "golden"
    ),
    # SFT: Preference / Alignment
    SourceEntry(
        "huggingface_golden/SmolTalk2-Preference/",
        "smoltalk2_preference",
        1.7,
        "golden",
    ),
    SourceEntry("huggingface_golden/ORPO-DPO-Mix-40K/", "orpo_dpo_mix", 0.08, "golden"),
    SourceEntry(
        "huggingface_golden/Skywork-Reward-Preference-80K/",
        "skywork_reward_preference",
        0.14,
        "golden",
    ),
    SourceEntry(
        "huggingface_golden/UltraFeedback-Binarized/",
        "ultrafeedback_binarized",
        0.1,
        "golden",
    ),
    SourceEntry(
        "huggingface_golden/Infinity-Preference/", "infinity_preference", 0.15, "golden"
    ),
    SourceEntry(
        "huggingface_golden/Arena-Human-Preference-100K/",
        "arena_human_preference",
        0.3,
        "golden",
    ),
    # SFT: Math (additional)
    SourceEntry(
        "huggingface_golden/UltraData-Math-L3/", "ultradata_math", 20.0, "golden"
    ),
]

# Combine all
ALL_SOURCES = NORMALIZED_SOURCES + GOLDEN_SOURCES

# NOTE: B5 benchmark sources (numina_math, open_math_reasoning, etc.) are
# EXCLUDED — they were already processed separately from HuggingFace.
# Also excluded: the_stack_v2 (merged with Starcoder), taco/code_contests/
# open_code_reasoning_2/open_math_instruct_2/theorem_qa/olympiad_bench (B5).


# ═══════════════════════════════════════════════════════════════════════════
# BATCH DEFINITIONS — grouped by approximate size to fit in ~1 TB local
# ═══════════════════════════════════════════════════════════════════════════


def build_batches() -> List[List[SourceEntry]]:
    """Group sources into ~6 batches, balanced by size."""
    # Batch 0: cc_head (700 GB) — single giant source
    # Batch 1: cc_tail (819 GB) — single giant source
    # Batch 2: cc_middle (903 GB) — single giant source
    # Batch 3: refinedweb (828 GB) — single giant source
    # Batch 4: C4 + Starcoder + reddit + pes2o (738 GB) — medium sources
    # Batch 5: Everything else (~300 GB) — all small sources + SFT/alignment

    by_name = {s.target_name: s for s in ALL_SOURCES}
    batches: List[List[SourceEntry]] = [[] for _ in range(6)]

    batch_assignments = {
        0: ["cc_head"],
        1: ["cc_tail"],
        2: ["cc_middle"],
        3: ["refinedweb"],
        4: ["C4", "Starcoder", "reddit", "pes2o"],
    }

    assigned = set()
    for batch_idx, names in batch_assignments.items():
        for name in names:
            if name in by_name:
                batches[batch_idx].append(by_name[name])
                assigned.add(name)

    # Batch 5: everything else
    for s in ALL_SOURCES:
        if s.target_name not in assigned:
            batches[5].append(s)

    return batches


# ═══════════════════════════════════════════════════════════════════════════
# SHARD OFFSET MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════


def load_shard_offset() -> int:
    """Load the current shard offset from persistent file."""
    if os.path.exists(SHARD_OFFSET_FILE):
        with open(SHARD_OFFSET_FILE) as f:
            data = json.load(f)
        return data.get("next_shard_offset", 0)
    return 0


def save_shard_offset(offset: int) -> None:
    """Save the next shard offset for the next batch."""
    with open(SHARD_OFFSET_FILE, "w") as f:
        json.dump(
            {
                "next_shard_offset": offset,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            f,
            indent=2,
        )
    log(f"  Saved shard offset {offset} to {SHARD_OFFSET_FILE}")


def count_shards_produced(output_dir: str) -> int:
    """Count total shards produced across all bands."""
    count = 0
    out = Path(output_dir)
    if out.exists():
        for band_dir in out.iterdir():
            if band_dir.is_dir() and band_dir.name.startswith("band_"):
                for shard_dir in band_dir.iterdir():
                    if shard_dir.is_dir() and shard_dir.name.startswith("shard_"):
                        count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def log(msg: str) -> None:
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)


def run_cmd(cmd: str, desc: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run shell command with logging."""
    log(f"  CMD: {desc}")
    log(f"       {cmd[:200]}{'...' if len(cmd) > 200 else ''}")
    t0 = time.time()
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"  FAILED ({elapsed:.0f}s): {result.stderr[-500:]}")
    else:
        log(f"  OK ({elapsed:.0f}s)")
    return result


def disk_free_gb(path: str) -> float:
    """Return free disk space in GB."""
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024**3)


def dir_size_gb(path: str) -> float:
    """Return directory size in GB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024**3)


# ═══════════════════════════════════════════════════════════════════════════
# BATCH EXECUTION
# ═══════════════════════════════════════════════════════════════════════════


def download_batch(sources: List[SourceEntry]) -> None:
    """Download all sources in a batch from S3 to local input dir."""
    os.makedirs(LOCAL_INPUT_DIR, exist_ok=True)
    total_gb = sum(s.size_gb for s in sources)
    log(f"Downloading {len(sources)} sources (~{total_gb:.0f} GB)...")
    log(f"  Free disk before: {disk_free_gb('/mnt/nvme1'):.0f} GB")

    for i, src in enumerate(sources):
        s3_path = f"{SOURCE_BUCKET}/{src.s3_prefix}"
        local_path = os.path.join(LOCAL_INPUT_DIR, f"source={src.target_name}")
        os.makedirs(local_path, exist_ok=True)

        log(
            f"  [{i+1}/{len(sources)}] {src.target_name} ({src.size_gb:.1f} GB) ← {src.s3_prefix}"
        )
        t0 = time.time()

        # aws s3 sync with progress
        cmd = f"aws s3 sync '{s3_path}' '{local_path}/' --quiet"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=3600
        )

        elapsed = time.time() - t0
        if result.returncode != 0:
            log(f"    FAILED after {elapsed:.0f}s: {result.stderr[-300:]}")
            raise RuntimeError(f"Download failed for {src.target_name}")

        # Count files downloaded
        n_files = len(list(Path(local_path).glob("*.parquet")))
        actual_gb = dir_size_gb(local_path)
        rate = actual_gb / elapsed * 1024 if elapsed > 0 else 0  # MB/s
        log(
            f"    OK: {n_files} files, {actual_gb:.1f} GB, {elapsed:.0f}s ({rate:.0f} MB/s)"
        )

    log(f"  Free disk after download: {disk_free_gb('/mnt/nvme1'):.0f} GB")
    log(f"  Input dir size: {dir_size_gb(LOCAL_INPUT_DIR):.1f} GB")


def process_batch(shard_offset: int) -> int:
    """Run process.py on local input dir. Returns number of shards produced."""
    log(f"Processing with {WORKERS} workers, shard_offset={shard_offset}...")

    cmd = (
        f"python3 {PROCESS_PY}"
        f" --input-dir {LOCAL_INPUT_DIR}"
        f" --output-dir {LOCAL_OUTPUT_DIR}"
        f" --tokenizer-dir {TOKENIZER_DIR}"
        f" --workers {WORKERS}"
        f" --shard-max-blocks {SHARD_MAX_BLOCKS}"
        f" --shard-offset {shard_offset}"
        f" --band-map {BAND_MAP_PATH}"
        f" --allow-unknown-band"
    )

    log(f"  CMD: {cmd}")
    t0 = time.time()

    # Stream output to both stdout and log file
    log_file = "/mnt/nvme1/batch_process.log"
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    with open(log_file, "a") as lf:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)

    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        log(f"  FAILED after {elapsed:.0f}s (exit code {proc.returncode})")
        raise RuntimeError("process.py failed")

    n_shards = count_shards_produced(LOCAL_OUTPUT_DIR)
    log(f"  Produced {n_shards} shards in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return n_shards


def upload_shards() -> None:
    """Upload output shards to S3."""
    out_size = dir_size_gb(LOCAL_OUTPUT_DIR)
    log(f"Uploading shards to {UPLOAD_BUCKET}/ ({out_size:.1f} GB)...")

    t0 = time.time()
    cmd = f"aws s3 sync '{LOCAL_OUTPUT_DIR}/' '{UPLOAD_BUCKET}/' --quiet"
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=3600
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"  UPLOAD FAILED after {elapsed:.0f}s: {result.stderr[-300:]}")
        raise RuntimeError("Upload failed")

    rate = out_size / elapsed * 1024 if elapsed > 0 else 0
    log(f"  Upload OK: {out_size:.1f} GB in {elapsed:.0f}s ({rate:.0f} MB/s)")


def verify_upload() -> bool:
    """Quick verify: check S3 has the shards we just uploaded."""
    log("Verifying upload...")

    # Count local shards
    local_count = count_shards_produced(LOCAL_OUTPUT_DIR)

    # Count S3 shards (just check for metadata.json files)
    cmd = f"aws s3 ls '{UPLOAD_BUCKET}/' --recursive | grep -c metadata.json"
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=120
    )
    s3_count = int(result.stdout.strip()) if result.returncode == 0 else -1

    log(f"  Local shards: {local_count}, S3 shards: {s3_count}")
    # S3 count includes shards from ALL batches, so it should be >= local
    if s3_count >= local_count:
        log("  Verification PASSED")
        return True
    else:
        log(f"  WARNING: S3 count ({s3_count}) < local count ({local_count})")
        return False


def cleanup_local() -> None:
    """Delete local input and output dirs to free disk."""
    log("Cleaning up local data...")
    for d in [LOCAL_INPUT_DIR, LOCAL_OUTPUT_DIR]:
        if os.path.exists(d):
            size = dir_size_gb(d)
            shutil.rmtree(d)
            log(f"  Deleted {d} ({size:.1f} GB)")
    log(f"  Free disk after cleanup: {disk_free_gb('/mnt/nvme1'):.0f} GB")


def run_batch(
    batch_idx: int, sources: List[SourceEntry], dry_run: bool = False
) -> None:
    """Execute one complete batch cycle."""
    total_gb = sum(s.size_gb for s in sources)
    log("=" * 70)
    log(f"BATCH {batch_idx}: {len(sources)} sources, ~{total_gb:.0f} GB")
    log("=" * 70)

    for s in sources:
        log(f"  {s.target_name:<35} {s.size_gb:>8.1f} GB  ({s.path_type})")

    if dry_run:
        log("DRY RUN — skipping download/process/upload")
        return

    free = disk_free_gb("/mnt/nvme1")
    required = total_gb * 1.3  # 30% headroom for output shards
    if free < required:
        log(f"  ERROR: Need ~{required:.0f} GB free, have {free:.0f} GB")
        log("  Run cleanup or use a larger disk")
        sys.exit(1)

    shard_offset = load_shard_offset()
    t_batch = time.time()

    # 1. Download
    t0 = time.time()
    download_batch(sources)
    dl_time = time.time() - t0

    # 2. Process
    t0 = time.time()
    n_shards = process_batch(shard_offset)
    proc_time = time.time() - t0

    # 3. Upload
    t0 = time.time()
    upload_shards()
    ul_time = time.time() - t0

    # 4. Verify
    ok = verify_upload()
    if not ok:
        log("UPLOAD VERIFICATION FAILED — keeping local data for retry")
        log("Fix the issue and re-run this batch")
        sys.exit(1)

    # 5. Save new shard offset
    new_offset = shard_offset + n_shards
    save_shard_offset(new_offset)

    # 6. Cleanup
    cleanup_local()

    total_time = time.time() - t_batch
    log(f"\nBATCH {batch_idx} COMPLETE:")
    log(f"  Download:  {dl_time/60:.1f} min")
    log(f"  Process:   {proc_time/60:.1f} min")
    log(f"  Upload:    {ul_time/60:.1f} min")
    log(f"  Total:     {total_time/60:.1f} min ({total_time/3600:.1f} hrs)")
    log(f"  Shards:    {n_shards} (offset {shard_offset} → {new_offset})")
    log(f"  Disk free: {disk_free_gb('/mnt/nvme1'):.0f} GB")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Streaming batch processing pipeline")
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Batch index (0-5) or 'all' to run all sequentially",
    )
    parser.add_argument(
        "--list-batches", action="store_true", help="Show batch plan and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview batch without downloading/processing",
    )
    parser.add_argument(
        "--reset-offset",
        action="store_true",
        help="Reset shard offset to 0 (use for fresh start)",
    )
    args = parser.parse_args()

    batches = build_batches()

    if args.reset_offset:
        save_shard_offset(0)
        log("Shard offset reset to 0")
        return

    if args.list_batches or args.batch is None:
        print("\n=== BATCH PLAN ===\n")
        grand_total = 0
        for i, batch in enumerate(batches):
            total = sum(s.size_gb for s in batch)
            grand_total += total
            sources_str = ", ".join(s.target_name for s in batch[:5])
            if len(batch) > 5:
                sources_str += f", ... (+{len(batch)-5} more)"
            print(
                f"  Batch {i}: {len(batch):>3} sources, {total:>7.0f} GB  [{sources_str}]"
            )
        print(
            f"\n  Total: {sum(len(b) for b in batches)} sources, {grand_total:.0f} GB"
        )
        print(f"  Current shard offset: {load_shard_offset()}")
        print(f"  Upload target: {UPLOAD_BUCKET}")
        return

    if args.batch == "all":
        log("Running ALL batches sequentially...")
        for i, batch in enumerate(batches):
            run_batch(i, batch, dry_run=args.dry_run)
        log("\n" + "=" * 70)
        log("ALL BATCHES COMPLETE!")
        log(f"  Final shard offset: {load_shard_offset()}")
        log("=" * 70)
    else:
        batch_idx = int(args.batch)
        if batch_idx < 0 or batch_idx >= len(batches):
            print(f"Error: batch index must be 0-{len(batches)-1}")
            sys.exit(1)
        run_batch(batch_idx, batches[batch_idx], dry_run=args.dry_run)


if __name__ == "__main__":
    main()

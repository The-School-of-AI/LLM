#!/usr/bin/env python3
"""
Performance diagnostic for selection engine bottleneck identification.
Run with: python profile_selection.py
"""

import json
import random
import time
from pathlib import Path

from src.core.config import PipelineConfig
from src.core.types import ChunkMetadata, DifficultyBand
from src.curriculum.loader import CurriculumLoader
from src.selection.engine import SelectionEngine


def profile_selection_at_scale(k: int):
    """Profile selection engine at given scale"""
    print(f"\n{'='*70}")
    print(f"Profiling selection engine with k={k} chunks")
    print(f"{'='*70}\n")

    large_path = Path("data/datasets/large_sample_chunks.jsonl")
    if not large_path.exists():
        print(f"ERROR: {large_path} not found")
        return

    # Reservoir sample k lines
    print(f"Sampling {k} chunks from {large_path.name}...", end=" ", flush=True)
    start = time.time()

    reservoir = []
    with open(large_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < k:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = line

    sample_time = time.time() - start
    print(f"DONE ({sample_time:.2f}s)\n")

    # Build chunks
    print("Building chunk metadata...", end=" ", flush=True)
    start = time.time()

    all_chunks = {}
    total_tokens = 0
    for line in reservoir:
        data = json.loads(line)
        cid = data.get("chunk_id")
        meta = ChunkMetadata(
            chunk_id=cid,
            dataset_id=data.get("dataset_id", "ds"),
            token_count=int(data.get("token_count", 0)),
            byte_length=int(data.get("byte_length", 0)),
            domain=data.get("domain", "clean_web"),
            language=data.get("language", "en"),
            band=DifficultyBand(data.get("band", "B0")),
            source_doc_id=data.get("source_doc_id", ""),
            source_url=data.get("source_url", None),
        )
        if "token_ids" in data:
            setattr(meta, "token_ids", list(data["token_ids"]))
            total_tokens += len(data["token_ids"])
        all_chunks[cid] = meta

    build_time = time.time() - start
    print(f"DONE ({build_time:.2f}s)")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Total tokens: {total_tokens:,}\n")

    # Load curriculum
    print("Loading curriculum...", end=" ", flush=True)
    start = time.time()

    curriculum_yaml = Path("data/datasets/curriculum_min_for_large_test.yaml")
    curriculum = CurriculumLoader(str(curriculum_yaml))
    ok, errors = curriculum.load()
    assert ok, f"Failed to load curriculum: {errors}"

    curriculum_time = time.time() - start
    print(f"DONE ({curriculum_time:.2f}s)\n")

    # Initialize engine
    print("Initializing SelectionEngine...", end=" ", flush=True)
    start = time.time()

    config = PipelineConfig()
    config.dedup.enable_near_dedup = False
    engine = SelectionEngine(config, curriculum)

    init_time = time.time() - start
    print(f"DONE ({init_time:.2f}s)\n")

    # Register chunks
    print("Registering chunks with engine...", end=" ", flush=True)
    start = time.time()

    chunks_list = [(cid, meta, None) for cid, meta in all_chunks.items()]
    engine.register_chunks(chunks_list)

    register_time = time.time() - start
    print(f"DONE ({register_time:.2f}s)\n")

    # Create buckets
    print("Creating stratified buckets...", end=" ", flush=True)
    start = time.time()

    engine._create_buckets(all_chunks, "1B")

    bucket_time = time.time() - start
    print(f"DONE ({bucket_time:.2f}s)")
    print(f"  Buckets created: {len(engine.buckets)}\n")

    # Score chunks
    print("Scoring chunks in buckets...", end=" ", flush=True)
    start = time.time()

    for bucket in engine.buckets.values():
        engine._score_chunks_in_bucket(bucket, all_chunks)

    score_time = time.time() - start
    print(f"DONE ({score_time:.2f}s)\n")

    # Stratified sample
    print("Stratified sampling from buckets...", end=" ", flush=True)
    start = time.time()

    selected = set()
    for bucket in engine.buckets.values():
        bucket_selection = engine._stratified_sample_from_bucket(bucket)
        selected.update(bucket_selection)

    sample_time = time.time() - start
    print(f"DONE ({sample_time:.2f}s)")
    print(f"  Selected chunks: {len(selected)}\n")

    # Summary
    total_time = (
        sample_time
        + build_time
        + curriculum_time
        + init_time
        + register_time
        + bucket_time
        + score_time
        + sample_time
    )

    print(f"{'='*70}")
    print(f"TIMING BREAKDOWN (k={k} chunks):")
    print(f"{'='*70}")
    print(
        f"  Sampling from file:      {sample_time:8.2f}s ({sample_time/total_time*100:5.1f}%)"
    )
    print(
        f"  Building metadata:       {build_time:8.2f}s ({build_time/total_time*100:5.1f}%)"
    )
    print(
        f"  Loading curriculum:      {curriculum_time:8.2f}s ({curriculum_time/total_time*100:5.1f}%)"
    )
    print(
        f"  Engine initialization:   {init_time:8.2f}s ({init_time/total_time*100:5.1f}%)"
    )
    print(
        f"  Registering chunks:      {register_time:8.2f}s ({register_time/total_time*100:5.1f}%)"
    )
    print(
        f"  Creating buckets:        {bucket_time:8.2f}s ({bucket_time/total_time*100:5.1f}%)"
    )
    print(
        f"  Scoring chunks:          {score_time:8.2f}s ({score_time/total_time*100:5.1f}%)"
    )
    print(
        f"  Stratified sampling:     {sample_time:8.2f}s ({sample_time/total_time*100:5.1f}%)"
    )
    print(f"{'='*70}")
    print(f"  TOTAL:                   {total_time:8.2f}s")
    print(f"{'='*70}\n")

    return {
        "k": k,
        "total_time": total_time,
        "score_time": score_time,
        "bucket_time": bucket_time,
        "register_time": register_time,
    }


if __name__ == "__main__":
    results = []

    # Profile at increasing scales
    for k in [50, 100, 200]:
        result = profile_selection_at_scale(k)
        if result:
            results.append(result)

    # Analyze scaling
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS:")
    print(f"{'='*70}")
    for i, r in enumerate(results):
        if i > 0:
            prev = results[i - 1]
            scale_factor = r["k"] / prev["k"]
            time_increase = r["total_time"] / prev["total_time"]
            print(
                f"k={prev['k']} → k={r['k']}: {scale_factor:.1f}x chunks, {time_increase:.1f}x time (O(n^{time_increase**0.5:.1f}))"
            )
        else:
            print(f"Baseline: k={r['k']}")
    print(f"{'='*70}\n")

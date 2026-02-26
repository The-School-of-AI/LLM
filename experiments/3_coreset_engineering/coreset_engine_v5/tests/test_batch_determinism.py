import json
from pathlib import Path

from src.io.batch_processor import BatchProcessor
from tools.verify_batch_determinism import add_stage_wise_labels, compute_signatures


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _collect_batch_chunk_ids(
    input_path: str, *, batch_size: int, shard_id: int, num_shards: int
):
    bp = BatchProcessor(batch_size=batch_size)
    files = bp.list_input_files(input_path, "jsonl")
    assert files

    row_level_shard = num_shards > 1 and len(files) == 1
    if not row_level_shard:
        files = bp.shard_files(files, shard_id, num_shards)

    out = []
    for f in files:
        for batch in bp.batch_iterator(
            f,
            shard_id=(shard_id if row_level_shard else 0),
            num_shards=(num_shards if row_level_shard else 1),
            shard_key="chunk_id",
        ):
            out.append([cid for cid, _ in batch])
    return out


def test_batch_membership_deterministic_across_runs(tmp_path):
    # Two files to exercise sorted file discovery and file-level sharding path.
    d = tmp_path / "ds"
    d.mkdir()

    _write_jsonl(
        d / "a.jsonl",
        [{"chunk_id": f"a_{i:03d}", "token_count_estimate": 10} for i in range(7)],
    )
    _write_jsonl(
        d / "b.jsonl",
        [{"chunk_id": f"b_{i:03d}", "token_count_estimate": 10} for i in range(8)],
    )

    run1 = _collect_batch_chunk_ids(str(d), batch_size=5, shard_id=0, num_shards=1)
    run2 = _collect_batch_chunk_ids(str(d), batch_size=5, shard_id=0, num_shards=1)

    assert run1 == run2


def test_row_level_sharding_deterministic_when_single_file(tmp_path):
    # Single file forces row-level sharding when num_shards>1.
    p = tmp_path / "single.jsonl"

    rows = []
    for i in range(50):
        # Mix of identifiers to exercise fallback normalization.
        if i % 10 == 0:
            rows.append({"uid": f"u_{i}", "token_count_estimate": 10})
        elif i % 11 == 0:
            rows.append({"chunk_id": "", "guid": f"g_{i}", "token_count_estimate": 10})
        else:
            rows.append({"chunk_id": f"c_{i}", "token_count_estimate": 10})

    _write_jsonl(p, rows)

    run1_s0 = _collect_batch_chunk_ids(str(p), batch_size=7, shard_id=0, num_shards=3)
    run2_s0 = _collect_batch_chunk_ids(str(p), batch_size=7, shard_id=0, num_shards=3)
    run1_s1 = _collect_batch_chunk_ids(str(p), batch_size=7, shard_id=1, num_shards=3)
    run2_s1 = _collect_batch_chunk_ids(str(p), batch_size=7, shard_id=1, num_shards=3)
    run1_s2 = _collect_batch_chunk_ids(str(p), batch_size=7, shard_id=2, num_shards=3)
    run2_s2 = _collect_batch_chunk_ids(str(p), batch_size=7, shard_id=2, num_shards=3)

    assert run1_s0 == run2_s0
    assert run1_s1 == run2_s1
    assert run1_s2 == run2_s2


def test_parquet_batch_signatures_deterministic_across_runs(tmp_path):
    import pandas as pd

    p = tmp_path / "ds.parquet"
    df = pd.DataFrame(
        {
            "chunk_id": [f"c_{i:03d}" for i in range(30)],
            "token_count_estimate": [10] * 30,
        }
    )
    df.to_parquet(p, index=False)

    run1 = compute_signatures(
        input_path=str(p),
        input_format="parquet",
        batch_size=7,
        max_rows=None,
        shard_id=0,
        num_shards=1,
        shard_key="chunk_id",
    )
    run2 = compute_signatures(
        input_path=str(p),
        input_format="parquet",
        batch_size=7,
        max_rows=None,
        shard_id=0,
        num_shards=1,
        shard_key="chunk_id",
    )

    assert run1["batches"] == run2["batches"]
    assert run1["total_chunks"] == run2["total_chunks"]
    assert run1["total_batches"] == run2["total_batches"]


def test_parquet_row_level_sharding_deterministic_when_single_file(tmp_path):
    import pandas as pd

    p = tmp_path / "single.parquet"
    df = pd.DataFrame(
        {
            "chunk_id": [f"c_{i:03d}" for i in range(50)],
            "token_count_estimate": [10] * 50,
        }
    )
    df.to_parquet(p, index=False)

    run1_s1 = compute_signatures(
        input_path=str(p),
        input_format="parquet",
        batch_size=8,
        max_rows=None,
        shard_id=1,
        num_shards=3,
        shard_key="chunk_id",
    )
    run2_s1 = compute_signatures(
        input_path=str(p),
        input_format="parquet",
        batch_size=8,
        max_rows=None,
        shard_id=1,
        num_shards=3,
        shard_key="chunk_id",
    )

    assert run1_s1["batches"] == run2_s1["batches"]


def test_stage_wise_labels_duplicate_same_signatures(tmp_path):
    import pandas as pd

    p = tmp_path / "ds.parquet"
    df = pd.DataFrame(
        {
            "chunk_id": [f"c_{i:03d}" for i in range(15)],
            "token_count_estimate": [10] * 15,
        }
    )
    df.to_parquet(p, index=False)

    base = compute_signatures(
        input_path=str(p),
        input_format="parquet",
        batch_size=5,
        max_rows=None,
        shard_id=0,
        num_shards=1,
        shard_key="chunk_id",
    )
    labeled = add_stage_wise_labels(base, ["1B", "3B", "8B", "70B"])

    assert labeled["batches"] == base["batches"]
    assert labeled["stage_reports"]["1B"]["batches"] == base["batches"]
    assert labeled["stage_reports"]["70B"]["batches"] == base["batches"]

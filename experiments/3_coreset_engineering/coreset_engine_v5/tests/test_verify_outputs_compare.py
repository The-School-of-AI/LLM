import json
from pathlib import Path

from tools.verify_batch_determinism import compare_output_dirs


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_manifest(
    p: Path, *, stage: str, actual_tokens: int, selected_chunks: int
) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "stage_name": stage,
        "target_tokens": actual_tokens,
        "actual_tokens": actual_tokens,
        "selected_chunks_count": selected_chunks,
        "shard_id": 0,
        "num_shards": 1,
        "composition": {
            "band_distribution": {"B0": 1.0},
            "domain_distribution": {
                "total": {"web": 1.0},
                "by_band": {"B0": {"web": 1.0}},
            },
            "language_distribution": {"languages": {"en": 1.0}},
        },
    }
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def test_compare_outputs_ok_when_identical(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"

    for root in (a, b):
        stage_dir = root / "1B"
        _write_jsonl(
            stage_dir / "selected_indices_part_shard000_batch000000.jsonl",
            [
                {
                    "chunk_id": "c1",
                    "token_count": 10,
                    "band": "B0",
                    "domain": "web",
                    "language": "en",
                },
                {
                    "chunk_id": "c2",
                    "token_count": 20,
                    "band": "B0",
                    "domain": "web",
                    "language": "en",
                },
            ],
        )
        _write_manifest(
            stage_dir / "manifest.json", stage="1B", actual_tokens=30, selected_chunks=2
        )

    report = compare_output_dirs(
        outputs_a=str(a), outputs_b=str(b), stages=["1B"], include_shard_manifests=False
    )
    assert report["ok"] is True
    assert report["stages"]["1B"]["match"] is True


def test_compare_outputs_detects_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"

    stage_a = a / "1B"
    stage_b = b / "1B"

    _write_jsonl(
        stage_a / "selected_indices_part_shard000_batch000000.jsonl",
        [
            {
                "chunk_id": "c1",
                "token_count": 10,
                "band": "B0",
                "domain": "web",
                "language": "en",
            },
            {
                "chunk_id": "c2",
                "token_count": 20,
                "band": "B0",
                "domain": "web",
                "language": "en",
            },
        ],
    )
    _write_manifest(
        stage_a / "manifest.json", stage="1B", actual_tokens=30, selected_chunks=2
    )

    _write_jsonl(
        stage_b / "selected_indices_part_shard000_batch000000.jsonl",
        [
            {
                "chunk_id": "c1",
                "token_count": 10,
                "band": "B0",
                "domain": "web",
                "language": "en",
            },
            {
                "chunk_id": "c3",
                "token_count": 20,
                "band": "B0",
                "domain": "web",
                "language": "en",
            },
        ],
    )
    _write_manifest(
        stage_b / "manifest.json", stage="1B", actual_tokens=30, selected_chunks=2
    )

    report = compare_output_dirs(
        outputs_a=str(a), outputs_b=str(b), stages=["1B"], include_shard_manifests=False
    )
    assert report["ok"] is False
    assert report["stages"]["1B"]["match"] is False
    assert "chunk_id_fingerprint_mismatch" in (report["stages"]["1B"]["reasons"] or [])

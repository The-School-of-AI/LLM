from pathlib import Path

import pandas as pd
import pytest
from tools.merge_selected_indices import merge_stage_parts


def test_merge_stage_parts_writes_output(tmp_path: Path):
    stage_dir = tmp_path / "1B"
    stage_dir.mkdir(parents=True, exist_ok=True)

    df1 = pd.DataFrame(
        [
            {
                "chunk_id": "a",
                "dataset_id": "ds",
                "token_count": 10,
                "token_count_estimate": 10,
                "band": "B0",
                "domain": "clean_web",
                "language": "en",
            },
            {
                "chunk_id": "b",
                "dataset_id": "ds",
                "token_count": 20,
                "token_count_estimate": 20,
                "band": "B1",
                "domain": "code",
                "language": "en",
            },
        ]
    )
    df2 = pd.DataFrame(
        [
            {
                "chunk_id": "c",
                "dataset_id": "ds2",
                "token_count": 30,
                "token_count_estimate": 30,
                "band": "B2",
                "domain": "math",
                "language": "hi",
            },
        ]
    )

    (stage_dir / "selected_indices_part_shard000_batch000000.parquet")
    df1.to_parquet(
        stage_dir / "selected_indices_part_shard000_batch000000.parquet", index=False
    )
    df2.to_parquet(
        stage_dir / "selected_indices_part_shard001_batch000000.parquet", index=False
    )

    result = merge_stage_parts(stage_dir, overwrite=True)

    assert result.output_path.exists()
    merged = pd.read_parquet(result.output_path)
    assert len(merged) == 3
    assert set(merged["chunk_id"].tolist()) == {"a", "b", "c"}


def test_merge_stage_parts_requires_parts(tmp_path: Path):
    stage_dir = tmp_path / "3B"
    stage_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        merge_stage_parts(stage_dir, overwrite=True)

import pytest


def test_parquet_batch_iterator_ignores_missing_columns(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd
    from src.io.batch_processor import BatchProcessor

    # Create a minimal parquet file that does NOT include band_score/difficulty_score/band_p_* columns.
    df = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "dataset_id": "ds",
                "token_count_estimate": 10,
                "byte_length": 5,
                "domain": "web",
                "language": "en",
                "band": "B0",
                "source_doc_id": "doc",
                "source_url": None,
            }
        ]
    )

    p = tmp_path / "mini.parquet"
    df.to_parquet(p, index=False)

    bp = BatchProcessor(batch_size=1)
    # Request a superset of columns including ones missing from the parquet schema.
    cols = [
        "chunk_id",
        "dataset_id",
        "token_count_estimate",
        "band_score",
        "difficulty_score",
        "band_p_B0",
        "band_p_B5",
    ]

    batches = list(bp.parquet_batch_iterator(str(p), batch_size_rows=1, columns=cols))
    assert len(batches) == 1
    assert len(batches[0]) == 1
    row = batches[0][0]

    # Required columns still present.
    assert row["chunk_id"] == "c1"
    assert row["dataset_id"] == "ds"

    # Missing optional columns should simply be absent from the row dict.
    assert "band_score" not in row
    assert "difficulty_score" not in row
    assert "band_p_B0" not in row

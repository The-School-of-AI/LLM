from __future__ import annotations

from pathlib import Path

from src.core.config import PipelineConfig
from src.core.types import ChunkMetadata, DifficultyBand, ProtectedSliceRule
from src.curriculum.loader import CurriculumLoader
from src.selection.engine_batched import BatchedSelectionEngine


def test_protected_slice_enforcement_respects_band_allowed_domains(
    tmp_path: Path,
) -> None:
    """Protected slice top-ups must not violate (band, domain) eligibility.

    Regression guard for cases like B4 selecting domain=web even though
    curriculum B4 allowed_domains excludes web.
    """

    cfg = PipelineConfig.load_from_file("config/pipeline.yaml")

    loader = CurriculumLoader("config/curriculum.yaml")
    ok, errors = loader.load()
    assert ok, f"Curriculum failed to load: {errors}"

    engine = BatchedSelectionEngine(cfg, loader)

    protected = [
        ProtectedSliceRule(
            band_or_domain="B4",
            minimum_preservation_ratio=0.95,
            reason="unit-test",
        )
    ]

    batch_chunks = {
        # Intentionally put the disallowed domain first lexicographically so the
        # bug would deterministically pick it without domain filtering.
        "c001": ChunkMetadata(
            chunk_id="c001",
            dataset_id="ds",
            token_count=10,
            byte_length=0,
            domain="web",
            language="en",
            band=DifficultyBand.B4,
            source_doc_id="doc",
            source_url=None,
        ),
        "c002": ChunkMetadata(
            chunk_id="c002",
            dataset_id="ds",
            token_count=10,
            byte_length=0,
            domain="code",
            language="en",
            band=DifficultyBand.B4,
            source_doc_id="doc",
            source_url=None,
        ),
    }

    # Pretend we have remaining B4 budget and stage budget.
    engine._remaining_stage_tokens = 10
    engine._remaining_band_tokens = {"B4": 10}
    engine._remaining_domain_tokens = {}

    out = engine._enforce_protected_slices_streaming(
        selected_in_batch=set(),
        batch_chunks=batch_chunks,
        protected_slices=protected,
        stage_name="1B",
    )

    assert "c001" not in out
    assert "c002" in out

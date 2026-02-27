"""
Unit tests for coreset selection engine.
Run with: pytest tests/
"""

from pathlib import Path

import pytest
from src.core.config import PipelineConfig
from src.core.types import (
    BandDistribution,
    ChunkMetadata,
    CoresetComposition,
    DifficultyBand,
    DomainDistribution,
    LanguageDistribution,
)
from src.curriculum.loader import CurriculumLoader
from src.dedup.deduplicator import ExactDeduplicator, MinHasher, SimHasher
from src.diversity.scorer import DiversityScorer, TokenFrequencyAnalyzer


class TestConfiguration:
    """Test configuration loading and validation"""

    def test_config_creation(self):
        """Test creating default config"""
        config = PipelineConfig()
        assert config.pipeline_version == "1.0.0"
        assert "70B" in config.stages

    def test_config_validation(self):
        """Test config validation"""
        config = PipelineConfig()
        valid, errors = config.validate()
        assert valid, f"Config validation failed: {errors}"

    def test_config_serialization(self):
        """Test config to/from JSON"""
        config = PipelineConfig()
        json_str = config.to_json()
        assert '"pipeline_version"' in json_str
        assert '"1.0.0"' in json_str

    def test_config_hashing(self):
        """Test config hash consistency"""
        config = PipelineConfig()

        hash1 = config.compute_hash()
        hash2 = config.compute_hash()

        assert hash1 == hash2, "Same config instance should produce same hash"

    def test_config_hash_changes_with_modification(self):
        """Test that config hash changes when config changes"""
        config1 = PipelineConfig()
        hash1 = config1.compute_hash()

        config2 = PipelineConfig()
        config2.dedup.enable_near_dedup = False
        hash2 = config2.compute_hash()

        assert hash1 != hash2, "Different configs should produce different hashes"


class TestTypes:
    """Test type definitions and structures"""

    def test_band_distribution_validation(self):
        """Test band distribution sums to ~1.0"""
        dist = BandDistribution(B0=0.45, B1=0.30, B2=0.20, B3=0.05, B4=0.0, B5=0.0)
        assert dist.validate(), "Valid distribution should pass"

    def test_band_distribution_invalid(self):
        """Test band distribution fails for invalid sums"""
        dist = BandDistribution(B0=0.5, B1=0.5, B2=0.5)  # Sums to 1.5
        assert not dist.validate(), "Invalid distribution should fail"

    def test_chunk_metadata_creation(self):
        """Test creating chunk metadata"""
        metadata = ChunkMetadata(
            chunk_id="chunk_001",
            dataset_id="test_dataset",
            token_count=4096,
            byte_length=24576,
            domain="code",
            language="en",
            band=DifficultyBand.B3,
            source_doc_id="doc_001",
        )
        assert metadata.chunk_id == "chunk_001"
        assert metadata.band == DifficultyBand.B3


class TestDeduplication:
    """Test deduplication engines"""

    def test_exact_dedup_finds_duplicates(self):
        """Test exact dedup identifies identical content"""
        dedup = ExactDeduplicator()

        text = "The quick brown fox jumps over the lazy dog"
        dedup.compute_hash("chunk_1", text)
        dedup.compute_hash("chunk_2", text)
        dedup.compute_hash("chunk_3", "Different content")

        duplicates = dedup.find_exact_duplicates()
        assert len(duplicates) == 1, "Should find one duplicate pair"
        assert duplicates[0] == ("chunk_1", "chunk_2")

    def test_simhash_similarity(self):
        """Test SimHash similarity computation"""
        text1 = "The quick brown fox"
        text2 = "The quick brown dog"  # Similar
        text3 = "Completely different content here"

        hash1 = SimHasher.compute_simhash(text1)
        hash2 = SimHasher.compute_simhash(text2)
        hash3 = SimHasher.compute_simhash(text3)

        sim_12 = SimHasher.hamming_similarity(hash1, hash2)
        sim_13 = SimHasher.hamming_similarity(hash1, hash3)

        assert sim_12 > sim_13, "Similar texts should have higher similarity"

    def test_minhash_similarity(self):
        """Test MinHash similarity computation"""
        hasher = MinHasher()

        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumps over the lazy cat"  # Similar

        hash1 = hasher.compute_minhash(text1)
        hash2 = hasher.compute_minhash(text2)

        similarity = MinHasher.jaccard_similarity(hash1, hash2)
        assert similarity > 0.5, "Similar texts should have Jaccard > 0.5"


class TestDiversity:
    """Test diversity scoring"""

    def test_token_frequency_analyzer(self):
        """Test token frequency analysis"""
        analyzer = TokenFrequencyAnalyzer(vocab_size=128_000)

        # Add tokens with different frequencies
        analyzer.add_tokens([10] * 100)  # Most frequent (0% percentile -> junk)
        analyzer.add_tokens([50000] * 5)  # Middle freq (33% percentile -> normal)
        analyzer.add_tokens([127999] * 1)  # Least frequent (100% percentile -> tail)

        assert analyzer.token_total == 106
        assert (
            analyzer.classify_token_band(10) == "junk"
        )  # Most frequent is <5%, classified as junk
        assert (
            analyzer.classify_token_band(50000) == "normal"
        )  # 33% percentile is 20-80% band

    def test_diversity_scorer(self):
        """Test diversity scoring"""
        analyzer = TokenFrequencyAnalyzer()
        analyzer.add_tokens(list(range(1000)))

        scorer = DiversityScorer(analyzer)

        token_ids = list(range(100))
        score = scorer.score_chunk_composite(
            token_ids=token_ids,
            domain="code",
            language="en",
        )

        assert 0.0 <= score <= 1.0, "Score should be in [0, 1]"


class TestCurriculum:
    """Test curriculum loading."""

    def test_curriculum_loading(self, tmp_path):
        """Test loading curriculum YAML (legacy schema)."""
        curriculum_yaml = """
version: "0.0.1"
status: "FROZEN"
guarantees:
    deterministic_sampling: true
    seed_required: true
stages:
    "70B":
        total_tokens: 240000000000
        band_ratios:
            B0: 0.05
            B1: 0.10
            B2: 0.20
            B3: 0.25
            B4: 0.25
            B5: 0.15
"""
        curriculum_path = tmp_path / "curriculum.yaml"
        curriculum_path.write_text(curriculum_yaml)

        loader = CurriculumLoader(str(curriculum_path))
        success, errors = loader.load()

        assert success, f"Failed to load curriculum: {errors}"
        assert loader.validate_curriculum_frozen()

    def test_curriculum_v06_secondary_language_list(self, tmp_path):
        """v0.6: secondary_languages supports lang as a list and earliest_stage gating."""
        import yaml

        curriculum_obj = {
            "version": "0.6",
            "status": "DRAFT",
            "global_contract": {
                "guarantees": {
                    "determinism": {
                        "sampling": True,
                        "batch_content": True,
                        "data_order": True,
                        "fixed_seed_required": True,
                    }
                }
            },
            "language_and_context": {
                "language_policy": {
                    "primary_languages": [{"lang": "en", "max_share": 0.92}],
                    "secondary_languages": [
                        {
                            "lang": ["hi", "bn"],
                            "max_share": 0.08,
                            "earliest_stage": "3B",
                        }
                    ],
                    "excluded_languages": ["fr"],
                }
            },
            "difficulty_system": {
                "bands": {
                    "B0": {
                        "allowed_domains": ["general_web_clean"],
                        "allowed_modalities": ["general_text"],
                    }
                }
            },
            "growth_schedule": {
                "stages": [
                    {"name": "1B", "order": 1, "profile": "base"},
                    {"name": "3B", "order": 2, "profile": "base"},
                ],
                "stage_profiles": {"base": {"band_weights": {"B0": 1.0}}},
            },
            "guardrails": {
                "rolling_window": {
                    "window_tokens": 100,
                    "max_band_delta": 0.1,
                    "max_domain_delta": 0.1,
                }
            },
            "domains": {
                "definition_method": "test",
                "domain_groups": [{"id": "general_web_clean"}],
                "band_domain_policy": {"B0": ["general_web_clean"]},
            },
        }

        curriculum_yaml = yaml.safe_dump(curriculum_obj, sort_keys=False)
        curriculum_path = tmp_path / "curriculum_v06.yaml"
        curriculum_path.write_text(curriculum_yaml)

        loader = CurriculumLoader(str(curriculum_path))
        success, errors = loader.load()
        assert success, f"Failed to load v0.6 curriculum: {errors}"

        allowed_1b = loader.get_allowed_languages_for_stage("1B")
        allowed_3b = loader.get_allowed_languages_for_stage("3B")
        assert "en" in allowed_1b
        assert "hi" not in allowed_1b
        assert "bn" not in allowed_1b
        assert "hi" in allowed_3b
        assert "bn" in allowed_3b


class TestIntegration:
    """Integration tests"""

    def test_pipeline_composition_creation(self):
        """Test creating coreset composition"""
        band_dist = BandDistribution(
            B0=0.05, B1=0.10, B2=0.20, B3=0.25, B4=0.25, B5=0.15
        )
        domain_dist = DomainDistribution(
            code=0.2, math=0.2, reasoning=0.2, agentic=0.1, indic=0.15, clean_web=0.15
        )
        lang_dist = LanguageDistribution(languages={"en": 0.92, "hi": 0.08})

        composition = CoresetComposition(
            band_distribution=band_dist,
            domain_distribution=domain_dist,
            language_distribution=lang_dist,
        )

        comp_dict = composition.to_dict()
        assert comp_dict["band_distribution"]["B5"] == 0.15

    def test_selection_using_real_sample(self, tmp_path):
        """Run selection engine on a small real sample from data/datasets/sample_chunks.jsonl"""
        import json

        from src.core.config import PipelineConfig
        from src.core.types import ChunkMetadata, DifficultyBand
        from src.curriculum.loader import CurriculumLoader
        from src.selection.engine import SelectionEngine

        sample_path = Path("data/datasets/sample_chunks.jsonl")
        if not sample_path.exists():
            pytest.skip(f"Sample dataset not found: {sample_path}")

        # Load a small sample (all lines) into all_chunks
        all_chunks = {}
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                chunk_id = data.get("chunk_id")
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    dataset_id=data.get("dataset_id", "ds"),
                    token_count=int(data.get("token_count_estimate", 0)),
                    byte_length=int(data.get("byte_length", 0)),
                    domain=data.get("domain", "clean_web"),
                    language=data.get("language", "en"),
                    band=DifficultyBand(data.get("band", "B0")),
                    source_doc_id=data.get("source_doc_id", ""),
                    source_url=data.get("source_url", None),
                )
                all_chunks[chunk_id] = meta

        # Minimal curriculum with 1B stage so engine can compute buckets
        curriculum_yaml = tmp_path / "curriculum_min.yaml"
        curriculum_yaml.write_text(
            """
version: "0.0.1"
status: "FROZEN"
stages:
  "1B":
    total_tokens: 20000000000
    band_ratios:
      B0: 0.10
      B1: 0.15
      B2: 0.20
      B3: 0.25
      B4: 0.15
      B5: 0.15
"""
        )

        curriculum = CurriculumLoader(str(curriculum_yaml))
        ok, errors = curriculum.load()
        assert ok, f"Failed to load minimal curriculum: {errors}"

        config = PipelineConfig()
        engine = SelectionEngine(config, curriculum)

        # Register chunks (no token ids available; engine uses placeholder for scoring)
        chunks_list = [(cid, meta, None) for cid, meta in all_chunks.items()]
        engine.register_chunks(chunks_list)

        selected, stats = engine.select_for_stage(
            all_chunks, "1B", protected_slices=None
        )

        # Validate selection: selected should be subset of all_chunks and stats consistent
        assert isinstance(selected, set)
        assert selected.issubset(set(all_chunks.keys()))
        assert "selected_tokens" in stats

    def test_selection_using_large_sample(self):
        """If a large sample file exists, reservoir-sample 200 rows and run selection."""
        import json
        import random

        from src.core.config import PipelineConfig
        from src.core.types import ChunkMetadata, DifficultyBand
        from src.curriculum.loader import CurriculumLoader
        from src.selection.engine import SelectionEngine

        large_path = Path("data/datasets/large_sample_chunks.jsonl")
        if not large_path.exists():
            import pytest

            pytest.skip("Large sample file not present; skip heavy integration test")

        # Reservoir sample 1000 lines for scale testing (fixed O(n³) scoring bottleneck)
        k = 1000
        reservoir = []
        with open(large_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < k:
                    reservoir.append(line)
                else:
                    j = random.randint(0, i)
                    if j < k:
                        reservoir[j] = line

        # Build all_chunks from sampled lines
        all_chunks = {}
        for line in reservoir:
            data = json.loads(line)
            cid = data.get("chunk_id")
            meta = ChunkMetadata(
                chunk_id=cid,
                dataset_id=data.get("dataset_id", "ds"),
                token_count=int(data.get("token_count_estimate", 0)),
                byte_length=int(data.get("byte_length", 0)),
                domain=data.get("domain", "clean_web"),
                language=data.get("language", "en"),
                band=DifficultyBand(data.get("band", "B0")),
                source_doc_id=data.get("source_doc_id", ""),
                source_url=data.get("source_url", None),
            )
            if "token_ids" in data:
                setattr(meta, "token_ids", list(data["token_ids"]))
            all_chunks[cid] = meta

        # Minimal curriculum for 1B
        curriculum_yaml = Path("data/datasets/curriculum_min_for_large_test.yaml")
        curriculum_yaml.write_text(
            """
version: "0.0.1"
status: "FROZEN"
stages:
  "1B":
    total_tokens: 20000000000
    band_ratios:
      B0: 0.10
      B1: 0.15
      B2: 0.20
      B3: 0.25
      B4: 0.15
      B5: 0.15
"""
        )

        curriculum = CurriculumLoader(str(curriculum_yaml))
        ok, errors = curriculum.load()
        assert ok, f"Failed to load minimal curriculum: {errors}"

        config = PipelineConfig()
        config.dedup.enable_near_dedup = (
            False  # Disable to avoid O(n^2) pairwise similarity
        )
        engine = SelectionEngine(config, curriculum)

        chunks_list = [(cid, meta, None) for cid, meta in all_chunks.items()]
        engine.register_chunks(chunks_list)

        selected, stats = engine.select_for_stage(
            all_chunks, "1B", protected_slices=None
        )

        assert isinstance(selected, set)
        assert selected.issubset(set(all_chunks.keys()))
        assert "selected_tokens" in stats


# Fixtures
@pytest.fixture
def temp_config(tmp_path):
    """Create temporary config file"""
    config = PipelineConfig()
    config_path = tmp_path / "config.yaml"
    config.save_to_file(str(config_path), format="yaml")
    return config_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

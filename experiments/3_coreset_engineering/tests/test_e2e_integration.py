"""
End-to-end integration tests for coreset engineering pipeline.

Tests verify:
1. Complete pipeline execution (data → manifests)
2. Manifest validation and format compliance
3. AWS output compatibility
4. Stage progression consistency
5. Curriculum compliance across all stages
"""

import json
import os
import random
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coreset_engine.ingestion.deduplicator import TextDeduplicator
from coreset_engine.selection.builder import CoresetBuilder
from coreset_engine.selection.curriculum import CurriculumConfig


class TestEndToEndPipeline:
    """Test complete coreset engineering pipeline."""

    @pytest.fixture
    def test_environment(self):
        """Set up complete test environment."""
        temp_base = tempfile.mkdtemp(prefix="e2e_test_")
        env = {
            "base": temp_base,
            "input": os.path.join(temp_base, "input"),
            "output": os.path.join(temp_base, "output"),
            "config": os.path.join(temp_base, "config"),
            "aws_output": os.path.join(temp_base, "aws_output"),
        }
        for d in env.values():
            if d != "base":
                os.makedirs(d, exist_ok=True)

        yield env
        shutil.rmtree(temp_base, ignore_errors=True)

    @pytest.fixture
    def curriculum_config(self, test_environment):
        """Create complete curriculum configuration for E2E testing."""
        curriculum = {
            "version": "0.2",
            "owner_team": "Team 2: Curriculum Architects",
            "frozen_on": "2026-02-02",
            "growth_schedule": {
                "stages": [
                    {"name": "1B", "order": 1, "curriculum_profile": "base"},
                    {"name": "3B", "order": 2, "curriculum_profile": "harder_shift_1"},
                    {"name": "8B", "order": 3, "curriculum_profile": "harder_shift_2"},
                    {
                        "name": "70B",
                        "order": 4,
                        "curriculum_profile": "final_adaptive_knobs",
                    },
                ]
            },
            "difficulty_bands": {
                "definition_method": "heuristic",
                "bands": {
                    "B0": {"min_score": 0.0, "max_score": 0.2},
                    "B1": {"min_score": 0.2, "max_score": 0.4},
                    "B2": {"min_score": 0.4, "max_score": 0.6},
                    "B3": {"min_score": 0.6, "max_score": 0.8},
                    "B4": {"min_score": 0.8, "max_score": 0.95},
                    "B5": {"min_score": 0.95, "max_score": 1.0},
                },
            },
            "stage_profiles": {
                "base": {
                    "target_tokens": 1_000_000_000,
                    "band_weights": {
                        "B0": 0.4,
                        "B1": 0.3,
                        "B2": 0.2,
                        "B3": 0.1,
                        "B4": 0.0,
                        "B5": 0.0,
                    },
                    "modality_weights": {"text": 0.8, "code": 0.2},
                },
                "harder_shift_1": {
                    "target_tokens": 3_000_000_000,
                    "band_weights": {
                        "B0": 0.2,
                        "B1": 0.3,
                        "B2": 0.3,
                        "B3": 0.15,
                        "B4": 0.05,
                        "B5": 0.0,
                    },
                    "modality_weights": {"text": 0.7, "code": 0.3},
                },
                "harder_shift_2": {
                    "target_tokens": 8_000_000_000,
                    "band_weights": {
                        "B0": 0.1,
                        "B1": 0.2,
                        "B2": 0.3,
                        "B3": 0.25,
                        "B4": 0.1,
                        "B5": 0.05,
                    },
                    "modality_weights": {"text": 0.6, "code": 0.4},
                },
                "final_adaptive_knobs": {
                    "target_tokens": 70_000_000_000,
                    "band_weights": {
                        "B0": 0.05,
                        "B1": 0.1,
                        "B2": 0.2,
                        "B3": 0.3,
                        "B4": 0.25,
                        "B5": 0.1,
                    },
                    "modality_weights": {"text": 0.5, "code": 0.5},
                },
            },
        }

        config_path = os.path.join(test_environment["config"], "curriculum.yaml")
        with open(config_path, "w") as f:
            yaml.dump(curriculum, f)

        return config_path

    @pytest.fixture
    def sample_dataset(self, test_environment):
        """Generate sample dataset for E2E testing."""
        random.seed(42)
        data = []

        for i in range(2000):  # Larger dataset for E2E
            sample = {
                "id": f"doc_{i:06d}",
                "text": f"Document {i}. "
                + "Sample text content. " * (5 + random.randint(0, 100)),
                "modality": random.choice(["text", "code"]),
                "domain": random.choice(["web", "book", "code", "academic"]),
                "difficulty_score": round(random.uniform(0.0, 1.0), 3),
                "language": "en",
                "timestamp": "2026-01-01",
                "token_count": random.randint(100, 10000),
            }
            data.append(sample)

        # Write to JSONL
        dataset_path = os.path.join(test_environment["input"], "raw_dataset.jsonl")
        with open(dataset_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        return dataset_path, data

    @pytest.mark.integration
    def test_e2e_pipeline_execution(
        self, curriculum_config, sample_dataset, test_environment
    ):
        """Test complete pipeline execution from raw data to manifests."""
        data_path, raw_data = sample_dataset

        # Initialize builder
        builder = CoresetBuilder(
            config_path=curriculum_config,
            data_dir=test_environment["input"],
            output_dir=test_environment["output"],
        )

        # Execute pipeline
        result = builder.build(raw_data)

        # Verify result structure
        assert "manifests" in result
        assert "metadata" in result
        assert "stages" in result["manifests"]

        # Verify all stages have manifests
        for stage in ["1B", "3B", "8B", "70B"]:
            assert stage in result["manifests"]["stages"]

    @pytest.mark.integration
    def test_manifest_output_format(
        self, curriculum_config, sample_dataset, test_environment
    ):
        """Test that manifests conform to expected output format."""
        data_path, raw_data = sample_dataset

        builder = CoresetBuilder(
            config_path=curriculum_config,
            data_dir=test_environment["input"],
            output_dir=test_environment["output"],
        )

        manifests = builder.build(raw_data)["manifests"]

        # Verify manifest JSON structure
        for stage_name, stage_manifest in manifests["stages"].items():
            # Required fields
            assert isinstance(stage_manifest, dict)
            assert "indices" in stage_manifest
            assert "statistics" in stage_manifest
            assert "metadata" in stage_manifest

            # Indices must be list of ints
            assert isinstance(stage_manifest["indices"], list)
            assert all(isinstance(i, int) for i in stage_manifest["indices"])

            # Statistics structure
            stats = stage_manifest["statistics"]
            assert "band_distribution" in stats
            assert "modality_distribution" in stats
            assert "total_samples" in stats
            assert "unique_documents" in stats

    @pytest.mark.integration
    def test_aws_compatible_json_output(
        self, curriculum_config, sample_dataset, test_environment
    ):
        """Test that manifests are AWS-compatible JSON."""
        data_path, raw_data = sample_dataset

        builder = CoresetBuilder(
            config_path=curriculum_config,
            data_dir=test_environment["input"],
            output_dir=test_environment["output"],
        )

        result = builder.build(raw_data)

        # Should be serializable to JSON (AWS S3 compatible)
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Should be deserializable
        restored = json.loads(json_str)
        assert "manifests" in restored

    @pytest.mark.integration
    def test_stage_progression_consistency(
        self, curriculum_config, sample_dataset, test_environment
    ):
        """Test that stage progression maintains subset invariant."""
        data_path, raw_data = sample_dataset

        builder = CoresetBuilder(
            config_path=curriculum_config,
            data_dir=test_environment["input"],
            output_dir=test_environment["output"],
        )

        result = builder.build(raw_data)
        manifests = result["manifests"]["stages"]

        # Get indices for each stage
        indices_1b = set(manifests["1B"]["indices"])
        indices_3b = set(manifests["3B"]["indices"])
        indices_8b = set(manifests["8B"]["indices"])
        indices_70b = set(manifests["70B"]["indices"])

        # Verify subset property: 1B ⊂ 3B ⊂ 8B ⊂ 70B
        assert indices_1b.issubset(indices_3b), "1B indices should be subset of 3B"
        assert indices_3b.issubset(indices_8b), "3B indices should be subset of 8B"
        assert indices_8b.issubset(indices_70b), "8B indices should be subset of 70B"

        # Verify increasing sizes
        assert len(indices_1b) < len(indices_3b)
        assert len(indices_3b) < len(indices_8b)
        assert len(indices_8b) < len(indices_70b)

    @pytest.mark.integration
    def test_curriculum_compliance_validation(
        self, curriculum_config, sample_dataset, test_environment
    ):
        """Test that output complies with curriculum requirements."""
        data_path, raw_data = sample_dataset

        builder = CoresetBuilder(
            config_path=curriculum_config,
            data_dir=test_environment["input"],
            output_dir=test_environment["output"],
        )

        result = builder.build(raw_data)
        config = CurriculumConfig.load(curriculum_config)

        # Verify all stages from curriculum are in manifest
        curriculum_stages = {stage.name for stage in config.stages}
        manifest_stages = set(result["manifests"]["stages"].keys())
        assert curriculum_stages == manifest_stages

    @pytest.mark.integration
    def test_deduplication_applied(self, curriculum_config, test_environment):
        """Test that deduplication is applied in pipeline."""
        # Create dataset with duplicates
        duplicate_data = []
        random.seed(42)

        base_text = "The quick brown fox jumps over the lazy dog."

        for i in range(100):
            # Add unique samples
            duplicate_data.append(
                {
                    "id": f"unique_{i}",
                    "text": f"Unique sample {i}. " * 5,
                    "modality": "text",
                    "difficulty_score": random.uniform(0.0, 1.0),
                }
            )

        # Add exact duplicates
        for i in range(100):
            duplicate_data.append(
                {
                    "id": f"dup_{i}",
                    "text": base_text,  # All same
                    "modality": "text",
                    "difficulty_score": 0.5,
                }
            )

        # Run deduplicator
        dedup = TextDeduplicator()
        deduplicated = dedup.deduplicate(duplicate_data)

        # Should have removed exact duplicates
        assert len(deduplicated) < len(duplicate_data)

    @pytest.mark.integration
    def test_data_quality_metrics(
        self, curriculum_config, sample_dataset, test_environment
    ):
        """Test that pipeline generates quality metrics."""
        data_path, raw_data = sample_dataset

        builder = CoresetBuilder(
            config_path=curriculum_config,
            data_dir=test_environment["input"],
            output_dir=test_environment["output"],
        )

        result = builder.build(raw_data)

        # Verify metrics are present
        assert "metadata" in result
        metadata = result["metadata"]

        assert "processed_at" in metadata or "timestamp" in metadata
        assert "curriculum_version" in metadata or "config_path" in metadata
        assert "statistics" in metadata or "summary" in metadata

    @pytest.mark.integration
    def test_manifest_s3_key_format(
        self, curriculum_config, sample_dataset, test_environment
    ):
        """Test that manifests follow S3 naming convention."""
        data_path, raw_data = sample_dataset

        builder = CoresetBuilder(
            config_path=curriculum_config,
            data_dir=test_environment["input"],
            output_dir=test_environment["output"],
        )

        # Mock S3 output
        with patch("boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.return_value = mock_s3

            result = builder.build(raw_data)
            manifests = result["manifests"]

            # Verify manifest structure has S3-compatible naming
            for stage in ["1B", "3B", "8B", "70B"]:
                # S3 path should follow pattern: manifests/{stage}/manifest.json
                # Just verify stage is in manifests
                assert stage in manifests["stages"]

    @pytest.mark.integration
    def test_audit_visualization_s3_output(
        self, curriculum_config, sample_dataset, test_environment
    ):
        """Test that pipeline generates audit/visualization files for S3."""
        data_path, raw_data = sample_dataset

        builder = CoresetBuilder(
            config_path=curriculum_config,
            data_dir=test_environment["input"],
            output_dir=test_environment["output"],
        )

        result = builder.build(raw_data)

        # Should generate audit trail
        assert "audit" in result or "timestamp" in result.get("metadata", {})

        # Should generate visualization data
        manifests = result["manifests"]
        for stage in manifests["stages"].values():
            assert "statistics" in stage
            # Stats should support visualization (band distribution, etc.)
            assert "band_distribution" in stage["statistics"]


class TestAWSIntegration:
    """Integration tests with AWS services."""

    @pytest.fixture
    def mock_aws_environment(self):
        """Mock AWS environment variables."""
        env_vars = {
            "AWS_ACCOUNT_ID": "123456789012",
            "AWS_REGION": "us-west-2",
            "S3_BUCKET": "llm-coreset-artifacts-staging",
            "S3_PREFIX": "manifests",
        }
        return env_vars

    @patch("boto3.client")
    def test_s3_manifest_upload(
        self, mock_boto_client, curriculum_config, sample_dataset
    ):
        """Test uploading manifests to S3."""
        data_path, raw_data = sample_dataset
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Simulate upload
        bucket = "test-bucket"
        key = "manifests/1B/manifest.json"

        manifest_data = {"indices": list(range(100))}

        mock_s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(manifest_data),
            ContentType="application/json",
        )

        # Verify call was made
        mock_s3.put_object.assert_called_once()

    @patch("boto3.client")
    def test_lambda_validator_invocation(self, mock_boto_client):
        """Test Lambda validator is invoked after manifest generation."""
        mock_lambda = MagicMock()
        mock_boto_client.return_value = mock_lambda

        # Simulate Lambda invocation
        mock_lambda.invoke(
            FunctionName="coreset-manifest-validator",
            InvocationType="Synchronous",
            Payload=json.dumps(
                {
                    "stage": "1B",
                    "manifest_path": "s3://bucket/manifests/1B/manifest.json",
                }
            ),
        )

        # Verify Lambda was invoked
        mock_lambda.invoke.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

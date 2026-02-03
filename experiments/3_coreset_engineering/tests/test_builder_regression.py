"""
Regression tests for coreset_engine.selection.builder module.

Tests verify:
1. Pipeline correctness across curriculum stages
2. Manifest generation consistency
3. Stratified sampling adherence
4. Deduplication accuracy
5. End-to-end workflow stability
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import pytest
import yaml

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from coreset_engine.selection.builder import CoresetBuilder
from coreset_engine.selection.curriculum import CurriculumConfig
from coreset_engine.selection.bucketer import DifficultyBucketer
from coreset_engine.selection.sampler import StratifiedSampler


class TestCoresetBuilderRegressions:
    """Test suite for coreset builder stability and correctness."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for test data."""
        temp_base = tempfile.mkdtemp(prefix="coreset_test_")
        dirs = {
            'input': os.path.join(temp_base, 'input'),
            'output': os.path.join(temp_base, 'output'),
            'config': os.path.join(temp_base, 'config'),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        yield dirs
        shutil.rmtree(temp_base)

    @pytest.fixture
    def mock_curriculum_yaml(self, temp_dirs):
        """Generate a minimal valid curriculum.yaml for testing."""
        curriculum = {
            'version': '0.2',
            'owner_team': 'Team 2: Curriculum Architects',
            'frozen_on': '2026-02-02',
            'growth_schedule': {
                'stages': [
                    {'name': '1B', 'order': 1, 'curriculum_profile': 'base'},
                    {'name': '3B', 'order': 2, 'curriculum_profile': 'harder_shift_1'},
                    {'name': '8B', 'order': 3, 'curriculum_profile': 'harder_shift_2'},
                    {'name': '70B', 'order': 4, 'curriculum_profile': 'final_adaptive_knobs'},
                ]
            },
            'difficulty_bands': {
                'definition_method': 'heuristic',
                'bands': {
                    'B0': {'min_score': 0.0, 'max_score': 0.2},
                    'B1': {'min_score': 0.2, 'max_score': 0.4},
                    'B2': {'min_score': 0.4, 'max_score': 0.6},
                    'B3': {'min_score': 0.6, 'max_score': 0.8},
                    'B4': {'min_score': 0.8, 'max_score': 0.95},
                    'B5': {'min_score': 0.95, 'max_score': 1.0},
                }
            },
            'stage_profiles': {
                'base': {
                    'target_tokens': 1_000_000_000,
                    'band_weights': {'B0': 0.4, 'B1': 0.3, 'B2': 0.2, 'B3': 0.1, 'B4': 0.0, 'B5': 0.0},
                    'modality_weights': {'text': 0.8, 'code': 0.2}
                },
                'harder_shift_1': {
                    'target_tokens': 3_000_000_000,
                    'band_weights': {'B0': 0.2, 'B1': 0.3, 'B2': 0.3, 'B3': 0.15, 'B4': 0.05, 'B5': 0.0},
                    'modality_weights': {'text': 0.7, 'code': 0.3}
                },
                'harder_shift_2': {
                    'target_tokens': 8_000_000_000,
                    'band_weights': {'B0': 0.1, 'B1': 0.2, 'B2': 0.3, 'B3': 0.25, 'B4': 0.1, 'B5': 0.05},
                    'modality_weights': {'text': 0.6, 'code': 0.4}
                },
                'final_adaptive_knobs': {
                    'target_tokens': 70_000_000_000,
                    'band_weights': {'B0': 0.05, 'B1': 0.1, 'B2': 0.2, 'B3': 0.3, 'B4': 0.25, 'B5': 0.1},
                    'modality_weights': {'text': 0.5, 'code': 0.5}
                }
            }
        }
        config_path = os.path.join(temp_dirs['config'], 'curriculum.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(curriculum, f)
        return config_path

    @pytest.fixture
    def mock_dataset(self, temp_dirs):
        """Generate a mock dataset with diverse samples."""
        import random
        
        data = []
        random.seed(42)
        
        for i in range(1000):
            sample = {
                'id': f'sample_{i}',
                'text': f'Sample text content {i}. ' * (10 + random.randint(0, 50)),
                'modality': random.choice(['text', 'code']),
                'domain': random.choice(['web', 'book', 'code', 'academic']),
                'difficulty_score': round(random.uniform(0.0, 1.0), 3),
                'language': 'en',
                'timestamp': '2026-01-01',
            }
            data.append(sample)
        
        # Create JSONL file
        data_path = os.path.join(temp_dirs['input'], 'dataset.jsonl')
        with open(data_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        return data_path, data

    def test_builder_initialization(self, mock_curriculum_yaml, temp_dirs):
        """Test CoresetBuilder initializes with valid config."""
        builder = CoresetBuilder(
            config_path=mock_curriculum_yaml,
            data_dir=temp_dirs['input'],
            output_dir=temp_dirs['output']
        )
        assert builder is not None
        assert builder.config_path == mock_curriculum_yaml

    def test_curriculum_parsing(self, mock_curriculum_yaml):
        """Test curriculum.yaml parsing and validation."""
        config = CurriculumConfig.load(mock_curriculum_yaml)
        assert config.version == '0.2'
        assert len(config.stages) == 4
        assert config.stages[0].name == '1B'
        assert config.stages[-1].name == '70B'

    def test_dataset_loading_jsonl(self, mock_dataset, temp_dirs):
        """Test loading JSONL dataset."""
        data_path, expected_data = mock_dataset
        
        # Verify JSONL integrity
        loaded = []
        with open(data_path, 'r') as f:
            for line in f:
                loaded.append(json.loads(line))
        
        assert len(loaded) == 1000
        assert all('id' in item for item in loaded)
        assert all('difficulty_score' in item for item in loaded)

    def test_deduplication_stability(self, mock_dataset, temp_dirs):
        """Test that deduplication is deterministic and repeatable."""
        from coreset_engine.ingestion.deduplicator import TextDeduplicator
        
        data_path, data = mock_dataset
        
        # Create two identical datasets
        data_path_1 = os.path.join(temp_dirs['input'], 'dataset_1.jsonl')
        data_path_2 = os.path.join(temp_dirs['input'], 'dataset_2.jsonl')
        
        with open(data_path_1, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        shutil.copy(data_path_1, data_path_2)
        
        # Deduplicate both
        dedup = TextDeduplicator()
        result_1 = dedup.deduplicate(data_path_1)
        result_2 = dedup.deduplicate(data_path_2)
        
        # Results should be identical
        assert len(result_1) == len(result_2)

    def test_difficulty_bucketing(self, mock_dataset, temp_dirs, mock_curriculum_yaml):
        """Test that difficulty bucketing respects band definitions."""
        data_path, data = mock_dataset
        config = CurriculumConfig.load(mock_curriculum_yaml)
        
        bucketer = DifficultyBucketer(config)
        
        # Bucket samples
        bucketed = bucketer.bucket(data)
        
        # Verify all samples are assigned to a band
        assigned_bands = [item.get('difficulty_band') for item in bucketed]
        assert all(band in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5'] for band in assigned_bands)
        
        # Verify band assignment matches difficulty score
        for item in bucketed:
            score = item['difficulty_score']
            band = item['difficulty_band']
            band_def = config.difficulty_bands['bands'][band]
            assert band_def['min_score'] <= score < band_def['max_score']

    def test_stratified_sampling_band_weights(self, mock_dataset, temp_dirs, mock_curriculum_yaml):
        """Test that stratified sampling respects band_weights."""
        data_path, data = mock_dataset
        config = CurriculumConfig.load(mock_curriculum_yaml)
        
        bucketer = DifficultyBucketer(config)
        bucketed = bucketer.bucket(data)
        
        sampler = StratifiedSampler(config)
        
        # Sample for 1B stage
        stage_1b = config.get_stage('1B')
        target_tokens = stage_1b['target_tokens']
        
        sampled = sampler.stratified_sample(bucketed, stage_1b, target_count=500)
        
        # Verify sample composition
        band_counts = {}
        for item in sampled:
            band = item['difficulty_band']
            band_counts[band] = band_counts.get(band, 0) + 1
        
        # Check that band_weights are approximately respected (with tolerance)
        expected_weights = stage_1b['band_weights']
        total_sampled = len(sampled)
        
        for band, weight in expected_weights.items():
            if weight > 0:
                actual_ratio = band_counts.get(band, 0) / total_sampled
                # Allow 20% tolerance in distribution
                assert abs(actual_ratio - weight) < 0.2 * weight

    def test_manifest_generation(self, mock_dataset, temp_dirs, mock_curriculum_yaml):
        """Test that manifests are generated correctly."""
        data_path, data = mock_dataset
        
        builder = CoresetBuilder(
            config_path=mock_curriculum_yaml,
            data_dir=temp_dirs['input'],
            output_dir=temp_dirs['output']
        )
        
        # Note: This assumes build() is implemented
        # If not, test individual components
        manifests = builder.generate_manifests(data)
        
        # Verify manifest structure
        assert 'metadata' in manifests
        assert 'stages' in manifests
        assert all(stage in manifests['stages'] for stage in ['1B', '3B', '8B', '70B'])
        
        # Verify stage manifests have required fields
        for stage_name, stage_manifest in manifests['stages'].items():
            assert 'indices' in stage_manifest
            assert 'statistics' in stage_manifest
            assert 'band_distribution' in stage_manifest['statistics']
            assert 'modality_distribution' in stage_manifest['statistics']

    def test_manifest_reproducibility(self, mock_dataset, temp_dirs, mock_curriculum_yaml):
        """Test that identical inputs produce identical manifests."""
        data_path, data = mock_dataset
        
        builder_1 = CoresetBuilder(
            config_path=mock_curriculum_yaml,
            data_dir=temp_dirs['input'],
            output_dir=os.path.join(temp_dirs['output'], 'run1')
        )
        
        builder_2 = CoresetBuilder(
            config_path=mock_curriculum_yaml,
            data_dir=temp_dirs['input'],
            output_dir=os.path.join(temp_dirs['output'], 'run2')
        )
        
        manifests_1 = builder_1.generate_manifests(data)
        manifests_2 = builder_2.generate_manifests(data)
        
        # Manifests should be identical
        assert json.dumps(manifests_1, sort_keys=True) == json.dumps(manifests_2, sort_keys=True)

    def test_stage_progression_difficulty(self, mock_curriculum_yaml):
        """Test that curriculum shows increasing difficulty across stages."""
        config = CurriculumConfig.load(mock_curriculum_yaml)
        
        profiles = list(config.stage_profiles.values())
        
        # Check that B0 weight decreases and B5 weight increases
        b0_weights = [stage['band_weights']['B0'] for stage in profiles]
        b5_weights = [stage['band_weights']['B5'] for stage in profiles]
        
        assert b0_weights[0] > b0_weights[-1], "B0 weight should decrease from 1B to 70B"
        assert b5_weights[-1] > b5_weights[0], "B5 weight should increase from 1B to 70B"

    def test_config_validation_missing_fields(self, temp_dirs):
        """Test that invalid curriculum config raises appropriate errors."""
        invalid_config = {
            'version': '0.2',
            # Missing required fields
        }
        config_path = os.path.join(temp_dirs['config'], 'invalid.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ValueError):
            CurriculumConfig.load(config_path)

    def test_output_directory_creation(self, temp_dirs, mock_curriculum_yaml):
        """Test that builder creates output directory if it doesn't exist."""
        output_dir = os.path.join(temp_dirs['output'], 'nonexistent', 'nested', 'path')
        
        builder = CoresetBuilder(
            config_path=mock_curriculum_yaml,
            data_dir=temp_dirs['input'],
            output_dir=output_dir
        )
        
        assert os.path.exists(output_dir)

    def test_large_dataset_stability(self, temp_dirs, mock_curriculum_yaml):
        """Regression test: handle large datasets without memory issues."""
        import random
        
        # Generate larger dataset
        large_data = []
        random.seed(42)
        
        for i in range(10000):  # 10x larger
            sample = {
                'id': f'sample_{i}',
                'text': f'Large dataset sample {i}. ' * 100,
                'modality': random.choice(['text', 'code']),
                'difficulty_score': round(random.uniform(0.0, 1.0), 3),
            }
            large_data.append(sample)
        
        data_path = os.path.join(temp_dirs['input'], 'large_dataset.jsonl')
        with open(data_path, 'w') as f:
            for item in large_data:
                f.write(json.dumps(item) + '\n')
        
        builder = CoresetBuilder(
            config_path=mock_curriculum_yaml,
            data_dir=temp_dirs['input'],
            output_dir=temp_dirs['output']
        )
        
        # Should complete without memory errors
        manifests = builder.generate_manifests(large_data)
        assert manifests is not None


class TestCurriculumConfigRegressions:
    """Regression tests for curriculum configuration loading and parsing."""

    @pytest.fixture
    def curriculum_yaml_path(self, tmp_path):
        """Create a test curriculum.yaml file."""
        config = {
            'version': '0.2',
            'owner_team': 'Team 2',
            'frozen_on': '2026-02-02',
            'growth_schedule': {
                'stages': [
                    {'name': '1B', 'order': 1, 'curriculum_profile': 'base'},
                ]
            },
        }
        path = tmp_path / "curriculum.yaml"
        with open(path, 'w') as f:
            yaml.dump(config, f)
        return str(path)

    def test_stage_ordering(self, curriculum_yaml_path):
        """Test that stages are ordered correctly."""
        config = CurriculumConfig.load(curriculum_yaml_path)
        stage_orders = [stage.order for stage in config.stages]
        assert stage_orders == sorted(stage_orders)

    def test_profile_resolution(self, curriculum_yaml_path):
        """Test that curriculum profiles are correctly resolved."""
        config = CurriculumConfig.load(curriculum_yaml_path)
        for stage in config.stages:
            profile = config.get_stage_profile(stage.curriculum_profile)
            assert profile is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

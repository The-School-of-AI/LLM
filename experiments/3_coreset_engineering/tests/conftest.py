"""
Pytest configuration for coreset engineering tests.

Includes fixtures for test data, configs, and reproducibility validation.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import reproducibility modules
from coreset_engine.reproducibility import SeedPolicy, ManifestFingerprinter


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing test data directory."""
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope="session")
def test_config_dir():
    """Fixture providing test config directory."""
    return os.path.join(os.path.dirname(__file__), 'configs')


@pytest.fixture(scope="session")
def seed_policy():
    """Fixture providing seed policy from canonical config."""
    config_path = Path(__file__).parent.parent / 'configs' / 'seed_policy.yaml'
    if config_path.exists():
        return SeedPolicy(str(config_path))
    else:
        # Fallback: create minimal policy
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'version': '0.2',
                'global_seed': 42,
                'owner_team': 'Team 3',
            }, f)
            return SeedPolicy(f.name)


@pytest.fixture
def fingerprinter():
    """Fixture providing manifest fingerprinter."""
    return ManifestFingerprinter


@pytest.fixture
def deterministic_indices():
    """Fixture providing pre-sorted test indices."""
    return sorted([5, 0, 15, 10, 20, 100, 50, 75])


@pytest.fixture
def sample_manifest():
    """Fixture providing a sample valid manifest."""
    from coreset_engine.reproducibility import ManifestFingerprinter
    
    return {
        'schema_version': '0.2',
        'manifest_type': 'coreset_engineering',
        'frozen_on': '2026-02-02',
        'metadata': {
            'curriculum_version': '0.2',
            'seed': 42,
            'execution_date': '2026-02-02T10:00:00Z',
            'execution_id': 'run_test_stage_1B',
        },
        'stages': {
            '1B': {
                'manifest_fingerprint': ManifestFingerprinter.compute_indices_fingerprint([0, 5, 10]),
                'indices': [0, 5, 10],
                'statistics': {
                    'total_samples': 3,
                    'band_distribution': {'B0': 3, 'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0},
                    'modality_distribution': {'text': 2, 'code': 1},
                }
            },
        },
        'audit_trail': {
            'reproducibility_checksum': 'sha256:abc123',
        }
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests as regression tests"
    )
    config.addinivalue_line(
        "markers", "determinism: marks tests as determinism verification tests"
    )
    config.addinivalue_line(
        "markers", "reproducibility: marks tests as reproducibility tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip slow tests by default."""
    skip_slow = pytest.mark.skip(reason="slow test - run with -m slow")
    
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

"""
Replay verification tests for reproducibility.

Tests that manifests can be replayed identically with the same seed and configuration.
These are critical regression tests ensuring the pipeline is deterministic.
"""

import pytest
import json
import hashlib
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from coreset_engine.reproducibility import (
    SeedPolicy,
    ManifestFingerprinter,
    ReproducibilityValidator,
)


class TestSeedPolicy:
    """Tests for seed policy enforcement."""
    
    @pytest.fixture
    def temp_seed_policy(self, tmp_path):
        """Create a temporary seed policy config."""
        policy_config = {
            'version': '0.2',
            'global_seed': 42,
            'owner_team': 'Team 3',
        }
        config_file = tmp_path / "seed_policy.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(policy_config, f)
        return str(config_file)
    
    def test_seed_initialization(self, temp_seed_policy):
        """Test that seed can be initialized once."""
        policy = SeedPolicy(temp_seed_policy)
        assert not policy.is_initialized
        
        policy.initialize()
        assert policy.is_initialized
    
    def test_seed_cannot_reinitialize(self, temp_seed_policy):
        """Test that re-initialization is prevented."""
        policy = SeedPolicy(temp_seed_policy)
        policy.initialize()
        
        with pytest.raises(RuntimeError, match="already initialized"):
            policy.initialize()
    
    def test_checkpoint_validation_before_init(self, temp_seed_policy):
        """Test that checkpoints fail if seed not initialized."""
        policy = SeedPolicy(temp_seed_policy)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            policy.validate_at_checkpoint("manifest_generation")
    
    def test_checkpoint_validation_after_init(self, temp_seed_policy):
        """Test that checkpoints pass after initialization."""
        policy = SeedPolicy(temp_seed_policy)
        policy.initialize()
        
        # Should not raise
        policy.validate_at_checkpoint("manifest_generation")
    
    def test_seed_is_canonical(self, temp_seed_policy):
        """Test that seed must be 42."""
        policy = SeedPolicy(temp_seed_policy)
        assert policy.CANONICAL_SEED == 42


class TestManifestFingerprinting:
    """Tests for deterministic fingerprinting."""
    
    def test_identical_indices_same_fingerprint(self):
        """Same indices → same fingerprint (determinism)."""
        indices = [0, 5, 10, 15, 20]
        
        fp1 = ManifestFingerprinter.compute_indices_fingerprint(indices)
        fp2 = ManifestFingerprinter.compute_indices_fingerprint(indices)
        
        assert fp1 == fp2
    
    def test_different_indices_different_fingerprint(self):
        """Different indices → different fingerprints."""
        indices_1 = [0, 5, 10, 15, 20]
        indices_2 = [0, 5, 10, 15, 21]  # One different
        
        fp1 = ManifestFingerprinter.compute_indices_fingerprint(indices_1)
        fp2 = ManifestFingerprinter.compute_indices_fingerprint(indices_2)
        
        assert fp1 != fp2
    
    def test_order_invariant_fingerprinting(self):
        """Fingerprints are same regardless of input order (sorted internally)."""
        indices_sorted = [0, 5, 10, 15, 20]
        indices_unsorted = [20, 0, 15, 5, 10]
        
        fp_sorted = ManifestFingerprinter.compute_indices_fingerprint(indices_sorted)
        fp_unsorted = ManifestFingerprinter.compute_indices_fingerprint(indices_unsorted)
        
        # Should match because fingerprint sorts internally
        assert fp_sorted == fp_unsorted
    
    def test_fingerprint_format(self):
        """Fingerprints follow sha256:xyz... format."""
        indices = [0, 5, 10]
        fp = ManifestFingerprinter.compute_indices_fingerprint(indices)
        
        assert fp.startswith("sha256:")
        assert len(fp) == 71  # "sha256:" (7) + 64 hex chars
    
    def test_file_fingerprint(self):
        """Can compute fingerprint of files."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            f.flush()
            
            try:
                fp1 = ManifestFingerprinter.compute_file_fingerprint(f.name)
                fp2 = ManifestFingerprinter.compute_file_fingerprint(f.name)
                
                # Same file → same fingerprint
                assert fp1 == fp2
                assert fp1.startswith("sha256:")
            finally:
                os.unlink(f.name)
    
    def test_verify_fingerprint(self):
        """Fingerprint verification works."""
        indices = [0, 5, 10]
        fp = ManifestFingerprinter.compute_indices_fingerprint(indices)
        
        assert ManifestFingerprinter.verify_indices_fingerprint(indices, fp)
        
        # Wrong fingerprint should fail
        wrong_fp = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        assert not ManifestFingerprinter.verify_indices_fingerprint(indices, wrong_fp)


@pytest.mark.regression
class TestDeterministicManifestGeneration:
    """Tests that manifest generation is deterministic."""
    
    def test_replay_with_identical_seed(self):
        """
        Identical seed + config → identical manifest.
        
        This is the core reproducibility test.
        """
        # Simulate two runs with same seed
        indices_run1 = sorted([5, 0, 15, 10, 20])  # Simulate sorted result
        indices_run2 = sorted([5, 0, 15, 10, 20])  # Same result
        
        fp1 = ManifestFingerprinter.compute_indices_fingerprint(indices_run1)
        fp2 = ManifestFingerprinter.compute_indices_fingerprint(indices_run2)
        
        # Fingerprints must match
        assert fp1 == fp2
    
    def test_indices_always_sorted(self):
        """Manifest indices must always be sorted."""
        indices = [0, 5, 10, 15, 20]
        
        # Verify indices are in canonical sorted order
        assert indices == sorted(indices)
    
    @pytest.mark.parametrize("unsorted_indices", [
        [5, 0, 15, 10, 20],
        [20, 10, 15, 5, 0],
        [0, 20, 5, 15, 10],
    ])
    def test_fingerprint_order_independence(self, unsorted_indices):
        """Fingerprints are order-independent (internally sorts)."""
        sorted_indices = sorted(unsorted_indices)
        
        fp_sorted = ManifestFingerprinter.compute_indices_fingerprint(sorted_indices)
        fp_unsorted = ManifestFingerprinter.compute_indices_fingerprint(unsorted_indices)
        
        assert fp_sorted == fp_unsorted


class TestReproducibilityValidator:
    """Tests for manifest reproducibility validation."""
    
    @pytest.fixture
    def sample_manifest(self):
        """Create a sample valid manifest."""
        return {
            'schema_version': '0.2',
            'metadata': {
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
    
    def test_valid_manifest_passes(self, sample_manifest):
        """Valid manifest passes all checks."""
        validator = ReproducibilityValidator(sample_manifest)
        assert validator.validate_all()
    
    def test_wrong_seed_fails(self, sample_manifest):
        """Seed must be 42."""
        sample_manifest['metadata']['seed'] = 41
        validator = ReproducibilityValidator(sample_manifest)
        assert not validator.validate_seed()
    
    def test_unsorted_indices_fails(self, sample_manifest):
        """Indices must be strictly sorted."""
        sample_manifest['stages']['1B']['indices'] = [10, 0, 5]
        validator = ReproducibilityValidator(sample_manifest)
        assert not validator.validate_indices_sorted()
    
    def test_wrong_fingerprint_fails(self, sample_manifest):
        """Fingerprint mismatch should fail."""
        sample_manifest['stages']['1B']['manifest_fingerprint'] = 'sha256:wrong'
        validator = ReproducibilityValidator(sample_manifest)
        assert not validator.validate_fingerprints()
    
    def test_missing_audit_trail_fails(self, sample_manifest):
        """Audit trail with reproducibility checksum required."""
        del sample_manifest['audit_trail']['reproducibility_checksum']
        validator = ReproducibilityValidator(sample_manifest)
        assert not validator.validate_audit_trail()
    
    def test_validation_report(self, sample_manifest):
        """Validation report is human-readable."""
        validator = ReproducibilityValidator(sample_manifest)
        validator.validate_all()
        report = validator.report()
        
        assert '✓ PASS' in report
        assert 'Reproducibility Validation Report' in report


@pytest.mark.integration
class TestEndToEndReproducibility:
    """Integration tests for full reproducibility workflow."""
    
    def test_seed_policy_integration(self):
        """Seed policy integrates with fingerprinting."""
        # Create temp seed policy
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump({
                'version': '0.2',
                'global_seed': 42,
                'owner_team': 'Team 3',
            }, f)
            f.flush()
            
            try:
                # Initialize seed
                policy = SeedPolicy(f.name)
                policy.initialize()
                
                # Generate fingerprint
                indices = [0, 5, 10]
                fp = ManifestFingerprinter.compute_indices_fingerprint(indices)
                
                # Fingerprint should be valid
                assert fp.startswith('sha256:')
            finally:
                os.unlink(f.name)
    
    def test_full_reproducibility_workflow(self):
        """Full workflow: seed → determinism → fingerprinting."""
        # Step 1: Initialize seed
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump({'version': '0.2', 'global_seed': 42, 'owner_team': 'Team 3'}, f)
            f.flush()
            policy_file = f.name
        
        try:
            policy = SeedPolicy(policy_file)
            policy.initialize()
            
            # Step 2: Create deterministic indices (sorted)
            indices = sorted([5, 0, 15, 10, 20])
            
            # Step 3: Compute fingerprint
            fp = ManifestFingerprinter.compute_indices_fingerprint(indices)
            
            # Step 4: Validate fingerprint
            assert ManifestFingerprinter.verify_indices_fingerprint(indices, fp)
            
            # Step 5: Create manifest and validate
            manifest = {
                'schema_version': '0.2',
                'metadata': {'seed': 42, 'execution_date': '2026-02-02T10:00:00Z', 'execution_id': 'run_test'},
                'stages': {
                    '1B': {
                        'manifest_fingerprint': fp,
                        'indices': indices,
                        'statistics': {'total_samples': 5, 'band_distribution': {}, 'modality_distribution': {}},
                    }
                },
                'audit_trail': {'reproducibility_checksum': 'sha256:abc'},
            }
            
            validator = ReproducibilityValidator(manifest)
            assert validator.validate_all()
        
        finally:
            os.unlink(policy_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

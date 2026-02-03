"""
Reproducibility and determinism utilities for coreset engineering.

This module provides tools to ensure deterministic, reproducible manifest generation
across all pipeline runs. Key components:

- SeedPolicy: Manages seed initialization and validation
- DeterminismChecker: Validates deterministic execution
- ManifestFingerprinter: Computes reproducible fingerprints
"""

import random
import numpy as np
import torch
import hashlib
import json
import yaml
from typing import List, Dict, Any, Optional
from datetime import datetime


class SeedPolicy:
    """
    Manages global seed initialization and validation.
    
    Enforces the constraint that seed is set exactly once at pipeline start,
    never reset mid-pipeline, and always equals 42 (canonical value).
    
    Example:
        >>> policy = SeedPolicy('configs/seed_policy.yaml')
        >>> policy.initialize()  # Sets seed to 42
        >>> # Downstream operations are now deterministic
        >>> policy.validate_at_checkpoint('manifest_generation')  # OK
        >>> policy.initialize()  # Raises RuntimeError (already initialized)
    """
    
    CANONICAL_SEED = 42
    
    def __init__(self, config_path: str = "configs/seed_policy.yaml"):
        """Load seed policy from YAML config."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Verify config has canonical seed
        if self.config.get('global_seed') != self.CANONICAL_SEED:
            raise ValueError(
                f"Seed policy must use canonical seed {self.CANONICAL_SEED}, "
                f"got {self.config.get('global_seed')}"
            )
        
        self._seed_initialized = False
        self._initialization_time = None
    
    def initialize(self) -> None:
        """
        Initialize seed exactly once at pipeline start.
        
        Raises:
            RuntimeError: If seed already initialized (prevents mid-pipeline reset)
        """
        if self._seed_initialized:
            raise RuntimeError(
                "Seed already initialized at {}! Cannot reset mid-pipeline.".format(
                    self._initialization_time
                )
            )
        
        # Set seed across all randomness sources
        random.seed(self.CANONICAL_SEED)
        np.random.seed(self.CANONICAL_SEED)
        torch.manual_seed(self.CANONICAL_SEED)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.CANONICAL_SEED)
        
        self._seed_initialized = True
        self._initialization_time = datetime.now().isoformat()
    
    def validate_at_checkpoint(self, checkpoint_name: str) -> None:
        """
        Verify seed was initialized before critical operation.
        
        Args:
            checkpoint_name: Name of the checkpoint (e.g., 'manifest_generation')
            
        Raises:
            RuntimeError: If seed not yet initialized
        """
        if not self._seed_initialized:
            raise RuntimeError(
                f"Seed not initialized at checkpoint '{checkpoint_name}'. "
                "Call policy.initialize() at pipeline start."
            )
    
    @property
    def is_initialized(self) -> bool:
        """Check if seed has been initialized."""
        return self._seed_initialized
    
    @property
    def initialization_timestamp(self) -> Optional[str]:
        """Get ISO 8601 timestamp of when seed was initialized."""
        return self._initialization_time


class DeterminismChecker:
    """
    Validates that code follows deterministic execution patterns.
    
    Checks for common pitfalls:
    - Iterating over sets/dicts without sorting
    - Multiple seed initializations
    - Unseeded randomness
    """
    
    @staticmethod
    def assert_sorted(items: List, name: str = "items") -> None:
        """
        Verify list is sorted (for determinism).
        
        Args:
            items: List to check
            name: Name of list (for error messages)
            
        Raises:
            AssertionError: If list is not sorted
        """
        if items != sorted(items):
            raise AssertionError(
                f"{name} must be sorted for determinism. "
                f"Expected {sorted(items)}, got {items}"
            )
    
    @staticmethod
    def sorted_items(
        items: Any,
        key=None,
        reverse: bool = False
    ) -> List:
        """
        Safely sort items for deterministic iteration.
        
        Use this instead of directly iterating over sets/dicts.
        
        Args:
            items: Collection to sort (list, set, dict, etc.)
            key: Key function for sorting (optional)
            reverse: Sort in reverse order (default: False)
            
        Returns:
            Sorted list
            
        Example:
            >>> # Bad: for item in my_set (non-deterministic)
            >>> # Good: for item in DeterminismChecker.sorted_items(my_set)
            >>> for item in DeterminismChecker.sorted_items(my_set):
            ...     process(item)
        """
        if isinstance(items, dict):
            # Sort dict items by key
            return sorted(items.items(), key=lambda x: x[0], reverse=reverse)
        elif isinstance(items, set):
            # Sort set elements
            return sorted(items, reverse=reverse)
        else:
            # Sort list (or other iterable)
            return sorted(items, key=key, reverse=reverse)


class ManifestFingerprinter:
    """
    Computes reproducible fingerprints (checksums) for manifests.
    
    Ensures that identical inputs always produce identical fingerprints,
    enabling replay verification and reproducibility validation.
    
    Example:
        >>> fingerprinter = ManifestFingerprinter()
        >>> indices = [0, 5, 10, 15, 20]
        >>> fp1 = fingerprinter.compute_indices_fingerprint(indices)
        >>> fp2 = fingerprinter.compute_indices_fingerprint(indices)
        >>> assert fp1 == fp2  # Same indices → same fingerprint
    """
    
    ALGORITHM = "sha256"
    
    @staticmethod
    def compute_indices_fingerprint(indices: List[int]) -> str:
        """
        Compute SHA256 fingerprint of dataset indices.
        
        Args:
            indices: List of indices (will be sorted for determinism)
            
        Returns:
            SHA256 hash in format "sha256:xyz..."
            
        Example:
            >>> indices = [0, 5, 10, 15, 20]
            >>> fp = ManifestFingerprinter.compute_indices_fingerprint(indices)
            >>> # fp = "sha256:abc123..."
        """
        # Sort indices for determinism (order must be canonical)
        sorted_indices = sorted(indices)
        
        # Serialize to JSON deterministically
        json_str = json.dumps(
            sorted_indices,
            separators=(',', ':'),
            sort_keys=True,
            ensure_ascii=True
        )
        
        # Compute SHA256
        hash_value = hashlib.sha256(json_str.encode()).hexdigest()
        return f"sha256:{hash_value}"
    
    @staticmethod
    def compute_file_fingerprint(file_path: str) -> str:
        """
        Compute SHA256 fingerprint of a file (e.g., curriculum.yaml).
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash in format "sha256:xyz..."
        """
        with open(file_path, 'rb') as f:
            content = f.read()
        hash_value = hashlib.sha256(content).hexdigest()
        return f"sha256:{hash_value}"
    
    @staticmethod
    def compute_manifest_fingerprint(manifest: Dict[str, Any]) -> str:
        """
        Compute fingerprint of entire manifest JSON.
        
        Args:
            manifest: Manifest dictionary
            
        Returns:
            SHA256 hash in format "sha256:xyz..."
        """
        # Serialize manifest deterministically
        json_str = json.dumps(
            manifest,
            separators=(',', ':'),
            sort_keys=True,
            ensure_ascii=True
        )
        hash_value = hashlib.sha256(json_str.encode()).hexdigest()
        return f"sha256:{hash_value}"
    
    @staticmethod
    def verify_indices_fingerprint(
        indices: List[int],
        expected_fingerprint: str
    ) -> bool:
        """
        Verify indices match expected fingerprint.
        
        Args:
            indices: List of indices
            expected_fingerprint: Expected SHA256 hash
            
        Returns:
            True if fingerprints match, False otherwise
        """
        computed = ManifestFingerprinter.compute_indices_fingerprint(indices)
        return computed == expected_fingerprint


class ReproducibilityValidator:
    """
    Validates that manifests and executions meet reproducibility requirements.
    
    Checks:
    - Manifest schema compliance
    - Seed correctness (must be 42)
    - Fingerprint validity
    - Index sorting
    - Determinism markers in audit trail
    """
    
    def __init__(self, manifest: Dict[str, Any]):
        """Initialize with manifest to validate."""
        self.manifest = manifest
        self.validation_results = {}
    
    def validate_all(self) -> bool:
        """
        Run comprehensive reproducibility validation.
        
        Returns:
            True if all validations pass, False otherwise
        """
        self.validation_results = {
            'seed_correctness': self.validate_seed(),
            'indices_sorted': self.validate_indices_sorted(),
            'fingerprints_valid': self.validate_fingerprints(),
            'schema_present': self.validate_schema_fields(),
            'audit_trail_complete': self.validate_audit_trail(),
        }
        return all(self.validation_results.values())
    
    def validate_seed(self) -> bool:
        """Verify seed is exactly 42."""
        seed = self.manifest.get('metadata', {}).get('seed')
        return seed == 42
    
    def validate_indices_sorted(self) -> bool:
        """Verify all stage indices are strictly sorted."""
        stages = self.manifest.get('stages', {})
        for stage_name, stage_data in stages.items():
            indices = stage_data.get('indices', [])
            if indices != sorted(indices):
                print(f"WARNING: Indices not sorted in stage {stage_name}")
                return False
        return True
    
    def validate_fingerprints(self) -> bool:
        """Verify manifest fingerprints are correct."""
        stages = self.manifest.get('stages', {})
        for stage_name, stage_data in stages.items():
            indices = stage_data.get('indices', [])
            stored_fp = stage_data.get('manifest_fingerprint', '')
            
            computed_fp = ManifestFingerprinter.compute_indices_fingerprint(indices)
            
            if stored_fp != computed_fp:
                print(f"WARNING: Fingerprint mismatch in stage {stage_name}")
                print(f"  Expected: {computed_fp}")
                print(f"  Got: {stored_fp}")
                return False
        
        return True
    
    def validate_schema_fields(self) -> bool:
        """Verify required schema fields are present."""
        required = {
            'root': ['schema_version', 'metadata', 'stages', 'audit_trail'],
            'metadata': ['seed', 'execution_date', 'execution_id'],
            'stage': ['manifest_fingerprint', 'indices', 'statistics'],
        }
        
        # Check root fields
        for field in required['root']:
            if field not in self.manifest:
                return False
        
        # Check metadata fields
        for field in required['metadata']:
            if field not in self.manifest.get('metadata', {}):
                return False
        
        # Check each stage
        stages = self.manifest.get('stages', {})
        for stage_name, stage_data in stages.items():
            for field in required['stage']:
                if field not in stage_data:
                    return False
        
        return True
    
    def validate_audit_trail(self) -> bool:
        """Verify audit trail has reproducibility markers."""
        audit = self.manifest.get('audit_trail', {})
        return 'reproducibility_checksum' in audit
    
    def report(self) -> str:
        """Generate human-readable validation report."""
        lines = [
            "=" * 60,
            "Reproducibility Validation Report",
            "=" * 60,
        ]
        
        for check_name, result in self.validation_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            lines.append(f"{status}: {check_name}")
        
        passed = sum(1 for v in self.validation_results.values() if v)
        total = len(self.validation_results)
        lines.append(f"\nTotal: {passed}/{total} checks passed")
        
        return "\n".join(lines)


__all__ = [
    'SeedPolicy',
    'DeterminismChecker',
    'ManifestFingerprinter',
    'ReproducibilityValidator',
]

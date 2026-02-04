"""Manifest generation for reproducibility and auditing."""

import json
import hashlib
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime


class ManifestGenerator:
    """Generates reproducible manifests for coreset outputs."""
    
    def __init__(self, stage_name: str):
        """Initialize manifest generator."""
        self.stage_name = stage_name
        
    def generate(
        self,
        selected_indices: List[int],
        metadata: List[Dict[str, Any]],
        config: Dict[str, Any],
        seed: int
    ) -> Dict[str, Any]:
        """
        Generate manifest for a coreset.
        
        Returns:
            Manifest dictionary
        """
        # Calculate statistics
        total_tokens = sum(m.get("token_count", 0) for m in metadata)
        
        # Band distribution
        band_counts = {}
        domain_counts = {}
        
        for m in metadata:
            band = m.get("band", "unknown")
            domain = m.get("domain", "unknown")
            band_counts[band] = band_counts.get(band, 0) + m.get("token_count", 0)
            domain_counts[domain] = domain_counts.get(domain, 0) + m.get("token_count", 0)
        
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        
        manifest = {
            "stage_name": self.stage_name,
            "timestamp": datetime.utcnow().isoformat(),
            "config_hash": config_hash,
            "seed": seed,
            "total_chunks": len(selected_indices),
            "total_tokens": total_tokens,
            "band_distribution": band_counts,
            "domain_distribution": domain_counts,
            "selected_indices": selected_indices,
            "config": config
        }
        
        return manifest
    
    def save(self, manifest: Dict[str, Any], output_path: Path):
        """Save manifest to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

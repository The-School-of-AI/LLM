import yaml
from typing import Dict, List, Any

class CurriculumLoader:
    """
    Parses and validates the Production Curriculum YAML file.
    Resolves 'growth_schedule' stages against 'stage_profiles'.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path

    def load(self) -> List[Dict[str, Any]]:
        """
        Returns a FLATTENED list of stage configurations ready for the Sampler.
        [
            {
                "stage_id": "1B",
                "target_tokens": 20_000_000_000,
                "band_weights": {"B0": 0.3, ...},
                "modality_weights": {"general_text": 0.86, ...}
            },
            ...
        ]
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return self._resolve_stages(config)

    def _resolve_stages(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        stages = []
        
        growth_schedule = config.get('growth_schedule', {}).get('stages', [])
        profiles = config.get('stage_profiles', {})
        
        # Default token counts (from requirements) since YAML doesn't have them
        # 1B: 20B, 3B: 40B, 8B: 100B, 70B: 240B
        # Scaled down by factor of 1000 for local test? 
        # No, let's keep big numbers but maybe scale in Sampler or here if "mock_mode" is on.
        # User asked for "10M -> 200M" scale in previous turn.
        # Let's use a scale factor mapping.
        
        # Mapping for the specific names in the YAML
        default_targets = {
            "1B": 10_000_000,   # Local Scale: 10M
            "3B": 40_000_000,   # Local Scale: 40M
            "8B": 100_000_000,  # Local Scale: 100M
            "70B": 50_000_000   # Local Scale: 50M (capped for local test)
        }

        for stage_def in growth_schedule:
            stage_name = stage_def['name']
            profile_name = stage_def['curriculum_profile']
            
            if profile_name not in profiles:
                raise ValueError(f"Stage {stage_name} references unknown profile {profile_name}")
            
            profile = profiles[profile_name]
            
            stage_config = {
                "stage_id": stage_name,
                "target_tokens": default_targets.get(stage_name, 10_000_000),
                "band_weights": profile.get('band_weights', {}),
                "modality_weights": profile.get('modality_weights', {}),
                "notes": profile.get('notes', [])
            }
            stages.append(stage_config)
            
        return stages

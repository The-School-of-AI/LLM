"""YAML configuration loader with dotted-key access support."""

from pathlib import Path
from typing import Any, Dict

import yaml


class CurriculumConfig:
    """Load and query curriculum YAML configuration with dotted-key syntax.

    Supports flexible queries like: config.get('difficulty_system.bands.B0.name')
    """

    def __init__(self, yaml_path: str | Path):
        """Load curriculum config from YAML file.

        Args:
            yaml_path: Path to curriculum YAML file
        """
        self.path = Path(yaml_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Curriculum YAML not found: {yaml_path}")

        with open(self.path) as f:
            self._data: Dict[str, Any] = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from config using dotted key notation.

        Args:
            key: Dotted key path (e.g., 'k1.k2.k3')
            default: Default value if key not found

        Returns:
            Value at key path or default

        Examples:
            >>> config.get('difficulty_bands.B0.name')
            'Nursery'
            >>> config.get('missing.key', 'default')
            'default'
        """
        keys = key.split(".")
        value = self._data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_required(self, key: str) -> Any:
        """Get value from config, raise error if not found.

        Args:
            key: Dotted key path

        Returns:
            Value at key path

        Raises:
            KeyError: If key not found
        """
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required config key not found: {key}")
        return value

    @property
    def version(self) -> str:
        """Get curriculum version."""
        return self.get("version", "unknown")

    @property
    def raw(self) -> Dict[str, Any]:
        """Access raw YAML data."""
        return self._data

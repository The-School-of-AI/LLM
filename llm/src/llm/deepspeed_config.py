from copy import deepcopy
from typing import Any


def apply_runtime_overrides(
    ds_config: dict[str, Any], overlap_communication: bool | None
) -> dict[str, Any]:
    """
    Return a DeepSpeed config with runtime overrides applied.

    The input config is copied so callers can safely reuse the parsed YAML.
    """
    resolved = deepcopy(ds_config)

    if overlap_communication is None:
        return resolved

    zero_optimization = resolved.get("zero_optimization")
    if zero_optimization is None:
        raise ValueError(
            "training.overlap_communication requires a DeepSpeed "
            "'zero_optimization' config block"
        )
    if not isinstance(zero_optimization, dict):
        raise ValueError(
            "DeepSpeed 'zero_optimization' must be a mapping when "
            "training.overlap_communication is set"
        )

    zero_optimization["overlap_comm"] = overlap_communication
    return resolved

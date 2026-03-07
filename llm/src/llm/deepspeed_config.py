from copy import deepcopy
from typing import Any


def apply_runtime_overrides(
    ds_config: dict[str, Any],
    overlap_communication: bool | None,
    reduce_bucket_size: int | None,
) -> dict[str, Any]:
    """
    Return a DeepSpeed config with runtime overrides applied.

    The input config is copied so callers can safely reuse the parsed YAML.
    """
    resolved = deepcopy(ds_config)

    if overlap_communication is None and reduce_bucket_size is None:
        return resolved

    zero_optimization = resolved.get("zero_optimization")
    if zero_optimization is None:
        raise ValueError(
            "runtime DeepSpeed overrides require a DeepSpeed "
            "'zero_optimization' config block"
        )
    if not isinstance(zero_optimization, dict):
        raise ValueError(
            "DeepSpeed 'zero_optimization' must be a mapping when "
            "runtime DeepSpeed overrides are set"
        )

    if overlap_communication is not None:
        zero_optimization["overlap_comm"] = overlap_communication
    if reduce_bucket_size is not None:
        zero_optimization["reduce_bucket_size"] = reduce_bucket_size
    return resolved

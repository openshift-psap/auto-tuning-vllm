"""
Compute the total cardinality of the parameter grid from a study config.
Uses the same logic as StudyController._create_search_space for consistency.
"""

from pathlib import Path
from typing import Union

from auto_tune_vllm.core.config import StudyConfig
from auto_tune_vllm.core.parameters import (
    BooleanParameter,
    EnvironmentParameter,
    ListParameter,
    ParameterConfig,
    RangeParameter,
)

_MAX_GRID_SIZE = 10000


def _count_parameter_values(param: ParameterConfig) -> int:
    """Count distinct values for one parameter (mirrors _create_search_space)."""
    if isinstance(param, (ListParameter, EnvironmentParameter)):
        return len(param.options)
    if isinstance(param, RangeParameter):
        min_val = param.min_value
        max_val = param.max_value
        if param.data_type is float:
            step = param.step
            if step is None:
                return _MAX_GRID_SIZE
            n_steps = int(round((max_val - min_val) / step)) + 1
            return min(n_steps, _MAX_GRID_SIZE)
        step = param.step or 1
        current = min_val
        count = 0
        while current <= max_val and count < _MAX_GRID_SIZE:
            count += 1
            current += step
        return count
    if isinstance(param, BooleanParameter):
        return 2
    raise ValueError(f"Unknown parameter type: {type(param)}")


def get_parameter_grid_cardinality(
    config: Union[str, Path, StudyConfig],
    vllm_version: str | None = None,
) -> int:
    """
    Return the total number of points in the parameter grid (product of enabled params).

    config: Path to YAML, or an already-loaded StudyConfig.
    """
    if isinstance(config, (str, Path)):
        config = StudyConfig.from_file(str(config), vllm_version=vllm_version)
    total = 1
    for param_config in config.parameters.values():
        if param_config.enabled:
            total *= _count_parameter_values(param_config)
    return total

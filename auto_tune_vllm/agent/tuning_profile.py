"""Load named runtime policies for agentic tuning sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TuningProfile:
    """Runtime workload and safety constraints, independent of pod manifests."""

    name: str
    benchmark_profiles: list[str]
    max_tensor_parallel_size: int | None = None
    max_experiments: int | None = None
    optimization_objective: str = "throughput"
    priority_items: list[str] = field(default_factory=list)
    recipe_model_id: str | None = None
    environment: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None


def load_tuning_profile(path: str | Path) -> TuningProfile:
    """Load and validate a tuning profile YAML file."""
    profile_path = Path(path)
    with profile_path.open(encoding="utf-8") as profile_file:
        data = yaml.safe_load(profile_file) or {}

    workload = data.get("workload") or {}
    constraints = data.get("constraints") or {}
    benchmark_profiles = workload.get("benchmark_profiles") or []
    if not isinstance(benchmark_profiles, list) or not all(
        isinstance(profile, str) for profile in benchmark_profiles
    ):
        raise ValueError("workload.benchmark_profiles must be a list of strings")

    max_tp = constraints.get("max_tensor_parallel_size")
    if max_tp is not None and (not isinstance(max_tp, int) or max_tp < 1):
        raise ValueError(
            "constraints.max_tensor_parallel_size must be a positive integer"
        )
    max_experiments = constraints.get("max_experiments")
    if max_experiments is not None and (
        not isinstance(max_experiments, int) or max_experiments < 1
    ):
        raise ValueError("constraints.max_experiments must be a positive integer")

    optimization = data.get("optimization") or {}
    objective = str(optimization.get("objective", "throughput")).lower()
    if objective not in {"throughput", "latency"}:
        raise ValueError("optimization.objective must be 'throughput' or 'latency'")
    priority_items = optimization.get("priority_items") or []
    if not isinstance(priority_items, list) or not all(
        isinstance(item, str) and item.strip() for item in priority_items
    ):
        raise ValueError("optimization.priority_items must be a list of non-empty strings")
    recipe_model_id = optimization.get("recipe_model_id")
    if recipe_model_id is not None and (
        not isinstance(recipe_model_id, str) or not recipe_model_id.strip()
    ):
        raise ValueError("optimization.recipe_model_id must be a non-empty string")

    environment = data.get("environment") or None
    if environment is not None and not isinstance(environment, dict):
        raise ValueError("environment must be a mapping")
    runtime = data.get("runtime") or None
    if runtime is not None and not isinstance(runtime, dict):
        raise ValueError("runtime must be a mapping")

    return TuningProfile(
        name=str(data.get("name") or profile_path.stem),
        benchmark_profiles=benchmark_profiles,
        max_tensor_parallel_size=max_tp,
        max_experiments=max_experiments,
        optimization_objective=objective,
        priority_items=priority_items,
        recipe_model_id=recipe_model_id,
        environment=environment,
        runtime=runtime,
    )

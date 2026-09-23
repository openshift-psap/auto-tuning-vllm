"""Load named runtime policies for agentic tuning sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TuningProfile:
    """Runtime workload and safety constraints, independent of pod manifests."""

    name: str
    benchmark_profiles: list[str]
    max_tensor_parallel_size: int | None = None
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
        environment=environment,
        runtime=runtime,
    )

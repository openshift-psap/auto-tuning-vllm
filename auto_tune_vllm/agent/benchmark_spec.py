"""Pinned benchmark + warmup spec for the agentic tuner.

The workload is decided once (CLI / settings.yaml), not by the agent per
experiment. Every scored GuideLLM run is preceded by an unscored warmup
at a fixed concurrency (duration or request count).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WarmupSpec:
    """Unscored load applied before every measured benchmark."""

    enabled: bool = True
    concurrency: int = 8
    seconds: int = 15
    requests: Optional[int] = None  # if set, stop after N samples instead of seconds

    def describe(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.requests:
            return (
                f"concurrency={self.concurrency}, requests={self.requests} (unscored)"
            )
        return f"concurrency={self.concurrency}, seconds={self.seconds} (unscored)"


@dataclass
class BenchmarkSpec:
    """Locked test workload shared by baseline and every experiment."""

    profile: str = "balanced"
    isl: int = 128
    osl: int = 128
    samples: int = 100
    description: str = ""
    concurrency_levels: list[int] = field(default_factory=lambda: [50])
    max_seconds: int = 60
    warmup: WarmupSpec = field(default_factory=WarmupSpec)

    @property
    def concurrency_csv(self) -> str:
        return ",".join(str(c) for c in self.concurrency_levels)

    @property
    def data_flag(self) -> str:
        import json

        return json.dumps(
            {
                "prompt_tokens": self.isl,
                "output_tokens": self.osl,
                "samples": self.samples,
            }
        )

    def describe(self) -> str:
        return (
            f"profile={self.profile} ISL={self.isl} OSL={self.osl} "
            f"concurrency={self.concurrency_csv} max_seconds={self.max_seconds} "
            f"| warmup: {self.warmup.describe()}"
        )


def spec_from_settings(
    settings: dict,
    profiles: dict,
    *,
    profile_name: Optional[str] = None,
    concurrency: Optional[str] = None,
    max_seconds: Optional[int] = None,
    warmup_concurrency: Optional[int] = None,
    warmup_seconds: Optional[int] = None,
    warmup_requests: Optional[int] = None,
    warmup_enabled: bool = True,
) -> BenchmarkSpec:
    """Build a spec from settings.yaml, then apply CLI overrides."""
    bench = settings.get("benchmark") or {}
    warm = settings.get("warmup") or {}
    name = profile_name or bench.get("profile") or "balanced"
    if name not in profiles:
        raise ValueError(
            f"Unknown profile '{name}'. Choose from: {list(profiles.keys())}"
        )
    profile = profiles[name]

    conc_src = concurrency or ",".join(
        str(c) for c in (bench.get("concurrency_levels") or [50])
    )
    conc_levels = [int(x.strip()) for x in str(conc_src).split(",") if x.strip()]
    if not conc_levels:
        conc_levels = [50]

    wr = warmup_requests if warmup_requests is not None else warm.get("requests")
    wr_int = int(wr) if wr else None

    warmup = WarmupSpec(
        enabled=warmup_enabled and bool(warm.get("enabled", True)),
        concurrency=int(
            warmup_concurrency
            if warmup_concurrency is not None
            else warm.get("concurrency", 8)
        ),
        seconds=int(
            warmup_seconds if warmup_seconds is not None else warm.get("seconds", 15)
        ),
        requests=wr_int,
    )
    return BenchmarkSpec(
        profile=name,
        isl=int(profile["isl"]),
        osl=int(profile["osl"]),
        samples=int(profile.get("samples", 100)),
        description=profile.get("description", name),
        concurrency_levels=conc_levels,
        max_seconds=int(
            max_seconds if max_seconds is not None else bench.get("max_seconds", 60)
        ),
        warmup=warmup,
    )

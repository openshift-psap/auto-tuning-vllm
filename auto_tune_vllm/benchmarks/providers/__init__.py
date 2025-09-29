"""Benchmark provider base classes and registry."""

from __future__ import annotations

from .guidellm import GuideLLMBenchmark
from .mlperf import MLPerfBenchmark
from .template import BenchmarkProvider, CustomBenchmarkTemplate

__all__ = [
    "BenchmarkProvider",
    "GuideLLMBenchmark",
    "MLPerfBenchmark",
]

BENCHMARK_PROVIDERS = {
    "guidellm": GuideLLMBenchmark,
    "custom_template": CustomBenchmarkTemplate,
    "mlperf": MLPerfBenchmark,
}


def get_benchmark_provider(provider_name: str) -> BenchmarkProvider:
    """Get benchmark provider by name."""
    if provider_name not in BENCHMARK_PROVIDERS:
        raise ValueError(
            f"Unknown benchmark provider: {provider_name}. "
            f"Available providers: {list(BENCHMARK_PROVIDERS.keys())}"
        )

    provider_class = BENCHMARK_PROVIDERS[provider_name]
    return provider_class()


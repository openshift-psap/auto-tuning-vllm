"""Benchmark provider implementations package.

This package contains the benchmark provider implementations for auto-tuning vLLM.
"""

from .base import BenchmarkProvider
from .guidellm import GuideLLMBenchmark
from .template import CustomBenchmarkTemplate

__all__ = [
    "BenchmarkProvider",
    "GuideLLMBenchmark",
    "CustomBenchmarkTemplate",
]

BENCHMARK_PROVIDERS = {
    "guidellm": GuideLLMBenchmark,
    "custom_template": CustomBenchmarkTemplate,
}

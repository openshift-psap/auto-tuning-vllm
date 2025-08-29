"""
Auto-Tune vLLM: Distributed hyperparameter optimization framework for vLLM serving.

This package provides:
- Distributed optimization using Ray or local execution
- Flexible benchmark providers (GuideLLM, custom)
- Centralized logging and result management
- Study management with PostgreSQL backend
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main public API
from .core.study_controller import StudyController
from .core.config import StudyConfig, ParameterConfig
from .execution.backends import RayExecutionBackend, LocalExecutionBackend
from .benchmarks.providers import GuideLLMBenchmark, BenchmarkProvider

__all__ = [
    "StudyController",
    "StudyConfig", 
    "ParameterConfig",
    "RayExecutionBackend",
    "LocalExecutionBackend", 
    "GuideLLMBenchmark",
    "BenchmarkProvider",
]
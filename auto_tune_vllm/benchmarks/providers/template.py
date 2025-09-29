"""Template for implementing custom benchmark providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..config import BenchmarkConfig

logger = logging.getLogger(__name__)


class BenchmarkProvider(ABC):
    """Abstract benchmark provider interface."""
    
    def __init__(self):
        self._logger = logger  # Default to module logger
        self._trial_context = None  # Store trial context for file paths
    
    def set_logger(self, custom_logger):
        """Set a custom logger for this benchmark provider."""
        self._logger = custom_logger
    
    def set_trial_context(self, study_name: str, trial_id: str):
        """Set trial context for benchmark result storage."""
        self._trial_context = {
            'study_name': study_name,
            'trial_id': trial_id
        }
    
    @abstractmethod
    def run_benchmark(self, model_url: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Run benchmark against model server.
        
        Args:
            model_url: URL of the vLLM server (e.g., "http://localhost:8000/v1")
            config: Benchmark configuration
            
        Returns:
            Dictionary with benchmark results. Must include metrics that can be
            converted to objective values for Optuna.
        """
        pass


class CustomBenchmarkTemplate(BenchmarkProvider):
    """Template for implementing custom benchmark providers."""

    def run_benchmark(self, model_url: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Template implementation for custom benchmarks.

        Override this method to implement your custom benchmark logic.
        The returned dictionary should contain metrics that will be used
        to compute objective values for Optuna optimization.
        """
        # Example custom benchmark implementation:

        # 1. Setup your benchmark client/tools
        # benchmark_client = YourBenchmarkTool(model_url)

        # 2. Run your benchmark logic
        # benchmark_results = benchmark_client.run_test(
        #     duration=config.max_seconds,
        #     concurrency=config.concurrency,
        #     # ... other config parameters
        # )

        # 3. Return structured results
        # The keys should be metric names, values should be numeric
        # Optuna will use these for optimization objectives

        return {
            "throughput": 0.0,  # requests/second or tokens/second
            "latency_p95": 0.0,  # 95th percentile latency in ms
            "latency_mean": 0.0,  # mean latency in ms
            "error_rate": 0.0,  # error percentage
            # Add any other metrics relevant to your benchmark
            # These will be stored in the detailed_metrics and can be
            # used for analysis and visualization
        }

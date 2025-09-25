"""Benchmark provider implementations."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from .config import BenchmarkConfig

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


class GuideLLMBenchmark(BenchmarkProvider):
    """GuideLLM benchmark provider implementation."""
    
    def run_benchmark(self, model_url: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run GuideLLM benchmark."""
        self._logger.info(f"Starting GuideLLM benchmark for {config.model}")
        
        # Create temporary file for results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            results_file = f.name
        
        try:
            # Build GuideLLM command
            cmd = self._build_guidellm_command(model_url, config, results_file)
            # Validate binary and basic inputs
            import shutil
            if shutil.which("guidellm") is None:
                raise RuntimeError("GuideLLM CLI not found on PATH. Please install or provide the full path.")
            if not (model_url.startswith("http://") or model_url.startswith("https://")):
                raise ValueError(f"Invalid model_url: {model_url!r} (expected http/https)")

            # Run GuideLLM
            self._logger.info(f"Running: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                timeout=config.max_seconds + 120,  # Add buffer for setup/teardown
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log GuideLLM output for debugging
            if process.stdout:
                self._logger.info(f"GuideLLM stdout:\n{process.stdout}")
            if process.stderr:
                self._logger.warning(f"GuideLLM stderr:\n{process.stderr}")
            
            self._logger.debug(f"GuideLLM process completed with return code: {process.returncode}")
            
            self._logger.info("GuideLLM completed successfully")
            
            # Parse results
            return self._parse_guidellm_results(results_file)
            
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"GuideLLM benchmark timed out after {config.max_seconds + 120} seconds"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GuideLLM failed: {e.stderr}") from e
        except OSError as e:
            # e.g., FileNotFoundError for the CLI
            raise RuntimeError("Failed to execute GuideLLM CLI") from e
        finally:
            # Copy results to permanent location before cleanup
            if os.path.exists(results_file):
                preserved_file = self._preserve_results_file(results_file)
                if preserved_file:
                    self._logger.info(f"Benchmark results preserved at: {preserved_file}")
                # Clean up temporary file
                os.unlink(results_file)
    
    def _preserve_results_file(self, temp_file: str) -> str:
        """Copy benchmark results to permanent location in study directory."""
        if self._trial_context is None:
            self._logger.warning("No trial context set, cannot preserve results file")
            return None
        
        try:
            # Create directory structure: /tmp/auto-tune-vllm-local-run/logs/{study_name}/benchmark_results/
            study_name = self._trial_context['study_name']
            trial_id = self._trial_context['trial_id']
            
            # Use /tmp as base directory for consistency with existing log structure
            base_dir = Path("/tmp/auto-tune-vllm-local-run/logs")
            benchmark_dir = base_dir / study_name / "benchmark_results"
            
            # Create directory if it doesn't exist
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            
            # Create permanent results file with trial-specific name
            permanent_file = benchmark_dir / f"{trial_id}_benchmark_results.json"
            
            # Copy the temporary file to permanent location
            import shutil
            shutil.copy2(temp_file, permanent_file)
            
            return str(permanent_file)
            
        except Exception as e:
            self._logger.warning(f"Failed to preserve benchmark results: {e}")
            return None
    
    def _build_guidellm_command(
        self, 
        model_url: str, 
        config: BenchmarkConfig, 
        results_file: str
    ) -> list[str]:
        """Build GuideLLM command arguments."""
        # Use processor if specified, otherwise default to model
        processor = config.processor if config.processor is not None else config.model
        
        cmd = [
            "guidellm", "benchmark",
            "--target", model_url,
            "--model", config.model,
            "--processor", processor,
            "--rate-type", "concurrent",
            "--max-seconds", str(config.max_seconds),
            "--rate", str(config.rate),
            "--output-path", results_file
        ]
        
        # Add dataset or synthetic data configuration
        if config.use_synthetic_data:
            # Build complex data JSON object with statistics
            data_config = {
                "prompt_tokens": config.prompt_tokens,
                "prompt_tokens_stdev": config.prompt_tokens_stdev,
                "prompt_tokens_min": config.prompt_tokens_min,
                "prompt_tokens_max": config.prompt_tokens_max,
                "output_tokens": config.output_tokens,
                "output_tokens_stdev": config.output_tokens_stdev,
                "output_tokens_min": config.output_tokens_min,
                "output_tokens_max": config.output_tokens_max
            }
            
            cmd.extend([
                "--data", json.dumps(data_config)
            ])
        else:
            if config.dataset.startswith("hf://"):
                # HuggingFace dataset
                dataset_name = config.dataset[5:]  # Remove "hf://" prefix
                cmd.extend(["--data-type", "huggingface", "--dataset", dataset_name])
            else:
                # Local file
                if not os.path.exists(config.dataset):
                    raise FileNotFoundError(f"Dataset file not found: {config.dataset}")
                cmd.extend(["--data-type", "file", "--dataset", config.dataset])
        return cmd
    
    def _parse_guidellm_results(self, results_file: str) -> Dict[str, Any]:
        """Parse GuideLLM JSON results."""
        if not os.path.exists(results_file):
            raise RuntimeError(f"GuideLLM results file not found: {results_file}")
        
        try:
            with open(results_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in results file: {e}")
        
        # Extract benchmark data structure
        try:
            benchmarks = data["benchmarks"]
            if not benchmarks:
                raise RuntimeError("No benchmark data found in results")
            
            # Get first (and typically only) benchmark result
            benchmark_data = benchmarks[0]
            metrics = benchmark_data["metrics"]
            
            # Extract key metrics with percentiles
            result = {}
            
            required_metrics = [
                "requests_per_second",
                "request_latency", 
                "output_tokens_per_second",
                "time_to_first_token_ms",
                "inter_token_latency_ms"
            ]
            
            for metric_name in required_metrics:
                if metric_name not in metrics:
                    self._logger.warning(f"Missing metric: {metric_name}")
                    continue
                
                metric_data = metrics[metric_name]["successful"]
                
                # Store median value and percentiles
                result[metric_name] = metric_data["median"]
                
                percentiles = metric_data.get("percentiles", {})
                for percentile, value in percentiles.items():
                    result[f"{metric_name}_{percentile}"] = value
            
            return result
            
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Invalid benchmark data structure: {e}")


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
            "error_rate": 0.0,   # error percentage
            # Add any other metrics relevant to your benchmark
            # These will be stored in the detailed_metrics and can be
            # used for analysis and visualization
        }


# Registry for dynamic benchmark provider loading
BENCHMARK_PROVIDERS = {
    "guidellm": GuideLLMBenchmark,
    "custom_template": CustomBenchmarkTemplate,
}


def get_benchmark_provider(provider_name: str) -> BenchmarkProvider:
    """Get benchmark provider by name."""
    if provider_name not in BENCHMARK_PROVIDERS:
        raise ValueError(f"Unknown benchmark provider: {provider_name}. "
                        f"Available providers: {list(BENCHMARK_PROVIDERS.keys())}")
    
    provider_class = BENCHMARK_PROVIDERS[provider_name]
    return provider_class()
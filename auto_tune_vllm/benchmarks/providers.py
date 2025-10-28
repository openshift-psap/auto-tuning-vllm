"""Benchmark provider implementations."""

from __future__ import annotations

import json
import logging
import os
import signal
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
        self._process = None  # Track running benchmark process for termination
        self._process_pid = None  # Store PID for cleanup even if process handle is gone
        self._process_pgid = None  # Store process group ID for cleanup
        self._cancellation_flag = None  # Function to check for cancellation
    
    def set_logger(self, custom_logger):
        """Set a custom logger for this benchmark provider."""
        self._logger = custom_logger

    def set_trial_context(self, study_name: str, trial_id: str):
        """Set trial context for benchmark result storage."""
        self._trial_context = {
            'study_name': study_name,
            'trial_id': trial_id
        }
    
    def terminate_benchmark(self):
        """Terminate the running benchmark process and its process group if active."""
        # Try to use stored PID/PGID first, in case process handle is gone
        pid = self._process_pid
        if pid is None:  # first check, if PID is still none after this we will return
            pid = self._process.pid if self._process else None
        pgid = self._process_pgid
        
        if pid is None:
            self._logger.debug("Benchmark: No benchmark process to terminate")
            return
            
        self._logger.info(
            f"Benchmark: Terminating benchmark process {pid} "
            f"and its process group..."
        )
        
        # Try to get process group ID if we don't have it
        if pgid is None:
            try:
                pgid = os.getpgid(pid)
                self._logger.debug(f"Benchmark: Retrieved process group ID: {pgid}")
            except (OSError, ProcessLookupError):
                self._logger.debug(
                    f"Benchmark: Process {pid} already gone or no process group"
                )
        
        # Try graceful shutdown with SIGTERM first
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
                self._logger.info(
                    f"Benchmark → Process Group: Sent SIGTERM to group {pgid}"
                )
            else:
                os.kill(pid, signal.SIGTERM)
                self._logger.info(f"Benchmark → Process: Sent SIGTERM to process {pid}")
        except (OSError, ProcessLookupError):
            # Process already gone
            self._logger.info(f"Benchmark: Process {pid} already terminated")
            self._process = None
            self._process_pid = None
            self._process_pgid = None
            return
        
        # Wait for graceful shutdown
        # (use a shorter timeout if process handle unavailable)
        wait_timeout = 5 if self._process else 2
        try:
            if self._process:
                self._process.wait(timeout=wait_timeout)
            else:
                # Wait by polling if no process handle
                import time
                for _ in range(int(wait_timeout * 10)):
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        time.sleep(0.1)
                    except (OSError, ProcessLookupError):
                        # Process is gone
                        break
                else:
                    # Timeout - process still exists
                    raise subprocess.TimeoutExpired(None, wait_timeout)
                    
            self._logger.info(
                f"Benchmark: ✓ Process {pid} terminated gracefully via SIGTERM"
            )
        except subprocess.TimeoutExpired:
            self._logger.warning(
                f"Benchmark: Process {pid} did not terminate within {wait_timeout}s. "
                f"Escalating to SIGKILL..."
            )
            
            # Force kill with SIGKILL
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                    self._logger.info(
                        f"Benchmark → Process Group: Sent SIGKILL to group {pgid}"
                    )
                else:
                    os.kill(pid, signal.SIGKILL)
                    self._logger.info(
                        f"Benchmark → Process: Sent SIGKILL to process {pid}"
                    )
                self._logger.info(
                    f"Benchmark: ✓ Process {pid} force killed via SIGKILL"
                )
            except (OSError, ProcessLookupError) as e:
                self._logger.debug(
                    f"Benchmark: Process {pid} already gone during SIGKILL: {e}"
                )
        finally:
            self._process = None
            self._process_pid = None
            self._process_pgid = None

    @abstractmethod
    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen:
        """
        Start benchmark subprocess (non-blocking).

        Args:
            model_url: URL of the vLLM server (e.g., "http://localhost:8000/v1")
            config: Benchmark configuration

        Returns:
            Popen process handle for polling by caller
        """
        pass

    @abstractmethod
    def parse_results(self) -> Dict[str, Any]:
        """
        Parse benchmark results from output file.

        Returns:
            Dictionary with benchmark results. Must include metrics that can be
            converted to objective values for Optuna.
        """
        pass


class GuideLLMBenchmark(BenchmarkProvider):
    """GuideLLM benchmark provider implementation."""

    # Maps metric names to their appropriate request category in GuideLLM output
    METRIC_CATEGORIES = {
        # Throughput metrics - use 'total' for overall system performance
        "output_tokens_per_second": "total",
        "requests_per_second": "total",
        "tokens_per_second": "total",
        # Quality metrics - use 'successful' for performance characteristics
        "request_latency": "successful",
        "time_to_first_token_ms": "successful",
        "inter_token_latency_ms": "successful",
        "time_per_output_token_ms": "successful",
        # Count metrics - use 'successful'
        "output_token_count": "successful",
        "prompt_token_count": "successful",
        "request_concurrency": "successful",
    }

    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen:
        """
        Start GuideLLM benchmark subprocess (non-blocking).
        
        Returns:
            Popen process handle for polling by caller
        """
        self._logger.info(f"Starting GuideLLM benchmark for {config.model}")

        # Create results file path directly in permanent location
        self._results_file = self._get_results_file_path()

        # Build GuideLLM command
        cmd = self._build_guidellm_command(model_url, config, self._results_file)
        
        # Validate binary and basic inputs
        import shutil

        if shutil.which("guidellm") is None:
            raise RuntimeError(
                "GuideLLM CLI not found on PATH. "
                "Please install or provide the full path."
            )
        if not (
            model_url.startswith("http://") or model_url.startswith("https://")
        ):
            raise ValueError(
                f"Invalid model_url: {model_url!r} (expected http/https)"
            )

        # Run GuideLLM
        self._logger.info(f"Running: {' '.join(cmd)}")
        self._logger.info(f"Results will be saved to: {self._results_file}")
        
        # Use Popen so we can terminate if vLLM dies
        # start_new_session=True puts it in its own process group for clean
        # termination
        env = os.environ.copy()
        env["GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL"] = config.logging_level

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True
        )
        
        # Store PID and PGID immediately for cleanup, even if process handle is lost
        self._process_pid = self._process.pid
        try:
            self._process_pgid = os.getpgid(self._process_pid)
            self._logger.debug(
                f"Started GuideLLM process {self._process_pid} "
                f"in process group {self._process_pgid}"
            )
        except (OSError, ProcessLookupError):
            self._logger.warning(
                f"Failed to get process group for GuideLLM process "
                f"{self._process_pid}"
            )
            self._process_pgid = None
        
        return self._process

    def parse_results(self) -> Dict[str, Any]:
        """
        Parse GuideLLM benchmark results from output file.
        
        Returns:
            Dictionary with benchmark metrics
        """
        results_file = self._results_file
        
        if not os.path.exists(results_file):
            raise RuntimeError(f"GuideLLM results file not found: {results_file}")
        
        try:
            with open(results_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in results file: {e}")

        return self._parse_guidellm_results(data)

    def _get_results_file_path(self) -> str:
        """
        Get the permanent results file path, creating directory structure if needed.
        """
        if self._trial_context is None:
            # Fallback to temporary file if no trial context
            self._logger.warning("No trial context set, using temporary file")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                return f.name

        try:
            # Create directory structure:
            # /tmp/auto-tune-vllm-local-run/logs/{study_name}/benchmark_results/
            study_name = self._trial_context["study_name"]
            trial_id = self._trial_context["trial_id"]

            # Use /tmp as base directory for consistency with existing log structure
            base_dir = Path("/tmp/auto-tune-vllm-local-run/logs")
            benchmark_dir = base_dir / study_name / "benchmark_results"

            # Create directory if it doesn't exist
            benchmark_dir.mkdir(parents=True, exist_ok=True)

            # Create permanent results file with trial-specific name
            permanent_file = benchmark_dir / f"{trial_id}_benchmark_results.json"

            return str(permanent_file)

        except Exception as e:
            self._logger.warning(
                f"Failed to create permanent results path: {e}, using temporary file"
            )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                return f.name

    def _build_guidellm_command(
        self, model_url: str, config: BenchmarkConfig, results_file: str
    ) -> list[str]:
        """Build GuideLLM command arguments."""
        # Use processor if specified, otherwise default to model
        processor = config.processor if config.processor is not None else config.model

        cmd = [
            "guidellm",
            "benchmark",
            "--target",
            model_url,
            "--model",
            config.model,
            "--processor",
            processor,
            "--rate-type",
            "concurrent",
            "--max-seconds",
            str(config.max_seconds),
            "--rate",
            str(config.rate),
            "--output-path",
            results_file,
            "--processor-args",
            '{"trust-remote-code":"true"}'
        ]

        # Add dataset or synthetic data configuration
        if config.use_synthetic_data:
            # Build data JSON object - only include statistical parameters if specified
            data_config = {
                "prompt_tokens": config.prompt_tokens,
                "output_tokens": config.output_tokens,
                "samples": config.samples
            }

            # Only add statistical distribution parameters if they were explicitly
            # specified
            if config.prompt_tokens_stdev is not None:
                data_config["prompt_tokens_stdev"] = config.prompt_tokens_stdev
            if config.prompt_tokens_min is not None:
                data_config["prompt_tokens_min"] = config.prompt_tokens_min
            if config.prompt_tokens_max is not None:
                data_config["prompt_tokens_max"] = config.prompt_tokens_max
            if config.output_tokens_stdev is not None:
                data_config["output_tokens_stdev"] = config.output_tokens_stdev
            if config.output_tokens_min is not None:
                data_config["output_tokens_min"] = config.output_tokens_min
            if config.output_tokens_max is not None:
                data_config["output_tokens_max"] = config.output_tokens_max

            cmd.extend(["--data", json.dumps(data_config)])
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

    def _parse_guidellm_results(self, data: dict) -> Dict[str, Any]:
        """Parse GuideLLM JSON results data structure."""
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
                "inter_token_latency_ms",
            ]

            for metric_name in required_metrics:
                if metric_name not in metrics:
                    # FAIL HARD - no fallbacks for missing metrics
                    raise RuntimeError(
                        f"Required metric '{metric_name}' not found in GuideLLM results"
                    )

                # Get the appropriate request category for this metric
                category = self.METRIC_CATEGORIES.get(metric_name, "successful")
                if metric_name not in self.METRIC_CATEGORIES:
                    self._logger.warning(
                        f"Unknown metric '{metric_name}', "
                        f"defaulting to 'successful' category"
                    )

                # Validate that the category exists in the results
                if category not in metrics[metric_name]:
                    raise RuntimeError(
                        f"Category '{category}' not found for metric '{metric_name}'. "
                        f"Available categories: {list(metrics[metric_name].keys())}"
                    )

                metric_data = metrics[metric_name][category]

                # Extract ALL statistical measures that GuideLLM provides
                statistical_measures = [
                    "mean",
                    "median",
                    "mode",
                    "min",
                    "max",
                    "std_dev",
                    "variance",
                ]

                for measure in statistical_measures:
                    if measure in metric_data:
                        result[f"{metric_name}_{measure}"] = metric_data[measure]

                # Extract percentiles
                percentiles = metric_data.get("percentiles", {})
                for percentile, value in percentiles.items():
                    result[f"{metric_name}_{percentile}"] = value

                # Store base metric as median for backward compatibility
                result[metric_name] = metric_data.get(
                    "median", metric_data.get("mean", 0.0)
                )

            return result

        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Invalid benchmark data structure: {e}")


class CustomBenchmarkTemplate(BenchmarkProvider):
    """Template for implementing custom benchmark providers."""

    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen:
        """
        Template implementation for starting custom benchmarks.

        Override this method to start your custom benchmark subprocess.
        Should return a Popen process handle for the caller to poll.
        
        Example:
            cmd = ["your-benchmark-tool", "--url", model_url, ...]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            self._process_pid = self._process.pid
            self._process_pgid = os.getpgid(self._process.pid)
            return self._process
        """
        raise NotImplementedError("CustomBenchmarkTemplate is a template only")

    def parse_results(self) -> Dict[str, Any]:
        """
        Template implementation for parsing benchmark results.

        Override this method to parse your benchmark output file.
        The returned dictionary should contain metrics that will be used
        to compute objective values for Optuna optimization.
        
        Example:
            with open(self._results_file) as f:
                data = json.load(f)
            return {
                "throughput": data["throughput"],
                "latency_p95": data["p95_latency"],
                ...
            }
        """
        # Return template structure showing expected metrics
        return {
            "throughput": 0.0,  # requests/second or tokens/second
            "latency_p95": 0.0,  # 95th percentile latency in ms
            "latency_mean": 0.0,  # mean latency in ms
            "error_rate": 0.0,  # error percentage
            # Add any other metrics relevant to your benchmark
            # These will be stored in the detailed_metrics and can be
            # used for analysis and visualization
        }


# Registry for dynamic benchmark provider loading (for reference/documentation)
BENCHMARK_PROVIDERS = {
    "guidellm": GuideLLMBenchmark,
    "custom_template": CustomBenchmarkTemplate,
}


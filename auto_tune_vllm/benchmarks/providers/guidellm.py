"""GuideLLM benchmark provider implementation."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from typing_extensions import override

from ..config import BenchmarkConfig
from .base import BenchmarkProvider

logger = logging.getLogger(__name__)


class GuideLLMBenchmark(BenchmarkProvider):
    """GuideLLM benchmark provider implementation."""

    # Maps metric names to their appropriate request category in GuideLLM output
    METRIC_CATEGORIES: dict[str, str] = {
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

    @override
    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen[str]:
        """
        Start GuideLLM benchmark subprocess (non-blocking).

        Returns:
            Popen process handle for polling by caller
        """
        self._logger.info(f"Starting GuideLLM benchmark for {config.model}")

        # Create results file path directly in permanent location
        self._results_file: str = self._get_results_file_path()

        # Build GuideLLM command
        cmd = self._build_guidellm_command(model_url, config, self._results_file)

        # Validate binary and basic inputs
        import shutil

        if shutil.which("guidellm") is None:
            raise RuntimeError(
                "GuideLLM CLI not found on PATH. "
                "Please install or provide the full path."
            )
        if not (model_url.startswith("http://") or model_url.startswith("https://")):
            raise ValueError(f"Invalid model_url: {model_url!r} (expected http/https)")

        # Run GuideLLM
        self._logger.info(f"Running: {' '.join(cmd)}")
        self._logger.info(f"Results will be saved to: {self._results_file}")

        # Use Popen so we can terminate if vLLM dies
        # start_new_session=True puts it in its own process group for clean termination
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
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
                f"Failed to get process group for GuideLLM process {self._process_pid}"
            )
            self._process_pgid = None

        return self._process

    @override
    def parse_results(self) -> dict[str, Any]:
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
            '{"trust-remote-code":"true"}',
        ]

        # Add synthetic data or specified dataset configuration
        if config.dataset is None:
            data_config = self._build_synthetic_dataset_args(config)
            cmd.extend(["--data", json.dumps(data_config)])
        else: # dataset is an explicit path
            if config.dataset.startswith("hf://"): # huggingface dataset
                dataset_name = config.dataset[5:]  # Remove "hf://" prefix
                cmd.extend(["--data-type", "huggingface", "--dataset", dataset_name])
            else:
                # Local file
                if not os.path.exists(config.dataset):
                    raise FileNotFoundError(f"Dataset file not found: {config.dataset}")
                cmd.extend(["--data-type", "file", "--dataset", config.dataset])

        return cmd

    def _build_synthetic_dataset_args(self, config: BenchmarkConfig):
        data_config = {
            "prompt_tokens": config.prompt_tokens,
            "output_tokens": config.output_tokens,
            "samples": config.samples,
        }
        # Only add statistical distribution parameters if they were explicitly specified
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

        return data_config

    def _parse_guidellm_results(self, data: dict[str, Any]) -> dict[str, Any]:
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
            result: dict[str, Any] = {}

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

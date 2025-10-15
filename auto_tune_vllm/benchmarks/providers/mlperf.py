"""MLPerf benchmark provider implementation."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict

from ..config import BenchmarkConfig
from .template import BenchmarkProvider

logger = logging.getLogger(__name__)


class MLPerfBenchmark(BenchmarkProvider):
    """MLPerf benchmark provider implementation."""

    def run_benchmark(self, model_url: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run MLPerf benchmark."""
        self._logger.info(f"Starting MLPerf benchmark for {config.model}")

        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as results_dir:
            try:
                # Build MLPerf command
                cmd = self._build_mlperf_command(model_url, config, results_dir)

                # Run MLPerf benchmark
                self._logger.info(f"Running: {' '.join(cmd)}")
                self._logger.info(f"Results will be saved to: {results_dir}")
                process = subprocess.run(
                    cmd,
                    timeout=config.max_seconds + 300,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Log output for debugging
                if process.stdout:
                    self._logger.info(f"MLPerf stdout:\n{process.stdout}")
                if process.stderr:
                    self._logger.warning(f"MLPerf stderr:\n{process.stderr}")

                self._logger.debug(
                    f"MLPerf process completed with return code: {process.returncode}"
                )
                self._logger.info("MLPerf completed successfully")

                # Parse results from output directory
                return self._parse_mlperf_results(results_dir)

            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"MLPerf benchmark timed out after {config.max_seconds + 300} seconds"
                ) from e
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"MLPerf failed: {e.stderr}") from e
            except OSError as e:
                raise RuntimeError("Failed to execute MLPerf benchmark") from e

    def _build_mlperf_command(
        self, model_url: str, config: BenchmarkConfig, results_dir: str
    ) -> list[str]:
        assert isinstance(config.dataset, str)
        # Strip "/v1" from the model_url if present
        if model_url.endswith("/v1"):
            model_url = model_url[:-3]
        return [
            "python3",
            config._mlperf_cmd_path,
            "--model",
            config.model,
            "--dataset-path",
            config.dataset,
            "--user-conf",
            config._mlperf_conf_path,
            "--test-mode",
            config._mlperf_test_mode,
            "--output-log-dir",
            results_dir,
            "--api-server-url",
            model_url,
        ]

    def _parse_mlperf_results(self, results_dir: str) -> Dict[str, Any]:
        """Parse MLPerf results from output directory."""
        # Look for MLPerf result files in the output directory
        result_files = []
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith(".txt") and "summary" in file.lower():
                    result_files.append(os.path.join(root, file))

        if not result_files:
            raise RuntimeError(f"No MLPerf result files found in {results_dir}")

        metrics = {
            "request_per_second": 0.0,
            "latency_p99": 0.0,
            "latency_mean": 0.0,
            "output_tokens_per_second": 0.0,
        }

        # Parse MLPerf log format
        for result_file in result_files:
            try:
                with open(result_file, "r") as f:
                    content = f.read()

                metrics = {}

                # Extract key metrics from MLPerf log format
                for line in content.split("\n"):
                    line = line.strip()

                    # Throughput (samples per second)
                    if line.startswith("Samples per second"):
                        throughput = float(line.split(":")[1].strip())
                        metrics["request_per_second"] = throughput

                    # Tokens per second
                    elif line.startswith("Tokens per second"):
                        if ":" in line:
                            tokens_per_second = float(line.split(":")[1].strip())
                            metrics["output_tokens_per_second"] = tokens_per_second

                    # Mean latency (convert from nanoseconds to milliseconds)
                    elif line.startswith("Mean latency (ns)"):
                        mean_latency_ns = float(line.split(":")[1].strip())
                        metrics["latency_mean"] = (
                            mean_latency_ns / 1_000_000
                        )  # Convert to ms

                    # 99th percentile latency (convert from nanoseconds to milliseconds)
                    elif line.startswith("99.00 percentile latency (ns)"):
                        p99_latency_ns = float(line.split(":")[1].strip())
                        metrics["latency_p99"] = (
                            p99_latency_ns / 1_000_000
                        )  # Convert to ms

                self._logger.info(f"{metrics=}")
                return metrics

            except (IOError, ValueError, IndexError):
                continue

        # If no valid files found, return default metrics
        return metrics 

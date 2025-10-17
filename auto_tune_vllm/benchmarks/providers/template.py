"""Template for implementing custom benchmark providers."""

from __future__ import annotations

import subprocess
from typing import Any

from typing_extensions import override

from ..config import BenchmarkConfig
from .base import BenchmarkProvider


class CustomBenchmarkTemplate(BenchmarkProvider):
    """Template for implementing custom benchmark providers."""

    @override
    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen[str]:
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

    @override
    def parse_results(self) -> dict[str, Any]:
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

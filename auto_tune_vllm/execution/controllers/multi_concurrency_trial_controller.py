"""Trial controller implementations for Ray and local execution."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

from auto_tune_vllm.core.config import OptimizationConfig

from ...benchmarks.providers import BenchmarkProvider, GuideLLMBenchmark
from ...core.trial import (
    ExecutionInfo,
    MultiConccurencyTrialResult,
    TrialConfig,
)
from ..vllmprocess import vLLMProcess
from .trial_loggers import TrialLoggers

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY_LIST = [4, 8, 16, 32, 64]
POLL_INTERVAL = 1


class MultiConcurrencyTrial:
    def __init__(self):
        # Trial-specific logger manager
        self.loggers: TrialLoggers = TrialLoggers()
        # Flag for external cancellation requests
        self._cancellation_requested: bool = False
        self._trial_context: dict[str, Any] = {}

    def _create_benchmark_provider(
        self, benchmark_type: str, logger: logging.Logger | None = None
    ) -> BenchmarkProvider:
        """Create appropriate benchmark provider."""
        if benchmark_type == "guidellm":
            return GuideLLMBenchmark(custom_logger=logger)
        else:
            raise NotImplementedError

    def run_single_trial(
        self,
        trial_config: TrialConfig,
        server_url: str,
        concurrency_level: int,
    ) -> dict[str, int | float]:
        controller_logger = self.loggers.get_logger("controller")
        controller_logger.info(f"vLLM server ready at {server_url}")
        controller_logger.info(f"Starting run with concurrency={concurrency_level}")

        benchmark_logger = self.loggers.get_logger("benchmark")
        benchmark_provider = self._create_benchmark_provider(
            trial_config.benchmark_config.benchmark_type, logger=benchmark_logger
        )
        benchmark_provider.set_trial_context(
            trial_config.study_name, trial_config.trial_id
        )

        # Clone benchmark config and set concurrency
        modified_benchmark_config = copy.deepcopy(trial_config.benchmark_config)
        modified_benchmark_config.rate = concurrency_level
        modified_benchmark_config.concurrency = concurrency_level
        benchmark_proc = benchmark_provider.start_benchmark(
            server_url, modified_benchmark_config
        )

        # Poll until benchmark completes
        poll_count = 0
        benchmark_start_time = time.time()
        while benchmark_proc.poll() is None:
            poll_count += 1
            time.sleep(POLL_INTERVAL)
        elapsed = time.time() - benchmark_start_time
        status = benchmark_proc.returncode
        if status == 0:
            benchmark_logger.info(f"Benchmark completed after {elapsed:.1f}s")
            return benchmark_provider.parse_results()

        return {}

    def run_trial(
        self,
        trial_config: TrialConfig,
    ) -> MultiConccurencyTrialResult:
        """Execute trial with proper error handling and cleanup.

        Args:
            trial_config: Configuration for this trial
        """
        execution_info = ExecutionInfo()
        self.loggers.setup(trial_config)
        controller_logger = self.loggers.get_logger("controller")
        controller_logger.info(
            f"Running trial {trial_config.trial_id} "
            f"with parameters: {trial_config.parameters}"
        )
        controller_logger.info(f"Study name: {trial_config.study_name}")

        current_study_name: str = trial_config.study_name
        self._trial_context = {
            "study_name": trial_config.study_name,
            "trial_id": trial_config.trial_id,
        }
        # Start vLLM server using vLLMProcess
        controller_logger.info("Starting vLLM server")
        execution_info.mark_vllm_started()
        vllm_server = vLLMProcess(
            model=trial_config.benchmark_config.model,
            vllm_args=trial_config.vllm_args,
            env_vars=trial_config.environment_vars,
            startup_timeout=trial_config.vllm_startup_timeout,
            health_check_interval=trial_config.health_check_interval,
            health_check_max_failures=trial_config.health_check_max_failures,
        )
        execution_info.mark_vllm_ready()

        controller_logger.info(f"Sweeping over {DEFAULT_CONCURRENCY_LIST}")
        server_url = vllm_server.get_url_for("v1")

        all_objective_values: dict[int, list[float]] = {}
        all_detailed_metrics: dict[int, dict[str, float | int]] = {}

        for concurrency_level in DEFAULT_CONCURRENCY_LIST:
            res = self.run_single_trial(
                trial_config=trial_config,
                server_url=server_url,
                concurrency_level=concurrency_level,
            )
            objective_value = self._extract_objectives(
                benchmark_result=res,
                optimization_config=trial_config.optimization_config,
            )
            all_objective_values[concurrency_level] = objective_value
            all_detailed_metrics[concurrency_level] = res

        execution_info.mark_completed(status="success")
        self.loggers.flush_all(trial_config.trial_id, current_study_name)

        return MultiConccurencyTrialResult(
            trial_id=trial_config.trial_id,
            trial_number=trial_config.trial_number,
            concurrencies=DEFAULT_CONCURRENCY_LIST,
            objective_values=all_objective_values,
            detailed_metrics=all_detailed_metrics,
            execution_info=execution_info,
            success=True,
        )

    def _extract_objectives(
        self,
        benchmark_result: dict[str, Any],
        optimization_config: OptimizationConfig | None,
    ) -> list[float]:
        """
        Extract objective values for Optuna from benchmark results based on optimization
        config.
        """
        if optimization_config is None or optimization_config.objectives is None:
            # fallback to throughput
            throughput = benchmark_result.get("output_tokens_per_second", 0.0)
            return [throughput]

        # objective_values: list[float] = []
        objective_values: list[float] = []
        for idx in range(len(optimization_config.objectives)):
            metric_key = optimization_config.get_metric_key(idx)
            # Extract the value from benchmark results - FAIL HARD if missing
            value = benchmark_result.get(metric_key, None)
            if value is None:
                raise RuntimeError(
                    f"Metric '{metric_key}' not found in benchmark results. "
                    f"Available metrics: {list(benchmark_result.keys())}"
                )
            value = float(value)
            objective_values.append(value)

        return objective_values

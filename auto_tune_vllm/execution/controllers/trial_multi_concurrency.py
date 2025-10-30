"""Trial controller implementations for Ray and local execution."""

from __future__ import annotations

import copy
import logging
import os
import platform
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable

import ray
from ray.exceptions import GetTimeoutError
from typing_extensions import override

from ...benchmarks.providers import BenchmarkProvider, GuideLLMBenchmark
from ...core.trial import ExecutionInfo, TrialConfig, TrialResult
from ...logging.manager import CentralizedLogger
from ..vllmprocess import vLLMProcess
from .utils import classify_error, validate_environment

logger = logging.getLogger(__name__)


class TrialState(Enum):
    """States for trial execution state machine."""

    WAITING_FOR_VLLM = auto()
    RUNNING_BENCHMARK = auto()


class TrialController(ABC):
    """Abstract base for trial execution controllers."""

    @abstractmethod
    def run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Execute a single optimization trial."""
        pass

    @abstractmethod
    def cleanup_resources(self):
        """Clean up any resources (servers, processes, etc.)."""
        pass

    @abstractmethod
    def request_cancellation(self):
        """Request cancellation of the running trial (non-blocking)."""
        pass


class MultiConcurrencyTrialController(TrialController):
    """Base implementation with common trial execution logic."""

    def __init__(self):
        self.vllm_server: vLLMProcess | None = None
        self.benchmark_provider: BenchmarkProvider | None = None
        self._environment_validated: bool = False
        # Dict to hold trial-specific loggers
        self.trial_loggers: dict[str, logging.Logger] = {}
        self._benchmark_process = None  # Track running benchmark process
        # Flag for external cancellation requests
        self._cancellation_requested: bool = False

    def _validate_environment(self, trial_config: TrialConfig) -> None:
        """Validate that all required packages are available on this worker."""
        if self._environment_validated:
            return

        try:
            _ = validate_environment(trial_config)
            self._environment_validated = True
        except Exception as error:
            raise error

    def _setup_trial_logging(self, trial_config: TrialConfig):
        """Setup trial-specific loggers based on logging configuration."""
        if not trial_config.logging_config:
            # No specific logging config, use default loggers
            return

        try:
            # Initialize CentralizedLogger for this trial
            log_database_url = trial_config.logging_config.get("database_url")
            log_file_path = trial_config.logging_config.get("file_path")
            log_level = trial_config.logging_config.get("log_level", "INFO")

            if log_database_url or log_file_path:
                centralized_logger = CentralizedLogger(
                    study_name=trial_config.study_name,
                    pg_url=log_database_url,
                    file_path=log_file_path,
                    log_level=log_level,
                )

                # Get trial-specific loggers for different components
                self.trial_loggers["controller"] = centralized_logger.get_trial_logger(
                    trial_config.trial_id, "controller"
                )
                self.trial_loggers["vllm"] = centralized_logger.get_trial_logger(
                    trial_config.trial_id, "vllm"
                )
                self.trial_loggers["benchmark"] = centralized_logger.get_trial_logger(
                    trial_config.trial_id, "benchmark"
                )

                # Log trial start
                self.trial_loggers["controller"].info(
                    f"Starting trial {trial_config.trial_id}"
                )
                self.trial_loggers["controller"].info(
                    f"Parameters: {trial_config.parameters}"
                )

        except Exception as e:
            # Fallback to default logger if setup fails
            logger.warning(f"Failed to setup trial logging: {e}")

    def _get_trial_logger(self, component: str) -> logging.Logger:
        """
        Get trial logger for specific component.
        Fallback to default if not available.
        """
        return self.trial_loggers.get(component, logger)

    def _flush_logger_handlers(self, target_logger: logging.Logger):
        """
        Immediately flush all handlers for a specific logger.
        This ensures logs appear in real-time during critical operations like cleanup.
        """
        for handler in target_logger.handlers:
            try:
                handler.flush()
            except Exception as e:
                # Silently ignore flush errors to avoid breaking cleanup
                logger.debug(f"Failed to flush handler: {e}")

    def _flush_trial_logs(self, trial_id: str):
        """Flush any buffered logs for the trial to ensure all records are written."""
        try:
            # Flush trial-specific loggers if we have them
            for component_logger in self.trial_loggers.values():
                for handler in component_logger.handlers:
                    try:
                        handler.flush()
                    except Exception as e:
                        logger.debug(f"Failed to flush handler: {e}")

            # Also try to flush by logger name pattern (fallback)
            import logging

            study_name = getattr(self, "_current_study_name", None)
            if study_name:
                for component in ["controller", "vllm", "benchmark"]:
                    logger_name = f"study_{study_name}.{trial_id}.{component}"
                    trial_logger = logging.getLogger(logger_name)
                    for handler in trial_logger.handlers:
                        try:
                            handler.flush()
                        except Exception as e:
                            logger.debug(
                                f"Failed to flush handler for {logger_name}: {e}"
                            )
        except Exception as e:
            logger.debug(f"Error flushing trial logs: {e}")

    @override
    def request_cancellation(self):
        """Request cancellation of the running trial (non-blocking).

        This method can be called via Ray .remote() while run_trial is executing.
        It sets a flag that causes the trial to terminate gracefully.
        """
        controller_logger = self.trial_loggers.get("controller", logger)
        controller_logger.info(
            "!!! CANCELLATION REQUESTED - Terminating trial immediately !!!"
        )
        self._flush_logger_handlers(controller_logger)

        self._cancellation_requested = True

        # Immediately terminate benchmark if running
        if self.benchmark_provider:
            controller_logger.info("Cancellation: Terminating benchmark process...")
            self._flush_logger_handlers(controller_logger)
            try:
                self.benchmark_provider.terminate_benchmark()
                controller_logger.info("Cancellation: Benchmark terminated")
            except Exception as e:
                controller_logger.warning(
                    f"Cancellation: Error terminating benchmark: {e}"
                )
            self._flush_logger_handlers(controller_logger)

    @override
    def run_trial(
        self, trial_config: TrialConfig, cancellation_flag_actor=None
    ) -> TrialResult:
        """Execute trial with proper error handling and cleanup.

        Args:
            trial_config: Configuration for this trial
            cancellation_flag_actor: Optional Ray actor that holds cancellation state.
                                    Can be checked via .is_cancelled().remote()
        """
        execution_info = ExecutionInfo()
        controller_logger = self._get_trial_logger("controller")
        controller_logger.info(
            f"Running trial {trial_config.trial_id} "
            f"with parameters: {trial_config.parameters}"
        )
        controller_logger.info(f"Study name: {trial_config.study_name}")

        try:
            self._current_study_name: str = trial_config.study_name
            self._trial_context: dict[str, Any] = {
                "study_name": trial_config.study_name,
                "trial_id": trial_config.trial_id,
            }
            self._setup_trial_logging(trial_config)
            self._validate_environment(trial_config)

            self.benchmark_provider = self._create_benchmark_provider(trial_config)

            def should_cancel():
                """Check if cancellation was requested.

                Works with both Ray actor and local flag.
                """
                if cancellation_flag_actor:
                    try:
                        # Use ray.get() with a small timeout to check cancellation
                        # This ensures the remote call has time to complete
                        is_cancelled = ray.get(
                            cancellation_flag_actor.is_cancelled.remote(), timeout=0.2
                        )
                        return is_cancelled
                    except (Exception, GetTimeoutError) as _:
                        return False
                return self._cancellation_requested

            # MULTI-CONCURRENCY BENCHMARK SWEEP
            # Start vLLM once, run benchmarks at multiple concurrency levels
            controller_logger.info("Starting multi-concurrency benchmark sweep")

            # Start vLLM server using vLLMProcess
            controller_logger.info("Starting vLLM server")
            execution_info.mark_vllm_started()
            vllm_start_time = time.time()

            self.vllm_server = vLLMProcess(
                model=trial_config.benchmark_config.model,
                vllm_args=trial_config.vllm_args,
                env_vars=trial_config.environment_vars,
                startup_timeout=trial_config.vllm_startup_timeout,
                health_check_interval=trial_config.health_check_interval,
                health_check_max_failures=trial_config.health_check_max_failures,
            )
            execution_info.worker_node_id = self._get_worker_id()
            execution_info.mark_vllm_ready()

            server_url = self.vllm_server.get_url_for("v1")
            controller_logger.info(f"vLLM server ready at {server_url}")

            # Extract concurrency levels for multi-concurrency sweep
            concurrency_levels = getattr(
                trial_config.benchmark_config, "concurrency_levels", [4, 8, 16, 32]
            )
            if not concurrency_levels or not isinstance(concurrency_levels, list):
                concurrency_levels = [1]
            controller_logger.info(
                f"Running benchmark sweep with concurrency levels: {concurrency_levels}"
            )

            # Run benchmark for each concurrency level
            all_objective_values = []
            all_detailed_metrics = {}

            for concurrency_idx, concurrency_level in enumerate(concurrency_levels):
                controller_logger.info(
                    f"Starting benchmark {concurrency_idx + 1}/{len(concurrency_levels)} "
                    f"with concurrency={concurrency_level}"
                )

                # Clone benchmark config and set concurrency
                modified_benchmark_config = copy.deepcopy(trial_config.benchmark_config)
                modified_benchmark_config.rate = concurrency_level
                modified_benchmark_config.concurrency = concurrency_level

                # Start benchmark
                execution_info.mark_benchmark_started()
                benchmark_logger: logging.Logger = self._get_trial_logger("benchmark")

                if hasattr(self.benchmark_provider, "set_logger"):
                    self.benchmark_provider.set_logger(benchmark_logger)

                if hasattr(self.benchmark_provider, "set_trial_context"):
                    self.benchmark_provider.set_trial_context(
                        trial_config.study_name, trial_config.trial_id
                    )

                benchmark_process = self.benchmark_provider.start_benchmark(
                    server_url, modified_benchmark_config
                )
                benchmark_start_time = time.time()

                # Poll until benchmark completes
                poll_count = 0
                poll_interval = 0.5
                while True:
                    poll_count += 1

                    # Check for cancellation
                    self._check_cancellation(
                        should_cancel,
                        poll_count,
                        TrialState.RUNNING_BENCHMARK,
                        vllm_start_time,
                        benchmark_process,
                        controller_logger,
                    )

                    # Check if benchmark completed
                    result = self._handle_benchmark_running(
                        benchmark_process,
                        benchmark_start_time,
                        trial_config,
                        execution_info,
                        controller_logger,
                    )

                    if result:  # Benchmark completed
                        controller_logger.info(
                            f"Concurrency {concurrency_level} completed: "
                            f"{len(result.objective_values)} objectives, "
                            f"{len(result.detailed_metrics)} metrics"
                        )
                        all_objective_values.append(result.objective_values)
                        all_detailed_metrics[f"concurrency_{concurrency_level}"] = (
                            result.detailed_metrics
                        )
                        execution_info.mark_benchmark_completed()
                        break

                    # Still running
                    time.sleep(poll_interval)

            # All concurrency levels complete - validate results
            if not all_objective_values:
                raise RuntimeError(
                    "No benchmark results collected - all concurrency levels failed"
                )

            execution_info.mark_completed(status="success")
            controller_logger.info(
                f"Completed {len(all_objective_values)}/{len(concurrency_levels)} "
                f"concurrency levels successfully"
            )
            controller_logger.info(
                f"Result structure: {len(all_objective_values)} concurrency levels, "
                f"{len(all_objective_values[0]) if all_objective_values else 0} objectives each"
            )

            return TrialResult(
                trial_id=trial_config.trial_id,
                trial_number=trial_config.trial_number,
                trial_type=trial_config.trial_type,
                objective_values=all_objective_values,
                detailed_metrics=all_detailed_metrics,
                execution_info=execution_info,
                success=True,
            )

        except KeyboardInterrupt as e:
            execution_info.mark_completed()
            controller_logger.warning(f"Trial {trial_config.trial_id} cancelled: {e}")

            return TrialResult(
                trial_id=trial_config.trial_id,
                trial_number=trial_config.trial_number,
                trial_type=trial_config.trial_type,
                objective_values=[],
                detailed_metrics={},
                execution_info=execution_info,
                success=False,
                error_message=f"Trial cancelled: {e}",
                error_type="Cancelled",
            )
        except Exception as e:
            # Check if this is a Ray cancellation exception
            exception_name = type(e).__name__
            if "Cancel" in exception_name or "cancel" in str(e).lower():
                execution_info.mark_completed()
                controller_logger.warning(
                    f"Trial {trial_config.trial_id} cancelled by Ray: {e}"
                )

                return TrialResult(
                    trial_id=trial_config.trial_id,
                    trial_number=trial_config.trial_number,
                    trial_type=trial_config.trial_type,
                    objective_values=[],
                    detailed_metrics={},
                    execution_info=execution_info,
                    success=False,
                    error_message=f"Trial cancelled: {e}",
                    error_type="Cancelled",
                )

            # Handle other exceptions normally
            # determine failure type based on error message
            error_str = str(e)
            status = (
                "vllm_crash"
                if "vLLM" in error_str or "health" in error_str
                else "benchmark_crash"
            )
            # Mark benchmark as completed (safe to call even if not started)
            execution_info.mark_benchmark_completed()
            execution_info.mark_completed(status=status)
            controller_logger.error(f"Trial {trial_config.trial_id} failed: {e}")

            # Classify error for database storage
            error_type = classify_error(e)

            return TrialResult(
                trial_id=trial_config.trial_id,
                trial_number=trial_config.trial_number,
                trial_type=trial_config.trial_type,
                objective_values=[],
                detailed_metrics={},
                execution_info=execution_info,
                success=False,
                error_message=str(e),
                error_type=error_type,
            )
        finally:
            # Flush any buffered logs before cleanup
            self._flush_trial_logs(trial_config.trial_id)
            self.cleanup_resources()

    def _create_benchmark_provider(
        self, trial_config: TrialConfig
    ) -> BenchmarkProvider:
        """Create appropriate benchmark provider."""
        benchmark_type = trial_config.benchmark_config.benchmark_type

        if benchmark_type == "guidellm":
            return GuideLLMBenchmark()
        else:
            raise NotImplementedError

    def _log_python_environment(self, logger: logging.Logger):
        """Log Python environment information for debugging."""

        logger.info("=" * 60)
        logger.info("PYTHON ENVIRONMENT INFORMATION")
        logger.info("=" * 60)

        # Python executable and version
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")

        # Virtual environment detection
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            logger.info(f"Virtual environment: {sys.prefix}")
            logger.info(f"Base Python: {sys.base_prefix}")
        else:
            logger.info("Virtual environment: None (using system Python)")

        # Python path
        logger.info(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

        # Environment variables

        env_vars = [
            "VIRTUAL_ENV",
            "CONDA_DEFAULT_ENV",
            "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES",
        ]
        for var in env_vars:
            value = os.environ.get(var, "Not set")
            logger.info(f"{var}: {value}")

        # Ray GPU resources and accelerator information
        try:
            if ray.is_initialized():
                cluster_resources = ray.cluster_resources()
                available_resources = ray.available_resources()

                # Log GPU resources
                gpu_count = cluster_resources.get("GPU", 0)
                available_gpus = available_resources.get("GPU", 0)
                logger.info(
                    f"Ray GPU resources: {gpu_count} total, {available_gpus} available"
                )

                # Log accelerator types (GPU models)
                accelerator_types = [
                    k
                    for k in cluster_resources.keys()
                    if k.startswith("accelerator_type:")
                ]
                if accelerator_types:
                    logger.info("GPU accelerator types:")
                    for acc_type in accelerator_types:
                        acc_name = acc_type.replace("accelerator_type:", "")
                        acc_count = cluster_resources[acc_type]
                        logger.info(f"  - {acc_name}: {acc_count}")
                else:
                    logger.info("No accelerator type information available")

                # Log assigned GPUs for this worker (from environment)
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cuda_visible:
                    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
                else:
                    logger.info("CUDA_VISIBLE_DEVICES: Not set")
            else:
                logger.info(
                    "Ray: Not initialized - cannot get GPU resource information"
                )
        except Exception as e:
            logger.info(f"Ray GPU detection error: {e}")

        # Get vLLM version from CLI command
        try:
            result = subprocess.run(
                ["vllm", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # vLLM CLI returns just the version number (e.g., "0.10.1.1")
                version_output = result.stdout.strip()
                logger.info(f"vLLM version: {version_output}")
            else:
                logger.info(
                    f"vLLM: Error getting version (exit code {result.returncode})"
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.info("vLLM: Not installed or not in PATH")
        except Exception as e:
            logger.info(f"vLLM: Error checking version: {e}")

        # Ray worker info (if available)
        try:
            if ray.is_initialized():
                runtime_ctx = ray.get_runtime_context()
                logger.info(f"Ray node ID: {runtime_ctx.get_node_id()}")
                logger.info(f"Ray worker ID: {runtime_ctx.get_worker_id()}")
                logger.info(f"Ray job ID: {runtime_ctx.get_job_id()}")
        except ImportError:
            logger.info("Ray: Not available")
        except Exception:
            logger.info("Ray: Available but not initialized")

        logger.info("=" * 60)

    def _check_cancellation(
        self,
        should_cancel: Callable[[], bool],
        poll_count: int,
        state: TrialState,
        vllm_start_time: float,
        benchmark_process,
        controller_logger: logging.Logger,
    ):
        """Check for cancellation request and handle cleanup if cancelled."""
        is_cancelled = should_cancel()

        # Log every 5th check to verify mechanism is working
        if poll_count % 5 == 0:
            controller_logger.debug(f"Cancellation check #{poll_count}: {is_cancelled}")

        # Log progress periodically
        if poll_count % 20 == 0:  # Every 10 seconds
            elapsed_total = time.time() - vllm_start_time
            logger.debug(
                f"Main loop iteration {poll_count}, elapsed: {elapsed_total:.1f}s, "
                f"state: {state.name}"
            )

        if is_cancelled:
            controller_logger.warning(
                "!!! CANCELLATION DETECTED IN MAIN LOOP - Terminating trial !!!"
            )
            controller_logger.info(f"Trial was in state: {state.name}")
            controller_logger.info(f"Detection occurred at iteration: {poll_count}")
            self._flush_logger_handlers(controller_logger)

            # Cleanup based on current state
            if benchmark_process and benchmark_process.poll() is None:
                controller_logger.info("Terminating running benchmark process...")
                if self.benchmark_provider:
                    self.benchmark_provider.terminate_benchmark()

            raise KeyboardInterrupt(f"Trial cancelled while {state.name}")

    def _handle_benchmark_running(
        self,
        benchmark_process,
        benchmark_start_time: float,
        trial_config: TrialConfig,
        execution_info,
        logger,
    ):
        """Handle benchmark running state.

        Returns TrialResult on completion, None otherwise.
        """
        # Check if benchmark completed
        returncode = benchmark_process.poll()
        if returncode is not None:
            logger.debug(f"Benchmark process completed with return code {returncode}")

            # Get benchmark output and parse results
            stdout, stderr = benchmark_process.communicate(timeout=5)

            if returncode != 0:
                raise RuntimeError(
                    f"Benchmark failed with exit code {returncode}: {stderr}"
                )

            # Parse benchmark results
            benchmark_result = self.benchmark_provider.parse_results()

            # Check if vLLM server died during benchmark
            if self.vllm_server and not self.vllm_server.is_healthy():
                failure_reason = self.vllm_server.get_failure_reason()
                raise RuntimeError(f"vLLM server health check failed: {failure_reason}")

            # Extract objectives
            objective_values = self._extract_objectives(
                benchmark_result, trial_config.optimization_config
            )
            logger.info(f"Trial completed with objectives: {objective_values}")

            return TrialResult(
                trial_id=trial_config.trial_id,
                trial_number=trial_config.trial_number,
                trial_type=trial_config.trial_type,
                objective_values=objective_values,
                detailed_metrics=benchmark_result,
                execution_info=execution_info,
                success=True,
            )

        # Check benchmark timeout
        elapsed = time.time() - benchmark_start_time
        max_benchmark_time = trial_config.benchmark_config.max_seconds * 1.5
        if elapsed > max_benchmark_time:
            logger.warning(f"Benchmark timeout after {elapsed:.1f}s, terminating...")
            if hasattr(self.benchmark_provider, "terminate_benchmark"):
                self.benchmark_provider.terminate_benchmark()
            raise RuntimeError(f"Benchmark timed out after {max_benchmark_time}s")

        return None  # Still running, continue polling

    def _extract_objectives(
        self, benchmark_result: dict, optimization_config=None
    ) -> list[float]:
        """
        Extract objective values for Optuna from benchmark results based on optimization
        config.
        """
        if optimization_config is None:
            # Fallback to default behavior
            throughput = benchmark_result.get("output_tokens_per_second", 0.0)
            return [throughput]

        objective_values = []

        for _ in optimization_config.objectives:
            # Get the metric key with percentile if specified
            metric_key = optimization_config.get_metric_key(len(objective_values))

            # Extract the value from benchmark results - FAIL HARD if missing
            value = benchmark_result.get(metric_key)

            if value is None:
                raise RuntimeError(
                    f"Metric '{metric_key}' not found in benchmark results. "
                    f"Available metrics: {list(benchmark_result.keys())}"
                )

            # Convert to float and handle potential conversion errors
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise RuntimeError(
                    f"Failed to convert metric '{metric_key}' value '{value}' to float"
                )

            objective_values.append(value)

        return objective_values

    @override
    def cleanup_resources(self):
        """Clean up vLLM server process and health monitoring."""
        # Use trial-specific logger if available, otherwise fall back to module logger
        controller_logger = self._get_trial_logger("controller")

        controller_logger.info(
            "!!! Trial Controller: Received cleanup request from backend !!!"
        )

        # IMMEDIATE FLUSH: Ensure user sees cleanup starting in real-time
        self._flush_logger_handlers(controller_logger)

        # Terminate any running benchmark process
        if self.benchmark_provider:
            try:
                controller_logger.info(
                    "Trial Controller: Terminating benchmark process..."
                )
                self._flush_logger_handlers(controller_logger)
                self.benchmark_provider.terminate_benchmark()
                controller_logger.info("Trial Controller: Benchmark process terminated")
                self._flush_logger_handlers(controller_logger)
            except Exception as e:
                controller_logger.warning(
                    f"Trial Controller: Error terminating benchmark process: {e}"
                )
                self._flush_logger_handlers(controller_logger)

        # Stop vLLM server
        if self.vllm_server:
            controller_logger.info("Trial Controller: Stopping vLLM server...")
            self._flush_logger_handlers(controller_logger)
            self.vllm_server.stop(timeout=10)
            self.vllm_server = None
            controller_logger.info("Trial Controller: vLLM server stopped")
            self._flush_logger_handlers(controller_logger)
        else:
            controller_logger.debug("Trial Controller: No vLLM server to cleanup")

        # Flush all trial logs to ensure cleanup messages are written
        controller_logger.info("Trial Controller: Cleanup complete, flushing logs...")
        for component_logger in self.trial_loggers.values():
            for handler in component_logger.handlers:
                try:
                    handler.flush()
                except Exception as e:
                    # Use module logger as fallback since trial logger might be affected
                    logger.debug(f"Failed to flush handler during cleanup: {e}")
        controller_logger.info("Trial Controller: Log flush complete")

    @abstractmethod
    def _get_worker_id(self) -> str:
        """Get worker identifier (Ray node ID or local machine info)."""
        pass


class RayWorkerMultiConcurrencyTrialController(MultiConcurrencyTrialController):
    """Ray worker node trial controller with Ray-specific functionality."""

    def _get_worker_id(self) -> str:
        """Get Ray worker node ID."""
        try:
            return ray.get_runtime_context().get_node_id()
        except Exception:
            return "ray_worker_unknown"


# Ray remote actor wrapper
@ray.remote
class RayMultiConcurrencyTrialActor(RayWorkerMultiConcurrencyTrialController):
    """Ray remote actor for distributed trial execution."""

    def run_trial(
        self, trial_config: TrialConfig, cancellation_flag_actor=None
    ) -> TrialResult:
        """Run trial on Ray worker with optional cancellation flag actor."""
        return super().run_trial(trial_config, cancellation_flag_actor)

    def __del__(self):
        """Ensure cleanup on actor destruction."""
        self.cleanup_resources()

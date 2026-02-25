"""Trial controller implementations for Ray and local execution."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, Optional

try:
    import ray
    from ray.exceptions import GetTimeoutError

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    GetTimeoutError = Exception  # Fallback exception

from ..benchmarks.providers import BenchmarkProvider, GuideLLMBenchmark, MLPerfBenchmark
from ..core.trial import ExecutionInfo, TrialConfig, TrialResult
from ..logging.manager import CentralizedLogger

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


class BaseTrialController(TrialController):
    """Base implementation with common trial execution logic."""

    # Error classification patterns for database storage
    # Maps error types to keyword patterns used by _classify_error()
    ERROR_PATTERNS = {
        "OOM": ["out of memory", "outofmemoryerror", "memory allocation failed"],
        "GPU_Memory": [
            "gpu memory",
            "free memory on device",
            "insufficient gpu memory",
        ],
        "Timeout": ["timeout", "timed out"],
        "CUDA_Error": ["cuda error", "cuda runtime error"],
        "Connection_Error": ["connection refused", "connection reset"],
        "Server_Startup": ["server startup", "failed to start", "died during startup"],
        "Benchmark_Error": ["benchmark", "guidellm"],
    }

    def __init__(self):
        self.vllm_process: Optional[subprocess.Popen] = None
        self.benchmark_provider: Optional[BenchmarkProvider] = None
        self._environment_validated = False
        self.trial_loggers = {}  # Dict to hold trial-specific loggers
        self._health_monitor_thread = None
        self._health_monitor_stop = False
        self._health_monitor_stop_event = None
        self._health_check_url = None
        self._health_check_failed = False
        self._health_check_failure_reason = None
        self._benchmark_process = None  # Track running benchmark process
        self._cancellation_requested = False  # Flag for external cancellation requests

    def _validate_environment(self, trial_config: Optional[TrialConfig] = None) -> None:
        """Validate that all required packages are available on this worker."""
        if self._environment_validated:
            return

        required_packages = {
            "vllm": "vLLM serving framework",
            "guidellm": "GuideLLM benchmarking tool",
            "optuna": "Optuna optimization framework",
            "ray": "Ray distributed computing",
        }

        # Only require psycopg2 if using PostgreSQL
        using_postgresql = False
        if trial_config and trial_config.logging_config:
            using_postgresql = (
                trial_config.logging_config.get("database_url") is not None
            )

        if using_postgresql:
            required_packages["psycopg2"] = "PostgreSQL client"

        missing_packages = []

        for package, description in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(f"{package} ({description})")

        if missing_packages:
            missing_list = "\n  - ".join(missing_packages)
            raise RuntimeError(
                f"Missing required packages on Ray worker node:\n  - {missing_list}\n\n"
                f"Ray worker nodes must have the same Python environment as the head node.\n"  # noqa: E501
                f"Install auto-tune-vllm on all Ray cluster nodes:\n"
                f"  pip install auto-tune-vllm"
            )

        # Check if commands are available in PATH
        required_commands = {
            "python3": "Python interpreter",
            "guidellm": "GuideLLM CLI tool",
        }

        import shutil

        missing_commands = []

        for command, description in required_commands.items():
            if not shutil.which(command):
                missing_commands.append(f"{command} ({description})")

        if missing_commands:
            missing_list = "\n  - ".join(missing_commands)
            raise RuntimeError(
                f"Missing required commands in PATH on Ray worker node:\n"
                f"  - {missing_list}\n\n"
                f"Ensure all dependencies are properly installed and available in PATH."
            )

        # Check GPU availability using Ray cluster resources
        try:
            if ray.is_initialized():
                cluster_resources = ray.cluster_resources()
                available_resources = ray.available_resources()

                gpu_count = cluster_resources.get("GPU", 0)
                available_gpus = available_resources.get("GPU", 0)

                if gpu_count > 0:
                    logger.info(
                        f"Ray cluster has {gpu_count} GPU(s) total, "
                        f"{available_gpus} available"
                    )

                    # Log accelerator types if available
                    accelerator_types = [
                        k
                        for k in cluster_resources.keys()
                        if k.startswith("accelerator_type:")
                    ]
                    if accelerator_types:
                        for acc_type in accelerator_types:
                            acc_name = acc_type.replace("accelerator_type:", "")
                            acc_count = cluster_resources[acc_type]
                            logger.info(f"GPU type: {acc_name} (count: {acc_count})")
                else:
                    logger.warning(
                        "No GPUs detected in Ray cluster. vLLM may fail to start."
                    )
            else:
                logger.warning("Ray not initialized. Cannot check GPU availability.")
        except Exception as e:
            logger.warning(f"Could not check GPU availability from Ray: {e}")

        self._environment_validated = True
        logger.info("Environment validation passed on Ray worker")

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

    def _get_trial_logger(self, component: str):
        """
        Get trial logger for specific component.
        Fallback to default if not available.
        """
        return self.trial_loggers.get(component, logger)

    def _flush_logger_handlers(self, target_logger):
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

    def request_cancellation(self):
        """Request cancellation of the running trial (non-blocking).

        This method can be called via Ray .remote() while run_trial is executing.
        It sets a flag that causes the trial to terminate gracefully.
        """
        controller_logger = self._get_trial_logger("controller")
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
            # Store study name for log flushing
            self._current_study_name = trial_config.study_name

            # Store trial context for benchmark subprocess
            self._trial_context = {
                "study_name": trial_config.study_name,
                "trial_id": trial_config.trial_id,
            }

            # Setup trial-specific logging first
            self._setup_trial_logging(trial_config)

            # Validate environment first
            self._validate_environment(trial_config)

            # Setup benchmark provider
            self.benchmark_provider = self._create_benchmark_provider(trial_config)

            # Setup cancellation checker function
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

            # UNIFIED EXECUTION LOOP - Handles vLLM startup AND benchmark
            # while allowing for cancellation at any point
            controller_logger.info(
                "Starting unified execution loop (vLLM startup + benchmark)"
            )

            # Start vLLM server
            controller_logger.info("Starting vLLM server")
            execution_info.mark_vllm_started()
            server_info = self._start_vllm_server(trial_config)
            execution_info.worker_node_id = self._get_worker_id()

            # State machine for trial execution
            state = TrialState.WAITING_FOR_VLLM
            benchmark_process = None
            vllm_start_time = time.time()
            benchmark_start_time = None
            poll_interval = 0.5  # 500ms
            poll_count = 0

            # Wait for server to be ready
            controller_logger.info(
                f"Waiting for server at {server_info['url']} to be ready "
                f"(timeout: {trial_config.vllm_startup_timeout}s)"
            )
            self._wait_for_server_ready(
                server_info["url"], trial_config.vllm_startup_timeout
            )
            execution_info.mark_vllm_ready()

            # Main execution loop - concise with extracted state handlers
            while True:
                poll_count += 1

                # Check for cancellation (every iteration)
                self._check_cancellation(
                    should_cancel,
                    poll_count,
                    state,
                    vllm_start_time,
                    benchmark_process,
                    controller_logger,
                )

                # Handle current state
                if state == TrialState.WAITING_FOR_VLLM:
                    result = self._handle_vllm_startup(
                        trial_config, server_info, vllm_start_time, controller_logger
                    )
                    if result:  # vLLM is ready, transition to benchmark
                        benchmark_process, benchmark_start_time = result
                        execution_info.mark_benchmark_started()
                        state = TrialState.RUNNING_BENCHMARK
                        controller_logger.debug(
                            f"State transition: {TrialState.WAITING_FOR_VLLM.name} "
                            f"→ {TrialState.RUNNING_BENCHMARK.name}"
                        )
                        continue  # Skip sleep, start benchmark immediately

                elif state == TrialState.RUNNING_BENCHMARK:
                    result = self._handle_benchmark_running(
                        benchmark_process,
                        benchmark_start_time,
                        trial_config,
                        execution_info,
                        controller_logger,
                    )
                    if result:  # Benchmark completed successfully
                        controller_logger.debug(
                            f"Benchmark handler returned result: "
                            f"success={result.success}, "
                            f"objectives={result.objective_values}"
                        )
                        execution_info.mark_benchmark_completed()
                        execution_info.mark_completed(status="success")
                        controller_logger.info(
                            f"Returning successful trial result with "
                            f"{len(result.objective_values)} objectives"
                        )
                        return result
                    # If None, benchmark still running - continue polling
                    controller_logger.debug(
                        f"Benchmark still running... "
                        f"(elapsed: {time.time() - benchmark_start_time:.1f}s)"
                    )

                # Sleep before next poll
                time.sleep(poll_interval)

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
            # Mark benchmark as completed if we were running it
            if state == TrialState.RUNNING_BENCHMARK:
                execution_info.mark_benchmark_completed()
            execution_info.mark_completed(status=status)
            controller_logger.error(f"Trial {trial_config.trial_id} failed: {e}")

            # Classify error for database storage
            error_type = self._classify_error(e)

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

    def _classify_error(self, exception: Exception) -> str:
        """Classify error type based on exception message.

        Uses ERROR_PATTERNS dictionary to categorize exceptions for
        structured failure analysis in the database.

        Returns:
            Error type string (e.g., "OOM", "Timeout") or "Unknown"
        """
        error_message = str(exception).lower()

        for error_type, patterns in self.ERROR_PATTERNS.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type

        return "Unknown"

    def _create_benchmark_provider(
        self, trial_config: TrialConfig
    ) -> BenchmarkProvider:
        """Create appropriate benchmark provider."""
        benchmark_type = trial_config.benchmark_config.benchmark_type

        if benchmark_type == "guidellm":
            return GuideLLMBenchmark()
        elif benchmark_type == "mlperf":
            return MLPerfBenchmark()
        else:
            # Import custom provider by name
            # This enables extensibility for custom benchmarks
            return self._import_custom_benchmark(benchmark_type)

    def _import_custom_benchmark(self, benchmark_type: str) -> BenchmarkProvider:
        """Dynamically import custom benchmark provider."""
        try:
            # Try to import from benchmarks.custom module
            module_name = f"auto_tune_vllm.benchmarks.custom.{benchmark_type}"
            module = __import__(module_name, fromlist=[benchmark_type])
            provider_class = getattr(module, f"{benchmark_type.title()}Benchmark")
            return provider_class()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unknown benchmark provider: {benchmark_type}") from e

    def _log_python_environment(self, logger):
        """Log Python environment information for debugging."""
        import platform
        import sys

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
        import os

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
        should_cancel,
        poll_count: int,
        state: TrialState,
        vllm_start_time: float,
        benchmark_process,
        controller_logger,
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
                if self.benchmark_provider and hasattr(
                    self.benchmark_provider, "terminate_benchmark"
                ):
                    self.benchmark_provider.terminate_benchmark()

            raise KeyboardInterrupt(f"Trial cancelled while {state.name}")

    def _handle_vllm_startup(
        self,
        trial_config: TrialConfig,
        server_info: dict,
        vllm_start_time: float,
        logger,
    ):
        """Handle vLLM startup state.

        Returns (benchmark_process, start_time) on success, None otherwise.
        """
        import requests

        # Check timeout
        elapsed = time.time() - vllm_start_time
        if elapsed > trial_config.vllm_startup_timeout:
            raise RuntimeError(
                f"vLLM server failed to start within "
                f"{trial_config.vllm_startup_timeout}s"
            )

        # Check if vLLM process died
        if self.vllm_process and self.vllm_process.poll() is not None:
            raise RuntimeError(
                f"vLLM process died during startup with exit code "
                f"{self.vllm_process.returncode}"
            )

        # Check if server is ready
        try:
            health_url = server_info["url"].replace("/v1", "/health")
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                logger.info(
                    f"vLLM server ready at {server_info['url']} (took {elapsed:.1f}s)"
                )

                # Start health monitoring
                logger.info("Starting runtime health monitoring")
                self._start_health_monitoring(
                    health_url,
                    check_interval=trial_config.health_check_interval,
                    max_failures=trial_config.health_check_max_failures,
                )

                # Setup and start benchmark
                logger.info("Starting benchmark run")
                benchmark_logger = self._get_trial_logger("benchmark")

                if hasattr(self.benchmark_provider, "set_logger"):
                    self.benchmark_provider.set_logger(benchmark_logger)

                if hasattr(self.benchmark_provider, "set_trial_context"):
                    self.benchmark_provider.set_trial_context(
                        trial_config.study_name, trial_config.trial_id
                    )

                # Start benchmark as subprocess
                benchmark_process = self.benchmark_provider.start_benchmark(
                    server_info["url"], trial_config.benchmark_config
                )
                return benchmark_process, time.time()

        except requests.exceptions.RequestException as e:
            # Health check failed, log and continue polling
            logger.debug(f"Health check failed: {e}")
        except Exception as e:
            # Other errors during benchmark setup - fatal
            logger.error(f"Error setting up benchmark: {e}")
            raise

        return None  # Not ready yet, continue polling

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
            self._check_health_status()

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

    def _start_vllm_server(self, trial_config: TrialConfig) -> dict:
        """Start vLLM server with trial parameters."""
        port = self._get_available_port()
        vllm_logger = self._get_trial_logger("vllm")

        # Log Python environment information for debugging
        self._log_python_environment(vllm_logger)

        # Build vLLM command
        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            trial_config.benchmark_config.model,
            "--port",
            str(port),
            "--host",
            "0.0.0.0",
        ]

        # Add trial-specific parameters
        cmd.extend(trial_config.vllm_args)

        vllm_logger.info(f"Starting vLLM server: {' '.join(cmd)}")

        # Prepare environment variables
        env = os.environ.copy()
        trial_env_vars = trial_config.environment_vars
        if trial_env_vars:
            env.update(trial_env_vars)
            vllm_logger.info(f"Environment variables: {trial_env_vars}")

        # Start process with captured output
        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env=env,  # Pass environment variables to the process
            start_new_session=True,  # Put child in its own process group/session
        )

        # Start a thread to capture and log vLLM output
        import threading

        def log_vllm_output():
            try:
                for line in iter(self.vllm_process.stdout.readline, ""):
                    if line.strip():
                        vllm_logger.info(line.strip())
            except Exception as e:
                vllm_logger.error(f"Error capturing vLLM output: {e}")

        log_thread = threading.Thread(target=log_vllm_output, daemon=True)
        log_thread.start()

        return {
            "port": port,
            "url": f"http://localhost:{port}/v1",
            "pid": self.vllm_process.pid,
        }

    def _get_available_port(self) -> int:
        """Get an available port for vLLM server."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]

        return port

    def _wait_for_server_ready(self, url: str, timeout: int = 300):
        """Wait for vLLM server to be ready."""
        import requests

        vllm_logger = self._get_trial_logger("vllm")
        start_time = time.time()
        health_url = url.replace("/v1", "/health")

        vllm_logger.info(
            f"Waiting for vLLM server to be ready at {health_url} (timeout: {timeout}s)"
        )

        while time.time() - start_time < timeout:
            # Check if vLLM process has died during startup
            if self.vllm_process and self.vllm_process.poll() is not None:
                vllm_logger.error(
                    f"vLLM process died during startup with exit code "
                    f"{self.vllm_process.returncode}"
                )
                raise RuntimeError(
                    f"vLLM process died during startup with exit code "
                    f"{self.vllm_process.returncode}"
                )

            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    vllm_logger.info(f"vLLM server ready at {url}")
                    return
            except requests.exceptions.RequestException as e:
                vllm_logger.debug(f"Health check failed: {e}")

            time.sleep(2)

        vllm_logger.error(f"vLLM server failed to start within {timeout} seconds")
        raise RuntimeError(f"vLLM server failed to start within {timeout} seconds")

    def _start_health_monitoring(
        self, health_url: str, check_interval: float = 1.0, max_failures: int = 3
    ):
        """
        Start background health monitoring of vLLM server.

        Args:
            health_url: URL to check for health status
            check_interval: Seconds between health checks (default: 30)
            max_failures: Number of consecutive failures before marking
                as dead (default: 3)

        Environment Variables:
            VLLM_HEALTH_CHECK_DEBUG: Set to '1' or 'true' to enable verbose logging
        """
        import threading

        import requests

        # TODO: Remove debug logging after verifying health monitoring works
        # Set VLLM_HEALTH_CHECK_DEBUG=1 to enable verbose logging
        debug = os.environ.get("VLLM_HEALTH_CHECK_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )

        self._health_check_url = health_url
        self._health_monitor_stop = False
        self._health_check_failed = False
        self._health_check_failure_reason = None

        vllm_logger = self._get_trial_logger("vllm")

        def monitor_health():
            consecutive_failures = 0
            # Normalize polling period to a positive float and default to 1.0s
            try:
                period = float(check_interval)
            except Exception:
                period = 1.0
            if period <= 0:
                period = 1.0
            # Detect if we're using Kubernetes backend
            use_k8s_monitoring = hasattr(self, "deployment_name") and hasattr(
                self, "namespace"
            )

            if use_k8s_monitoring:
                vllm_logger.info(
                    f"Starting health monitoring: checking Deployment {self.deployment_name} pod status every {period}s"
                    + (" (DEBUG MODE: verbose logging enabled)" if debug else "")
                )
            else:
                vllm_logger.info(
                    f"Starting health monitoring: checking {health_url} every {period}s"
                    + (" (DEBUG MODE: verbose logging enabled)" if debug else "")
                )

            # Event used to interrupt waits for responsive shutdown
            import threading as _threading

            if self._health_monitor_stop_event is None:
                self._health_monitor_stop_event = _threading.Event()
            stop_event = self._health_monitor_stop_event

            # Schedule first run immediately, then maintain fixed cadence
            next_deadline = time.monotonic() + period

            while not self._health_monitor_stop and not stop_event.is_set():
                # Check if vLLM process itself has died
                if self.vllm_process and self.vllm_process.poll() is not None:
                    self._health_check_failed = True
                    self._health_check_failure_reason = (
                        f"vLLM process died unexpectedly with exit code "
                        f"{self.vllm_process.returncode}"
                    )
                    vllm_logger.error(
                        f"Health monitoring detected process death: "
                        f"exit code {self.vllm_process.returncode}"
                    )
                    # Terminate running benchmark immediately (if using local benchmark provider)
                    if self.benchmark_provider and hasattr(
                        self.benchmark_provider, "terminate_benchmark"
                    ):
                        self.benchmark_provider.terminate_benchmark()
                    break

                try:
                    # Kubernetes backend: Check pod status via API instead of HTTP
                    if use_k8s_monitoring:
                        from kubernetes import client, config

                        # Load kubeconfig
                        kubeconfig = getattr(self, "kubeconfig", None)
                        if kubeconfig:
                            config.load_kube_config(config_file=kubeconfig)
                        else:
                            try:
                                config.load_incluster_config()
                            except config.ConfigException:
                                config.load_kube_config()

                        core_v1 = client.CoreV1Api()

                        # Find pod for this deployment
                        # Deployments use labels: app=vllm-server, trial-id=trial-X
                        # NOTE: trial-id is sanitized (lowercase, underscores to hyphens)
                        # Try to get trial_id from instance variable (set in run_trial)
                        trial_id = getattr(self, "_current_trial_id", None)
                        if trial_id:
                            # Import sanitize function to match deployment's label format
                            from .k8s_utils import sanitize_k8s_name

                            sanitized_trial_id = sanitize_k8s_name(trial_id)
                            label_selector = (
                                f"app=vllm-server,trial-id={sanitized_trial_id}"
                            )
                        else:
                            # Fallback: use generic vllm-server label
                            label_selector = "app=vllm-server"

                        pods = core_v1.list_namespaced_pod(
                            namespace=self.namespace, label_selector=label_selector
                        )

                        # DEBUG: Log what we're searching for
                        if debug or consecutive_failures == 0:
                            vllm_logger.info(
                                f"[HEALTH CHECK DEBUG] Searching namespace={self.namespace}, "
                                f"label_selector={label_selector}, found {len(pods.items)} pods"
                            )

                        if not pods.items:
                            consecutive_failures += 1
                            vllm_logger.warning(
                                f"Health check failed: No pods found with labels {label_selector} "
                                f"(failure {consecutive_failures}/{max_failures})"
                            )
                        else:
                            pod = pods.items[0]  # Get first pod
                            pod_phase = pod.status.phase

                            # Check for error states
                            error_states = ["Failed", "Unknown"]
                            container_errors = []

                            if pod.status.container_statuses:
                                for container_status in pod.status.container_statuses:
                                    if (
                                        container_status.state
                                        and container_status.state.waiting
                                    ):
                                        if container_status.state.waiting.reason in [
                                            "CrashLoopBackOff",
                                            "Error",
                                            "ImagePullBackOff",
                                        ]:
                                            container_errors.append(
                                                f"{container_status.name}: {container_status.state.waiting.reason}"
                                            )
                                    elif (
                                        container_status.state
                                        and container_status.state.terminated
                                    ):
                                        if (
                                            container_status.state.terminated.exit_code
                                            != 0
                                        ):
                                            container_errors.append(
                                                f"{container_status.name}: terminated with exit code {container_status.state.terminated.exit_code}"
                                            )

                            if pod_phase in error_states or container_errors:
                                consecutive_failures += 1
                                error_msg = f"pod phase={pod_phase}"
                                if container_errors:
                                    error_msg += (
                                        f", containers={', '.join(container_errors)}"
                                    )
                                vllm_logger.warning(
                                    f"Health check failed: {error_msg} "
                                    f"(failure {consecutive_failures}/{max_failures})"
                                )
                            elif pod_phase == "Running":
                                # Pod is healthy
                                if debug:
                                    vllm_logger.info(
                                        f"[DEBUG] Health check PASSED: pod {pod.metadata.name} is Running, "
                                        f"consecutive_failures={consecutive_failures}"
                                    )
                                if consecutive_failures > 0:
                                    vllm_logger.info(
                                        f"Health check recovered after {consecutive_failures} failures"
                                    )
                                consecutive_failures = 0
                            else:
                                # Pod in transitional state (Pending, etc.) - don't count as failure
                                if debug:
                                    vllm_logger.info(
                                        f"[DEBUG] Health check: pod phase={pod_phase} (transitional, not counting as failure)"
                                    )

                    # Local/Docker backends: HTTP health check
                    else:
                        response = requests.get(health_url, timeout=5)
                        if response.status_code == 200:
                            # TODO: Remove debug logging after verifying health
                            # monitoring works
                            if debug:
                                vllm_logger.info(
                                    f"[DEBUG] Health check PASSED: "
                                    f"status={response.status_code}, "
                                    f"consecutive_failures={consecutive_failures}"
                                )
                            # Health check passed - reset failure counter
                            if consecutive_failures > 0:
                                vllm_logger.info(
                                    f"Health check recovered after "
                                    f"{consecutive_failures} failures"
                                )
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            vllm_logger.warning(
                                f"Health check returned status {response.status_code} "
                                f"(failure {consecutive_failures}/{max_failures})"
                            )
                except requests.exceptions.RequestException as e:
                    # HTTP health check failed (local/docker backends only)
                    consecutive_failures += 1
                    # TODO: Remove debug logging after verifying health
                    # monitoring works
                    log_msg = (
                        f"Health check failed: {e} "
                        f"(failure {consecutive_failures}/{max_failures})"
                    )
                    if debug:
                        log_msg = f"[DEBUG] Health check FAILED: {log_msg}"
                    vllm_logger.warning(log_msg)
                except Exception as e:
                    # Kubernetes API errors or other unexpected errors
                    consecutive_failures += 1
                    # TODO: Remove debug logging after verifying health
                    # monitoring works
                    log_msg = (
                        f"Unexpected health check error: {e} "
                        f"(failure {consecutive_failures}/{max_failures})"
                    )
                    if debug:
                        log_msg = f"[DEBUG] Health check ERROR: {log_msg}"
                    vllm_logger.warning(log_msg)

                # Check if we've exceeded max failures
                if max_failures > 0 and consecutive_failures >= max_failures:
                    self._health_check_failed = True
                    self._health_check_failure_reason = (
                        f"vLLM server failed {consecutive_failures} consecutive "
                        f"health checks"
                    )
                    vllm_logger.error(
                        f"Health monitoring detected server failure: "
                        f"{self._health_check_failure_reason}"
                    )
                    # Terminate running benchmark immediately (if using local benchmark provider)
                    if self.benchmark_provider and hasattr(
                        self.benchmark_provider, "terminate_benchmark"
                    ):
                        self.benchmark_provider.terminate_benchmark()
                    break

                # Maintain fixed-cadence scheduling based on monotonic time
                now = time.monotonic()
                # Catch up if the last cycle overran the period
                while next_deadline <= now:
                    next_deadline += period
                sleep_duration = max(0.0, next_deadline - now)
                if stop_event.wait(timeout=sleep_duration):
                    break

            vllm_logger.info("Health monitoring stopped")

        self._health_monitor_thread = threading.Thread(
            target=monitor_health, daemon=True, name="vllm-health-monitor"
        )
        self._health_monitor_thread.start()
        vllm_logger.info("Health monitoring thread started")

    def _stop_health_monitoring(self):
        """Stop the health monitoring thread."""
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            vllm_logger = self._get_trial_logger("vllm")
            vllm_logger.info("Stopping health monitoring thread")
            self._health_monitor_stop = True
            try:
                if self._health_monitor_stop_event is not None:
                    self._health_monitor_stop_event.set()
            except Exception:
                pass
            # Wait briefly for the thread to exit cooperatively
            self._health_monitor_thread.join(timeout=10)
            if self._health_monitor_thread.is_alive():
                vllm_logger.debug(
                    "Health monitoring thread did not stop within "
                    "timeout; continuing cleanup"
                )

    def _check_health_status(self):
        """Check if health monitoring has detected a failure."""
        if self._health_check_failed:
            raise RuntimeError(
                f"vLLM server health check failed: {self._health_check_failure_reason}"
            )

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

    def cleanup_resources(self):
        """Clean up vLLM server process and health monitoring."""
        # Use trial-specific logger if available, otherwise fall back to module logger
        controller_logger = self._get_trial_logger("controller")

        controller_logger.info(
            "!!! Trial Controller: Received cleanup request from backend !!!"
        )

        # IMMEDIATE FLUSH: Ensure user sees cleanup starting in real-time
        self._flush_logger_handlers(controller_logger)

        # Stop health monitoring first
        controller_logger.info("Trial Controller: Stopping health monitoring...")
        self._stop_health_monitoring()
        controller_logger.debug("Trial Controller: Health monitoring stopped")
        self._flush_logger_handlers(controller_logger)

        # Terminate any running benchmark process (if using local benchmark provider)
        if self.benchmark_provider and hasattr(
            self.benchmark_provider, "terminate_benchmark"
        ):
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

        if self.vllm_process:
            pid = self.vllm_process.pid
            controller_logger.info(
                f"Trial Controller: Cleaning up vLLM server process (PID: {pid})..."
            )
            self._flush_logger_handlers(controller_logger)

            # Try to resolve the child's process group id upfront. If the process
            # has already exited, fall back to signaling the process directly.
            try:
                pgid = os.getpgid(pid)
                controller_logger.debug(
                    f"Trial Controller: vLLM process group ID: {pgid}"
                )
            except (OSError, ProcessLookupError):
                controller_logger.warning(
                    f"Trial Controller: Failed to get process group id for {pid}"
                )
                pgid = None

            # Attempt graceful shutdown with SIGTERM first
            controller_logger.info(
                f"Trial Controller → vLLM: Sending SIGTERM to process {pid} "
                f"for graceful shutdown..."
            )
            self._flush_logger_handlers(controller_logger)
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGTERM)
                    controller_logger.debug(
                        f"Trial Controller: Sent SIGTERM to process group {pgid}"
                    )
                else:
                    controller_logger.debug(
                        f"Trial Controller: No process group, sending SIGTERM "
                        f"to process {pid}"
                    )
                    self.vllm_process.terminate()
            except (OSError, ProcessLookupError):
                # Process already gone
                controller_logger.info(
                    f"Trial Controller: vLLM process {pid} already terminated"
                )
                self.vllm_process = None
                return

            try:
                controller_logger.debug(
                    f"Trial Controller: Waiting up to 10s for vLLM process {pid} "
                    f"to terminate..."
                )
                self.vllm_process.wait(timeout=10)
                controller_logger.info(
                    f"Trial Controller: ✓ vLLM process {pid} terminated "
                    f"gracefully via SIGTERM"
                )
                self._flush_logger_handlers(controller_logger)
                self.vllm_process = None
                return
            except subprocess.TimeoutExpired:
                controller_logger.warning(
                    f"Trial Controller: vLLM process {pid} did not respond to "
                    f"SIGTERM within 10s. "
                    f"Escalating to SIGKILL..."
                )
                self._flush_logger_handlers(controller_logger)

            # Final fallback: SIGKILL
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                    controller_logger.info(
                        f"Trial Controller: Sent SIGKILL to process group {pgid}"
                    )
                else:
                    self.vllm_process.kill()
                    controller_logger.info(
                        f"Trial Controller: Sent SIGKILL to process {pid}"
                    )
                controller_logger.info(
                    f"Trial Controller: ✓ vLLM process {pid} force killed via SIGKILL"
                )
                self._flush_logger_handlers(controller_logger)
            except (OSError, ProcessLookupError) as e:
                controller_logger.warning(
                    f"Trial Controller: Failed to kill process {pid}: {e}"
                )
                self._flush_logger_handlers(controller_logger)
            finally:
                self.vllm_process = None
        else:
            controller_logger.debug("Trial Controller: No vLLM process to cleanup")

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


class LocalTrialController(BaseTrialController):
    """Local execution trial controller."""

    def _get_worker_id(self) -> str:
        """Get local machine identifier."""
        import socket

        return f"local_{socket.gethostname()}"


class RayWorkerTrialController(BaseTrialController):
    """Ray worker node trial controller with Ray-specific functionality."""

    def _get_worker_id(self) -> str:
        """Get Ray worker node ID."""
        try:
            return ray.get_runtime_context().get_node_id()
        except Exception:
            return "ray_worker_unknown"

    def _start_vllm_server(self, trial_config: TrialConfig) -> dict:
        """Start vLLM server with GPU assignment from Ray."""
        # Log Ray-specific GPU assignment info to vLLM logger
        vllm_logger = self._get_trial_logger("vllm")

        # Ray handles GPU allocation via resources={"GPU": 1}
        # We can get the assigned GPU from CUDA_VISIBLE_DEVICES if needed
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        vllm_logger.info(f"Ray assigned GPUs: {gpu_ids}")

        # Log Ray worker context information
        try:
            if ray.is_initialized():
                runtime_ctx = ray.get_runtime_context()
                vllm_logger.info(f"Ray worker node: {runtime_ctx.get_node_id()[:8]}")
                vllm_logger.info(
                    f"Ray worker process: {runtime_ctx.get_worker_id()[:8]}"
                )
        except Exception as e:
            vllm_logger.warning(f"Could not get Ray context: {e}")

        # Check for CUDA_VISIBLE_DEVICES override in trial environment variables
        trial_env_vars = trial_config.environment_vars
        if "CUDA_VISIBLE_DEVICES" in trial_env_vars:
            vllm_logger.warning(
                f"Trial specifies "
                f"CUDA_VISIBLE_DEVICES={trial_env_vars['CUDA_VISIBLE_DEVICES']}, "
                f"but Ray has already assigned GPUs: {gpu_ids}. "
                f"Trial setting will override Ray assignment."
            )

        # Call parent implementation
        # (which handles environment variables and logs full Python environment)
        return super()._start_vllm_server(trial_config)


class HelmTrialController(BaseTrialController):
    """Helm-based Kubernetes trial controller."""

    def __init__(
        self,
        release_name: str,
        namespace: str = "default",
        benchmark_image: Optional[str] = None,
        helm_config: Optional[Dict[str, Any]] = None,
        benchmark_pvc: Optional[str] = None,
        model_pvc: Optional[str] = None,
    ):
        """Initialize Helm trial controller.

        Args:
            release_name: Helm release name
            namespace: Kubernetes namespace
            benchmark_image: Container image for benchmark Jobs
            helm_config: Helm configuration dictionary (for full stack deployment checks)
            benchmark_pvc: PersistentVolumeClaim name for benchmark results storage
            model_pvc: PersistentVolumeClaim name for model and dataset storage
        """
        super().__init__()
        self.release_name = release_name
        self.namespace = namespace
        self.benchmark_image = benchmark_image
        self.helm_config = helm_config or {}
        self.benchmark_pvc = benchmark_pvc
        self.model_pvc = model_pvc
        self.server_url: Optional[str] = None
        self.benchmark_job_name: Optional[str] = None

    def _get_worker_id(self) -> str:
        """Get Kubernetes pod/node identifier."""
        try:
            from kubernetes import client, config

            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

            v1 = client.CoreV1Api()
            node_name = os.environ.get("NODE_NAME") or "kubernetes-node"
            return f"k8s_{node_name}"
        except Exception:
            return "kubernetes_unknown"

    def _validate_environment(self, trial_config: Optional[TrialConfig] = None) -> None:
        """Validate environment for Helm-based deployments.

        For Helm deployments, vLLM and benchmarks run remotely in Kubernetes,
        so we only need to validate controller-side dependencies (optuna, kubernetes client).
        """
        if self._environment_validated:
            return

        # For Helm deployments, only require controller-side packages
        required_packages = {
            "optuna": "Optuna optimization framework",
            "kubernetes": "Kubernetes client",
        }

        # Only require psycopg2 if using PostgreSQL
        using_postgresql = False
        if trial_config and trial_config.logging_config:
            using_postgresql = (
                trial_config.logging_config.get("database_url") is not None
            )

        if using_postgresql:
            required_packages["psycopg2"] = "PostgreSQL client"

        missing_packages = []

        for package, description in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(f"{package} ({description})")

        if missing_packages:
            missing_list = "\n  - ".join(missing_packages)
            raise RuntimeError(
                f"Missing required packages for Helm controller:\n  - {missing_list}\n\n"
                f"For Helm deployments, vLLM and benchmarks run remotely in Kubernetes.\n"
                f"Install auto-tune-vllm[helm] on the controller:\n"
                f"  pip install auto-tune-vllm[helm]"
            )

        # Only check for python3, not guidellm (benchmarks run in Kubernetes Jobs)
        import shutil

        if not shutil.which("python3"):
            raise RuntimeError(
                "Missing required command: python3\n\n"
                "Python 3 interpreter is required on the controller."
            )

        self._environment_validated = True
        logger.info(
            "Environment validation passed for Helm controller (remote execution)"
        )

    def _start_vllm_server(self, trial_config: TrialConfig) -> dict:
        """Start vLLM server via Helm (already deployed by backend).

        Args:
            trial_config: Trial configuration

        Returns:
            Dictionary with server info (port, url, pid)
        """
        from .helm_utils import get_service_url, wait_for_service_ready

        vllm_logger = self._get_trial_logger("vllm")
        vllm_logger.info(
            f"Using Helm-deployed vLLM server (release: {self.release_name})"
        )

        # Get service URL from Helm release
        # Pass helm_config if available to check for full stack deployment
        helm_config = getattr(self, "helm_config", None)
        # #region agent log - DISABLED
        # import json
        # import time
        # with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        #     f.write(json.dumps({"sessionId":"debug-session","runId":"trial-controller","hypothesisId":"A","location":"trial_controller.py:1425","message":"About to call get_service_url","data":{"release_name":self.release_name,"namespace":self.namespace,"helm_config":str(helm_config)},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion
        self.server_url = get_service_url(
            self.release_name, self.namespace, helm_config
        )
        # #region agent log - DISABLED
        # with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        #     f.write(json.dumps({"sessionId":"debug-session","runId":"trial-controller","hypothesisId":"A","location":"trial_controller.py:1430","message":"get_service_url returned","data":{"server_url":self.server_url},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion
        vllm_logger.info(f"vLLM server URL: {self.server_url}")

        # Wait for service to be ready
        # Parse URL - handle both IP addresses and DNS names
        from urllib.parse import urlparse

        parsed = urlparse(self.server_url)
        hostname = parsed.hostname
        # If it's an IP address, pass the full URL; otherwise extract service name
        if (
            hostname
            and all(c.isdigit() or c == "." for c in hostname)
            and hostname.count(".") == 3
        ):
            # It's an IP address - pass the full URL
            service_name = self.server_url
        else:
            # It's a DNS name - extract service name
            service_name = (
                hostname.split(".")[0]
                if hostname
                else self.server_url.split("://")[1].split("/")[0]
            )
        ready = wait_for_service_ready(
            service_name, self.namespace, trial_config.vllm_startup_timeout
        )
        if not ready:
            raise RuntimeError(
                f"vLLM service not ready after {trial_config.vllm_startup_timeout}s"
            )

        # Extract port from URL
        port = 8000
        if ":" in self.server_url:
            try:
                port = int(self.server_url.split(":")[-1].split("/")[0])
            except ValueError:
                pass

        return {
            "port": port,
            "url": self.server_url,
            "pid": None,  # No process PID for Helm deployments
        }

    def _start_benchmark(
        self, trial_config: TrialConfig, benchmark_image: Optional[str] = None
    ) -> str:
        """Start benchmark as Kubernetes Job.

        Args:
            trial_config: Trial configuration
            benchmark_image: Optional container image for benchmark

        Returns:
            Job name
        """
        from .helm_utils import create_benchmark_job

        controller_logger = self._get_trial_logger("controller")
        controller_logger.info("Creating Kubernetes Job for benchmark")

        if not self.server_url:
            raise RuntimeError("vLLM server URL not available")

        # Create benchmark Job
        self.benchmark_job_name = create_benchmark_job(
            trial_config,
            self.server_url,
            self.namespace,
            benchmark_image,
            kubeconfig=None,
            benchmark_pvc=self.benchmark_pvc,
            model_pvc=self.model_pvc,
        )

        controller_logger.info(f"Created benchmark Job: {self.benchmark_job_name}")
        return self.benchmark_job_name

    def _wait_for_benchmark_completion(
        self, job_name: str, timeout: int = 3600
    ) -> bool:
        """Wait for benchmark Job to complete.

        Args:
            job_name: Job name
            timeout: Timeout in seconds

        Returns:
            True if completed successfully, False otherwise
        """
        from .helm_utils import wait_for_job_completion

        controller_logger = self._get_trial_logger("controller")
        controller_logger.info(f"Waiting for benchmark Job {job_name} to complete")

        kubeconfig = getattr(self, "kubeconfig", None)
        return wait_for_job_completion(
            job_name, self.namespace, timeout, kubeconfig=kubeconfig
        )

    def _extract_benchmark_results(self, job_name: str) -> dict:
        """Extract benchmark results from Kubernetes Job.

        Args:
            job_name: Job name

        Returns:
            Dictionary of benchmark results
        """
        from .helm_utils import extract_job_results

        controller_logger = self._get_trial_logger("controller")
        controller_logger.info(f"Extracting results from Job {job_name}")

        kubeconfig = getattr(self, "kubeconfig", None)
        return extract_job_results(job_name, self.namespace, kubeconfig=kubeconfig)

    def run_trial(
        self, trial_config: TrialConfig, cancellation_flag_actor=None
    ) -> TrialResult:
        """Execute trial with Helm-deployed vLLM and Kubernetes Job benchmark.

        Args:
            trial_config: Trial configuration
            cancellation_flag_actor: Optional cancellation flag (not used for Helm)

        Returns:
            TrialResult with benchmark results
        """
        from ..core.trial import ExecutionInfo, TrialResult
        from .helm_utils import delete_benchmark_job

        execution_info = ExecutionInfo()
        execution_info.helm_release_name = self.release_name
        controller_logger = self._get_trial_logger("controller")

        controller_logger.info(
            f"Running Helm trial {trial_config.trial_id} "
            f"with parameters: {trial_config.parameters}"
        )

        try:
            # Setup trial-specific logging
            self._setup_trial_logging(trial_config)

            # Start vLLM server (already deployed via Helm, just get URL)
            controller_logger.info("Getting vLLM server URL from Helm release")
            execution_info.mark_vllm_started()
            server_info = self._start_vllm_server(trial_config)
            execution_info.worker_node_id = self._get_worker_id()

            # Note: _start_vllm_server already calls wait_for_service_ready which uses
            # a Kubernetes Job to verify readiness for ClusterIP services.
            # No need to call _wait_for_server_ready again (which would fail for ClusterIP
            # services since it tries HTTP from outside the cluster).
            controller_logger.info(
                f"Server ready at {server_info['url']} (verified via readiness check Job)"
            )
            execution_info.mark_vllm_ready()

            # Start health monitoring
            health_url = server_info["url"].replace("/v1", "/health")
            self._start_health_monitoring(
                health_url,
                check_interval=trial_config.health_check_interval,
                max_failures=trial_config.health_check_max_failures,
            )

            # Start benchmark as Kubernetes Job
            controller_logger.info("Starting benchmark as Kubernetes Job")
            execution_info.mark_benchmark_started()
            job_name = self._start_benchmark(trial_config, self.benchmark_image)
            execution_info.benchmark_job_name = job_name

            # Wait for benchmark completion
            # Timeout should account for: pod initialization, PVC mounting, init container, benchmark duration
            # Minimum 5 minutes for initialization, plus 1.5x benchmark duration
            min_init_time = 300  # 5 minutes minimum for pod/PVC initialization
            benchmark_time = trial_config.benchmark_config.max_seconds * 1.5
            max_benchmark_time = max(min_init_time, benchmark_time)
            completed = self._wait_for_benchmark_completion(
                job_name, int(max_benchmark_time)
            )

            if not completed:
                # Collect logs before raising error
                try:
                    from .helm_utils import collect_job_logs

                    kubeconfig = getattr(self, "kubeconfig", None)
                    logs = collect_job_logs(
                        job_name, self.namespace, kubeconfig=kubeconfig
                    )
                    controller_logger.error(
                        f"Benchmark Job {job_name} did not complete. Logs:\n{logs}"
                    )
                except Exception as log_e:
                    controller_logger.warning(
                        f"Failed to collect logs for incomplete benchmark job "
                        f"{job_name}: {log_e}"
                    )
                raise RuntimeError(
                    f"Benchmark Job {job_name} did not complete within timeout"
                )

            # Extract results
            benchmark_result = self._extract_benchmark_results(job_name)

            # Check if vLLM server died during benchmark
            self._check_health_status()

            # Extract objectives
            objective_values = self._extract_objectives(
                benchmark_result, trial_config.optimization_config
            )
            controller_logger.info(
                f"Trial completed with objectives: {objective_values}"
            )

            execution_info.mark_benchmark_completed()
            execution_info.mark_completed(status="success")

            return TrialResult(
                trial_id=trial_config.trial_id,
                trial_number=trial_config.trial_number,
                trial_type=trial_config.trial_type,
                objective_values=objective_values,
                detailed_metrics=benchmark_result,
                execution_info=execution_info,
                success=True,
            )

        except Exception as e:
            execution_info.mark_completed(status="failed")
            error_type = self._classify_error(e)
            controller_logger.error(f"Trial {trial_config.trial_id} failed: {e}")

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
            # Stop health monitoring first
            controller_logger.info("Trial Controller: Stopping health monitoring...")
            self._stop_health_monitoring()
            controller_logger.debug("Trial Controller: Health monitoring stopped")

            # Collect benchmark pod logs BEFORE deletion
            if self.benchmark_job_name:
                controller_logger.info(
                    f"Collecting logs from benchmark Job {self.benchmark_job_name}"
                )
                try:
                    from .pod_log_collector import collect_and_store_job_pod_logs

                    pg_url = (
                        trial_config.logging_config.get("database_url")
                        if trial_config.logging_config
                        else None
                    )
                    if pg_url:
                        success = collect_and_store_job_pod_logs(
                            job_name=self.benchmark_job_name,
                            namespace=self.namespace,
                            study_name=trial_config.study_name,
                            trial_id=trial_config.trial_id,
                            pg_url=pg_url,
                            kubeconfig=None,
                        )
                        if success:
                            controller_logger.info("Benchmark pod logs collected successfully")
                        else:
                            controller_logger.warning("Failed to collect benchmark pod logs")
                    else:
                        controller_logger.debug("No PostgreSQL URL, skipping pod log collection")
                except Exception as e:
                    controller_logger.warning(f"Error collecting benchmark pod logs: {e}")

            # Cleanup benchmark Job
            if self.benchmark_job_name:
                try:
                    delete_benchmark_job(self.benchmark_job_name, self.namespace)
                except Exception as e:
                    controller_logger.warning(f"Failed to delete benchmark Job: {e}")

            self._flush_trial_logs(trial_config.trial_id)
            # Note: Helm release cleanup is handled by backend

    def cleanup_resources(self):
        """Clean up resources (benchmark Job cleanup handled in run_trial)."""
        controller_logger = self._get_trial_logger("controller")
        controller_logger.info(
            "Helm trial controller cleanup (release managed by backend)"
        )


class KubernetesTrialController(BaseTrialController):
    """Kubernetes Deployment/Pod-based trial controller."""

    def __init__(
        self,
        deployment_name: str,
        service_name: str,
        namespace: str = "default",
        benchmark_image: Optional[str] = None,
        service_type: str = "ClusterIP",
        service_port: int = 8000,
        kubeconfig: Optional[str] = None,
        benchmark_pvc: Optional[str] = None,
        model_pvc: Optional[str] = None,
    ):
        """Initialize Kubernetes trial controller.

        Args:
            deployment_name: Kubernetes Deployment name
            service_name: Kubernetes Service name
            namespace: Kubernetes namespace
            benchmark_image: Container image for benchmark Jobs
            service_type: Service type (ClusterIP, NodePort, LoadBalancer)
            service_port: Service port
            kubeconfig: Path to kubeconfig file
            benchmark_pvc: PersistentVolumeClaim name for benchmark results storage
            model_pvc: PersistentVolumeClaim name for model and dataset storage
        """
        super().__init__()
        self.deployment_name = deployment_name
        self.service_name = service_name
        self.namespace = namespace
        self.benchmark_image = benchmark_image
        self.service_type = service_type
        self.service_port = service_port
        self.kubeconfig = kubeconfig
        self.benchmark_pvc = benchmark_pvc
        self.model_pvc = model_pvc
        self.server_url: Optional[str] = None
        self.benchmark_job_name: Optional[str] = None

    def _get_worker_id(self) -> str:
        """Get Kubernetes pod/node identifier."""
        try:
            from kubernetes import client, config

            try:
                config.load_incluster_config()
            except config.ConfigException:
                if self.kubeconfig:
                    config.load_kube_config(config_file=self.kubeconfig)
                else:
                    config.load_kube_config()

            v1 = client.CoreV1Api()
            node_name = os.environ.get("NODE_NAME") or "kubernetes-node"
            return f"k8s_{node_name}"
        except Exception:
            return "kubernetes_unknown"

    def _validate_environment(self, trial_config: Optional[TrialConfig] = None) -> None:
        """Validate environment for Kubernetes-based deployments."""
        if self._environment_validated:
            return

        # For Kubernetes deployments, only require controller-side packages
        required_packages = {
            "optuna": "Optuna optimization framework",
            "kubernetes": "Kubernetes client",
        }

        # Only require psycopg2 if using PostgreSQL
        using_postgresql = False
        if trial_config and trial_config.logging_config:
            using_postgresql = (
                trial_config.logging_config.get("database_url") is not None
            )

        if using_postgresql:
            required_packages["psycopg2"] = "PostgreSQL client"

        missing_packages = []

        for package, description in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(f"{package} ({description})")

        if missing_packages:
            missing_list = "\n  - ".join(missing_packages)
            raise RuntimeError(
                f"Missing required packages for Kubernetes controller:\n  - {missing_list}\n\n"
                f"For Kubernetes deployments, vLLM and benchmarks run remotely in Kubernetes.\n"
                f"Install auto-tune-vllm[helm] on the controller:\n"
                f"  pip install auto-tune-vllm[helm]"
            )

        import shutil

        if not shutil.which("python3"):
            raise RuntimeError(
                "Missing required command: python3\n\n"
                "Python 3 interpreter is required on the controller."
            )

        self._environment_validated = True
        logger.info(
            "Environment validation passed for Kubernetes controller (remote execution)"
        )

    def _start_vllm_server(self, trial_config: TrialConfig) -> dict:
        """Start vLLM server via Kubernetes Deployment (already deployed by backend).

        Args:
            trial_config: Trial configuration

        Returns:
            Dictionary with server info (port, url, pid)
        """
        from .k8s_utils import get_service_url, wait_for_deployment_ready

        vllm_logger = self._get_trial_logger("vllm")
        vllm_logger.info(
            f"Using Kubernetes-deployed vLLM server "
            f"(Deployment: {self.deployment_name}, Service: {self.service_name})"
        )

        # Wait for deployment to be ready
        ready = wait_for_deployment_ready(
            deployment_name=self.deployment_name,
            namespace=self.namespace,
            timeout=trial_config.vllm_startup_timeout,
            kubeconfig=self.kubeconfig,
        )
        if not ready:
            raise RuntimeError(
                f"vLLM Deployment not ready after {trial_config.vllm_startup_timeout}s"
            )

        # Get service URL
        self.server_url = get_service_url(
            service_name=self.service_name,
            namespace=self.namespace,
            service_type=self.service_type,
            kubeconfig=self.kubeconfig,
        )
        vllm_logger.info(f"vLLM server URL: {self.server_url}")

        # Extract port from URL
        port = self.service_port
        if ":" in self.server_url:
            try:
                port = int(self.server_url.split(":")[-1].split("/")[0])
            except ValueError:
                pass

        return {
            "port": port,
            "url": self.server_url,
            "pid": None,  # No process PID for Kubernetes deployments
        }

    def _start_benchmark(
        self, trial_config: TrialConfig, benchmark_image: Optional[str] = None
    ) -> str:
        """Start benchmark as Kubernetes Job.

        Args:
            trial_config: Trial configuration
            benchmark_image: Optional container image for benchmark

        Returns:
            Job name
        """
        from .helm_utils import create_benchmark_job

        controller_logger = self._get_trial_logger("controller")
        controller_logger.info("Creating Kubernetes Job for benchmark")

        if not self.server_url:
            raise RuntimeError("vLLM server URL not available")

        # Create benchmark Job
        self.benchmark_job_name = create_benchmark_job(
            trial_config,
            self.server_url,
            self.namespace,
            benchmark_image or self.benchmark_image,
            self.kubeconfig,
            benchmark_pvc=self.benchmark_pvc,
            model_pvc=self.model_pvc,
        )

        controller_logger.info(f"Created benchmark Job: {self.benchmark_job_name}")
        return self.benchmark_job_name

    def _wait_for_benchmark_completion(
        self, job_name: str, timeout: int = 3600
    ) -> bool:
        """Wait for benchmark Job to complete.

        Args:
            job_name: Job name
            timeout: Timeout in seconds

        Returns:
            True if completed successfully, False otherwise
        """
        from .helm_utils import wait_for_job_completion

        controller_logger = self._get_trial_logger("controller")
        controller_logger.info(f"Waiting for benchmark Job {job_name} to complete")

        kubeconfig = getattr(self, "kubeconfig", None)
        return wait_for_job_completion(
            job_name, self.namespace, timeout, kubeconfig=kubeconfig
        )

    def _extract_benchmark_results(self, job_name: str) -> dict:
        """Extract benchmark results from Kubernetes Job.

        Args:
            job_name: Job name

        Returns:
            Dictionary of benchmark results
        """
        from .helm_utils import extract_job_results

        controller_logger = self._get_trial_logger("controller")
        controller_logger.info(f"Extracting results from Job {job_name}")

        kubeconfig = getattr(self, "kubeconfig", None)
        return extract_job_results(job_name, self.namespace, kubeconfig=kubeconfig)

    def run_trial(
        self, trial_config: TrialConfig, cancellation_flag_actor=None
    ) -> TrialResult:
        """Execute trial with Kubernetes-deployed vLLM and Kubernetes Job benchmark.

        Args:
            trial_config: Trial configuration
            cancellation_flag_actor: Optional cancellation flag (not used for Kubernetes)

        Returns:
            TrialResult with benchmark results
        """
        from ..core.trial import ExecutionInfo, TrialResult
        from .helm_utils import delete_benchmark_job

        execution_info = ExecutionInfo()

        # Store trial_id for health monitoring label selector
        self._current_trial_id = trial_config.trial_id

        controller_logger = self._get_trial_logger("controller")
        controller_logger.info(
            f"Running trial {trial_config.trial_id} "
            f"with parameters: {trial_config.parameters}"
        )

        try:
            # Setup trial-specific logging
            self._setup_trial_logging(trial_config)

            # Start vLLM server (already deployed via Kubernetes, just get URL)
            controller_logger.info("Getting vLLM server URL from Kubernetes Service")
            execution_info.mark_vllm_started()
            server_info = self._start_vllm_server(trial_config)
            execution_info.worker_node_id = self._get_worker_id()

            # Wait for server to actually be ready (HTTP health check)
            # Note: wait_for_deployment_ready only checks pod status, not HTTP readiness
            controller_logger.info(
                f"Waiting for server at {server_info['url']} to be ready "
                f"(timeout: {trial_config.vllm_startup_timeout}s)"
            )
            from .helm_utils import wait_for_service_ready

            ready = wait_for_service_ready(
                service_name=server_info["url"],
                namespace=self.namespace,
                timeout=trial_config.vllm_startup_timeout,
                kubeconfig=self.kubeconfig,
            )
            if not ready:
                raise RuntimeError(
                    f"vLLM server at {server_info['url']} not ready after {trial_config.vllm_startup_timeout}s"
                )

            controller_logger.info(
                f"Server ready at {server_info['url']} "
                f"(Deployment: {self.deployment_name})"
            )
            execution_info.mark_vllm_ready()

            # Start health monitoring
            health_url = server_info["url"].replace("/v1", "/health")
            self._start_health_monitoring(
                health_url,
                check_interval=trial_config.health_check_interval,
                max_failures=trial_config.health_check_max_failures,
            )

            # Start benchmark as Kubernetes Job
            controller_logger.info("Starting benchmark as Kubernetes Job")
            execution_info.mark_benchmark_started()
            job_name = self._start_benchmark(trial_config, self.benchmark_image)
            execution_info.benchmark_job_name = job_name

            # Wait for benchmark completion
            # Timeout should account for: pod initialization, PVC mounting, init container, benchmark duration
            # Minimum 5 minutes for initialization, plus 1.5x benchmark duration
            min_init_time = 300  # 5 minutes minimum for pod/PVC initialization
            benchmark_time = trial_config.benchmark_config.max_seconds * 1.5
            max_benchmark_time = max(min_init_time, benchmark_time)
            completed = self._wait_for_benchmark_completion(
                job_name, int(max_benchmark_time)
            )

            if not completed:
                # Collect logs before raising error
                try:
                    from .helm_utils import collect_job_logs

                    kubeconfig = getattr(self, "kubeconfig", None)
                    logs = collect_job_logs(
                        job_name, self.namespace, kubeconfig=kubeconfig
                    )
                    controller_logger.error(
                        f"Benchmark Job {job_name} did not complete. Logs:\n{logs}"
                    )
                except Exception as log_e:
                    controller_logger.warning(
                        f"Failed to collect logs for incomplete benchmark job "
                        f"{job_name}: {log_e}"
                    )
                raise RuntimeError(
                    f"Benchmark Job {job_name} did not complete within timeout"
                )

            # Extract results
            benchmark_result = self._extract_benchmark_results(job_name)

            # Check if vLLM server died during benchmark
            self._check_health_status()

            # Extract objectives
            objective_values = self._extract_objectives(
                benchmark_result, trial_config.optimization_config
            )
            controller_logger.info(
                f"Trial completed with objectives: {objective_values}"
            )

            execution_info.mark_benchmark_completed()
            execution_info.mark_completed(status="success")

            return TrialResult(
                trial_id=trial_config.trial_id,
                trial_number=trial_config.trial_number,
                trial_type=trial_config.trial_type,
                objective_values=objective_values,
                detailed_metrics=benchmark_result,
                execution_info=execution_info,
                success=True,
            )

        except Exception as e:
            execution_info.mark_completed(status="failed")
            error_type = self._classify_error(e)
            controller_logger.error(f"Trial {trial_config.trial_id} failed: {e}")

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
            # Stop health monitoring first
            controller_logger.info("Trial Controller: Stopping health monitoring...")
            self._stop_health_monitoring()
            controller_logger.debug("Trial Controller: Health monitoring stopped")

            # Collect vLLM pod logs BEFORE backend cleanup
            controller_logger.info(
                f"Collecting logs from vLLM Deployment {self.deployment_name}"
            )
            try:
                from .pod_log_collector import collect_and_store_deployment_pod_logs

                pg_url = (
                    trial_config.logging_config.get("database_url")
                    if trial_config.logging_config
                    else None
                )
                if pg_url:
                    success = collect_and_store_deployment_pod_logs(
                        deployment_name=self.deployment_name,
                        trial_id=trial_config.trial_id,
                        namespace=self.namespace,
                        study_name=trial_config.study_name,
                        pg_url=pg_url,
                        kubeconfig=self.kubeconfig,
                    )
                    if success:
                        controller_logger.info("vLLM pod logs collected successfully")
                    else:
                        controller_logger.warning("Failed to collect vLLM pod logs")
                else:
                    controller_logger.debug("No PostgreSQL URL, skipping pod log collection")
            except Exception as e:
                controller_logger.warning(f"Error collecting vLLM pod logs: {e}")

            # Collect benchmark pod logs BEFORE deletion
            if self.benchmark_job_name:
                controller_logger.info(
                    f"Collecting logs from benchmark Job {self.benchmark_job_name}"
                )
                try:
                    from .pod_log_collector import collect_and_store_job_pod_logs

                    pg_url = (
                        trial_config.logging_config.get("database_url")
                        if trial_config.logging_config
                        else None
                    )
                    if pg_url:
                        success = collect_and_store_job_pod_logs(
                            job_name=self.benchmark_job_name,
                            namespace=self.namespace,
                            study_name=trial_config.study_name,
                            trial_id=trial_config.trial_id,
                            pg_url=pg_url,
                            kubeconfig=self.kubeconfig,
                        )
                        if success:
                            controller_logger.info("Benchmark pod logs collected successfully")
                        else:
                            controller_logger.warning("Failed to collect benchmark pod logs")
                    else:
                        controller_logger.debug("No PostgreSQL URL, skipping pod log collection")
                except Exception as e:
                    controller_logger.warning(f"Error collecting benchmark pod logs: {e}")

            # Cleanup benchmark Job
            if self.benchmark_job_name:
                try:
                    delete_benchmark_job(self.benchmark_job_name, self.namespace)
                except Exception as e:
                    controller_logger.warning(f"Failed to delete benchmark Job: {e}")

            self._flush_trial_logs(trial_config.trial_id)
            # Note: Kubernetes Deployment/Service cleanup is handled by backend

    def cleanup_resources(self):
        """Clean up resources (benchmark Job cleanup handled in run_trial)."""
        controller_logger = self._get_trial_logger("controller")
        controller_logger.info(
            "Kubernetes trial controller cleanup (Deployment/Service managed by backend)"
        )

    def request_cancellation(self):
        """Request cancellation of the running trial."""
        controller_logger = self._get_trial_logger("controller")
        controller_logger.info("Cancellation requested for Kubernetes trial")
        # Note: Actual cancellation is handled by backend cleanup


# Helper functions for endpoint discovery
def _is_kubernetes_env() -> bool:
    """Check if running in Kubernetes environment."""
    return os.environ.get("KUBERNETES_SERVICE_HOST") is not None


def _get_node_ip() -> str:
    """Get node IP address for server URL.

    Returns:
        Node IP address or hostname, falls back to localhost
    """
    try:
        # Try to get hostname first
        hostname = socket.gethostname()

        # Try to resolve to IP address
        try:
            node_ip = socket.gethostbyname(hostname)
            # If it resolves to 127.0.0.1, use hostname instead
            if node_ip != "127.0.0.1":
                return node_ip
            return hostname
        except socket.gaierror:
            # If resolution fails, use hostname
            return hostname
    except Exception:
        # Fallback to localhost
        return "localhost"


def _get_kubernetes_namespace() -> str:
    """Get Kubernetes namespace from environment or service account."""
    # Try environment variable first
    namespace = os.environ.get("POD_NAMESPACE")
    if namespace:
        return namespace

    # Try reading from service account namespace file
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            return f.read().strip()
    except (FileNotFoundError, IOError):
        pass

    # Default namespace
    return "default"


# SharedState Actor for server URL communication
if RAY_AVAILABLE:

    @ray.remote
    class SharedState:
        """Ray actor to hold vLLM server URL for communication between actors."""

        def __init__(self):
            self.server_url: Optional[str] = None

        def set_server_url(self, url: str):
            """Set the vLLM server URL."""
            self.server_url = url

        def get_server_url(self) -> Optional[str]:
            """Get the vLLM server URL."""
            return self.server_url
else:
    # Dummy class when Ray is not available
    class SharedState:
        """Dummy SharedState when Ray is not available."""

        def __init__(self):
            pass

        def set_server_url(self, url: str):
            pass

        def get_server_url(self) -> Optional[str]:
            return None


# VLLM Server Actor
if RAY_AVAILABLE:

    @ray.remote
    class VLLMServerActor:
        """Ray actor that manages vLLM server lifecycle."""

        def __init__(self):
            pass

        def start_server(self, trial_config):
            return {}

        def get_server_url(self):
            return None

        def is_ready(self):
            return False

        def wait_for_ready(self, timeout=300):
            return False

        def cleanup(self):
            pass
else:

    class VLLMServerActor:
        """Dummy VLLMServerActor when Ray is not available."""

        pass


# Workload Actor
if RAY_AVAILABLE:

    @ray.remote
    class WorkloadActor:
        """Ray actor that runs benchmark workload."""

        def __init__(self):
            pass

        def run_benchmark(self, *args, **kwargs):
            return {}
else:

    class WorkloadActor:
        """Dummy WorkloadActor when Ray is not available."""

        pass


# Ray remote actor wrapper (kept for backward compatibility)
if RAY_AVAILABLE:

    @ray.remote
    class RayTrialActor(RayWorkerTrialController):
        """Ray remote actor for distributed trial execution."""

        pass
else:

    class RayTrialActor(RayWorkerTrialController):
        """Dummy RayTrialActor when Ray is not available."""

        pass

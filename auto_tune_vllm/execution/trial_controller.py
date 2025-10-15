"""Trial controller implementations for Ray and local execution."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Optional

import ray

from ..benchmarks.providers import BenchmarkProvider, GuideLLMBenchmark
from ..core.trial import ExecutionInfo, TrialConfig, TrialResult
from ..logging.manager import CentralizedLogger

logger = logging.getLogger(__name__)


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
        self._health_check_url = None
        self._health_check_failed = False
        self._health_check_failure_reason = None
        self._benchmark_process = None  # Track running benchmark process

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

    def run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Execute trial with proper error handling and cleanup."""
        execution_info = ExecutionInfo()

        print(
            f"Running trial {trial_config.trial_id} "
            f"with parameters: {trial_config.parameters}"
        )
        print(f"Study name: {trial_config.study_name}")

        print("")

        # If the parallelism is set lower than MIN_GPUS in the
        # static_environment_variables, cannot start the trial
        # if trial_config.parameters.get(
        #     "tensor_parallel_size"
        # ) < trial_config.static_environment_variables.get("MIN_GPUS"):
        #     logger.error(
        #         f"Trial {trial_config.trial_id} cannot start "
        #         f"because the parallelism is set lower than MIN_GPUS "
        #         f" in the static_environment_variables"
        #     )
        #     return TrialResult(
        #         trial_id=trial_config.trial_id,
        #         trial_number=trial_config.trial_number,
        #         trial_type=trial_config.trial_type,
        #         objective_values=[],
        #         detailed_metrics={},
        #         execution_info=execution_info,
        #         success=False,
        #         error_message=(
        #             "Parallelism is set lower than MIN_GPUS "
        #             "in the static_environment_variables"
        #         ),
        #     )

        try:
            # Store study name for log flushing
            self._current_study_name = trial_config.study_name

            # Setup trial-specific logging first
            self._setup_trial_logging(trial_config)

            # Validate environment first
            # (with trial config for conditional psycopg2 check)
            self._validate_environment(trial_config)

            # Setup benchmark provider
            self.benchmark_provider = self._create_benchmark_provider(trial_config)

            # Start vLLM server
            controller_logger = self._get_trial_logger("controller")
            controller_logger.info("Starting vLLM server")
            execution_info.mark_vllm_started()
            server_info = self._start_vllm_server(trial_config)
            execution_info.worker_node_id = self._get_worker_id()

            # Wait for server to be ready
            controller_logger.info(
                f"Waiting for server at {server_info['url']} to be ready "
                f"(timeout: {trial_config.vllm_startup_timeout}s)"
            )
            self._wait_for_server_ready(
                server_info["url"], trial_config.vllm_startup_timeout
            )
            execution_info.mark_vllm_ready()

            # Start health monitoring after server is ready
            health_url = server_info["url"].replace("/v1", "/health")
            controller_logger.info("Starting runtime health monitoring")
            self._start_health_monitoring(
                health_url,
                check_interval=trial_config.health_check_interval,
                max_failures=trial_config.health_check_max_failures,
            )

            # Run benchmark
            controller_logger.info("Starting benchmark run")
            benchmark_logger = self._get_trial_logger("benchmark")

            # Pass benchmark logger to provider if it supports it
            if hasattr(self.benchmark_provider, "set_logger"):
                self.benchmark_provider.set_logger(benchmark_logger)

            # Pass trial context for benchmark result storage
            if hasattr(self.benchmark_provider, "set_trial_context"):
                self.benchmark_provider.set_trial_context(
                    trial_config.study_name, trial_config.trial_id
                )

            execution_info.mark_benchmark_started()
            benchmark_result = self.benchmark_provider.run_benchmark(
                model_url=server_info["url"], config=trial_config.benchmark_config
            )
            execution_info.mark_benchmark_completed()

            # Check if vLLM server died during benchmark execution
            self._check_health_status()

            # Extract objectives for Optuna using optimization configuration
            objective_values = self._extract_objectives(
                benchmark_result, trial_config.optimization_config
            )
            controller_logger.info(
                f"Trial completed with objectives: {objective_values}"
            )

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
            # Determine failure type based on error message
            error_str = str(e)
            status = (
                "vllm_crash"
                if "vLLM" in error_str or "health" in error_str
                else "benchmark_crash"
            )
            execution_info.mark_completed(status=status)
            error_logger = self._get_trial_logger("controller")
            error_logger.error(f"Trial {trial_config.trial_id} failed: {e}")

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
            "--no-enable-prefix-caching",
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
        self, health_url: str, check_interval: int = 30, max_failures: int = 3
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
            check_count = 0  # TODO: Remove after verifying health monitoring works
            vllm_logger.info(
                f"Starting health monitoring: checking {health_url} every "
                f"{check_interval}s"
                + (" (DEBUG MODE: verbose logging enabled)" if debug else "")
            )

            while not self._health_monitor_stop:
                check_count += 1  # TODO: Remove after verifying health monitoring works

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
                    # Terminate running benchmark immediately
                    self._terminate_benchmark()
                    break

                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        # TODO: Remove debug logging after verifying health
                        # monitoring works
                        if debug:
                            vllm_logger.info(
                                f"[DEBUG] Health check #{check_count} PASSED: "
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
                    consecutive_failures += 1
                    # TODO: Remove debug logging after verifying health
                    # monitoring works
                    log_msg = (
                        f"Health check failed: {e} "
                        f"(failure {consecutive_failures}/{max_failures})"
                    )
                    if debug:
                        log_msg = (
                            f"[DEBUG] Health check #{check_count} FAILED: {log_msg}"
                        )
                    vllm_logger.warning(log_msg)
                except Exception as e:
                    consecutive_failures += 1
                    # TODO: Remove debug logging after verifying health
                    # monitoring works
                    log_msg = (
                        f"Unexpected health check error: {e} "
                        f"(failure {consecutive_failures}/{max_failures})"
                    )
                    if debug:
                        log_msg = (
                            f"[DEBUG] Health check #{check_count} ERROR: {log_msg}"
                        )
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
                    # Terminate running benchmark immediately
                    self._terminate_benchmark()
                    break

                # Wait before next check (but check stop flag periodically)
                time.sleep(check_interval)

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
            # Give the thread a moment to stop gracefully
            self._health_monitor_thread.join(timeout=5)
            if self._health_monitor_thread.is_alive():
                vllm_logger.warning("Health monitoring thread did not stop gracefully")

    def _terminate_benchmark(self):
        """Terminate the running benchmark process if vLLM has failed."""
        if self.benchmark_provider and hasattr(
            self.benchmark_provider, "terminate_benchmark"
        ):
            try:
                self.benchmark_provider.terminate_benchmark()
            except Exception as e:
                logger.warning(f"Failed to terminate benchmark: {e}")

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
        # Stop health monitoring first
        self._stop_health_monitoring()

        if self.vllm_process:
            pid = self.vllm_process.pid
            try:
                # Send SIGINT first for graceful cleanup (what vLLM expects)
                logger.info(
                    f"Sending SIGINT to vLLM process {pid} for graceful shutdown"
                )
                self.vllm_process.send_signal(signal.SIGINT)
                self.vllm_process.wait(
                    timeout=15
                )  # Give more time for graceful cleanup
                logger.info(f"Gracefully terminated vLLM process {pid}")
            except subprocess.TimeoutExpired:
                # If graceful shutdown fails, then use SIGTERM
                logger.warning(f"Graceful shutdown timed out, sending SIGTERM to {pid}")
                self.vllm_process.terminate()
                try:
                    self.vllm_process.wait(timeout=5)
                    logger.info(f"Terminated vLLM process {pid} with SIGTERM")
                except subprocess.TimeoutExpired:
                    # Last resort: kill entire process group to catch multiprocessing
                    # children
                    logger.warning(f"SIGTERM timeout, killing process group for {pid}")
                    try:
                        # Kill the entire process group (negative PID)
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        logger.info(f"Killed process group for {pid}")
                    except (OSError, ProcessLookupError) as e:
                        logger.warning(f"Failed to kill process group for {pid}: {e}")
                        # Fallback to regular kill if killpg fails
                        try:
                            self.vllm_process.kill()
                            logger.warning(f"Force killed vLLM process {pid}")
                        except (OSError, ProcessLookupError):
                            logger.warning(f"Process {pid} already dead")
            finally:
                self.vllm_process = None

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


# Ray remote actor wrapper
@ray.remote
class RayTrialActor(RayWorkerTrialController):
    """Ray remote actor for distributed trial execution."""

    def run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Run trial on Ray worker."""
        return super().run_trial(trial_config)

    def __del__(self):
        """Ensure cleanup on actor destruction."""
        self.cleanup_resources()

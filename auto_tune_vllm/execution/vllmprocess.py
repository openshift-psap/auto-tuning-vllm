"""vLLM process management for auto-tuning framework."""

import logging
import os
import signal
import socket
import subprocess
import threading
import time

import requests

logger = logging.getLogger(__name__)


class vLLMProcess:
    """Manages the lifecycle of a vLLM server process.

    This class handles:
    - Starting vLLM server with configurable parameters
    - Waiting for server readiness via health checks
    - Background health monitoring
    - Process output logging
    - Graceful shutdown and cleanup
    """

    def _wait_for_server_start(self):
        """Wait for vLLM server to become healthy.

        Polls the health endpoint until it returns 200 or timeout is reached.

        Raises:
            RuntimeError: If server fails to start within timeout or process dies
        """
        logger.info(f"Waiting for vLLM server to start on port {self.port}...")
        health_url = self.get_url_for("health")
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            # Check if process died during startup
            if self.proc and self.proc.poll() is not None:
                exit_code = self.proc.returncode
                raise RuntimeError(
                    f"vLLM process died during startup with exit code {exit_code}"
                )

            # Try health check
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    logger.info(f"vLLM server ready after {elapsed:.1f} seconds")
                    return
            except requests.exceptions.RequestException:
                pass  # Expected during startup

            # Wait before retrying
            time.sleep(2)

        # Timeout reached
        elapsed = time.time() - start_time
        raise RuntimeError(
            f"vLLM server failed to start within {self.startup_timeout} seconds "
            f"(waited {elapsed:.1f} seconds)"
        )

    def _start_server(self):
        """Start the vLLM server process.

        This method:
        1. Builds the vLLM command
        2. Starts the process with a new session
        3. Waits for server to become healthy
        4. Starts background logging and health monitoring

        Raises:
            RuntimeError: If process fails to start or become healthy
        """
        # Build command and environment
        cmd = self._build_command()
        env = os.environ.copy()
        env.update(self.env_vars)

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")

        # Start process
        self.proc: subprocess.Popen[str] = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            start_new_session=True,  # Create new process group for clean cleanup
        )
        pgid = os.getpgid(self.proc.pid)
        logger.info(f"Started vLLM process (PID: {self.proc.pid}, PGID: {pgid})")
        # Wait for server to become healthy
        try:
            self._wait_for_server_start()
        except Exception as e:
            # Cleanup on startup failure
            logger.error(f"Server startup failed, cleaning up: {e}")
            self.stop(timeout=5)
            raise
        # Start background threads
        self._start_logging_thread()
        self._start_health_monitoring()

        logger.info(f"vLLM server fully initialized at {self.get_url_for('v1')}")

    def __init__(
        self,
        model: str,
        port: int | None = None,
        host: str = "0.0.0.0",
        vllm_args: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        startup_timeout: int = 300,
        health_check_interval: float = 1.0,
        health_check_max_failures: int = 3,
    ):
        """Initialize vLLM process manager.

        Args:
            model: Model name or path to serve
            port: Server port (auto-allocated if not provided)
            host: Server host address
            max_num_seqs: Maximum number of sequences per iteration
            gpu_memory_utilization: GPU memory utilization (0-1)
            tensor_parallel_size: Number of tensor parallel replicas
            enable_prefix_caching: Whether to enable prefix caching
            extra_args: Additional vLLM CLI arguments
            env_vars: Environment variables for the process
            startup_timeout: Maximum time to wait for server startup (seconds)
            health_check_interval: Interval between health checks (seconds)
            health_check_max_failures: Max consecutive health check failures before
                termination
        """
        self.model: str = model
        self.port: int = port or self._get_available_port()
        self.host: str = host
        self.vllm_args: list[str] = vllm_args or []
        self.env_vars: dict[str, str] = env_vars or {}
        self.startup_timeout: int = startup_timeout
        self.health_check_interval: float = health_check_interval
        self.health_check_max_failures: int = health_check_max_failures

        # Health monitoring state
        self._health_monitor_thread: threading.Thread | None = None
        self._health_monitor_stop_event: threading.Event = threading.Event()
        self._health_check_failed: bool = False
        self._health_check_failure_reason: str | None = None

        # Logging state
        self._logging_thread: threading.Thread | None = None
        self._logging_stop_event: threading.Event = threading.Event()
        self._start_server()

    def _get_available_port(self) -> int:
        """Find an available port using socket binding.

        Returns:
            Available port number
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _build_command(self) -> list[str]:
        """Build the vLLM server command.

        Returns:
            Command arguments as list
        """
        cmd = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--host",
            self.host,
        ]
        # Add extra arguments
        cmd.extend(self.vllm_args)

        return cmd

    def _start_logging_thread(self):
        """Start background thread to log process output."""

        def log_output():
            logger.info(f"Starting vLLM output logging for PID {self.proc.pid}")
            if self.proc.stdout:
                for line in iter(self.proc.stdout.readline, ""):
                    if self._logging_stop_event.is_set():
                        break
                    if line:
                        logger.debug(f"[vLLM {self.proc.pid}] {line.rstrip()}")
            else:
                logger.warning(f"Unable to read stdout of {self.proc.pid}")

        self._logging_thread = threading.Thread(target=log_output, daemon=True)
        self._logging_thread.start()

    def _start_health_monitoring(self):
        """Start background thread to monitor server health."""

        def monitor_health():
            logger.info(
                f"Starting health monitoring for vLLM server on port {self.port}"
            )
            consecutive_failures = 0
            health_url = self.get_url_for("health")

            while not self._health_monitor_stop_event.is_set():
                # Check if process is still alive
                if self.proc and self.proc.poll() is not None:
                    exit_code = self.proc.returncode
                    self._health_check_failed = True
                    self._health_check_failure_reason = (
                        f"Process died with exit code {exit_code}"
                    )
                    logger.error(
                        f"vLLM process died: {self._health_check_failure_reason}"
                    )
                    break

                # Check health endpoint
                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        logger.warning(
                            f"Health check returned status {response.status_code} "
                            f"({consecutive_failures}/{self.health_check_max_failures})"
                        )
                except requests.exceptions.RequestException as e:
                    consecutive_failures += 1
                    logger.warning(
                        f"Health check failed: {e} "
                        f"({consecutive_failures}/{self.health_check_max_failures})"
                    )

                # Check if max failures exceeded
                if consecutive_failures >= self.health_check_max_failures:
                    self._health_check_failed = True
                    self._health_check_failure_reason = (
                        f"Health check failed {consecutive_failures} consecutive times"
                    )
                    logger.error(
                        f"vLLM server unhealthy: {self._health_check_failure_reason}"
                    )
                    break

                # Wait before next check
                self._health_monitor_stop_event.wait(self.health_check_interval)

            logger.info(
                f"Stopped health monitoring for vLLM server on port {self.port}"
            )

        self._health_monitor_thread = threading.Thread(
            target=monitor_health, daemon=True
        )
        self._health_monitor_thread.start()

    def stop(self, timeout: int = 10):
        """Stop the vLLM server gracefully.

        This method:
        1. Stops health monitoring and logging threads
        2. Sends SIGTERM to process group
        3. Waits for graceful shutdown
        4. Forces SIGKILL if timeout exceeded

        Args:
            timeout: Maximum time to wait for graceful shutdown (seconds)
        """
        logger.info(f"Stopping vLLM server (PID: {self.proc.pid})...")

        # Stop monitoring threads
        self._health_monitor_stop_event.set()
        self._logging_stop_event.set()

        # Check if process is already dead
        if self.proc.poll() is not None:
            logger.info(
                f"vLLM process already terminated with code {self.proc.returncode}"
            )
            self._cleanup()
            return

        try:
            # Try graceful shutdown with SIGTERM
            pgid = os.getpgid(self.proc.pid)
            logger.info(f"Sending SIGTERM to process group {pgid}")
            os.killpg(pgid, signal.SIGTERM)

            # Wait for graceful shutdown
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.proc.poll() is not None:
                    logger.info("vLLM process terminated gracefully")
                    self._cleanup()
                    return
                time.sleep(0.5)

            # Force kill if still alive
            logger.warning(
                f"vLLM process did not terminate within {timeout}s, forcing SIGKILL"
            )
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait(timeout=5)
            logger.info("vLLM process force killed")

        except ProcessLookupError:
            # Process already gone
            logger.info("vLLM process already terminated")
        except Exception as e:
            logger.error(f"Error during vLLM process termination: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up process resources."""
        if self.proc:
            if self.proc.stdout:
                self.proc.stdout.close()

            # Wait for threads to finish
            if self._health_monitor_thread and self._health_monitor_thread.is_alive():
                self._health_monitor_thread.join(timeout=2)

            if self._logging_thread and self._logging_thread.is_alive():
                self._logging_thread.join(timeout=2)

    def is_healthy(self) -> bool:
        """Check if the server is currently healthy.

        Returns:
            True if server is running and healthy, False otherwise
        """
        if self.proc.poll() is not None:
            return False

        if self._health_check_failed:
            return False

        return True

    def get_base_url(self) -> str:
        """Get the base server URL without any path extension.

        Returns:
            Base URL (e.g., "http://localhost:8000")
        """
        return f"http://localhost:{self.port}"

    def get_url_for(self, extension: str) -> str:
        """Get the server URL with a specific path extension.

        Args:
            extension: Path extension to append (e.g., "/v1", "/health")

        Returns:
            Complete URL with extension
        """
        base = self.get_base_url()
        # Ensure extension starts with /
        if not extension.startswith("/"):
            extension = f"/{extension}"
        return f"{base}{extension}"

    def get_failure_reason(self) -> str | None:
        """Get the reason for health check failure, if any.

        Returns:
            Failure reason string or None if healthy
        """
        return self._health_check_failure_reason

    def __enter__(self):
        """Context manager entry."""
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        """Context manager exit."""
        self.stop()
        return False

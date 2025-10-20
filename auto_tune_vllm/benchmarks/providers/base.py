"""Abstract benchmark provider base class."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any

from ..config import BenchmarkConfig

logger = logging.getLogger(__name__)


class BenchmarkProvider(ABC):
    """Abstract benchmark provider interface."""

    def __init__(self):
        # Default to module logger
        self._logger: logging.Logger = logger
        # Store trial context for file paths
        self._trial_context: dict[str, str] | None = None
        # Track running benchmark process for termination
        self._process: subprocess.Popen[str] | None = None
        # Store PID for cleanup even if process handle is gone
        self._process_pid: int | None = None
        # Store process group ID for cleanup
        self._process_pgid: int | None = None
        # Function to check for cancellation
        self._cancellation_flag: bool | None = None
        self._started: bool = False

    def __del__(self):
        self.terminate_benchmark()

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        """Context manager exit point - ensures benchmark process cleanup."""
        self.terminate_benchmark()
        return False  # Don't suppress exceptions

    def set_logger(self, custom_logger: logging.Logger):
        """Set a custom logger for this benchmark provider."""
        self._logger = custom_logger

    def set_trial_context(self, study_name: str, trial_id: str):
        """Set trial context for benchmark result storage."""
        self._trial_context = {"study_name": study_name, "trial_id": trial_id}

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
            f"Benchmark: Terminating benchmark process {pid} and its process group..."
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
                    f"Benchmark -> Process Group: Sent SIGTERM to group {pgid}"
                )
            else:
                os.kill(pid, signal.SIGTERM)
                self._logger.info(
                    f"Benchmark -> Process: Sent SIGTERM to process {pid}"
                )
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
                for _ in range(int(wait_timeout * 10)):
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        time.sleep(0.1)
                    except (OSError, ProcessLookupError):
                        # Process is gone
                        break
                else:
                    # Timeout - process still exists
                    raise subprocess.TimeoutExpired("", wait_timeout)

            self._logger.info(
                f"Benchmark: Process {pid} terminated gracefully via SIGTERM"
            )
        except subprocess.TimeoutExpired:
            self._logger.warning(
                f"Benchmark: Process {pid} did not terminate within {wait_timeout}s. "
                + "Escalating to SIGKILL..."
            )

            # Force kill with SIGKILL
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                    self._logger.info(
                        f"Benchmark -> Process Group: Sent SIGKILL to group {pgid}"
                    )
                else:
                    os.kill(pid, signal.SIGKILL)
                    self._logger.info(
                        f"Benchmark -> Process: Sent SIGKILL to process {pid}"
                    )
                self._logger.info(f"Benchmark: Process {pid} force killed via SIGKILL")
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
        self,
        model_url: str,
        config: BenchmarkConfig,
    ) -> subprocess.Popen[str]:
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
    def parse_results(self) -> dict[str, Any]:
        """
        Parse benchmark results from output file.

        Returns:
            Dictionary with benchmark results. Must include metrics that can be
            converted to objective values for Optuna.
        """
        pass

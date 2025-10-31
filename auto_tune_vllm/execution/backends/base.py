"""Execution backend abstractions for Ray and local execution."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import ray

from ...core.trial import TrialConfig, TrialResult

logger = logging.getLogger(__name__)

# Simple Ray actor to hold cancellation state that can be modified externally


class ExecutionBackend(ABC):
    """Abstract execution backend - supports Ray or local execution."""

    @abstractmethod
    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit a trial for execution."""
        pass

    @abstractmethod
    def poll_trials(
        self, job_handles: List[JobHandle]
    ) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll for completed trials, return completed results and remaining handles."""
        pass

    @abstractmethod
    def shutdown(self):
        """Clean shutdown of backend resources."""
        pass

    @abstractmethod
    def cleanup_all_trials(self):
        """Force cleanup of all active trials and their resources (vLLM processes)."""
        pass


@ray.remote
class CancellationFlag:
    """Lightweight Ray actor to hold mutable cancellation state."""

    def __init__(self):
        self.cancelled: bool = False

    def request_cancellation(self):
        """Set cancellation flag to True."""
        self.cancelled = True
        return True

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self.cancelled


@dataclass
class JobHandle:
    """Handle for submitted trial job."""

    trial_id: str
    backend_job_id: str  # Ray ObjectRef ID, process PID, etc.
    status: str = "running"  # "running", "completed", "failed"
    submitted_at: float = 0.0

    def __post_init__(self):
        if self.submitted_at == 0.0:
            self.submitted_at = time.time()

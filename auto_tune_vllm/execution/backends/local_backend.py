"""Execution backend abstractions for Ray and local execution."""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Dict, List, Tuple

from ...core.trial import TrialConfig, TrialResult
from ..controllers.trial_controller import LocalTrialController
from .base import ExecutionBackend, JobHandle

logger = logging.getLogger(__name__)


class LocalExecutionBackend(ExecutionBackend):
    """Local execution backend using thread/process pool."""

    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent
        )
        self.active_futures: Dict[str, concurrent.futures.Future] = {}

    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit trial for local execution."""

        # Create controller and submit to executor
        controller = LocalTrialController()

        future = self.executor.submit(controller.run_trial, trial_config)

        job_id = str(id(future))  # Use future object ID as job ID
        self.active_futures[job_id] = future

        logger.info(f"Submitted trial {trial_config.trial_id} for local execution")
        return JobHandle(trial_config.trial_id, job_id)

    def poll_trials(
        self, job_handles: List[JobHandle]
    ) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll for completed local trials."""
        if not job_handles:
            return [], []

        completed_results = []
        remaining_handles = []

        for handle in job_handles:
            future = self.active_futures.get(handle.backend_job_id)

            if future and future.done():
                try:
                    result = future.result()
                    completed_results.append(result)
                    logger.info(f"Completed local trial {handle.trial_id}")
                    # Remove from active futures
                    del self.active_futures[handle.backend_job_id]
                except Exception as e:
                    # Trial failed - create error result
                    from ..core.trial import ExecutionInfo, TrialResult

                    error_result = TrialResult(
                        trial_id=handle.trial_id,
                        objective_values=[],
                        detailed_metrics={},
                        execution_info=ExecutionInfo(),
                        success=False,
                        error_message=str(e),
                    )
                    completed_results.append(error_result)
                    logger.error(f"Local trial {handle.trial_id} failed: {e}")
                    # Remove from active futures
                    del self.active_futures[handle.backend_job_id]
            else:
                remaining_handles.append(handle)

        return completed_results, remaining_handles

    def cleanup_all_trials(self):
        """Cleanup all active trials (stub implementation for local backend)."""
        logger.info("Local backend does not require explicit trial cleanup")
        # Local backend doesn't need to do anything special here
        # Individual trial controllers handle their own cleanup when they complete

    def shutdown(self):
        """Shutdown thread pool executor."""
        self.executor.shutdown(wait=True)
        logger.info("Shutdown local execution backend")

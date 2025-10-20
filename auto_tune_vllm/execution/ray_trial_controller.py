"""Trial controller implementations for Ray and local execution."""

from __future__ import annotations

import logging
import os
from typing import Any

import ray
from typing_extensions import override

from ..core.trial import TrialConfig, TrialResult
from .trial_controller import BaseTrialController


class RayWorkerTrialController(BaseTrialController):
    """Ray worker node trial controller with Ray-specific functionality."""

    @override
    def _get_worker_id(self) -> str:
        """Get Ray worker node ID."""
        try:
            return ray.get_runtime_context().get_node_id()
        except Exception:
            return "ray_worker_unknown"

    @override
    def _start_vllm_server(self, trial_config: TrialConfig) -> dict[str, Any]:
        """Start vLLM server with GPU assignment from Ray."""
        # Log Ray-specific GPU assignment info to vLLM logger
        vllm_logger: logging.Logger = self._get_trial_logger("vllm")

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

    @override
    def run_trial(
        self, trial_config: TrialConfig, cancellation_flag_actor=None
    ) -> TrialResult:
        """Run trial on Ray worker with optional cancellation flag actor."""
        return super().run_trial(trial_config, cancellation_flag_actor)

    def __del__(self):
        """Ensure cleanup on actor destruction."""
        self.cleanup_resources()

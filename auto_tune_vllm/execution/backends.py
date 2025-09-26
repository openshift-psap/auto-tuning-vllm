"""Execution backend abstractions for Ray and local execution."""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

from ..core.trial import TrialConfig, TrialResult

logger = logging.getLogger(__name__)


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


class ExecutionBackend(ABC):
    """Abstract execution backend - supports Ray or local execution."""
    
    @abstractmethod
    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit a trial for execution."""
        pass
    
    @abstractmethod
    def poll_trials(self, job_handles: List[JobHandle]) -> Tuple[List[TrialResult], List[JobHandle]]:
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




class RayExecutionBackend(ExecutionBackend):
    """Ray-based distributed execution backend."""
    
    def __init__(
        self, 
        resource_requirements: Optional[Dict[str, float]] = None, 
        start_ray_head: bool = False,
        python_executable: Optional[str] = None,
        venv_path: Optional[str] = None,
        conda_env: Optional[str] = None
    ):
        # Legacy: resource_requirements per backend (now calculated per trial)
        self.resource_requirements = resource_requirements or {"num_gpus": 1, "num_cpus": 4}
        self.active_jobs: Dict[str, object] = {}  # job_id -> ray_ref
        self.active_actors: Dict[str, object] = {}  # job_id -> ray_actor
        self.start_ray_head = start_ray_head
        self._started_ray_head = False  # Track if we started Ray head for cleanup
        
        # Python environment configuration
        self.python_executable = python_executable
        self.venv_path = venv_path
        self.conda_env = conda_env
        
        self._ensure_ray_initialized()
    
    def _build_runtime_env(self) -> Dict:
        """Build Ray runtime environment configuration for Python."""
        runtime_env = {}
        
        # Method 1: Explicit Python executable path
        if self.python_executable:
            runtime_env["python"] = self.python_executable
            logger.info(f"Ray workers will use Python executable: {self.python_executable}")
        
        # Method 2: Virtual environment path
        elif self.venv_path:
            venv_python = Path(self.venv_path) / "bin" / "python"
            if venv_python.exists():
                runtime_env["python"] = str(venv_python)
                logger.info(f"Ray workers will use venv Python: {venv_python}")
            else:
                logger.warning(f"Virtual environment not found at {self.venv_path}, trying python3")
                venv_python3 = Path(self.venv_path) / "bin" / "python3"
                if venv_python3.exists():
                    runtime_env["python"] = str(venv_python3)
                    logger.info(f"Ray workers will use venv Python3: {venv_python3}")
                else:
                    raise RuntimeError(f"No Python executable found in venv: {self.venv_path}")
        
        # Method 3: Conda environment
        elif self.conda_env:
            runtime_env["conda"] = self.conda_env
            logger.info(f"Ray workers will use conda environment: {self.conda_env}")
        
        # Method 4: Auto-detect current environment
        else:
            current_python = sys.executable
            
            # Check if we're in a virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                # We're in a virtual environment
                runtime_env["python"] = current_python
                logger.info(f"Auto-detected virtual environment, Ray workers will use: {current_python}")
            else:
                logger.warning(
                    "No Python environment specified and not in a virtual environment. "
                    "Ray workers may use different Python installations. "
                    "Consider using --python-executable, --venv-path, or --conda-env options."
                )
        
        return runtime_env
    
    def _ensure_ray_initialized(self):
        """Initialize Ray if not already initialized."""
        try:
            import ray
            if not ray.is_initialized():
                try:
                    # First try to connect to existing cluster
                    ray.init(address="auto", ignore_reinit_error=True)
                    logger.info("Connected to existing Ray cluster")
                except Exception as e:
                    if self.start_ray_head:
                        logger.info("No existing Ray cluster found, starting Ray head...")
                        self._start_ray_head()
                        logger.info("Started Ray head successfully")
                    else:
                        raise RuntimeError(
                            f"Failed to connect to Ray cluster: {e}\n"
                            f"Use --start-ray-head to automatically start a Ray head, or start one manually:\n"
                            f"  ray start --head --port=10001"
                        )
        except ImportError:
            raise ImportError("Ray is required for RayExecutionBackend. Install with: pip install ray[default]")
    
    def _start_ray_head(self):
        """Start a Ray head node."""
        import time
        import ray
        
        try:
            # Start Ray head node with default settings (let Ray choose ports)
            cmd = [
                "ray", "start", 
                "--head", 
                "--dashboard-host=0.0.0.0"
            ]
            
            logger.info(f"Starting Ray head with command: {' '.join(cmd)}")
            
            # Check for ray available in path
            if shutil.which("ray") is None:
                raise RuntimeError("Ray is not installed. Cannot start Ray head. Install or add Ray to PATH.")
            
            # Start Ray head as subprocess
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            logger.info(f"Ray head start output: {process.stdout}")
            if process.stderr:
                logger.warning(f"Ray head start stderr: {process.stderr}")
            
            # Wait a moment for Ray head to initialize
            time.sleep(3)
            
            # Connect to the newly started Ray head using auto-discovery
            ray.init(address="auto", ignore_reinit_error=True)
            self._started_ray_head = True
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start Ray head: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Ray head start timed out after 30 seconds") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error starting Ray head: {e}") from e
    
    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit trial to Ray cluster."""
        from .trial_controller import RayTrialActor
        
        # Create Ray actor with resource requirements from trial config
        # Extract num_gpus and num_cpus from trial's resource_requirements
        num_gpus = trial_config.resource_requirements.get("num_gpus", 1)
        num_cpus = trial_config.resource_requirements.get("num_cpus", 1)
        
        # Filter out any other custom resources from trial config
        custom_resources = {k: v for k, v in trial_config.resource_requirements.items() 
                          if k not in ["num_gpus", "num_cpus"]}
        
        # Build runtime environment for Python configuration
        runtime_env = self._build_runtime_env()
        
        # Create controller with runtime environment
        controller_options = {
            "num_gpus": num_gpus,
            "num_cpus": num_cpus
        }
        
        # Add custom resources if any
        if custom_resources:
            controller_options["resources"] = custom_resources
        
        # Add runtime environment if configured
        if runtime_env:
            controller_options["runtime_env"] = runtime_env
        
        controller = RayTrialActor.options(**controller_options).remote()

        # Submit trial execution
        future_ref = controller.run_trial.remote(trial_config)
        job_id = str(future_ref)  # Use Ray ObjectRef as job ID

        # Track active job and actor
        self.active_jobs[job_id] = future_ref
        self.active_actors[job_id] = controller
        
        logger.info(f"Submitted trial {trial_config.trial_id} to Ray cluster")
        return JobHandle(trial_config.trial_id, job_id)
    
    def poll_trials(self, job_handles: List[JobHandle]) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll for completed Ray trials."""
        import ray
        
        if not job_handles:
            return [], []
        
        # Get Ray refs for active jobs
        active_refs = []
        handle_map = {}
        
        for handle in job_handles:
            if handle.backend_job_id in self.active_jobs:
                ray_ref = self.active_jobs[handle.backend_job_id]
                active_refs.append(ray_ref)
                handle_map[ray_ref] = handle
        
        if not active_refs:
            return [], job_handles
        
        # Check which trials are ready (non-blocking)
        ready_refs, pending_refs = ray.wait(active_refs, num_returns=len(active_refs), timeout=0)
        
        completed_results = []
        remaining_handles = []
        
        for handle in job_handles:
            ray_ref = self.active_jobs.get(handle.backend_job_id)
            
            if ray_ref in ready_refs:
                try:
                    result = ray.get(ray_ref)  # Get completed result
                    completed_results.append(result)
                    logger.info(f"Completed trial {handle.trial_id}")
                    # Remove from active jobs and actors
                    del self.active_jobs[handle.backend_job_id]
                    if handle.backend_job_id in self.active_actors:
                        del self.active_actors[handle.backend_job_id]
                except Exception as e:
                    # Trial failed - create error result
                    from ..core.trial import TrialResult, ExecutionInfo
                    error_result = TrialResult(
                        trial_id=handle.trial_id,
                        objective_values=[],
                        detailed_metrics={},
                        execution_info=ExecutionInfo(),
                        success=False,
                        error_message=str(e)
                    )
                    completed_results.append(error_result)
                    logger.error(f"Trial {handle.trial_id} failed: {e}")
                    # Remove from active jobs and actors
                    del self.active_jobs[handle.backend_job_id]
                    if handle.backend_job_id in self.active_actors:
                        del self.active_actors[handle.backend_job_id]
            else:
                remaining_handles.append(handle)

        return completed_results, remaining_handles

    def cleanup_all_trials(self):
        """Force cleanup of all active trials and their vLLM processes."""
        import ray

        if not self.active_actors:
            logger.info("No active trials to cleanup")
            return

        logger.info(f"Cleaning up {len(self.active_actors)} active trials...")

        # Call cleanup_resources on all active actors with timeout
        cleanup_futures = []
        for job_id, actor in self.active_actors.items():
            try:
                # Call cleanup asynchronously with timeout
                cleanup_ref = actor.cleanup_resources.remote()
                cleanup_futures.append((job_id, cleanup_ref))
                logger.info(f"Initiated cleanup for trial {job_id}")
            except Exception as e:
                logger.warning(f"Failed to initiate cleanup for trial {job_id}: {e}")

        # Wait for cleanup with timeout
        timeout = 30  # 30 seconds timeout
        if cleanup_futures:
            try:
                refs_only = [ref for _, ref in cleanup_futures]
                ready_refs, remaining_refs = ray.wait(refs_only, num_returns=len(refs_only), timeout=timeout)

                # Log results
                if ready_refs:
                    logger.info(f"Successfully cleaned up {len(ready_refs)} trials")
                if remaining_refs:
                    logger.warning(f"Cleanup timeout for {len(remaining_refs)} trials")

            except Exception as e:
                logger.error(f"Error during cleanup wait: {e}")

        # Force kill actors that didn't cleanup properly
        for job_id, actor in list(self.active_actors.items()):
            try:
                ray.kill(actor)
                logger.info(f"Force killed actor for trial {job_id}")
            except Exception as e:
                logger.warning(f"Failed to kill actor for trial {job_id}: {e}")

        # Clear tracking
        self.active_actors.clear()
        self.active_jobs.clear()
        logger.info("Completed cleanup of all active trials")

    def shutdown(self):
        """Shutdown Ray cluster connection."""
        import ray
        
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Shutdown Ray cluster connection")
        
        # If we started the Ray head, stop it
        if self._started_ray_head:
            try:
                logger.info("Stopping Ray head that we started...")
                process = subprocess.run(
                    ["ray", "stop"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                logger.info(f"Ray stop output: {process.stdout}")
                if process.stderr:
                    logger.warning(f"Ray stop stderr: {process.stderr}")
                self._started_ray_head = False
                logger.info("Successfully stopped Ray head")
            except Exception as e:
                logger.error(f"Failed to stop Ray head: {e}")
                # Try force stop
                try:
                    logger.info("Attempting force stop of Ray processes...")
                    subprocess.run(["pkill", "-f", "ray::"], timeout=5)
                except Exception:
                    logger.error("Force stop also failed")




class LocalExecutionBackend(ExecutionBackend):
    """Local execution backend using thread/process pool."""
    
    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
        self.active_futures: Dict[str, concurrent.futures.Future] = {}
    
    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit trial for local execution."""
        from .trial_controller import LocalTrialController
        
        # Create controller and submit to executor
        controller = LocalTrialController()
        
        future = self.executor.submit(
            controller.run_trial,
            trial_config
        )
        
        job_id = str(id(future))  # Use future object ID as job ID
        self.active_futures[job_id] = future
        
        logger.info(f"Submitted trial {trial_config.trial_id} for local execution")
        return JobHandle(trial_config.trial_id, job_id)
    
    def poll_trials(self, job_handles: List[JobHandle]) -> Tuple[List[TrialResult], List[JobHandle]]:
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
                    from ..core.trial import TrialResult, ExecutionInfo
                    error_result = TrialResult(
                        trial_id=handle.trial_id,
                        objective_values=[],
                        detailed_metrics={},
                        execution_info=ExecutionInfo(),
                        success=False,
                        error_message=str(e)
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
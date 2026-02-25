"""Execution backend abstractions for Ray and local execution."""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

import yaml

from ..core.trial import TrialConfig, TrialResult

logger = logging.getLogger(__name__)


# Simple Ray actor to hold cancellation state that can be modified externally

if RAY_AVAILABLE:

    @ray.remote
    class CancellationFlag:
        """Lightweight Ray actor to hold mutable cancellation state."""

        def __init__(self):
            self.cancelled = False

        def request_cancellation(self):
            """Set cancellation flag to True."""
            self.cancelled = True
            return True

        def is_cancelled(self):
            """Check if cancellation was requested."""
            return self.cancelled
else:
    # Dummy class when Ray is not available
    class CancellationFlag:
        """Dummy cancellation flag when Ray is not available."""

        pass


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


class RayExecutionBackend(ExecutionBackend):
    """Ray-based distributed execution backend."""

    # Cleanup timeouts (in seconds)
    CANCELLATION_FLAG_TIMEOUT = 2
    CANCELLATION_DETECTION_WAIT = 5
    TASK_CANCELLATION_WAIT = 2
    GRACEFUL_CLEANUP_TIMEOUT = 180  # 3 minutes for vLLM + GuideLLM shutdown

    def __init__(
        self,
        resource_requirements: Optional[Dict[str, float]] = None,
        start_ray_head: bool = True,
        python_executable: Optional[str] = None,
        venv_path: Optional[str] = None,
        conda_env: Optional[str] = None,
    ):
        # Legacy: resource_requirements per backend (now calculated per trial)
        self.resource_requirements = resource_requirements or {
            "num_gpus": 1,
            "num_cpus": 4,
        }
        self.active_jobs: Dict[str, object] = {}  # job_id -> ray_ref (workload future)
        self.active_actors: Dict[str, object] = {}  # job_id -> workload_actor
        self.vllm_actors: Dict[str, object] = {}  # job_id -> vllm_actor
        self.shared_states: Dict[str, object] = {}  # job_id -> shared_state_actor
        self.cancellation_flags: Dict[
            str, object
        ] = {}  # job_id -> cancellation_flag_actor
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
            logger.info(
                f"Ray workers will use Python executable: {self.python_executable}"
            )

        # Method 2: Virtual environment path
        elif self.venv_path:
            venv_python = Path(self.venv_path) / "bin" / "python"
            if venv_python.exists():
                runtime_env["python"] = str(venv_python)
                logger.info(f"Ray workers will use venv Python: {venv_python}")
            else:
                logger.warning(
                    f"Virtual environment not found at {self.venv_path}, trying python3"
                )
                venv_python3 = Path(self.venv_path) / "bin" / "python3"
                if venv_python3.exists():
                    runtime_env["python"] = str(venv_python3)
                    logger.info(f"Ray workers will use venv Python3: {venv_python3}")
                else:
                    raise RuntimeError(
                        f"No Python executable found in venv: {self.venv_path}"
                    )

        # Method 3: Conda environment
        elif self.conda_env:
            runtime_env["conda"] = self.conda_env
            logger.info(f"Ray workers will use conda environment: {self.conda_env}")

        # Method 4: Auto-detect current environment
        else:
            current_python = sys.executable

            # Check if we're in a virtual environment
            if hasattr(sys, "real_prefix") or (
                hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
            ):
                # We're in a virtual environment
                runtime_env["python"] = current_python
                logger.info(
                    f"Auto-detected virtual environment, "
                    f"Ray workers will use: {current_python}"
                )
            else:
                logger.warning(
                    "No Python environment specified and not in a virtual environment. "
                    "Ray workers may use different Python installations. Consider using"
                    " --python-executable, --venv-path, or --conda-env options."
                )

        return runtime_env

    def _ensure_ray_initialized(self):
        """Initialize Ray if not already initialized."""
        try:
            if not ray.is_initialized():
                try:
                    # First try to connect to existing cluster
                    ray.init(address="auto", ignore_reinit_error=True)
                    logger.info("Connected to existing Ray cluster")
                except Exception as e:
                    if self.start_ray_head:
                        logger.info(
                            "No existing Ray cluster found, starting Ray head..."
                        )
                        self._start_ray_head()
                        logger.info("Started Ray head successfully")
                    else:
                        raise RuntimeError(
                            f"Failed to connect to Ray cluster: {e}\n"
                            f"Use --start-ray-head to automatically start a Ray head, "
                            f"or start one manually:\n"
                            f"  ray start --head --port=10001"
                        )
        except ImportError:
            raise ImportError(
                "Ray is required for RayExecutionBackend. "
                "Install with: pip install ray[default]"
            )

    def _start_ray_head(self):
        """Start a Ray head node."""
        try:
            # Start Ray head node with default settings (let Ray choose ports)
            cmd = ["ray", "start", "--head", "--dashboard-host=0.0.0.0"]

            logger.info(f"Starting Ray head with command: {' '.join(cmd)}")

            # Check for ray available in path
            if shutil.which("ray") is None:
                raise RuntimeError(
                    "Ray is not installed. Cannot start Ray head. "
                    "Install or add Ray to PATH."
                )

            # Start Ray head as subprocess
            process = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=30
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
        """Submit trial to Ray cluster using separate vLLM and workload actors."""
        from .trial_controller import SharedState, VLLMServerActor, WorkloadActor

        # Create cancellation flag actor
        cancellation_flag_actor = CancellationFlag.remote()

        # Extract resource requirements
        num_gpus = trial_config.resource_requirements.get("num_gpus", 1)
        num_cpus = trial_config.resource_requirements.get("num_cpus", 1)

        # Filter out any other custom resources from trial config
        custom_resources = {
            k: v
            for k, v in trial_config.resource_requirements.items()
            if k not in ["num_gpus", "num_cpus"]
        }

        # Build runtime environment for Python configuration
        runtime_env = self._build_runtime_env()

        # Create SharedState actor (lightweight, no resources needed)
        shared_state_actor = SharedState.remote()

        # Create VLLMServerActor with GPU resources
        vllm_options = {
            "num_gpus": num_gpus,
            "num_cpus": 1,
        }  # Minimal CPU for vLLM actor
        if custom_resources:
            vllm_options["resources"] = custom_resources
        if runtime_env:
            vllm_options["runtime_env"] = runtime_env

        vllm_actor = VLLMServerActor.options(**vllm_options).remote()

        # Start vLLM server
        server_info_ref = vllm_actor.start_server.remote(trial_config)
        server_info = ray.get(server_info_ref)
        server_url = server_info["url"]

        # Store server URL in shared state
        shared_state_actor.set_server_url.remote(server_url)

        # Wait for server to be ready
        ready = ray.get(
            vllm_actor.wait_for_ready.remote(trial_config.vllm_startup_timeout)
        )
        if not ready:
            # Cleanup on failure
            vllm_actor.cleanup.remote()
            raise RuntimeError(
                f"vLLM server failed to start within {trial_config.vllm_startup_timeout}s"
            )

        # Start health monitoring
        health_url = server_url.replace("/v1", "/health")
        vllm_actor.start_health_monitoring.remote(
            health_url,
            check_interval=trial_config.health_check_interval,
            max_failures=trial_config.health_check_max_failures,
        )

        # Create WorkloadActor with CPU resources
        workload_options = {"num_cpus": num_cpus}
        if runtime_env:
            workload_options["runtime_env"] = runtime_env

        workload_actor = WorkloadActor.options(**workload_options).remote()

        # Start workload execution
        workload_future = workload_actor.run_workload.remote(
            shared_state_actor, trial_config, cancellation_flag_actor
        )

        job_id = str(workload_future)  # Use Ray ObjectRef as job ID

        # Track all actors and futures
        self.active_jobs[job_id] = workload_future
        self.active_actors[job_id] = workload_actor
        self.vllm_actors[job_id] = vllm_actor
        self.shared_states[job_id] = shared_state_actor
        self.cancellation_flags[job_id] = cancellation_flag_actor

        logger.info(
            f"Submitted trial {trial_config.trial_id} to Ray cluster "
            f"(vLLM actor + workload actor)"
        )
        return JobHandle(trial_config.trial_id, job_id)

    def poll_trials(
        self, job_handles: List[JobHandle]
    ) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll for completed Ray trials."""
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
        ready_refs, _ = ray.wait(active_refs, num_returns=len(active_refs), timeout=0)

        completed_results = []
        remaining_handles = []

        for handle in job_handles:
            ray_ref = self.active_jobs.get(handle.backend_job_id)

            if ray_ref in ready_refs:
                try:
                    result = ray.get(ray_ref)  # Get completed result
                    completed_results.append(result)
                    logger.info(f"Completed trial {handle.trial_id}")

                    # Cleanup actors for this trial
                    self._cleanup_trial_actors(handle.backend_job_id)
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
                    logger.error(f"Trial {handle.trial_id} failed: {e}")

                    # Cleanup actors for this trial
                    self._cleanup_trial_actors(handle.backend_job_id)
            else:
                remaining_handles.append(handle)

        return completed_results, remaining_handles

    def _cleanup_trial_actors(self, job_id: str):
        """Cleanup all actors for a specific trial."""
        # Cleanup vLLM actor
        if job_id in self.vllm_actors:
            try:
                ray.get(self.vllm_actors[job_id].cleanup.remote(), timeout=30)
            except Exception as e:
                logger.warning(f"Error cleaning up vLLM actor for {job_id}: {e}")
            finally:
                del self.vllm_actors[job_id]

        # Remove from tracking dictionaries
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        if job_id in self.active_actors:
            del self.active_actors[job_id]
        if job_id in self.shared_states:
            del self.shared_states[job_id]
        if job_id in self.cancellation_flags:
            del self.cancellation_flags[job_id]

    def _execute_remote_calls(
        self, items: dict, method_name: str, description: str
    ) -> list:
        """Execute remote method calls on multiple actors/refs with error handling.

        Args:
            items: Dict of job_id -> actor/ref
            method_name: Name of remote method to call
            description: Description for logging

        Returns:
            List of (job_id, remote_ref) tuples for successful calls
        """
        futures = []
        for job_id, item in items.items():
            try:
                if method_name:
                    remote_ref = getattr(item, method_name).remote()
                else:
                    remote_ref = item  # Already a ref (for task cancellation)
                futures.append((job_id, remote_ref))
                logger.debug(f"{description}: {job_id}")
            except Exception as e:
                logger.warning(f"Failed {description} for {job_id}: {e}")
        return futures

    def _wait_for_refs(self, futures: list, timeout: float, description: str) -> tuple:
        """Wait for remote refs to complete with timeout.

        Returns:
            Tuple of (ready_count, remaining_count)
        """
        if not futures:
            return 0, 0

        try:
            refs_only = [ref for _, ref in futures]
            ready_refs, remaining_refs = ray.wait(
                refs_only, num_returns=len(refs_only), timeout=timeout
            )

            if ready_refs:
                logger.info(f"✓ {len(ready_refs)} {description} completed")
            if remaining_refs:
                logger.warning(
                    f"⚠ {len(remaining_refs)} {description} timed out after {timeout}s"
                )

            return len(ready_refs), len(remaining_refs)
        except Exception as e:
            logger.error(f"Error waiting for {description}: {e}")
            return 0, len(futures)

    def cleanup_all_trials(self):
        """Force cleanup of all active trials and their vLLM processes.

        Cleanup phases:
        1. Set cancellation flags (triggers polling loop detection)
        2. Cancel Ray tasks (sends cancellation signal)
        3. Call cleanup on vLLM actors (graceful SIGTERM)
        4. Force kill unresponsive actors (SIGKILL)
        """
        if not self.active_actors and not self.vllm_actors:
            logger.debug("No active trials to cleanup")
            return

        total_trials = max(len(self.active_actors), len(self.vllm_actors))
        logger.info(f"Cleaning up {total_trials} active trial(s)")

        # Phase 1: Set cancellation flags
        logger.info("Phase 1 - Setting cancellation flags...")
        cancel_futures = self._execute_remote_calls(
            self.cancellation_flags, "request_cancellation", "Set cancellation flag"
        )
        self._wait_for_refs(
            cancel_futures, self.CANCELLATION_FLAG_TIMEOUT, "cancellation flags"
        )

        # Give polling loops time to detect and terminate benchmarks
        if cancel_futures:
            logger.info(
                f"Waiting {self.CANCELLATION_DETECTION_WAIT}s for polling loops "
                f"to detect cancellation..."
            )
            time.sleep(self.CANCELLATION_DETECTION_WAIT)

        # Phase 2: Cancel Ray tasks (workload futures)
        logger.info("Phase 2 - Cancelling Ray tasks...")
        cancelled = 0
        for job_id, task_ref in self.active_jobs.items():
            try:
                ray.cancel(task_ref, force=False)
                cancelled += 1
            except Exception as e:
                logger.warning(f"Failed to cancel task {job_id}: {e}")

        if cancelled:
            logger.info(
                f"Cancelled {cancelled} Ray task(s), waiting "
                f"{self.TASK_CANCELLATION_WAIT}s..."
            )
            time.sleep(self.TASK_CANCELLATION_WAIT)

        # Phase 3: Call cleanup on vLLM actors
        logger.info("Phase 3 - Requesting graceful cleanup from vLLM actors...")
        vllm_cleanup_futures = self._execute_remote_calls(
            self.vllm_actors, "cleanup", "Sent cleanup request to vLLM actor"
        )
        logger.info(
            f"Waiting up to {self.GRACEFUL_CLEANUP_TIMEOUT}s "
            f"for graceful vLLM cleanup..."
        )
        self._wait_for_refs(
            vllm_cleanup_futures, self.GRACEFUL_CLEANUP_TIMEOUT, "vLLM actor cleanups"
        )

        # Phase 4: Force kill unresponsive actors
        all_actors = {}
        all_actors.update(self.active_actors)
        all_actors.update(self.vllm_actors)

        if all_actors:
            logger.warning(f"Force killing {len(all_actors)} unresponsive actor(s)...")
            for job_id, actor in list(all_actors.items()):
                try:
                    ray.kill(actor)
                    logger.debug(f"Force killed actor {job_id}")
                except Exception as e:
                    logger.warning(f"Failed to kill actor {job_id}: {e}")

        # Clear tracking
        self.active_actors.clear()
        self.active_jobs.clear()
        self.vllm_actors.clear()
        self.shared_states.clear()
        self.cancellation_flags.clear()
        logger.info("✓ Completed cleanup of all active trials")

    def shutdown(self):
        """Shutdown Ray cluster connection."""

        if ray.is_initialized():
            ray.shutdown()
            logger.info("Shutdown Ray cluster connection")

        # If we started the Ray head, stop it
        if self._started_ray_head:
            try:
                logger.info("Stopping Ray head that we started...")
                process = subprocess.run(
                    ["ray", "stop"], capture_output=True, text=True, timeout=10
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
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent
        )
        self.active_futures: Dict[str, concurrent.futures.Future] = {}

    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit trial for local execution."""
        from .trial_controller import LocalTrialController

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


class HelmExecutionBackend(ExecutionBackend):
    """Helm-based Kubernetes execution backend."""

    def __init__(
        self,
        helm_config: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        kubeconfig: Optional[str] = None,
        study_name: Optional[str] = None,
    ):
        """Initialize Helm execution backend.

        Args:
            helm_config: Helm configuration dictionary
            namespace: Kubernetes namespace
            kubeconfig: Path to kubeconfig file
            study_name: Study name/prefix for cleanup purposes
        """
        self.helm_config = helm_config or {}
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.study_name = study_name  # Store study name for cleanup

        # Track active trials: trial_id -> (release_name, future, executor)
        self.active_trials: Dict[str, Tuple[str, Any, Any]] = {}

        # Validate Helm CLI is available
        if shutil.which("helm") is None:
            raise RuntimeError(
                "Helm CLI not found. Please install Helm: "
                "https://helm.sh/docs/intro/install/"
            )

        # Validate Kubernetes access
        self._validate_kubernetes_access()

        # Ensure namespace exists
        self._ensure_namespace_exists()

        logger.info(f"Initialized Helm execution backend (namespace: {self.namespace})")

    def _validate_kubernetes_access(self):
        """Validate Kubernetes cluster access."""
        try:
            from kubernetes import client, config
        except ImportError:
            raise RuntimeError(
                "kubernetes library not available. "
                "Install with: pip install 'auto-tune-vllm[helm]'"
            )

        try:
            if self.kubeconfig:
                config.load_kube_config(config_file=self.kubeconfig)
            else:
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()

            # Test API access
            v1 = client.CoreV1Api()
            v1.list_namespace()  # Note: method is list_namespace (singular), not list_namespaces
            logger.info("Kubernetes cluster access validated")
        except Exception as e:
            raise RuntimeError(f"Failed to access Kubernetes cluster: {e}")

    def _ensure_namespace_exists(self):
        """Ensure Kubernetes namespace exists, create if it doesn't."""
        try:
            from kubernetes import client, config
        except ImportError:
            return  # Already validated in _validate_kubernetes_access

        try:
            if self.kubeconfig:
                config.load_kube_config(config_file=self.kubeconfig)
            else:
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()

            v1 = client.CoreV1Api()

            # Check if namespace exists
            try:
                v1.read_namespace(name=self.namespace)
                logger.debug(f"Namespace {self.namespace} already exists")
            except Exception:
                # Namespace doesn't exist, create it
                logger.info(f"Creating namespace: {self.namespace}")
                namespace_body = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=self.namespace)
                )
                v1.create_namespace(body=namespace_body)
                logger.info(f"Created namespace: {self.namespace}")
        except Exception as e:
            logger.warning(f"Failed to ensure namespace exists: {e}")
            # Don't fail - namespace might be created by cluster admin

    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit trial for Helm-based execution.

        Args:
            trial_config: Trial configuration

        Returns:
            JobHandle with release name as backend_job_id
        """
        import concurrent.futures

        from .helm_utils import (
            generate_helm_values,
            sanitize_release_name,
        )
        from .trial_controller import HelmTrialController

        # Generate release name
        release_name = sanitize_release_name(
            f"{trial_config.study_name}-{trial_config.trial_id}"
        )

        # Generate Helm values
        helm_values = generate_helm_values(trial_config, self.helm_config)

        # Create values file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(helm_values, f)
            values_file = f.name

        try:
            # For prefix-aware routing, deploy full stack: infra, gaie, modelservice
            # Use release name postfix for gaie (e.g., "kv-events")
            # NOTE: infra and gaie are shared across all trials, so only deploy once
            release_name_postfix = self.helm_config.get(
                "release_name_postfix", "kv-events"
            )

            # Deploy infra chart first (only if not already deployed)
            infra_release_name = f"infra-{release_name_postfix}"
            if self.helm_config.get("deploy_full_stack", False):
                # Check if infra release already exists
                result = subprocess.run(
                    ["helm", "list", "-n", self.namespace, "-q"],
                    capture_output=True,
                    text=True,
                )
                infra_exists = infra_release_name in result.stdout

                if not infra_exists:
                    logger.info("Deploying full stack: infra, gaie, modelservice")

                    # 1. Deploy infra chart
                    infra_chart_repo = self.helm_config.get(
                        "infra_chart_repo",
                        "https://llm-d-incubation.github.io/llm-d-infra/",
                    )
                    infra_chart_name = self.helm_config.get(
                        "infra_chart_name", "llm-d-infra"
                    )
                    infra_chart_version = self.helm_config.get(
                        "infra_chart_version", "v1.3.4"
                    )

                    infra_repo_name = (
                        f"helm-repo-infra-{abs(hash(infra_chart_repo)) % 10000}"
                    )
                    result = subprocess.run(
                        ["helm", "repo", "list"],
                        capture_output=True,
                        text=True,
                    )
                    if infra_repo_name not in result.stdout:
                        subprocess.run(
                            ["helm", "repo", "add", infra_repo_name, infra_chart_repo],
                            check=True,
                            capture_output=True,
                        )
                    subprocess.run(
                        ["helm", "repo", "update", infra_repo_name],
                        check=False,
                        capture_output=True,
                    )

                    infra_cmd = [
                        "helm",
                        "install",
                        infra_release_name,
                        f"{infra_repo_name}/{infra_chart_name}",
                        "--version",
                        infra_chart_version,
                        "--namespace",
                        self.namespace,
                        "--wait",
                        "--timeout",
                        "10m",
                    ]
                    logger.info(f"Installing infra Helm release: {infra_release_name}")
                    result = subprocess.run(
                        infra_cmd, check=True, capture_output=True, text=True
                    )
                    logger.info(f"Infra Helm release installed: {result.stdout}")
                else:
                    logger.info(
                        f"Infra Helm release {infra_release_name} already exists, skipping"
                    )

                # 2. Deploy/upgrade gaie chart (depends on infra, redeploy for each trial with trial-specific parameters)
                gaie_release_name = f"gaie-{release_name_postfix}"
                gaie_chart_ref = "oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool"
                gaie_chart_version = self.helm_config.get(
                    "gaie_chart_version", "v1.2.0"
                )

                # Generate GAIE values from trial configuration
                import tempfile

                from .helm_utils import generate_gaie_values

                gaie_values = generate_gaie_values(trial_config, self.helm_config)

                # Write GAIE values to temp file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False
                ) as f:
                    yaml.dump(
                        gaie_values, f, default_flow_style=False, allow_unicode=True
                    )
                    gaie_values_file = f.name

                # Check if GAIE release already exists
                result = subprocess.run(
                    ["helm", "list", "-n", self.namespace, "-q"],
                    capture_output=True,
                    text=True,
                )
                gaie_exists = gaie_release_name in result.stdout

                if gaie_exists:
                    # Upgrade existing release with trial-specific parameters
                    gaie_cmd = [
                        "helm",
                        "upgrade",
                        gaie_release_name,
                        gaie_chart_ref,
                        "--version",
                        gaie_chart_version,
                        "--namespace",
                        self.namespace,
                        "--values",
                        gaie_values_file,
                        "--wait",
                        "--timeout",
                        "10m",
                    ]
                    logger.info(
                        f"Upgrading gaie Helm release: {gaie_release_name} with trial-specific parameters"
                    )
                else:
                    # Install new release
                    gaie_cmd = [
                        "helm",
                        "install",
                        gaie_release_name,
                        gaie_chart_ref,
                        "--version",
                        gaie_chart_version,
                        "--namespace",
                        self.namespace,
                        "--values",
                        gaie_values_file,
                        "--wait",
                        "--timeout",
                        "10m",
                    ]
                    logger.info(
                        f"Installing gaie Helm release: {gaie_release_name} with trial-specific parameters"
                    )

                result = subprocess.run(
                    gaie_cmd, check=True, capture_output=True, text=True
                )
                logger.info(
                    f"Gaie Helm release {'upgraded' if gaie_exists else 'installed'}: {result.stdout}"
                )

                # Clean up temp file
                import os

                if os.path.exists(gaie_values_file):
                    os.unlink(gaie_values_file)

            # 3. Deploy modelservice chart (depends on infra and gaie)
            # Update helm_values to include GAIE_RELEASE_NAME_POSTFIX
            if "decode" not in helm_values:
                helm_values["decode"] = {}
            if "containers" not in helm_values["decode"]:
                helm_values["decode"]["containers"] = [{}]
            container = helm_values["decode"]["containers"][0]
            if "env" not in container:
                container["env"] = []

            # Add GAIE_RELEASE_NAME_POSTFIX to environment variables
            env_dict = {
                env["name"]: env.get("value", "")
                for env in container["env"]
                if "name" in env
            }
            env_dict["GAIE_RELEASE_NAME_POSTFIX"] = release_name_postfix
            container["env"] = [
                {"name": k, "value": str(v)} for k, v in env_dict.items()
            ]

            # Re-write values file with updated env vars
            with open(values_file, "w") as f:
                yaml.dump(helm_values, f)

            # Install modelservice Helm release
            helm_cmd = [
                "helm",
                "install",
                release_name,
            ]

            # Add chart source
            if self.helm_config.get("chart_path"):
                helm_cmd.append(self.helm_config["chart_path"])
            elif self.helm_config.get("chart_repo") and self.helm_config.get(
                "chart_name"
            ):
                repo_url = self.helm_config["chart_repo"]
                chart_name = self.helm_config["chart_name"]
                chart_version = self.helm_config.get("chart_version")

                # Add repo if needed (check if exists first)
                repo_name = f"helm-repo-{abs(hash(repo_url)) % 10000}"
                # Check if repo already exists
                result = subprocess.run(
                    ["helm", "repo", "list"],
                    capture_output=True,
                    text=True,
                )
                repo_exists = repo_name in result.stdout

                if not repo_exists:
                    subprocess.run(
                        ["helm", "repo", "add", repo_name, repo_url],
                        check=True,
                        capture_output=True,
                    )
                # Update repo to ensure we have latest charts
                subprocess.run(
                    ["helm", "repo", "update", repo_name],
                    check=False,  # Don't fail if update fails
                    capture_output=True,
                )

                chart_ref = f"{repo_name}/{chart_name}"
                helm_cmd.append(chart_ref)
                if chart_version:
                    helm_cmd.extend(["--version", chart_version])
            else:
                raise ValueError(
                    "Helm config must specify either chart_path or (chart_repo + chart_name)"
                )

            helm_cmd.extend(
                [
                    "--namespace",
                    self.namespace,
                    "--values",
                    values_file,
                    "--wait",  # Wait for resources to be ready
                    "--timeout",
                    "10m",
                ]
            )

            logger.info(f"Installing Helm release: {release_name}")
            result = subprocess.run(
                helm_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Helm release installed: {result.stdout}")

            # Create Service for llm-d-modelservice if chart doesn't create one
            # The chart may not create a Service, so we create one to target the decode pods
            try:
                from kubernetes import client, config
                from kubernetes.client.rest import ApiException

                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()

                v1 = client.CoreV1Api()
                apps_v1 = client.AppsV1Api()

                # Check if service already exists
                service_name = f"{release_name}-llm-d-modelservice-decode"
                try:
                    v1.read_namespaced_service(
                        name=service_name, namespace=self.namespace
                    )
                    logger.info(f"Service {service_name} already exists")
                except ApiException as e:
                    if e.status == 404:
                        logger.debug(
                            f"Kubernetes API: Service '{service_name}' in namespace '{self.namespace}' not found "
                            f"(status={e.status}, reason={e.reason}). Will create it."
                        )
                        # Service doesn't exist, create it
                        # Get deployment to find selector labels
                        # Try different possible deployment name formats
                        deployment_names = [
                            f"{release_name}-llm-d-modelservice-decode",
                            f"{release_name}-llm-d-m-decode",
                            f"{release_name}-l-decode",  # llm-d-modelservice chart uses -l-decode suffix
                            f"{release_name}-decode",  # Fallback pattern
                        ]
                        deployment = None
                        deployment_name = None
                        for dep_name in deployment_names:
                            try:
                                deployment = apps_v1.read_namespaced_deployment(
                                    name=dep_name, namespace=self.namespace
                                )
                                deployment_name = dep_name
                                logger.debug(
                                    f"Found deployment '{deployment_name}' for release '{release_name}'"
                                )
                                break
                            except ApiException as e:
                                logger.debug(
                                    f"Kubernetes API: Deployment '{dep_name}' in namespace '{self.namespace}' not found "
                                    f"(status={e.status}, reason={e.reason}). Trying next name."
                                )
                                continue

                        if deployment is None:
                            # List all deployments and find one matching the release by label
                            logger.debug(
                                f"Listing all deployments in namespace '{self.namespace}' to find one for release '{release_name}'"
                            )
                            deployments = apps_v1.list_namespaced_deployment(
                                namespace=self.namespace
                            )
                            for dep in deployments.items:
                                labels = dep.metadata.labels or {}
                                instance_label = labels.get(
                                    "app.kubernetes.io/instance"
                                )
                                release_label = labels.get("release")
                                # Match by instance label or release label, or by name prefix
                                if (
                                    instance_label == release_name
                                    or release_label == release_name
                                    or dep.metadata.name.startswith(release_name)
                                ):
                                    # Prefer decode deployment over prefill
                                    if "decode" in dep.metadata.name.lower():
                                        deployment = dep
                                        deployment_name = dep.metadata.name
                                        logger.debug(
                                            f"Found deployment '{deployment_name}' by label/prefix matching"
                                        )
                                        break
                                    elif deployment is None:
                                        # Keep prefill as fallback if no decode found
                                        deployment = dep
                                        deployment_name = dep.metadata.name
                                        logger.debug(
                                            f"Found deployment '{deployment_name}' by label/prefix matching (prefill, keeping as fallback)"
                                        )

                        if deployment is None:
                            logger.error(
                                f"Could not find deployment for release '{release_name}' in namespace '{self.namespace}'. Available deployments: {[d.metadata.name for d in apps_v1.list_namespaced_deployment(namespace=self.namespace).items]}"
                            )
                            raise ApiException(
                                status=404, reason="Deployment not found"
                            )

                        try:
                            # Get selector labels from deployment
                            selector = deployment.spec.selector.match_labels

                            # Get service port from values (default 8000)
                            service_port = 8000
                            target_port = (
                                8200  # vLLM container port for prefix-aware routing
                            )

                            # Create Service with shorter name to avoid DNS label length limits
                            # Kubernetes DNS labels have a 63 character limit
                            # Use a shorter service name
                            short_service_name = (
                                f"{release_name[:40]}-svc"
                                if len(release_name) > 40
                                else f"{release_name}-svc"
                            )
                            # Ensure it doesn't exceed 63 chars total
                            if len(short_service_name) > 63:
                                short_service_name = f"{release_name[:55]}-svc"

                            service = client.V1Service(
                                metadata=client.V1ObjectMeta(
                                    name=short_service_name,
                                    labels={
                                        "app.kubernetes.io/instance": release_name,
                                        "app.kubernetes.io/managed-by": "auto-tune-vllm",
                                    },
                                ),
                                spec=client.V1ServiceSpec(
                                    type="ClusterIP",
                                    selector=selector,
                                    ports=[
                                        client.V1ServicePort(
                                            port=service_port,
                                            target_port=target_port,
                                            protocol="TCP",
                                            name="http",
                                        ),
                                    ],
                                ),
                            )
                            service_name = (
                                short_service_name  # Update service_name for logging
                            )
                            v1.create_namespaced_service(
                                namespace=self.namespace, body=service
                            )
                            logger.info(
                                f"Created Service {service_name} for Helm release {release_name}"
                            )
                        except ApiException as de:
                            logger.warning(
                                f"Kubernetes API error while creating Service '{service_name}' in namespace '{self.namespace}': "
                                f"status={de.status}, reason={de.reason}, message={de.body if hasattr(de, 'body') else str(de)}. "
                                f"Deployment: {deployment_name}"
                            )
                    else:
                        logger.warning(
                            f"Kubernetes API error while checking for Service '{service_name}' in namespace '{self.namespace}': "
                            f"status={e.status}, reason={e.reason}, message={e.body if hasattr(e, 'body') else str(e)}"
                        )
            except Exception as e:
                logger.warning(f"Could not create Service for {release_name}: {e}")

            # Create Helm trial controller and run in executor
            benchmark_image = self.helm_config.get("benchmark_image")
            benchmark_pvc = self.helm_config.get("benchmark_pvc")
            model_pvc = self.helm_config.get("model_pvc")
            controller = HelmTrialController(
                release_name,
                self.namespace,
                benchmark_image,
                self.helm_config,
                benchmark_pvc=benchmark_pvc,
                model_pvc=model_pvc,
            )
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(controller.run_trial, trial_config)

            # Track trial
            self.active_trials[trial_config.trial_id] = (release_name, future, executor)

            return JobHandle(trial_config.trial_id, release_name)

        except subprocess.CalledProcessError as e:
            logger.error(f"Helm install failed: {e.stderr}")
            raise RuntimeError(f"Failed to install Helm release: {e.stderr}")
        finally:
            # Clean up values file
            import os

            if os.path.exists(values_file):
                os.unlink(values_file)

    def poll_trials(
        self, job_handles: List[JobHandle]
    ) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll for completed Helm trials.

        Args:
            job_handles: List of job handles to poll

        Returns:
            Tuple of (completed results, remaining handles)
        """
        if not job_handles:
            return [], []

        completed_results = []
        remaining_handles = []

        for handle in job_handles:
            trial_id = handle.trial_id
            if trial_id not in self.active_trials:
                logger.warning(f"Trial {trial_id} not found in active trials")
                remaining_handles.append(handle)
                continue

            release_name, future, executor = self.active_trials[trial_id]

            # Check if future completed
            if future.done():
                try:
                    result = future.result()
                    completed_results.append(result)
                    logger.info(f"Completed Helm trial {trial_id}")

                    # Clean up Helm release before removing from active trials
                    try:
                        import subprocess

                        uninstall_cmd = [
                            "helm",
                            "uninstall",
                            release_name,
                            "--namespace",
                            self.namespace,
                        ]
                        if self.kubeconfig:
                            uninstall_cmd.extend(["--kubeconfig", self.kubeconfig])

                        result = subprocess.run(
                            uninstall_cmd,
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        if result.returncode == 0:
                            logger.debug(f"Uninstalled Helm release: {release_name}")
                        else:
                            logger.warning(
                                f"Helm uninstall returned non-zero exit code "
                                f"for {release_name}: {result.stderr}"
                            )
                    except Exception as cleanup_e:
                        logger.warning(
                            f"Failed to cleanup Helm release {release_name} "
                            f"for trial {trial_id}: {cleanup_e}"
                        )

                    # Clean up
                    executor.shutdown(wait=False)
                    del self.active_trials[trial_id]

                except Exception as e:
                    logger.error(f"Trial {trial_id} failed: {e}")
                    from ..core.trial import ExecutionInfo, TrialResult

                    execution_info = ExecutionInfo()
                    execution_info.helm_release_name = release_name

                    error_result = TrialResult(
                        trial_id=trial_id,
                        objective_values=[],
                        detailed_metrics={},
                        execution_info=execution_info,
                        success=False,
                        error_message=str(e),
                    )
                    completed_results.append(error_result)

                    # Clean up Helm release even on error
                    try:
                        import subprocess

                        uninstall_cmd = [
                            "helm",
                            "uninstall",
                            release_name,
                            "--namespace",
                            self.namespace,
                        ]
                        if self.kubeconfig:
                            uninstall_cmd.extend(["--kubeconfig", self.kubeconfig])

                        result = subprocess.run(
                            uninstall_cmd,
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        if result.returncode == 0:
                            logger.debug(f"Uninstalled Helm release: {release_name}")
                        else:
                            logger.warning(
                                f"Helm uninstall returned non-zero exit code "
                                f"for {release_name}: {result.stderr}"
                            )
                    except Exception as cleanup_e:
                        logger.warning(
                            f"Failed to cleanup Helm release {release_name} "
                            f"for failed trial {trial_id}: {cleanup_e}"
                        )

                    executor.shutdown(wait=False)
                    del self.active_trials[trial_id]
            else:
                remaining_handles.append(handle)

        return completed_results, remaining_handles

    def cleanup_all_trials(self):
        """Cleanup all active Helm releases and benchmark Jobs.

        This method:
        1. Cleans up releases tracked in active_trials
        2. Also searches for and uninstalls ALL Helm releases matching the study name pattern
           (to catch releases from previous runs or orphaned releases)
        """
        # DEBUG DISABLED: #region agent log
        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1263","message":"Starting cleanup_all_trials","data":{"active_trials_count":len(self.active_trials),"active_trials_keys":list(self.active_trials.keys()),"study_name":self.study_name},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion

        # Step 1: Clean up tracked active trials
        if self.active_trials:
            logger.info(
                f"Cleaning up {len(self.active_trials)} tracked active trial(s)"
            )
            # DEBUG DISABLED: #region agent log
            # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
            # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1275","message":"Cleaning up tracked active trials","data":{"count":len(self.active_trials)},"timestamp":int(time.time()*1000)})+"\n")
            # DEBUG DISABLED: #endregion

            for trial_id, (release_name, future, executor) in list(
                self.active_trials.items()
            ):
                try:
                    # DEBUG DISABLED: #region agent log
                    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1281","message":"Processing trial cleanup","data":{"trial_id":trial_id,"release_name":release_name},"timestamp":int(time.time()*1000)})+"\n")
                    # DEBUG DISABLED: #endregion

                    # Shutdown executor
                    if executor:
                        executor.shutdown(wait=False)
                        # DEBUG DISABLED: #region agent log
                        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1287","message":"Executor shutdown","data":{"trial_id":trial_id},"timestamp":int(time.time()*1000)})+"\n")
                        # DEBUG DISABLED: #endregion

                    # Uninstall Helm release
                    logger.info(f"Uninstalling Helm release: {release_name}")
                    # DEBUG DISABLED: #region agent log
                    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1293","message":"Uninstalling Helm release","data":{"release_name":release_name,"namespace":self.namespace},"timestamp":int(time.time()*1000)})+"\n")
                    # DEBUG DISABLED: #endregion

                    result = subprocess.run(
                        [
                            "helm",
                            "uninstall",
                            release_name,
                            "--namespace",
                            self.namespace,
                        ],
                        check=False,  # Don't fail if already deleted
                        capture_output=True,
                        text=True,
                    )

                    # DEBUG DISABLED: #region agent log
                    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1301","message":"Helm uninstall result","data":{"release_name":release_name,"returncode":result.returncode,"stdout":result.stdout[:200] if result.stdout else "","stderr":result.stderr[:200] if result.stderr else ""},"timestamp":int(time.time()*1000)})+"\n")
                    # DEBUG DISABLED: #endregion

                    if result.returncode != 0:
                        logger.warning(
                            f"Helm uninstall for {release_name} returned code {result.returncode}: "
                            f"{result.stderr or result.stdout}"
                        )
                    else:
                        logger.info(
                            f"Successfully uninstalled Helm release: {release_name}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error cleaning up trial {trial_id}: {e}", exc_info=True
                    )
                    # DEBUG DISABLED: #region agent log
                    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1313","message":"Error during cleanup","data":{"trial_id":trial_id,"error":str(e)},"timestamp":int(time.time()*1000)})+"\n")
                    # DEBUG DISABLED: #endregion

            self.active_trials.clear()

        # Step 2: Search for and uninstall ALL Helm releases matching study name pattern
        # This catches orphaned releases from previous runs
        if self.study_name:
            logger.info(
                f"Searching for orphaned Helm releases matching study: {self.study_name}"
            )
            # DEBUG DISABLED: #region agent log
            # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
            # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1324","message":"Searching for orphaned releases","data":{"study_name":self.study_name},"timestamp":int(time.time()*1000)})+"\n")
            # DEBUG DISABLED: #endregion

            try:
                # List all Helm releases in namespace
                result = subprocess.run(
                    ["helm", "list", "-n", self.namespace, "-q"],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode == 0 and result.stdout:
                    releases = [
                        r.strip()
                        for r in result.stdout.strip().split("\n")
                        if r.strip()
                    ]
                    # Find releases that start with study name
                    study_releases = [
                        r for r in releases if r.startswith(self.study_name)
                    ]

                    # DEBUG DISABLED: #region agent log
                    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1338","message":"Found matching releases","data":{"total_releases":len(releases),"study_releases":study_releases},"timestamp":int(time.time()*1000)})+"\n")
                    # DEBUG DISABLED: #endregion

                    if study_releases:
                        logger.info(
                            f"Found {len(study_releases)} orphaned Helm release(s) matching study name"
                        )
                        for release in study_releases:
                            try:
                                logger.info(
                                    f"Uninstalling orphaned Helm release: {release}"
                                )
                                # DEBUG DISABLED: #region agent log
                                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1346","message":"Uninstalling orphaned release","data":{"release_name":release},"timestamp":int(time.time()*1000)})+"\n")
                                # DEBUG DISABLED: #endregion

                                uninstall_result = subprocess.run(
                                    [
                                        "helm",
                                        "uninstall",
                                        release,
                                        "--namespace",
                                        self.namespace,
                                    ],
                                    check=False,
                                    capture_output=True,
                                    text=True,
                                )

                                # DEBUG DISABLED: #region agent log
                                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1354","message":"Orphaned release uninstall result","data":{"release_name":release,"returncode":uninstall_result.returncode,"stdout":uninstall_result.stdout[:200] if uninstall_result.stdout else "","stderr":uninstall_result.stderr[:200] if uninstall_result.stderr else ""},"timestamp":int(time.time()*1000)})+"\n")
                                # DEBUG DISABLED: #endregion

                                if uninstall_result.returncode == 0:
                                    logger.info(
                                        f"Successfully uninstalled orphaned Helm release: {release}"
                                    )
                                else:
                                    logger.warning(
                                        f"Failed to uninstall orphaned release {release}: "
                                        f"{uninstall_result.stderr or uninstall_result.stdout}"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error uninstalling orphaned release {release}: {e}",
                                    exc_info=True,
                                )
                                # DEBUG DISABLED: #region agent log
                                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1366","message":"Error uninstalling orphaned release","data":{"release_name":release,"error":str(e)},"timestamp":int(time.time()*1000)})+"\n")
                                # DEBUG DISABLED: #endregion
                    else:
                        logger.debug(
                            f"No orphaned Helm releases found matching study: {self.study_name}"
                        )
                else:
                    logger.debug(
                        f"Could not list Helm releases: {result.stderr or result.stdout}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error searching for orphaned Helm releases: {e}", exc_info=True
                )
                # DEBUG DISABLED: #region agent log
                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1375","message":"Error searching for orphaned releases","data":{"error":str(e)},"timestamp":int(time.time()*1000)})+"\n")
                # DEBUG DISABLED: #endregion

        logger.info("Completed cleanup of all active trials and orphaned releases")
        # DEBUG DISABLED: #region agent log
        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"cleanup-all-trials","hypothesisId":"CLEANUP","location":"backends.py:1380","message":"Cleanup completed","data":{},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion

    def shutdown(self):
        """Shutdown Helm execution backend."""
        self.cleanup_all_trials()
        logger.info("Shutdown Helm execution backend")


class KubernetesExecutionBackend(ExecutionBackend):
    """Kubernetes Deployment/Pod-based execution backend (no Helm)."""

    def __init__(
        self,
        k8s_config: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        kubeconfig: Optional[str] = None,
        study_name: Optional[str] = None,
    ):
        """Initialize Kubernetes execution backend.

        Args:
            k8s_config: Kubernetes configuration dictionary
            namespace: Kubernetes namespace
            kubeconfig: Path to kubeconfig file
            study_name: Study name/prefix for cleanup purposes
        """
        self.k8s_config = k8s_config or {}
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.study_name = study_name

        # Track active trials: trial_id -> (deployment_name, service_name, future, executor)
        self.active_trials: Dict[str, Tuple[str, str, Any, Any]] = {}

        # Validate Kubernetes access
        self._validate_kubernetes_access()

        # Ensure namespace exists
        self._ensure_namespace_exists()

        logger.info(
            f"Initialized Kubernetes execution backend (namespace: {self.namespace})"
        )

    def _validate_kubernetes_access(self):
        """Validate Kubernetes cluster access."""
        try:
            from kubernetes import client, config
        except ImportError:
            raise RuntimeError(
                "kubernetes library not available. "
                "Install with: pip install 'auto-tune-vllm[helm]'"
            )

        try:
            if self.kubeconfig:
                config.load_kube_config(config_file=self.kubeconfig)
            else:
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()

            # Test API access
            v1 = client.CoreV1Api()
            v1.list_namespace()
            logger.info("Kubernetes cluster access validated")
        except Exception as e:
            raise RuntimeError(f"Failed to access Kubernetes cluster: {e}")

    def _ensure_namespace_exists(self):
        """Ensure Kubernetes namespace exists, create if it doesn't."""
        try:
            from kubernetes import client, config
        except ImportError:
            return

        try:
            if self.kubeconfig:
                config.load_kube_config(config_file=self.kubeconfig)
            else:
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()

            v1 = client.CoreV1Api()

            # Check if namespace exists
            try:
                v1.read_namespace(name=self.namespace)
                logger.debug(f"Namespace {self.namespace} already exists")
            except Exception:
                # Namespace doesn't exist, create it
                logger.info(f"Creating namespace: {self.namespace}")
                namespace_body = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=self.namespace)
                )
                v1.create_namespace(body=namespace_body)
                logger.info(f"Created namespace: {self.namespace}")
        except Exception as e:
            logger.warning(f"Failed to ensure namespace exists: {e}")

    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit trial for Kubernetes-based execution.

        Args:
            trial_config: Trial configuration

        Returns:
            JobHandle with deployment name as backend_job_id
        """
        import concurrent.futures

        from .k8s_utils import (
            create_vllm_deployment,
            create_vllm_service,
            sanitize_k8s_name,
        )
        from .trial_controller import KubernetesTrialController

        # Generate deployment and service names
        base_name = sanitize_k8s_name(
            f"{trial_config.study_name}-{trial_config.trial_id}"
        )
        deployment_name = f"{base_name}-vllm"
        service_name = f"{base_name}-svc"

        # Get vLLM image
        vllm_image = self.k8s_config.get("vllm_image")
        if not vllm_image:
            # Default vLLM image - could be made configurable
            vllm_image = "vllm/vllm-openai:latest"
            logger.warning(f"vLLM image not specified, using default: {vllm_image}")

        # Create vLLM Deployment
        create_vllm_deployment(
            trial_config=trial_config,
            deployment_name=deployment_name,
            namespace=self.namespace,
            vllm_image=vllm_image,
            resource_requests=self.k8s_config.get("resource_requests"),
            resource_limits=self.k8s_config.get("resource_limits"),
            kubeconfig=self.kubeconfig,
            model_pvc=self.k8s_config.get("model_pvc"),
        )

        # Create vLLM Service
        service_type = self.k8s_config.get("service_type", "ClusterIP")
        service_port = self.k8s_config.get("service_port", 8000)
        create_vllm_service(
            trial_config=trial_config,
            service_name=service_name,
            deployment_name=deployment_name,
            namespace=self.namespace,
            service_type=service_type,
            service_port=service_port,
            kubeconfig=self.kubeconfig,
        )

        # Create trial controller
        benchmark_image = self.k8s_config.get("benchmark_image")
        benchmark_pvc = self.k8s_config.get("benchmark_pvc")
        model_pvc = self.k8s_config.get("model_pvc")
        controller = KubernetesTrialController(
            deployment_name=deployment_name,
            service_name=service_name,
            namespace=self.namespace,
            benchmark_image=benchmark_image,
            service_type=service_type,
            service_port=service_port,
            kubeconfig=self.kubeconfig,
            benchmark_pvc=benchmark_pvc,
            model_pvc=model_pvc,
        )

        # Submit trial execution in thread pool
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(controller.run_trial, trial_config)

        # Track trial
        self.active_trials[trial_config.trial_id] = (
            deployment_name,
            service_name,
            future,
            executor,
        )

        logger.info(
            f"Submitted trial {trial_config.trial_id} "
            f"(Deployment: {deployment_name}, Service: {service_name})"
        )

        return JobHandle(
            trial_id=trial_config.trial_id,
            backend_job_id=deployment_name,
            status="running",
        )

    def poll_trials(
        self, job_handles: List[JobHandle]
    ) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll for completed trials.

        Args:
            job_handles: List of job handles to poll

        Returns:
            Tuple of (completed results, remaining handles)
        """
        completed_results = []
        remaining_handles = []

        for handle in job_handles:
            trial_id = handle.trial_id
            if trial_id not in self.active_trials:
                # Trial not found - mark as failed
                logger.warning(f"Trial {trial_id} not found in active trials")
                completed_results.append(
                    TrialResult(
                        trial_id=trial_id,
                        trial_number=0,
                        trial_type="unknown",
                        objective_values=[],
                        detailed_metrics={},
                        execution_info=None,
                        success=False,
                        error_message="Trial not found in active trials",
                    )
                )
                continue

            deployment_name, service_name, future, executor = self.active_trials[
                trial_id
            ]

            # Check if future is done
            if future.done():
                try:
                    result = future.result(timeout=1)
                    completed_results.append(result)
                    # Cleanup Kubernetes resources before removing from active trials
                    try:
                        from .k8s_utils import delete_vllm_resources

                        delete_vllm_resources(
                            deployment_name=deployment_name,
                            service_name=service_name,
                            namespace=self.namespace,
                            kubeconfig=self.kubeconfig,
                        )
                        logger.debug(f"Cleaned up resources for trial {trial_id}")
                    except Exception as cleanup_e:
                        logger.warning(
                            f"Failed to cleanup resources for trial {trial_id}: {cleanup_e}"
                        )
                    # Remove from active trials
                    del self.active_trials[trial_id]
                except Exception as e:
                    logger.error(f"Error getting result for trial {trial_id}: {e}")
                    completed_results.append(
                        TrialResult(
                            trial_id=trial_id,
                            trial_number=0,
                            trial_type="unknown",
                            objective_values=[],
                            detailed_metrics={},
                            execution_info=None,
                            success=False,
                            error_message=str(e),
                        )
                    )
                    # Cleanup Kubernetes resources even on error
                    try:
                        from .k8s_utils import delete_vllm_resources

                        delete_vllm_resources(
                            deployment_name=deployment_name,
                            service_name=service_name,
                            namespace=self.namespace,
                            kubeconfig=self.kubeconfig,
                        )
                        logger.debug(
                            f"Cleaned up resources for failed trial {trial_id}"
                        )
                    except Exception as cleanup_e:
                        logger.warning(
                            f"Failed to cleanup resources for failed trial {trial_id}: {cleanup_e}"
                        )
                    del self.active_trials[trial_id]
            else:
                remaining_handles.append(handle)

        return completed_results, remaining_handles

    def cleanup_all_trials(self):
        """Force cleanup of all active trials and their resources."""
        from .k8s_utils import delete_vllm_resources

        logger.info(f"Cleaning up {len(self.active_trials)} active Kubernetes trials")

        for trial_id, (deployment_name, service_name, future, executor) in list(
            self.active_trials.items()
        ):
            try:
                # Cancel future if still running
                if not future.done():
                    future.cancel()

                # Delete Kubernetes resources
                delete_vllm_resources(
                    deployment_name=deployment_name,
                    service_name=service_name,
                    namespace=self.namespace,
                    kubeconfig=self.kubeconfig,
                )

                # Shutdown executor
                executor.shutdown(wait=False)

                logger.info(f"Cleaned up trial {trial_id}")
            except Exception as e:
                logger.error(f"Error cleaning up trial {trial_id}: {e}")

        self.active_trials.clear()
        logger.info("Completed cleanup of all active Kubernetes trials")

    def shutdown(self):
        """Shutdown Kubernetes execution backend."""
        self.cleanup_all_trials()
        logger.info("Shutdown Kubernetes execution backend")

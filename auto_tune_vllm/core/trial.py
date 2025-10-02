"""Trial data structures and execution info."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import BenchmarkConfig to avoid circular imports
from ..benchmarks.config import BenchmarkConfig


def calculate_gpu_requirements(parameters: Dict[str, Any]) -> int:
    """Calculate GPU requirements based on parallelism parameters.

    Args:
        parameters: Trial parameters dictionary

    Returns:
        Number of GPUs required (product of parallelism settings)
    """

    # Extract parallelism parameters with defaults and safe coercion
    def _as_pos_int(v, default=1):
        try:
            i = int(v)
        except (TypeError, ValueError):
            return default
        return default if i < 1 else i

    tensor_parallel = _as_pos_int(parameters.get("tensor_parallel_size"), 1)
    pipeline_parallel = _as_pos_int(parameters.get("pipeline_parallel_size"), 1)
    data_parallel = _as_pos_int(parameters.get("data_parallel_size"), 1)

    # Calculate total GPU requirement
    total_gpus = tensor_parallel * pipeline_parallel * data_parallel

    return max(1, total_gpus)  # Ensure at least 1 GPU


@dataclass
class TrialConfig:
    """Configuration for a single optimization trial."""

    study_name: str
    trial_id: str  # "trial_123" | "baseline_concurrency_50" | "probe_warmup_1"
    trial_number: Optional[int] = None  # Only for Optuna trials (for study.tell())
    trial_type: str = "optimization"  # "baseline" | "optimization" | "probe"
    parameters: Dict[str, Any] = field(
        default_factory=dict
    )  # vLLM parameters from Optuna
    parameter_configs: Optional[Dict[str, Any]] = (
        None  # Parameter configuration metadata for determining env vars
    )
    static_environment_variables: Dict[str, str] = field(
        default_factory=dict
    )  # Static environment variables
    benchmark_config: BenchmarkConfig = None
    optimization_config: Optional[Any] = None  # Optimization configuration from study
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    logging_config: Optional[Dict[str, Any]] = None  # Logging configuration from study
    vllm_startup_timeout: int = 300  # vLLM server startup timeout in seconds

    def __post_init__(self):
        """Calculate resource requirements after initialization."""
        if not self.resource_requirements:
            # Calculate GPU requirements based on parallelism parameters
            num_gpus = calculate_gpu_requirements(self.parameters)
            self.resource_requirements = {
                "num_gpus": num_gpus,
                "num_cpus": 4,  # Default CPU count, could be made configurable
            }

    @property
    def vllm_args(self) -> List[str]:
        """Convert non-environment parameters to vLLM command-line arguments."""
        args = []
        for param_name, value in self.parameters.items():
            # Skip environment variables - they're handled separately
            if self._is_environment_parameter(param_name):
                continue

            # Convert underscore to dash for CLI
            cli_param = param_name.replace("_", "-")

            if isinstance(value, bool):
                if value:
                    args.append(f"--{cli_param}")
            else:
                args.extend([f"--{cli_param}", str(value)])

        return args

    @property
    def environment_vars(self) -> Dict[str, str]:
        """Get all environment variables as string dictionary (optimizable + static)."""
        env_vars = {}

        # Add static environment variables first
        env_vars.update(self.static_environment_variables)

        # Add optimizable environment variables from parameters
        # (can override static ones)
        for param_name, value in self.parameters.items():
            if self._is_environment_parameter(param_name):
                env_vars[param_name] = str(value)

        return env_vars

    def _is_environment_parameter(self, param_name: str) -> bool:
        """Check if a parameter is an environment variable."""
        if self.parameter_configs:
            # Use parameter configuration metadata to identify environment parameters
            from .config import EnvironmentParameter

            param_config = self.parameter_configs.get(param_name)
            return param_config and isinstance(param_config, EnvironmentParameter)
        else:
            # Fallback: use heuristic for common environment variable names
            env_var_names = {
                "VLLM_ATTENTION_BACKEND",
                "CUDA_VISIBLE_DEVICES",
                "VLLM_CACHE_ROOT",
                "VLLM_CONFIGURE_LOGGING",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "TOKENIZERS_PARALLELISM",
                "VLLM_ENGINE_ITERATION_TIMEOUT_S",
                "VLLM_API_SERVER_CHAT_TEMPLATE",
            }
            return param_name in env_var_names


@dataclass
class ExecutionInfo:
    """Information about trial execution."""

    worker_node_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None

    def mark_completed(self):
        """Mark execution as completed."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time


@dataclass
class TrialResult:
    """Results from a completed trial."""

    trial_id: str  # Replace trial_number with composite identifier
    trial_number: Optional[int] = None  # Keep for Optuna compatibility
    trial_type: str = "optimization"
    objective_values: List[float] = field(
        default_factory=list
    )  # For Optuna (throughput, latency, etc.)
    detailed_metrics: Dict[str, Any] = field(
        default_factory=dict
    )  # Rich percentile data from benchmarks
    execution_info: ExecutionInfo = None
    success: bool = True
    error_message: Optional[str] = None

    @property
    def primary_objective(self) -> float:
        """Get primary objective value for single-objective optimization."""
        return self.objective_values[0] if self.objective_values else 0.0

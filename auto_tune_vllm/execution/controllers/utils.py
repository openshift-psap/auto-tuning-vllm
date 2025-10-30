import logging
import shutil

import ray

from ...core.trial import TrialConfig

logger = logging.getLogger(__name__)

# Error classification patterns for database storage
# Maps error types to keyword patterns used by _classify_error()
ERROR_PATTERNS: dict[str, list[str]] = {
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

def classify_error(exception: Exception) -> str:
    """Classify error type based on exception message.

    Uses ERROR_PATTERNS dictionary to categorize exceptions for
    structured failure analysis in the database.

    Returns:
        Error type string (e.g., "OOM", "Timeout") or "Unknown"
    """
    error_message = str(exception).lower()

    for error_type, patterns in ERROR_PATTERNS.items():
        if any(pattern in error_message for pattern in patterns):
            return error_type

    return "Unknown"

def validate_environment(trial_config: TrialConfig) -> bool:
    """Validate that all required packages are available on this worker."""

    required_packages = {
        "vllm": "vLLM serving framework",
        "guidellm": "GuideLLM benchmarking tool",
        "optuna": "Optuna optimization framework",
        "ray": "Ray distributed computing",
    }

    # Only require psycopg2 if using PostgreSQL
    using_postgresql = False
    if trial_config.logging_config:
        using_postgresql = trial_config.logging_config.get("database_url") is not None

    if using_postgresql:
        required_packages["psycopg2"] = "PostgreSQL client"

    missing_packages: list[str] = []
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

    required_commands = {
        "python3": "Python interpreter",
        "guidellm": "GuideLLM CLI tool",
        "vllm": "vLLM CLI command",
    }

    missing_commands: list[str] = []
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
                accelerator_types: list[str] = [
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
    logger.info("Environment validation passed on Ray worker")

    return True

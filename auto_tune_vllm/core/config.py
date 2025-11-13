"""Configuration management with validation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from auto_tune_vllm.core.constraint import Constraint

from ..benchmarks.config import BenchmarkConfig
from .parameters import (
    BooleanParameter,
    EnvironmentParameter,
    ListParameter,
    ParameterConfig,
    RangeParameter,
)


@dataclass
class ObjectiveConfig:
    """Configuration for a single optimization objective."""

    metric: str  # "output_tokens_per_second", "request_latency", etc.
    direction: str  # "maximize" or "minimize"
    percentile: str = "median"  # "median", "p50", "p95", "p90", "p99", "mean"

    def __post_init__(self):
        """Validate objective configuration."""
        valid_metrics = {
            "output_tokens_per_second",
            "request_latency",
            "time_to_first_token_ms",
            "inter_token_latency_ms",
            "requests_per_second",
        }
        valid_directions = {"maximize", "minimize"}
        valid_percentiles = {"median", "p50", "p95", "p90", "p99", "mean"}

        if self.metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{self.metric}'. Valid options: {valid_metrics}"
            )
        if self.direction not in valid_directions:
            raise ValueError(
                f"Invalid direction '{self.direction}'. "
                f"Valid options: {valid_directions}"
            )
        if self.percentile not in valid_percentiles:
            raise ValueError(
                f"Invalid percentile '{self.percentile}'. "
                f"Valid options: {valid_percentiles}"
            )


@dataclass
class OptimizationConfig:
    """Optimization configuration with support for new structured format.

    Includes backward compatibility.
    """

    # Backward compatibility fields
    objective: Union[str, List[str]] = None  # Old format: "maximize", "minimize", list
    sampler: str = "tpe"  # "tpe", "random", "gp", "botorch", "nsga2", "grid"
    n_trials: int = 100
    n_startup_trials: int = 10  # Number of random startup trials
    max_concurrent_trials: Optional[int] = (
        None  # Maximum concurrent trials (required for resource management)
    )

    # New structured format fields
    approach: Optional[str] = None  # "single_objective" or "multi_objective"
    objectives: Optional[List[ObjectiveConfig]] = None  # For multi-objective
    preset: Optional[str] = None  # "high_throughput", "low_latency", "balanced"

    def __post_init__(self):
        """Process and validate optimization configuration."""
        # Handle preset configurations
        if self.preset:
            self._apply_preset()
            return

        # Handle new structured format
        if self.approach:
            self._validate_structured_format()
            return

        # Handle backward compatibility (old format)
        if self.objective:
            self._convert_old_format()
            return

        # Default fallback
        self._apply_default_config()

    def _apply_preset(self):
        """Apply preset optimization configurations."""
        if self.preset == "high_throughput":
            self.approach = "single_objective"
            self.objectives = [
                ObjectiveConfig(
                    metric="output_tokens_per_second",
                    direction="maximize",
                    percentile="mean",
                )
            ]
        elif self.preset == "low_latency":
            self.approach = "single_objective"
            self.objectives = [
                ObjectiveConfig(
                    metric="request_latency", direction="minimize", percentile="p95"
                )
            ]
        elif self.preset == "balanced":
            self.approach = "multi_objective"
            self.objectives = [
                ObjectiveConfig(
                    metric="output_tokens_per_second",
                    direction="maximize",
                    percentile="mean",
                ),
                ObjectiveConfig(
                    metric="request_latency", direction="minimize", percentile="median"
                ),
            ]
        else:
            raise ValueError(
                f"Unknown preset '{self.preset}'. "
                f"Valid options: high_throughput, low_latency, balanced"
            )

    def _validate_structured_format(self):
        """Validate new structured format."""
        if self.approach not in ["single_objective", "multi_objective"]:
            raise ValueError(
                f"Invalid approach '{self.approach}'. "
                f"Valid options: single_objective, multi_objective"
            )

        if not self.objectives:
            raise ValueError("Objectives must be specified for structured format")

        if self.approach == "single_objective" and len(self.objectives) != 1:
            raise ValueError(
                "Single objective optimization requires exactly one objective"
            )

        if self.approach == "multi_objective" and len(self.objectives) < 2:
            raise ValueError(
                "Multi-objective optimization requires at least two objectives"
            )

    def _convert_old_format(self):
        """Convert old format to new structured format for backward compatibility."""
        if isinstance(self.objective, str):
            # Single objective
            self.approach = "single_objective"
            if self.objective == "maximize":
                # Default to maximizing throughput
                self.objectives = [
                    ObjectiveConfig(
                        metric="output_tokens_per_second",
                        direction="maximize",
                        percentile="median",
                    )
                ]
            elif self.objective == "minimize":
                # Default to minimizing latency
                self.objectives = [
                    ObjectiveConfig(
                        metric="request_latency",
                        direction="minimize",
                        percentile="median",
                    )
                ]
            else:
                raise ValueError(
                    f"Invalid objective '{self.objective}'. "
                    f"Use 'maximize' or 'minimize'"
                )
        elif isinstance(self.objective, list):
            # Multi-objective (legacy format)
            self.approach = "multi_objective"
            # Default to throughput vs latency
            self.objectives = [
                ObjectiveConfig(
                    metric="output_tokens_per_second",
                    direction="maximize",
                    percentile="median",
                ),
                ObjectiveConfig(
                    metric="request_latency", direction="minimize", percentile="median"
                ),
            ]

    def _apply_default_config(self):
        """Apply default configuration when none is specified."""
        self.approach = "single_objective"
        self.objectives = [
            ObjectiveConfig(
                metric="output_tokens_per_second",
                direction="maximize",
                percentile="mean",
            )
        ]

    @property
    def is_multi_objective(self) -> bool:
        """Check if this is multi-objective optimization."""
        return self.approach == "multi_objective"

    @property
    def optuna_directions(self) -> List[str]:
        """Get Optuna directions for study creation."""
        if self.objectives is not None:
            return [obj.direction for obj in self.objectives]
        else:
            return []

    def get_metric_key(self, objective_index: int = 0) -> str:
        """Get the metric key for extracting values from benchmark results."""
        assert self.objectives is not None
        if objective_index >= len(self.objectives):
            raise IndexError(f"Objective index {objective_index} out of range")

        obj = self.objectives[objective_index]
        if obj.percentile == "median":
            return obj.metric
        else:
            return f"{obj.metric}_{obj.percentile}"


@dataclass
class BaselineConfig:
    """Configuration for baseline trials."""

    enabled: bool = True
    concurrency_levels: List[int] = field(
        default_factory=lambda: [50]
    )  # Concurrency levels to test
    parameters: Dict[str, Any] = field(
        default_factory=dict
    )  # Custom parameters for baseline trials

    def __post_init__(self):
        """Validate baseline configuration."""
        if self.enabled and not self.concurrency_levels:
            raise ValueError(
                "Baseline configuration requires at least one concurrency level"
            )

        if self.enabled:
            for concurrency in self.concurrency_levels:
                if not isinstance(concurrency, int) or concurrency <= 0:
                    raise ValueError(
                        f"Invalid concurrency level: {concurrency}. "
                        + "Must be positive integer"
                    )


@dataclass
class StudyConfig:
    """Complete study configuration."""

    study_name: str
    database_url: Optional[str]
    optimization: OptimizationConfig
    benchmark: BenchmarkConfig
    parameters: Dict[str, ParameterConfig] = field(default_factory=dict)
    static_environment_variables: Dict[str, str] = field(
        default_factory=dict
    )  # Static environment variables
    static_parameters: Dict[str, Any] = field(
        default_factory=dict
    )  # Static vLLM parameters for all trials
    baseline: Optional[BaselineConfig] = None  # Baseline configuration
    logging_config: Optional[Dict[str, Any]] = None
    storage_file: Optional[str] = (
        None  # Alternative to database_url for file-based storage
    )
    study_prefix: Optional[str] = (
        None  # For auto-generated study names with custom prefix
    )
    use_explicit_name: bool = (
        False  # Flag to indicate explicit name usage (affects load_if_exists behavior)
    )
    constraints: list[Constraint] = field(default_factory=list)

    @classmethod
    def from_file(
        cls,
        config_path: str,
        vllm_version: str | None = None,
    ) -> StudyConfig:
        """Load and validate configuration from YAML file."""
        config_validator = ConfigValidator(vllm_version)
        return config_validator.load_and_validate(config_path)


class ConfigValidator:
    """Validates study configurations against parameter"""

    def __init__(
        self,
        vllm_version: str | None = None,
    ):
        self.vllm_version: str = vllm_version if vllm_version else "No Version Set"

    def load_and_validate(self, config_path: str) -> StudyConfig:
        """Load and validate study configuration with environment variable expansion."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Read yaml contents
        with open(config_path) as f:
            raw_config = f.read()

        expanded_config = self.expand_environment_variables(raw_config)
        raw_config = yaml.safe_load(expanded_config)

        return self._validate_config(raw_config)

    def expand_environment_variables(self, yaml_content: str) -> str:
        """
        Expand environment variables in YAML content.

        Supports patterns:
        - ${VAR_NAME} - expands to environment variable value
          or empty string if not set
        - ${VAR_NAME:-default_value} - expands to environment variable value
          or default_value if not set

        Args:
            yaml_content: Raw YAML content as string

        Returns:
            YAML content with environment variables expanded

        Examples:
            ${POSTGRES_PASSWORD} -> value of POSTGRES_PASSWORD env var
            ${LOG_LEVEL:-INFO} -> value of LOG_LEVEL env var or "INFO" if not set
        """

        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""

            env_value = os.getenv(var_name)
            if env_value is None:
                if default_value:
                    return default_value
                else:
                    # Log warning for missing required env vars without defaults
                    print(
                        f"Warning: Environment variable '{var_name}' not found, "
                        f"using empty string"
                    )
                    return ""
            return env_value

        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*?)(?::-(.*?))?\}"
        expanded_content = re.sub(pattern, replace_env_var, yaml_content)

        return expanded_content

    def _generate_unique_study_name(self, prefix: Optional[str] = None) -> str:
        """Generate a unique study name in the format {prefix}_{N} or study_N."""
        # Use timestamp seconds as a simple unique number
        import time

        timestamp_seconds = int(time.time())
        # Take last 6 digits to keep numbers reasonable
        study_number = timestamp_seconds % 1000000

        if prefix:
            return f"{prefix}_{study_number}"
        else:
            return f"study_{study_number}"

    def _handle_study_naming(
        self, study_info: Dict[str, Any]
    ) -> tuple[str, Optional[str], bool]:
        """
        Handle study naming logic with prefix support.

        Returns:
            tuple: (study_name, study_prefix, use_explicit_name)

        Rules:
        1. If just 'name' is provided: Use exact name, fail if exists
           (use_explicit_name=True)
        2. If just 'prefix' is provided: Auto-generate with prefix
           (use_explicit_name=False)
        3. If both provided: Validation error
        4. If neither provided: Auto-generate with default prefix
           (use_explicit_name=False)
        """
        name = study_info.get("name")
        prefix = study_info.get("prefix")

        # Scenario 3: Both name and prefix provided - ERROR
        if name and prefix:
            raise ValueError(
                "Cannot specify both 'name' and 'prefix' in study configuration. "
                "Use 'name' for exact study names that must be unique, "
                "or 'prefix' for auto-generated names."
            )

        # Scenario 1: Just name provided - use exact name, fail if exists
        if name and not prefix:
            print(
                f"Using explicit study name: {name} (will fail if study already exists)"
            )
            return name, None, True

        # Scenario 2: Just prefix provided - auto-generate with prefix
        if prefix and not name:
            auto_name = self._generate_unique_study_name(prefix)
            print(f"Generated study name: {auto_name} from prefix: {prefix}")
            return auto_name, prefix, False

        # Scenario 4: Neither provided - auto-generate with default prefix
        auto_name = self._generate_unique_study_name()
        print(f"Auto-generated study name: {auto_name}")
        return auto_name, None, False

    def _validate_config(self, raw_config: dict[str, dict[str, Any]]) -> StudyConfig:
        # Validate and build parameter configs
        validated_params = {}

        for param_name, param_config in raw_config.get("parameters", {}).items():
            if not param_config.get("enabled", True):
                continue

            # Check if this is an environment variable
            is_env_var = param_config.get("env_var", False)

            if is_env_var:
                # This is an environment variable parameter
                if (
                    "range" in param_config
                    or "min" in param_config
                    or "max" in param_config
                    or "step" in param_config
                ):
                    raise ValueError(
                        f"Environment parameter '{param_name}' cannot use "
                        f"range configurations. Only list options are allowed."
                    )

                if "options" not in param_config:
                    raise ValueError(
                        f"Environment parameter '{param_name}' "
                        f"must specify options as a list"
                    )

                validated_param = EnvironmentParameter(
                    name=param_name,
                    enabled=param_config.get("enabled", True),
                    options=param_config["options"],
                    data_type=param_config.get("data_type", "str"),
                    description=param_config.get(
                        "description", f"Environment variable {param_name}"
                    ),
                )
            else:
                # Build parameter config based on schema type
                validated_param = self._build_parameter_config(param_name, param_config)

            validated_params[param_name] = validated_param

        # Validate static environment variables (simple key-value pairs)
        static_env_vars = {}
        for env_name, env_value in raw_config.get(
            "static_environment_variables", {}
        ).items():
            if not isinstance(env_value, (str, int, float, bool)):
                raise ValueError(
                    f"Static environment variable '{env_name}' must be a simple "
                    f"value (string, number, or boolean), got {type(env_value)}"
                )
            static_env_vars[env_name] = str(env_value)

        # Validate static parameters (simple key-value pairs for vLLM CLI args)
        static_params = {}

        raw_static_parameters = raw_config.get("static_parameters")
        if raw_static_parameters is None:
            raw_static_parameters = {}
        elif not isinstance(raw_static_parameters, dict):
            raise TypeError(
                "Static parameters must be provided as a mapping of CLI flag "
                "names to simple values."
            )

        for param_name, param_value in raw_static_parameters.items():
            if not isinstance(param_value, (str, int, float, bool)):
                raise ValueError(
                    f"Static parameter '{param_name}' must be a simple value "
                    f"(string, number, or boolean), got {type(param_value)}"
                )

            # Keep the original type (don't convert to string like env vars)
            static_params[param_name] = param_value
        # Build other configs
        study_info = raw_config.get("study", None)
        if study_info is None:
            study_info = {}
        # Handle study naming logic with prefix support
        study_name, study_prefix, use_explicit_name = self._handle_study_naming(
            study_info
        )

        # Handle optimization config with validation
        opt_config_data = raw_config["optimization"]

        # Convert objective config if using new structured format
        if "objective" in opt_config_data and isinstance(
            opt_config_data["objective"], dict
        ):
            # Single objective structured format
            obj_data = opt_config_data["objective"]
            opt_config_data["objectives"] = [ObjectiveConfig(**obj_data)]
            del opt_config_data["objective"]
        elif "objectives" in opt_config_data:
            # Multi-objective structured format
            objectives_data = opt_config_data["objectives"]
            opt_config_data["objectives"] = [
                ObjectiveConfig(**obj) for obj in objectives_data
            ]

        optimization = OptimizationConfig(**opt_config_data)
        benchmark = BenchmarkConfig(**raw_config["benchmark"])

        # Handle optional database_url and storage_file
        database_url = study_info.get("database_url")
        storage_file = study_info.get("storage_file")

        # Validate storage configuration
        if not database_url and not storage_file:
            # Default to file-based storage using study name
            storage_file = f"./optuna_studies/{study_name}/study.db"

        if database_url and storage_file:
            raise ValueError(
                "Cannot specify both database_url and storage_file. "
                "Choose one storage method."
            )

        # Handle baseline configuration
        baseline_config = None
        if "baseline" in raw_config:
            baseline_data = raw_config["baseline"]
            # If concurrency_levels not specified, inherit from benchmark rate
            if "concurrency_levels" not in baseline_data:
                baseline_data["concurrency_levels"] = [benchmark.rate]
            if baseline_data.get("enabled", True):  # Default is now True
                # Ensure parameters field is a dict, not None
                # (YAML can parse empty as None)
                if baseline_data.get("parameters") is None:
                    baseline_data["parameters"] = {}
                elif not isinstance(baseline_data.get("parameters"), dict):
                    raise TypeError(
                        "Baseline parameters must be provided as a mapping of "
                        "CLI flag names to simple values."
                    )
                baseline_config = BaselineConfig(**baseline_data)
        else:
            # No baseline section in config
            # Create default baseline with benchmark rate
            baseline_config = BaselineConfig(
                enabled=True, concurrency_levels=[benchmark.rate]
            )

        # Handle constraint parsing
        constraints = []
        if "constraints" in raw_config:
            constraint_data = raw_config["constraints"]
            if constraint_data is not None:
                if not isinstance(constraint_data, list):
                    msg = "Constraints must be provided as a list of expression strings"
                    raise TypeError(msg)
                constraints = [Constraint(expression=expr) for expr in constraint_data]

        return StudyConfig(
            study_name=study_name,
            database_url=database_url,
            optimization=optimization,
            benchmark=benchmark,
            parameters=validated_params,
            static_environment_variables=static_env_vars,
            static_parameters=static_params,
            baseline=baseline_config,
            logging_config=raw_config.get("logging"),
            storage_file=storage_file,
            study_prefix=study_prefix,
            use_explicit_name=use_explicit_name,
            constraints=constraints,
        )

    def _infer_parameter_type(self, parameter_config: dict[str, Any]):
        range_check = ["max" in parameter_config, "min" in parameter_config]
        list_check = "options" in parameter_config
        if any(range_check):
            min_value = parameter_config.get("min")
            max_value = parameter_config.get("max")
            min_is_numeric = isinstance(min_value, (int, float))
            max_is_numeric = isinstance(max_value, (int, float))
            if not min_is_numeric or not max_is_numeric:
                raise TypeError("'min' and 'max' must be numbers (int or float)")
            # check or clause incase users do (0, 1.0) as a range
            if isinstance(max_value, float) or isinstance(min_value, float):
                return float
            else:
                return int
        elif list_check:
            options = parameter_config.get("options")
            if not isinstance(options, list):
                raise TypeError("'options' must be a list")
            if options == [True, False] or options == [False, True]:
                return bool
            return list

        raise ValueError(f"Unable to parse parameter {parameter_config}")

    def _build_parameter_config(
        self, name: str, user_config: dict[str, Any]
    ) -> ParameterConfig:
        """Build parameter config from user config."""
        try:
            param_type = self._infer_parameter_type(user_config)
        except ValueError as _:
            raise ValueError(f"Unable to parse {name}: {user_config}")

        common = {
            "name": name,
            "enabled": user_config.get("enabled", True),
        }
        common.update(user_config)
        if param_type is float or param_type is int:
            # TODO: Separate out into RangeIntParamter and RangeFloatParameter types
            return RangeParameter(
                name=name,
                min=user_config.get("min"),
                max=user_config.get("max"),
                step=common.get("step", None),
                data_type=param_type,
            )
        elif param_type is list:
            return ListParameter(name=name, options=user_config.get("options"))
        elif param_type is bool:
            return BooleanParameter(name=name)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

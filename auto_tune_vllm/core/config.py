"""Configuration management with validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import os
import re
import yaml
from pydantic import BaseModel, Field

# Import BenchmarkConfig from the benchmarks module
from ..benchmarks.config import BenchmarkConfig


class ParameterConfig(BaseModel, ABC):
    """Base class for parameter configurations."""
    
    name: str
    enabled: bool = True
    description: Optional[str] = None
    
    @abstractmethod
    def generate_optuna_suggest(self, trial) -> Any:
        """Generate Optuna trial suggestion for this parameter."""
        pass


class RangeParameter(ParameterConfig):
    """Range-based parameter (continuous or discrete)."""
    
    min_value: Union[int, float] = Field(alias="min")
    max_value: Union[int, float] = Field(alias="max") 
    step: Optional[Union[int, float]] = None
    data_type: str = "float"  # "int" or "float"
    
    def generate_optuna_suggest(self, trial) -> Union[int, float]:
        """Generate Optuna range suggestion."""
        if self.data_type == "int":
            return trial.suggest_int(
                self.name, 
                int(self.min_value), 
                int(self.max_value),
                step=int(self.step) if self.step else None
            )
        else:
            return trial.suggest_float(
                self.name,
                float(self.min_value),
                float(self.max_value), 
                step=float(self.step) if self.step else None
            )


class ListParameter(ParameterConfig):
    """List-based parameter (categorical choices)."""
    
    options: List[Any]
    data_type: str = "str"
    
    def generate_optuna_suggest(self, trial) -> Any:
        """Generate Optuna categorical suggestion."""
        return trial.suggest_categorical(self.name, self.options)


class BooleanParameter(ParameterConfig):
    """Boolean parameter."""
    
    data_type: str = "bool"
    
    def generate_optuna_suggest(self, trial) -> bool:
        """Generate Optuna boolean suggestion."""
        return trial.suggest_categorical(self.name, [True, False])


class EnvironmentParameter(ParameterConfig):
    """Environment variable parameter (list-only choices)."""
    
    options: List[Any]
    data_type: str = "str"
    
    def generate_optuna_suggest(self, trial) -> Any:
        """Generate Optuna categorical suggestion for environment variable."""
        return trial.suggest_categorical(self.name, self.options)


@dataclass
class ObjectiveConfig:
    """Configuration for a single optimization objective."""
    
    metric: str  # "output_tokens_per_second", "request_latency", etc.
    direction: str  # "maximize" or "minimize"
    percentile: str = "median"  # "median", "p50", "p95", "p90", "p99", "mean"
    
    def __post_init__(self):
        """Validate objective configuration."""
        valid_metrics = {
            "output_tokens_per_second", "request_latency", "time_to_first_token_ms",
            "inter_token_latency_ms", "requests_per_second"
        }
        valid_directions = {"maximize", "minimize"}
        valid_percentiles = {"median", "p50", "p95", "p90", "p99", "mean"}
        
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{self.metric}'. Valid options: {valid_metrics}")
        if self.direction not in valid_directions:
            raise ValueError(f"Invalid direction '{self.direction}'. Valid options: {valid_directions}")
        if self.percentile not in valid_percentiles:
            raise ValueError(f"Invalid percentile '{self.percentile}'. Valid options: {valid_percentiles}")


@dataclass 
class OptimizationConfig:
    """Optimization configuration with support for new structured format and backward compatibility."""
    
    # Backward compatibility fields
    objective: Union[str, List[str]] = None  # Old format: "maximize", "minimize", or list
    sampler: str = "tpe"  # "tpe", "random", "gp", "botorch", "nsga2", "grid" 
    n_trials: int = 100
    n_startup_trials: Optional[int] = None  # Number of startup trials for samplers that support it
    max_concurrent: Optional[int] = None  # Maximum concurrent trials (required for resource management)
    
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
            self.objectives = [ObjectiveConfig(
                metric="output_tokens_per_second",
                direction="maximize",
                percentile="mean"
            )]
        elif self.preset == "low_latency":
            self.approach = "single_objective"
            self.objectives = [ObjectiveConfig(
                metric="request_latency",
                direction="minimize",
                percentile="p95"
            )]
        elif self.preset == "balanced":
            self.approach = "multi_objective"
            self.objectives = [
                ObjectiveConfig(
                    metric="output_tokens_per_second",
                    direction="maximize",
                    percentile="mean"
                ),
                ObjectiveConfig(
                    metric="request_latency",
                    direction="minimize",
                    percentile="median"
                )
            ]
        else:
            raise ValueError(f"Unknown preset '{self.preset}'. Valid options: high_throughput, low_latency, balanced")
    
    def _validate_structured_format(self):
        """Validate new structured format."""
        if self.approach not in ["single_objective", "multi_objective"]:
            raise ValueError(f"Invalid approach '{self.approach}'. Valid options: single_objective, multi_objective")
        
        if not self.objectives:
            raise ValueError("Objectives must be specified for structured format")
        
        if self.approach == "single_objective" and len(self.objectives) != 1:
            raise ValueError("Single objective optimization requires exactly one objective")
        
        if self.approach == "multi_objective" and len(self.objectives) < 2:
            raise ValueError("Multi-objective optimization requires at least two objectives")
    
    def _convert_old_format(self):
        """Convert old format to new structured format for backward compatibility."""
        if isinstance(self.objective, str):
            # Single objective
            self.approach = "single_objective"
            if self.objective == "maximize":
                # Default to maximizing throughput
                self.objectives = [ObjectiveConfig(
                    metric="output_tokens_per_second",
                    direction="maximize",
                    percentile="median"
                )]
            elif self.objective == "minimize":
                # Default to minimizing latency
                self.objectives = [ObjectiveConfig(
                    metric="request_latency",
                    direction="minimize",
                    percentile="median"
                )]
            else:
                raise ValueError(f"Invalid objective '{self.objective}'. Use 'maximize' or 'minimize'")
        elif isinstance(self.objective, list):
            # Multi-objective (legacy format)
            self.approach = "multi_objective"
            # Default to throughput vs latency
            self.objectives = [
                ObjectiveConfig(
                    metric="output_tokens_per_second",
                    direction="maximize",
                    percentile="median"
                ),
                ObjectiveConfig(
                    metric="request_latency",
                    direction="minimize",
                    percentile="median"
                )
            ]
    
    def _apply_default_config(self):
        """Apply default configuration when none is specified."""
        self.approach = "single_objective"
        self.objectives = [ObjectiveConfig(
            metric="output_tokens_per_second",
            direction="maximize",
            percentile="mean"
        )]
    
    @property
    def is_multi_objective(self) -> bool:
        """Check if this is multi-objective optimization."""
        return self.approach == "multi_objective"
    
    @property
    def optuna_directions(self) -> List[str]:
        """Get Optuna directions for study creation."""
        return [obj.direction for obj in self.objectives]
    
    def get_metric_key(self, objective_index: int = 0) -> str:
        """Get the metric key for extracting values from benchmark results."""
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

    enabled: bool = False
    concurrency_levels: List[int] = field(default_factory=lambda: [50])  # Concurrency levels to test
    
    def __post_init__(self):
        """Validate baseline configuration."""
        if self.enabled and not self.concurrency_levels:
            raise ValueError("Baseline configuration requires at least one concurrency level")
        
        if self.enabled:
            for concurrency in self.concurrency_levels:
                if not isinstance(concurrency, int) or concurrency <= 0:
                    raise ValueError(f"Invalid concurrency level: {concurrency}. Must be positive integer")


@dataclass
class StudyConfig:
    """Complete study configuration."""
    
    study_name: str
    database_url: Optional[str]
    optimization: OptimizationConfig
    benchmark: BenchmarkConfig
    parameters: Dict[str, ParameterConfig] = field(default_factory=dict)
    static_environment_variables: Dict[str, str] = field(default_factory=dict)  # NEW: Static environment variables
    baseline: Optional[BaselineConfig] = None  # NEW: Baseline configuration
    logging_config: Optional[Dict[str, Any]] = None
    storage_file: Optional[str] = None  # Alternative to database_url for file-based storage
    study_prefix: Optional[str] = None  # For auto-generated study names with custom prefix
    use_explicit_name: bool = False  # Flag to indicate explicit name usage (affects load_if_exists behavior)
    
    @classmethod
    def from_file(cls, config_path: str, schema_path: Optional[str] = None, 
                 defaults_path: Optional[str] = None, vllm_version: Optional[str] = None) -> StudyConfig:
        """Load and validate configuration from YAML file."""
        config_validator = ConfigValidator(schema_path, defaults_path, vllm_version)
        return config_validator.load_and_validate(config_path)


class ConfigValidator:
    """Validates study configurations against parameter schema."""
    
    def __init__(self, schema_path: Optional[str] = None, defaults_path: Optional[str] = None, 
                 vllm_version: Optional[str] = None):
        """Initialize with parameter schema and optional defaults."""
        if schema_path is None:
            # Resolve vLLM version: prefer explicit, else auto-detect, else None
            resolved_version = vllm_version
            if resolved_version is None:
                try:
                    import vllm  # noqa: F401
                    from importlib.metadata import version as _dist_version
                    resolved_version = _dist_version("vllm")
                except Exception:
                    resolved_version = None
            # Choose schema
            if resolved_version and resolved_version.startswith("0.10.0"):
                schema_path = Path(__file__).parent.parent / "schemas" / "v0_10_0.yaml"
            else:
                schema_path = Path(__file__).parent.parent / "schemas" / "v0_10_1_1.yaml"
        # Store resolved version for defaults handling
        self.vllm_version = vllm_version or locals().get("resolved_version")
        
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        
        # Load defaults - support versioned defaults
        self.defaults = {}
        self.vllm_version = vllm_version
        
        if defaults_path is not None:
            self.defaults_path = Path(defaults_path)
            self.defaults = self._load_defaults()
        elif vllm_version is not None:
            # Load version-specific defaults
            self._load_version_defaults(vllm_version)
        else:
            # Try to load latest defaults
            self._load_latest_defaults()
    
    def _get_versioned_schema_path(self, vllm_version: str) -> Path:
        """
        Get the schema path for a specific vLLM version.
        
        Args:
            vllm_version: vLLM version string (e.g., "0.10.1.1")
            
        Returns:
            Path to versioned schema file
            
        Raises:
            FileNotFoundError: If versioned schema doesn't exist
        """
        # Convert version to filename format (e.g., "0.10.1.1" -> "v0_10_1_1.yaml")
        safe_version = vllm_version.replace('.', '_')
        schemas_dir = Path(__file__).parent.parent / "schemas"
        versioned_schema_path = schemas_dir / f"v{safe_version}.yaml"
        
        if versioned_schema_path.exists():
            print(f"Using version-specific schema for vLLM {vllm_version}: {versioned_schema_path}")
            return versioned_schema_path
        else:
            # Fallback to default schema with warning
            fallback_schema = schemas_dir / "parameter_schema_original.yaml"
            print(f"Warning: No schema found for vLLM version {vllm_version}, falling back to default schema: {fallback_schema}")
            return fallback_schema
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load parameter schema from YAML."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path) as f:
            return yaml.safe_load(f)
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load defaults from YAML file."""
        if not self.defaults_path.exists():
            raise FileNotFoundError(f"Defaults file not found: {self.defaults_path}")
        
        with open(self.defaults_path) as f:
            defaults_data = yaml.safe_load(f)
        
        return self._flatten_defaults(defaults_data)
    
    def _load_version_defaults(self, version: str):
        """Load version-specific defaults."""
        try:
            from ..utils.version_manager import VLLMVersionManager
            manager = VLLMVersionManager()
            defaults_data = manager.load_defaults(version)
            self.defaults = self._flatten_defaults(defaults_data)
            self.defaults_path = manager.get_defaults_path(version)
        except Exception as e:
            print(f"Warning: Could not load vLLM defaults for version {version}: {e}")
            self.defaults = {}
            self.defaults_path = None
    
    def _load_latest_defaults(self):
        """Try to load the latest available defaults."""
        try:
            from ..utils.version_manager import VLLMVersionManager
            manager = VLLMVersionManager()
            defaults_data = manager.load_defaults()  # Load latest
            self.defaults = self._flatten_defaults(defaults_data)
            self.defaults_path = manager.get_defaults_path()
        except Exception:
            # No versioned defaults available, use empty defaults
            self.defaults = {}
            self.defaults_path = None
    
    def _flatten_defaults(self, defaults_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten the nested defaults structure for easy lookup."""
        flat_defaults = {}
        for section, section_defaults in defaults_data.get("defaults", {}).items():
            for param_name, default_value in section_defaults.items():
                flat_defaults[param_name] = default_value
        return flat_defaults
    
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
        - ${VAR_NAME} - expands to environment variable value or empty string if not set
        - ${VAR_NAME:-default_value} - expands to environment variable value or default_value if not set

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
                    print(f"Warning: Environment variable '{var_name}' not found, using empty string")
                    return ""
            return env_value

        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*?)(?::-(.*?))?\}'
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
    
    def _handle_study_naming(self, study_info: Dict[str, Any]) -> tuple[str, Optional[str], bool]:
        """
        Handle study naming logic with prefix support.
        
        Returns:
            tuple: (study_name, study_prefix, use_explicit_name)
        
        Rules:
        1. If just 'name' is provided: Use exact name, fail if exists (use_explicit_name=True)
        2. If just 'prefix' is provided: Auto-generate with prefix (use_explicit_name=False)
        3. If both provided: Validation error
        4. If neither provided: Auto-generate with default prefix (use_explicit_name=False)
        """
        name = study_info.get("name")
        prefix = study_info.get("prefix")
        
        # Scenario 3: Both name and prefix provided - ERROR
        if name and prefix:
            raise ValueError(
                "Cannot specify both 'name' and 'prefix' in study configuration. "
                "Use 'name' for exact study names that must be unique, or 'prefix' for auto-generated names."
            )
        
        # Scenario 1: Just name provided - use exact name, fail if exists
        if name and not prefix:
            print(f"Using explicit study name: {name} (will fail if study already exists)")
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
    
    def _validate_config(self, raw_config: Dict[str, Any]) -> StudyConfig:
        """Validate configuration against schema."""
        # Validate and build parameter configs
        validated_params = {}
        
        for param_name, param_config in raw_config.get("parameters", {}).items():
            if not param_config.get("enabled", True):
                continue
            
            # Check if this is an environment variable (has explicit type or not in schema)
            param_type = param_config.get("type")
            schema_def = self.schema.get("parameters", {}).get(param_name)
            
            if param_type == "environment" or (not schema_def and "options" in param_config):
                # This is an environment variable parameter
                if "range" in param_config or "min" in param_config or "max" in param_config or "step" in param_config:
                    raise ValueError(f"Environment parameter '{param_name}' cannot use range configurations. Only list options are allowed.")
                
                if "options" not in param_config:
                    raise ValueError(f"Environment parameter '{param_name}' must specify options as a list")
                
                validated_param = EnvironmentParameter(
                    name=param_name,
                    enabled=param_config.get("enabled", True),
                    options=param_config["options"],
                    data_type=param_config.get("data_type", "str"),
                    description=param_config.get("description", f"Environment variable {param_name}")
                )
            else:
                # This is a regular vLLM parameter
                if not schema_def:
                    raise ValueError(f"Unknown parameter in schema: {param_name}")
                
                # Build parameter config based on schema type
                validated_param = self._build_parameter_config(param_name, param_config, schema_def)
            
            validated_params[param_name] = validated_param
        
        # Validate static environment variables (simple key-value pairs)
        static_env_vars = {}
        
        for env_name, env_value in raw_config.get("static_environment_variables", {}).items():
            if not isinstance(env_value, (str, int, float, bool)):
                raise ValueError(f"Static environment variable '{env_name}' must be a simple value (string, number, or boolean), got {type(env_value)}")
            
            static_env_vars[env_name] = str(env_value)
        
        # Build other configs
        study_info = raw_config.get("study", {})
        if study_info is None:
            study_info = {}
        
        # Handle study naming logic with prefix support
        study_name, study_prefix, use_explicit_name = self._handle_study_naming(study_info)
        
        # Handle optimization config with validation
        opt_config_data = raw_config["optimization"]
        
        # Convert objective config if using new structured format
        if "objective" in opt_config_data and isinstance(opt_config_data["objective"], dict):
            # Single objective structured format
            obj_data = opt_config_data["objective"]
            opt_config_data["objectives"] = [ObjectiveConfig(**obj_data)]
            del opt_config_data["objective"]
        elif "objectives" in opt_config_data:
            # Multi-objective structured format
            objectives_data = opt_config_data["objectives"]
            opt_config_data["objectives"] = [ObjectiveConfig(**obj) for obj in objectives_data]
        
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
            raise ValueError("Cannot specify both database_url and storage_file. Choose one storage method.")
        
        # Handle baseline configuration
        baseline_config = None
        if "baseline" in raw_config:
            baseline_data = raw_config["baseline"]
            if baseline_data.get("enabled", False):
                baseline_config = BaselineConfig(**baseline_data)
        
        return StudyConfig(
            study_name=study_name,
            database_url=database_url,
            optimization=optimization,
            benchmark=benchmark, 
            parameters=validated_params,
            static_environment_variables=static_env_vars,
            baseline=baseline_config,
            logging_config=raw_config.get("logging"),
            storage_file=storage_file,
            study_prefix=study_prefix,
            use_explicit_name=use_explicit_name
        )
    
    def _build_parameter_config(
        self, 
        name: str, 
        user_config: Dict[str, Any], 
        schema_def: Dict[str, Any]
    ) -> ParameterConfig:
        """Build parameter config from user config, defaults, and schema."""
        param_type = schema_def["type"]
        description = schema_def.get("description")
        
        base_config = {
            "name": name,
            "enabled": user_config.get("enabled", True),
            "description": description
        }
        
        def get_value(key: str, schema_fallback=None, allow_defaults: bool = False):
            """
            Priority: user_config > (defaults if allow_defaults) > schema > schema_fallback.
            Note: defaults are parameter values, not bounds; do NOT use for min/max/step/options.
            """
            if key in user_config:
                return user_config[key]
            if allow_defaults and name in self.defaults:
                return self.defaults[name]
            if key in schema_def:
                return schema_def[key]
            return schema_fallback
        
        if param_type == "range":
            return RangeParameter(
                **base_config,
                min=get_value("min", schema_def["min"], allow_defaults=False),
                max=get_value("max", schema_def["max"], allow_defaults=False),
                step=get_value("step", schema_def.get("step"), allow_defaults=False),
                data_type=schema_def["data_type"]
            )
        elif param_type == "list":
            # Allow user to restrict schema options
            schema_options = schema_def["options"]
            user_options = user_config.get("options", schema_options)
            
            # Validate user options are subset of schema options
            invalid_options = set(user_options) - set(schema_options)
            if invalid_options:
                raise ValueError(f"Invalid options for {name}: {invalid_options}")
            
            return ListParameter(
                **base_config,
                options=user_options,
                data_type=schema_def["data_type"]
            )
        elif param_type == "boolean":
            return BooleanParameter(**base_config)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
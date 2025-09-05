# Trial Configuration Examples

This directory contains configuration files and Python examples that demonstrate how to use the versioned vLLM defaults system for hyperparameter optimization.

## File Overview

**Configuration Files:**
- `study_config.yaml` - Basic optimization configuration for getting started
- `trial_config_comprehensive.yaml` - Comprehensive optimization with 11 parameters
- `trial_config_version_specific.yaml` - Version-specific config for vLLM v0.10.1.1
- `test_defaults_config.yaml` - Test configuration for the defaults system

**Python Examples:**
- `basic_usage.py` - Python API usage example with StudyController and Ray
- `versioned_defaults_demo.py` - Interactive demo of versioned defaults system
- `vllm_cli_demo.py` - CLI parsing and schema generation demonstration

## Overview

All trial configurations now support **versioned vLLM defaults**, which means:
- ✅ Defaults are automatically extracted from your installed vLLM version
- ✅ No hardcoded values - always synchronized with vLLM
- ✅ Version-specific optimization for different vLLM releases
- ✅ Backward compatibility with older vLLM versions

## Available Examples

### Configuration Files

#### `study_config.yaml`
- **Purpose**: Basic optimization configuration with essential parameters
- **Parameters**: 3 core parameters (max_num_batched_tokens, gpu_memory_utilization, kv_cache_dtype)
- **Use case**: Getting started with vLLM optimization
- **Runtime**: ~100 trials, moderate resources

#### `trial_config_comprehensive.yaml`
- **Purpose**: Comprehensive optimization across all parameter categories
- **Parameters**: 11 parameters covering cache, model, scheduler, and parallel configs
- **Use case**: Thorough optimization when you have time and resources
- **Runtime**: ~200 trials, high resources

#### `trial_config_version_specific.yaml`
- **Purpose**: Optimized specifically for vLLM v0.10.1.1
- **Parameters**: 13 version-specific parameters
- **Use case**: When you want version-specific optimization
- **Runtime**: ~100 trials, moderate resources

#### `test_defaults_config.yaml`
- **Purpose**: Testing the defaults system functionality
- **Parameters**: 4 parameters demonstrating schema and vLLM defaults
- **Use case**: Development and testing of the defaults system

### Python Examples

#### `basic_usage.py`
- **Purpose**: Demonstrates using the Python API for optimization
- **Features**: StudyController setup, Ray/Local backends, Optuna integration
- **Use case**: When you want to run optimization from Python code instead of CLI
- **Dependencies**: Requires Ray for distributed execution (or can use LocalExecutionBackend)
- **Configuration**: Uses `study_config.yaml` as the configuration file

#### `versioned_defaults_demo.py`
- **Purpose**: Interactive demo of the versioned defaults system
- **Features**: Version management, configuration loading, version comparison
- **Use case**: Understanding how versioned defaults work and exploring available versions
- **Output**: Shows available versions, demonstrates loading configs, version comparison
- **Requirements**: Needs generated vLLM defaults (run `generate_vllm_defaults.py` first)

#### `vllm_cli_demo.py`
- **Purpose**: Demonstrates vLLM CLI parsing and schema generation
- **Features**: CLI argument parsing, parameter schema generation, configuration creation
- **Use case**: Development and understanding of CLI integration
- **Output**: Generates schemas, defaults YAML, and example configurations in `output/` directory
- **Requirements**: Needs vLLM installed in environment

## Usage

### Configuration Files

#### Basic Usage (Auto-detect vLLM version)

```python
from auto_tune_vllm.core.config import StudyConfig

# Automatically uses defaults from installed vLLM version
config = StudyConfig.from_file("examples/study_config.yaml")
```

#### Explicit Version Usage

```python
from auto_tune_vllm.core.config import StudyConfig

# Use defaults from specific vLLM version
config = StudyConfig.from_file(
    "examples/trial_config_version_specific.yaml",
    vllm_version="0.10.1.1"
)
```

#### CLI Usage

```bash
# Run optimization with auto-detected defaults
python -m auto_tune_vllm examples/study_config.yaml

# Run comprehensive optimization
python -m auto_tune_vllm examples/trial_config_comprehensive.yaml

# Run with specific version defaults
python -m auto_tune_vllm examples/trial_config_version_specific.yaml --vllm-version 0.10.1.1
```

### Python API Examples

#### Run Basic Optimization

```python
# Use the Python API example
python examples/basic_usage.py
```

#### Explore Versioned Defaults

```python
# Interactive demo of versioned defaults system
python examples/versioned_defaults_demo.py
```

#### Generate vLLM Schemas

```python
# Demo CLI parsing and schema generation
python examples/vllm_cli_demo.py
```

## Parameter Categories

The configurations cover four main vLLM parameter categories:

### Cache Configuration (`cacheconfig`)
- `gpu_memory_utilization`: GPU memory fraction (vLLM default: 0.9)
- `kv_cache_dtype`: KV cache data type (vLLM default: auto)
- `swap_space`: CPU swap space in GB (vLLM default: 4)
- `calculate_kv_scales`: Dynamic KV scale calculation (vLLM default: False)
- `prefix_caching_hash_algo`: Prefix caching hash (vLLM default: builtin)

### Model Configuration (`modelconfig`)
- `max_seq_len_to_capture`: Max sequence length (vLLM default: 8192)
- `dtype`: Model data type (vLLM default: auto)
- `enforce_eager`: Disable CUDA graphs (vLLM default: False)
- `tokenizer_mode`: Tokenizer mode (vLLM default: auto)
- `disable_cascade_attn`: Disable cascade attention (vLLM default: False)

### Scheduler Configuration (`schedulerconfig`)
- `max_num_batched_tokens`: Max tokens per batch (schema defaults: 1024-32768)
- `async_scheduling`: Enable async scheduling (vLLM default: False)
- `scheduling_policy`: Scheduling policy (vLLM default: fcfs)
- `long_prefill_token_threshold`: Chunked prefill threshold (vLLM default: 0)
- `max_num_partial_prefills`: Partial prefill count (vLLM default: 1)

### Parallel Configuration (`parallelconfig`)
- `tensor_parallel_size`: Tensor parallelism (vLLM default: 1)
- `data_parallel_size`: Data parallelism (vLLM default: 1)
- `disable_custom_all_reduce`: Use NCCL vs custom (vLLM default: False)

## Default Value Sources

Parameters can get default values from three sources (in priority order):

1. **User configuration**: Explicitly specified in your YAML file
2. **vLLM CLI defaults**: Extracted from `vllm serve --help` for your version
3. **Schema defaults**: Fallback defaults defined in `parameter_schema.yaml`

## Version Management

### List Available Versions

```python
from auto_tune_vllm.utils import VLLMVersionManager

manager = VLLMVersionManager()
print(manager.get_version_info_summary())
```

### Generate Defaults for Current vLLM Version

```bash
# Generate versioned defaults
python scripts/generate_vllm_defaults.py --verbose

# Generate for specific sections only
python scripts/generate_vllm_defaults.py --sections cacheconfig,schedulerconfig
```

### Compare Versions

```python
from auto_tune_vllm.utils import VLLMVersionManager

manager = VLLMVersionManager()
comparison = manager.compare_versions("0.10.0", "0.10.1.1")
print(f"Changed parameters: {comparison['summary']['changed_count']}")
```

## Best Practices

### 1. Start Simple
- Begin with `study_config.yaml` for basic optimization
- Use `basic_usage.py` to understand the Python API
- Add more parameters as you understand their impact

### 2. Resource Planning
- **Basic configs** (`study_config.yaml`): 2-4 hours on modest hardware
- **Comprehensive configs** (`trial_config_comprehensive.yaml`): 8-12 hours on powerful hardware
- **Version-specific configs** (`trial_config_version_specific.yaml`): 4-6 hours with targeted optimization

### 3. Parameter Selection
- Focus on parameters that impact your specific use case
- Use performance profiling to identify bottlenecks first
- Start with cache and scheduler parameters for most workloads

### 4. Version Management
- Regenerate defaults when updating vLLM
- Keep configs version-specific for reproducibility
- Use explicit versions in production deployments

### 5. Monitoring
- Enable detailed logging for parameter impact analysis
- Monitor both throughput and memory usage
- Use file-based logging for long-running optimizations

## Troubleshooting

### Version Not Found
```bash
# Generate defaults for your vLLM version
python scripts/generate_vllm_defaults.py --verbose
```

### Parameter Not in Schema
- Check if parameter exists in vLLM CLI: `vllm serve --help`
- Regenerate parameter schema if needed
- Use exact parameter names from CLI (convert `--param-name` to `param_name`)

### Memory Issues
- Reduce `gpu_memory_utilization` range
- Lower `max_num_batched_tokens` values
- Increase `swap_space` for CPU offloading

### Performance Issues
- Use file-based logging instead of database
- Reduce number of trials for initial testing
- Focus on fewer parameters initially

## Examples

### Quick Start
```bash
# 1. Generate defaults for your vLLM version
python scripts/generate_vllm_defaults.py

# 2. Run basic optimization
python -m auto_tune_vllm examples/study_config.yaml

# 3. View results
tail -f /tmp/auto-tune-vllm-logs/optimization.log
```

### Comprehensive Optimization Workflow
```bash
# 1. Generate version-specific defaults
python scripts/generate_vllm_defaults.py --verbose

# 2. Run comprehensive optimization
python -m auto_tune_vllm examples/trial_config_comprehensive.yaml

# 3. Deploy with best parameters
# Use the optimal parameters in your production vLLM config
```

### Python API Workflow
```bash
# 1. Run Python API example
python examples/basic_usage.py

# 2. Explore versioned defaults interactively
python examples/versioned_defaults_demo.py

# 3. Generate schemas and understand CLI integration
python examples/vllm_cli_demo.py
```

### Development Workflow
```bash
# 1. Test defaults system
python -m auto_tune_vllm examples/test_defaults_config.yaml

# 2. Test version-specific optimization
python -m auto_tune_vllm examples/trial_config_version_specific.yaml --vllm-version 0.10.1.1

# 3. Generate new schemas for development
python examples/vllm_cli_demo.py
```

For more information, see the main project documentation.

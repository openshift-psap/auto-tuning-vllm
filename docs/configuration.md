# Configuration Guide

This guide provides detailed explanations of all configuration options available in auto-tune-vllm for optimizing vLLM performance.

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Study Configuration](#study-configuration)
3. [Optimization Configuration](#optimization-configuration)
4. [Benchmark Configuration](#benchmark-configuration)
5. [Logging Configuration](#logging-configuration)
6. [Parameter Configuration](#parameter-configuration)
7. [Environment Variables](#environment-variables)
8. [Configuration Examples](#configuration-examples)

## Configuration File Structure

Auto-tune-vllm uses YAML configuration files with five main sections. Each section controls a different aspect of the optimization process:

- **`study`**: Defines study metadata, naming, and storage backend
- **`optimization`**: Specifies what metrics to optimize and how
- **`benchmark`**: Configures how performance benchmarks are executed
- **`logging`**: Controls logging output and verbosity (optional)
- **`parameters`**: Defines which vLLM parameters to optimize and their ranges

The basic structure looks like:

```yaml
study:
  # Study identity and storage
  
optimization:
  # What to optimize and optimization strategy
  
benchmark:
  # How to measure performance
  
logging:
  # Where and how to log (optional)
  
parameters:
  # Which vLLM parameters to tune
```

## Study Configuration

The `study` section controls study identity, naming, and where optimization results are stored. This section is required in every configuration file.

### Study Naming

Studies need unique identifiers to track optimization progress. Auto-tune-vllm provides flexible naming options:

#### `name` (string, optional)
Specifies an explicit study name that must be unique. If a study with this name already exists, the optimization will fail unless you're resuming it. Use this when you need predictable, exact study names.

#### `prefix` (string, optional)  
Used for auto-generating unique study names in the format `{prefix}_{timestamp}`. This ensures uniqueness while providing meaningful prefixes. If omitted, defaults to "study".

**Naming Rules:**
- You cannot specify both `name` and `prefix` - choose one approach
- If neither is specified, auto-generates names like `study_123456`
- Explicit names (`name`) fail if the study already exists
- Prefixed names (`prefix`) automatically create unique variants

### Storage Backend

Auto-tune-vllm supports two storage backends for persisting optimization results:

#### `database_url` (string, optional)
PostgreSQL connection URL for production environments. Supports concurrent optimization workers and provides robust persistence. The URL format is: `postgresql://username:password@host:port/database`

#### `storage_file` (string, optional)
Path to SQLite database file for single-machine optimization. Simpler to set up than PostgreSQL but doesn't support concurrent workers. If not specified, defaults to `./optuna_studies/{study_name}/study.db`.

**Storage Rules:**
- You cannot specify both `database_url` and `storage_file`
- If neither is specified, uses SQLite with default file location
- PostgreSQL is recommended for production with multiple workers
- SQLite is suitable for development and single-worker optimization

### Additional Options

#### `study_prefix` (string, optional)
Internal option for advanced study naming scenarios. Generally not needed in user configurations.

#### `use_explicit_name` (boolean, optional)
Internal flag that controls study loading behavior. Automatically set based on your naming choice.

## Optimization Configuration

The `optimization` section defines what performance metrics to optimize and the strategy for finding optimal parameter combinations. This section is required and controls the core optimization behavior.

### Configuration Approaches

Auto-tune-vllm supports three ways to configure optimization, from simple to advanced:

#### Preset-Based Configuration (Recommended for Beginners)

Use `preset` for common optimization scenarios:

##### `preset` (string, optional)
Pre-configured optimization strategies for typical use cases:

- **`"high_throughput"`**: Maximizes token generation rate (output_tokens_per_second)
- **`"low_latency"`**: Minimizes 95th percentile request latency  
- **`"balanced"`**: Multi-objective optimization balancing throughput and latency

When using presets, you only need to specify `n_trials`. The preset automatically configures the approach, objectives, and sampler.

#### Structured Configuration (Advanced)

Use `approach` and `objectives` for full control over optimization:

##### `approach` (string, optional)
Defines the optimization strategy:

- **`"single_objective"`**: Optimize one metric only. Best when you have a clear primary goal.
- **`"multi_objective"`**: Optimize multiple metrics simultaneously, finding trade-off solutions.

##### `objectives` (list, required when using approach)
List of optimization objectives. Each objective specifies:

**`metric`** (string, required): The performance metric to optimize. Available metrics:
- `output_tokens_per_second`: Token generation throughput (tokens/sec)
- `request_latency`: End-to-end request latency (milliseconds) 
- `time_to_first_token_ms`: Time until first token appears (milliseconds)
- `inter_token_latency_ms`: Latency between consecutive tokens (milliseconds)
- `requests_per_second`: Request processing throughput (requests/sec)

**`direction`** (string, required): Optimization direction:
- `"maximize"`: Increase the metric value (for throughput metrics)
- `"minimize"`: Decrease the metric value (for latency metrics)

**`percentile`** (string, optional): Which percentile of the metric to optimize:
- `"median"` or `"p50"`: 50th percentile (most stable, default)
- `"p95"`: 95th percentile (good for SLA optimization)
- `"p90"`: 90th percentile (balanced approach)
- `"p99"`: 99th percentile (extreme tail optimization)

#### Legacy Configuration (Backward Compatibility)

##### `objective` (string, optional) 
Legacy single-objective format:
- `"maximize"`: Defaults to maximizing throughput
- `"minimize"`: Defaults to minimizing latency

This format is deprecated but still supported for backward compatibility.

### Optimization Algorithm Settings

#### `sampler` (string, optional)
The optimization algorithm to use. Default is "tpe":

- **`"tpe"`**: Tree-structured Parzen Estimator. Best general-purpose sampler for single-objective optimization.
- **`"nsga2"`**: Non-dominated Sorting Genetic Algorithm II. Recommended for multi-objective optimization.
- **`"botorch"`**: Bayesian Optimization with Torch. Advanced sampler that can be faster but may get stuck in local optima.
- **`"random"`**: Random sampling. Useful for baselines and quick testing only.
- **`"grid"`**: Exhaustive grid search. Tests all parameter combinations (use with small parameter spaces only).

#### `n_trials` (integer, required)
Number of optimization trials to run. Each trial tests one parameter combination:
- **Development**: 10-50 trials for quick testing
- **Production**: 100-500 trials for thorough optimization
- **Multi-objective**: Typically needs 2x more trials than single-objective

#### `n_startup_trials` (integer, optional)
Number of random trials to run before starting the main sampler algorithm. Only supported by some samplers (TPE, BoTorch). Helps initialize the sampler with diverse data points.

### Preset Configurations Explained

#### High Throughput Preset
Optimizes for maximum token generation speed using median throughput. Best for batch processing and high-volume serving where latency is less critical.

#### Low Latency Preset  
Optimizes for minimal 95th percentile request latency. Best for interactive applications where response time matters more than maximum throughput.

#### Balanced Preset
Multi-objective optimization finding the best trade-offs between throughput and latency. Provides a Pareto front of solutions rather than a single optimum. Best when you need to balance performance characteristics.

## Benchmark Configuration

The `benchmark` section controls how performance measurements are conducted. This section is required and defines the workload used to evaluate different vLLM parameter combinations.

### Core Benchmark Settings

#### `benchmark_type` (string, optional)
The benchmarking framework to use. Currently only "guidellm" is supported. Defaults to "guidellm".

#### `model` (string, required)
The HuggingFace model identifier to benchmark. This should match the model you plan to serve in production. Examples:
- `"facebook/opt-125m"` (small model for testing)
- `"Qwen/Qwen3-30B-A3B-FP8"` (production model)
- `"microsoft/DialoGPT-medium"` (conversational model)

#### `max_seconds` (integer, optional)
Duration in seconds for each benchmark run. Longer benchmarks provide more accurate measurements but take more time. Typical values:
- **Development**: 60-120 seconds for quick feedback
- **Production**: 300-600 seconds for stable measurements
Default: 300 seconds

### Workload Configuration

The benchmark needs a workload to test against. Auto-tune-vllm supports both synthetic data generation and real datasets.

#### Synthetic Data (Recommended)

##### `dataset` (null)
Set to `null` to use synthetic data generation. This creates artificial prompts and responses based on your specifications.

##### `prompt_tokens` (integer, optional)
Base number of tokens in generated prompts. Default: 1000

##### `output_tokens` (integer, optional) 
Base number of tokens in generated responses. Default: 1000

##### Advanced Synthetic Data Distribution

For more realistic workloads, control the distribution of prompt and output lengths:

##### `prompt_tokens_stdev` (integer, optional)
Standard deviation for prompt token lengths. Creates variation around the base `prompt_tokens` value. Default: 128

##### `prompt_tokens_min` (integer, optional)
Minimum prompt length in tokens. Default: 256

##### `prompt_tokens_max` (integer, optional)
Maximum prompt length in tokens. Default: 1024

##### `output_tokens_stdev` (integer, optional)
Standard deviation for output token lengths. Default: 512

##### `output_tokens_min` (integer, optional)
Minimum output length in tokens. Default: 1024

##### `output_tokens_max` (integer, optional)
Maximum output length in tokens. Default: 3072

#### Real Dataset Configuration

##### `dataset` (string)
Path to a real dataset file or HuggingFace dataset identifier. Supported formats:
- Local JSONL files: `"path/to/dataset.jsonl"`
- HuggingFace datasets: `"huggingface/dataset_name"`

When using real datasets, the `prompt_tokens` and `output_tokens` settings are ignored.

### Load Configuration

#### `rate` (integer, optional)
Number of concurrent requests to maintain during benchmarking. This simulates realistic server load:
- **Light load**: 10-20 requests
- **Moderate load**: 50-100 requests  
- **Heavy load**: 200+ requests
Default: 50

### Advanced Options

#### `processor` (string, optional)
Separate model for request processing if different from the served model. Rarely needed - defaults to the same value as `model`.

## Logging Configuration

The `logging` section controls where and how detailed logging information is recorded. This section is optional - if omitted, logs are only displayed on the console.

#### `file_path` (string, optional)
Directory path where log files should be written. Auto-tune-vllm will create log files in this directory for:
- Optimization progress and results
- vLLM server output
- Benchmark execution details
- Error and debugging information

If not specified, no log files are created and all output goes to the console only.

#### `log_level` (string, optional)
Controls the verbosity of logging output. Available levels:
- **`"DEBUG"`**: Detailed debugging information, including parameter values and internal state
- **`"INFO"`**: General information about optimization progress (default)
- **`"WARNING"`**: Only warnings and errors
- **`"ERROR"`**: Only error messages

**Note**: File logging is recommended for production optimization runs to preserve detailed results and troubleshooting information.

## Parameter Configuration

The `parameters` section defines which vLLM server parameters should be optimized and the ranges or options to explore for each parameter. This section controls the parameter space that the optimization algorithm searches through.

### Parameter Structure

Each parameter has a common structure with required and optional fields:

#### `enabled` (boolean, required)
Controls whether this parameter should be included in optimization:
- `true`: Include this parameter in optimization
- `false`: Skip this parameter (use vLLM defaults)

#### Parameter Type-Specific Configuration

Auto-tune-vllm supports three types of parameters, each with different configuration options:

### Range Parameters

Range parameters define continuous or discrete numeric ranges to optimize over. Used for parameters like memory utilization or batch sizes.

#### Configuration Fields:
- **`min`** (number, optional): Minimum value to test. Uses schema default if not specified.
- **`max`** (number, optional): Maximum value to test. Uses schema default if not specified.  
- **`step`** (number, optional): Step size between values. Uses schema default if not specified.

The optimizer will test values between `min` and `max` in increments of `step`. For floating-point parameters, the step can be a decimal value.

### List Parameters  

List parameters define categorical choices from a fixed set of options. Used for parameters like data types or scheduling policies.

#### Configuration Fields:
- **`options`** (array, optional): List of valid values to test. Uses schema defaults if not specified.

The optimizer will test each value in the options list.

### Boolean Parameters

Boolean parameters test both true and false values. Used for feature flags and enable/disable options.

#### Configuration Fields:
No additional configuration needed - automatically tests both `true` and `false` values.

### Available Parameters

Auto-tune-vllm supports optimization of 27+ vLLM server parameters. For detailed descriptions of each parameter, run:

```bash
vllm serve --help
```

The available parameters include:

**Memory & Cache**: `gpu_memory_utilization`, `swap_space`, `block_size`, `kv_cache_dtype`  
**Model & Computation**: `dtype`, `enforce_eager`, `max_seq_len_to_capture`, `compilation_config`  
**Batching & Scheduling**: `max_num_batched_tokens`, `scheduling_policy`, `scheduler_delay_factor`, `max_num_partial_prefills`  
**CUDA Graphs**: `cuda_graph_sizes`, `long_prefill_token_threshold`  
**Parallelism**: `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`  
**Caching**: `enable_prefix_caching`

### Parameter Configuration Guidelines

#### Schema Defaults
If you don't specify `min`, `max`, `step`, or `options` for a parameter, auto-tune-vllm uses built-in schema defaults based on typical vLLM usage patterns.

#### Important Notes
- **`gpu_memory_utilization`**: Should not go below 0.9 as it significantly reduces performance
- **Parallelism parameters**: `tensor_parallel_size * pipeline_parallel_size * data_parallel_size` must not exceed your GPU count

#### Performance Impact
Focus on high-impact parameters first:
- **High impact**: `gpu_memory_utilization`, `max_num_batched_tokens`, `kv_cache_dtype`
- **Medium impact**: `block_size`, `dtype`  
- **Low impact**: `scheduler_delay_factor`, `compilation_config`

#### Performance Notes for High-Impact Parameters

**`kv_cache_dtype`**: FP8 typically provides 2x+ throughput improvement over "auto". Consider using FP8 if your model supports it.

**`gpu_memory_utilization`**: Values below 0.9 significantly reduce performance. Start optimization around 0.9-0.95 range.

**`max_num_batched_tokens`**: Higher values generally improve throughput but increase memory usage. Balance with `gpu_memory_utilization`.

## Baseline Configuration

Baseline trials establish performance baselines using pure vLLM defaults before running optimization. This helps measure optimization improvements and provides reference performance data.

### Baseline Configuration Fields

```yaml
baseline:
  enabled: true
  concurrency_levels: [50, 100, 150]  # Test multiple load levels
```

#### Configuration Fields:
- **`enabled`** (boolean): Enable baseline trials
- **`concurrency_levels`** (array): List of concurrency levels to test baseline performance

### Baseline Trial Behavior

Baseline trials use **pure vLLM defaults** with only one parameter modified:
- `--max-num-seqs` is set to the concurrency level being tested
- All other parameters use vLLM's built-in defaults (not hardcoded values)

This provides clean baseline performance data for comparison with optimized configurations.

### Example Configuration

```yaml
baseline:
  enabled: true
  concurrency_levels: [50, 100]

optimization:
  preset: "high_throughput"
  n_trials: 50

parameters:
  gpu_memory_utilization:
    enabled: true
    min: 0.88
    max: 0.95
    
  kv_cache_dtype:
    enabled: true
    options: ["auto", "fp8"]
    # PERFORMANCE NOTE: fp8 typically provides 2x+ throughput improvement
```

This configuration will:
1. **First** run baseline trials at concurrency 50 and 100 using pure vLLM defaults
2. **Then** run 50 optimization trials to find the best parameter settings
3. **Finally** compare the best optimized performance against the baseline

## Environment Variables

Auto-tune-vllm configuration files support environment variable expansion, allowing you to externalize sensitive information and environment-specific settings.

### Variable Expansion Syntax

Environment variables are referenced using `${VARIABLE_NAME}` syntax within YAML values:

#### Basic Expansion: `${VAR_NAME}`
Expands to the environment variable value, or an empty string if the variable is not set.

#### Default Values: `${VAR_NAME:-default_value}`  
Expands to the environment variable value if set, otherwise uses the provided default value.

### Usage Patterns

#### Database Credentials
Keep database passwords out of configuration files:
```yaml
study:
  database_url: "postgresql://user:${POSTGRES_PASSWORD}@localhost:5432/optuna"
```

#### Environment-Specific Settings
Use different log levels per environment:
```yaml
logging:
  log_level: "${LOG_LEVEL:-INFO}"
```

#### Model Configuration
Allow model selection via environment:
```yaml
benchmark:
  model: "${MODEL_NAME:-facebook/opt-125m}"
```

### Common Environment Variables

These environment variables are commonly used with auto-tune-vllm:

- **`POSTGRES_PASSWORD`**: Database password for PostgreSQL storage
- **`DATABASE_URL`**: Complete database connection URL  
- **`LOG_LEVEL`**: Logging verbosity (DEBUG/INFO/WARNING/ERROR)
- **`LOG_PATH`**: Directory for log files
- **`MODEL_NAME`**: HuggingFace model identifier
- **`BENCHMARK_DURATION`**: Benchmark runtime in seconds

Set these in your shell or deployment environment before running auto-tune-vllm.

## vLLM Environment Variables

Auto-tune-vllm supports passing environment variables to vLLM processes in two ways:

- **Environment parameters**: Add `type: environment` in the `parameters` section (list-only options required)
- **Static environment variables**: Use `static_environment_variables` section for consistent key-value pairs

```yaml
parameters:
  VLLM_ATTENTION_BACKEND:
    enabled: true
    type: environment
    options: ["FLASH_ATTN", "XFORMERS"]
    
static_environment_variables:
  VLLM_CACHE_ROOT: "/tmp/vllm_cache"
  VLLM_DEBUG: "0"
```

## Configuration Examples

### Basic Development Configuration

A minimal configuration for quick testing and development:

```yaml
study:
  prefix: "dev_test"

optimization:
  preset: "high_throughput"
  n_trials: 10

benchmark:
  model: "facebook/opt-125m"
  max_seconds: 60
  dataset: null
  prompt_tokens: 100
  output_tokens: 100

parameters:
  gpu_memory_utilization:
    enabled: true
  max_num_batched_tokens:
    enabled: true
```

### Production Configuration

A comprehensive production setup with PostgreSQL storage and multi-objective optimization:

```yaml
study:
  name: "production_optimization_v1"
  database_url: "postgresql://tuner:${POSTGRES_PASSWORD}@localhost:5432/optuna"

optimization:
  approach: "multi_objective"
  objectives:
    - metric: "output_tokens_per_second"
      direction: "maximize"
      percentile: "median"
    - metric: "request_latency"
      direction: "minimize"
      percentile: "p95"
  sampler: "nsga2"
  n_trials: 200

benchmark:
  model: "Qwen/Qwen3-30B-A3B-FP8"
  max_seconds: 300
  dataset: null
  prompt_tokens: 1000
  output_tokens: 1000
  rate: 100

logging:
  file_path: "/var/log/auto-tune-vllm"
  log_level: "INFO"

parameters:
  gpu_memory_utilization:
    enabled: true
    min: 0.85
    max: 0.95
    step: 0.01
  max_num_batched_tokens:
    enabled: true
  kv_cache_dtype:
    enabled: true
    options: ["auto", "fp8"]
```

### SQLite-Only Configuration  

A file-based configuration without PostgreSQL requirements:

```yaml
study:
  name: "local_optimization"
  storage_file: "/tmp/optimization/study.db"

optimization:
  preset: "balanced"
  n_trials: 50

benchmark:
  model: "facebook/opt-125m"
  max_seconds: 120
  dataset: null

logging:
  file_path: "/tmp/optimization/logs"
  log_level: "INFO"

parameters:
  gpu_memory_utilization:
    enabled: true
  max_num_batched_tokens:
    enabled: true
  kv_cache_dtype:
    enabled: true
```

---

## Configuration Validation

Before running optimization, validate your configuration file:

```bash
auto-tune-vllm validate --config your_config.yaml
```

The validation checks:
- YAML syntax correctness
- Required field presence  
- Parameter range and option validity
- Optimization configuration consistency
- Environment variable expansion
- Study naming conflicts

## Best Practices

1. **Start simple**: Begin with preset configurations before customizing
2. **Use meaningful names**: Choose descriptive study names for production
3. **Choose appropriate storage**: PostgreSQL for production, SQLite for development
4. **Scale trials appropriately**: 10-50 for development, 100+ for production
5. **Select relevant percentiles**: p95 for SLAs, median for general optimization  
6. **Enable file logging**: Essential for troubleshooting and result analysis
7. **Secure credentials**: Use environment variables for sensitive information
8. **Start with high-impact parameters**: Focus on memory and batching parameters first

## Quick Start Guide

The steps below help you set up the environment, validate your configuration, and launch an optimization study with Ray.

### Prerequisites

- Python 3.12 recommended
- NVIDIA GPU with recent drivers for GPU-backed studies (CPU runs are possible but limited)
- Internet access for pulling Python wheels and models

### 1) Create a Virtual Environment

Use either uv (recommended) or venv. Pick one.

```bash
# uv (installs Python if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 venv
source venv/bin/activate

# OR: venv
python3.12 -m venv venv
source venv/bin/activate
```

### 2) Install the Project (editable)

Install in editable mode so the CLI is available and source edits are reflected immediately.

```bash
# Using uv
uv pip install -e .

# OR: pip
pip install -e .
```

Verify the CLI is on PATH:

```bash
auto-tune-vllm --help
```

### 3) Configure the Study

Start from [`examples/study_config_local_exec.yaml`](../examples/study_config_local_exec.yaml) for a full example configuration file.

Key configuration areas:
- Set/confirm the study name and model ([Study Configuration](configuration.md#study-configuration))
- Choose the optimization objective(s) (e.g., throughput) ([Optimization Configuration](configuration.md#optimization-configuration))
- Adjust parameter ranges for tunables you want to explore ([Parameter Configuration](configuration.md#parameter-configuration))

Here's what a basic configuration looks like:

```yaml
# Basic study configuration
study:
  # name: "my_optimization_study"  # Optional: auto-generates if omitted

optimization:
  preset: "high_throughput"  # Use preset for common scenarios
  n_trials: 200             # Number of optimization trials

benchmark:
  benchmark_type: "guidellm"
  model: "RedHatAI/Qwen3-1.7B-FP8-dynamic"
  max_seconds: 240          # Benchmark duration per trial
  rate: 30                  # Request rate (req/sec)
  prompt_tokens: 2000       # Input length
  output_tokens: 2000       # Output length

logging:
  file_path: "/tmp/auto-tune-vllm-local-run/logs"
  log_level: "INFO"

parameters:
  gpu_memory_utilization:
    enabled: true
    min: 0.9
    max: 0.95
  kv_cache_dtype:
    enabled: true
    options: ["auto", "fp8"]
```



#### Using Optimization Presets (Recommended)

**Use presets for common optimization scenarios** - they're pre-configured for best results:

```yaml
optimization:
  preset: "high_throughput"  # Maximize token generation rate
  n_trials: 100
```

**Available presets:**
- `"high_throughput"`: Maximize output tokens per second
- `"low_latency"`: Minimize request latency (95th percentile)
- `"balanced"`: Multi-objective optimization (throughput vs latency)

**To customize or add presets:**
- Edit the `_apply_preset()` method in `auto_tune_vllm/core/config.py` (line 139)
- Add new preset conditions or modify existing ones

#### What Gets Optimized

The optimizer will tune parameters you mark as `enabled: true` in your config. Parameters you don't specify use vLLM defaults.

**Quick example:**
```yaml
parameters:
  gpu_memory_utilization:
    enabled: true
    min: 0.85
    max: 0.95
  # Other parameters use vLLM defaults
```

**To customize optimization ranges** (for better results after initial studies):
- Edit `auto_tune_vllm/schemas/v0_10_1_1.yaml` or `auto_tune_vllm/schemas/v0_10_0_0.yaml`
- Narrow ranges around promising values from previous runs
- **Don't change the `default` values** - only adjust `min`/`max`/`options`

> 💡 **Tip**: Start simple with 2-3 key parameters, then expand based on results. See [Parameter Configuration](configuration.md#parameter-configuration) for all available parameters.

### 4) Validate Configuration

Always validate before launching optimization:

```bash
auto-tune-vllm validate --config examples/study_config_local_exec.yaml
```

### 5) Run Optimization (Ray)

Ray workers must run in a Python environment you control. Provide one of:
- `--venv-path /absolute/path/to/venv`
- `--python-executable /absolute/path/to/python`
- `--conda-env <conda-env-name>`

Use an existing Ray cluster (recommended if one is already running):

```bash
auto-tune-vllm optimize \
  --config examples/study_config_local_exec.yaml \
  --venv-path "$(pwd)/venv" \
  --max-concurrent <count>
```

Start a new Ray head locally (when no cluster is running):

```bash
auto-tune-vllm optimize \
  --config examples/study_config_local_exec.yaml \
  --venv-path "$(pwd)/venv" \
  --max-concurrent <count> \
  --start-ray-head
```

Notes:
- If a Ray cluster is already running, omit `--start-ray-head` to avoid port conflicts.
- To manually start Ray (and then omit `--start-ray-head`):
  ```bash
  ray start --head --dashboard-host=0.0.0.0
  ```

### 6) Monitor Logs

Open a separate terminal. After optimization starts, the CLI prints an exact logs command. For file-based logging:

```bash
auto-tune-vllm logs --study-id <your_study_id> --log-path ./logs
```

If you configured PostgreSQL logging:

```bash
auto-tune-vllm logs --study-name <your_study_name> --database-url postgresql://user:pass@host:5432/db
```

> See [Logging Configuration](configuration.md#logging-configuration) for detailed logging options.

### View Study in Optuna Dashboard

Use the Optuna Dashboard Web UI (no local install needed):

- Open the dashboard in your browser: https://optuna.github.io/optuna-dashboard/#/
- After your study finishes, locate the SQLite file created by the run:
  - `optuna_studies/<study_name>/study.db`
- Drag-and-drop the `study.db` file into the dashboard page.

You can now explore optimization history, parameter importance, and parallel coordinates to see how configurations evolved over the study.

### Troubleshooting

- Error: "At least one Python environment option must be specified"
  - Provide one of `--venv-path`, `--python-executable`, or `--conda-env`.
  - Example:
    ```bash
    auto-tune-vllm optimize \
      --config examples/study_config_local_exec.yaml \
      --venv-path "$(pwd)/venv"
    ```

- Ray is already running on this port
  - A cluster is active. Either connect without `--start-ray-head`, or manually start Ray on a different port and then omit `--start-ray-head`.
  - Example manual start on another port:
    ```bash
    ray stop --force
    ray start --head --port=6380 --dashboard-host=0.0.0.0
    ```

- Ray version mismatch between cluster and local
  - Align your local Ray version with the cluster (or stop the cluster and start fresh):
    ```bash
    # Match local to cluster
    uv pip install "ray==<cluster_version>"
    # Or stop the old cluster and start a new one locally
    ray stop --force
    ray start --head --dashboard-host=0.0.0.0
    ```
- **Important**: Add `--max-concurrent <count>` or set `max_concurrent: <count>` in your YAML config.

### Next Steps

- Inspect resources on your cluster: `auto-tune-vllm check-env --ray-cluster`
- Calibrate concurrency based on available GPUs (e.g., 1 GPU per trial to start)
- Explore advanced configuration options: [Configuration Guide](configuration.md)

Project home: [Auto‑Tune vLLM (GitHub)](https://github.com/openshift-psap/auto-tuning-vllm/tree/main)

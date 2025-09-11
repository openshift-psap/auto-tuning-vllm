# Auto-Tune vLLM

[![PyPI version](https://badge.fury.io/py/auto-tune-vllm.svg)](https://badge.fury.io/py/auto-tune-vllm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A distributed hyperparameter optimization framework for vLLM serving, built with Ray and Optuna.

## Features

- 🚀 **Distributed Optimization**: Scale across multiple GPUs and nodes using Ray
- 🎯 **Flexible Backends**: Run locally or on Ray clusters  
- 📊 **Rich Benchmarking**: Built-in GuideLLM support + custom benchmark providers
- 🗄️ **Centralized Storage**: PostgreSQL for trials, metrics, and logs
- ⚙️ **Easy Configuration**: YAML-based study and parameter configuration
- 📈 **Multi-Objective**: Support for throughput vs latency trade-offs
- 🔧 **Extensible**: Plugin system for custom benchmarks

## Quick Start (5 minutes)
For a detailed starter guide, see the [Quick Start Guide](docs/quick_start.md).

### Installation

```bash
pip install auto-tune-vllm
```

### Basic Usage

#### CLI Interface
```bash
# Run optimization study
auto-tune-vllm optimize --config study_config.yaml

# Stream live logs  
auto-tune-vllm logs --study-id 42 --trial-number 15

# Resume interrupted study
auto-tune-vllm resume --study-id 42
```


## Documentation

- [Ray Cluster Setup](docs/ray_cluster_setup.md) - **Important for distributed optimization**
- [Configuration Reference](docs/configuration.md) 

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- PostgreSQL database

All ML dependencies (vLLM, Ray, GuideLLM, BoTorch) are included automatically.

## Known Issues

### Ray Cluster Concurrency Validation

**Issue**: The `--max-concurrent` parameter is not validated against available Ray cluster resources.

**Details**: When using Ray backend, the system doesn't check if the requested concurrency level is feasible given the cluster's GPU/CPU resources. For example, setting `--max-concurrent 10` on a cluster with only 4 GPUs will not warn the user that only 4 trials can actually run concurrently.

**Reason**: There is not a clear answer if all the trials would use the exact same number of GPUs. For example, we might have different parallelism related tunings for different trials which might result in different number of GPUs being required for the trial.

**Current Behavior**: 
- Excess trials are queued by Ray until resources become available
- No warning or guidance is provided to users
- May lead to confusion about why trials aren't running as expected

**Workaround**: 
- Use `auto-tune-vllm check-env --ray-cluster` to inspect available resources
- Set concurrency based on available GPUs (typically 1 GPU per trial)
- Monitor Ray dashboard at `http://<head-node>:8265` for resource utilization

**Example**:
```bash
# Check cluster resources first
auto-tune-vllm check-env --ray-cluster

# Set realistic concurrency (e.g., if you have 4 GPUs)
auto-tune-vllm optimize --config study.yaml --max-concurrent 4
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.
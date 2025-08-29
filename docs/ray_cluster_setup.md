# Ray Cluster Setup for auto-tune-vllm

This guide explains how to set up a Ray cluster for distributed vLLM optimization.

## Overview

Ray **does NOT automatically install dependencies** on worker nodes. Each node in your Ray cluster must have the same Python environment with `auto-tune-vllm` installed.

**New**: auto-tune-vllm now supports configuring Python environments for Ray workers! This solves the common issue where Ray workers use different Python installations than the head node.

## Prerequisites

### Hardware Requirements
- **Head Node**: CPU for orchestration, can be GPU-less
- **Worker Nodes**: NVIDIA GPUs with CUDA support for vLLM serving
- **Network**: Fast interconnect between nodes (10GbE+ recommended)

### Software Requirements
All nodes must have:
- Python 3.10+
- NVIDIA drivers and CUDA toolkit
- Same Python environment with auto-tune-vllm installed

## Setup Instructions

### 1. Install auto-tune-vllm on All Nodes

**On every node** (head and workers):

```bash
# Install auto-tune-vllm (includes all dependencies)
pip install auto-tune-vllm

# Verify installation
auto-tune-vllm check-env
```

### 2. Start Ray Cluster

#### Head Node
```bash
# Start Ray head node
ray start --head --port=10001 --dashboard-host=0.0.0.0

# Note the connection command shown (ray start --address=...)
```

#### Worker Nodes
```bash
# On each worker node, connect to head node
ray start --address=<head_node_ip>:10001

# Verify worker joined
ray status  # Run on head node
```

### 3. Verify Cluster Environment

After setting up the cluster, verify all nodes have the required dependencies:

```bash
# Check local environment
auto-tune-vllm check-env

# Check entire Ray cluster
auto-tune-vllm check-env --ray-cluster
```

### 4. Run Distributed Optimization

#### Basic Usage
```bash
# Run optimization with Ray backend
auto-tune-vllm optimize --config study.yaml --backend ray
```

#### Python Environment Configuration

To ensure Ray workers use the correct Python environment, use one of these options:

##### Option 1: Virtual Environment Path
```bash
# Specify virtual environment directory
auto-tune-vllm optimize --config study.yaml --backend ray --venv-path /path/to/your/venv

# Example with common venv locations
auto-tune-vllm optimize --config study.yaml --backend ray --venv-path ./venv
auto-tune-vllm optimize --config study.yaml --backend ray --venv-path ~/.venvs/myproject
```

##### Option 2: Explicit Python Executable
```bash
# Specify exact Python executable
auto-tune-vllm optimize --config study.yaml --backend ray --python-executable /path/to/python

# Examples
auto-tune-vllm optimize --config study.yaml --backend ray --python-executable ./venv/bin/python
auto-tune-vllm optimize --config study.yaml --backend ray --python-executable /usr/bin/python3.11
```

##### Option 3: Conda Environment
```bash
# Specify conda environment name
auto-tune-vllm optimize --config study.yaml --backend ray --conda-env myenv

# Example with conda environment
auto-tune-vllm optimize --config study.yaml --backend ray --conda-env auto-tune-vllm
```

##### Option 4: Auto-Detection (Default)
```bash
# No options - auto-detects current virtual environment
auto-tune-vllm optimize --config study.yaml --backend ray
```

**Note**: Only specify one Python environment option at a time. The system will validate this and show an error if multiple options are provided.

## Common Issues and Solutions

### Issue: "Missing required packages on Ray worker node"

**Cause**: Worker node doesn't have auto-tune-vllm installed

**Solution**:
```bash
# On the problematic worker node
pip install auto-tune-vllm

# Restart the worker
ray stop
ray start --address=<head_node_ip>:10001
```

### Issue: Ray workers using wrong Python installation

**Symptoms**: 
- ImportError for packages that are installed in your environment
- Workers can't find vLLM or other dependencies
- Different Python versions on workers vs head node

**Solution**: Use Python environment configuration options

```bash
# If you're using a virtual environment
auto-tune-vllm optimize --config study.yaml --backend ray --venv-path ./venv

# If you have a specific Python executable
auto-tune-vllm optimize --config study.yaml --backend ray --python-executable /path/to/python

# If you're using conda
auto-tune-vllm optimize --config study.yaml --backend ray --conda-env myenv
```

### Issue: "No Python executable found in venv"

**Cause**: Invalid virtual environment path or corrupted venv

**Solution**:
```bash
# Verify the venv exists and has Python
ls /path/to/venv/bin/python*

# If missing, recreate the virtual environment
python -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install auto-tune-vllm

# Use explicit Python path instead
auto-tune-vllm optimize --config study.yaml --backend ray --python-executable /path/to/venv/bin/python
```

### Issue: Auto-detection not working

**Symptoms**: Warning about "not in a virtual environment"

**Solution**: Explicitly specify your Python environment
```bash
# Check current Python
which python
python --version

# Use explicit path
auto-tune-vllm optimize --config study.yaml --backend ray --python-executable $(which python)
```

### Issue: "No CUDA GPUs detected on Ray worker"

**Cause**: GPU drivers not installed or not accessible

**Solution**:
```bash
# Check GPU availability on worker node
nvidia-smi

# If no GPUs shown, install NVIDIA drivers
# If GPUs shown but not accessible to Python:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "guidellm command not found"

**Cause**: GuideLLM not properly installed or not in PATH

**Solution**:
```bash
# Reinstall auto-tune-vllm to ensure all CLI tools are available
pip install --force-reinstall auto-tune-vllm

# Verify guidellm is available
which guidellm
guidellm --help
```

### Issue: Ray cluster connection fails

**Cause**: Network connectivity or firewall issues

**Solution**:
```bash
# Check if head node port is accessible
telnet <head_node_ip> 10001

# If connection fails, check firewall settings
# Allow Ray ports (10001, 8265 for dashboard)
sudo ufw allow 10001
sudo ufw allow 8265
```

## Performance Optimization

### Resource Allocation

Configure Ray to properly utilize GPUs:

```python
# In your study config or when creating backend
backend = RayExecutionBackend({
    "num_gpus": 1,        # 1 GPU per trial
    "num_cpus": 4,        # 4 CPUs per trial  
    "memory": 8_000_000_000  # 8GB RAM per trial
})
```

### Node Placement

For optimal performance:
- **Head node**: CPU-only machine for orchestration
- **Worker nodes**: GPU machines with sufficient VRAM for your models
- **Separate database**: Run PostgreSQL on dedicated machine if possible

### Cluster Scaling

Ray supports dynamic scaling:

```bash
# Add more worker nodes anytime
ray start --address=<head_node_ip>:10001

# Remove worker nodes
ray stop  # Run on worker node
```

## Environment Variables

Set these on all nodes for consistent behavior:

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Available GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Ray settings  
export RAY_TMPDIR=/tmp/ray  # Temp directory
export RAY_ADDRESS=<head_node_ip>:10001  # For workers
```

## Monitoring

### Ray Dashboard
Access at: `http://<head_node_ip>:8265`

### Resource Usage
```bash
# Check cluster status
ray status

# Monitor resource utilization
watch -n 5 'ray status'
```

### Logs
```bash
# View Ray logs
ray logs

# View auto-tune-vllm logs
auto-tune-vllm logs --study-id <id> --database-url <postgres_url>
```

## Security Considerations

### Network Security
- Use private networks for Ray cluster communication
- Restrict Ray dashboard access (port 8265)
- Enable authentication if needed

### Resource Isolation
- Consider containerization (Docker/Kubernetes) for better isolation
- Use Ray's resource allocation to prevent resource conflicts

## Troubleshooting

### Debug Mode
Run with verbose logging:
```bash
auto-tune-vllm optimize --config study.yaml --backend ray --verbose
```

### Environment Validation
Always run environment checks before starting optimization:
```bash
auto-tune-vllm check-env --ray-cluster --verbose
```

### Clean Restart
If issues persist:
```bash
# On all nodes
ray stop
killall -9 raylet  # Force kill if needed

# Restart cluster
# Head: ray start --head --port=10001
# Workers: ray start --address=<head_node_ip>:10001
```
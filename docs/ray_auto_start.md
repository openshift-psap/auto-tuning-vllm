# Ray Auto-Start Feature

## Overview

The `--start-ray-head` option allows auto-tune-vllm to automatically start a Ray head node when no existing Ray cluster is detected. This simplifies deployment for single-machine setups and development environments.

## Usage

```bash
# Auto-start Ray head if no cluster found
auto-tune-vllm optimize --config study.yaml --backend ray --start-ray-head

# Traditional usage (requires existing Ray cluster)
auto-tune-vllm optimize --config study.yaml --backend ray
```

## How It Works

1. **Detection**: When using Ray backend, the system first attempts to connect to an existing Ray cluster
2. **Fallback**: If no cluster is found and `--start-ray-head` is enabled, automatically starts a Ray head
3. **Connection**: Connects to the newly started Ray head using auto-discovery
4. **Cleanup**: Automatically stops the Ray head when optimization completes

## Technical Implementation

### CLI Changes
- Added `--start-ray-head` boolean option to the `optimize` command
- Option is passed to `RayExecutionBackend` constructor

### Backend Changes
- `RayExecutionBackend` now accepts `start_ray_head` parameter
- Enhanced Ray initialization logic with fallback behavior
- Automatic Ray head lifecycle management

### Ray Head Configuration
When auto-starting, the Ray head is configured with:
- Ports: Automatically chosen by Ray (avoids conflicts)
- Dashboard: Enabled on all interfaces (0.0.0.0)
- Connection: Uses Ray's auto-discovery mechanism

## Error Handling

### Without `--start-ray-head`
```
Failed to connect to Ray cluster: [connection error]
Use --start-ray-head to automatically start a Ray head, or start one manually:
  ray start --head
```

### With `--start-ray-head`
- Automatically attempts to start Ray head
- Provides detailed error messages if Ray head startup fails
- Graceful cleanup on both success and failure

## Logging

The feature provides comprehensive logging:
```
INFO: No existing Ray cluster found, starting Ray head...
INFO: Starting Ray head with command: ray start --head --dashboard-host=0.0.0.0
INFO: Started Ray head successfully
INFO: Connected to newly started Ray head
...
INFO: Stopping Ray head that we started...
INFO: Successfully stopped Ray head
```

## Use Cases

### Development Environment
```bash
# Quick start for local development
auto-tune-vllm optimize --config dev-study.yaml --start-ray-head
```

### Single Machine Deployment
```bash
# Production single-machine setup
auto-tune-vllm optimize --config production.yaml --backend ray --start-ray-head
```

### CI/CD Pipelines
```bash
# Automated testing with ephemeral Ray cluster
auto-tune-vllm optimize --config test-study.yaml --start-ray-head --trials 5
```

## Considerations

### Resource Requirements
- Ray head requires minimal resources for orchestration
- Actual compute happens on worker nodes (or head if single-machine)
- Consider available CPU/memory when using single-machine setup

### Network Configuration
- Auto-started Ray head uses automatically chosen ports
- Dashboard available on `http://localhost:8265` (default)
- For multi-machine clusters, still requires manual Ray cluster setup

### Cleanup Behavior
- Ray head is automatically stopped when optimization completes
- Force cleanup attempted if graceful shutdown fails
- Temporary Ray files cleaned up in `/tmp/ray`

## Limitations

1. **Single Machine Only**: Auto-start only creates head node on localhost
2. **Fixed Configuration**: Uses default Ray head settings
3. **No Worker Auto-Start**: Additional workers must be added manually

## Troubleshooting

### Ray Processes Already Running
```bash
# Check for existing Ray processes
ray status

# Stop existing Ray processes if needed
ray stop
```

### Permission Issues
```bash
# Ensure Ray command is in PATH
which ray

# Check write permissions for temp directory
ls -la /tmp/ray
```

### Manual Cleanup
```bash
# If auto-cleanup fails
ray stop
pkill -f "ray::"
rm -rf /tmp/ray
```

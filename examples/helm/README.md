# Helm Backend Example

This directory contains a complete example for using the Helm backend with auto-tune-vllm.

## Contents

- **`Chart.yaml`**: Helm chart metadata
- **`values.yaml`**: Default Helm values template (includes PostgreSQL configuration)
- **`templates/`**: Helm chart templates (Deployment, Service, PostgreSQL StatefulSet, helpers)
- **`study_config.yaml`**: Example study configuration using Helm backend
- **`postgres-standalone.yaml`**: Standalone PostgreSQL StatefulSet manifest
- **`deploy-postgres.sh`**: Script to deploy PostgreSQL using Helm
- **`deploy-postgres-standalone.sh`**: Script to deploy PostgreSQL using standalone YAML
- **`postgres-test-pod.yaml`**: Test pod manifest for standalone PostgreSQL deployment
- **`postgres-test-pod-helm.yaml`**: Test pod manifest for Helm PostgreSQL deployment
- **`test-postgres.sh`**: Automated test script for PostgreSQL connectivity

## Step-by-Step Guide

Follow these steps to run your first optimization using the Helm backend.

### Step 1: Prerequisites

Ensure you have the following installed and configured:

1. **Kubernetes cluster** with GPU nodes accessible
2. **Helm CLI** (v3.0+):
   ```bash
   # Check if Helm is installed
   helm version
   
   # If not installed, install Helm:
   # https://helm.sh/docs/intro/install/
   ```

3. **kubectl** configured to access your cluster:
   ```bash
   # Verify cluster access
   kubectl cluster-info
   kubectl get nodes
   ```

4. **auto-tune-vllm with Helm support**:
   ```bash
   pip install 'auto-tune-vllm[helm]'
   ```

### Step 2: Prepare Kubernetes Cluster

1. **Create namespace** for your trials:
   ```bash
   kubectl create namespace llm-d-trials
   ```

2. **Verify GPU nodes** are available:
   ```bash
   # Check for GPU nodes
   kubectl get nodes -l accelerator=nvidia-tesla-v100  # Adjust label as needed
   
   # Or check all nodes for GPU resources
   kubectl describe nodes | grep nvidia.com/gpu
   ```

3. **Create HuggingFace token secret** (if your model requires authentication):
   ```bash
   kubectl create secret generic llm-d-hf-token \
     --from-literal=HF_TOKEN=your_huggingface_token_here \
     -n llm-d-trials
   ```
   
   **Note**: If your model is public, you can skip this step.

4. **Deploy PostgreSQL StatefulSet** (if using PostgreSQL for study storage):
   
   This Helm chart includes PostgreSQL StatefulSet templates with NFS-backed persistent storage.
   
   **Option A: Deploy using Helm (recommended)**:
   ```bash
   # export KUBECONFIG=/path/to/your/kubeconfig  # Optional: use if not using default ~/.kube/config
   cd examples/helm
   ./deploy-postgres.sh [release-name]
   ```
   
   **Option B: Deploy using standalone YAML**:
   ```bash
   # export KUBECONFIG=/path/to/your/kubeconfig  # Optional: use if not using default ~/.kube/config
   cd examples/helm
   ./deploy-postgres-standalone.sh
   ```
   
   **Option C: Manual deployment with Helm**:
   ```bash
   # export KUBECONFIG=/path/to/your/kubeconfig  # Optional: use if not using default ~/.kube/config
   helm upgrade --install postgresql examples/helm \
     --namespace llm-d-trials \
     --set postgresql.enabled=true \
     --set postgresql.namespace=llm-d-trials \
     --set postgresql.persistence.storageClass=nfs-storage \
     --wait --timeout 10m
   ```
   
   **Option D: Manual deployment with kubectl**:
   ```bash
   # export KUBECONFIG=/path/to/your/kubeconfig  # Optional: use if not using default ~/.kube/config
   kubectl apply -f examples/helm/postgres-standalone.yaml
   ```
   
   **Verify PostgreSQL is running**:
   ```bash
   kubectl get statefulset -n llm-d-trials
   kubectl get pods -n llm-d-trials -l app=postgresql
   kubectl get pvc -n llm-d-trials
   ```
   
   **Get connection details**:
   ```bash
   # For Helm deployment
   kubectl get secret <release-name>-postgresql-secret -n llm-d-trials \
     -o jsonpath='{.data.postgres-user}' | base64 -d && echo
   kubectl get secret <release-name>-postgresql-secret -n llm-d-trials \
     -o jsonpath='{.data.postgres-password}' | base64 -d && echo
   
   # For standalone deployment
   kubectl get secret postgresql-secret -n llm-d-trials \
     -o jsonpath='{.data.postgres-user}' | base64 -d && echo
   kubectl get secret postgresql-secret -n llm-d-trials \
     -o jsonpath='{.data.postgres-password}' | base64 -d && echo
   ```
   
   **Service endpoint**:
   - Helm: `<release-name>-vllm-server-postgresql.llm-d-trials.svc.cluster.local:5432`
   - Standalone: `postgresql.llm-d-trials.svc.cluster.local:5432`
   
   **Note**: Default credentials are `postgres/postgres`. Change these in production!

5. **Test PostgreSQL deployment** (optional but recommended):
   
   **Option A: Automated test script** (recommended):
   ```bash
   # export KUBECONFIG=/path/to/your/kubeconfig  # Optional: use if not using default ~/.kube/config
   cd examples/helm
   
   # For standalone deployment
   ./test-postgres.sh postgresql
   
   # For Helm deployment (adjust service name based on release name)
   ./test-postgres.sh <release-name>-vllm-server-postgresql
   ```
   
   The test script will:
   - Verify PostgreSQL service exists
   - Check pod readiness
   - Test connectivity with `pg_isready`
   - Test database connection
   - Create and query a test table
   - List available databases
   
   **Option B: Manual test pod**:
   ```bash
   # export KUBECONFIG=/path/to/your/kubeconfig  # Optional: use if not using default ~/.kube/config
   
   # For standalone deployment
   kubectl apply -f examples/helm/postgres-test-pod.yaml
   
   # For Helm deployment (update service/secret names in the file first)
   kubectl apply -f examples/helm/postgres-test-pod-helm.yaml
   
   # Wait for pod to be ready
   kubectl wait --for=condition=ready pod postgres-test-pod -n llm-d-trials --timeout=60s
   
   # Connect to the test pod
   kubectl exec -it postgres-test-pod -n llm-d-trials -- /bin/bash
   
   # Inside the pod, test PostgreSQL
   pg_isready -h postgresql.llm-d-trials.svc.cluster.local -p 5432 -U postgres
   psql -h postgresql.llm-d-trials.svc.cluster.local -U postgres -d postgres -c "SELECT version();"
   
   # Clean up test pod when done
   kubectl delete pod postgres-test-pod -n llm-d-trials
   ```
   
   **Option C: Quick connectivity test**:
   ```bash
   # Create a temporary test pod
   kubectl run postgres-test --image=postgres:15 --rm -it --restart=Never \
     --namespace=llm-d-trials \
     --env="PGHOST=postgresql.llm-d-trials.svc.cluster.local" \
     --env="PGUSER=postgres" \
     --env="PGPASSWORD=postgres" \
     -- psql -h postgresql.llm-d-trials.svc.cluster.local -U postgres -d postgres -c "SELECT version();"
   ```

### Step 3: Configure Study

1. **Navigate to the example directory**:
   ```bash
   cd examples/helm
   ```

2. **Edit `study_config.yaml`**:
   
   a. **Set database URL** (or use file-based storage):
      ```yaml
      study:
        name: "helm_optimization_example"
        database_url: "postgresql://user:password@localhost:5432/optuna"
      ```
      
      **Alternative**: Remove `database_url` to use file-based storage:
      ```yaml
      study:
        name: "helm_optimization_example"
        # Will use: ./optuna_studies/helm_optimization_example/study.db
      ```

   b. **Configure Helm backend**:
      
      **REQUIRED**: You must explicitly specify `chart_type`:
      - `"vllm"`: For vanilla vLLM charts (all args provided in container.args)
      - `"llm-d-modelservice"`: For llm-d-modelservice charts (chart constructs args from fields)
      
      **Option 1: Use local chart (vanilla vLLM)**:
      ```yaml
      execution:
        backend: "helm"
        helm:
          chart_type: "vllm"  # REQUIRED: Explicit chart type
          chart_path: "./examples/helm"
          namespace: "llm-d-trials"
          benchmark_image: "guidellm:latest"
      ```
      
      **Option 2: Use llm-d-modelservice chart from repository**:
      ```yaml
      execution:
        backend: "helm"
        helm:
          chart_type: "llm-d-modelservice"  # REQUIRED: Explicit chart type
          chart_repo: "https://llm-d-incubation.github.io/llm-d-modelservice/"
          chart_name: "llm-d-modelservice"
          chart_version: "v0.3.8"
          namespace: "llm-d-trials"
          benchmark_image: "guidellm:latest"
      ```
      
      **Important**: If `chart_type` is not specified, auto-tune-vllm will fail with an error.

   c. **Adjust model** (if needed):
      ```yaml
      benchmark:
        model: "Qwen/Qwen3-0.6B"  # Use a model that fits your GPU memory
      ```

   d. **Set optimization parameters**:
      ```yaml
      optimization:
        n_trials: 20  # Start with fewer trials for testing
        max_concurrent_trials: 2  # Adjust based on cluster capacity
      ```

   e. **Configure tunable vLLM parameters**:
      ```yaml
      parameters:
        max_num_seqs:
          enabled: true
          min: 32
          max: 256
          step: 32
        gpu_memory_utilization:
          enabled: true
          min: 0.8
          max: 0.95
          step: 0.05
      ```
      
      These parameters will be converted to vLLM CLI arguments (e.g., `--max-num-seqs`, `--gpu-memory-utilization`) and added to the Helm chart's `decode.containers[0].args` field.

### Step 4: Validate Configuration

1. **Test Helm chart** (if using local chart):
   ```bash
   helm lint examples/helm
   ```

2. **Dry-run Helm install** (optional):
   ```bash
   helm install test-release examples/helm \
     --namespace llm-d-trials \
     --dry-run \
     --debug
   ```

3. **Validate study config**:
   ```bash
   auto-tune-vllm optimize \
     --config examples/helm/study_config.yaml \
     --backend helm \
     --dry-run  # If supported, otherwise just check for errors
   ```

### Step 5: Run Optimization

1. **Start optimization**:
   ```bash
   # From project root
   auto-tune-vllm optimize \
     --config examples/helm/study_config.yaml \
     --backend helm
   ```

2. **Monitor progress**:
   
   In another terminal, watch Kubernetes resources:
   ```bash
   # Watch Helm releases
   watch -n 5 'helm list -n llm-d-trials'
   
   # Watch deployments
   kubectl get deployments -n llm-d-trials -w
   
   # Watch jobs
   kubectl get jobs -n llm-d-trials -w
   
   # Watch pods
   kubectl get pods -n llm-d-trials -w
   ```

3. **Check logs**:
   ```bash
   # View optimization logs
   tail -f /tmp/auto-tune-vllm-logs/optimization.log
   
   # View specific trial logs
   kubectl logs -n llm-d-trials <pod-name> -f
   ```

### Step 6: Verify Results

1. **Check trial results**:
   ```bash
   # If using PostgreSQL
   psql -h localhost -U user -d optuna -c \
     "SELECT trial_id, value FROM trial_values ORDER BY trial_id;"
   
   # If using file storage
   sqlite3 ./optuna_studies/helm_optimization_example/study.db \
     "SELECT trial_id, value FROM trial_values;"
   ```

2. **Inspect Helm releases**:
   ```bash
   # List all releases
   helm list -n llm-d-trials
   
   # Get release details
   helm get manifest <release-name> -n llm-d-trials
   
   # View release values
   helm get values <release-name> -n llm-d-trials
   ```

3. **Check for completed jobs**:
   ```bash
   kubectl get jobs -n llm-d-trials
   kubectl logs job/<job-name> -n llm-d-trials
   ```

### Step 7: Cleanup (Optional)

After optimization completes, you can clean up resources:

1. **Delete Helm releases** (if not auto-cleaned):
   ```bash
   helm list -n llm-d-trials
   helm uninstall <release-name> -n llm-d-trials
   ```

2. **Delete completed jobs** (they auto-delete after TTL, but you can force):
   ```bash
   kubectl delete jobs --all -n llm-d-trials
   ```

3. **Delete namespace** (if you want to remove everything):
   ```bash
   kubectl delete namespace llm-d-trials
   ```

## Quick Start (Summary)

For experienced users, here's the condensed version:

```bash
# 1. Install dependencies
pip install 'auto-tune-vllm[helm]'

# 2. Create namespace
kubectl create namespace llm-d-trials

# 3. Configure study_config.yaml (database URL, model, etc.)

# 4. Run optimization
auto-tune-vllm optimize \
  --config examples/helm/study_config.yaml \
  --backend helm
```

### Using a Chart from Repository

To use a chart from a Helm repository instead of the local chart:

1. Update `study_config.yaml`:
   ```yaml
   execution:
     backend: "helm"
     helm:
       chart_type: "llm-d-modelservice"  # REQUIRED: Must specify chart type
       chart_repo: "https://llm-d-incubation.github.io/llm-d-modelservice/"
       chart_name: "llm-d-modelservice"
       chart_version: "v0.3.8"
       namespace: "llm-d-trials"
   ```

2. Comment out the `chart_path` line

**Important**: The `chart_type` field is **REQUIRED** and must be explicitly set to either:
- `"vllm"`: For vanilla vLLM charts
- `"llm-d-modelservice"`: For llm-d-modelservice charts

If `chart_type` is not specified, auto-tune-vllm will fail with a clear error message.

**Note**: The `llm-d-modelservice` chart supports `modelCommand` which can be:
- `"vllmServe"`: Uses `command: ["vllm", "serve"]` (default, set automatically)
- `"imageDefault"`: Uses image's default entrypoint
- `"custom"`: Uses user-provided command

The auto-tune-vllm Helm backend automatically sets `modelCommand: "vllmServe"` when generating values for llm-d-modelservice charts.

## Preparing Your Helm Chart for Auto-Tune

To use a Helm chart with auto-tune-vllm, the chart must be structured to accept vLLM parameters that will be optimized. Here's how to prepare your chart:

### Required Chart Structure

Your Helm chart must support the following structure in `values.yaml`:

```yaml
decode:
  create: true
  containers:
    - name: vllm
      image: "your-vllm-image:tag"
      modelCommand: "vllmServe"  # or "imageDefault" or "custom"
      args: []  # This will be populated by auto-tune-vllm
      env: []   # Environment variables (can be optimized)
      resources:
        limits:
          nvidia.com/gpu: "1"
        requests:
          nvidia.com/gpu: "1"
```

### Chart Type Specification (REQUIRED)

**You must explicitly specify the chart type** in your Helm configuration:

```yaml
execution:
  backend: "helm"
  helm:
    chart_type: "llm-d-modelservice"  # or "vllm" - REQUIRED
    # ... other config
```

**Why?** Different chart types handle vLLM arguments differently:
- **`"llm-d-modelservice"`**: Chart constructs vLLM args from `parallelism`, `modelArtifacts`, etc.
- **`"vllm"`**: All vLLM args must be provided directly in `container.args`

If `chart_type` is not specified, auto-tune-vllm will fail with an error.

### How Auto-Tune Maps Parameters to Helm Values

Auto-tune-vllm converts trial parameters to Helm values based on the specified `chart_type`:

#### Standard vLLM Parameters (Auto-Mapped)

**Important**: The behavior differs between vanilla vLLM charts and llm-d-modelservice charts:

**For Vanilla vLLM Charts:**
1. **vLLM CLI Parameters** → `decode.containers[0].args`
   - All parameters defined in `study_config.yaml` under `parameters:` section
   - Converted from `snake_case` to `--kebab-case` format
   - Example: `max_num_seqs: 256` → `["--max-num-seqs", "256"]`
   - All args are added directly to `container.args`

**For llm-d-modelservice Charts:**
1. **Chart constructs standard vLLM args automatically** from:
   - `modelArtifacts.uri` → `--model` argument
   - `decode.parallelism.tensor` → `--tensor-parallel-size`
   - `decode.parallelism.data` → `--data-parallel-size`
   - `decode.parallelism.dataLocal` → `--data-parallel-size-local`
   - `modelArtifacts.name` → `--served-model-name`
   - Port configuration → `--port`
   
2. **Non-standard vLLM parameters** → `decode.containers[0].args` (appended)
   - Parameters like `max_num_seqs`, `gpu_memory_utilization`, etc.
   - These are appended to the chart-constructed args
   - Example: `max_num_seqs: 256` → `["--max-num-seqs", "256"]` (added to args)
   
3. **Parallelism parameters** → `decode.parallelism.*` (NOT in args)
   - `tensor_parallel_size` → `decode.parallelism.tensor`
   - `data_parallel_size` → `decode.parallelism.data`
   - Chart uses these to construct `--tensor-parallel-size` and `--data-parallel-size` args

2. **Environment Variables** → `decode.containers[0].env`
   - Parameters marked as environment variables
   - Static env vars from `static_environment_variables`
   - Example: `VLLM_LOGGING_LEVEL: "INFO"`

3. **Model Name** → `decode.containers[0].args` (--model flag) + `modelArtifacts`
   - Automatically added from `benchmark.model` config
   - Example: `["--model", "Qwen/Qwen3-0.6B"]` + `modelArtifacts.uri: "hf://Qwen/Qwen3-0.6B"`

4. **Resource Requirements** → `decode.containers[0].resources`
   - GPU requirements calculated from parallelism parameters
   - Automatically set based on `tensor_parallel_size`, etc.

#### llm-d-modelservice Specific Fields (Custom Mapping Required)

For llm-d-modelservice charts, you can optimize additional fields beyond vLLM args. These require custom parameter names or extending the Helm values generation:

1. **Parallelism Configuration** → `decode.parallelism.*`
   - `tensor_parallel_size` → `decode.parallelism.tensor`
   - `data_parallel_size` → `decode.parallelism.data`
   - `data_parallel_size_local` → `decode.parallelism.dataLocal`
   - `workers` → `decode.parallelism.workers`

2. **Replicas** → `decode.replicas`
   - Number of pod replicas for horizontal scaling

3. **Container Resources** → `decode.containers[0].resources.*`
   - CPU, memory limits/requests beyond GPU count
   - Resource claims for DRA (Dynamic Resource Allocation)

4. **Routing Proxy Settings** → `routing.proxy.*`
   - `routing.proxy.enabled`
   - `routing.proxy.targetPort`
   - `routing.proxy.connector`

5. **Model Artifacts** → `modelArtifacts.*`
   - `modelArtifacts.size` (volume size)
   - `modelArtifacts.mountPath`
   - `modelArtifacts.mountModelVolume`

6. **Node Selectors** → `decode.nodeSelector`
   - Node affinity for pod placement

7. **Volumes** → `decode.volumes`
   - Additional volumes for model storage or configuration

**Note**: To optimize these fields, you'll need to either:
- Use parameter names that match the Helm values path (see "Custom Field Mapping" below)
- Extend `generate_helm_values()` in `helm_utils.py` to map custom parameters
- Use a values template with placeholders that get replaced

### Step-by-Step: Making Fields Tunable

#### Step 1: Identify Tunable Parameters

Decide which vLLM parameters you want to optimize. Common tunable parameters include:

- `max_num_seqs`: Maximum number of sequences to process
- `max_num_batched_tokens`: Maximum batched tokens
- `gpu_memory_utilization`: GPU memory usage (0.0-1.0)
- `max_model_len`: Maximum model length
- `tensor_parallel_size`: Tensor parallelism
- `pipeline_parallel_size`: Pipeline parallelism
- `enable_cuda_graphs`: Enable CUDA graphs (boolean)
- `kv_cache_dtype`: KV cache data type (e.g., "auto", "fp8")

#### Step 2: Specify Chart Type

**REQUIRED**: Set `chart_type` in your Helm configuration:

```yaml
execution:
  backend: "helm"
  helm:
    chart_type: "llm-d-modelservice"  # or "vllm"
    # ... rest of config
```

#### Step 3: Configure Parameters in study_config.yaml

Add parameters to optimize in your `study_config.yaml`:

```yaml
parameters:
  max_num_seqs:
    enabled: true
    min: 32
    max: 256
    step: 32
    
  gpu_memory_utilization:
    enabled: true
    min: 0.8
    max: 0.95
    step: 0.05
    
  max_num_batched_tokens:
    enabled: true
    min: 1024
    max: 8192
    step: 1024
    
  kv_cache_dtype:
    enabled: true
    options: ["auto", "fp8"]
    
  enable_cuda_graphs:
    enabled: true
    # Boolean parameter - will be True/False
```

#### Step 4: Ensure Chart Template Accepts Args

Your deployment template must pass `args` to the container. Example:

```yaml
# templates/deployment.yaml
containers:
  - name: {{ .container.name }}
    image: {{ .container.image }}
    {{- if eq .container.modelCommand "vllmServe" }}
    command: ["vllm", "serve"]
    {{- end }}
    args:
      {{- range .container.args }}
      - {{ . | quote }}
      {{- end }}
```

#### Step 5: Set Static (Non-Tunable) Parameters

Parameters that should be the same for all trials go in `static_parameters`:

```yaml
# study_config.yaml
static_parameters:
  tensor_parallel_size: 1  # Fixed for all trials
  max_model_len: 16384      # Fixed model length
  port: 8000                 # Fixed port
```

These will be added to `args` for every trial.

#### Step 6: Configure Model and Resources

Set the model in `benchmark` section (not in Helm values):

```yaml
benchmark:
  model: "Qwen/Qwen3-0.6B"  # Model to optimize
```

The model will automatically be added as `--model` argument.

### Example: Complete Values Template

Here's a complete example `values.yaml` ready for auto-tune:

```yaml
# values.yaml - Template for auto-tune-vllm

modelArtifacts:
  uri: ""  # Will be set from benchmark.model
  name: ""  # Will be set from benchmark.model
  authSecretName: ""  # Optional: HF token secret

decode:
  create: true
  replicas: 1
  containers:
    - name: vllm
      image: "ghcr.io/vllm-project/vllm-openai:latest"
      modelCommand: "vllmServe"  # auto-tune will set this
      
      # These will be populated by auto-tune-vllm:
      args: []  # vLLM CLI arguments from trial parameters
      env: []   # Environment variables
      
      ports:
        - containerPort: 8000
          name: http
          protocol: TCP
      
      resources:
        limits:
          nvidia.com/gpu: "1"  # Will be adjusted based on parallelism
        requests:
          nvidia.com/gpu: "1"
      
      livenessProbe:
        httpGet:
          path: /health
          port: http
        periodSeconds: 10
      
      readinessProbe:
        httpGet:
          path: /health
          port: http
        periodSeconds: 5

service:
  type: ClusterIP
  port: 8000
  targetPort: 8000
```

### How Values Are Merged

Auto-tune-vllm merges values in this order:

1. **Base template** (if `values_template` is specified)
2. **Trial-specific values**:
   - vLLM args from `parameters` section
   - Model from `benchmark.model`
   - Environment variables
   - Resource requirements

The merged values are written to a temporary file and passed to `helm install --values`.

### Advanced: Optimizing llm-d-modelservice Specific Fields

#### Method 1: Direct Parameter Mapping (Recommended for llm-d-modelservice)

For llm-d-modelservice charts, use parameter names that match vLLM parameter names. The code automatically maps them:

```yaml
# study_config.yaml
parameters:
  # These automatically map to decode.parallelism.* for llm-d-modelservice
  tensor_parallel_size:  # Maps to decode.parallelism.tensor
    enabled: true
    min: 1
    max: 4
    step: 1
  
  data_parallel_size:  # Maps to decode.parallelism.data
    enabled: true
    min: 1
    max: 2
    step: 1
  
  # Standard vLLM parameters (added to container.args, appended after chart-constructed args)
  max_num_seqs:
    enabled: true
    min: 32
    max: 256
    step: 32
  
  gpu_memory_utilization:
    enabled: true
    min: 0.8
    max: 0.95
    step: 0.05
```

**Note**: The `generate_helm_values()` function automatically:
- Detects llm-d-modelservice charts (by chart name or presence of `parallelism` field)
- Maps `tensor_parallel_size` → `decode.parallelism.tensor`
- Maps `data_parallel_size` → `decode.parallelism.data`
- Maps `data_parallel_size_local` → `decode.parallelism.dataLocal`
- Adds non-standard vLLM args (like `max_num_seqs`) to `container.args` (appended)

For other chart-specific fields, extend `generate_helm_values()`:

```python
# In generate_helm_values() after parallelism mapping

# Map replicas
if "decode_replicas" in trial_config.parameters:
    values["decode"]["replicas"] = trial_config.parameters["decode_replicas"]

# Map routing proxy settings
if "routing_proxy_enabled" in trial_config.parameters:
    if "routing" not in values:
        values["routing"] = {}
    if "proxy" not in values["routing"]:
        values["routing"]["proxy"] = {}
    values["routing"]["proxy"]["enabled"] = trial_config.parameters["routing_proxy_enabled"]
```

#### Method 2: Using Values Template with Placeholders

Create a values template with placeholders that get replaced:

```yaml
# values-template.yaml
decode:
  parallelism:
    tensor: {{ .Values.decode.parallelism.tensor | default 1 }}
    data: {{ .Values.decode.parallelism.data | default 1 }}
  replicas: {{ .Values.decode.replicas | default 1 }}
```

Then in `generate_helm_values()`, set these values directly:

```python
# Set parallelism from trial parameters
if "tensor_parallel_size" in trial_config.parameters:
    if "parallelism" not in values["decode"]:
        values["decode"]["parallelism"] = {}
    values["decode"]["parallelism"]["tensor"] = trial_config.parameters["tensor_parallel_size"]
```

#### Method 3: Environment Variable Mapping

Some chart fields can be controlled via environment variables. Map parameters to env vars:

```yaml
# study_config.yaml
parameters:
  # This will go to decode.containers[0].env
  DECODE_REPLICAS:
    enabled: true
    options: ["1", "2", "3"]
```

The chart template can then read this env var:

```yaml
# In chart template
replicas: {{ .Values.decode.replicas | default (env "DECODE_REPLICAS" | int) }}
```

#### Example: Complete llm-d-modelservice Optimization

Here's a complete example optimizing both vLLM args and llm-d-specific fields:

```yaml
# study_config.yaml
parameters:
  # llm-d-modelservice parallelism (auto-mapped to decode.parallelism.*)
  # Chart will construct --tensor-parallel-size and --data-parallel-size from these
  tensor_parallel_size:  # Maps to decode.parallelism.tensor
    enabled: true
    min: 1
    max: 4
    step: 1
  
  data_parallel_size:  # Maps to decode.parallelism.data
    enabled: true
    min: 1
    max: 2
    step: 1
  
  # Standard vLLM parameters (added to container.args, appended after chart-constructed args)
  # Chart constructs: --model, --port, --tensor-parallel-size, --data-parallel-size, --served-model-name
  # These get appended: --max-num-seqs, --gpu-memory-utilization
  max_num_seqs:
    enabled: true
    min: 32
    max: 256
    step: 32
  
  gpu_memory_utilization:
    enabled: true
    min: 0.8
    max: 0.95
    step: 0.05

static_parameters:
  # Fixed values (added to container.args)
  max_model_len: 16384
```

**How it works:**
1. Chart constructs base args from `decode.parallelism.*` and `modelArtifacts.*`:
   ```
   --model Qwen/Qwen3-0.6B
   --port 8000
   --tensor-parallel-size 2
   --data-parallel-size 1
   --served-model-name Qwen/Qwen3-0.6B
   ```

2. Additional args from `container.args` are appended:
   ```
   --max-num-seqs 128
   --gpu-memory-utilization 0.9
   --max-model-len 16384
   ```

**No code changes needed** - `generate_helm_values()` automatically handles this mapping for llm-d-modelservice charts!

#### Available llm-d-modelservice Fields for Optimization

| Field Path | Description | Type | Example Values |
|------------|-------------|------|----------------|
| `decode.parallelism.tensor` | Tensor parallelism | int | 1, 2, 4, 8 |
| `decode.parallelism.data` | Data parallelism | int | 1, 2, 4 |
| `decode.parallelism.dataLocal` | Local data parallelism | int | 1, 2 |
| `decode.parallelism.workers` | Number of workers | int | 1, 2, 4 |
| `decode.replicas` | Number of pod replicas | int | 1, 2, 3 |
| `decode.containers[0].resources.limits.cpu` | CPU limit | string | "4", "8" |
| `decode.containers[0].resources.limits.memory` | Memory limit | string | "16Gi", "32Gi" |
| `routing.proxy.enabled` | Enable routing proxy | bool | true, false |
| `routing.proxy.targetPort` | vLLM target port | int | 8000, 8200 |
| `routing.proxy.connector` | Proxy connector | string | "nixl", "nixlv2" |
| `modelArtifacts.size` | Model volume size | string | "20Gi", "50Gi" |
| `modelArtifacts.mountModelVolume` | Mount model volume | bool | true, false |

### Verification Checklist

Before running optimization, verify:

- [ ] Chart has `decode.containers[0].args` field
- [ ] Deployment template uses `args` in container spec
- [ ] `modelCommand` is set (or defaults to "vllmServe")
- [ ] Resource limits are configurable
- [ ] Health probes are configured
- [ ] Service is created for vLLM API access
- [ ] All tunable parameters are in `study_config.yaml` `parameters:` section
- [ ] Static parameters are in `static_parameters:` section

## Chart Structure

The example chart includes:

- **Deployment**: vLLM server deployment with configurable resources
- **Service**: ClusterIP service for vLLM API access
- **Health Probes**: Liveness, readiness, and startup probes

### modelCommand Support

The chart supports the `modelCommand` field (matching llm-d-modelservice pattern):

- **`vllmServe`**: Uses `command: ["vllm", "serve"]` with args (default)
- **`imageDefault`**: Uses image's default entrypoint with args
- **`custom`**: Uses user-provided command and args

This allows compatibility with llm-d-modelservice charts and various vLLM image configurations.

## Customization

### Modifying Values Template

Edit `values.yaml` to customize:
- Container image
- Resource limits
- Health probe settings
- Volume mounts
- Environment variables

### Adding Chart Templates

You can extend the chart with additional templates:
- ConfigMaps for configuration
- Secrets for credentials
- Ingress for external access
- ServiceMonitor for Prometheus

## Troubleshooting

### Helm Release Fails to Install

- Check Helm chart syntax: `helm lint examples/helm`
- Verify Kubernetes resources: `kubectl get all -n llm-d-trials`
- Check Helm release status: `helm list -n llm-d-trials`

### Service Not Found

- Verify service was created: `kubectl get svc -n llm-d-trials`
- Check service labels match chart expectations
- Review service discovery logic in `helm_utils.py`

### Benchmark Job Fails

- Check Job logs: `kubectl logs job/<job-name> -n llm-d-trials`
- Verify benchmark image is accessible
- Check vLLM service is reachable from Job pods

### Results Not Extracted

- Ensure GuideLLM outputs JSON to stdout or writes to `/tmp/benchmark-results.json`
- Check Job pod logs for results
- Verify results file exists in pod: `kubectl exec -it <pod-name> -n llm-d-trials -- cat /tmp/benchmark-results.json`

## Notes

- Each trial creates a separate Helm release for isolation
- Benchmark Jobs are automatically cleaned up after completion
- Helm releases are tracked in trial metadata for reproducibility
- Results are extracted from Job logs or pod exec

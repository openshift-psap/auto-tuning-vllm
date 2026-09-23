# Agentic vLLM Tuner

Claude-driven alternative to Optuna search. The agent keeps a **read-only baseline**
server, creates a **fresh pod per experiment**, benchmarks with GuideLLM, compares
metrics with a 2% regression threshold, then deletes the experiment pod.

This is complementary to `auto-tune-vllm optimize`: Optuna does Bayesian search over
a declared parameter space; the agent reasons over logs, recipes, and results and
chooses the next config itself.

## Install

```bash
pip install -e ".[agent]"
```

Requires: `oc` (OpenShift mode) or SSH access, and either `ANTHROPIC_API_KEY`
or Vertex AI credentials. A profile-based OpenShift run provisions its own cache,
model downloader, baseline service, and experiment template.

## OpenShift with a tuning profile (recommended)

The tuning profile is the source of truth for the model, model cache, GPU
resources, TP ceiling, vLLM runtime flags, and benchmark workload. By default,
the agent provisions missing cache/download/baseline resources first, reuses a
healthy baseline without restarting it, creates a local baseline port-forward,
and passes the rendered experiment template to the agent.

```bash
auto-tune-vllm agent \
    --tuning-profile examples/agent/profiles/janus-gemma-32k1k.yaml \
    --kubeconfig /path/to/kubeconfig \
    --max-iterations 30
```

Use `--no-provision` only when the profile environment is already present and
you are supplying the normal connection options yourself. A completed downloader
Job is reused, so routine profile runs do not download the model again.

## Manually managed OpenShift

For an existing environment that is intentionally not profile-managed, supply
the endpoint, model, pod, namespace, and pod template explicitly:

```bash
auto-tune-vllm agent \
    --vllm-endpoint http://localhost:8000 \
    --model facebook/opt-125m \
    --oc-mode --oc-pod <baseline-pod> --oc-namespace <namespace> \
    --pod-template examples/agent/experiment-pod.yaml
```

## SSH mode

```bash
auto-tune-vllm agent \
    --vllm-endpoint http://gpu-host:8000 \
    --vllm-host gpu-host \
    --ssh-user root \
    --model meta-llama/Llama-3.1-8B \
    --api-key "$ANTHROPIC_API_KEY"
```

## Vertex AI

```bash
export CLAUDE_CODE_USE_VERTEX=1
export ANTHROPIC_VERTEX_PROJECT_ID=<project>
export CLOUD_ML_REGION=us-east5

auto-tune-vllm agent --vertex --vllm-endpoint http://localhost:8000 --model <model> ...
```

## What the agent does

1. **Baseline** — `nvidia-smi`, launch args, GuideLLM `balanced` profile, parse vLLM logs.
2. **Experiments** — `create_vllm_pod` with extra CLI args, benchmark, `compare_benchmarks`, `delete_vllm_pod`.
3. **Stop** — 10 consecutive experiments with no >2% gain, or `--max-iterations`.
4. **Report** — markdown + JSON under `agent_reports/`.

The baseline pod is never killed or restarted. Experiment pods are cleaned up on exit (`atexit`).

## Tools

| Tool | Where it runs |
|------|----------------|
| `run_command` / `read_file` / `write_file` / `fetch_vllm_logs` | Remote (pod or SSH host) |
| `run_benchmark` / `read_benchmark_results` / `compare_benchmarks` | Local (hits the port-forward) |
| `create_vllm_pod` / `delete_vllm_pod` | OpenShift (`oc apply` / `oc delete`) |
| `analyze_trace` / `map_kernel` / `check_preemptions` | Local analysis |

## Profiles

Defaults are conservative (tuned for small models like opt-125m). Edit
`auto_tune_vllm/agent/settings.yaml` to raise ISL/OSL for production models.

| Profile | ISL | OSL |
|---------|-----|-----|
| balanced | 128 | 128 |
| decode_heavy | 128 | 512 |
| prefill_heavy | 512 | 64 |
| long_context | 1024 | 128 |

## Origin

Ported from the Team TOA hackathon agent
([toa-vllm-hackathon](https://github.com/aas008/toa-vllm-hackathon)):
pod-per-experiment architecture, GuideLLM harness, and 120+ vLLM log patterns.

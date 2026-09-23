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

Requires: `oc` (OpenShift mode) or SSH access, a running baseline vLLM server,
and either `ANTHROPIC_API_KEY` or Vertex AI credentials.

## OpenShift (recommended)

1. Port-forward the **baseline** pod (never modified):

```bash
oc port-forward -n <namespace> <baseline-pod> 8000:8000
```

2. Copy and edit `examples/agent/experiment-pod.yaml` for your image, model path, PVC, and GPU count.

3. Run:

```bash
auto-tune-vllm agent \
    --vllm-endpoint http://localhost:8000 \
    --model facebook/opt-125m \
    --oc-mode \
    --oc-pod <baseline-pod> \
    --oc-namespace <namespace> \
    --pod-template examples/agent/experiment-pod.yaml \
    --max-iterations 30 \
    --profiles balanced \
    --concurrency 50 \
    --max-seconds 60
```

The **scored test is locked** for the whole session (same ISL/OSL/concurrency for baseline and every experiment). Defaults come from `auto_tune_vllm/agent/settings.yaml` (`benchmark:` + `warmup:`): profile `balanced` (ISL=128, OSL=128), concurrency 50 for 60s.

Before each scored run the controller hits an **unscored warmup** at a fixed concurrency (default 8 concurrent for 15s). Pass `--warmup-requests N` to stop after N requests instead of seconds. Warmup JSON is discarded. Use `--no-warmup` to skip.

```bash
# Example: decode-heavy scored test, 10s warmup at conc 4
auto-tune-vllm agent ... \
    --profiles decode_heavy \
    --concurrency 50 \
    --max-seconds 60 \
    --warmup-concurrency 4 \
    --warmup-seconds 10
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

1. **Baseline** — `nvidia-smi`, launch args, locked GuideLLM workload (warmup then scored), parse vLLM logs.
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

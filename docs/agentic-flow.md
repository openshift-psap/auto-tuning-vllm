# Agentic Tuning Flow

The agent is a Claude-driven tool loop: Claude proposes actions, the runner
executes them, feeds results back, and repeats. The baseline vLLM pod is
read-only; every configuration change is tested in a fresh ephemeral pod.

```mermaid
flowchart TD
    CLI["auto-tune-vllm agent<br/>CLI options"] --> Validate["Validate connectivity<br/>SSH or oc exec + Claude API"]
    Validate --> Setup["Create ClaudeClient, AgentTools,<br/>optional PodManager"]
    Setup --> Runner["AgenticRunner<br/>system prompt + message history"]

    Runner --> Claude["Claude / Vertex AI<br/>tool-use response"]
    Claude -->|text / end_turn| Continue["Append response +<br/>ask agent to continue"]
    Continue --> Runner

    Claude -->|tool calls| Dispatch["AgenticRunner executes tools<br/>and records decision log"]
    Dispatch -->|recipe discovery| RecipeCatalog["recipes.vllm.ai catalog<br/>exact-model lookup"]
    RecipeCatalog -->|no exact match| RecipeRepo["vllm-project/recipes<br/>family / architecture YAML lookup"]
    RecipeCatalog -->|recommended args| Feedback
    RecipeRepo -->|candidate args| Feedback

    subgraph Baseline["Baseline: never restarted or modified"]
      Remote["SSHExecutor or OcExecutor"] --> BasePod["Baseline vLLM pod"]
      BenchBase["GuideLLM Job<br/>(in-cluster; baseline Service)"] --> BasePod
      BasePod --> LogsBase["fetch_vllm_logs<br/>parse config, memory, warnings"]
      BenchBase --> ResultsBase["read_benchmark_results<br/>throughput, TTFT, ITL, TPOT"]
    end

    Dispatch -->|diagnostics / log fetch| Remote
    Dispatch -->|baseline benchmark| BenchBase
    LogsBase --> Feedback["Tool results returned<br/>to Claude message history"]
    ResultsBase --> Feedback

    subgraph Experiment["One isolated experiment per tuning attempt"]
      Create["create_vllm_pod<br/>vLLM args"] --> PodMgr["PodManager"]
      PodMgr --> Manifest["Copy YAML template;<br/>append vLLM args"]
      Manifest --> ExpPod["New OpenShift vLLM pod"]
      ExpPod --> PF["oc port-forward<br/>unique localhost port"]
      PF --> BenchExp["GuideLLM Job<br/>(in-cluster; experiment Service)"]
      ExpPod --> LogsExp["fetch_vllm_logs"]
      BenchExp --> ResultsExp["read_benchmark_results"]
      ResultsBase --> Compare["compare_benchmarks<br/>baseline vs experiment"]
      ResultsExp --> Compare
      Compare --> Delete["delete_vllm_pod<br/>stop port-forward + delete pod"]
    end

    Dispatch -->|create / delete pod| Create
    Dispatch -->|benchmark / analysis| BenchExp
    Dispatch -->|optional profiling| Trace["analyze_trace / map_kernel"]
    LogsExp --> Feedback
    ResultsExp --> Feedback
    Compare --> Feedback
    Delete --> Feedback
    Trace --> Feedback

    Feedback --> Runner

    Claude -->|done| Complete["AgentState complete"]
    Runner -->|iteration cap| Fallback["Extract results from decision log"]
    Complete --> Report["Reporter writes Markdown + JSON<br/>metrics, actions, token cost, decisions"]
    Fallback --> Report
    Report --> Cleanup["Finally: PodManager.cleanup_all"]
```

## Decision cycle

1. Check the vLLM Recipes catalog and, when needed, its source repository for
   exact-model or family/architecture-level YAML recommendations.
2. Inspect and benchmark the baseline, then parse its logs and metrics.
3. Claude selects one vLLM configuration change.
4. A new pod is created from the template with the change.
5. The experiment is benchmarked and inspected.
6. Its results are compared with the baseline; its pod is deleted.
7. The loop continues until Claude signals completion or the configured
   iteration limit is reached, then the reporter writes Markdown and JSON
   outputs.

## Current enforcement detail

The system prompt asks Claude to stop after 10 consecutive non-improving
experiments. This is prompt-guided, rather than mechanically enforced by the
runner. The hard stop is `max_iterations`; at that cap, the runner derives a
fallback summary from the decision log.

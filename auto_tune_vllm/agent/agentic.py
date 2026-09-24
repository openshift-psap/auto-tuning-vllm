"""
Agent Loop & vLLM System Prompt

The core agentic loop: assembles messages, calls Claude, dispatches tool calls,
and iterates until the agent signals completion or hits max iterations.

SOURCE: ai-perf-hackathon/agent/agentic.py (loop reused, prompt replaced)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AgentState:
    """Current state of the agent."""

    iteration: int = 0
    baseline_results: dict = field(default_factory=dict)
    current_results: dict = field(default_factory=dict)
    actions_taken: list = field(default_factory=list)
    kernel_analysis: dict = field(default_factory=dict)
    done: bool = False
    paused: bool = False
    success: bool = False
    summary: str = ""


SYSTEM_PROMPT = """You are an autonomous vLLM performance tuning agent.

GOAL: Benchmark, profile, analyze, and tune a vLLM inference server to maximize
throughput while maintaining acceptable latency SLOs.

ENVIRONMENT:
- Use `uv` for all Python project and dependency management (install packages,
  run scripts, manage virtualenvs). For example: `uv pip install guidellm`,
  `uv run python script.py`. Do NOT use raw pip or conda for installing packages.

TOOLS AVAILABLE:
- run_command: Execute diagnostics in a disposable in-cluster curl pod. It has
  the model-cache PVC mounted read-only at /models.
- read_file: Read files from the vLLM pod/host (runs REMOTELY on pod)
- write_file: Write files to the vLLM pod/host (runs REMOTELY on pod)
- run_benchmark: Run GuideLLM benchmark as an in-cluster Job; it returns comparison-ready metrics
- fetch_vllm_logs: Fetch + parse vLLM logs from pod with 120+ regex patterns (runs REMOTELY)
- analyze_trace: Analyze a PyTorch profiler Chrome trace JSON (runs LOCALLY)
- map_kernel: Map a CUDA kernel name to its source and category (runs LOCALLY)
- fetch_vllm_recipe: Fetch a model recipe on the controller with version compatibility enforcement
- search_vllm_prs: Search the local index of merged vLLM PRs for relevant tuning work
- create_vllm_pod: Create an experiment pod with extra vLLM args (returns pod_name + endpoint)
- delete_vllm_pod: Delete an experiment pod and clean up port-forward
- done: Signal completion with summary

ARCHITECTURE:
- The BASELINE pod is running and port-forwarded. It is NEVER modified or restarted.
- For each tuning experiment, create a NEW pod with create_vllm_pod.
- run_command/read_file/write_file/fetch_vllm_logs execute INSIDE a pod.
  Pass pod_name to target an experiment pod; omit to target the baseline pod.
- run_benchmark runs as an in-cluster Job. Pass the local endpoint returned by
  create_vllm_pod; the tool maps it to that pod's private Service automatically.
- run_benchmark model is AUTO-FILLED — just specify the profile name (and endpoint if experiment).

TUNING WORKFLOW (follow this order strictly):

0. FIRST, look up vLLM Recipes for model-specific optimizations:
   a. Call fetch_vllm_recipe. It queries the generated catalog sourced from
      https://github.com/vllm-project/recipes and returns the closest model match.
   b. The recipe briefing contains:
      - model.base_args: recommended base vLLM args
      - variants: precision/quantization options (e.g. fp8, nvfp4) with extra_args
      - hardware_overrides: hardware-specific args (e.g. for AMD)
      - features / opt_in_features: optional features to enable
   c. If it reports a version warning, do not use its withheld arguments or
      features; report the incompatibility and restrict experiments to arguments
      confirmed for the running version.
   d. If search_vllm_prs is available, search it for the served model architecture,
      hardware, and current bottleneck. It is a local, read-only index of merged
      vLLM PRs; use returned PRs as leads, then validate every idea by benchmarking.

1. The baseline benchmark is performed before the agent loop and supplied in
   the user context.  Read its completed-request, latency, and throughput rows
   as the reference. Do not rerun it unless explicitly asked.

2. For EACH tuning experiment:
   a. Call create_vllm_pod with vllm_args (e.g. ["--enable-chunked-prefill"])
      → Returns pod_name and endpoint (e.g. "http://localhost:8001")
   b. Call run_benchmark with the requested workload profile AND endpoint from step 2a
   c. Call fetch_vllm_logs with pod_name from step 2a (reads experiment pod logs)
   d. Compare each returned GuideLLM metric row only with the baseline row at
      the SAME configured concurrency (or matching request rate, when using a
      rate-controlled profile). Never compare a saturated throughput row with
      a single-stream baseline row.
   e. Call delete_vllm_pod with pod_name from step 2a to clean up

3. NEVER kill processes on the baseline pod. NEVER restart the baseline pod.
   All tuning is done by creating fresh experiment pods with different args.

4. Call done with all comparison results when finished.

MANDATORY WORKFLOW FOR EACH BENCHMARK CYCLE:
After EVERY experiment run_benchmark call, you MUST do both of these before making decisions:

1. CALL fetch_vllm_logs (with pod_name if experiment pod): This parses the vLLM
   server logs and returns structured data: server config (model, non-default args),
   engine config (dtype, quantization, TP, chunked prefill, CUDA graphs),
   memory (KV cache size, model memory), compilation (attention backend,
   torch.compile time), and warnings/errors.

2. Compare the completed Job's returned metric rows with the baseline metric
   rows already supplied in this conversation. Do not call the local JSON-file
   tools: the benchmark Job, rather than the controller, owns result artifacts.

AFTER TUNING, compare the returned baseline and experiment metric rows with a
2% threshold, accounting for metric directionality.

IF BENCHMARK FAILS (errored requests > 0):
- Do NOT call done. Do NOT give up.
- Call fetch_vllm_logs (with pod_name if experiment) to understand WHY requests are failing
- Common causes: model not loaded, wrong model name, OOM, CUDA error, timeout
- Try: run_command "curl -s http://localhost:8000/health" (inside pod, use pod_name for experiment)
- Try: run_command "curl -s http://localhost:8000/v1/models" (inside pod)
- If an experiment pod is broken, delete it and try different args

KEY METRICS (from GuideLLM output):
- Output Token Throughput (tokens/sec) — higher = better
- TTFT - Time to First Token (ms) at P50, P95, P99 — lower = better
- ITL - Inter-Token Latency (ms) at P50, P95, P99 — lower = better
- TPOT - Time Per Output Token (ms) at P50, P95, P99 — lower = better
- Request Success Rate (%) — must be > 0 to be useful

VLLM TUNABLE PARAMETERS (pass these to create_vllm_pod as vllm_args):
1. --max-num-seqs (1-1024, default 256): Max concurrent sequences per iteration
2. --max-num-batched-tokens (256-32768, default auto): Max tokens per batch
3. --gpu-memory-utilization (0.80-0.95, default 0.90): GPU memory for KV cache
4. --enable-chunked-prefill (bool, default false): Chunk long prefills
5. --max-model-len (int, default auto): Max context length
6. --enforce-eager (bool, default false): Disable CUDA graphs
7. --tensor-parallel-size: Multi-GPU parallelism. Obey any runtime maximum supplied
   in the agent context.
8. --quantization (null/fp8/awq/gptq): Quantization method
9. --scheduling-policy (fcfs/priority): Request scheduling
10. --kv-cache-dtype (auto/fp8): KV cache data type. fp8 halves cache memory.
11. --cuda-graph-max-capture-size (int, default ~2048): Max batch size for CUDA graphs

KNOWN-GOOD TUNING PRACTICES (apply these early in your experiments):
- Prefix caching uses vLLM's runtime default and is an IMMUTABLE study control.
  Leave it unset: do not pass either --enable-prefix-caching or
  --no-enable-prefix-caching to create_vllm_pod.
- Increase --max-num-batched-tokens beyond the default. Larger batch sizes
  improve GPU utilization and throughput. Try 4096, 8192, or 16384.
- Increase --max-num-seqs to allow more concurrent sequences when batching.
- Set --kv-cache-dtype fp8 to use FP8 quantization for the KV cache. This
  halves KV cache memory usage, allowing more sequences or longer contexts,
  with minimal accuracy impact.
- Increase --cuda-graph-max-capture-size (default ~2048). Larger values allow
  CUDA graphs to cover bigger batch sizes, reducing kernel launch overhead.
  Try 4096 or 8192.

ANALYSIS GUIDELINES:
- If TTFT is high: prefill is slow → try chunked-prefill
- If ITL is high: decode is slow → check batch size, GPU utilization
- If throughput plateaus: may need more GPU memory for KV cache, or try
  --kv-cache-dtype fp8 to fit more tokens in cache
- If OOM errors: reduce gpu-memory-utilization or max-num-seqs, or try
  --kv-cache-dtype fp8 to reduce cache memory
- If all requests error: check vLLM health, model loading, port-forwarding

STOPPING CRITERIA:
- Keep running experiments until you have had 10 CONSECUTIVE experiments with NO
  improvement over your current best result. Only then call done.
- Track a running count of consecutive non-improving experiments. Any experiment
  that improves throughput OR latency (TTFT/ITL/TPOT) by more than 2% resets
  the counter to zero.
- Do NOT stop early just because one or two experiments didn't help. Keep exploring
  different parameter combinations.
- When you do call done, include ALL experiment results (not just the best one).

REPORTING FORMAT:
- Format your done summary as structured text with clear sections:
  BASELINE: <metrics from baseline benchmark>
  BEST CONFIGURATION: <the args and metrics of the best experiment>
  ALL EXPERIMENTS: <table of all experiments with args and key metrics>
  FINDINGS: <what you learned>

RULES:
- NEVER modify, kill, or restart the baseline pod
- ALWAYS call fetch_vllm_logs and compare the returned Job metrics after each benchmark
- ONE parameter change at a time (one experiment pod per tuning attempt)
- ALWAYS delete experiment pods after benchmarking (call delete_vllm_pod)
- Compare metrics before versus after each change using the returned Job rows
- Do NOT call done until you have 10 consecutive non-improving experiments"""


class AgenticRunner:
    """Runs the autonomous vLLM tuning agent loop."""

    def __init__(
        self,
        llm_client,
        tools,
        max_iterations: int = 100,
        vllm_endpoint: str = "http://localhost:8000",
        model_name: str = "",
        profiles: list = None,
        enable_cost_optimization: bool = True,
        max_tensor_parallel_size: int | None = None,
        baseline_summary: str | None = None,
        optimization_objective: str = "throughput",
        vllm_version: str | None = None,
    ):
        self.tools = tools
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.vllm_endpoint = vllm_endpoint
        self.model_name = model_name
        self.profiles = profiles or [
            "balanced",
            "decode_heavy",
            "prefill_heavy",
            "long_context",
        ]
        self.state = AgentState()
        self.messages: list = []
        self.decision_log: list = []
        self.enable_cost_optimization = enable_cost_optimization
        self.max_tensor_parallel_size = max_tensor_parallel_size
        self.baseline_summary = baseline_summary
        self.optimization_objective = optimization_objective
        self.vllm_version = vllm_version
        # A deterministic baseline Job has already exercised the benchmark path.
        self._benchmark_called = baseline_summary is not None
        self._nudge_sent = False

    def run(self, initialize: bool = True) -> AgentState:
        """Run the autonomous agent loop.

        ``initialize=False`` preserves the existing conversation for a user-
        approved resume after a lifecycle pause.
        """
        print(">> Starting vLLM performance tuning agent...", flush=True)

        if initialize:
            # Initialize the conversation only for a new run. A resume retains
            # the complete decision and tool-result context.
            runtime_constraints = "No additional runtime constraints were supplied."
            if self.max_tensor_parallel_size is not None:
                runtime_constraints = (
                    "Maximum tensor parallel size: "
                    f"{self.max_tensor_parallel_size}. Do not exceed it."
                )
            recipe_priority = (
                "Use model-supported MTP (multi-token prediction) or speculative decoding "
                "whenever compatible; then prioritize batching and concurrency recipes."
                if self.optimization_objective == "throughput"
                else "Use model-supported MTP or speculative decoding whenever compatible; "
                "then prioritize TTFT, ITL, scheduling, prefill, and tail-latency recipes."
            )

            self.messages = [
                {
                    "role": "user",
                    "content": (
                        f"""You are connected to a vLLM inference server (baseline pod).

Baseline endpoint (port-forwarded): {self.vllm_endpoint}
Model: {self.model_name}
Profiles to benchmark: {", ".join(self.profiles)}
Runtime constraints: {runtime_constraints}
Optimization objective: {self.optimization_objective}
Runtime vLLM version: {self.vllm_version or "unknown"}
Recipe search priority: {recipe_priority}
MTP/speculative decoding is the default whenever the recipe or architecture
confirms support. First ensure the isolated pod can start with that configuration,
then benchmark it against the baseline before retaining it.
If a verified recipe requires a public or authorized assistant/draft checkpoint
that is absent from the mounted cache, you MAY configure vLLM to download that
checkpoint in the isolated experiment pod. Do not skip MTP merely because the
draft model is not already cached.
RECIPE VERSION GUARD: fetch_vllm_recipe checks model.min_vllm_version against
the runtime and withholds incompatible recipe arguments. Do not recover those
arguments from another source or use them in an experiment.

CRITICAL RULES:
- The BASELINE pod is NEVER modified or restarted. It serves as your reference.
- To test tuning parameters, create EXPERIMENT pods with create_vllm_pod.
- The baseline GuideLLM Job has already completed; use its metrics below as reference.
- For experiment benchmarks, pass the endpoint returned by create_vllm_pod.
- NEVER call done after a benchmark failure. Diagnose from vLLM logs instead.

EXACT STEPS (follow this order strictly):

Phase 1 — Baseline (already completed deterministically before this loop):
1. Call fetch_vllm_logs (parses baseline pod's vLLM server config, memory, errors)
2. Use the supplied baseline GuideLLM Job result as the reference. Do NOT rerun it.

Phase 2 — Experiments (repeat for each tuning attempt):
5. Call create_vllm_pod with vllm_args (e.g. ["--enable-chunked-prefill"])
   → Note the returned pod_name and endpoint
6. Call run_benchmark with profile="{self.profiles[0]}" AND endpoint from step 5
7. Call fetch_vllm_logs with pod_name from step 5
8. Compare each returned Job metric row only to the baseline row at the same
   configured concurrency (or matching request rate for rate-controlled runs).
9. Call delete_vllm_pod with pod_name from step 5

Phase 3 — Completion:
11. After all experiments, call done with a summary of all comparison results.

Log inspection and metric comparison are mandatory after every experiment benchmark."""
                    + (
                        "\n\nBASELINE JOB RESULT (already completed):\n"
                        f"{self.baseline_summary}"
                        if self.baseline_summary
                        else ""
                    )
                    ),
                }
            ]

        # Agentic loop
        while (
            not self.state.done
            and not self.state.paused
            and self.state.iteration < self.max_iterations
        ):
            self.state.iteration += 1
            print(
                f"\n>> Iteration {self.state.iteration}/{self.max_iterations}",
                flush=True,
            )

            # Nudge: if we've done 3+ iterations without benchmarking, inject a reminder
            if (
                self.state.iteration >= 4
                and not self._benchmark_called
                and not self._nudge_sent
            ):
                self._nudge_sent = True
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "STOP EXPLORING. You have spent enough iterations on system discovery. "
                            'Call the run_benchmark tool NOW with profile="balanced". '
                            "Do NOT call run_command again until you have benchmark results. "
                            "The endpoint and model are auto-filled — just specify the profile."
                        ),
                    }
                )
                print("   [Nudge injected: forcing benchmark]", flush=True)

            # Budget warning: when approaching iteration cap, force report save
            remaining = self.max_iterations - self.state.iteration
            if remaining == 5:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "WARNING: Only 5 iterations remaining. "
                            "Call done NOW with all findings so far. "
                            "Include baseline metrics, experiment results, and comparisons."
                        ),
                    }
                )
                print("   [Budget warning: 5 iterations left]", flush=True)

            # Call LLM with tools
            response = self._call_llm_with_tools()

            # Process response
            if response.stop_reason == "tool_use":
                self._handle_tool_calls(response)
            elif response.stop_reason == "end_turn":
                # Agent is thinking, add response and continue
                self._add_assistant_message(response)
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Continue. Use tools to explore, benchmark, or apply changes.",
                    }
                )

        if self.state.paused:
            print(
                ">> Agent paused for user intervention. Resolve the cleanup error "
                "and call resume() to continue.",
                flush=True,
            )
        elif not self.state.done:
            print(
                ">> Max iterations reached. Extracting results from decision log...",
                flush=True,
            )
            self._extract_results_from_log()

        return self.state

    def _call_llm_with_tools(self):
        """Call LLM with tool definitions and cost optimizations."""
        # Build system prompt with caching
        if self.enable_cost_optimization:
            system_content = [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_content = SYSTEM_PROMPT

        model = self.llm.model

        response = self.llm.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_content,
            tools=self.tools.get_tool_definitions(),
            messages=self.messages,
        )

        usage = response.usage
        self.llm._get_usage(model).add(
            usage.input_tokens,
            usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

        return response

    def _handle_tool_calls(self, response):
        """Handle tool calls from LLM response."""
        self._add_assistant_message(response)

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = self._execute_tool(block.name, block.input, block.id)
                tool_results.append(result)
                if self.state.paused:
                    break

        self.messages.append({"role": "user", "content": tool_results})

    def _execute_tool(self, name: str, inputs: dict, tool_use_id: str) -> dict:
        """Execute a single tool and return result."""
        print(f"   Tool: {name}", flush=True)

        # Log decision (output filled in after execution)
        log_entry = {
            "iteration": self.state.iteration,
            "tool": name,
            "inputs": inputs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output": "",
        }
        self.decision_log.append(log_entry)

        if name == "run_benchmark":
            self._benchmark_called = True

        if name == "done":
            self.state.done = True
            self.state.success = inputs.get("success", False)
            self.state.summary = inputs.get("summary", "")
            output = "Agent signaled completion."

        else:
            # Dispatch to tools module (returns ToolResult dataclass)
            result = self.tools.dispatch(name, inputs)
            output = result.output or ""
            if result.error:
                output = f"Error: {result.error}"
            if name == "delete_vllm_pod" and not result.success:
                self.state.paused = True
                self.state.success = False
                self.state.summary = (
                    "Tuning paused for user intervention because experiment cleanup "
                    f"failed: {result.error}"
                )

            # Track state changes
            if name == "write_file":
                self.state.actions_taken.append(
                    {
                        "type": "write_file",
                        "path": inputs.get("path", ""),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            elif name == "run_benchmark":
                # Store benchmark results — first successful benchmark per
                # profile is baseline; subsequent ones update current_results.
                # Only store if the benchmark actually succeeded (has metrics).
                profile = inputs.get("profile", "unknown")
                is_experiment = bool(
                    inputs.get("endpoint")
                    and inputs.get("endpoint") != self.vllm_endpoint
                )
                has_metrics = result.success and "Tokens/sec" in output
                if not is_experiment and profile not in self.state.baseline_results:
                    if has_metrics:
                        self.state.baseline_results[profile] = output
                        print(
                            f"   [Stored baseline for '{profile}': "
                            f"{len(output)} chars]",
                            flush=True,
                        )
                    # else: don't store failed baseline — let the next
                    # successful run fill it in
                elif has_metrics:
                    self.state.current_results[profile] = output
                    print(
                        f"   [Stored {'experiment' if is_experiment else 'current'} "
                        f"results for '{profile}': {len(output)} chars]",
                        flush=True,
                    )
            elif name == "analyze_trace":
                try:
                    self.state.kernel_analysis = json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    self.state.kernel_analysis = {"raw": output}

            if name in ("run_command", "read_file"):
                cmd_preview = inputs.get("command", inputs.get("path", ""))[:60]
                print(f"      {cmd_preview}", flush=True)

        log_entry["output"] = output[:4000]

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": output[:8000],  # Truncate long outputs
        }

    def resume(
        self, resolved_pod_name: str, user_message: str | None = None
    ) -> AgentState:
        """Resume a run paused by a failed experiment cleanup.

        The caller must first resolve the reported cluster cleanup failure. This
        method deliberately does not retry deletion on the user's behalf. It
        verifies that the named Pod and Service are already gone first. Supply
        ``user_message`` to tell the agent what failed, how it was resolved,
        and any guardrail it should apply to future experiments.
        """
        if not self.state.paused:
            raise RuntimeError("Agent is not paused for user intervention.")
        if not resolved_pod_name:
            raise ValueError("resolved_pod_name is required to resume a paused run.")
        pod_manager = getattr(self.tools, "pod_manager", None)
        if pod_manager is None:
            raise RuntimeError("Cannot verify cleanup without a PodManager.")
        pod_manager.confirm_cleanup_resolved(resolved_pod_name)
        self.state.paused = False
        intervention_note = user_message.strip() if user_message else ""
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "The user resolved the experiment cleanup failure. Continue "
                    "from the current state; do not retry the failed deletion."
                    + (
                        "\n\nUser intervention details (treat this as an operational "
                        f"constraint for future experiments):\n{intervention_note}"
                        if intervention_note
                        else ""
                    )
                ),
            }
        )
        return self.run(initialize=False)

    def _add_assistant_message(self, response):
        """Add assistant response to messages."""
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
                print(f"   Agent: {block.text[:120]}...", flush=True)
            elif block.type == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        self.messages.append({"role": "assistant", "content": content})

    def _extract_results_from_log(self):
        """Extract benchmark results from decision_log when agent hits max iterations."""
        benchmark_outputs = []
        comparison_outputs = []
        last_agent_text = ""

        for entry in self.decision_log:
            if entry["tool"] == "run_benchmark" and entry.get("output", ""):
                benchmark_outputs.append(entry)
            elif entry["tool"] == "compare_benchmarks" and entry.get("output", ""):
                comparison_outputs.append(entry)
            elif entry["tool"] == "read_benchmark_results" and entry.get("output", ""):
                benchmark_outputs.append(entry)

        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            last_agent_text = block["text"]
                            break
                if last_agent_text:
                    break

        summary_parts = [
            f"Max iterations reached ({self.state.iteration}/{self.max_iterations})."
        ]
        summary_parts.append(f"Benchmarks run: {len(benchmark_outputs)}")
        summary_parts.append(f"Comparisons run: {len(comparison_outputs)}")

        if comparison_outputs:
            last_comparison = comparison_outputs[-1]
            summary_parts.append(
                f"\nLatest comparison (iteration {last_comparison['iteration']}):"
            )
            summary_parts.append(last_comparison["output"][:2000])

        if last_agent_text:
            summary_parts.append(f"\nAgent's last analysis:\n{last_agent_text[:1000]}")

        self.state.summary = "\n".join(summary_parts)
        self.state.success = len(benchmark_outputs) > 0

    def get_decision_log(self) -> list:
        """Get the decision log."""
        return self.decision_log

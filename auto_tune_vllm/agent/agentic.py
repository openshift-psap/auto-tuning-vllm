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
- run_command: Execute shell commands on the vLLM pod/host (runs REMOTELY on pod)
- read_file: Read files from the vLLM pod/host (runs REMOTELY on pod)
- write_file: Write files to the vLLM pod/host (runs REMOTELY on pod)
- run_benchmark: Run the LOCKED GuideLLM test (warmup then scored run).
  Workload is fixed for the session. Pass endpoint for experiment pods; omit for baseline.
- fetch_vllm_logs: Fetch + parse vLLM logs from pod with 120+ regex patterns (runs REMOTELY)
- read_benchmark_results: Read a GuideLLM JSON file and extract structured metrics (runs LOCALLY)
- compare_benchmarks: Compare two benchmark JSON files to detect regressions (runs LOCALLY)
- analyze_trace: Analyze a PyTorch profiler Chrome trace JSON (runs LOCALLY)
- map_kernel: Map a CUDA kernel name to its source and category (runs LOCALLY)
- create_vllm_pod: Create an experiment pod with extra vLLM args (returns pod_name + endpoint)
- delete_vllm_pod: Delete an experiment pod and clean up port-forward
- done: Signal completion with summary

ARCHITECTURE:
- The BASELINE pod is running and port-forwarded. It is NEVER modified or restarted.
- For each tuning experiment, create a NEW pod with create_vllm_pod.
- run_command/read_file/write_file/fetch_vllm_logs execute INSIDE a pod.
  Pass pod_name to target an experiment pod; omit to target the baseline pod.
- run_benchmark runs LOCALLY. Pass endpoint from create_vllm_pod to benchmark
  an experiment pod; omit endpoint to benchmark the baseline.
- run_benchmark model is AUTO-FILLED. Do NOT change profile, ISL/OSL, or concurrency.
  An unscored warmup (fixed concurrency, short duration or N requests) runs automatically
  before every scored test. Only pass endpoint when benchmarking an experiment pod.

TUNING WORKFLOW (follow this order strictly):

0. CHECK YOUR INITIAL CONTEXT for the PRE-FETCHED vLLM RECIPE block.
   The controller fetched recipes.vllm.ai at startup and injected the recipe
   (base_args, hardware-specific args, variants, spec_decoding) directly into
   your first message — you do NOT need to curl anything.
   - If the recipe block is present: use the FIRST EXPERIMENT args as your
     initial create_vllm_pod call. Then build on top with variants.
   - If the initial context says "no recipe found": skip this step and use the
     KNOWN-GOOD TUNING PRACTICES below as your first experiment.

1. Benchmark the BASELINE pod (already running, uses default endpoint):
   a. Call run_benchmark (no profile needed — the session workload is locked)
   b. Call fetch_vllm_logs (no pod_name — reads baseline pod logs)
   c. Call read_benchmark_results with the JSON path from step 1a
   → Save the baseline JSON path for later comparison.

2. For EACH tuning experiment:
   a. Call create_vllm_pod with vllm_args (e.g. ["--enable-chunked-prefill"])
      → Returns pod_name and endpoint (e.g. "http://localhost:8001")
   b. Call run_benchmark AND endpoint from step 2a (same locked workload + warmup)
   c. Call fetch_vllm_logs with pod_name from step 2a (reads experiment pod logs)
   d. Call read_benchmark_results with the JSON path from step 2b
   e. Call compare_benchmarks with baseline JSON (from step 1c) and experiment JSON (from step 2d)
   f. Call delete_vllm_pod with pod_name from step 2a to clean up

3. NEVER kill processes on the baseline pod. NEVER restart the baseline pod.
   All tuning is done by creating fresh experiment pods with different args.

4. Call done with all comparison results when finished.

MANDATORY WORKFLOW FOR EACH BENCHMARK CYCLE:
After EVERY run_benchmark call, you MUST do BOTH of these before making any decisions:

1. CALL fetch_vllm_logs (with pod_name if experiment pod): This parses the vLLM
   server logs and returns structured data: server config (model, non-default args),
   engine config (dtype, quantization, TP, chunked prefill, CUDA graphs),
   memory (KV cache size, model memory), compilation (attention backend,
   torch.compile time), and warnings/errors.

2. CALL read_benchmark_results with the JSON path from run_benchmark output:
   This returns structured per-concurrency metrics: success rate, throughput,
   TTFT, ITL, TPOT with P50/P95/P99 percentiles.

AFTER TUNING, use compare_benchmarks:
   Pass the baseline JSON path and the post-tuning JSON path to get a side-by-side
   comparison with regression detection (2% threshold, metric directionality aware).

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
5. --enable-prefix-caching (bool, default false): Cache common prefixes
6. --max-model-len (int, default auto): Max context length
7. --enforce-eager (bool, default false): Disable CUDA graphs
8. --tensor-parallel-size (1-8, default 1): Multi-GPU parallelism
9. --quantization (null/fp8/awq/gptq): Quantization method
10. --scheduling-policy (fcfs/priority): Request scheduling
11. --kv-cache-dtype (auto/fp8): KV cache data type. fp8 halves cache memory.
12. --cuda-graph-max-capture-size (int, default ~2048): Max batch size for CUDA graphs

KNOWN-GOOD TUNING PRACTICES (apply these early in your experiments):
- ALWAYS enable prefix caching (--enable-prefix-caching). It reduces redundant
  computation for repeated prefixes and almost never hurts performance.
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
- If TTFT is high: prefill is slow → try chunked-prefill or prefix-caching
- If ITL is high: decode is slow → check batch size, GPU utilization
- If throughput plateaus: may need more GPU memory for KV cache, or try
  --kv-cache-dtype fp8 to fit more tokens in cache
- If OOM errors: reduce gpu-memory-utilization or max-num-seqs, or try
  --kv-cache-dtype fp8 to reduce cache memory
- If all requests error: check vLLM health, model loading, port-forwarding

STOPPING CRITERIA:
- Keep running experiments until you have had 10 CONSECUTIVE experiments with NO
  improvement over your current best result. Only then proceed to verification.
- Track a running count of consecutive non-improving experiments. Any experiment
  that improves throughput OR latency (TTFT/ITL/TPOT) by more than 2% resets
  the counter to zero.
- Do NOT stop early just because one or two experiments didn't help. Keep exploring
  different parameter combinations.

FINAL VERIFICATION (before calling done):
- Once you have identified the best configuration, call run_benchmark with
  repeat=3 on that configuration to verify reproducibility.
- This runs the scored test 3 times and returns mean ± stddev per metric.
- A coefficient of variation (CV) below 5% on throughput indicates a stable result.
- When you do call done, include ALL experiment results AND the verification stats.

REPORTING FORMAT:
- Format your done summary as structured text with clear sections:
  BASELINE: <metrics from baseline benchmark>
  BEST CONFIGURATION: <the args and metrics of the best experiment>
  ALL EXPERIMENTS: <table of all experiments with args and key metrics>
  FINDINGS: <what you learned>

RULES:
- NEVER modify, kill, or restart the baseline pod
- ALWAYS call fetch_vllm_logs AND read_benchmark_results after each benchmark
- ONE parameter change at a time (one experiment pod per tuning attempt)
- ALWAYS delete experiment pods after benchmarking (call delete_vllm_pod)
- Compare metrics before vs after each change using compare_benchmarks
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
        workload_description: str = "",
        vllm_recipe: dict = None,
    ):
        self.tools = tools
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.vllm_endpoint = vllm_endpoint
        self.model_name = model_name
        self.profiles = profiles or ["balanced"]
        self.workload_description = workload_description or (
            f"profiles={', '.join(self.profiles)}"
        )
        self.state = AgentState()
        self.messages: list = []
        self.decision_log: list = []
        self.enable_cost_optimization = enable_cost_optimization
        self._benchmark_called = False
        self._nudge_sent = False
        self.vllm_recipe = vllm_recipe

    def run(self) -> AgentState:
        """Run the autonomous agent loop."""
        print(">> Starting vLLM performance tuning agent...", flush=True)

        # Build recipe block for injection into the first user message
        if self.vllm_recipe:
            from .recipe_fetcher import format_recipe_briefing
            recipe_block = "\n" + format_recipe_briefing(self.vllm_recipe) + "\n"
        else:
            recipe_block = (
                "\nNO vLLM RECIPE FOUND for this model in recipes.vllm.ai. "
                "Apply KNOWN-GOOD TUNING PRACTICES as your first experiment.\n"
            )

        # Initialize conversation
        self.messages = [
            {
                "role": "user",
                "content": f"""You are connected to a vLLM inference server (baseline pod).

Baseline endpoint (port-forwarded): {self.vllm_endpoint}
Model: {self.model_name}
LOCKED WORKLOAD (do not change): {self.workload_description}
{recipe_block}

CRITICAL RULES:
- The BASELINE pod is NEVER modified or restarted. It serves as your reference.
- To test tuning parameters, create EXPERIMENT pods with create_vllm_pod.
- For baseline benchmarks, call run_benchmark with no extra args (endpoint auto-filled).
- For experiment benchmarks, pass the endpoint returned by create_vllm_pod.
- Do NOT pick a different profile, ISL/OSL, or concurrency. Warmup is automatic.
- NEVER call done after a benchmark failure. Diagnose from vLLM logs instead.

EXACT STEPS (follow this order strictly):

Phase 1 — Baseline:
1. Call run_command with command="nvidia-smi" (1 tool call)
2. Call run_command with command="cat /proc/1/cmdline | tr '\\0' ' '" (see vLLM launch args)
3. Call run_benchmark (THIS IS MANDATORY ON STEP 3 — uses baseline + locked workload)
4. AFTER benchmark completes, ALWAYS call BOTH:
   a. fetch_vllm_logs (parses baseline pod's vLLM server config, memory, errors)
   b. read_benchmark_results with the JSON path from step 3 output
   → SAVE the baseline JSON path for later compare_benchmarks calls.

Phase 2 — Experiments (repeat for each tuning attempt):
5. Call create_vllm_pod with vllm_args (e.g. ["--enable-chunked-prefill"])
   → Note the returned pod_name and endpoint
6. Call run_benchmark with endpoint from step 5 (same locked workload + warmup)
7. Call fetch_vllm_logs with pod_name from step 5
8. Call read_benchmark_results with the JSON path from step 6
9. Call compare_benchmarks with baseline JSON (step 4b) and experiment JSON (step 8)
10. Call delete_vllm_pod with pod_name from step 5

Phase 3 — Completion:
11. After all experiments, call done with a summary of all comparison results.

Steps 4a/4b (and 7/8 for experiments) are MANDATORY after every benchmark.""",
            }
        ]

        # Agentic loop
        while not self.state.done and self.state.iteration < self.max_iterations:
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
                            "Call the run_benchmark tool NOW (no profile arg — workload is locked). "
                            "Do NOT call run_command again until you have benchmark results."
                        ),
                    }
                )
                print("   [Nudge injected: forcing benchmark]", flush=True)

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

        if not self.state.done:
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

    def _add_assistant_message(self, response):
        """Add assistant response to messages."""
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
                print(f"   Agent: {block.text[:120]}...", flush=True)
            elif block.type == "tool_use":
                raw_input = block.input
                if isinstance(raw_input, dict):
                    raw_input = dict(raw_input)
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": raw_input,
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

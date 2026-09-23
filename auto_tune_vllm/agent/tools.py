"""
Tool Definitions & Dispatch

Defines all tools available to the Claude agent and their handler functions.

SOURCE: ai-perf-hackathon/agent/tools.py (core tools reused, new tools added)

Core Tools (from ai-perf-hackathon, adapted for RemoteExecutor):
    - run_command(command, timeout)    -- Execute shell command on vLLM host/pod
    - read_file(path)                 -- Read file contents from vLLM host/pod
    - write_file(path, content)       -- Write file to vLLM host/pod
    - done(summary, success)          -- Signal agent completion

New Benchmark Tool:
    - run_benchmark(endpoint, model)
        Warmup then scored GuideLLM run. Workload is locked for the session
        (profile / ISL / OSL / concurrency from settings.yaml or CLI).
        Output: throughput (tok/sec), TTFT, ITL, TPOT at P50/P95/P99

New Analysis Tools:
    - analyze_trace(trace_json_path)
        Calls analysis/trace_analyzer.py to extract kernel stats, category breakdown.
    - map_kernel(kernel_name)
        Calls analysis/kernel_mapper.py to identify vLLM source for hot kernels.

Architecture:
    RemoteExecutor abstracts over SSH and OpenShift (`oc exec`) execution modes.
    - OcExecutor(namespace, pod_name, kubeconfig) -- runs `oc exec` via subprocess
    - SSHExecutor(ssh_client) -- wraps the existing SSHClient

Tool Definition Format:
    Each tool is a dict with: name, description, input_schema (JSON Schema)
    Compatible with Claude's tool_use API format.

Dispatch:
    dispatch_tool(name, args, executor) -> ToolResult
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .benchmark_spec import BenchmarkSpec
from .pod_manager import PodManager
from .ssh_client import SSHClient

# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool: str
    success: bool
    output: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"tool": self.tool, "success": self.success, "output": self.output}
        if self.error:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# RemoteExecutor abstraction
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """Unified result from a remote command execution."""

    stdout: str
    stderr: str
    returncode: int
    success: bool

    @property
    def output(self) -> str:
        return self.stdout if self.success else self.stderr


class RemoteExecutor(ABC):
    """Abstract base for executing commands on a remote vLLM host/pod."""

    @abstractmethod
    def run(self, command: str, timeout: int = 60) -> CommandResult:
        """Execute a command on the remote target."""
        ...

    @abstractmethod
    def read_file(self, path: str) -> CommandResult:
        """Read a file from the remote target."""
        ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> CommandResult:
        """Write content to a file on the remote target."""
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Test connectivity to the remote target."""
        ...


class SSHExecutor(RemoteExecutor):
    """Execute commands on the vLLM host via SSH.

    Wraps the existing SSHClient for backward compatibility with the
    ai-perf-hackathon SSH-based workflow.
    """

    def __init__(self, ssh_client: SSHClient):
        self._client = ssh_client

    def run(self, command: str, timeout: int = 60) -> CommandResult:
        result = self._client.run(command, timeout=timeout)
        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            success=result.success,
        )

    def read_file(self, path: str) -> CommandResult:
        result = self._client.read_file(path)
        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            success=result.success,
        )

    def write_file(self, path: str, content: str) -> CommandResult:
        result = self._client.write_file(path, content)
        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            success=result.success,
        )

    def test_connection(self) -> bool:
        return self._client.test_connection()


class OcExecutor(RemoteExecutor):
    """Execute commands on a vLLM pod via ``oc exec`` (OpenShift).

    Parameters
    ----------
    namespace : str
        Kubernetes namespace where the vLLM pod runs.
    pod_name : str
        Name (or label selector) of the vLLM pod.
    kubeconfig : str or None
        Path to a kubeconfig file. If *None*, the default kubeconfig
        (``~/.kube/config`` or ``$KUBECONFIG``) is used.
    container : str or None
        Container name inside the pod. If *None*, the default container
        is used (Kubernetes picks the first one).
    """

    def __init__(
        self,
        namespace: str,
        pod_name: str,
        kubeconfig: Optional[str] = None,
        container: Optional[str] = None,
    ):
        self.namespace = namespace
        self.pod_name = pod_name
        self.kubeconfig = kubeconfig
        self.container = container

    def _build_oc_cmd(self, command: str) -> list[str]:
        """Build the full ``oc exec`` command list."""
        cmd = ["oc"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += ["exec", "-n", self.namespace]
        if self.container:
            cmd += ["-c", self.container]
        cmd += [self.pod_name, "--"]
        # Use /bin/sh -c to support shell features (pipes, redirects, etc.)
        cmd += ["/bin/sh", "-c", command]
        return cmd

    def run(self, command: str, timeout: int = 60) -> CommandResult:
        oc_cmd = self._build_oc_cmd(command)
        try:
            result = subprocess.run(
                oc_cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )
            stdout = result.stdout.replace("\r\n", "\n").replace("\r", "\n")
            stderr = result.stderr.replace("\r\n", "\n").replace("\r", "\n")
            return CommandResult(
                stdout=stdout,
                stderr=stderr,
                returncode=result.returncode,
                success=result.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="",
                stderr=f"oc exec timed out after {timeout}s",
                returncode=-1,
                success=False,
            )
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=str(e),
                returncode=-1,
                success=False,
            )

    def read_file(self, path: str) -> CommandResult:
        return self.run(f"cat {shlex.quote(path)}")

    def write_file(self, path: str, content: str) -> CommandResult:
        return self.run(f"cat > {shlex.quote(path)} << 'EOFAGENT'\n{content}\nEOFAGENT")

    def test_connection(self) -> bool:
        result = self.run("echo ok", timeout=15)
        return result.success and "ok" in result.stdout


# ---------------------------------------------------------------------------
# Benchmark profiles (loaded from settings.yaml)
# ---------------------------------------------------------------------------


def _load_settings() -> dict:
    settings_path = Path(__file__).with_name("settings.yaml")
    with open(settings_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_benchmark_profiles(settings: dict) -> dict:
    profiles = {}
    for name, profile in (settings.get("profiles") or {}).items():
        isl = int(profile["input_sequence_length"])
        osl = int(profile["output_sequence_length"])
        samples = int(profile.get("samples", 100))
        profiles[name] = {
            "isl": isl,
            "osl": osl,
            "samples": samples,
            "description": profile.get("description", name),
            "data_flag": json.dumps(
                {"prompt_tokens": isl, "output_tokens": osl, "samples": samples}
            ),
        }
    return profiles


_SETTINGS = _load_settings()
BENCHMARK_PROFILES = _build_benchmark_profiles(_SETTINGS)
DEFAULT_CONCURRENCY_LEVELS = list(
    _SETTINGS.get("default_concurrency_levels") or [1, 50]
)
PROFILE_CHOICES = list(BENCHMARK_PROFILES.keys()) or [
    "balanced",
    "decode_heavy",
    "prefill_heavy",
    "long_context",
]


# ---------------------------------------------------------------------------
# Tool Definitions (for Claude's tool_use API)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "run_command",
        "description": (
            "Run a shell command on the vLLM host/pod for diagnostics. "
            "Use this to check GPU status, inspect vLLM configs, and read process state. "
            "The command runs on the remote target (SSH host or OpenShift pod). "
            "Never kill or restart the baseline pod. "
            "Optionally specify pod_name to run on an experiment pod instead of the baseline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute on the vLLM host/pod",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Command timeout in seconds. Default is 60. "
                        "Increase for long-running operations (e.g. model reload)."
                    ),
                    "default": 60,
                },
                "pod_name": {
                    "type": "string",
                    "description": (
                        "Optional: name of an experiment pod to run the command on "
                        "(created by create_vllm_pod). If omitted, runs on the baseline pod."
                    ),
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file on the vLLM host/pod. "
            "Use this to inspect configuration files, logs, profiler output, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file on the vLLM host/pod",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file on the vLLM host/pod. "
            "Use this to modify vLLM configuration, create scripts, "
            "or write profiler setup files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file on the vLLM host/pod",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_benchmark",
        "description": (
            "Run the LOCKED GuideLLM test against a vLLM endpoint from the LOCAL machine. "
            "Workload (profile, ISL/OSL, concurrency, duration) is fixed for the session; "
            "do not pick a different profile. An unscored warmup (fixed concurrency for a "
            "short duration, or N requests) runs automatically first. Then the scored run "
            "measures throughput, TTFT, ITL, TPOT. Pass endpoint only for experiment pods. "
            "For FINAL VERIFICATION of the best config, pass repeat=3 to run 3 times and "
            "get mean ± stddev — warmup runs before the first repeat only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": PROFILE_CHOICES,
                    "description": (
                        "Ignored when a session spec is locked. Kept for compatibility."
                    ),
                },
                "concurrency": {
                    "type": "string",
                    "description": (
                        "Ignored when a session spec is locked. Kept for compatibility."
                    ),
                },
                "endpoint": {
                    "type": "string",
                    "description": (
                        "vLLM endpoint URL. Auto-filled from CLI args if omitted."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": "Model name served by vLLM. Auto-filled from CLI args if omitted.",
                },
                "max_seconds": {
                    "type": "integer",
                    "description": (
                        "Ignored when a session spec is locked. Kept for compatibility."
                    ),
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Path to save the GuideLLM JSON results. "
                        "Default: ./benchmark_results/<profile>_<timestamp>.json"
                    ),
                },
                "repeat": {
                    "type": "integer",
                    "description": (
                        "Number of times to run the scored benchmark. Default: 1. "
                        "Use repeat=3 for FINAL VERIFICATION of the best config to get "
                        "mean ± stddev and confirm reproducibility. "
                        "Warmup runs only before the first repeat."
                    ),
                    "default": 1,
                },
            },
            "required": [],
        },
    },
    {
        "name": "analyze_trace",
        "description": (
            "Analyze a PyTorch profiler trace JSON file to extract kernel-level "
            "performance statistics. Returns top kernels by GPU time, category "
            "breakdown (attention, GEMM, normalization, etc.), and total GPU time. "
            "Use this after collecting a trace to identify performance bottlenecks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trace_json_path": {
                    "type": "string",
                    "description": (
                        "Path to the Chrome trace JSON file from PyTorch profiler. "
                        "This is a local file path (on the machine running the agent)."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top kernels to return. Default: 20",
                    "default": 20,
                },
            },
            "required": ["trace_json_path"],
        },
    },
    {
        "name": "map_kernel",
        "description": (
            "Map a CUDA kernel name to its source code location in vLLM or PyTorch. "
            "Returns the source file, function name, description, category, and whether "
            "it is a PyTorch standard library kernel. Use this to understand what a hot "
            "kernel does and where to look for optimization opportunities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_name": {
                    "type": "string",
                    "description": (
                        "CUDA kernel name from the profiler trace "
                        "(e.g. 'flash_fwd_kernel', 'rms_norm_kernel')"
                    ),
                },
            },
            "required": ["kernel_name"],
        },
    },
    {
        "name": "fetch_vllm_logs",
        "description": (
            "Fetch and parse vLLM server logs from the pod. Runs REMOTELY on the pod "
            "to collect log output, then parses it with 120+ regex patterns to extract "
            "structured information: server config (vLLM version, non-default args), "
            "engine config (dtype, quantization, TP/PP, CUDA graphs, chunked prefill), "
            "compilation (attention backend, torch.compile time, CUDA graph capture), "
            "memory (model memory, KV cache size, weights load time), "
            "timing (engine init time), and warnings/errors. "
            "IMPORTANT: Call this AFTER every benchmark to understand the server state. "
            "Optionally specify pod_name to fetch logs from an experiment pod."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "log_source": {
                    "type": "string",
                    "description": (
                        "How to collect logs. Options: "
                        "'process' (default) reads /proc/1/fd/1 for stdout of vLLM process, "
                        "'file' reads a specific log file path, "
                        "'dmesg' reads kernel messages for OOM/GPU errors."
                    ),
                    "enum": ["process", "file", "dmesg"],
                    "default": "process",
                },
                "log_path": {
                    "type": "string",
                    "description": (
                        "Path to log file on the pod (only used when log_source='file'). "
                        "Default: /tmp/vllm.log"
                    ),
                },
                "tail_lines": {
                    "type": "integer",
                    "description": "Number of recent log lines to fetch. Default: 200",
                    "default": 200,
                },
                "pod_name": {
                    "type": "string",
                    "description": (
                        "Optional: name of an experiment pod to fetch logs from "
                        "(created by create_vllm_pod). If omitted, uses the baseline pod."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_benchmark_results",
        "description": (
            "Read and parse a GuideLLM benchmark results JSON file from a previous "
            "run_benchmark call. Returns structured metrics per concurrency level: "
            "request totals (successful/errored), output tokens/sec, TTFT, ITL, TPOT "
            "with P50/P95/P99 percentiles, and request latency. "
            "Use this to review detailed metrics from a completed benchmark. "
            "The file path is shown in run_benchmark output as 'Results saved to: ...'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "results_path": {
                    "type": "string",
                    "description": "Path to the GuideLLM JSON results file (local path).",
                },
            },
            "required": ["results_path"],
        },
    },
    {
        "name": "compare_benchmarks",
        "description": (
            "Compare two benchmark runs to detect performance regressions or improvements. "
            "Takes paths to two GuideLLM JSON result files (baseline and current), "
            "extracts key metrics from each, and compares them with a configurable "
            "threshold (default 2%). Reports per-metric changes with direction "
            "(improvement/regression/neutral), accounting for metric directionality "
            "(higher throughput = better, lower latency = better). "
            "Use this after applying a tuning change to verify the impact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "baseline_path": {
                    "type": "string",
                    "description": "Path to the baseline GuideLLM JSON results file.",
                },
                "current_path": {
                    "type": "string",
                    "description": "Path to the current (post-tuning) GuideLLM JSON results file.",
                },
                "threshold": {
                    "type": "number",
                    "description": (
                        "Threshold for flagging a change as regression/improvement, "
                        "as a fraction (e.g. 0.02 = 2%). Default: 0.02"
                    ),
                    "default": 0.02,
                },
            },
            "required": ["baseline_path", "current_path"],
        },
    },
    {
        "name": "check_preemptions",
        "description": (
            "Query the vLLM Prometheus /metrics endpoint and return the preemption count. "
            "Preemptions happen when vLLM evicts KV cache entries under memory pressure, "
            "which severely hurts latency and throughput. If preemptions are occurring, "
            "further tuning is unlikely to help — the workload exceeds GPU memory capacity. "
            "Call this AFTER each benchmark to detect preemptions early."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": (
                        "vLLM endpoint URL to query. If omitted, uses the baseline endpoint. "
                        "For experiment pods, pass the endpoint returned by create_vllm_pod."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_vllm_pod",
        "description": (
            "Create a new experiment pod from the pod template with extra vLLM CLI args. "
            "The pod is created in the configured namespace, waits for readiness, and "
            "a port-forward is set up automatically. Returns the pod name and endpoint URL "
            "that can be passed to run_benchmark. Use this to test tuning parameters without "
            "modifying the baseline pod."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vllm_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Extra vLLM CLI args to add to the pod's launch command. "
                        'e.g. ["--enable-chunked-prefill", "--gpu-memory-utilization", "0.95"]'
                    ),
                },
            },
            "required": ["vllm_args"],
        },
    },
    {
        "name": "delete_vllm_pod",
        "description": (
            "Delete an experiment pod created by create_vllm_pod and clean up its "
            "port-forward. Call this after benchmarking an experiment pod to free resources."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pod_name": {
                    "type": "string",
                    "description": "Name of the experiment pod to delete (returned by create_vllm_pod).",
                },
            },
            "required": ["pod_name"],
        },
    },
    {
        "name": "done",
        "description": (
            "Signal that the tuning session is complete. Call this when you have "
            "achieved the performance target, exhausted viable tuning options, "
            "or want to report final results. Provide a summary of actions taken "
            "and results achieved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "Summary of what was done: tunings applied, benchmarks run, "
                        "performance changes observed, and final recommendations."
                    ),
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the performance target was achieved",
                },
            },
            "required": ["summary", "success"],
        },
    },
]


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------


def _handle_run_command(
    args: dict,
    executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Execute a shell command on the vLLM host/pod."""
    command = args["command"]
    timeout = args.get("timeout", 60)

    result = executor.run(command, timeout=timeout)

    command_history.append(
        {
            "tool": "run_command",
            "command": command,
            "success": result.success,
            "output": result.output[:2000],
        }
    )

    return ToolResult(
        tool="run_command",
        success=result.success,
        output=result.output,
        error=result.stderr if not result.success else None,
    )


def _handle_read_file(
    args: dict,
    executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Read a file from the vLLM host/pod."""
    path = args["path"]

    result = executor.read_file(path)

    command_history.append(
        {
            "tool": "read_file",
            "path": path,
            "success": result.success,
        }
    )

    return ToolResult(
        tool="read_file",
        success=result.success,
        output=result.output,
        error=result.stderr if not result.success else None,
    )


def _handle_write_file(
    args: dict,
    executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Write content to a file on the vLLM host/pod."""
    path = args["path"]
    content = args["content"]

    result = executor.write_file(path, content)

    command_history.append(
        {
            "tool": "write_file",
            "path": path,
            "success": result.success,
        }
    )

    return ToolResult(
        tool="write_file",
        success=result.success,
        output=result.output,
        error=result.stderr if not result.success else None,
    )


def _extract_guidellm_metrics(bench_data: dict) -> str:
    """Extract structured metrics from GuideLLM JSON output.

    Returns a human-readable summary of key performance metrics from each
    benchmark run (one per concurrency level).
    """
    lines = ["=== GUIDELLM METRICS SUMMARY ==="]

    benchmarks = bench_data.get("benchmarks", [])
    if not benchmarks:
        return "No benchmark data found in JSON output."

    for i, bench in enumerate(benchmarks):
        config = bench.get("config", {})
        strategy = config.get("strategy", {})
        conc = strategy.get("max_concurrency", strategy.get("worker_count", "?"))
        lines.append(f"\n--- Concurrency: {conc} ---")

        metrics = bench.get("metrics", {})

        # Request totals
        totals = metrics.get("request_totals", {})
        successful = totals.get("successful", 0)
        errored = totals.get("errored", 0)
        total = totals.get("total", 0)
        lines.append(
            f"  Requests: {successful} successful, {errored} errored, {total} total"
        )
        if total > 0:
            lines.append(f"  Success Rate: {successful / total * 100:.1f}%")

        if successful == 0:
            lines.append(
                "  WARNING: No successful requests — metrics below will be zeros."
            )
            lines.append("  → Check vLLM logs for errors (model loading, OOM, etc.)")

        # Throughput
        def _stat_line(label, stat_dict, keys=("mean", "median")):
            """Format a statistics dict into a readable line."""
            if not stat_dict:
                return f"  {label}: N/A"
            # Look for 'successful' sub-dict first (GuideLLM v0.5 format)
            d = stat_dict.get("successful", stat_dict)
            parts = []
            for k in keys:
                v = d.get(k)
                if v is not None:
                    parts.append(f"{k}={v:.2f}")
            # Add percentiles if present
            pcts = d.get("percentiles", {})
            for p in ("p50", "p95", "p99"):
                v = pcts.get(p)
                if v is not None:
                    parts.append(f"{p}={v:.2f}")
            return f"  {label}: {', '.join(parts)}" if parts else f"  {label}: N/A"

        lines.append(
            _stat_line("Output Tokens/sec", metrics.get("output_tokens_per_second"))
        )
        lines.append(
            _stat_line("Prompt Tokens/sec", metrics.get("prompt_tokens_per_second"))
        )
        lines.append(_stat_line("Total Tokens/sec", metrics.get("tokens_per_second")))
        lines.append(_stat_line("TTFT (ms)", metrics.get("time_to_first_token_ms")))
        lines.append(_stat_line("ITL (ms)", metrics.get("inter_token_latency_ms")))
        lines.append(_stat_line("TPOT (ms)", metrics.get("time_per_output_token_ms")))
        lines.append(_stat_line("Request Latency (s)", metrics.get("request_latency")))
        lines.append(_stat_line("Requests/sec", metrics.get("requests_per_second")))

        # Duration
        duration = bench.get("duration")
        if duration is not None:
            lines.append(f"  Duration: {duration:.1f}s")

    return "\n".join(lines)


def _guidellm_request_type(model: str, override: Optional[str] = None) -> str:
    if override:
        return override
    model_lower = model.lower()
    chat_indicators = ("chat", "instruct", "it-", "-it", "rlhf")
    if any(ind in model_lower for ind in chat_indicators):
        return "chat_completions"
    return "text_completions"


def _processor_name(model: str, processor: Optional[str] = None) -> str:
    name = processor or model
    if name.startswith("/models/"):
        return name[len("/models/") :]
    return name


def _build_guidellm_cmd(
    *,
    target_url: str,
    model: str,
    processor: str,
    request_type: str,
    concurrency: str,
    max_seconds: int,
    output_path: str,
    data_flag: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "guidellm",
        "benchmark",
        "run",
        f"--target={target_url}",
        f"--model={model}",
        f"--processor={processor}",
        "--rate-type=concurrent",
        f"--max-seconds={max_seconds}",
        f"--rate={concurrency}",
        f"--output-path={output_path}",
        f"--request-type={request_type}",
        "--processor-args",
        '{"trust-remote-code":"true"}',
        "--data",
        data_flag,
    ]


def _run_guidellm(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Run GuideLLM and stream stdout so the operator can see progress.

    GuideLLM's Rich progress bar rewrites a single line (no newline). A
    blocking ``readline()`` never returns, so the timeout never fires and
    orphaned workers keep hitting the experiment port-forward after the
    agent is killed. Read available bytes with ``select`` and kill the
    whole process group on timeout.
    """
    import os
    import select
    import signal
    import time

    def _kill_tree(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    chunks: list[bytes] = []
    deadline = time.time() + timeout
    try:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                _kill_tree(proc)
                raise subprocess.TimeoutExpired(cmd, timeout)
            ready, _, _ = select.select([fd], [], [], min(1.0, remaining))
            if ready:
                data = os.read(fd, 4096)
                if data:
                    chunks.append(data)
                    text = data.decode("utf-8", errors="replace")
                    for raw_line in text.splitlines():
                        if raw_line.strip():
                            print(f"   GuideLLM: {raw_line.rstrip()}", flush=True)
                    continue
            if proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    chunks.append(rest)
                    text = rest.decode("utf-8", errors="replace")
                    if text.strip():
                        print(text, end="", flush=True)
                break
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        raise
    out = b"".join(chunks).decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(cmd, proc.returncode or 0, out, "")


def _aggregate_repeat_metrics(
    all_flat_metrics: list[list[dict]],
    repeat: int,
) -> str:
    """Compute mean ± stddev across repeat runs, grouped by concurrency level."""
    import math

    conc_groups: dict[int, list[dict]] = {}
    for run_metrics in all_flat_metrics:
        for m in run_metrics:
            conc = m.get("concurrency", 0)
            conc_groups.setdefault(conc, []).append(m)

    lines = [f"=== AGGREGATED METRICS ({repeat} repeat runs) ==="]
    for conc in sorted(conc_groups.keys()):
        runs = conc_groups[conc]
        lines.append(f"\n--- Concurrency: {conc} (n={len(runs)}) ---")
        all_keys = sorted(
            {k for r in runs for k, v in r.items() if isinstance(v, (int, float)) and k != "concurrency"}
        )
        for key in all_keys:
            values = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
            if not values:
                continue
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                stddev = math.sqrt(variance)
                cv = stddev / mean * 100 if mean != 0 else 0
                lines.append(f"  {key}: mean={mean:.3f} ± {stddev:.3f}  (CV={cv:.1f}%)")
            else:
                lines.append(f"  {key}: {mean:.3f}")
    return "\n".join(lines)


def _log_benchmark_to_mlflow(
    mlflow_uri: str,
    experiment_name: str,
    *,
    model: str,
    profile: str,
    concurrency: str,
    is_baseline: bool,
    bench_data: dict,
    extra_tags: dict = None,
) -> None:
    """Log a benchmark run to MLflow. Silently skips if mlflow is not installed."""
    try:
        import mlflow
    except ImportError:
        return

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)
        flat_metrics_list = _extract_flat_metrics(bench_data)

        with mlflow.start_run():
            mlflow.set_tag("model", model)
            mlflow.set_tag("profile", profile)
            mlflow.set_tag("concurrency", concurrency)
            mlflow.set_tag("is_baseline", str(is_baseline))
            for k, v in (extra_tags or {}).items():
                mlflow.set_tag(str(k), str(v))

            for flat in flat_metrics_list:
                conc = flat.get("concurrency", 0)
                for key, value in flat.items():
                    if isinstance(value, (int, float)) and key != "concurrency":
                        mlflow.log_metric(f"{key}_conc{conc}", value)
    except Exception:
        pass  # never let MLflow errors crash the agent


def _handle_run_benchmark(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Warm up, then run the locked scored GuideLLM test against the endpoint."""
    import datetime
    import os
    import tempfile

    spec: Optional[BenchmarkSpec] = args.get("_benchmark_spec")
    endpoint = args["endpoint"]
    model = args["model"]

    if spec is None:
        profile_name = args.get("profile", "balanced")
        if profile_name not in BENCHMARK_PROFILES:
            return ToolResult(
                tool="run_benchmark",
                success=False,
                output="",
                error=(
                    f"Unknown profile '{profile_name}'. "
                    f"Choose from: {list(BENCHMARK_PROFILES.keys())}"
                ),
            )
        profile = BENCHMARK_PROFILES[profile_name]
        max_seconds = args.get("max_seconds", 60)
        concurrency = args.get(
            "concurrency", ",".join(str(c) for c in DEFAULT_CONCURRENCY_LEVELS)
        )
        data_flag = profile["data_flag"]
        description = profile["description"]
        warmup = None
    else:
        profile_name = spec.profile
        profile = BENCHMARK_PROFILES[profile_name]
        max_seconds = spec.max_seconds
        concurrency = spec.concurrency_csv
        data_flag = spec.data_flag
        description = spec.description
        warmup = spec.warmup if spec.warmup.enabled else None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        args.get("output_path")
        or f"./benchmark_results/{profile_name}_{timestamp}.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    target_url = endpoint.rstrip("/")
    if not target_url.endswith("/v1"):
        target_url += "/v1"

    processor = _processor_name(model, args.get("processor"))
    request_type = _guidellm_request_type(model, args.get("request_type"))

    repeat = max(1, int(args.get("repeat", 1)))
    mlflow_uri = args.get("_mlflow_uri")
    mlflow_experiment = args.get("_mlflow_experiment", "vllm-autotuning")
    is_baseline = bool(args.get("_is_baseline", False))

    command_history.append(
        {
            "tool": "run_benchmark",
            "profile": profile_name,
            "concurrency": concurrency,
            "endpoint": endpoint,
            "model": model,
            "warmup": warmup.describe() if warmup else "disabled",
            "repeat": repeat,
        }
    )

    n_levels = max(len(concurrency.split(",")), 1)
    # Cap wait to the scored duration plus setup/drain, not a 10min floor.
    bench_timeout = max_seconds * n_levels + 180

    try:
        # Warmup runs ONCE before all repeats — it already warms up the JIT.
        if warmup:
            if warmup.requests:
                warm_seconds = 300
                warm_samples = int(warmup.requests)
            else:
                warm_seconds = max(int(warmup.seconds), 5)
                warm_samples = max(
                    int(warmup.concurrency) * int(warmup.seconds) * 8, 256
                )
            isl = spec.isl if spec is not None else profile["isl"]
            osl = spec.osl if spec is not None else profile["osl"]
            warm_data = json.dumps(
                {"prompt_tokens": isl, "output_tokens": osl, "samples": warm_samples}
            )
            warm_fd, warm_path = tempfile.mkstemp(suffix="_warmup.json")
            os.close(warm_fd)
            warm_cmd = _build_guidellm_cmd(
                target_url=target_url,
                model=model,
                processor=processor,
                request_type=request_type,
                concurrency=str(warmup.concurrency),
                max_seconds=warm_seconds,
                output_path=warm_path,
                data_flag=warm_data,
            )
            print(f"   Warmup (unscored): {warmup.describe()}", flush=True)
            warm_result = _run_guidellm(warm_cmd, timeout=warm_seconds + 120)
            try:
                os.unlink(warm_path)
            except OSError:
                pass
            if warm_result.returncode != 0:
                err = (warm_result.stderr or warm_result.stdout or "warmup failed")[-2000:]
                command_history[-1]["success"] = False
                return ToolResult(
                    tool="run_benchmark",
                    success=False,
                    output="",
                    error=f"Warmup failed (scored run skipped):\n{err}",
                )
            print("   Warmup complete. Starting scored run...", flush=True)

        # Scored runs — loop for repeat.
        all_flat_metrics: list[list[dict]] = []
        last_bench_data: Optional[dict] = None
        last_output_path = output_path
        last_result = None

        for run_idx in range(repeat):
            run_output_path = (
                output_path.replace(".json", f"_r{run_idx}.json")
                if repeat > 1
                else output_path
            )
            last_output_path = run_output_path

            guidellm_cmd = _build_guidellm_cmd(
                target_url=target_url,
                model=model,
                processor=processor,
                request_type=request_type,
                concurrency=concurrency,
                max_seconds=max_seconds,
                output_path=run_output_path,
                data_flag=data_flag,
            )
            if run_idx == 0:
                command_history[-1]["command"] = " ".join(guidellm_cmd)

            if repeat > 1:
                print(f"   Scored run {run_idx + 1}/{repeat}...", flush=True)

            result = _run_guidellm(guidellm_cmd, timeout=bench_timeout)
            last_result = result

            if result.returncode == 0 and os.path.exists(run_output_path):
                try:
                    with open(run_output_path, encoding="utf-8") as fh:
                        bench_data = json.load(fh)
                    last_bench_data = bench_data
                    if repeat > 1:
                        all_flat_metrics.append(_extract_flat_metrics(bench_data))
                    if mlflow_uri:
                        _log_benchmark_to_mlflow(
                            mlflow_uri,
                            mlflow_experiment,
                            model=model,
                            profile=profile_name,
                            concurrency=concurrency,
                            is_baseline=is_baseline,
                            bench_data=bench_data,
                            extra_tags={
                                "repeat_index": run_idx,
                                "repeat_total": repeat,
                            },
                        )
                except (json.JSONDecodeError, OSError):
                    pass
            elif result.returncode != 0:
                break

        stdout = last_result.stdout.replace("\r\n", "\n").replace("\r", "\n")
        stderr = last_result.stderr.replace("\r\n", "\n").replace("\r", "\n")

        if last_result.returncode == 0:
            summary_parts = [
                f"LOCKED TEST: {profile_name} ({description})",
                f"ISL={profile['isl']} OSL={profile['osl']}",
                f"Scored concurrency: {concurrency}",
                f"Max seconds per level: {max_seconds}",
                f"Warmup: {warmup.describe() if warmup else 'disabled'}",
                f"Repeat: {repeat}",
                f"Results saved to: {last_output_path}",
            ]

            if last_bench_data:
                summary_parts.append("")
                summary_parts.append(_extract_guidellm_metrics(last_bench_data))

            if repeat > 1 and all_flat_metrics:
                summary_parts.append("")
                summary_parts.append(_aggregate_repeat_metrics(all_flat_metrics, repeat))

            summary_parts.append("")
            summary_parts.append("--- GuideLLM stdout (last 2000 chars) ---")
            summary_parts.append(stdout[-2000:] if len(stdout) > 2000 else stdout)
            output_text = "\n".join(summary_parts)
        else:
            output_text = (
                f"GuideLLM exited with code {last_result.returncode}\n"
                f"stdout:\n{stdout[-2000:]}\n"
                f"stderr:\n{stderr[-2000:]}"
            )

        command_history[-1]["success"] = last_result.returncode == 0

        return ToolResult(
            tool="run_benchmark",
            success=last_result.returncode == 0,
            output=output_text,
            error=stderr if last_result.returncode != 0 else None,
        )

    except subprocess.TimeoutExpired:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="run_benchmark",
            success=False,
            output="",
            error="GuideLLM timed out",
        )
    except FileNotFoundError:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="run_benchmark",
            success=False,
            output="",
            error=(
                "GuideLLM is not installed or not on PATH. "
                "Install with: pip install guidellm"
            ),
        )
    except Exception as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="run_benchmark",
            success=False,
            output="",
            error=f"Failed to run GuideLLM: {e}",
        )


def _handle_analyze_trace(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Analyze a PyTorch profiler trace JSON file.

    Loads the Chrome trace from disk and delegates to
    ``analysis.trace_analyzer.analyze_trace``. Falls back to a basic JSON
    parse if the analyzer cannot be imported.
    """
    trace_path = args["trace_json_path"]
    top_n = args.get("top_n", 20)

    command_history.append(
        {
            "tool": "analyze_trace",
            "trace_json_path": trace_path,
        }
    )

    try:
        from .analysis.trace_analyzer import analyze_trace as analyze_trace_fn

        if not os.path.exists(trace_path):
            command_history[-1]["success"] = False
            return ToolResult(
                tool="analyze_trace",
                success=False,
                output="",
                error=f"Trace file not found: {trace_path}",
            )
        with open(trace_path, encoding="utf-8") as fh:
            trace_data = json.load(fh)
        result = analyze_trace_fn(trace_data, top_n=top_n)
        command_history[-1]["success"] = True
        return ToolResult(
            tool="analyze_trace",
            success=True,
            output=json.dumps(result, indent=2, default=str),
        )
    except (ImportError, AttributeError, json.JSONDecodeError, OSError):
        pass

    # Fallback: basic trace parsing
    try:
        if not os.path.exists(trace_path):
            command_history[-1]["success"] = False
            return ToolResult(
                tool="analyze_trace",
                success=False,
                output="",
                error=f"Trace file not found: {trace_path}",
            )

        with open(trace_path, "r") as f:
            trace_data = json.load(f)

        # Basic Chrome trace event parsing
        events = (
            trace_data
            if isinstance(trace_data, list)
            else trace_data.get("traceEvents", [])
        )
        gpu_events = [
            e
            for e in events
            if isinstance(e, dict)
            and e.get("cat") in ("kernel", "gpu_memcpy", "cuda_runtime")
            and e.get("dur", 0) > 0
        ]

        # Aggregate by kernel name
        kernel_stats: dict[str, dict] = {}
        for e in gpu_events:
            name = e.get("name", "unknown")
            dur_us = e.get("dur", 0)
            if name not in kernel_stats:
                kernel_stats[name] = {
                    "count": 0,
                    "total_dur_us": 0,
                    "min_dur_us": dur_us,
                    "max_dur_us": dur_us,
                }
            stats = kernel_stats[name]
            stats["count"] += 1
            stats["total_dur_us"] += dur_us
            stats["min_dur_us"] = min(stats["min_dur_us"], dur_us)
            stats["max_dur_us"] = max(stats["max_dur_us"], dur_us)

        # Sort by total duration and take top N
        sorted_kernels = sorted(
            kernel_stats.items(), key=lambda x: x[1]["total_dur_us"], reverse=True
        )
        top_kernels = [
            {
                "kernel": name,
                **stats,
                "avg_dur_us": round(stats["total_dur_us"] / stats["count"], 2),
            }
            for name, stats in sorted_kernels[:top_n]
        ]

        total_gpu_time_us = sum(s["total_dur_us"] for s in kernel_stats.values())

        result = {
            "trace_file": trace_path,
            "total_events": len(events),
            "gpu_events": len(gpu_events),
            "unique_kernels": len(kernel_stats),
            "total_gpu_time_us": total_gpu_time_us,
            "total_gpu_time_ms": round(total_gpu_time_us / 1000, 2),
            "top_kernels": top_kernels,
            "note": "Basic fallback parser (trace_analyzer module not yet populated)",
        }

        command_history[-1]["success"] = True
        return ToolResult(
            tool="analyze_trace",
            success=True,
            output=json.dumps(result, indent=2),
        )

    except json.JSONDecodeError as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="analyze_trace",
            success=False,
            output="",
            error=f"Failed to parse trace JSON: {e}",
        )
    except Exception as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="analyze_trace",
            success=False,
            output="",
            error=f"Failed to analyze trace: {e}",
        )


def _handle_map_kernel(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Map a CUDA kernel name to its source code location.

    Delegates to agent.analysis.kernel_mapper.  Falls back to a stub
    response if the module is not yet populated.
    """
    kernel_name = args["kernel_name"]

    command_history.append(
        {
            "tool": "map_kernel",
            "kernel_name": kernel_name,
        }
    )

    try:
        from .analysis import kernel_mapper

        if hasattr(kernel_mapper, "find_kernel_mapping"):
            mapping = kernel_mapper.find_kernel_mapping(kernel_name)
            is_stdlib = False
            if hasattr(kernel_mapper, "is_pytorch_stdlib"):
                is_stdlib = kernel_mapper.is_pytorch_stdlib(kernel_name)
            result = {
                "kernel_name": kernel_name,
                "mapping": mapping,
                "is_pytorch_stdlib": is_stdlib,
            }
            command_history[-1]["success"] = True
            return ToolResult(
                tool="map_kernel",
                success=True,
                output=json.dumps(result, indent=2, default=str),
            )
    except (ImportError, AttributeError):
        pass

    # Fallback: pattern-based heuristic when the full mapper is not available
    result = _fallback_kernel_mapping(kernel_name)
    command_history[-1]["success"] = True
    return ToolResult(
        tool="map_kernel",
        success=True,
        output=json.dumps(result, indent=2),
    )


def _fallback_kernel_mapping(kernel_name: str) -> dict:
    """Simple heuristic kernel mapping when kernel_mapper is not populated."""
    name_lower = kernel_name.lower()

    # Pattern-based classification
    patterns = [
        (
            ["flash_fwd", "flash_bwd", "flash_attn"],
            "attention",
            "Flash Attention kernel",
            "vllm/attention/",
        ),
        (
            ["paged_attention", "paged_attn"],
            "attention",
            "PagedAttention kernel",
            "vllm/attention/",
        ),
        (["fmha"], "attention", "Fused multi-head attention", "vllm/attention/"),
        (
            ["cutlass", "gemm", "cublas", "matmul", "sgemm", "hgemm"],
            "linear/gemm",
            "Matrix multiplication kernel",
            "torch or cutlass",
        ),
        (
            ["rms_norm", "rmsnorm"],
            "normalization",
            "RMS normalization kernel",
            "vllm/model_executor/layers/",
        ),
        (
            ["layer_norm", "layernorm"],
            "normalization",
            "Layer normalization kernel",
            "torch/nn/",
        ),
        (
            ["silu", "gelu", "relu", "swiglu"],
            "activation",
            "Activation function kernel",
            "vllm/model_executor/layers/",
        ),
        (
            ["rotary", "rope"],
            "positional_encoding",
            "Rotary positional embedding kernel",
            "vllm/model_executor/layers/",
        ),
        (["memcpy", "memset"], "memory", "Memory operation", "CUDA runtime"),
        (
            ["nccl", "allreduce", "allgather"],
            "communication",
            "Collective communication kernel",
            "NCCL",
        ),
        (["softmax"], "attention", "Softmax kernel", "torch or custom"),
        (
            ["elementwise", "binary", "unary"],
            "elementwise",
            "Elementwise operation",
            "torch",
        ),
        (
            ["topk", "sampling", "argmax"],
            "sampling",
            "Sampling/selection kernel",
            "vllm/model_executor/layers/sampler",
        ),
        (
            ["quantize", "dequantize", "fp8", "awq", "gptq"],
            "quantization",
            "Quantization kernel",
            "vllm/model_executor/layers/quantization/",
        ),
    ]

    for keywords, category, description, source_hint in patterns:
        if any(kw in name_lower for kw in keywords):
            return {
                "kernel_name": kernel_name,
                "category": category,
                "description": description,
                "source_hint": source_hint,
                "is_pytorch_stdlib": any(
                    kw in name_lower for kw in ["cublas", "memcpy", "memset", "nccl"]
                ),
                "note": "Heuristic mapping (kernel_mapper module not yet populated)",
            }

    return {
        "kernel_name": kernel_name,
        "category": "unknown",
        "description": "Kernel not recognized by heuristic mapper",
        "source_hint": "unknown",
        "is_pytorch_stdlib": False,
        "note": "Heuristic mapping (kernel_mapper module not yet populated)",
    }


def _handle_fetch_vllm_logs(
    args: dict,
    executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Fetch vLLM logs from the pod and parse them into structured data.

    Runs remotely on the pod to collect log text, then applies the vLLM log
    parser (120+ regex patterns) to extract server config, engine config,
    compilation, memory, timing, and warnings.
    """
    log_source = args.get("log_source", "process")
    tail_lines = args.get("tail_lines", 200)

    command_history.append(
        {
            "tool": "fetch_vllm_logs",
            "log_source": log_source,
            "tail_lines": tail_lines,
        }
    )

    # Prefer `oc logs` on OpenShift: `cat /proc/1/fd/1` never EOFs on a live
    # vLLM process, so oc exec hits the 30s timeout and returns nothing useful.
    if isinstance(executor, OcExecutor):
        oc_cmd = ["oc"]
        if executor.kubeconfig:
            oc_cmd += ["--kubeconfig", executor.kubeconfig]
        oc_cmd += [
            "logs",
            "-n",
            executor.namespace,
            executor.pod_name,
            f"--tail={tail_lines}",
            "-c",
            executor.container or "vllm",
        ]
        try:
            result = subprocess.run(
                oc_cmd, capture_output=True, text=True, timeout=30
            )
            result = CommandResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                success=result.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            result = CommandResult(
                stdout="",
                stderr="oc logs timed out after 30s",
                returncode=-1,
                success=False,
            )
    else:
        if log_source == "file":
            log_path = args.get("log_path", "/tmp/vllm.log")
            cmd = f"tail -{tail_lines} {log_path} 2>/dev/null || echo 'Log file not found: {log_path}'"
        elif log_source == "dmesg":
            cmd = f"dmesg | tail -{tail_lines} 2>/dev/null || echo 'dmesg not available'"
        else:
            cmd = (
                f"(timeout 5 tail -n {tail_lines} /proc/1/fd/1 2>/dev/null) || "
                f"(tail -{tail_lines} /tmp/vllm*.log 2>/dev/null) || "
                f"echo 'No vLLM logs found. Try log_source=file with a specific path.'"
            )
        result = executor.run(cmd, timeout=30)

    if not result.success:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="fetch_vllm_logs",
            success=False,
            output="",
            error=f"Failed to fetch logs: {result.stderr}",
        )

    raw_logs = result.stdout

    # Also fetch current vLLM launch args
    cmdline_result = executor.run(
        "cat /proc/1/cmdline 2>/dev/null | tr '\\0' ' '", timeout=10
    )
    cmdline = cmdline_result.stdout.strip() if cmdline_result.success else ""

    # Parse logs with the structured parser
    try:
        from .analysis.vllm_log_parser import parse_vllm_log

        parsed = parse_vllm_log(raw_logs)
    except Exception as e:
        parsed = {"parse_error": str(e)}

    # Add cmdline to parsed result
    if cmdline:
        parsed.setdefault("server_config", {})["cmdline"] = cmdline

    # Build output
    output_parts = ["=== PARSED VLLM LOG ANALYSIS ==="]

    for section, data in parsed.items():
        if section == "warnings_errors":
            if data:
                output_parts.append(f"\n--- Warnings/Errors ({len(data)} found) ---")
                for w in data[:20]:  # Limit to 20
                    output_parts.append(f"  {w[:200]}")
        elif isinstance(data, dict):
            output_parts.append(f"\n--- {section.replace('_', ' ').title()} ---")
            for k, v in data.items():
                if k == "non_default_args" and isinstance(v, dict):
                    output_parts.append(f"  {k}:")
                    for ak, av in v.items():
                        output_parts.append(f"    {ak}: {av}")
                else:
                    output_parts.append(f"  {k}: {v}")

    output_parts.append(
        f"\n--- Raw Log Tail ({min(tail_lines, len(raw_logs.splitlines()))} lines) ---"
    )
    # Include last 50 lines of raw logs for context
    raw_tail = "\n".join(raw_logs.splitlines()[-50:])
    output_parts.append(raw_tail)

    output_text = "\n".join(output_parts)
    command_history[-1]["success"] = True
    command_history[-1]["parsed_sections"] = list(parsed.keys())

    return ToolResult(
        tool="fetch_vllm_logs",
        success=True,
        output=output_text,
    )


def _extract_flat_metrics(bench_data: dict) -> list[dict]:
    """Extract flat metric dicts per concurrency level from GuideLLM JSON.

    Returns a list of dicts, one per benchmark (concurrency level), with
    flat key-value pairs suitable for comparison.
    """
    results = []
    benchmarks = bench_data.get("benchmarks", [])

    for bench in benchmarks:
        config = bench.get("config", {})
        strategy = config.get("strategy", {})
        conc = strategy.get("max_concurrency", strategy.get("worker_count", 0))
        metrics = bench.get("metrics", {})

        flat: dict = {"concurrency": conc}

        # Request totals
        totals = metrics.get("request_totals", {})
        flat["successful_requests"] = totals.get("successful", 0)
        flat["errored_requests"] = totals.get("errored", 0)
        flat["total_requests"] = totals.get("total", 0)

        # Extract mean and percentiles from each metric
        metric_keys = [
            ("output_tokens_per_second", "output_tok/sec"),
            ("prompt_tokens_per_second", "prompt_tok/sec"),
            ("tokens_per_second", "total_tok/sec"),
            ("time_to_first_token_ms", "ttft"),
            ("inter_token_latency_ms", "itl"),
            ("time_per_output_token_ms", "tpot"),
            ("request_latency", "request_latency"),
            ("requests_per_second", "requests/sec"),
        ]

        for src_key, dst_prefix in metric_keys:
            stat_dict = metrics.get(src_key, {})
            d = stat_dict.get("successful", stat_dict)
            if isinstance(d, dict):
                for stat in ("mean", "median"):
                    v = d.get(stat)
                    if v is not None:
                        flat[f"{dst_prefix}_{stat}"] = round(v, 4)
                pcts = d.get("percentiles", {})
                for p in ("p50", "p95", "p99"):
                    v = pcts.get(p)
                    if v is not None:
                        flat[f"{dst_prefix}_{p}"] = round(v, 4)

        flat["duration_s"] = bench.get("duration")
        results.append(flat)

    return results


def _handle_read_benchmark_results(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Read and parse a GuideLLM benchmark JSON file into structured metrics."""
    import os

    results_path = args["results_path"]

    command_history.append(
        {
            "tool": "read_benchmark_results",
            "results_path": results_path,
        }
    )

    if not os.path.exists(results_path):
        command_history[-1]["success"] = False
        return ToolResult(
            tool="read_benchmark_results",
            success=False,
            output="",
            error=f"Results file not found: {results_path}",
        )

    try:
        with open(results_path, "r") as f:
            bench_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="read_benchmark_results",
            success=False,
            output="",
            error=f"Failed to parse JSON: {e}",
        )

    # Extract structured metrics
    flat_metrics = _extract_flat_metrics(bench_data)

    # Also get the GuideLLM summary
    guidellm_summary = _extract_guidellm_metrics(bench_data)

    # Build output
    output_parts = [guidellm_summary, ""]
    output_parts.append("=== FLAT METRICS PER CONCURRENCY (for compare_benchmarks) ===")
    for fm in flat_metrics:
        output_parts.append(json.dumps(fm, indent=2))

    output_text = "\n".join(output_parts)
    command_history[-1]["success"] = True

    return ToolResult(
        tool="read_benchmark_results",
        success=True,
        output=output_text,
    )


def _handle_compare_benchmarks(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Compare two GuideLLM benchmark runs and detect regressions/improvements."""
    import os

    baseline_path = args["baseline_path"]
    current_path = args["current_path"]
    threshold = args.get("threshold", 0.02)

    command_history.append(
        {
            "tool": "compare_benchmarks",
            "baseline_path": baseline_path,
            "current_path": current_path,
            "threshold": threshold,
        }
    )

    # Load both files
    for path, label in [(baseline_path, "Baseline"), (current_path, "Current")]:
        if not os.path.exists(path):
            command_history[-1]["success"] = False
            return ToolResult(
                tool="compare_benchmarks",
                success=False,
                output="",
                error=f"{label} file not found: {path}",
            )

    try:
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
        with open(current_path, "r") as f:
            current_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="compare_benchmarks",
            success=False,
            output="",
            error=f"Failed to parse JSON: {e}",
        )

    # Extract flat metrics
    baseline_metrics_list = _extract_flat_metrics(baseline_data)
    current_metrics_list = _extract_flat_metrics(current_data)

    if not baseline_metrics_list or not current_metrics_list:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="compare_benchmarks",
            success=False,
            output="",
            error="One or both benchmark files contain no benchmark data.",
        )

    # Import regression detector
    try:
        from .analysis.regression import detect_regression
    except ImportError:
        # Inline minimal comparison if module not available
        detect_regression = None

    output_parts = ["=== BENCHMARK COMPARISON ==="]
    output_parts.append(f"Baseline: {baseline_path}")
    output_parts.append(f"Current:  {current_path}")
    output_parts.append(f"Threshold: {threshold * 100:.1f}%")

    # Compare each concurrency level
    baseline_by_conc = {m.get("concurrency", 0): m for m in baseline_metrics_list}
    current_by_conc = {m.get("concurrency", 0): m for m in current_metrics_list}

    common_conc = sorted(set(baseline_by_conc.keys()) & set(current_by_conc.keys()))

    if not common_conc:
        # Fall back to comparing first entry from each
        output_parts.append("\nNo common concurrency levels. Comparing first entries.")
        common_conc = [baseline_metrics_list[0].get("concurrency", 0)]
        current_by_conc[common_conc[0]] = current_metrics_list[0]

    all_comparisons = []
    for conc in common_conc:
        baseline_flat = baseline_by_conc.get(conc, baseline_metrics_list[0])
        current_flat = current_by_conc.get(conc, current_metrics_list[0])

        output_parts.append(f"\n--- Concurrency: {conc} ---")

        if detect_regression is not None:
            result = detect_regression(baseline_flat, current_flat, threshold=threshold)
            if result.get("status") == "success":
                output_parts.append(f"  Verdict: {result['summary']['verdict']}")
                output_parts.append(f"  {result['message']}")

                for comp in result.get("regressions", []):
                    output_parts.append(
                        f"  REGRESSION: {comp['metric']}: "
                        f"{comp['baseline_value']} -> {comp['current_value']} "
                        f"({comp['percent_change']:+.1f}%)"
                    )
                for comp in result.get("improvements", []):
                    output_parts.append(
                        f"  IMPROVED: {comp['metric']}: "
                        f"{comp['baseline_value']} -> {comp['current_value']} "
                        f"({comp['percent_change']:+.1f}%)"
                    )
                all_comparisons.append(result)
            else:
                output_parts.append(f"  Error: {result.get('message', 'unknown')}")
        else:
            # Manual comparison
            for key in sorted(set(baseline_flat.keys()) & set(current_flat.keys())):
                bv = baseline_flat[key]
                cv = current_flat[key]
                if (
                    isinstance(bv, (int, float))
                    and isinstance(cv, (int, float))
                    and bv != 0
                ):
                    pct = ((cv - bv) / bv) * 100
                    output_parts.append(f"  {key}: {bv} -> {cv} ({pct:+.1f}%)")

    output_text = "\n".join(output_parts)
    command_history[-1]["success"] = True

    return ToolResult(
        tool="compare_benchmarks",
        success=True,
        output=output_text,
    )


def _handle_check_preemptions(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Query vLLM /metrics endpoint for preemption count."""
    import re
    import urllib.error
    import urllib.request

    endpoint = args.get("endpoint", "").rstrip("/")
    if not endpoint:
        return ToolResult(
            tool="check_preemptions",
            success=False,
            output="",
            error="No endpoint provided and no default available. Pass endpoint explicitly.",
        )

    metrics_url = f"{endpoint}/metrics"

    command_history.append(
        {
            "tool": "check_preemptions",
            "endpoint": endpoint,
        }
    )

    try:
        req = urllib.request.Request(metrics_url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
    except Exception as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="check_preemptions",
            success=False,
            output="",
            error=f"Failed to fetch {metrics_url}: {e}",
        )

    # Parse preemption counter
    preemption_count = 0.0
    for line in body.splitlines():
        if line.startswith("vllm:num_preemptions_total"):
            m = re.search(r"}\s+([\d.eE+\-]+)", line)
            if m:
                preemption_count += float(m.group(1))

    command_history[-1]["success"] = True
    command_history[-1]["preemption_count"] = preemption_count

    if preemption_count > 0:
        output = (
            f"WARNING: {int(preemption_count)} preemption(s) detected!\n"
            f"vLLM is evicting KV cache entries under memory pressure.\n"
            f"This means the workload exceeds available GPU memory for the current config.\n"
            f"Further tuning is unlikely to help. Consider:\n"
            f"  - Reducing max-num-seqs or max-num-batched-tokens\n"
            f"  - Reducing max-model-len\n"
            f"  - Using quantization to free memory\n"
            f"  - If preemptions persist across configs, stop tuning and report findings."
        )
    else:
        output = "No preemptions detected. KV cache pressure is within limits."

    return ToolResult(
        tool="check_preemptions",
        success=True,
        output=output,
    )


def _handle_create_vllm_pod(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
    *,
    pod_manager: Optional[PodManager] = None,
) -> ToolResult:
    """Create a new experiment pod with extra vLLM CLI args."""
    vllm_args = args["vllm_args"]

    command_history.append(
        {
            "tool": "create_vllm_pod",
            "vllm_args": vllm_args,
        }
    )

    if pod_manager is None:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="create_vllm_pod",
            success=False,
            output="",
            error="PodManager not configured. Use --pod-template and --oc-mode to enable pod management.",
        )

    try:
        pod_name, endpoint = pod_manager.create_pod(vllm_args)
        command_history[-1]["success"] = True
        command_history[-1]["pod_name"] = pod_name
        command_history[-1]["endpoint"] = endpoint
        return ToolResult(
            tool="create_vllm_pod",
            success=True,
            output=(
                f"Experiment pod created successfully.\n"
                f"  Pod name: {pod_name}\n"
                f"  Endpoint: {endpoint}\n"
                f"  vLLM args: {' '.join(vllm_args)}\n\n"
                f'Use this endpoint when calling run_benchmark (pass endpoint="{endpoint}").\n'
                f'Use pod_name="{pod_name}" with run_command or fetch_vllm_logs to inspect the pod.\n'
                f'Call delete_vllm_pod(pod_name="{pod_name}") when done.'
            ),
        )
    except Exception as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="create_vllm_pod",
            success=False,
            output="",
            error=f"Failed to create experiment pod: {e}",
        )


def _handle_delete_vllm_pod(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
    *,
    pod_manager: Optional[PodManager] = None,
) -> ToolResult:
    """Delete an experiment pod and clean up its port-forward."""
    pod_name = args["pod_name"]

    command_history.append(
        {
            "tool": "delete_vllm_pod",
            "pod_name": pod_name,
        }
    )

    if pod_manager is None:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="delete_vllm_pod",
            success=False,
            output="",
            error="PodManager not configured.",
        )

    try:
        pod_manager.delete_pod(pod_name)
        command_history[-1]["success"] = True
        return ToolResult(
            tool="delete_vllm_pod",
            success=True,
            output=f"Pod {pod_name} deleted and port-forward cleaned up.",
        )
    except Exception as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="delete_vllm_pod",
            success=False,
            output="",
            error=f"Failed to delete pod {pod_name}: {e}",
        )


def _handle_done(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Signal that the agent has completed its work."""
    summary = args["summary"]
    success = args.get("success", False)

    command_history.append(
        {
            "tool": "done",
            "summary": summary,
            "success": success,
        }
    )

    return ToolResult(
        tool="done",
        success=success,
        output=summary,
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "run_command": _handle_run_command,
    "read_file": _handle_read_file,
    "write_file": _handle_write_file,
    "run_benchmark": _handle_run_benchmark,
    "fetch_vllm_logs": _handle_fetch_vllm_logs,
    "read_benchmark_results": _handle_read_benchmark_results,
    "compare_benchmarks": _handle_compare_benchmarks,
    "analyze_trace": _handle_analyze_trace,
    "map_kernel": _handle_map_kernel,
    "check_preemptions": _handle_check_preemptions,
    "create_vllm_pod": _handle_create_vllm_pod,
    "delete_vllm_pod": _handle_delete_vllm_pod,
    "done": _handle_done,
}

# Handlers that accept pod_manager as a keyword argument
_POD_MANAGER_HANDLERS = {"create_vllm_pod", "delete_vllm_pod"}

# Handlers that accept an executor override via pod_name arg
_POD_AWARE_HANDLERS = {"run_command", "fetch_vllm_logs"}


def dispatch_tool(
    name: str,
    args: dict,
    executor: RemoteExecutor,
    command_history: Optional[list[dict]] = None,
    *,
    pod_manager: Optional[PodManager] = None,
    namespace: Optional[str] = None,
    kubeconfig: Optional[str] = None,
) -> ToolResult:
    """Route a tool call to the appropriate handler.

    Parameters
    ----------
    name : str
        Tool name (must match one of the keys in TOOL_DEFINITIONS).
    args : dict
        Tool input arguments from the Claude API response.
    executor : RemoteExecutor
        The configured executor (SSHExecutor or OcExecutor) for remote commands.
    command_history : list[dict] or None
        Mutable list to track command history across the agent session.
        If None, a temporary list is used (history is discarded).
    pod_manager : PodManager or None
        Pod manager for create/delete pod operations.
    namespace : str or None
        OpenShift namespace (used to create temp OcExecutors for experiment pods).
    kubeconfig : str or None
        Kubeconfig path (used to create temp OcExecutors for experiment pods).

    Returns
    -------
    ToolResult
        Result of the tool execution.
    """
    if command_history is None:
        command_history = []

    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return ToolResult(
            tool=name,
            success=False,
            output="",
            error=f"Unknown tool: '{name}'. Available tools: {list(_TOOL_HANDLERS.keys())}",
        )

    # For pod manager tools, pass pod_manager as keyword arg
    if name in _POD_MANAGER_HANDLERS:
        return handler(args, executor, command_history, pod_manager=pod_manager)

    # For pod-aware tools, create a temp OcExecutor if pod_name is specified
    target_executor = executor
    pod_name = args.pop("pod_name", None) if name in _POD_AWARE_HANDLERS else None
    if pod_name and namespace:
        target_executor = OcExecutor(
            namespace=namespace,
            pod_name=pod_name,
            kubeconfig=kubeconfig,
        )

    return handler(args, target_executor, command_history)


# ---------------------------------------------------------------------------
# Convenience: AgentTools class (higher-level wrapper, optional)
# ---------------------------------------------------------------------------


class AgentTools:
    """Higher-level wrapper that bundles an executor with tool dispatch.

    Provides a class-based interface similar to the original ai-perf-hackathon
    AgentTools, but backed by the RemoteExecutor abstraction.
    """

    def __init__(
        self,
        executor: RemoteExecutor,
        vllm_endpoint: str = "http://localhost:8000",
        model_name: str = "",
        pod_manager: Optional[PodManager] = None,
        namespace: Optional[str] = None,
        kubeconfig: Optional[str] = None,
        benchmark_spec: Optional[BenchmarkSpec] = None,
        mlflow_uri: Optional[str] = None,
        mlflow_experiment: str = "vllm-autotuning",
    ):
        self.executor = executor
        self.vllm_endpoint = vllm_endpoint
        self.model_name = model_name
        self.pod_manager = pod_manager
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.benchmark_spec = benchmark_spec
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment = mlflow_experiment
        self.command_history: list[dict] = []

    def get_tool_definitions(self) -> list[dict]:
        """Return the tool definitions list for Claude's API."""
        return TOOL_DEFINITIONS

    def dispatch(self, name: str, args: dict) -> ToolResult:
        """Dispatch a tool call, tracking history on this instance.

        For run_benchmark, uses the agent-supplied endpoint if provided,
        otherwise falls back to the CLI-provided baseline endpoint.
        Model name is always filled from CLI args.
        """
        # Copy so we never mutate Claude's tool_use input (that object is
        # stored in messages and must stay JSON-serializable).
        args = dict(args)
        if name == "run_benchmark":
            if "endpoint" not in args or not args.get("endpoint"):
                args["endpoint"] = self.vllm_endpoint  # baseline default
            args["model"] = self.model_name
            if self.benchmark_spec is not None:
                args["_benchmark_spec"] = self.benchmark_spec
                args["profile"] = self.benchmark_spec.profile
                args["concurrency"] = self.benchmark_spec.concurrency_csv
                args["max_seconds"] = self.benchmark_spec.max_seconds
            # MLflow — injected as internal params (not exposed to Claude's tool schema)
            if self.mlflow_uri:
                args["_mlflow_uri"] = self.mlflow_uri
                args["_mlflow_experiment"] = self.mlflow_experiment
                args["_is_baseline"] = (
                    not args.get("endpoint") or args["endpoint"] == self.vllm_endpoint
                )
        elif name == "check_preemptions":
            if "endpoint" not in args or not args.get("endpoint"):
                args["endpoint"] = self.vllm_endpoint  # baseline default
        return dispatch_tool(
            name,
            args,
            self.executor,
            self.command_history,
            pod_manager=self.pod_manager,
            namespace=self.namespace,
            kubeconfig=self.kubeconfig,
        )

    # Convenience methods for direct (non-agent) use

    def run_command(self, command: str, timeout: int = 60) -> ToolResult:
        return self.dispatch("run_command", {"command": command, "timeout": timeout})

    def read_file(self, path: str) -> ToolResult:
        return self.dispatch("read_file", {"path": path})

    def write_file(self, path: str, content: str) -> ToolResult:
        return self.dispatch("write_file", {"path": path, "content": content})

    def run_benchmark(
        self,
        profile: str,
        endpoint: str,
        model: str,
        concurrency: Optional[str] = None,
        max_seconds: int = 120,
        output_path: Optional[str] = None,
    ) -> ToolResult:
        args: dict = {
            "profile": profile,
            "endpoint": endpoint,
            "model": model,
            "max_seconds": max_seconds,
        }
        if concurrency:
            args["concurrency"] = concurrency
        if output_path:
            args["output_path"] = output_path
        return self.dispatch("run_benchmark", args)

    def analyze_trace(self, trace_json_path: str, top_n: int = 20) -> ToolResult:
        return self.dispatch(
            "analyze_trace", {"trace_json_path": trace_json_path, "top_n": top_n}
        )

    def map_kernel(self, kernel_name: str) -> ToolResult:
        return self.dispatch("map_kernel", {"kernel_name": kernel_name})

    def done(self, summary: str, success: bool = False) -> ToolResult:
        return self.dispatch("done", {"summary": summary, "success": success})

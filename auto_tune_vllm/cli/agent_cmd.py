"""Typer command for the Claude-driven agentic vLLM tuner."""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Optional

import typer
from rich.console import Console

console = Console()


def agent_command(
    vllm_endpoint: str = typer.Option(
        ..., "--vllm-endpoint", help="URL of the running vLLM server"
    ),
    model: str = typer.Option(..., "--model", help="Model served by vLLM"),
    vllm_host: Optional[str] = typer.Option(
        None, "--vllm-host", help="SSH hostname (required unless --oc-mode)"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Anthropic API key (default: $ANTHROPIC_API_KEY)"
    ),
    claude_model: str = typer.Option(
        "sonnet", "--claude-model", help="Claude model: sonnet, opus, haiku, or full ID"
    ),
    max_iterations: int = typer.Option(
        100, "--max-iterations", help="Max agent loop iterations"
    ),
    profiles: Optional[list[str]] = typer.Option(
        None, "--profiles", help="Locked workload profile (first value is used)"
    ),
    concurrency: Optional[str] = typer.Option(
        None, "--concurrency", help="Scored concurrency levels, comma-separated"
    ),
    max_seconds: Optional[int] = typer.Option(
        None, "--max-seconds", help="Scored seconds per concurrency level"
    ),
    warmup_concurrency: Optional[int] = typer.Option(
        None, "--warmup-concurrency", help="Fixed concurrency for unscored warmup"
    ),
    warmup_seconds: Optional[int] = typer.Option(
        None, "--warmup-seconds", help="Unscored warmup duration in seconds"
    ),
    warmup_requests: Optional[int] = typer.Option(
        None,
        "--warmup-requests",
        help="If set, warmup stops after this many requests instead of seconds",
    ),
    no_warmup: bool = typer.Option(
        False, "--no-warmup", help="Skip unscored warmup before each scored run"
    ),
    ssh_user: str = typer.Option("root", "--ssh-user", help="SSH user for vLLM host"),
    output: str = typer.Option(
        "agent_reports", "--output", "-o", help="Output directory for reports"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    oc_mode: bool = typer.Option(False, "--oc-mode", help="Use oc exec instead of SSH"),
    oc_namespace: Optional[str] = typer.Option(
        None, "--oc-namespace", help="OpenShift namespace (required with --oc-mode)"
    ),
    oc_pod: Optional[str] = typer.Option(
        None, "--oc-pod", help="Baseline pod name (required with --oc-mode)"
    ),
    kubeconfig: Optional[str] = typer.Option(
        None, "--kubeconfig", help="Path to kubeconfig (default: $KUBECONFIG)"
    ),
    pod_template: Optional[str] = typer.Option(
        None,
        "--pod-template",
        help="Pod YAML for experiment pods (examples/agent/experiment-pod.yaml)",
    ),
    vertex: bool = typer.Option(
        False, "--vertex", help="Use Google Cloud Vertex AI for Claude"
    ),
    vertex_project_id: Optional[str] = typer.Option(
        None, "--vertex-project-id", help="Vertex AI project ID"
    ),
    vertex_region: str = typer.Option(
        "us-east5", "--vertex-region", help="Vertex AI region"
    ),
    mlflow_uri: Optional[str] = typer.Option(
        None,
        "--mlflow-uri",
        help="MLflow tracking server URI. Enables automatic logging of all benchmark runs.",
    ),
    mlflow_experiment: str = typer.Option(
        "vllm-autotuning",
        "--mlflow-experiment",
        help="MLflow experiment name (default: vllm-autotuning)",
    ),
):
    """Run the Claude-driven pod-per-experiment vLLM tuner.

    The baseline pod/server is never restarted. Each experiment creates a
    fresh pod (when --pod-template is set), benchmarks it with GuideLLM,
    compares against baseline, then deletes it.
    """
    try:
        from ..agent.main import run_agent
    except ImportError as exc:
        console.print(
            "[red]Agent extras are not installed.[/red] "
            "Install with: [bold]pip install -e '.[agent]'[/bold]"
        )
        console.print(f"[dim]{exc}[/dim]")
        raise typer.Exit(1) from exc

    args = Namespace(
        vllm_endpoint=vllm_endpoint,
        vllm_host=vllm_host,
        model=model,
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        claude_model=claude_model,
        max_iterations=max_iterations,
        profiles=profiles or ["balanced"],
        concurrency=concurrency,
        max_seconds=max_seconds,
        warmup_concurrency=warmup_concurrency,
        warmup_seconds=warmup_seconds,
        warmup_requests=warmup_requests,
        no_warmup=no_warmup,
        ssh_user=ssh_user,
        output=output,
        verbose=verbose,
        oc_mode=oc_mode,
        oc_namespace=oc_namespace,
        oc_pod=oc_pod,
        kubeconfig=kubeconfig or os.environ.get("KUBECONFIG"),
        pod_template=pod_template,
        vertex=vertex or os.environ.get("CLAUDE_CODE_USE_VERTEX", "0") == "1",
        vertex_project_id=vertex_project_id
        or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"),
        vertex_region=vertex_region or os.environ.get("CLOUD_ML_REGION", "us-east5"),
        mlflow_uri=mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI"),
        mlflow_experiment=mlflow_experiment,
    )
    run_agent(args)

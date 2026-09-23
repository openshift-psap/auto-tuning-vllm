"""Typer command for the Claude-driven agentic vLLM tuner."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()


def agent_command(
    vllm_endpoint: Optional[str] = typer.Option(
        None, "--vllm-endpoint", help="URL of the running vLLM server"
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model served by vLLM"),
    vllm_host: Optional[str] = typer.Option(
        None, "--vllm-host", help="SSH hostname (required unless --oc-mode)"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Anthropic API key (default: $ANTHROPIC_API_KEY)"
    ),
    api_key_file: Optional[Path] = typer.Option(
        None,
        "--api-key-file",
        exists=True,
        dir_okay=False,
        help="File containing an Anthropic API key",
    ),
    claude_model: str = typer.Option(
        "sonnet", "--claude-model", help="Claude model: sonnet, opus, haiku, or full ID"
    ),
    max_iterations: int = typer.Option(
        100, "--max-iterations", help="Max agent loop iterations"
    ),
    max_tensor_parallel_size: Optional[int] = typer.Option(
        None,
        "--max-tensor-parallel-size",
        min=1,
        help="Optional runtime ceiling for tensor parallel size",
    ),
    tuning_profile: Optional[Path] = typer.Option(
        None,
        "--tuning-profile",
        exists=True,
        dir_okay=False,
        help="YAML profile; provisions its environment before tuning by default",
    ),
    provision: bool = typer.Option(
        True,
        "--provision/--no-provision",
        help="Provision the environment declared by --tuning-profile",
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
    vertex: Optional[bool] = typer.Option(
        None,
        "--vertex/--no-vertex",
        help="Use Google Cloud Vertex AI for Claude",
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
    port_forward = None
    benchmark_config = None
    benchmark_target = None
    if tuning_profile:
        from ..agent.tuning_profile import load_tuning_profile

        profile = load_tuning_profile(tuning_profile)
        if profiles is None and profile.benchmark_profiles:
            profiles = profile.benchmark_profiles
        if max_tensor_parallel_size is None:
            max_tensor_parallel_size = profile.max_tensor_parallel_size
        environment = profile.environment
        if environment is not None:
            model_config = environment.get("model")
            if isinstance(model_config, dict):
                profile_model = model_config.get("id")
                if isinstance(profile_model, str) and profile_model:
                    model = model or profile_model
        if provision:
            if not kubeconfig:
                raise typer.BadParameter(
                    "--kubeconfig is required to provision a tuning profile"
                )
            from ..agent.provision import (
                provision_environment,
                start_baseline_port_forward,
            )

            if environment is None:
                raise typer.BadParameter(
                    "The tuning profile needs an environment mapping to provision"
                )
            baseline = environment.get("baseline")
            benchmark_config = environment.get("benchmark")
            model_config = environment.get("model")
            if not isinstance(baseline, dict) or not isinstance(model_config, dict):
                raise typer.BadParameter(
                    "Profile environment needs baseline and model mappings"
                )
            baseline_name = baseline.get("name")
            profile_model = model_config.get("id")
            if not isinstance(baseline_name, str) or not isinstance(profile_model, str):
                raise typer.BadParameter(
                    "Profile baseline.name and model.id must be non-empty strings"
                )
            if not isinstance(benchmark_config, dict):
                raise typer.BadParameter(
                    "Profile environment needs a benchmark mapping"
                )
            benchmark_target = f"http://{baseline_name}:8000"

            template_path = Path(".auto_tune") / f"{profile.name}-experiment-pod.yaml"
            console.print(f"[cyan]Provisioning tuning profile: {profile.name}[/cyan]")
            provisioned = provision_environment(profile, kubeconfig, template_path)
            oc_mode = True
            oc_namespace = provisioned.namespace
            oc_pod = f"deployment/{provisioned.baseline_deployment}"
            pod_template = str(provisioned.experiment_template)
            if vllm_endpoint is None:
                port_forward = start_baseline_port_forward(
                    kubeconfig=kubeconfig,
                    namespace=provisioned.namespace,
                    service=provisioned.baseline_deployment,
                )
                vllm_endpoint = port_forward.endpoint
                console.print(
                    f"[cyan]Baseline available locally at {vllm_endpoint}[/cyan]"
                )

    if vllm_endpoint is None:
        raise typer.BadParameter(
            "--vllm-endpoint is required unless a provisioned tuning profile "
            "supplies it"
        )
    if model is None:
        raise typer.BadParameter(
            "--model is required unless a provisioned tuning profile supplies it"
        )

    try:
        from ..agent.main import run_agent
    except ImportError as exc:
        if port_forward is not None:
            port_forward.close()
        console.print(
            "[red]Agent extras are not installed.[/red] "
            "Install with: [bold]pip install -e '.[agent]'[/bold]"
        )
        console.print(f"[dim]{exc}[/dim]")
        raise typer.Exit(1) from exc

    resolved_api_key = api_key
    if api_key_file is not None:
        resolved_api_key = api_key_file.read_text(encoding="utf-8").strip()
        if not resolved_api_key:
            raise typer.BadParameter("--api-key-file is empty")

    args = Namespace(
        vllm_endpoint=vllm_endpoint,
        vllm_host=vllm_host,
        model=model,
        api_key=resolved_api_key or os.environ.get("ANTHROPIC_API_KEY"),
        claude_model=claude_model,
        max_iterations=max_iterations,
        max_tensor_parallel_size=max_tensor_parallel_size,
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
        benchmark_config=benchmark_config,
        benchmark_target=benchmark_target,
        vertex=(
            vertex
            if vertex is not None
            else os.environ.get("CLAUDE_CODE_USE_VERTEX", "0") == "1"
        ),
        vertex_project_id=vertex_project_id
        or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"),
        vertex_region=vertex_region or os.environ.get("CLOUD_ML_REGION", "us-east5"),
        mlflow_uri=mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI"),
        mlflow_experiment=mlflow_experiment,
    )
    try:
        run_agent(args)
    finally:
        if port_forward is not None:
            port_forward.close()

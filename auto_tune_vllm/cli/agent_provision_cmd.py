"""Typer command for profile-defined agentic environment provisioning."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ..agent.provision import provision_environment
from ..agent.tuning_profile import load_tuning_profile

console = Console()


def provision_agent_environment_command(
    tuning_profile: Path = typer.Option(
        ...,
        "--tuning-profile",
        exists=True,
        dir_okay=False,
        help="Profile that defines the environment and tuning policy",
    ),
    kubeconfig: Path = typer.Option(
        ..., "--kubeconfig", help="Target cluster kubeconfig"
    ),
    experiment_template: Path = typer.Option(
        Path(".auto_tune/experiment-pod.yaml"),
        "--experiment-template",
        help="Generated experiment pod template path",
    ),
    timeout: str = typer.Option(
        "20m", "--timeout", help="Download and rollout timeout"
    ),
) -> None:
    """Provision the profile's baseline and model cache on OpenShift."""
    profile = load_tuning_profile(tuning_profile)
    try:
        result = provision_environment(
            profile,
            str(kubeconfig),
            experiment_template,
            timeout,
        )
    except Exception as exc:
        console.print(f"[red]Provisioning failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Provisioned namespace: {result.namespace}[/green]")
    console.print(f"Baseline deployment: {result.baseline_deployment}")
    console.print(f"Experiment template: {result.experiment_template}")

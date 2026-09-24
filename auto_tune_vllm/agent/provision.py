"""Provision a profile-defined agentic-tuning environment on OpenShift."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .environment import render_environment_resources
from .tuning_profile import TuningProfile


@dataclass(frozen=True)
class ProvisionedEnvironment:
    """Stable resource identifiers produced by profile provisioning."""

    namespace: str
    baseline_deployment: str
    experiment_template: Path


def cleanup_baseline(
    *, kubeconfig: str, namespace: str, deployment: str, service: str
) -> None:
    """Remove a study's baseline once its benchmark result is recorded.

    The cache PVC is deliberately not touched. A deletion error is propagated so
    the controller stops rather than starting trials while reserved GPUs remain.
    """
    _oc(
        kubeconfig,
        ["delete", "deployment", deployment, "-n", namespace, "--wait=true"],
    )
    _oc(
        kubeconfig,
        ["delete", "service", service, "-n", namespace, "--wait=true"],
    )


@dataclass
class BaselinePortForward:
    """A local port-forward to the profile's baseline service."""

    endpoint: str
    process: subprocess.Popen[str]

    def close(self) -> None:
        """Stop the port-forward process if it is still running."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def provision_environment(
    profile: TuningProfile,
    kubeconfig: str,
    experiment_template: str | Path,
    timeout: str = "20m",
) -> ProvisionedEnvironment:
    """Provision cache and a freshly restarted baseline for every study.

    The profile's model-access secret is copied by reference from its declared
    source namespace without writing its data to disk or command output.
    """
    environment = profile.environment
    if environment is None:
        raise ValueError("A provisionable profile requires an environment mapping")
    namespace = _required_string(environment, "namespace")
    baseline = _required_mapping(environment, "baseline")
    access = _required_mapping(environment, "model_access")
    resources = render_environment_resources(profile)
    pvc, results_pvc, downloader, deployment, service, experiment = resources

    namespace_result = _oc(
        kubeconfig,
        ["get", "namespace", namespace],
        capture_output=True,
        check=False,
    )
    if namespace_result.returncode != 0:
        _oc(kubeconfig, ["create", "namespace", namespace])
    _copy_secret(
        kubeconfig=kubeconfig,
        source_namespace=_required_string(access, "source_namespace"),
        target_namespace=namespace,
        secret_name=_required_string(access, "secret_name"),
    )
    _apply_resource(kubeconfig, namespace, pvc)
    _apply_resource(kubeconfig, namespace, results_pvc)

    downloader_name = downloader["metadata"]["name"]
    if not _job_complete(kubeconfig, namespace, downloader_name):
        _oc(
            kubeconfig,
            ["delete", "job", downloader_name, "-n", namespace, "--ignore-not-found"],
        )
        _apply_resource(kubeconfig, namespace, downloader)
        _oc(
            kubeconfig,
            [
                "wait",
                "-n",
                namespace,
                "--for=condition=complete",
                f"job/{downloader_name}",
                f"--timeout={timeout}",
            ],
        )

    deployment_name = _required_string(baseline, "name")
    deployment_exists = _resource_exists(
        kubeconfig, namespace, "deployment", deployment_name
    )
    _apply_resource(kubeconfig, namespace, deployment)
    _apply_resource(kubeconfig, namespace, service)
    if deployment_exists:
        # Keep the cached model, but give every study a fresh server process.
        _oc(
            kubeconfig,
            [
                "rollout",
                "restart",
                "-n",
                namespace,
                f"deployment/{deployment_name}",
            ],
        )
    _oc(
        kubeconfig,
        [
            "rollout",
            "status",
            "-n",
            namespace,
            f"deployment/{deployment_name}",
            f"--timeout={timeout}",
        ],
    )

    template_path = Path(experiment_template)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8"
    )
    return ProvisionedEnvironment(namespace, deployment_name, template_path)


def start_baseline_port_forward(
    *, kubeconfig: str, namespace: str, service: str
) -> BaselinePortForward:
    """Expose a profile baseline service locally for the agent's HTTP client."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        local_port = listener.getsockname()[1]

    process = subprocess.Popen(
        [
            "oc",
            "--kubeconfig",
            kubeconfig,
            "-n",
            namespace,
            "port-forward",
            f"service/{service}",
            f"{local_port}:8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    for _ in range(10):
        time.sleep(0.1)
        if process.poll() is None:
            return BaselinePortForward(f"http://127.0.0.1:{local_port}", process)
    stderr = process.stderr.read() if process.stderr is not None else ""
    raise RuntimeError(f"Could not start baseline port-forward: {stderr.strip()}")


def _copy_secret(
    *, kubeconfig: str, source_namespace: str, target_namespace: str, secret_name: str
) -> None:
    source = _oc(
        kubeconfig,
        ["get", "secret", secret_name, "-n", source_namespace, "-o", "json"],
        capture_output=True,
    )
    secret = json.loads(source.stdout)
    metadata = secret.setdefault("metadata", {})
    for field in (
        "namespace",
        "uid",
        "resourceVersion",
        "creationTimestamp",
        "managedFields",
        "ownerReferences",
    ):
        metadata.pop(field, None)
    _oc(
        kubeconfig,
        ["apply", "-n", target_namespace, "-f", "-"],
        input=json.dumps(secret),
    )


def _apply_resource(kubeconfig: str, namespace: str, resource: dict) -> None:
    _oc(kubeconfig, ["apply", "-n", namespace, "-f", "-"], input=json.dumps(resource))


def _job_complete(kubeconfig: str, namespace: str, job_name: str) -> bool:
    result = _oc(
        kubeconfig,
        ["get", "job", job_name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    status = json.loads(result.stdout).get("status", {})
    return status.get("succeeded", 0) >= 1


def _resource_exists(
    kubeconfig: str, namespace: str, kind: str, name: str
) -> bool:
    result = _oc(
        kubeconfig,
        ["get", kind, name, "-n", namespace],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _oc(
    kubeconfig: str,
    args: list[str],
    *,
    input: str | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["oc", "--kubeconfig", kubeconfig, *args],
        input=input,
        capture_output=capture_output,
        text=True,
        check=check,
    )


def _required_mapping(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"environment.{key} must be a mapping")
    return value


def _required_string(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value

"""
Pod Manager for Isolated Pod-per-Experiment Architecture.

Creates ephemeral vLLM pods from a YAML template for each tuning experiment.
The baseline pod is never modified; experiment pods are created with different
vLLM launch args, benchmarked, compared against baseline, then deleted.

Usage:
    pm = PodManager(namespace="llm-d", kubeconfig=None, base_pod_yaml_path="examples/agent/experiment-pod.yaml")
    pod_name, endpoint = pm.create_pod(["--enable-chunked-prefill", "--gpu-memory-utilization", "0.95"])
    # ... run benchmarks against endpoint ...
    pm.delete_pod(pod_name)
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import yaml


class ExperimentLifecycleError(RuntimeError):
    """A terminal experiment startup failure with cluster diagnostics."""


class PodManager:
    """Manages ephemeral vLLM experiment pods on OpenShift.

    Creates pods from a YAML template with modified vLLM args, waits for
    readiness, sets up port-forwarding, and cleans up on deletion.

    Parameters
    ----------
    namespace : str
        Kubernetes/OpenShift namespace.
    kubeconfig : str or None
        Path to kubeconfig. If None, uses default (~/.kube/config or $KUBECONFIG).
    base_pod_yaml_path : str
        Path to the pod YAML template (e.g. examples/agent/experiment-pod.yaml).
    base_port : int
        First local port to assign for port-forwarding experiment pods.
    """

    def __init__(
        self,
        namespace: str,
        kubeconfig: Optional[str] = None,
        base_pod_yaml_path: str = "examples/agent/experiment-pod.yaml",
        base_port: int = 8001,
        artifact_dir: str | Path | None = None,
    ):
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.base_yaml_path = base_pod_yaml_path
        self.active_pods: dict[str, dict] = {}
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None

        # Load and validate the template once
        with open(self.base_yaml_path, "r") as f:
            self._template = yaml.safe_load(f)

        if self._template.get("kind") != "Pod":
            raise ValueError(
                f"Template {base_pod_yaml_path} is not a Pod manifest (kind={self._template.get('kind')})"
            )

    def _build_oc_base(self) -> list[str]:
        """Build the base ``oc`` command with kubeconfig and namespace."""
        cmd = ["oc"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += ["-n", self.namespace]
        return cmd

    def _generate_pod_name(self) -> str:
        """Generate a unique pod name based on timestamp."""
        ts = int(time.time())
        return f"vllm-tune-{ts}"

    def _build_pod_manifest(self, pod_name: str, vllm_args: list[str]) -> dict:
        """Create a pod manifest from the template with extra vLLM args.

        Appends ``vllm_args`` to the existing ``args`` list of the first
        container (assumed to be the vLLM container).
        """
        prefix_cache_args = [
            arg
            for arg in vllm_args
            if arg.split("=", 1)[0]
            in {"--enable-prefix-caching", "--no-enable-prefix-caching"}
        ]
        if prefix_cache_args:
            raise ValueError(
                "Prefix caching is a fixed study control and cannot be changed "
                f"by an experiment: {prefix_cache_args}"
            )
        quantization_args = [
            arg for arg in vllm_args if arg.split("=", 1)[0] == "--quantization"
        ]
        if quantization_args:
            raise ValueError(
                "Do not use vLLM --quantization in this study. Select an externally "
                "quantized model checkpoint instead."
            )
        manifest = copy.deepcopy(self._template)

        # Set unique pod name
        manifest["metadata"]["name"] = pod_name

        # Add a label to identify experiment pods for easy cleanup
        labels = manifest["metadata"].setdefault("labels", {})
        labels["vllm-experiment"] = "true"
        # A unique selector lets an in-cluster GuideLLM Job reach exactly this
        # experiment rather than another concurrent tuning pod.
        labels["vllm-experiment-id"] = pod_name

        # Append tuning args to the container's args list
        container = manifest["spec"]["containers"][0]
        existing_args = list(container.get("args", []))
        existing_args.extend(vllm_args)
        container["args"] = existing_args

        return manifest

    def _build_service_manifest(self, pod_name: str) -> dict:
        """Build the private Service used by in-cluster benchmark Jobs."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": pod_name,
                "labels": {"app": "auto-tune-vllm", "vllm-experiment": "true"},
            },
            "spec": {
                "selector": {"vllm-experiment-id": pod_name},
                "ports": [{"name": "http", "port": 8000, "targetPort": "http"}],
            },
        }

    def _apply_manifest(self, manifest: dict, prefix: str) -> None:
        """Apply a manifest through ``oc`` without retaining it on disk."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{prefix}_", delete=False
        ) as tmp:
            yaml.dump(manifest, tmp, default_flow_style=False)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                self._build_oc_base() + ["apply", "-f", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"oc apply failed: {result.stderr}")
        finally:
            os.unlink(tmp_path)

    def _wait_for_ready(
        self, pod_name: str, timeout: int = 900, poll_interval: int = 5
    ) -> bool:
        """Poll pod readiness until ready or timeout.

        Returns True if the pod reached Running + Ready state.
        """
        oc_base = self._build_oc_base()
        deadline = time.time() + timeout

        while time.time() < deadline:
            # Check pod phase
            cmd = oc_base + [
                "get",
                "pod",
                pod_name,
                "-o",
                "jsonpath={.status.phase}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            phase = result.stdout.strip()

            if phase == "Failed" or phase == "Unknown":
                raise ExperimentLifecycleError(
                    self._failure_diagnostics(pod_name, f"phase={phase}")
                )

            if phase == "Running":
                # Check readiness condition
                ready_cmd = oc_base + [
                    "get",
                    "pod",
                    pod_name,
                    "-o",
                    "jsonpath={.status.conditions[?(@.type=='Ready')].status}",
                ]
                ready_result = subprocess.run(
                    ready_cmd, capture_output=True, text=True, timeout=15
                )
                if ready_result.stdout.strip() == "True":
                    return True

            time.sleep(poll_interval)

        raise ExperimentLifecycleError(
            self._failure_diagnostics(
                pod_name, f"not Ready after {timeout}s (last phase={phase})"
            )
        )

    def _failure_diagnostics(self, pod_name: str, reason: str) -> str:
        """Collect bounded startup diagnostics before cleanup removes the pod."""
        oc_base = self._build_oc_base()
        commands = {
            "pod": oc_base + ["get", "pod", pod_name, "-o", "json"],
            "events": oc_base
            + [
                "get",
                "events",
                "--field-selector",
                f"involvedObject.name={pod_name}",
                "--sort-by=.lastTimestamp",
            ],
            "logs": oc_base + ["logs", pod_name, "--tail=80"],
        }
        diagnostics: dict[str, str] = {"reason": reason}
        for name, command in commands.items():
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=20
                )
                text = (result.stdout or result.stderr).strip()
                if text:
                    diagnostics[name] = text[-4000:]
            except subprocess.TimeoutExpired:
                diagnostics[name] = "timed out while collecting diagnostics"
        diagnostics["cleanup"] = "Resources were cleaned up."
        return json.dumps({"experiment_lifecycle": "failed", **diagnostics})

    def archive_pod_logs(self, pod_name: str) -> Path | None:
        """Archive experiment diagnostics before deleting its Pod.

        The current and previous container logs are preserved in full, rather
        than the bounded tail supplied to the agent. Pod state and events are
        saved alongside them so a failed experiment remains reproducible after
        its cluster resources are gone.
        """
        if self.artifact_dir is None:
            return None

        pod_dir = self.artifact_dir / "pods" / pod_name
        pod_dir.mkdir(parents=True, exist_ok=True)
        oc_base = self._build_oc_base()
        captures = {
            "current.log": oc_base
            + ["logs", pod_name, "--all-containers=true", "--prefix=true"],
            "previous.log": oc_base
            + [
                "logs",
                pod_name,
                "--all-containers=true",
                "--prefix=true",
                "--previous=true",
            ],
            "pod.json": oc_base + ["get", "pod", pod_name, "-o", "json"],
            "events.txt": oc_base
            + [
                "get",
                "events",
                "--field-selector",
                f"involvedObject.name={pod_name}",
                "--sort-by=.lastTimestamp",
            ],
        }
        current_log_failure = None
        for filename, command in captures.items():
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=60
                )
            except subprocess.TimeoutExpired as exc:
                if filename == "current.log":
                    current_log_failure = f"timed out: {exc}"
                content = f"archive command timed out: {exc}\n"
            else:
                content = result.stdout
                if result.stderr:
                    content += f"\n--- stderr ---\n{result.stderr}"
                if filename == "current.log" and result.returncode:
                    current_log_failure = result.stderr.strip() or "oc logs failed"
            (pod_dir / filename).write_text(content, encoding="utf-8")

        if current_log_failure:
            raise ExperimentLifecycleError(
                f"Could not archive logs for pod {pod_name}: {current_log_failure}. "
                "The pod was retained for user intervention."
            )
        print(f"   Archived pod diagnostics: {pod_dir}", flush=True)
        return pod_dir

    def _start_port_forward(
        self, pod_name: str, local_port: int, remote_port: int = 8000
    ) -> subprocess.Popen:
        """Start ``oc port-forward`` as a background subprocess."""
        cmd = self._build_oc_base() + [
            "port-forward",
            pod_name,
            f"{local_port}:{remote_port}",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Give port-forward a moment to bind
        time.sleep(2)

        # Check it didn't immediately die
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(
                f"oc port-forward for {pod_name} exited immediately (rc={proc.returncode}): {stderr}"
            )

        return proc

    def create_pod(self, vllm_args: list[str]) -> tuple[str, str]:
        """Create an experiment pod with extra vLLM args, wait for readiness,
        and start port-forwarding.

        Parameters
        ----------
        vllm_args : list[str]
            Extra CLI args for vLLM (e.g. ["--enable-chunked-prefill",
            "--gpu-memory-utilization", "0.95"]).

        Returns
        -------
        tuple[str, str]
            (pod_name, endpoint_url) e.g. ("vllm-tune-1714000000", "http://localhost:8001")
        """
        pod_name = self._generate_pod_name()
        print(
            f">> PodManager: Creating pod {pod_name} with args {vllm_args}", flush=True
        )

        # Build manifest
        manifest = self._build_pod_manifest(pod_name, vllm_args)

        self._apply_manifest(manifest, pod_name)
        print(f"   Pod {pod_name} created.", flush=True)

        self._apply_manifest(self._build_service_manifest(pod_name), f"{pod_name}_svc")
        print(f"   Service {pod_name} created.", flush=True)

        try:
            # Wait for pod to be ready
            print(f"   Waiting for pod {pod_name} to be ready...", flush=True)
            self._wait_for_ready(pod_name)
            print(f"   Pod {pod_name} is ready.", flush=True)

        except Exception as exc:
            # A failed readiness check happens before this pod is recorded in
            # active_pods. Preserve its diagnostics before cleaning resources.
            self.archive_pod_logs(pod_name)
            self._delete_untracked_experiment(pod_name)
            raise

        endpoint = f"http://{pod_name}:8000"

        self.active_pods[pod_name] = {
            "vllm_args": vllm_args,
            "endpoint": endpoint,
            "cluster_endpoint": endpoint,
        }

        return pod_name, endpoint

    def _delete_untracked_experiment(self, pod_name: str) -> None:
        """Remove resources created before an experiment became active."""
        for resource in ("pod", "service"):
            try:
                subprocess.run(
                    self._build_oc_base()
                    + ["delete", resource, pod_name, "--ignore-not-found"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                # Preserve the original startup diagnostic; deletion is retried
                # by the next profile provisioning/explicit cleanup operation.
                pass

    def get_cluster_target(self, endpoint: str) -> Optional[str]:
        """Return the private Service URL corresponding to a local endpoint."""
        for info in self.active_pods.values():
            if info["endpoint"] == endpoint:
                return info["cluster_endpoint"]
        return None

    def delete_pod(self, pod_name: str) -> None:
        """Delete an experiment pod and kill its port-forward process.

        Parameters
        ----------
        pod_name : str
            Name of the pod to delete.
        """
        info = self.active_pods.get(pod_name)

        # Never delete evidence before preserving it for post-experiment
        # analysis. A failed archive intentionally leaves the Pod intact.
        self.archive_pod_logs(pod_name)

        # Delete gracefully. Do not force-delete an experiment: that can lose
        # shutdown diagnostics and makes a lifecycle failure harder to inspect.
        delete_cmd = self._build_oc_base() + [
            "delete",
            "pod",
            pod_name,
            "--wait=true",
        ]
        delete_error: str | None = None
        try:
            result = subprocess.run(
                delete_cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                print(f"   Pod {pod_name} deleted.", flush=True)
            else:
                delete_error = result.stderr.strip() or "oc delete returned non-zero"
        except subprocess.TimeoutExpired:
            delete_error = "oc delete timed out after 30s"

        # The Service belongs exclusively to this experiment selector.
        try:
            service_result = subprocess.run(
                self._build_oc_base()
                + ["delete", "service", pod_name, "--ignore-not-found"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if service_result.returncode != 0:
                service_error = (
                    "failed to delete Service: "
                    f"{service_result.stderr.strip() or 'oc delete returned non-zero'}"
                )
                delete_error = (
                    f"{delete_error}; {service_error}" if delete_error else service_error
                )
        except subprocess.TimeoutExpired:
            service_error = "Service deletion timed out after 30s"
            delete_error = (
                f"{delete_error}; {service_error}" if delete_error else service_error
            )
        if delete_error:
            raise ExperimentLifecycleError(
                f"Experiment cleanup failed for pod {pod_name}: {delete_error}. "
                "User intervention is required; tuning has paused."
            )
        self.active_pods.pop(pod_name, None)

    def cleanup_all(self) -> None:
        """Delete all active experiment pods or raise a terminal cleanup error."""
        pod_names = list(self.active_pods.keys())
        if not pod_names:
            return

        print(
            f">> PodManager: Cleaning up {len(pod_names)} experiment pod(s)...",
            flush=True,
        )
        failures: list[str] = []
        for pod_name in pod_names:
            try:
                self.delete_pod(pod_name)
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise ExperimentLifecycleError("; ".join(failures))

    def confirm_cleanup_resolved(self, pod_name: str) -> None:
        """Acknowledge user-remediated cleanup after verifying resources are gone.

        This is intentionally an explicit operation: it never deletes or retries
        resources. It only permits a paused controller to resume after the user
        has fixed the reported cluster-side failure.
        """
        for resource in ("pod", "service"):
            result = subprocess.run(
                self._build_oc_base() + ["get", resource, pod_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                raise ExperimentLifecycleError(
                    f"Cannot resume: {resource} {pod_name} still exists. "
                    "Remove it, then confirm cleanup again."
                )
            stderr = result.stderr.lower()
            if "notfound" not in stderr and "not found" not in stderr:
                raise ExperimentLifecycleError(
                    f"Cannot verify cleanup for {resource} {pod_name}: "
                    f"{result.stderr.strip() or 'oc get failed'}"
                )
        self.active_pods.pop(pod_name, None)

    def get_active_pods(self) -> dict[str, dict]:
        """Return info about active experiment pods."""
        return dict(self.active_pods)

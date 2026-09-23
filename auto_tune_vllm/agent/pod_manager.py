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
import os
import signal
import subprocess
import tempfile
import time
from typing import Optional

import yaml


def _merge_cli_args(existing: list[str], extra: list[str]) -> list[str]:
    """Append extra CLI args, replacing any flag already present in existing."""
    merged = list(existing)
    i = 0
    while i < len(extra):
        flag = extra[i]
        j = 0
        while j < len(merged):
            if merged[j] == flag:
                drop = (
                    2
                    if j + 1 < len(merged)
                    and not str(merged[j + 1]).startswith("-")
                    else 1
                )
                del merged[j : j + drop]
            else:
                j += 1
        merged.append(flag)
        if i + 1 < len(extra) and not str(extra[i + 1]).startswith("-"):
            merged.append(extra[i + 1])
            i += 2
        else:
            i += 1
    return merged


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
    ):
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.base_yaml_path = base_pod_yaml_path
        self.base_port = base_port
        self.active_pods: dict[
            str, dict
        ] = {}  # pod_name -> {"port_forward_proc": ..., "local_port": ...}
        self._next_port = base_port

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

        Reloads the YAML each time so template edits (CPU, image, model)
        apply to later experiments in the same session.
        """
        with open(self.base_yaml_path, "r") as f:
            self._template = yaml.safe_load(f)
        manifest = copy.deepcopy(self._template)

        # Set unique pod name
        manifest["metadata"]["name"] = pod_name

        # Add a label to identify experiment pods for easy cleanup
        labels = manifest["metadata"].setdefault("labels", {})
        labels["vllm-experiment"] = "true"

        # Merge tuning args, replacing flags already in the template
        # (e.g. --gpu-memory-utilization 0.85 must not leave a duplicate 0.90).
        container = manifest["spec"]["containers"][0]
        container["args"] = _merge_cli_args(
            list(container.get("args", [])), list(vllm_args or [])
        )

        return manifest

    def _wait_for_ready(
        self, pod_name: str, timeout: int = 120, poll_interval: int = 5
    ) -> bool:
        """Poll pod readiness until ready or timeout.

        Returns True if the pod reached Running + Ready state.
        """
        oc_base = self._build_oc_base()
        deadline = time.time() + timeout

        last_print = 0.0
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
            elapsed = int(time.time() - (deadline - timeout))
            if time.time() - last_print >= 15:
                print(
                    f"   still waiting for {pod_name}: phase={phase or '?'} "
                    f"({elapsed}s / {timeout}s)",
                    flush=True,
                )
                last_print = time.time()

            if phase == "Failed" or phase == "Unknown":
                # Pod failed to start — get events for debugging
                events_cmd = oc_base + [
                    "get",
                    "events",
                    "--field-selector",
                    f"involvedObject.name={pod_name}",
                    "--sort-by=.lastTimestamp",
                ]
                events_result = subprocess.run(
                    events_cmd, capture_output=True, text=True, timeout=15
                )
                raise RuntimeError(
                    f"Pod {pod_name} entered phase '{phase}'. Events:\n{events_result.stdout}"
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

        raise TimeoutError(
            f"Pod {pod_name} not ready after {timeout}s (last phase: {phase})"
        )

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
        local_port = self._next_port
        self._next_port += 1

        print(
            f">> PodManager: Creating pod {pod_name} with args {vllm_args}", flush=True
        )

        # Build manifest
        manifest = self._build_pod_manifest(pod_name, vllm_args)

        # Write to temp file and apply
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{pod_name}_", delete=False
        ) as tmp:
            yaml.dump(manifest, tmp, default_flow_style=False)
            tmp_path = tmp.name

        try:
            apply_cmd = self._build_oc_base() + ["apply", "-f", tmp_path]
            result = subprocess.run(
                apply_cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"oc apply failed: {result.stderr}")
            print(f"   Pod {pod_name} created.", flush=True)
        finally:
            os.unlink(tmp_path)

        # Wait for pod to be ready
        print(f"   Waiting for pod {pod_name} to be ready...", flush=True)
        # Model load on H200 can take several minutes (Gemma 26B ~5–15 min).
        self._wait_for_ready(pod_name, timeout=900)
        print(f"   Pod {pod_name} is ready.", flush=True)

        # Start port-forward
        print(f"   Starting port-forward :{local_port} -> {pod_name}:8000", flush=True)
        pf_proc = self._start_port_forward(pod_name, local_port)
        print(f"   Port-forward active (pid={pf_proc.pid}).", flush=True)

        endpoint = f"http://localhost:{local_port}"

        self.active_pods[pod_name] = {
            "port_forward_proc": pf_proc,
            "local_port": local_port,
            "vllm_args": vllm_args,
            "endpoint": endpoint,
        }

        return pod_name, endpoint

    def delete_pod(self, pod_name: str) -> None:
        """Delete an experiment pod and kill its port-forward process.

        Parameters
        ----------
        pod_name : str
            Name of the pod to delete.
        """
        info = self.active_pods.pop(pod_name, None)

        # Kill port-forward
        if info and info.get("port_forward_proc"):
            proc = info["port_forward_proc"]
            try:
                os.kill(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            print(f"   Port-forward for {pod_name} killed.", flush=True)

        # Delete pod
        delete_cmd = self._build_oc_base() + [
            "delete",
            "pod",
            pod_name,
            "--grace-period=0",
            "--force",
        ]
        result = subprocess.run(delete_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"   Pod {pod_name} deleted.", flush=True)
        else:
            print(
                f"   Warning: Failed to delete pod {pod_name}: {result.stderr}",
                flush=True,
            )

    def cleanup_all(self) -> None:
        """Delete all active experiment pods. Called at agent exit."""
        pod_names = list(self.active_pods.keys())
        if not pod_names:
            return

        print(
            f">> PodManager: Cleaning up {len(pod_names)} experiment pod(s)...",
            flush=True,
        )
        for pod_name in pod_names:
            try:
                self.delete_pod(pod_name)
            except Exception as e:
                print(f"   Warning: cleanup failed for {pod_name}: {e}", flush=True)

    def get_active_pods(self) -> dict[str, dict]:
        """Return info about active experiment pods."""
        return dict(self.active_pods)

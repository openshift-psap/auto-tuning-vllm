"""Run GuideLLM benchmarks as OpenShift Jobs, never on the controller."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClusterBenchmarkRunner:
    """Create a short-lived in-cluster GuideLLM Job and return its logs."""

    namespace: str
    kubeconfig: str
    config: dict[str, Any]

    def run(
        self,
        *,
        profile: dict[str, Any],
        target: str,
        model: str,
        concurrency: str,
        max_seconds: int,
    ) -> str:
        job_name = f"guidellm-{int(time.time())}"
        streams = [int(value) for value in concurrency.split(",")]
        resources = self.config["resources"]
        result_pvc = self.config["results_pvc"]["name"]
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "labels": {
                    "app.kubernetes.io/name": "guidellm",
                    "app.kubernetes.io/component": "benchmark",
                },
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 86400,
                "template": {
                    "spec": {
                        "automountServiceAccountToken": False,
                        "restartPolicy": "Never",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "containers": [
                            {
                                "name": "guidellm",
                                "image": self.config["image"],
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "capabilities": {"drop": ["ALL"]},
                                    "readOnlyRootFilesystem": True,
                                },
                                "command": [
                                    "guidellm",
                                    "run",
                                    "--backend",
                                    f"kind=openai_http,target={target},model={model}",
                                    "--tokenizer",
                                    f"kind=huggingface_auto,model={model}",
                                    "--data",
                                    "kind=synthetic_text,"
                                    f"prompt_tokens={profile['isl']},"
                                    f"output_tokens={profile['osl']}",
                                    "--profile",
                                    json.dumps({"kind": "concurrent", "streams": streams}),
                                    "--constraint",
                                    f"kind=max_duration,seconds={max_seconds}",
                                    "--output",
                                    f"kind=json,path=/results/{job_name}.json",
                                    "--output",
                                    f"kind=csv,path=/results/{job_name}.csv",
                                ],
                                "env": [
                                    {"name": "HF_HOME", "value": "/tmp/huggingface"},
                                    {"name": "XDG_CACHE_HOME", "value": "/tmp"},
                                ],
                                "resources": resources,
                                "volumeMounts": [
                                    {"name": "tmp", "mountPath": "/tmp"},
                                    {"name": "results", "mountPath": "/results"},
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "tmp", "emptyDir": {}},
                            {
                                "name": "results",
                                "persistentVolumeClaim": {"claimName": result_pvc},
                            },
                        ],
                    }
                },
            },
        }
        self._oc(["apply", "-n", self.namespace, "-f", "-"], input=json.dumps(manifest))
        timeout = max(max_seconds * len(streams) + 300, 900)
        result = self._oc(
            [
                "wait", "-n", self.namespace, "--for=condition=complete",
                f"job/{job_name}", f"--timeout={timeout}s",
            ],
            check=False,
        )
        logs = self._oc(["logs", "-n", self.namespace, f"job/{job_name}"], check=False)
        if result.returncode:
            raise RuntimeError(f"GuideLLM Job {job_name} failed:\n{logs.stdout}\n{logs.stderr}")
        return self._summarize(job_name, target, logs.stdout)

    @staticmethod
    def _summarize(job_name: str, target: str, logs: str) -> str:
        """Return the completed Job's comparison-ready console metrics.

        The result PVC is retained for operators, but it is intentionally not
        presented as a controller-local filename.  The agent receives the
        canonical GuideLLM summary emitted by the Job itself.
        """
        def row_after(heading: str) -> str:
            position = logs.find(heading)
            if position < 0:
                return "not found"
            for line in logs[position:].splitlines()[1:]:
                if "| concurrent |" in line:
                    return line.strip()
            return "not found"

        return "\n".join(
            [
                f"GuideLLM Job: {job_name}",
                f"Target Service: {target}",
                "=== BENCHMARK METRICS (compare these rows directly) ===",
                "Completed requests (input/output token totals):",
                row_after("Run Summary Info"),
                "Latency (request seconds, TTFT/TTFOT/ITL/TPOT milliseconds; median/p95):",
                row_after("Request Latency Statistics"),
                "Throughput (request, input-token, output-token, total-token per second):",
                row_after("Server Throughput Statistics"),
                "=== END BENCHMARK METRICS ===",
            ]
        )

    def _oc(self, args: list[str], *, input: str | None = None, check: bool = True):
        return subprocess.run(
            ["oc", "--kubeconfig", self.kubeconfig, *args],
            input=input, text=True, capture_output=True, check=check,
        )

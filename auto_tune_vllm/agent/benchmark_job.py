"""Run GuideLLM benchmarks as OpenShift Jobs, never on the controller."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClusterBenchmarkRunner:
    """Create a short-lived in-cluster GuideLLM Job and return its logs."""

    namespace: str
    kubeconfig: str
    config: dict[str, Any]
    mlflow_uri: str | None = None
    mlflow_experiment: str = "vllm-autotuning"
    mlflow_workspace: str | None = None
    artifact_dir: str | Path | None = None

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
        self._archive_job_diagnostics(job_name, logs.stdout, logs.stderr)
        if result.returncode:
            raise RuntimeError(f"GuideLLM Job {job_name} failed:\n{logs.stdout}\n{logs.stderr}")
        summary = self._summarize(job_name, target, logs.stdout)
        self._log_to_mlflow(job_name, model, profile, target, concurrency, logs.stdout)
        return summary

    def _archive_job_diagnostics(
        self, job_name: str, logs: str, log_stderr: str = ""
    ) -> Path | None:
        """Archive every GuideLLM Job-pod's logs and cluster diagnostics."""
        if self.artifact_dir is None:
            return None

        job_dir = Path(self.artifact_dir) / "benchmarks" / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        content = logs
        if log_stderr:
            content += f"\n--- stderr ---\n{log_stderr}"
        (job_dir / "guidellm.log").write_text(content, encoding="utf-8")

        captures = {
            "job.json": ["get", "job", job_name, "-n", self.namespace, "-o", "json"],
            "pods.json": [
                "get",
                "pods",
                "-n",
                self.namespace,
                "-l",
                f"job-name={job_name}",
                "-o",
                "json",
            ],
            "events.txt": [
                "get",
                "events",
                "-n",
                self.namespace,
                "--field-selector",
                f"involvedObject.name={job_name}",
                "--sort-by=.lastTimestamp",
            ],
        }
        for filename, command in captures.items():
            result = self._oc(command, check=False)
            captured = result.stdout
            if result.stderr:
                captured += f"\n--- stderr ---\n{result.stderr}"
            (job_dir / filename).write_text(captured, encoding="utf-8")

            if filename != "pods.json":
                continue
            try:
                pod_names = [
                    item["metadata"]["name"]
                    for item in json.loads(result.stdout).get("items", [])
                ]
            except (json.JSONDecodeError, KeyError, TypeError):
                pod_names = []
            for pod_name in pod_names:
                current = self._oc(
                    [
                        "logs",
                        "-n",
                        self.namespace,
                        f"pod/{pod_name}",
                        "--all-containers=true",
                        "--prefix=true",
                    ],
                    check=False,
                )
                current_content = current.stdout
                if current.stderr:
                    current_content += f"\n--- stderr ---\n{current.stderr}"
                (job_dir / f"{pod_name}.current.log").write_text(
                    current_content, encoding="utf-8"
                )
                if current.returncode:
                    raise RuntimeError(
                        f"Could not archive GuideLLM pod logs for {pod_name}: "
                        f"{current.stderr.strip() or 'oc logs failed'}"
                    )
                previous = self._oc(
                    [
                        "logs",
                        "-n",
                        self.namespace,
                        f"pod/{pod_name}",
                        "--all-containers=true",
                        "--prefix=true",
                        "--previous=true",
                    ],
                    check=False,
                )
                previous_content = previous.stdout
                if previous.stderr:
                    previous_content += f"\n--- stderr ---\n{previous.stderr}"
                (job_dir / f"{pod_name}.previous.log").write_text(
                    previous_content, encoding="utf-8"
                )
        print(f"Archived benchmark diagnostics: {job_dir}", flush=True)
        return job_dir

    def _log_to_mlflow(
        self,
        job_name: str,
        model: str,
        profile: dict[str, Any],
        target: str,
        concurrency: str,
        logs: str,
    ) -> None:
        """Log the authoritative in-cluster Job output without affecting a run."""
        if not self.mlflow_uri:
            return
        try:
            import mlflow
            from mlflow.utils.workspace_context import WorkspaceContext

            with WorkspaceContext(self.mlflow_workspace):
                mlflow.set_tracking_uri(self.mlflow_uri)
                mlflow.set_experiment(self.mlflow_experiment)
                with mlflow.start_run(run_name=job_name):
                    mlflow.set_tags(
                        {
                            "model": model,
                            "target_service": target,
                            "guidellm_job": job_name,
                            "concurrency": concurrency,
                            "workspace": self.mlflow_workspace or "default",
                        }
                    )
                    mlflow.log_params(
                        {
                            "input_sequence_length": profile["isl"],
                            "output_sequence_length": profile["osl"],
                        }
                    )
                    mlflow.log_metrics(self._mlflow_metrics(logs, concurrency))
                    mlflow.log_text(logs, "guidellm.log")
        except Exception as exc:
            # Observability must never make a benchmark result unusable.
            print(f"MLflow logging skipped for {job_name}: {exc}", flush=True)

    @staticmethod
    def _mlflow_metrics(logs: str, concurrency: str) -> dict[str, float]:
        """Extract comparison metrics from GuideLLM's textual result tables."""
        def rows_after(heading: str) -> list[list[str]]:
            position = logs.find(heading)
            if position < 0:
                return []
            section = logs[position:].split("\n\n", 1)[0]
            return [
                [cell.strip() for cell in line.split("|")[1:-1]]
                for line in section.splitlines()[1:]
                if "| concurrent |" in line
            ]

        def key_prefix(concurrency: str) -> str:
            value = float(concurrency)
            label = str(int(value)) if value.is_integer() else str(value).replace(".", "_")
            return f"concurrency_{label}"

        metrics: dict[str, float] = {}
        streams = concurrency.split(",")
        for stream, row in zip(streams, rows_after("Request Latency Statistics")):
            if len(row) < 9:
                continue
            prefix = key_prefix(stream)
            metrics[f"{prefix}_ttft_p50_ms"] = float(row[3])
            metrics[f"{prefix}_ttft_p95_ms"] = float(row[4])
            metrics[f"{prefix}_itl_p50_ms"] = float(row[7])
            metrics[f"{prefix}_itl_p95_ms"] = float(row[8])
        for stream, row in zip(streams, rows_after("Server Throughput Statistics")):
            if len(row) < 6:
                continue
            prefix = key_prefix(stream)
            metrics[f"{prefix}_output_tokens_per_second"] = float(row[5])
        return metrics

    @staticmethod
    def _summarize(job_name: str, target: str, logs: str) -> str:
        """Return the completed Job's comparison-ready console metrics.

        The result PVC is retained for operators, but it is intentionally not
        presented as a controller-local filename.  The agent receives the
        canonical GuideLLM summary emitted by the Job itself.
        """
        def rows_after(heading: str) -> str:
            position = logs.find(heading)
            if position < 0:
                return "not found"
            section = logs[position:].split("\n\n", 1)[0]
            rows = [
                line.strip()
                for line in section.splitlines()[1:]
                if "| concurrent |" in line
            ]
            return "\n".join(rows) if rows else "not found"

        return "\n".join(
            [
                f"GuideLLM Job: {job_name}",
                f"Target Service: {target}",
                "=== BENCHMARK METRICS (compare these rows directly) ===",
                "Completed requests (input/output token totals):",
                rows_after("Run Summary Info"),
                "Latency (request seconds, TTFT/TTFOT/ITL/TPOT milliseconds; median/p95):",
                rows_after("Request Latency Statistics"),
                "Throughput (request, input-token, output-token, total-token per second):",
                rows_after("Server Throughput Statistics"),
                "=== END BENCHMARK METRICS ===",
            ]
        )

    def _oc(self, args: list[str], *, input: str | None = None, check: bool = True):
        return subprocess.run(
            ["oc", "--kubeconfig", self.kubeconfig, *args],
            input=input, text=True, capture_output=True, check=check,
        )

"""Tests for profile-driven agentic environment rendering."""

from __future__ import annotations

import json
from types import SimpleNamespace

from auto_tune_vllm.agent.environment import render_environment_resources
from auto_tune_vllm.agent.main import write_controller_metadata
from auto_tune_vllm.agent.pod_manager import PodManager
from auto_tune_vllm.agent.provision import _job_complete, cleanup_baseline
from auto_tune_vllm.agent.tuning_profile import TuningProfile, load_tuning_profile


def _profile() -> TuningProfile:
    return TuningProfile(
        name="test-profile",
        benchmark_profiles=["long_context_16k_1k"],
        max_tensor_parallel_size=4,
        environment={
            "namespace": "test",
            "node_selector": {"accelerator": "h200"},
            "image": "example/vllm:latest",
            "model": {"id": "org/model", "cache_path": "/models/model"},
            "cache": {
                "pvc_name": "models",
                "storage_class": "fast",
                "size": "100Gi",
            },
            "model_access": {
                "source_namespace": "source",
                "secret_name": "hf-token",
                "secret_key": "HF_TOKEN",
            },
            "download_job_name": "download-model",
            "baseline": {
                "name": "model-baseline",
                "resources": {"cpu": "8", "memory": "32Gi", "gpus": 4},
            },
            "benchmark": {
                "image": "ghcr.io/vllm-project/guidellm:v0.7.3",
                "results_pvc": {
                    "name": "guidellm-results",
                    "storage_class": "fast",
                    "size": "20Gi",
                },
                "resources": {
                    "requests": {"cpu": "1", "memory": "1Gi"},
                    "limits": {"cpu": "2", "memory": "2Gi"},
                },
            },
        },
        runtime={"max_model_len": 32768, "base_vllm_args": []},
    )


def test_rendered_environment_keeps_profile_constraints_out_of_manifests():
    _, results_pvc, job, deployment, service, experiment = render_environment_resources(
        _profile()
    )

    assert results_pvc["metadata"]["name"] == "guidellm-results"
    assert job["metadata"]["name"] == "download-model"
    assert deployment["spec"]["strategy"]["type"] == "Recreate"
    assert deployment["spec"]["selector"]["matchLabels"] == service["spec"]["selector"]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert "--tensor-parallel-size=4" in container["args"]
    assert "--max-model-len=32768" in container["args"]
    assert experiment["spec"]["containers"][0]["args"] == container["args"]
    assert container["volumeMounts"][0].get("readOnly") is None
    for workload in (deployment["spec"]["template"]["spec"], experiment["spec"]):
        assert {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}} in workload[
            "volumes"
        ]
    assert {"name": "dshm", "mountPath": "/dev/shm"} in container["volumeMounts"]


def test_job_complete_recognizes_completed_download(monkeypatch):
    def fake_oc(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": {"succeeded": 1}}),
        )

    monkeypatch.setattr("auto_tune_vllm.agent.provision._oc", fake_oc)

    assert _job_complete("kubeconfig", "namespace", "download-model")


def test_cleanup_baseline_deletes_only_deployment_and_service(monkeypatch):
    calls = []

    def fake_oc(kubeconfig, args, **_kwargs):
        calls.append((kubeconfig, args))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("auto_tune_vllm.agent.provision._oc", fake_oc)

    cleanup_baseline(
        kubeconfig="kubeconfig",
        namespace="namespace",
        deployment="model-baseline",
        service="model-baseline",
    )

    assert calls == [
        (
            "kubeconfig",
            ["delete", "deployment", "model-baseline", "-n", "namespace", "--wait=true"],
        ),
        (
            "kubeconfig",
            ["delete", "service", "model-baseline", "-n", "namespace", "--wait=true"],
        ),
    ]


def test_controller_metadata_excludes_credentials(tmp_path):
    args = SimpleNamespace(
        controller_metadata_dir=tmp_path,
        oc_namespace="test",
        model="org/model",
        vllm_version="0.24.0",
        recipe_hardware="H200",
        profiles=["long_context_16k_1k"],
        max_iterations=100,
        benchmark_config={"concurrency": [1, 50]},
        cleanup_baseline_after_benchmark=True,
        api_key="must-not-appear",
    )

    path = write_controller_metadata(args)

    assert path is not None
    recorded = path.read_text(encoding="utf-8")
    assert "must-not-appear" not in recorded
    assert '"event": "controller_started"' in recorded


def test_experiment_delete_is_graceful_and_archives_first(tmp_path, monkeypatch):
    template = tmp_path / "pod.yaml"
    template.write_text(
        "apiVersion: v1\nkind: Pod\nmetadata: {name: template}\nspec: {containers: [{name: vllm}]}",
        encoding="utf-8",
    )
    manager = PodManager(
        namespace="test",
        base_pod_yaml_path=str(template),
        artifact_dir=tmp_path / "artifacts",
    )
    manager.active_pods["experiment"] = {}
    calls = []

    monkeypatch.setattr(manager, "archive_pod_logs", lambda _pod_name: tmp_path)

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("auto_tune_vllm.agent.pod_manager.subprocess.run", fake_run)

    manager.delete_pod("experiment")

    pod_delete = next(command for command in calls if "pod" in command)
    assert "--wait=true" in pod_delete
    assert "--force" not in pod_delete
    assert "--grace-period=0" not in pod_delete


def test_profile_experiment_priorities_and_recipe_source(tmp_path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """optimization:
  objective: throughput
  recipe_model_id: org/base-model
  priority_items:
    - FP8 KV cache
    - batching
workload:
  benchmark_profiles: [long_context_16k_1k]
constraints:
  max_experiments: 7
""",
        encoding="utf-8",
    )

    profile = load_tuning_profile(profile_path)

    assert profile.recipe_model_id == "org/base-model"
    assert profile.priority_items == ["FP8 KV cache", "batching"]
    assert profile.max_experiments == 7


def test_experiment_rejects_vllm_quantization_flag(tmp_path):
    template = tmp_path / "pod.yaml"
    template.write_text(
        "apiVersion: v1\nkind: Pod\nmetadata: {name: template}\nspec: {containers: [{name: vllm}]}",
        encoding="utf-8",
    )
    manager = PodManager(namespace="test", base_pod_yaml_path=str(template))

    try:
        manager._build_pod_manifest("experiment", ["--quantization=fp8"])
    except ValueError as exc:
        assert "--quantization" in str(exc)
    else:
        raise AssertionError("Expected --quantization to be rejected")

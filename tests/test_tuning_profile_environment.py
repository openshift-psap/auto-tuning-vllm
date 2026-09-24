"""Tests for profile-driven agentic environment rendering."""

from __future__ import annotations

import json
from types import SimpleNamespace

from auto_tune_vllm.agent.environment import render_environment_resources
from auto_tune_vllm.agent.main import write_controller_metadata
from auto_tune_vllm.agent.provision import _job_complete, cleanup_baseline
from auto_tune_vllm.agent.tuning_profile import TuningProfile


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

"""Tests for profile-driven agentic environment rendering."""

from __future__ import annotations

import json
from types import SimpleNamespace

from auto_tune_vllm.agent.environment import render_environment_resources
from auto_tune_vllm.agent.provision import _job_complete
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
        runtime={"max_model_len": 32768, "base_vllm_args": ["--enable-prefix-caching"]},
    )


def test_rendered_environment_keeps_profile_constraints_out_of_manifests():
    _, results_pvc, job, deployment, service, experiment = render_environment_resources(
        _profile()
    )

    assert results_pvc["metadata"]["name"] == "guidellm-results"
    assert job["metadata"]["name"] == "download-model"
    assert deployment["spec"]["selector"]["matchLabels"] == service["spec"]["selector"]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert "--tensor-parallel-size=4" in container["args"]
    assert "--max-model-len=32768" in container["args"]
    assert experiment["spec"]["containers"][0]["args"] == container["args"]


def test_job_complete_recognizes_completed_download(monkeypatch):
    def fake_oc(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": {"succeeded": 1}}),
        )

    monkeypatch.setattr("auto_tune_vllm.agent.provision._oc", fake_oc)

    assert _job_complete("kubeconfig", "namespace", "download-model")

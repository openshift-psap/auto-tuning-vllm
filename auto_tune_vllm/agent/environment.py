"""Render agentic-tuning Kubernetes resources from a tuning profile."""

from __future__ import annotations

from typing import Any

from .tuning_profile import TuningProfile


def render_environment_resources(profile: TuningProfile) -> list[dict[str, Any]]:
    """Render cache, downloader, baseline, service, and experiment template.

    The caller is responsible for copying the referenced model-access secret into
    the profile namespace before applying the returned resources.
    """
    environment = profile.environment
    if environment is None:
        raise ValueError("A provisionable profile requires an environment mapping")
    runtime = profile.runtime
    if runtime is None:
        raise ValueError("A provisionable profile requires a runtime mapping")

    model = _required_mapping(environment, "model")
    cache = _required_mapping(environment, "cache")
    baseline = _required_mapping(environment, "baseline")
    benchmark = _required_mapping(environment, "benchmark")
    benchmark_results = _required_mapping(benchmark, "results_pvc")
    access = _required_mapping(environment, "model_access")
    node_selector = _required_mapping(environment, "node_selector")
    image = _required_string(environment, "image")
    model_id = _required_string(model, "id")
    model_path = _required_string(model, "cache_path")
    pvc_name = _required_string(cache, "pvc_name")
    baseline_name = _required_string(baseline, "name")
    downloader_name = environment.get("download_job_name", f"download-{baseline_name}")
    if not isinstance(downloader_name, str) or not downloader_name:
        raise ValueError("environment.download_job_name must be a non-empty string")
    resources = _required_mapping(baseline, "resources")
    max_tp = profile.max_tensor_parallel_size
    if max_tp is None:
        raise ValueError("A provisionable profile requires max_tensor_parallel_size")

    container = _vllm_container(
        image=image,
        model_path=model_path,
        model_id=model_id,
        max_tp=max_tp,
        runtime=runtime,
        resources=resources,
        pvc_name=pvc_name,
    )
    downloader = _downloader_container(
        image=image,
        model_id=model_id,
        model_path=model_path,
        secret_name=_required_string(access, "secret_name"),
        secret_key=_required_string(access, "secret_key"),
        pvc_name=pvc_name,
    )
    return [
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": pvc_name},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": _required_string(cache, "storage_class"),
                "resources": {"requests": {"storage": _required_string(cache, "size")}},
            },
        },
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": _required_string(benchmark_results, "name")},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": _required_string(benchmark_results, "storage_class"),
                "resources": {
                    "requests": {"storage": _required_string(benchmark_results, "size")}
                },
            },
        },
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": downloader_name},
            "spec": {
                "backoffLimit": 1,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "nodeSelector": node_selector,
                        "containers": [downloader],
                        "volumes": _model_cache_volume(pvc_name),
                    }
                },
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": baseline_name},
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {"app.kubernetes.io/name": baseline_name}
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app.kubernetes.io/name": baseline_name,
                            "app.kubernetes.io/part-of": "agentic-tuning",
                        }
                    },
                    "spec": {
                        "nodeSelector": node_selector,
                        "containers": [container],
                        "volumes": _model_cache_volume(pvc_name),
                    },
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": baseline_name},
            "spec": {
                "selector": {"app.kubernetes.io/name": baseline_name},
                "ports": [{"name": "http", "port": 8000, "targetPort": "http"}],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "vllm-tune-placeholder",
                "labels": {"vllm-experiment": "true"},
            },
            "spec": {
                "restartPolicy": "Never",
                "nodeSelector": node_selector,
                "containers": [container],
                "volumes": _model_cache_volume(pvc_name),
            },
        },
    ]


def _vllm_container(
    *,
    image: str,
    model_path: str,
    model_id: str,
    max_tp: int,
    runtime: dict[str, Any],
    resources: dict[str, Any],
    pvc_name: str,
) -> dict[str, Any]:
    max_model_len = runtime.get("max_model_len")
    if not isinstance(max_model_len, int) or max_model_len < 1:
        raise ValueError("runtime.max_model_len must be a positive integer")
    base_args = runtime.get("base_vllm_args") or []
    if not isinstance(base_args, list) or not all(
        isinstance(arg, str) for arg in base_args
    ):
        raise ValueError("runtime.base_vllm_args must be a list of strings")
    return {
        "name": "vllm",
        "image": image,
        "args": [
            model_path,
            f"--served-model-name={model_id}",
            f"--tensor-parallel-size={max_tp}",
            f"--max-model-len={max_model_len}",
            *base_args,
        ],
        "env": [
            {"name": "HOME", "value": "/tmp"},
            {"name": "USER", "value": "vllm"},
            {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
            {"name": "TORCHINDUCTOR_CACHE_DIR", "value": "/tmp/torchinductor"},
        ],
        "ports": [{"name": "http", "containerPort": 8000}],
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": "http"},
            "initialDelaySeconds": 10,
            "periodSeconds": 10,
        },
        "resources": {
            "requests": {
                "cpu": resources["cpu"],
                "memory": resources["memory"],
                "nvidia.com/gpu": resources["gpus"],
            },
            "limits": {
                "cpu": resources["cpu"],
                "memory": resources["memory"],
                "nvidia.com/gpu": resources["gpus"],
            },
        },
        "volumeMounts": [
            {"name": "model-cache", "mountPath": "/models", "readOnly": True}
        ],
    }


def _downloader_container(
    *,
    image: str,
    model_id: str,
    model_path: str,
    secret_name: str,
    secret_key: str,
    pvc_name: str,
) -> dict[str, Any]:
    del pvc_name
    return {
        "name": "downloader",
        "image": image,
        "command": [
            "python3",
            "-c",
            (
                "import os\nfrom huggingface_hub import snapshot_download\n"
                f"snapshot_download(repo_id={model_id!r}, local_dir={model_path!r}, "
                'token=os.environ["HF_TOKEN"])\n'
            ),
        ],
        "env": [
            {"name": "HOME", "value": "/tmp"},
            {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
            {"name": "HF_HOME", "value": "/tmp/.cache/huggingface"},
            {"name": "HF_HUB_DISABLE_XET", "value": "1"},
            {
                "name": "HF_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": secret_key}},
            },
        ],
        "volumeMounts": [{"name": "model-cache", "mountPath": "/models"}],
    }


def _model_cache_volume(pvc_name: str) -> list[dict[str, Any]]:
    return [{"name": "model-cache", "persistentVolumeClaim": {"claimName": pvc_name}}]


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"environment.{key} must be a mapping")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value

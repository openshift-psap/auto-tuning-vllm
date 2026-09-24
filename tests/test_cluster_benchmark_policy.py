"""Study-level consistency checks for cluster GuideLLM benchmarks."""

from auto_tune_vllm.agent.benchmark_job import ClusterBenchmarkRunner
from auto_tune_vllm.agent.tools import ClusterCurlExecutor, _handle_run_benchmark


class FakeRunner:
    config = {"concurrency": [1, 50], "max_seconds": 45}

    def run(self, **kwargs):
        return str(kwargs)


def test_cluster_benchmark_uses_configured_policy_when_agent_omits_it():
    result = _handle_run_benchmark(
        {
            "profile": "balanced",
            "endpoint": "http://candidate:8000",
            "model": "test/model",
        },
        None,
        [],
        benchmark_runner=FakeRunner(),
        benchmark_target="http://candidate:8000",
    )

    assert result.success
    assert "'concurrency': '1,50'" in result.output
    assert "'max_seconds': 45" in result.output


def test_cluster_benchmark_rejects_different_concurrency():
    result = _handle_run_benchmark(
        {
            "profile": "balanced",
            "endpoint": "http://candidate:8000",
            "model": "test/model",
            "concurrency": "1",
        },
        None,
        [],
        benchmark_runner=FakeRunner(),
        benchmark_target="http://candidate:8000",
    )

    assert not result.success
    assert "fixed by the study benchmark policy" in result.error


def test_summary_includes_each_concurrency_row():
    logs = """Run Summary Info
| concurrent | 1.0 | 1.0 |
| concurrent | 50.0 | 49.5 |

Server Throughput Statistics
| concurrent | 1.0 | 1.0 | 0.2 |
| concurrent | 50.0 | 49.5 | 2.0 |
"""

    summary = ClusterBenchmarkRunner._summarize("job", "http://target", logs)

    assert summary.count("| concurrent | 1.0") == 2
    assert summary.count("| concurrent | 50.0") == 2


def test_mlflow_metrics_include_throughput_ttft_and_itl_percentiles():
    logs = """Request Latency Statistics
| concurrent | 4.3 | 4.5 | 70.0 | 80.0 | 70.0 | 80.0 | 4.2 | 5.0 | 4.2 | 5.0 |
| concurrent | 18.3 | 31.1 | 610.6 | 11365.8 | 610.6 | 11365.8 | 14.8 | 19.5 | 17.1 | 28.2 |

Server Throughput Statistics
| concurrent | 1.0 | 1.0 | 0.2 | 4418.9 | 237.2 | 4035.1 |
| concurrent | 50.0 | 48.5 | 2.0 | 47794.2 | 2511.0 | 50305.2 |
"""

    metrics = ClusterBenchmarkRunner._mlflow_metrics(logs, "1,50")

    assert metrics["concurrency_1_output_tokens_per_second"] == 237.2
    assert metrics["concurrency_50_ttft_p95_ms"] == 11365.8
    assert metrics["concurrency_50_itl_p50_ms"] == 14.8


def test_curl_pod_manifest_mounts_model_cache_without_overrides():
    manifest = ClusterCurlExecutor("test", cache_pvc_name="models")._build_manifest(
        "agent-curl-test", "ls /models"
    )

    container = manifest["spec"]["containers"][0]
    assert container["image"] == "curlimages/curl:8.10.1"
    assert container["volumeMounts"][0]["readOnly"] is True
    assert manifest["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"] == "models"

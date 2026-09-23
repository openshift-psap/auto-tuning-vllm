import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).parents[1] / "auto_tune_vllm" / "knowledge" / "vllm_pr_index.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "vllm_pr_index_test_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
VllmPullRequestIndex = _MODULE.VllmPullRequestIndex


def _pull(number: int, title: str, body: str) -> dict:
    return {
        "number": number,
        "title": title,
        "body": body,
        "html_url": f"https://github.com/vllm-project/vllm/pull/{number}",
        "user": {"login": "maintainer"},
        "merged_at": "2026-01-02T03:04:05Z",
        "updated_at": "2026-01-02T03:04:05Z",
        "labels": [{"name": "performance"}],
    }


def test_search_indexes_pr_text_files_and_facets(tmp_path):
    index = VllmPullRequestIndex(tmp_path / "index.sqlite")
    index.upsert_pull_request(
        _pull(
            101,
            "Improve H100 decode latency",
            "Use FP8 KV cache for Qwen MoE serving.",
        ),
        files=["vllm/v1/core/kv_cache_manager.py"],
    )

    results = index.search("KV cache decode", architecture="qwen", hardware="h100")

    assert [result.number for result in results] == [101]
    assert results[0].facets["architecture"] == ["qwen", "moe"]
    assert results[0].facets["hardware"] == ["h100"]


def test_search_without_query_returns_most_recent_pull_request(tmp_path):
    index = VllmPullRequestIndex(tmp_path / "index.sqlite")
    older = _pull(1, "Older change", "attention backend")
    older["merged_at"] = "2025-01-01T00:00:00Z"
    newer = _pull(2, "Newer change", "attention backend")
    newer["merged_at"] = "2026-01-01T00:00:00Z"
    index.upsert_pull_request(older)
    index.upsert_pull_request(newer)

    assert [result.number for result in index.search()] == [2, 1]


def test_manifest_reports_index_provenance(tmp_path):
    index = VllmPullRequestIndex(tmp_path / "index.sqlite")
    index.upsert_pull_request(_pull(12, "CUDA graph tuning", "H200 compilation"))

    manifest = index.manifest()

    assert manifest["repository"] == "vllm-project/vllm"
    assert manifest["document_count"] == 1


def test_sync_fetches_merged_prs_and_changed_files(tmp_path, monkeypatch):
    index = VllmPullRequestIndex(tmp_path / "index.sqlite")
    pull = _pull(44, "H100 cache tuning", "Reduce KV cache overhead")

    def fake_github_get(path, _params=None):
        if path.endswith("/pulls"):
            return [pull]
        if path.endswith("/pulls/44/files"):
            return [{"filename": "vllm/worker/gpu_model_runner.py"}]
        if path.endswith("/pulls/44"):
            return pull
        raise AssertionError(f"Unexpected API path: {path}")

    monkeypatch.setattr(index, "_github_get", fake_github_get)

    summary = index.sync()

    assert summary["indexed"] == 1
    assert index.search("gpu model runner", hardware="h100")[0].number == 44

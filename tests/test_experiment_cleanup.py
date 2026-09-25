"""Lifecycle safeguards for isolated experiment pods."""

from subprocess import CompletedProcess, TimeoutExpired

import pytest
import yaml

from auto_tune_vllm.agent.pod_manager import ExperimentLifecycleError, PodManager
from auto_tune_vllm.agent.agentic import AgenticRunner
from auto_tune_vllm.agent.tools import ToolResult


@pytest.fixture
def pod_manager(tmp_path):
    template = tmp_path / "experiment.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": "experiment"},
                "spec": {"containers": [{"name": "vllm", "image": "test"}]},
            }
        ),
        encoding="utf-8",
    )
    return PodManager(namespace="test", base_pod_yaml_path=str(template))


def test_delete_failure_is_terminal_and_keeps_experiment_tracked(
    monkeypatch, pod_manager
):
    pod_manager.active_pods["vllm-tune-1"] = {}
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if "pod" in command:
            raise TimeoutExpired(command, 30)
        return CompletedProcess(command, 0, "service/vllm-tune-1 deleted", "")

    monkeypatch.setattr("auto_tune_vllm.agent.pod_manager.subprocess.run", fake_run)

    with pytest.raises(ExperimentLifecycleError, match="User intervention is required"):
        pod_manager.delete_pod("vllm-tune-1")

    assert "vllm-tune-1" in pod_manager.active_pods
    assert any("service" in command for command in calls)


def test_service_delete_failure_is_terminal(monkeypatch, pod_manager):
    pod_manager.active_pods["vllm-tune-1"] = {}

    def fake_run(command, **_kwargs):
        if "service" in command:
            return CompletedProcess(command, 1, "", "forbidden")
        return CompletedProcess(command, 0, "pod/vllm-tune-1 deleted", "")

    monkeypatch.setattr("auto_tune_vllm.agent.pod_manager.subprocess.run", fake_run)

    with pytest.raises(ExperimentLifecycleError, match="failed to delete Service"):
        pod_manager.delete_pod("vllm-tune-1")

    assert "vllm-tune-1" in pod_manager.active_pods


def test_user_can_acknowledge_resolved_cleanup(monkeypatch, pod_manager):
    pod_manager.active_pods["vllm-tune-1"] = {}

    def fake_run(command, **_kwargs):
        return CompletedProcess(command, 1, "", "Error from server (NotFound)")

    monkeypatch.setattr("auto_tune_vllm.agent.pod_manager.subprocess.run", fake_run)

    pod_manager.confirm_cleanup_resolved("vllm-tune-1")

    assert "vllm-tune-1" not in pod_manager.active_pods


def test_cleanup_all_propagates_deletion_failure(monkeypatch, pod_manager):
    pod_manager.active_pods["vllm-tune-1"] = {}

    def fake_run(command, **_kwargs):
        if "pod" in command:
            raise TimeoutExpired(command, 30)
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("auto_tune_vllm.agent.pod_manager.subprocess.run", fake_run)

    with pytest.raises(ExperimentLifecycleError, match="User intervention is required"):
        pod_manager.cleanup_all()


@pytest.mark.parametrize(
    "vllm_args",
    [
        ["--enable-prefix-caching"],
        ["--no-enable-prefix-caching"],
        ["--enable-prefix-caching=false"],
    ],
)
def test_experiment_cannot_change_prefix_caching(pod_manager, vllm_args):
    with pytest.raises(ValueError, match="fixed study control"):
        pod_manager._build_pod_manifest("vllm-tune-1", vllm_args)


def test_resume_adds_user_intervention_context(monkeypatch):
    class FakePodManager:
        def __init__(self):
            self.confirmed_pod = None

        def confirm_cleanup_resolved(self, pod_name):
            self.confirmed_pod = pod_name

    class FakeTools:
        pod_manager = FakePodManager()

    runner = AgenticRunner(llm_client=None, tools=FakeTools())
    runner.state.paused = True
    runner.messages = [{"role": "user", "content": "prior context"}]

    def fake_run(*, initialize):
        assert initialize is False
        return runner.state

    monkeypatch.setattr(runner, "run", fake_run)

    result = runner.resume(
        "vllm-tune-1",
        "The API server was unavailable; I restored it. Avoid deleting while it is down.",
    )

    assert result is runner.state
    assert runner.tools.pod_manager.confirmed_pod == "vllm-tune-1"
    assert "Avoid deleting while it is down" in runner.messages[-1]["content"]


def test_experiment_limit_counts_launched_pods_only():
    class FakeTools:
        def __init__(self):
            self.calls = []

        def dispatch(self, name, inputs):
            self.calls.append((name, inputs))
            return ToolResult(tool=name, success=True, output="pod created")

    tools = FakeTools()
    runner = AgenticRunner(llm_client=None, tools=tools, max_experiments=1)

    first = runner._execute_tool("create_vllm_pod", {}, "first")
    second = runner._execute_tool("create_vllm_pod", {}, "second")
    diagnostic = runner._execute_tool("fetch_vllm_logs", {}, "logs")

    assert runner.state.experiments_started == 1
    assert "1/1" in first["content"]
    assert "experiment limit reached" in second["content"]
    assert diagnostic["content"] == "pod created"
    assert [call[0] for call in tools.calls] == ["create_vllm_pod", "fetch_vllm_logs"]

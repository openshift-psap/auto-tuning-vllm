from auto_tune_vllm.agent.recipe_fetcher import (
    fetch_vllm_recipe,
    format_recipe_briefing,
    is_vllm_version_compatible,
)


def test_recipe_version_compatibility_compares_release_versions():
    assert is_vllm_version_compatible("0.25.0", "0.25.0")
    assert is_vllm_version_compatible("0.26.1", "0.25.0")
    assert not is_vllm_version_compatible("0.24.0", "0.25.0")


def test_incompatible_recipe_withholds_recipe_only_arguments():
    recipe = {
        "matched_hf_id": "example/model",
        "min_vllm_version": "0.25.0",
        "runtime_vllm_version": "0.24.0",
        "version_compatible": False,
        "base_args": [],
        "hardware_args": [],
        "variants": {},
        "spec_decoding": None,
        "recommended_command": "vllm serve example/model --new-option",
    }

    briefing = format_recipe_briefing(recipe)

    assert "WARNING: Recipe requires vLLM >= 0.25.0" in briefing
    assert "FIRST EXPERIMENT" not in briefing


def test_fetcher_removes_incompatible_recipe_arguments(monkeypatch):
    responses = [
        [{"hf_id": "example/model"}],
        {
            "model": {
                "min_vllm_version": "0.25.0",
                "base_args": ["--new-option"],
                "recommended_command": "vllm serve example/model --new-option",
            },
            "variants": {"fp8": {"extra_args": ["--quantization", "fp8"]}},
        },
    ]
    monkeypatch.setattr(
        "auto_tune_vllm.agent.recipe_fetcher._fetch_json",
        lambda *_args, **_kwargs: responses.pop(0),
    )

    recipe = fetch_vllm_recipe("example/model", runtime_vllm_version="0.24.0")

    assert recipe is not None
    assert recipe["version_compatible"] is False
    assert recipe["base_args"] == []
    assert recipe["variants"] == {}
    assert recipe["recommended_command"] == ""

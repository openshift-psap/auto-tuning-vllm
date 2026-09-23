"""
Fetch and parse vLLM serving recipes from recipes.vllm.ai.

Called once at agent startup on the controller — not from inside the pod.
Returns a structured briefing that is injected into the agent's first message,
so Claude sees the full recipe without token truncation.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

RECIPES_INDEX_URL = "https://recipes.vllm.ai/models.json"
RECIPE_URL_TEMPLATE = "https://recipes.vllm.ai/{hf_id}.json"

_HARDWARE_ALIASES: dict[str, list[str]] = {
    "H200": ["H200", "h200"],
    "H100": ["H100", "h100"],
    "A100": ["A100", "a100"],
    "MI300X": ["MI300X", "mi300x"],
}


def _fetch_json(url: str, timeout: int = 10) -> Optional[dict | list]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-tune-vllm/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None


def _normalize(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace("/", "")


def _find_best_match(model_id: str, catalog: list[dict]) -> Optional[dict]:
    """Find the closest recipe entry for model_id."""
    # 1. Exact match on hf_id (case-insensitive)
    for entry in catalog:
        if entry.get("hf_id", "").lower() == model_id.lower():
            return entry

    # 2. Normalised match (strip punctuation)
    norm_target = _normalize(model_id)
    for entry in catalog:
        if _normalize(entry.get("hf_id", "")) == norm_target:
            return entry

    # 3. Token-overlap score — compare model-name tokens (after /), require same org prefix
    # when possible and use whole-token matching to avoid substring false positives.
    target_org = model_id.lower().split("/")[0] if "/" in model_id else ""
    target_name_tokens = [
        t for t in model_id.lower().split("/")[-1].split("-") if len(t) > 2
    ]
    target_token_set = set(target_name_tokens)

    best_score: float = 0.0
    best_entry = None
    for entry in catalog:
        hf_id = entry.get("hf_id", "")
        entry_org = hf_id.lower().split("/")[0] if "/" in hf_id else ""
        entry_name = hf_id.lower().split("/")[-1]
        entry_name_tokens = set(t for t in entry_name.split("-") if len(t) > 2)

        # Whole-token intersection (not substring) to avoid "gemma" matching "diffusiongemma"
        overlap = target_token_set & entry_name_tokens
        if len(overlap) < 2:
            continue

        score = float(len(overlap))
        # Boost if org matches
        if target_org and entry_org == target_org:
            score += 0.5

        if score > best_score:
            best_score, best_entry = score, entry

    if best_score >= 2:
        return best_entry
    return None


def _extract_hardware_args(recipe: dict, hardware: str) -> list[str]:
    overrides = recipe.get("hardware_overrides", {})
    for alias in _HARDWARE_ALIASES.get(hardware, [hardware, hardware.lower()]):
        if alias in overrides:
            hw = overrides[alias]
            if isinstance(hw, dict):
                return hw.get("extra_args", [])
    return []


def _extract_spec_decoding(recipe: dict) -> Optional[dict]:
    for section in ("features", "opt_in_features"):
        val = recipe.get(section)
        if isinstance(val, dict):
            sd = val.get("spec_decoding")
            if sd:
                return sd
    return None


def fetch_vllm_recipe(
    model_id: str,
    hardware: str = "H200",
    timeout: int = 10,
) -> Optional[dict]:
    """Fetch and parse the vLLM recipe for *model_id* from recipes.vllm.ai.

    Returns a structured dict suitable for passing to format_recipe_briefing,
    or None if the model is not in the catalog or the network is unavailable.
    """
    catalog = _fetch_json(RECIPES_INDEX_URL, timeout=timeout)
    if not isinstance(catalog, list):
        return None

    entry = _find_best_match(model_id, catalog)
    if not entry:
        return None

    hf_id = entry["hf_id"]
    recipe = _fetch_json(RECIPE_URL_TEMPLATE.format(hf_id=hf_id), timeout=timeout)
    if not isinstance(recipe, dict):
        return None

    return {
        "matched_hf_id": hf_id,
        "base_args": recipe.get("model", {}).get("base_args", []),
        "recommended_command": recipe.get("model", {}).get("recommended_command", ""),
        "variants": recipe.get("variants", {}),
        "hardware_args": _extract_hardware_args(recipe, hardware),
        "spec_decoding": _extract_spec_decoding(recipe),
        "features": recipe.get("features", {}),
        "opt_in_features": recipe.get("opt_in_features", {}),
    }


def format_recipe_briefing(recipe: dict, hardware: str = "H200") -> str:
    """Format a recipe dict into a structured briefing block for the agent."""
    hf_id = recipe["matched_hf_id"]
    lines = [f"=== PRE-FETCHED vLLM RECIPE: {hf_id} (hardware: {hardware}) ==="]

    if recipe.get("recommended_command"):
        lines.append(f"\nRecommended serving command:\n  {recipe['recommended_command'][:300]}")

    base_args = recipe.get("base_args", [])
    hw_args = recipe.get("hardware_args", [])
    combined_first_exp = base_args + hw_args
    if combined_first_exp:
        lines.append(f"\nFIRST EXPERIMENT (base_args + {hardware} hardware_args):")
        lines.append(f"  {' '.join(combined_first_exp)}")
    elif base_args:
        lines.append(f"\nBase args: {' '.join(base_args)}")
    if hw_args and not combined_first_exp:
        lines.append(f"Hardware-specific args ({hardware}): {' '.join(hw_args)}")

    variants = recipe.get("variants", {})
    if variants:
        lines.append("\nVariants (precision / quantization options):")
        for name, variant in variants.items():
            extra = variant.get("extra_args", [])
            desc = variant.get("description", "")
            line = f"  {name}: {' '.join(extra)}" if extra else f"  {name}: (no extra args)"
            if desc:
                line += f"  # {desc}"
            lines.append(line)

    sd = recipe.get("spec_decoding")
    if sd:
        lines.append(f"\nSpeculative decoding config:")
        if isinstance(sd, dict):
            for k, v in sd.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {sd}")

    lines.append(
        "\nINSTRUCTION: Use the FIRST EXPERIMENT args above as your initial experiment "
        "(create_vllm_pod with those vllm_args). Then explore variants. "
        "Do NOT curl recipes.vllm.ai from the pod — this recipe is already complete."
    )
    return "\n".join(lines)

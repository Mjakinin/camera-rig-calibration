from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any


_INSTALLED = False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _candidate(payload: dict[str, Any], marker_id: int) -> dict[str, Any]:
    for item in payload.get("ap03_single_scale_marker", {}).get("candidates", []):
        try:
            if int(item.get("id")) == int(marker_id):
                return item
        except (TypeError, ValueError):
            continue
    return {}


def _root_candidate(payload: dict[str, Any], root_camera: str) -> dict[str, Any]:
    for item in payload.get("ap01_root_camera", {}).get("candidates", []):
        if str(item.get("id")) == str(root_camera):
            return item
    return {}


def _ap02_reference_candidate(payload: dict[str, Any], marker_id: int) -> dict[str, Any]:
    for item in payload.get("ap02_reference_marker", {}).get("candidates", []):
        try:
            if int(item.get("id")) == int(marker_id):
                return item
        except (TypeError, ValueError):
            continue
    return {}


def _preferred_export_compatible(config: Any, resolved: Any, marker_id: int) -> bool:
    """Check whether enabled method exporters can express their result in marker_id.

    AP01 may use any reachable solved static camera that observes the marker; the
    AP01 root itself does not need to see it. AP02 requires the marker in the
    selected combined graph component. AP03 requires compatible repeated moving
    support. Ground Truth is never consulted.
    """

    payload = resolved.payload
    details = _candidate(payload, marker_id)
    if not details or not bool(details.get("automatic_candidate", False)):
        return False
    if not details.get("static_cameras") or int(details.get("moving_frames") or 0) < 1:
        return False

    enabled = set(config.methods.enabled)
    if "ap01" in enabled:
        root = _root_candidate(payload, resolved.root_camera)
        reachable = {str(value) for value in root.get("reachable_cameras", [])}
        supporting = {str(value) for value in details.get("static_cameras", [])}
        if not (reachable & supporting):
            return False

    if "ap02" in enabled:
        reference = _ap02_reference_candidate(
            payload, int(resolved.ap02_reference_marker_id)
        )
        component_markers = {
            int(value)
            for value in reference.get("combined_graph_reachable_marker_ids", [])
        }
        if int(marker_id) not in component_markers:
            return False

    if "ap03" in enabled and not bool(details.get("ap03_compatible", False)):
        return False

    return True


def _install_selection_compatibility() -> None:
    from .. import observations

    original = observations.resolve_selections
    if getattr(original, "_rigcal_common_anchor_export_compatible", False):
        return

    def resolve_selections(config, observations_root):
        resolved = original(config, observations_root)
        if not config.evaluation.enabled:
            return resolved
        preferred = config.evaluation.anchor_marker_id
        if (
            config.evaluation.anchor_selection_mode != "auto"
            or not isinstance(preferred, int)
            or int(resolved.evaluation_anchor_marker_id or -1) == int(preferred)
        ):
            return resolved
        if not _preferred_export_compatible(config, resolved, int(preferred)):
            return resolved

        payload = copy.deepcopy(resolved.payload)
        anchor = payload.setdefault("evaluation_anchor", {})
        observation_candidates = {
            int(value) for value in anchor.get("observation_candidates", [])
        }
        automatic_candidates = {
            int(value)
            for value in anchor.get("automatic_observation_candidates", [])
        }
        observation_candidates.add(int(preferred))
        automatic_candidates.add(int(preferred))
        anchor.update(
            {
                "selected": int(preferred),
                "configured": int(preferred),
                "preferred_marker_id": int(preferred),
                "selection_mode": "preferred_with_auto_fallback",
                "fallback_used": False,
                "observation_candidates": sorted(observation_candidates),
                "automatic_observation_candidates": sorted(automatic_candidates),
                "reason": (
                    f"preferred common anchor {preferred} is reconstructable by "
                    "the enabled method exporters; AP01 root visibility is not required"
                ),
            }
        )
        payload.setdefault("automatic_recommendations", {})[
            "evaluation_anchor_marker_id"
        ] = int(preferred)
        category = payload.setdefault("category_marker_preference", {})
        category["evaluation_anchor"] = {
            "preferred": int(preferred),
            "selected": int(preferred),
            "fallback_used": False,
        }
        category["ground_truth_used"] = False

        final = replace(
            resolved,
            evaluation_anchor_marker_id=int(preferred),
            payload=payload,
        )
        root = Path(observations_root)
        text = json.dumps(payload, indent=2) + "\n"
        for name in ("SELECTION_CANDIDATES.json", "REFERENCE_SELECTIONS.json"):
            path = root / name
            if path.parent.is_dir():
                path.write_text(text, encoding="utf-8")
        try:
            observations.write_selection_candidates_csv(root, payload)
        except (KeyError, OSError, ValueError):
            pass
        return final

    resolve_selections._rigcal_common_anchor_export_compatible = True  # type: ignore[attr-defined]
    observations.resolve_selections = resolve_selections


def _install_anchor_export_authority() -> None:
    from ..anchor_export import exporter

    original = exporter._config_for_result
    if getattr(original, "_rigcal_common_anchor_authority", False):
        return

    def config_for_result(method_root: Path):
        config = original(method_root)
        if config is None:
            return None
        experiment_root = method_root.parents[2]
        selected = _read_json(
            experiment_root / "evaluations" / "SELECTED_COMMON_EVALUATION.json"
        )
        anchor_value = selected.get("anchor_marker_id")
        if anchor_value is None:
            selection = _read_json(
                experiment_root / "observations" / "SELECTION_CANDIDATES.json"
            )
            anchor_value = selection.get("evaluation_anchor", {}).get("selected")
        try:
            anchor = int(anchor_value)
        except (TypeError, ValueError):
            return config
        evaluation = config.evaluation.model_copy(update={"anchor_marker_id": anchor})
        return config.model_copy(update={"evaluation": evaluation}, deep=True)

    config_for_result._rigcal_common_anchor_authority = True  # type: ignore[attr-defined]
    exporter._config_for_result = config_for_result


def install_common_anchor_authority_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_selection_compatibility()
    _install_anchor_export_authority()
    _INSTALLED = True

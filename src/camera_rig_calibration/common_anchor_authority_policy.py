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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    """Match preflight common-anchor eligibility to the actual anchor exporters.

    AP01 can reconstruct the anchor from any reachable static camera that sees it;
    the root camera itself does not have to see the marker. AP02 requires the marker
    to be in the selected combined component. AP03 requires repeated moving support;
    final four-corner triangulation remains a post-COLMAP export check.
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
    from . import observations

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
                    "the enabled methods using their actual anchor-export geometry; "
                    "AP01 root visibility is not required"
                ),
            }
        )
        recommendations = payload.setdefault("automatic_recommendations", {})
        recommendations["evaluation_anchor_marker_id"] = int(preferred)
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
    from .anchor_export import exporter

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
        evaluation = config.evaluation.model_copy(
            update={"anchor_marker_id": anchor}
        )
        return config.model_copy(update={"evaluation": evaluation}, deep=True)

    config_for_result._rigcal_common_anchor_authority = True  # type: ignore[attr-defined]
    exporter._config_for_result = config_for_result


def _preferred_from_selection(selection: dict[str, Any]) -> int | None:
    evaluation = selection.get("evaluation_anchor", {})
    category = selection.get("category_marker_preference", {}).get(
        "evaluation_anchor", {}
    )
    value = evaluation.get("preferred_marker_id", category.get("preferred"))
    mode = str(evaluation.get("selection_mode", ""))
    if value is None or mode not in {
        "preferred_with_auto_fallback",
        "published_common_evaluation",
    }:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _primary_method_roots(experiment_root: Path) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for method in ("ap01", "ap02", "ap03"):
        method_root = experiment_root / "methods" / method
        if not method_root.is_dir():
            continue
        candidates = sorted(
            path.parent
            for path in method_root.glob("*/RESULT.json")
            if path.is_file()
        )
        if not candidates:
            continue
        selected = next((path for path in candidates if path.name == "baseline"), candidates[0])
        roots.append((method, selected))
    return roots


def repair_published_preferred_anchor(experiment_root: Path) -> dict[str, Any]:
    """Repair only derived common-anchor authority for runs affected by the old root-only gate.

    Native calibration estimates are never modified. The old preflight selection remains
    preserved in observations/SELECTION_CANDIDATES.json; a derived authoritative
    SELECTED_COMMON_EVALUATION.json records the repaired anchor and evidence.
    """

    from .anchor_export import adapters, exporter

    experiment_root = experiment_root.resolve()
    selection = _read_json(
        experiment_root / "observations" / "SELECTION_CANDIDATES.json"
    )
    preferred = _preferred_from_selection(selection)
    if preferred is None:
        return {"status": "not_applicable", "reason": "no category-preferred anchor request"}
    current = selection.get("evaluation_anchor", {}).get("selected")
    checks: list[dict[str, Any]] = []
    roots = _primary_method_roots(experiment_root)
    if not roots:
        return {"status": "unavailable", "reason": "no completed primary method outputs"}

    for method, root in roots:
        config = exporter._config_for_result(root)
        native = adapters.load_camera_poses(root)
        if config is None or not native:
            checks.append(
                {
                    "method": method,
                    "label": root.name,
                    "available": False,
                    "code": "METHOD_OUTPUT_UNAVAILABLE",
                }
            )
            continue
        resolution = adapters.resolve_method_anchor(
            root, config, method, preferred, native
        )
        checks.append(
            {
                "method": method,
                "label": root.name,
                "available": bool(resolution.available),
                "code": resolution.code,
                "warnings": list(resolution.warnings),
            }
        )

    if not checks or not all(item["available"] for item in checks):
        payload = {
            "schema_version": 1,
            "status": "preferred_anchor_not_exportable_by_every_method",
            "preferred_anchor_marker_id": preferred,
            "preflight_anchor_marker_id": current,
            "ground_truth_used": False,
            "method_anchor_checks": checks,
        }
        _write_json(
            experiment_root / "evaluations" / "COMMON_ANCHOR_REPAIR.json",
            payload,
        )
        return payload

    selected_path = (
        experiment_root / "evaluations" / "SELECTED_COMMON_EVALUATION.json"
    )
    existing = _read_json(selected_path)
    repaired = {
        **existing,
        "schema_version": max(int(existing.get("schema_version") or 0), 5),
        "anchor_marker_id": preferred,
        "success_for_every_method": True,
        "selection_mode": "repaired_category_preference_export_compatibility",
        "configured_preferred_anchor_marker_id": preferred,
        "superseded_preflight_anchor_marker_id": current,
        "ground_truth_used": False,
        "reason": (
            "The original preflight used an obsolete AP01 root-visibility gate. "
            "Completed method outputs prove that the configured preferred marker "
            "is exportable by every primary method; only derived anchor/report/"
            "visualization outputs are refreshed."
        ),
        "method_anchor_checks": checks,
    }
    _write_json(selected_path, repaired)
    audit = {
        "schema_version": 1,
        "status": "repaired",
        "preferred_anchor_marker_id": preferred,
        "preflight_anchor_marker_id": current,
        "published_anchor_marker_id": preferred,
        "method_rerun": False,
        "colmap_rerun": False,
        "native_method_outputs_modified": False,
        "ground_truth_used": False,
        "method_anchor_checks": checks,
    }
    _write_json(
        experiment_root / "evaluations" / "COMMON_ANCHOR_REPAIR.json",
        audit,
    )
    return audit


def install_common_anchor_authority_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_selection_compatibility()
    _install_anchor_export_authority()
    _INSTALLED = True

from __future__ import annotations

import json
from pathlib import Path
from typing import Any



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


def _category(experiment_root: Path) -> str:
    dataset = _read_json(experiment_root / "dataset.json")
    value = dataset.get("category") or dataset.get("dataset", {}).get("category")
    return str(value or "real_vehicle")


def _manual_or_explicit(selection: dict[str, Any]) -> bool:
    evaluation = selection.get("evaluation_anchor", {})
    mode = str(evaluation.get("selection_mode", "")).strip().lower()
    if bool(evaluation.get("warning_confirmed")):
        return True
    return any(token in mode for token in ("manual", "explicit", "forced"))


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
        selected = next(
            (path for path in candidates if path.name == "baseline"),
            candidates[0],
        )
        roots.append((method, selected))
    return roots


def _legacy_preference_evidence(
    experiment_root: Path,
    selection: dict[str, Any],
) -> tuple[int | None, list[dict[str, Any]]]:
    """Infer only category defaults lost by the old queue common-anchor freeze.

    This is intentionally conservative. A manual/explicit common-anchor request is
    never rewritten. For an old automatic real-vehicle publication, marker 0 is
    inferred only when completed method configs still retain independent marker-0
    category-default evidence (AP02 auto reference and/or AP03 Single marker 0).
    Simulation analogously uses marker 14.
    """

    if _manual_or_explicit(selection):
        return None, []
    category = _category(experiment_root)
    preferred = 14 if category == "simulation" else 0
    evidence: list[dict[str, Any]] = []

    from .anchor_export import exporter

    for method, root in _primary_method_roots(experiment_root):
        config = exporter._config_for_result(root)
        if config is None:
            continue
        if method == "ap02":
            ref_mode = str(config.methods.ap02.reference_marker_selection_mode)
            ref = config.methods.ap02.reference_marker_id
            matched = ref_mode == "auto" and ref == preferred
            evidence.append(
                {
                    "method": method,
                    "kind": "auto_reference_marker",
                    "value": ref,
                    "mode": ref_mode,
                    "matches_category_preference": matched,
                }
            )
        elif method == "ap03":
            single = config.methods.ap03.single.scale_marker_id
            matched = single == preferred
            evidence.append(
                {
                    "method": method,
                    "kind": "single_scale_marker",
                    "value": single,
                    "matches_category_preference": matched,
                }
            )

    matched = [item for item in evidence if item["matches_category_preference"]]
    # Require two independent persisted settings when both AP02 and AP03 exist.
    represented = {item["method"] for item in evidence}
    required = 2 if {"ap02", "ap03"}.issubset(represented) else 1
    if len(matched) < required:
        return None, evidence
    return preferred, evidence


def repair_legacy_preferred_anchor(experiment_root: Path) -> dict[str, Any]:
    """Recover a category-preferred anchor lost by the historical queue gate.

    Native method estimates and COLMAP outputs are read-only. The preferred marker
    becomes authoritative only if every completed primary method can express its
    already-computed camera solution in that marker frame. Otherwise the existing
    published common anchor remains authoritative and the failure evidence is
    written for audit.
    """

    from .anchor_export import adapters, exporter

    experiment_root = experiment_root.resolve()
    selection = _read_json(
        experiment_root / "observations" / "SELECTION_CANDIDATES.json"
    )
    preferred, preference_evidence = _legacy_preference_evidence(
        experiment_root, selection
    )
    current_value = selection.get("evaluation_anchor", {}).get("selected")
    try:
        current = int(current_value)
    except (TypeError, ValueError):
        current = None

    if preferred is None:
        payload = {
            "schema_version": 1,
            "status": "not_applicable",
            "reason": (
                "No safe legacy category-preference inference is available; "
                "manual/explicit selections are never rewritten."
            ),
            "preflight_anchor_marker_id": current,
            "preference_evidence": preference_evidence,
            "ground_truth_used": False,
        }
        _write_json(
            experiment_root / "evaluations" / "COMMON_ANCHOR_REPAIR.json",
            payload,
        )
        return payload

    if current == preferred:
        payload = {
            "schema_version": 1,
            "status": "already_preferred",
            "preferred_anchor_marker_id": preferred,
            "preflight_anchor_marker_id": current,
            "preference_evidence": preference_evidence,
            "method_rerun": False,
            "colmap_rerun": False,
            "native_method_outputs_modified": False,
            "ground_truth_used": False,
        }
        _write_json(
            experiment_root / "evaluations" / "COMMON_ANCHOR_REPAIR.json",
            payload,
        )
        return payload

    checks: list[dict[str, Any]] = []
    roots = _primary_method_roots(experiment_root)
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
                "diagnostics": resolution.diagnostics,
            }
        )

    if not checks or not all(item["available"] for item in checks):
        payload = {
            "schema_version": 1,
            "status": "preferred_anchor_not_exportable_by_every_method",
            "preferred_anchor_marker_id": preferred,
            "preflight_anchor_marker_id": current,
            "preference_evidence": preference_evidence,
            "method_anchor_checks": checks,
            "method_rerun": False,
            "colmap_rerun": False,
            "native_method_outputs_modified": False,
            "ground_truth_used": False,
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
    authoritative = {
        **existing,
        "schema_version": max(int(existing.get("schema_version") or 0), 5),
        "anchor_marker_id": preferred,
        "success_for_every_method": True,
        "selection_mode": "repaired_legacy_category_preference",
        "configured_preferred_anchor_marker_id": preferred,
        "superseded_preflight_anchor_marker_id": current,
        "ground_truth_used": False,
        "reason": (
            "A historical automatic queue freeze lost the category-preferred "
            "common anchor. Persisted method settings retain the preference and "
            "every completed primary method can re-express its native solution "
            "in that marker frame. Only derived outputs are regenerated."
        ),
        "preference_evidence": preference_evidence,
        "method_anchor_checks": checks,
    }
    _write_json(selected_path, authoritative)

    payload = {
        "schema_version": 1,
        "status": "repaired_legacy_category_preference",
        "preferred_anchor_marker_id": preferred,
        "preflight_anchor_marker_id": current,
        "published_anchor_marker_id": preferred,
        "preference_evidence": preference_evidence,
        "method_anchor_checks": checks,
        "method_rerun": False,
        "colmap_rerun": False,
        "native_method_outputs_modified": False,
        "ground_truth_used": False,
    }
    _write_json(
        experiment_root / "evaluations" / "COMMON_ANCHOR_REPAIR.json",
        payload,
    )
    return payload

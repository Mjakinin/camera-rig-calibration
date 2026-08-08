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


def _result_candidates(method_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    if not method_root.is_dir():
        return []
    for pattern in (
        "*/RESULT.json",
        "*/provenance/resolved_config.yaml",
        "*/camera_extrinsics.csv",
        "*/camera_extrinsics_anchor.json",
    ):
        for artifact in method_root.glob(pattern):
            if artifact.is_file():
                root = artifact.parent
                if artifact.parent.name == "provenance":
                    root = artifact.parent.parent
                candidates.add(root)
    return sorted(candidates)


def _choose_result_root(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    preferred = next((path for path in candidates if path.name == "baseline"), None)
    return preferred or sorted(candidates, key=lambda path: path.name)[0]


def _primary_method_roots(experiment_root: Path) -> list[tuple[str, Path]]:
    """Return the actual public primary result roots for the three method families.

    Older publications used an AP03 container under methods/ap03 while current
    publications expose AP03 Single/Multi as derived result families. Discovery
    therefore follows published artifacts rather than assuming one historical
    directory layout. AP03 Multi is the public primary AP03 estimate.
    """

    roots: list[tuple[str, Path]] = []
    for method in ("ap01", "ap02"):
        selected = _choose_result_root(
            _result_candidates(experiment_root / "methods" / method)
        )
        if selected is not None:
            roots.append((method, selected))

    ap03_multi = _choose_result_root(
        _result_candidates(experiment_root / "methods" / "ap03_multi")
    )
    if ap03_multi is not None:
        roots.append(("ap03_multi", ap03_multi))
    else:
        ap03_container = _choose_result_root(
            _result_candidates(experiment_root / "methods" / "ap03")
        )
        if ap03_container is not None:
            roots.append(("ap03", ap03_container))
    return roots


def _legacy_preference_evidence(
    experiment_root: Path,
    selection: dict[str, Any],
) -> tuple[int | None, list[dict[str, Any]]]:
    """Infer only category defaults lost by the old queue common-anchor freeze."""

    if _manual_or_explicit(selection):
        return None, []
    category = _category(experiment_root)
    preferred = 14 if category == "simulation" else 0
    evidence: list[dict[str, Any]] = []

    from .anchor_export import exporter

    for method, root in _primary_method_roots(experiment_root):
        config = exporter._config_for_result(root)
        if config is None:
            evidence.append(
                {
                    "method": method,
                    "kind": "resolved_config",
                    "value": None,
                    "matches_category_preference": False,
                    "result_root": str(root),
                }
            )
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
                    "result_root": str(root),
                }
            )
        elif method in {"ap03", "ap03_multi"}:
            single = config.methods.ap03.single.scale_marker_id
            matched = single == preferred
            evidence.append(
                {
                    "method": "ap03",
                    "published_method": method,
                    "kind": "single_scale_marker",
                    "value": single,
                    "matches_category_preference": matched,
                    "result_root": str(root),
                }
            )

    matched = [item for item in evidence if item["matches_category_preference"]]
    represented = {
        "ap03" if item["method"] in {"ap03", "ap03_multi"} else item["method"]
        for item in evidence
    }
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
                    "result_root": str(root),
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
                "result_root": str(root),
            }
        )

    expected_families = {"ap01", "ap02", "ap03"}
    checked_families = {
        "ap03" if item["method"] in {"ap03", "ap03_multi"} else item["method"]
        for item in checks
    }
    all_primary_present = expected_families.issubset(checked_families)
    if (
        not checks
        or not all_primary_present
        or not all(item["available"] for item in checks)
    ):
        payload = {
            "schema_version": 1,
            "status": "preferred_anchor_not_exportable_by_every_method",
            "preferred_anchor_marker_id": preferred,
            "preflight_anchor_marker_id": current,
            "preference_evidence": preference_evidence,
            "method_anchor_checks": checks,
            "expected_method_families": sorted(expected_families),
            "checked_method_families": sorted(checked_families),
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

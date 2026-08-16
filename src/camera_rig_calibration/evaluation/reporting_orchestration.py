"""Focused scientific reporting responsibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..anchor_export import ensure_experiment_anchor_exports
from ..anchor_export.aggregate import (
    build_experiment_anchor_aggregate,
    experiment_anchor_aggregate_text,
)
from ..visualization.scene import ensure_visualization_artifacts
from .ap03_derived import ensure_ap03_derived_results
from .simulation_ground_truth import resolve_simulation_ground_truth
from .reporting_core import _now, _read_json, _write_json, _write_text
from .reporting_method import refresh_method_reports
from .reporting_real import _real_results_text
from .reporting_simulation import (
    _refresh_factor_reports,
    _simulation_results,
    _write_route2_baseline_comparison,
)
from .reporting_bindings import current_reporting_bindings


def write_scientific_experiment_reports(
    experiment_root: Path,
    *,
    dataset_root: Path,
    category: str,
) -> dict[str, Any]:
    """Write the canonical human and machine result front doors."""
    hooks = current_reporting_bindings()
    _read_json = hooks.read_json
    _write_json = hooks.write_json
    _real_results_text = hooks.real_results_text
    _simulation_results = hooks.simulation_results
    refresh_method_reports = hooks.refresh_method_reports
    ensure_ap03_derived_results = hooks.ensure_ap03_derived_results
    ensure_visualization_artifacts = hooks.ensure_visualization_artifacts
    ensure_ap03_derived_results(experiment_root)
    if category == "simulation":
        resolve_simulation_ground_truth(dataset_root, backfilled=True)
    ensure_experiment_anchor_exports(experiment_root)
    method_payloads = refresh_method_reports(experiment_root)
    if category == "simulation":
        text, payload = _simulation_results(
            experiment_root, dataset_root, method_payloads
        )
    else:
        text, payload = _real_results_text(
            experiment_root, method_payloads, dataset_root
        )
    evaluation_by_method: dict[tuple[str, str], str] = {}
    evaluation_metrics_by_method: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    if category == "simulation":
        summaries = payload.get(
            "anchor_camera_ground_truth", {}
        ).get("summaries", [])
        pair_summaries = payload.get(
            "primary_camera_pairwise", {}
        ).get("summaries", [])
        evaluation_by_method = {
            (str(item.get("method")), str(item.get("label"))): str(
                item.get("evaluation_status", "evaluation_unavailable")
            )
            for item in summaries
            if isinstance(item, dict)
        }
        anchor_by_key = {
            (str(item.get("method")), str(item.get("label"))): item
            for item in summaries
            if isinstance(item, dict)
        }
        for item in pair_summaries:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("method")),
                str(item.get("label")),
            )
            evaluation_metrics_by_method[key] = {
                "pairwise_gt": item,
                "anchor_camera_gt": anchor_by_key.get(key, {}),
            }
    else:
        marker_available = bool(payload.get("marker_consistency_path"))
        evaluation_by_method = {
            (str(item.get("method")), str(item.get("label"))): (
                "available"
                if marker_available
                and bool(item.get("anchor_export_available"))
                else "unavailable"
            )
            for item in method_payloads
        }
    statuses_changed = False
    for result_path in sorted(
        (experiment_root / "methods").glob("*/*/RESULT.json")
    ):
        method_result = _read_json(result_path)
        key = (
            str(method_result.get("method") or result_path.parents[1].name),
            str(method_result.get("label") or result_path.parent.name),
        )
        evaluation_status = evaluation_by_method.get(key, "unavailable")
        current_metrics = (
            dict(method_result.get("metrics", {}))
            if isinstance(method_result.get("metrics"), dict)
            else {}
        )
        evaluation_metrics = evaluation_metrics_by_method.get(key)
        metrics_changed = (
            evaluation_metrics is not None
            and current_metrics.get("evaluation") != evaluation_metrics
        )
        if metrics_changed:
            current_metrics["evaluation"] = evaluation_metrics
            method_result["metrics"] = current_metrics
        if (
            method_result.get("evaluation_status") != evaluation_status
            or metrics_changed
        ):
            method_result["evaluation_status"] = evaluation_status
            _write_json(result_path, method_result)
            statuses_changed = True
    if statuses_changed:
        method_payloads = refresh_method_reports(experiment_root)
        if category == "simulation":
            text, payload = _simulation_results(
                experiment_root, dataset_root, method_payloads
            )
        else:
            text, payload = _real_results_text(
                experiment_root, method_payloads, dataset_root
            )
    visualization = ensure_visualization_artifacts(experiment_root)
    method_payloads = refresh_method_reports(experiment_root)
    if category == "simulation":
        text, payload = _simulation_results(
            experiment_root, dataset_root, method_payloads
        )
    else:
        text, payload = _real_results_text(
            experiment_root, method_payloads, dataset_root
        )

    # The per-method anchor exports above are authoritative.  Rebuild the
    # experiment-level compatibility/front-door files on every report pass so
    # an older AP02/AP03 subset can never remain visible after new variants are
    # published (notably AP03 Single and later AP02 settings).
    anchor_aggregate = build_experiment_anchor_aggregate(experiment_root)
    payload["common_anchor_6dof_export"] = {
        "contract": anchor_aggregate.get("contract"),
        "anchor_marker_id": anchor_aggregate.get("anchor_marker_id"),
        "variant_count": len(anchor_aggregate.get("variants", [])),
        "camera_row_count": len(anchor_aggregate.get("rows", [])),
        "json": "CAMERA_EXTRINSICS_COMMON_ANCHOR.json",
        "csv": "CAMERA_EXTRINSICS_COMMON_ANCHOR.csv",
        "yaml": "CAMERA_EXTRINSICS_COMMON_ANCHOR.yaml",
    }
    text = (
        text.rstrip()
        + "\n\n"
        + experiment_anchor_aggregate_text(anchor_aggregate).rstrip()
        + "\n"
    )

    baseline_comparison = (
        _write_route2_baseline_comparison(experiment_root, payload)
        if category == "simulation"
        else None
    )
    if baseline_comparison is not None:
        payload["baseline_comparison"] = baseline_comparison
        text = (
            text.rstrip()
            + "\n\n"
            + (
                experiment_root / "BASELINE_COMPARISON.txt"
            ).read_text(encoding="utf-8")
        )
    existing_results = _read_json(experiment_root / "RESULTS.json")
    generated_at = existing_results.get("generated_at") or _now()
    payload.update(
        {
            "schema_version": 5,
            "layout_version": 2,
            "generated_at": generated_at,
            "human_report": "RESULTS.txt",
            "visualization": visualization,
        }
    )
    text = (
        text.rstrip()
        + "\n\nRVIZ VISUALIZATION\n"
        + "-" * 72
        + "\n"
        + f"Status: {visualization.get('status', 'unavailable')}\n"
        + "Manifest: visualization/visualization_manifest.json\n"
        + (
            "Open from rigcal View results; each window uses an isolated "
            "ROS_DOMAIN_ID.\n"
            if visualization.get("available")
            else f"Reason: {visualization.get('reason', '-')}\n"
        )
    )
    _write_text(experiment_root / "RESULTS.txt", text)
    _write_json(experiment_root / "RESULTS.json", payload)
    for obsolete in ("SUMMARY.txt", "COMPARISON.txt"):
        (experiment_root / obsolete).unlink(missing_ok=True)
    if category == "simulation":
        _refresh_factor_reports(experiment_root, payload)
    return payload


__all__ = [
    "write_scientific_experiment_reports",
]

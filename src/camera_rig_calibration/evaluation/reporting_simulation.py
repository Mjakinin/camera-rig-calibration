"""Focused scientific reporting responsibility."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..anchor_export import ensure_experiment_anchor_exports
from ..anchor_export.geometry import rotation_to_quaternion
from ..visualization.scene import ensure_visualization_artifacts
from .ap03_derived import ensure_ap03_derived_results
from .simulation_ground_truth import (
    ensure_simulation_ground_truth,
    resolve_simulation_ground_truth,
)

from ..methods.common.geometry import (
    R_to_rpy_deg,
    R_to_rvec,
    invT,
    make_T,
    rot_error_deg,
    rpy_to_R,
    rvec_to_R,
)

from .reporting_configuration import (
    _baseline_contract,
)
from .reporting_core import (
    _fmt,
    _maximum,
    _read_json,
    _text_table,
    _write_csv,
    _write_json,
    _write_text,
)
from .reporting_diagnostics import (
    _scale_comparison_rows,
)
from .reporting_simulation_geometry import (
    _anchor_camera_gt_rows,
    _anchor_pose_records,
    _ap02_marker_map,
    _camera_map_rows,
    _camera_map_text,
    _ground_truth_anchor_records,
    _simulation_gt_maps,
    _simulation_pairwise,
    _simulation_primary_text,
    _summary,
)
from .reporting_bindings import current_reporting_bindings

def _simulation_results(
    experiment_root: Path,
    dataset_root: Path,
    method_payloads: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    hooks = current_reporting_bindings()
    _anchor_camera_gt_rows = hooks.anchor_camera_gt_rows
    _baseline_contract = hooks.baseline_contract
    _read_json = hooks.read_json
    _write_json = hooks.write_json
    gt_payload = ensure_simulation_ground_truth(
        dataset_root, backfilled=True
    )
    if gt_payload.get("status") != "available":
        text = (
            "SIMULATION CALIBRATION RESULTS\n"
            "==============================\n\n"
            f"Ground truth unavailable: {gt_payload.get('reason', 'unknown')}\n"
        )
        return text, {
            "category": "simulation",
            "experiment": experiment_root.name,
            "status": "evaluation_unavailable",
            "ground_truth": gt_payload,
            "methods": method_payloads,
        }
    gt_cameras, gt_markers = _simulation_gt_maps(gt_payload)
    pair_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    marker_results: list[dict[str, Any]] = []
    marker_texts: list[str] = []
    selection = _read_json(
        dataset_root / "observations" / "SELECTION_CANDIDATES.json"
    )
    anchor_value = selection.get("evaluation_anchor", {}).get("selected")
    anchor_marker_id = int(anchor_value) if anchor_value is not None else None
    gt_anchor_cameras = (
        _ground_truth_anchor_records(
            anchor_marker_id=anchor_marker_id,
            gt_cameras=gt_cameras,
            gt_markers=gt_markers,
        )
        if anchor_marker_id is not None
        else {}
    )
    expected_cameras = sorted(gt_cameras)
    anchor_gt_rows: list[dict[str, Any]] = []
    anchor_gt_summaries: list[dict[str, Any]] = []
    for payload in method_payloads:
        method = str(payload["method"])
        label = str(payload["label"])
        root = experiment_root / "methods" / method / label
        anchor_payload = _read_json(
            root / "camera_extrinsics_anchor.json"
        )
        estimated = _anchor_pose_records(anchor_payload)
        complete_camera_set = set(estimated) == set(expected_cameras)
        rows = (
            _simulation_pairwise(
                method, label, estimated, gt_anchor_cameras
            )
            if complete_camera_set
            and set(gt_anchor_cameras) == set(expected_cameras)
            else []
        )
        pair_rows.extend(rows)
        evaluation_status = (
            "available"
            if len(rows)
            == len(expected_cameras) * (len(expected_cameras) - 1) // 2
            and bool(rows)
            else "evaluation_unavailable"
        )
        summaries.append(
            {
                "method": method,
                "label": label,
                "evaluation_status": evaluation_status,
                "expected_camera_count": len(expected_cameras),
                "evaluated_camera_count": len(estimated),
                "missing_cameras": sorted(
                    set(expected_cameras) - set(estimated)
                ),
                "reason": (
                    None
                    if evaluation_status == "available"
                    else (
                        "The direct anchor-relative estimate does not contain "
                        "the exact Ground-Truth camera set; no pair subset is "
                        "published as a complete evaluation."
                    )
                ),
                **_summary(rows),
            }
        )
        if anchor_marker_id is not None:
            direct_rows = _anchor_camera_gt_rows(
                method,
                label,
                anchor_payload,
                anchor_marker_id=anchor_marker_id,
                gt_cameras=gt_cameras,
                gt_markers=gt_markers,
            )
            anchor_gt_rows.extend(direct_rows)
            anchor_gt_summaries.append(
                {
                    "method": method,
                    "label": label,
                    "evaluation_status": (
                        "available"
                        if len(direct_rows) == len(expected_cameras)
                        and complete_camera_set
                        else "evaluation_unavailable"
                    ),
                    **_summary(direct_rows),
                }
            )
        try:
            map_rows.extend(
                _camera_map_rows(
                    method, label, estimated, gt_anchor_cameras
                )
            )
        except RuntimeError:
            pass
        if method == "ap02":
            marker_result, marker_text = _ap02_marker_map(
                root, gt_cameras, gt_markers
            )
            marker_results.append(marker_result)
            marker_texts.append(marker_text)
    evaluation_root = experiment_root / "evaluations"
    _write_csv(evaluation_root / "camera_pairwise_gt.csv", pair_rows)
    _write_json(
        evaluation_root / "camera_pairwise_gt.json",
        {"summaries": summaries, "rows": pair_rows},
    )
    _write_csv(evaluation_root / "camera_map_gt.csv", map_rows)
    _write_json(
        evaluation_root / "camera_map_gt.json",
        {"rows": map_rows},
    )
    _write_json(
        evaluation_root / "ap02_marker_map_gt.json",
        {"variants": marker_results},
    )
    _write_csv(
        evaluation_root / "anchor_camera_gt.csv", anchor_gt_rows
    )
    _write_json(
        evaluation_root / "anchor_camera_gt.json",
        {
            "anchor_marker_id": anchor_marker_id,
            "evaluation": (
                "direct_anchor_relative_posthoc_gt_no_fit_no_scale"
            ),
            "summaries": anchor_gt_summaries,
            "rows": anchor_gt_rows,
        },
    )
    map_text = _camera_map_text(experiment_root.name, map_rows)
    (experiment_root / "SECONDARY_CAMERA_MAP_RESULTS.txt").write_text(
        map_text, encoding="utf-8"
    )
    (experiment_root / "SECONDARY_AP02_MARKER_MAP_RESULTS.txt").write_text(
        "\n".join(marker_texts)
        if marker_texts
        else (
            "SECONDARY AP02 MARKER-MAP RESULTS\n"
            "=================================\n\n"
            "Unavailable: no successful AP02 result exists.\n"
        ),
        encoding="utf-8",
    )
    dataset = _read_json(dataset_root / "dataset.json")
    parameters = dataset.get("simulation_parameters", {}) or {}
    text = _simulation_primary_text(
        experiment_root.name,
        parameters,
        summaries,
        pair_rows,
        method_payloads,
        _read_json(
            dataset_root
            / "observations"
            / "SELECTION_CANDIDATES.json"
        ).get("evaluation_anchor", {}),
    )
    text += (
        "\n\nDIRECT COMMON-ANCHOR CAMERA POSES VS GROUND TRUTH\n"
        + "-" * 138
        + "\n"
        + (
            _text_table(
                [
                    "Method",
                    "Variant",
                    "Cameras",
                    "mean translation [cm]",
                    "max translation [cm]",
                    "mean rotation [deg]",
                    "max rotation [deg]",
                ],
                [
                    [
                        row["method"],
                        row["label"],
                        row["count"],
                        _fmt(row["mean_translation_error_cm"]),
                        _fmt(row["max_translation_error_cm"]),
                        _fmt(row["mean_rotation_error_deg"]),
                        _fmt(row["max_rotation_error_deg"]),
                    ]
                    for row in anchor_gt_summaries
                ],
            )
            if any(row["count"] for row in anchor_gt_summaries)
            else (
                "Unavailable: the frozen anchor or a method-specific "
                "anchor export is missing."
            )
        )
        + "\nDetailed values: evaluations/anchor_camera_gt.csv\n"
    )
    return text, {
        "category": "simulation",
        "experiment": experiment_root.name,
        "status": "available",
        "simulation_parameters": parameters,
        "storage": dataset.get("storage", {}),
        "ground_truth": {
            key: gt_payload.get(key)
            for key in (
                "snapshot_origin",
                "world_snapshot",
                "world_sha256",
                "camera_transform_convention",
                "marker_transform_convention",
            )
        },
        "methods": method_payloads,
        "baseline_contract": _baseline_contract(
            category="simulation",
            method_payloads=method_payloads,
            evaluation_anchor=selection.get("evaluation_anchor", {}),
        ),
        "scale_comparison": _scale_comparison_rows(method_payloads),
        "primary_camera_pairwise": {
            "summaries": summaries,
            "rows": pair_rows,
        },
        "anchor_camera_ground_truth": {
            "anchor_marker_id": anchor_marker_id,
            "summaries": anchor_gt_summaries,
            "rows": anchor_gt_rows,
            "path": "evaluations/anchor_camera_gt.csv",
        },
        "secondary_camera_map": {
            "rows": map_rows,
            "path": "SECONDARY_CAMERA_MAP_RESULTS.txt",
        },
        "secondary_ap02_marker_map": {
            "variants": marker_results,
            "path": "SECONDARY_AP02_MARKER_MAP_RESULTS.txt",
        },
    }


def _factor_report(factor_root: Path, factor: str) -> None:
    simulation_root = factor_root.parent
    experiment_payloads: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(factor_root.glob("*/RESULTS.json")):
        payload = _read_json(path)
        if payload.get("category") == "simulation":
            experiment_payloads.append((path.parent.name, payload))
    for path in sorted(
        (simulation_root / "baseline").glob("*/RESULTS.json")
    ):
        payload = _read_json(path)
        if payload.get("category") == "simulation":
            experiment_payloads.append(
                (f"baseline/{path.parent.name}", payload)
            )
    if not any(not name.startswith("baseline/") for name, _ in experiment_payloads):
        return
    rows: list[dict[str, Any]] = []
    for name, payload in experiment_payloads:
        storage = payload.get("storage", {})
        if not name.startswith("baseline/") and storage.get("factor") != factor:
            continue
        value = (
            f"baseline ({storage.get('value', path_name(name))})"
            if name.startswith("baseline/")
            else str(storage.get("value") or name)
        )
        for row in payload.get("primary_camera_pairwise", {}).get("rows", []):
            rows.append({"factor_value": value, "experiment": name, **row})
    if not rows:
        return
    lines = [
        f"SIMULATION {factor.upper()} COMPARISON — CAMERA-TO-CAMERA VS GT",
        "=" * 142,
    ]
    for value, method, label in sorted(
        {
            (row["factor_value"], row["method"], row["label"])
            for row in rows
        }
    ):
        selected = [
            row
            for row in rows
            if (
                row["factor_value"],
                row["method"],
                row["label"],
            )
            == (value, method, label)
        ]
        summary = _summary(selected)
        lines.extend(
            [
                "",
                f"{value} — {method}/{label}",
                "-" * 142,
                (
                    f"Summary: mean {_fmt(summary['mean_translation_error_cm'])} "
                    f"cm / {_fmt(summary['mean_rotation_error_deg'])} deg; "
                    f"max {_fmt(summary['max_translation_error_cm'])} cm / "
                    f"{_fmt(summary['max_rotation_error_deg'])} deg"
                ),
                _text_table(
                    [
                        "Pair",
                        "t err [cm]",
                        "r err [deg]",
                        "GT base [m]",
                        "Est base [m]",
                        "base err [cm]",
                        "dir err [deg]",
                    ],
                    [
                        [
                            row["pair"],
                            _fmt(row["translation_error_cm"]),
                            _fmt(row["rotation_error_deg"]),
                            _fmt(row["gt_baseline_m"]),
                            _fmt(row["estimated_baseline_m"]),
                            _fmt(row["baseline_error_cm"]),
                            _fmt(row["direction_error_deg"]),
                        ]
                        for row in selected
                    ],
                ),
            ]
        )
    lines.append("")
    (factor_root / "RESULTS.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    _write_csv(factor_root / "RESULTS.csv", rows)
    _write_json(
        factor_root / "RESULTS.json",
        {
            "schema_version": 5,
            "layout_version": 2,
            "kind": "simulation_factor_comparison",
            "factor": factor,
            "rows": rows,
        },
    )


def path_name(value: str) -> str:
    return Path(value).name


def _refresh_factor_reports(experiment_root: Path, payload: dict[str, Any]) -> None:
    simulation_root = next(
        (
            parent
            for parent in experiment_root.parents
            if parent.name == "simulation"
        ),
        None,
    )
    if simulation_root is None:
        return
    factor = str(payload.get("storage", {}).get("factor", ""))
    if factor in {
        "route",
        "density",
        "resolution",
        "fov",
        "lighting",
        "motion_blur",
    }:
        _factor_report(simulation_root / factor, factor)
    elif factor == "baseline":
        for candidate in (
            "route",
            "density",
            "resolution",
            "fov",
            "lighting",
            "motion_blur",
        ):
            if (simulation_root / candidate).is_dir():
                _factor_report(simulation_root / candidate, candidate)


def _write_route2_baseline_comparison(
    experiment_root: Path,
    current: dict[str, Any],
) -> dict[str, Any] | None:
    """Compare the controlled CPU repair with the immutable Route-2 run."""

    if experiment_root.name != "route2_cpu_ref14_50x50":
        return None
    previous_root = experiment_root.parent / "route2"
    previous = _read_json(previous_root / "RESULTS.json")
    if previous.get("category") != "simulation":
        return None

    rows: list[dict[str, Any]] = []
    for experiment_name, payload in (
        ("route2", previous),
        (experiment_root.name, current),
    ):
        methods = {
            (str(item.get("method")), str(item.get("label"))): item
            for item in payload.get("methods", [])
            if isinstance(item, dict)
        }
        summaries = payload.get("primary_camera_pairwise", {}).get(
            "summaries", []
        )
        anchor_summaries = {
            (str(item.get("method")), str(item.get("label"))): item
            for item in payload.get(
                "anchor_camera_ground_truth", {}
            ).get("summaries", [])
            if isinstance(item, dict)
        }
        pair_rows = payload.get("primary_camera_pairwise", {}).get(
            "rows", []
        )
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            key = (
                str(summary.get("method")),
                str(summary.get("label")),
            )
            method = methods.get(key, {})
            anchor = anchor_summaries.get(key, {})
            edge5 = [
                item
                for item in pair_rows
                if isinstance(item, dict)
                and str(item.get("method")) == key[0]
                and str(item.get("label")) == key[1]
                and "cam_edge_5" in str(item.get("pair", ""))
            ]
            registration = method.get("metrics", {}).get(
                "ap03_registration", {}
            )
            rows.append(
                {
                    "experiment": experiment_name,
                    "method": key[0],
                    "label": key[1],
                    "runtime_seconds": method.get("runtime_seconds"),
                    "execution_status": method.get("execution_status"),
                    "solver_status": method.get("solver_status"),
                    "quality_status": method.get("quality_status"),
                    "pair_count": summary.get("count"),
                    "anchor_camera_count": anchor.get("count"),
                    "mean_pair_translation_error_cm": summary.get(
                        "mean_translation_error_cm"
                    ),
                    "mean_pair_rotation_error_deg": summary.get(
                        "mean_rotation_error_deg"
                    ),
                    "maximum_pair_translation_error_cm": summary.get(
                        "max_translation_error_cm"
                    ),
                    "maximum_pair_rotation_error_deg": summary.get(
                        "max_rotation_error_deg"
                    ),
                    "cam_edge_5_pair_count": len(edge5),
                    "cam_edge_5_maximum_translation_error_cm": _maximum(
                        item.get("translation_error_cm") for item in edge5
                    ),
                    "cam_edge_5_maximum_rotation_error_deg": _maximum(
                        item.get("rotation_error_deg") for item in edge5
                    ),
                    "registered_static_cameras": registration.get(
                        "registered_static_cameras"
                    ),
                    "registered_moving_frames": registration.get(
                        "registered_moving_frames"
                    ),
                    "sparse_points": registration.get("sparse_points"),
                    "configuration": method.get("config_summary", {}),
                }
            )
    if not rows:
        return None
    comparison_payload = {
        "schema_version": 5,
        "comparison": (
            "same_published_route2_input_no_alignment_no_best_fit"
        ),
        "old_experiment": "route2",
        "new_experiment": experiment_root.name,
        "method_rerun_of_old_experiment": False,
        "rows": rows,
    }
    _write_json(
        experiment_root / "BASELINE_COMPARISON.json",
        comparison_payload,
    )
    _write_csv(experiment_root / "BASELINE_COMPARISON.csv", rows)
    text = "\n".join(
        [
            "ROUTE-2 BASELINE REPAIR COMPARISON",
            "=" * 138,
            "",
            "Input: the same published Route-2 images and observations.",
            "Evaluation: direct common-anchor and camera-pair Ground Truth; "
            "no global alignment and no best-fit.",
            "",
            _text_table(
                [
                    "Experiment",
                    "Method",
                    "Variant",
                    "Runtime [s]",
                    "Quality",
                    "Pairs",
                    "Anchors",
                    "mean t [cm]",
                    "mean r [deg]",
                    "cam_edge_5 max t [cm]",
                    "cam_edge_5 max r [deg]",
                    "Static reg.",
                    "Moving reg.",
                    "Sparse points",
                ],
                [
                    [
                        row["experiment"],
                        row["method"],
                        row["label"],
                        _fmt(row["runtime_seconds"], 1),
                        row["quality_status"],
                        row["pair_count"],
                        row["anchor_camera_count"],
                        _fmt(row["mean_pair_translation_error_cm"]),
                        _fmt(row["mean_pair_rotation_error_deg"]),
                        _fmt(
                            row[
                                "cam_edge_5_maximum_translation_error_cm"
                            ]
                        ),
                        _fmt(
                            row[
                                "cam_edge_5_maximum_rotation_error_deg"
                            ]
                        ),
                        row["registered_static_cameras"],
                        row["registered_moving_frames"],
                        row["sparse_points"],
                    ]
                    for row in rows
                ],
            ),
            "",
        ]
    )
    _write_text(experiment_root / "BASELINE_COMPARISON.txt", text)
    return {
        "status": "available",
        "text": "BASELINE_COMPARISON.txt",
        "json": "BASELINE_COMPARISON.json",
        "csv": "BASELINE_COMPARISON.csv",
        "rows": len(rows),
    }



__all__ = [
    '_simulation_results',
    '_factor_report',
    'path_name',
    '_refresh_factor_reports',
    '_write_route2_baseline_comparison',
]

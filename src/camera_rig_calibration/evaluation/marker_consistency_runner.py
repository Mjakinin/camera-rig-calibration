"""CLI orchestration for marker-consistency evaluation.

The numerical implementation and method-specific pose loaders are injected
explicitly.  This lets the native-metric real-data evaluation reuse geometry
without replacing functions on another module at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from . import marker_consistency_core as core
from .marker_consistency_reporting import report as default_report


PoseLoader = Callable[..., tuple[Any, Any, Path, dict[str, Any]]]
Evaluator = Callable[
    ..., tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]
]
ReportWriter = Callable[..., None]


SUMMARY_FIELDS = [
    "method",
    "status",
    "original_method_status",
    "original_method_success",
    "available_static_camera_count",
    "available_static_cameras",
    "registered_moving_frames",
    "pose_frame_count",
    "pose_frame_ids",
    "marker_frame_count",
    "marker_frame_ids",
    "candidate_frame_count",
    "candidate_frame_ids",
    "inlier_frame_count",
    "inlier_frame_ids",
    "rejected_frame_ids",
    "static_pose_file",
    "anchor_marker_id",
    "anchor_expected_size_cm",
    "anchor_raw_reconstructed_size_units",
    "anchor_scale_m_per_unit",
    "reconstructed_markers_total",
    "evaluated_non_anchor_markers",
    "markers_with_moving_to_static_validation",
    "marker_length_rmse_cm",
    "marker_length_rmse_percent",
    "median_absolute_size_error_cm",
    "median_relative_size_error_percent",
    "moving_fit_reprojection_rmse_px",
    "moving_fit_reprojection_median_px",
    "moving_to_static_reprojection_rmse_px",
    "moving_to_static_reprojection_median_px",
    "moving_to_static_reprojection_observations",
    "error",
]

MARKER_FIELDS = [
    "method",
    "marker_id",
    "is_scale_anchor",
    "moving_inlier_frame_count",
    "moving_inlier_frames",
    "max_triangulation_angle_deg",
    "static_validation_cameras",
    "static_validation_camera_count",
    "raw_reconstructed_size_units",
    "anchor_scale_m_per_unit",
    "estimated_marker_size_cm",
    "expected_marker_size_cm",
    "absolute_size_error_cm",
    "relative_size_error_percent",
    "moving_fit_reprojection_rmse_px",
    "moving_to_static_reprojection_rmse_px",
    "moving_to_static_reprojection_median_px",
    "moving_to_static_reprojection_observations",
]


def _method_directories(args: argparse.Namespace) -> dict[str, str]:
    if not args.method:
        return dict(core.METHOD_DIRS)
    parsed: dict[str, str] = {}
    for value in args.method:
        if "=" not in value:
            raise RuntimeError(f"Invalid --method value: {value!r}")
        name, directory = (item.strip() for item in value.split("=", 1))
        if not name or not directory:
            raise RuntimeError(f"Invalid --method value: {value!r}")
        parsed[name] = directory
    return parsed


def _method_anchors(
    args: argparse.Namespace, methods: dict[str, str]
) -> dict[str, int]:
    anchors: dict[str, int] = {}
    for value in args.method_anchor:
        if "=" not in value:
            raise RuntimeError(f"Invalid --method-anchor value: {value!r}")
        name, marker_id = (item.strip() for item in value.split("=", 1))
        if name not in methods:
            raise RuntimeError(f"--method-anchor references unknown method {name!r}")
        anchors[name] = int(marker_id)
    return anchors


def _unavailable_summary(
    method: str, method_status: dict[str, Any], error: BaseException
) -> dict[str, Any]:
    cameras = method_status.get("available_static_cameras", [])
    return {
        "method": method,
        "status": "NOT_AVAILABLE",
        "original_method_status": method_status.get("status", "MISSING"),
        "original_method_success": method_status.get("success", False),
        "error": f"{type(error).__name__}: {error}",
        "available_static_cameras": cameras,
        "available_static_camera_count": len(cameras),
        "registered_moving_frames": 0,
        "evaluated_non_anchor_markers": 0,
        "pose_frame_ids": [],
        "pose_frame_count": 0,
        "marker_frame_ids": [],
        "marker_frame_count": 0,
        "candidate_frame_ids": [],
        "candidate_frame_count": 0,
        "inlier_frame_ids": [],
        "inlier_frame_count": 0,
        "rejected_frame_ids": {},
    }


def _common_support(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        summary
        for summary in summaries
        if not str(summary.get("status", "")).startswith("NOT_AVAILABLE")
    ]
    pose_support = set(successful[0].get("pose_frame_ids", [])) if successful else set()
    inlier_support = (
        set(successful[0].get("inlier_frame_ids", [])) if successful else set()
    )
    for summary in successful[1:]:
        pose_support.intersection_update(summary.get("pose_frame_ids", []))
        inlier_support.intersection_update(summary.get("inlier_frame_ids", []))
    return {
        "schema_version": 4,
        "methods": [summary.get("method") for summary in summaries],
        "successful_methods": [summary.get("method") for summary in successful],
        "all_methods_available": len(successful) == len(summaries),
        "common_pose_frame_ids": sorted(pose_support),
        "common_pose_frame_count": len(pose_support),
        "common_evaluation_inlier_frame_ids": sorted(inlier_support),
        "common_evaluation_inlier_frame_count": len(inlier_support),
        "per_method": {
            summary.get("method"): {
                "pose_frame_ids": summary.get("pose_frame_ids", []),
                "marker_frame_ids": summary.get("marker_frame_ids", []),
                "candidate_frame_ids": summary.get("candidate_frame_ids", []),
                "inlier_frame_ids": summary.get("inlier_frame_ids", []),
                "rejected_frame_ids": summary.get("rejected_frame_ids", {}),
            }
            for summary in summaries
        },
    }


def _summary_csv_rows(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row = dict(summary)
        row["available_static_cameras"] = ";".join(
            summary.get("available_static_cameras", [])
        )
        for name in (
            "pose_frame_ids",
            "marker_frame_ids",
            "candidate_frame_ids",
            "inlier_frame_ids",
        ):
            row[name] = ";".join(map(str, summary.get(name, [])))
        row["rejected_frame_ids"] = json.dumps(
            summary.get("rejected_frame_ids", {}), sort_keys=True
        )
        rows.append(row)
    return rows


def run_evaluation(
    args: argparse.Namespace,
    *,
    load_poses_fn: PoseLoader | None = None,
    evaluate_fn: Evaluator | None = None,
    report_fn: ReportWriter | None = None,
) -> Path:
    """Execute one evaluation with explicit replaceable scientific adapters."""
    load_poses_fn = load_poses_fn or core.load_poses
    evaluate_fn = evaluate_fn or core.evaluate
    report_fn = report_fn or default_report

    dataset = Path(args.dataset).resolve()
    results_root = Path(args.results_root).resolve()
    observations_root = (
        Path(args.observations_root).resolve()
        if args.observations_root
        else dataset / "aruco_observations"
    )
    core.CAMERAS = tuple(
        value.strip() for value in args.cameras.split(",") if value.strip()
    )
    if not core.CAMERAS:
        raise RuntimeError("--cameras must contain at least one camera ID")

    methods = _method_directories(args)
    method_anchors = _method_anchors(args, methods)
    anchor_file = observations_root / "REFERENCE_MARKER_ID.txt"
    anchor = (
        args.anchor_marker_id
        if args.anchor_marker_id is not None
        else int(anchor_file.read_text(encoding="utf-8").strip())
    )
    static_rows = core.best_static(
        core.read_csv(observations_root / "shared_static_aruco_observations.csv")
    )
    moving_rows = core.moving_rows(
        core.read_csv(observations_root / "shared_moving_aruco_observations.csv")
    )
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path("workspace/standalone_evaluation").resolve()
    )
    details_root = output_root / "marker_consistency"
    details_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    reprojection_rows: list[dict[str, Any]] = []
    for method, directory in methods.items():
        method_root = results_root / directory
        method_status = core.status(method_root)
        method_anchor = method_anchors.get(method, anchor)
        try:
            static_poses, moving_poses, pose_file, metadata = load_poses_fn(
                method,
                method_root,
                static_rows,
                moving_rows,
                method_anchor,
            )
            summary, markers, reprojections = evaluate_fn(
                method,
                static_poses,
                moving_poses,
                pose_file,
                metadata,
                static_rows,
                moving_rows,
                method_anchor,
                args.marker_length_m,
                args,
            )
            summary["original_method_status"] = method_status.get("status", "UNKNOWN")
            summary["original_method_success"] = method_status.get("success", False)
        except Exception as error:
            summary = _unavailable_summary(method, method_status, error)
            markers = []
            reprojections = []
        summaries.append(summary)
        marker_rows.extend(markers)
        reprojection_rows.extend(reprojections)

    (details_root / "REAL_DATA_MARKER_CONSISTENCY_SUMMARY.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    (details_root / "COMMON_SUPPORT_REPORT.json").write_text(
        json.dumps(_common_support(summaries), indent=2) + "\n",
        encoding="utf-8",
    )
    core.write_csv(
        details_root / "REAL_DATA_MARKER_CONSISTENCY_SUMMARY.csv",
        _summary_csv_rows(summaries),
        SUMMARY_FIELDS,
    )
    core.write_csv(
        details_root / "REAL_DATA_MARKER_CONSISTENCY_BY_MARKER.csv",
        marker_rows,
        MARKER_FIELDS,
    )
    core.write_csv(
        details_root / "REAL_DATA_MOVING_TO_STATIC_REPROJECTION.csv",
        reprojection_rows,
        [
            "method",
            "marker_id",
            "corner_index",
            "static_camera",
            "cross_camera_reprojection_error_px",
        ],
    )
    report_path = output_root / "REAL_DATA_MARKER_CONSISTENCY.txt"
    report_fn(
        report_path,
        dataset,
        anchor,
        args.marker_length_m,
        summaries,
        marker_rows,
    )
    print(report_path.read_text(encoding="utf-8"))
    print(
        "[OK] marker consistency written\n"
        f" report: {report_path}\n details: {details_root}"
    )
    return report_path


def main() -> None:
    run_evaluation(core.args_parse())


__all__ = ["main", "run_evaluation"]

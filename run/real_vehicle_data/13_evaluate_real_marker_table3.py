#!/usr/bin/env python3
"""Run the common real-data marker evaluation with a Table-III-style report.

The numerical pose estimates from AP01/AP02/AP03 are reused unchanged. Marker 3
(or the configured reference marker) alone fixes metric scale. For every marker,
the reported reconstructed size is the arithmetic mean of its four triangulated
edge lengths, matching the interpretation used in Table III of the in-cabin
monitoring paper. Cross-camera reprojection is reported separately.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_SCRIPT = Path(__file__).with_name("12_evaluate_real_marker_consistency.py")
CLEAR_CSV_NAME = "REAL_DATA_MARKER_EDGE_LENGTHS_AND_REPROJECTION.csv"


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "real_marker_consistency_base",
        BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base evaluator: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mean_four_marker_edges(corners):
    """Return one value so the base evaluator uses the four-edge arithmetic mean."""
    c = np.asarray(corners, dtype=np.float64)
    sides = [
        float(np.linalg.norm(c[1] - c[0])),
        float(np.linalg.norm(c[2] - c[1])),
        float(np.linalg.norm(c[3] - c[2])),
        float(np.linalg.norm(c[0] - c[3])),
    ]
    return [float(np.mean(sides))]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def mean(values):
    clean = [value for value in (finite_float(x) for x in values) if value is not None]
    return float(np.mean(clean)) if clean else None


def median(values):
    clean = [value for value in (finite_float(x) for x in values) if value is not None]
    return float(np.median(clean)) if clean else None


def fmt(value, digits=4):
    value = finite_float(value)
    return "NA" if value is None else f"{value:.{digits}f}"


def truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def paper_style_report(path, dataset, anchor, length, summaries):
    report_path = Path(path)
    detail_dir = report_path.parent / "marker_consistency"
    source_csv = detail_dir / "REAL_DATA_MARKER_CONSISTENCY_BY_MARKER.csv"
    marker_rows = read_csv(source_csv)

    clear_rows = []
    grouped = defaultdict(list)
    for row in marker_rows:
        clear = {
            "method": row.get("method", ""),
            "marker_id": row.get("marker_id", ""),
            "is_scale_anchor": row.get("is_scale_anchor", ""),
            "reconstructed_mean_edge_length_cm": row.get("estimated_marker_size_cm", ""),
            "absolute_edge_length_error_cm": row.get("absolute_size_error_cm", ""),
            "relative_edge_length_error_percent": row.get("relative_size_error_percent", ""),
            "cross_camera_reprojection_rmse_px": row.get(
                "moving_to_static_reprojection_rmse_px", ""
            ),
            "static_validation_cameras": row.get("static_validation_cameras", ""),
            "static_validation_camera_count": row.get(
                "static_validation_camera_count", ""
            ),
            "cross_camera_reprojection_corner_count": row.get(
                "moving_to_static_reprojection_observations", ""
            ),
            "moving_inlier_frame_count": row.get("moving_inlier_frame_count", ""),
            "max_triangulation_angle_deg": row.get(
                "max_triangulation_angle_deg", ""
            ),
        }
        clear_rows.append(clear)
        grouped[clear["method"]].append(clear)

    clear_fields = [
        "method",
        "marker_id",
        "is_scale_anchor",
        "reconstructed_mean_edge_length_cm",
        "absolute_edge_length_error_cm",
        "relative_edge_length_error_percent",
        "cross_camera_reprojection_rmse_px",
        "static_validation_cameras",
        "static_validation_camera_count",
        "cross_camera_reprojection_corner_count",
        "moving_inlier_frame_count",
        "max_triangulation_angle_deg",
    ]
    write_csv(detail_dir / CLEAR_CSV_NAME, clear_rows, clear_fields)

    width = 132
    lines = [
        "REAL-DATA SINGLE-ANCHOR MARKER EDGE-LENGTH AND CROSS-CAMERA EVALUATION",
        "=" * width,
        "",
        f"Dataset: {dataset}",
        f"Metric anchor: marker {anchor} = {100.0 * length:.2f} cm",
        "All physical markers are assumed to have a 17.00 cm edge length; therefore no repeated target-size column is shown.",
        "",
        "METHOD COMPARISON (NON-ANCHOR MARKERS)",
        "-" * width,
        f"{'Method':<8}{'Status':<27}{'Cams':>6}{'Move':>7}{'Markers':>9}"
        f"{'Mean edge [cm]':>17}{'Median abs err [cm]':>22}"
        f"{'Mean abs err [cm]':>20}{'Cross RMSE [px]':>18}{'Cross corners':>15}",
    ]

    for summary in summaries:
        method = str(summary.get("method", "-"))
        validation_rows = [
            row for row in grouped.get(method, [])
            if not truthy(row.get("is_scale_anchor"))
        ]
        edge_lengths = [
            row.get("reconstructed_mean_edge_length_cm")
            for row in validation_rows
        ]
        absolute_errors = [
            row.get("absolute_edge_length_error_cm")
            for row in validation_rows
        ]
        lines.append(
            f"{method:<8}{str(summary.get('status', '-')):<27}"
            f"{int(summary.get('available_static_camera_count', 0)):>6}"
            f"{int(summary.get('registered_moving_frames', 0)):>7}"
            f"{len(validation_rows):>9}"
            f"{fmt(mean(edge_lengths)):>17}"
            f"{fmt(median(absolute_errors)):>22}"
            f"{fmt(mean(absolute_errors)):>20}"
            f"{fmt(summary.get('moving_to_static_reprojection_rmse_px')):>18}"
            f"{int(summary.get('moving_to_static_reprojection_observations', 0)):>15}"
        )

    for summary in summaries:
        method = str(summary.get("method", "-"))
        rows = grouped.get(method, [])
        lines.extend([
            "",
            f"{method}: RECONSTRUCTED MARKER EDGE LENGTHS",
            "-" * width,
            f"{'Marker':>8}{'Mean edge [cm]':>18}{'Abs error [cm]':>18}"
            f"{'Rel error [%]':>16}{'Cross RMSE [px]':>19}"
            f"{'Static cameras':>27}{'Cross corners':>16}",
        ])
        if not rows:
            lines.append("No marker evaluation available for this method.")
            continue
        for row in sorted(rows, key=lambda item: int(float(item["marker_id"]))):
            marker_label = str(row["marker_id"])
            if truthy(row.get("is_scale_anchor")):
                marker_label += "*"
            cameras = row.get("static_validation_cameras", "") or "-"
            lines.append(
                f"{marker_label:>8}"
                f"{fmt(row.get('reconstructed_mean_edge_length_cm')):>18}"
                f"{fmt(row.get('absolute_edge_length_error_cm')):>18}"
                f"{fmt(row.get('relative_edge_length_error_percent')):>16}"
                f"{fmt(row.get('cross_camera_reprojection_rmse_px')):>19}"
                f"{cameras:>27}"
                f"{str(row.get('cross_camera_reprojection_corner_count', '0')):>16}"
            )

    lines.extend([
        "",
        "HOW TO READ CROSS-CAMERA REPROJECTION",
        "-" * width,
        "A marker corner is reconstructed in 3D using only moving-camera frames. The estimated camera-to-camera calibration",
        "then predicts where that 3D corner should appear in a static-camera image. Cross RMSE is the root-mean-square pixel",
        "distance between this prediction and the actually detected ArUco corner. It therefore tests whether the moving and",
        "static camera geometries agree. Zero pixels would be perfect. As a rough 4K interpretation, about 1-3 px is good,",
        "3-5 px is still plausible, above 10 px is suspicious, and errors of tens or hundreds of pixels indicate wrong geometry.",
        "These ranges are practical guidance rather than universal acceptance thresholds.",
        "",
        "CALCULATION",
        "-" * width,
        "1. Existing AP01/AP02/AP03 poses are frozen; no method is re-run or optimized.",
        "2. Each marker corner is triangulated only from moving-camera observations using RANSAC.",
        f"3. Marker {anchor} alone sets metric scale to {100.0 * length:.2f} cm.",
        "4. The reported size is the arithmetic mean of the four reconstructed marker edges, analogous to Table III of the paper.",
        "5. Every other marker is compared with the common physical edge length of 17.00 cm.",
        "6. Cross-camera reprojection uses static images only for evaluation, not for triangulating the marker corners.",
        "",
        "INTERPRETATION LIMIT",
        "-" * width,
        "This is a common post-hoc geometric-consistency evaluation, not independent camera-pose ground truth.",
        "The marker-size check evaluates metric reconstruction consistency; Cross RMSE additionally tests moving-to-static",
        "camera agreement. Camera/method failures remain visible through missing rows and coverage counts.",
        "",
        "* Scale anchor: its reconstructed edge is fixed to 17.00 cm by construction and is not included in summary errors.",
        f"Detailed CSV: {detail_dir / CLEAR_CSV_NAME}",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    base = load_base_module()
    base.marker_lengths = mean_four_marker_edges
    base.report = paper_style_report
    base.main()


if __name__ == "__main__":
    main()

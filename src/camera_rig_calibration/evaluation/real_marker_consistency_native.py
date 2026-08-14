#!/usr/bin/env python3
"""Real-vehicle marker consistency on native metric method outputs.

This front-end reuses the established triangulation/reprojection implementation
but deliberately separates two concepts that must not be conflated:

* evaluation/export anchor: coordinate/alignment reference (normally marker 0),
* metric scale: the native metric scale already produced by AP01/AP02/AP03.

No marker is forced to the configured 17 cm length during this evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import marker_consistency_core as core
from .marker_consistency_reporting import report as base_report
from .marker_consistency_runner import run_evaluation


def _load_ap01_any_static(
    root,
    static_pose,
    static_obs,
    moving_obs,
    anchor,
):
    diag_path = root / "static_extrinsics" / "AP01_DIAGNOSTICS.json"
    if not diag_path.is_file():
        diag_path = root / "03_static_extrinsics" / "AP01_DIAGNOSTICS.json"
    if not diag_path.is_file():
        diag_path = next(iter(root.rglob("AP01_DIAGNOSTICS.json")))
    diag = json.loads(diag_path.read_text())
    scale = core.num(diag["metric_scale"]["scale_m_per_colmap_unit"])

    images = root / "moving_colmap" / "sparse_txt_best" / "images.txt"
    if not images.is_file():
        images = root / "01_moving_colmap" / "sparse_txt_best" / "images.txt"
    if not images.is_file():
        images = next(iter(root.rglob("sparse_txt_best/images.txt")))
    col = core.scaled_colmap_poses(images, scale)

    anchor_candidates = []
    anchor_weights = []
    anchor_cameras = []
    for (camera_id, marker_id), row in static_obs.items():
        if marker_id != anchor or camera_id not in static_pose:
            continue
        anchor_candidates.append(static_pose[camera_id] @ core.pnp_pose(row))
        anchor_weights.append(core.quality(row))
        anchor_cameras.append(camera_id)
    if not anchor_candidates:
        raise RuntimeError(
            f"Evaluation anchor {anchor} is not observed by any solved AP01 static camera"
        )
    method_anchor = core.mean_transform(anchor_candidates, anchor_weights)

    alignment_candidates = []
    alignment_weights = []
    for row in moving_obs:
        if int(float(row["marker_id"])) != anchor:
            continue
        frame = core.frame_id(row.get("frame_id", row.get("observer_id")))
        if frame not in col:
            continue
        method_moving = method_anchor @ core.inv(core.pnp_pose(row))
        alignment_candidates.append(method_moving @ core.inv(col[frame]))
        alignment_weights.append(core.quality(row))
    if not alignment_candidates:
        raise RuntimeError("No AP01 moving-frame observation supports anchor alignment")
    method_colmap = core.mean_transform(alignment_candidates, alignment_weights)
    return (
        {frame: method_colmap @ pose for frame, pose in col.items()},
        {
            "source": str(images),
            "native_scale": scale,
            "metric_scale_source": "AP01 native metric scale",
            "evaluation_anchor_marker_id": anchor,
            "anchor_static_cameras": sorted(set(anchor_cameras)),
            "anchor_alignment_observations": len(alignment_candidates),
        },
    )


def _load_native_poses(method, root, static_obs, moving_obs, anchor):
    """Load native metric trajectories without replacing shared functions."""
    static_poses, static_pose_file = core.static_poses(root)
    if method.startswith("AP01"):
        moving_poses, metadata = _load_ap01_any_static(
            root,
            static_poses,
            static_obs,
            moving_obs,
            anchor,
        )
    elif method.startswith("AP02"):
        moving_poses, metadata = core.load_ap02(root)
    else:
        moving_poses, metadata = core.load_ap03(root)
    return static_poses, moving_poses, static_pose_file, metadata


def _evaluate_native_metric(
    method,
    sposes,
    mposes,
    spfile,
    meta,
    srows,
    mrows,
    anchor,
    length,
    args,
):
    pose_frames = sorted(mposes)
    marker_frames = sorted(
        {
            core.frame_id(row.get("frame_id", row.get("observer_id")))
            for row in mrows
            if core.frame_id(row.get("frame_id", row.get("observer_id"))) is not None
        }
    )
    candidate_frames = sorted(set(pose_frames).intersection(marker_frames))
    markers = sorted({int(float(row["marker_id"])) for row in mrows})
    reconstructed = {}
    reproj_rows = []

    for marker in markers:
        rows = core.selected_marker_rows(
            mrows, mposes, marker, args.max_moving_observations_per_marker
        )
        if len(rows) < args.min_inliers:
            continue
        corners = []
        fit = []
        frames = set()
        angles = []
        ok = True
        for corner_index in range(4):
            observations = []
            frame_ids = []
            for row in rows:
                frame = core.frame_id(row.get("frame_id", row.get("observer_id")))
                if frame not in mposes:
                    continue
                observations.append(core.obs(row, mposes[frame], corner_index))
                frame_ids.append(frame)
            point, inside, errors, angle = core.robust_triangulate(
                observations,
                args,
                7919 + 17 * marker + corner_index,
            )
            if point is None:
                ok = False
                break
            corners.append(point)
            angles.append(angle)
            for index in inside:
                fit.append(errors[index])
                frames.add(frame_ids[index])
        if not ok:
            continue

        static_errors = []
        static_cameras = []
        for camera_id in core.CAMERAS:
            row = srows.get((camera_id, marker))
            if row is None or camera_id not in sposes:
                continue
            camera_errors = []
            for corner_index, point in enumerate(corners):
                observation = core.obs(row, sposes[camera_id], corner_index)
                projected = core.project(point, observation)
                error = (
                    float("inf")
                    if projected is None
                    else float(np.linalg.norm(projected - observation["point"]))
                )
                if np.isfinite(error):
                    static_errors.append(error)
                    camera_errors.append(error)
                reproj_rows.append(
                    {
                        "method": method,
                        "marker_id": marker,
                        "corner_index": corner_index,
                        "static_camera": camera_id,
                        "cross_camera_reprojection_error_px": error,
                    }
                )
            if camera_errors:
                static_cameras.append(camera_id)

        reconstructed[marker] = {
            "raw": float(np.median(core.marker_lengths(corners))),
            "fit": fit,
            "cross": static_errors,
            "cams": static_cameras,
            "frames": frames,
            "angle": max(angles),
        }

    inlier_frames = sorted(
        set().union(*(data["frames"] for data in reconstructed.values()))
        if reconstructed
        else set()
    )
    frame_support = {
        "pose_frame_ids": pose_frames,
        "pose_frame_count": len(pose_frames),
        "marker_frame_ids": marker_frames,
        "marker_frame_count": len(marker_frames),
        "candidate_frame_ids": candidate_frames,
        "candidate_frame_count": len(candidate_frames),
        "inlier_frame_ids": inlier_frames,
        "inlier_frame_count": len(inlier_frames),
        "rejected_frame_ids": {
            "pose_without_marker": sorted(set(pose_frames) - set(marker_frames)),
            "marker_without_pose": sorted(set(marker_frames) - set(pose_frames)),
            "candidate_not_used_as_inlier": sorted(
                set(candidate_frames) - set(inlier_frames)
            ),
        },
    }

    marker_rows = []
    size_cm = []
    size_pct = []
    cross = []
    fit = []
    validated = 0
    for marker, data in sorted(reconstructed.items()):
        # The moving/static trajectories are already metric. Do NOT re-scale them
        # so that any marker equals the expected physical length.
        estimated = data["raw"]
        error_cm = 100.0 * abs(estimated - length)
        error_pct = 100.0 * abs(estimated - length) / length
        size_cm.append(error_cm)
        size_pct.append(error_pct)
        cross += data["cross"]
        fit += data["fit"]
        validated += int(bool(data["cross"]))
        marker_rows.append(
            {
                "method": method,
                "marker_id": marker,
                "is_scale_anchor": False,
                "moving_inlier_frame_count": len(data["frames"]),
                "moving_inlier_frames": ";".join(map(str, sorted(data["frames"]))),
                "max_triangulation_angle_deg": data["angle"],
                "static_validation_cameras": ";".join(data["cams"]),
                "static_validation_camera_count": len(data["cams"]),
                "raw_reconstructed_size_units": data["raw"],
                "anchor_scale_m_per_unit": 1.0,
                "estimated_marker_size_cm": 100.0 * estimated,
                "expected_marker_size_cm": 100.0 * length,
                "absolute_size_error_cm": error_cm,
                "relative_size_error_percent": error_pct,
                "moving_fit_reprojection_rmse_px": core.rmse(data["fit"]),
                "moving_to_static_reprojection_rmse_px": core.rmse(data["cross"]),
                "moving_to_static_reprojection_median_px": core.med(data["cross"]),
                "moving_to_static_reprojection_observations": len(data["cross"]),
            }
        )

    expected = len(core.CAMERAS)
    summary = {
        "method": method,
        "status": (
            "OK" if len(sposes) == expected else f"PARTIAL_{len(sposes)}_OF_{expected}"
        ),
        "available_static_cameras": sorted(sposes),
        "available_static_camera_count": len(sposes),
        "registered_moving_frames": len(mposes),
        "static_pose_file": str(spfile),
        "trajectory": meta,
        "anchor_marker_id": anchor,
        "evaluation_anchor_marker_id": anchor,
        "metric_scale_source": "native_method_metric_output_no_evaluation_rescale",
        "reconstructed_markers_total": len(reconstructed),
        "evaluated_non_anchor_markers": len(size_cm),
        "markers_with_moving_to_static_validation": validated,
        "marker_length_rmse_cm": core.rmse(size_cm),
        "marker_length_rmse_percent": core.rmse(size_pct),
        "median_absolute_size_error_cm": core.med(size_cm),
        "median_relative_size_error_percent": core.med(size_pct),
        "moving_fit_reprojection_rmse_px": core.rmse(fit),
        "moving_fit_reprojection_median_px": core.med(fit),
        "moving_to_static_reprojection_rmse_px": core.rmse(cross),
        "moving_to_static_reprojection_median_px": core.med(cross),
        "moving_to_static_reprojection_observations": len(cross),
        **frame_support,
    }
    return summary, marker_rows, reproj_rows


def _report_native(path, dataset, anchor, length, summaries, marker_rows):
    base_report(path, dataset, anchor, length, summaries, marker_rows)
    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8")
    note = (
        "Metric scale: native method output; the evaluation/export anchor is NOT "
        "used to force any marker to 17 cm.\n"
        f"Evaluation/export anchor for frame alignment: marker {anchor}.\n"
    )
    text = text.replace(
        f"Expected marker edge length: {100 * length:.2f} cm\n",
        f"Expected marker edge length: {100 * length:.2f} cm\n{note}",
        1,
    )
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    run_evaluation(
        core.args_parse(),
        load_poses_fn=_load_native_poses,
        evaluate_fn=_evaluate_native_metric,
        report_fn=_report_native,
    )


if __name__ == "__main__":
    main()

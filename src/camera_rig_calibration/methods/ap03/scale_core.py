#!/usr/bin/env python3
"""
AP03 marker-size-only metric scale.

Allowed method input:
- COLMAP sparse model from targetless reconstruction
- registered images
- camera intrinsics
- ArUco marker detections
- known physical marker side length

Forbidden method input:
- GT camera poses
- GT marker poses
- SDF marker map
- known marker-map coordinates

Output:
- scaled static camera poses in arbitrary scaled COLMAP frame
- scale metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np

from camera_rig_calibration.methods.common.constants import (
    ARUCO_DICT_NAME,
    MARKER_LENGTH_M,
    STATIC_CAMERAS,
)
from camera_rig_calibration.methods.common.geometry import (
    R_to_rpy_deg,
    R_to_rvec,
    make_T,
)
from camera_rig_calibration.methods.common.aruco_utils import make_aruco_detector
from camera_rig_calibration.methods.common.projection import (
    reproj_errors_px,
    robust_triangulate_point,
)

from .scale_common import load_best_colmap_model, source_id_from_image_name


AP03_ROOT = Path("workspace/standalone_methods/ap03")
DEFAULT_OUT = AP03_ROOT / "07_final_results"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_marker_ids(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def pose_row(entity_type: str, entity_id: str, T: np.ndarray, source: str) -> dict:
    rpy = R_to_rpy_deg(T[:3, :3])
    rvec = R_to_rvec(T[:3, :3])
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "source": source,
        "x_m": float(T[0, 3]),
        "y_m": float(T[1, 3]),
        "z_m": float(T[2, 3]),
        "roll_deg": float(rpy[0]),
        "pitch_deg": float(rpy[1]),
        "yaw_deg": float(rpy[2]),
        "rvec_x": float(rvec[0]),
        "rvec_y": float(rvec[1]),
        "rvec_z": float(rvec[2]),
    }


def pose_fields() -> list[str]:
    return [
        "entity_type", "entity_id", "source",
        "x_m", "y_m", "z_m",
        "roll_deg", "pitch_deg", "yaw_deg",
        "rvec_x", "rvec_y", "rvec_z",
    ]


def detect_marker_observations(
    images: dict,
    marker_ids: set[int],
    dictionary: str,
    image_dir: Path,
    detection_mode: str = "baseline",
    minimum_marker_area_px2: float = 0.0,
) -> list[dict]:
    detect = make_aruco_detector(dictionary, detection_mode)
    obs = []

    for image_name in sorted(images.keys()):
        image_path = image_dir / image_name
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray)
        if ids is None or len(ids) == 0:
            continue

        ids = ids.reshape(-1)
        for idx, marker_id in enumerate(ids.tolist()):
            marker_id = int(marker_id)
            if marker_id not in marker_ids:
                continue

            pts = np.asarray(corners[idx], dtype=np.float64).reshape(4, 2)
            area = float(cv2.contourArea(pts.astype(np.float32)))
            if area < minimum_marker_area_px2:
                continue

            for corner_idx in range(4):
                obs.append({
                    "image_name": image_name,
                    "marker_id": marker_id,
                    "corner_idx": corner_idx,
                    "u": float(pts[corner_idx, 0]),
                    "v": float(pts[corner_idx, 1]),
                    "area_px2": area,
                })

    return obs


def select_observations_per_marker(
    observations: list[dict],
    maximum_observations_per_marker: int | None,
) -> tuple[list[dict], list[dict]]:
    """Select whole image/marker detections before corner triangulation."""

    grouped: dict[int, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in observations:
        grouped[int(row["marker_id"])][str(row["image_name"])].append(row)
    selected: list[dict] = []
    diagnostics: list[dict] = []
    for marker_id, by_image in sorted(grouped.items()):
        ranked_images = sorted(
            by_image,
            key=lambda image_name: (
                -max(
                    float(row.get("selection_score") or 0.0)
                    for row in by_image[image_name]
                ),
                -max(
                    float(row.get("area_px2") or 0.0)
                    for row in by_image[image_name]
                ),
                image_name,
            ),
        )
        chosen = (
            ranked_images
            if maximum_observations_per_marker is None
            else ranked_images[:maximum_observations_per_marker]
        )
        selected_images = set(chosen)
        for rank, image_name in enumerate(ranked_images, 1):
            image_rows = by_image[image_name]
            diagnostics.append(
                {
                    "marker_id": marker_id,
                    "image_name": image_name,
                    "quality_rank": rank,
                    "selected": image_name in selected_images,
                    "selection_score": max(
                        float(row.get("selection_score") or 0.0)
                        for row in image_rows
                    ),
                    "area_px2": max(
                        float(row.get("area_px2") or 0.0)
                        for row in image_rows
                    ),
                    "maximum_observations_per_marker": (
                        maximum_observations_per_marker
                    ),
                    "tie_breaker": "stable ascending image name",
                }
            )
            if image_name in selected_images:
                selected.extend(image_rows)
    return selected, diagnostics


def triangulate_marker_corners(obs: list[dict], images: dict, cameras: dict, args) -> tuple[dict, list[dict]]:
    grouped = defaultdict(list)
    for o in obs:
        grouped[(int(o["marker_id"]), int(o["corner_idx"]))].append(o)

    corners_3d = {}
    tri_rows = []

    for key in sorted(grouped.keys()):
        marker_id, corner_idx = key
        corner_obs = grouped[key]
        if len(corner_obs) < args.min_inliers:
            continue

        try:
            res = robust_triangulate_point(
                corner_obs,
                images,
                cameras,
                ransac_iters=args.ransac_iters,
                reproj_thresh_px=args.reproj_thresh_px,
                min_inliers=args.min_inliers,
                random_seed=1100 + marker_id * 10 + corner_idx,
            )
        except Exception as exc:
            tri_rows.append({
                "marker_id": marker_id,
                "corner_idx": corner_idx,
                "status": "TRIANGULATION_FAILED",
                "note": str(exc),
            })
            continue

        X = np.asarray(res["X"], dtype=np.float64)
        errs = reproj_errors_px(X, corner_obs, images, cameras)
        inlier_errs = [errs[i] for i in res["inlier_idx"]]

        corners_3d[key] = X
        tri_rows.append({
            "marker_id": marker_id,
            "corner_idx": corner_idx,
            "status": "OK",
            "x_colmap": float(X[0]),
            "y_colmap": float(X[1]),
            "z_colmap": float(X[2]),
            "obs_count": int(res["obs_count"]),
            "inlier_count": int(res["inlier_count"]),
            "mean_reproj_px": float(np.mean(inlier_errs)) if inlier_errs else float("nan"),
            "median_reproj_px": float(np.median(inlier_errs)) if inlier_errs else float("nan"),
            "max_reproj_px": float(np.max(inlier_errs)) if inlier_errs else float("nan"),
            "note": "",
        })

    return corners_3d, tri_rows


def compute_scale_observations(corners_3d: dict, marker_length_m: float) -> list[dict]:
    by_marker = defaultdict(dict)
    for (marker_id, corner_idx), X in corners_3d.items():
        by_marker[marker_id][corner_idx] = X

    specs = [
        (0, 1, marker_length_m, "side_01"),
        (1, 2, marker_length_m, "side_12"),
        (2, 3, marker_length_m, "side_23"),
        (3, 0, marker_length_m, "side_30"),
        (0, 2, math.sqrt(2.0) * marker_length_m, "diag_02"),
        (1, 3, math.sqrt(2.0) * marker_length_m, "diag_13"),
    ]

    rows = []
    for marker_id in sorted(by_marker.keys()):
        corners = by_marker[marker_id]
        if not all(i in corners for i in range(4)):
            continue

        for a, b, metric_len, kind in specs:
            col_len = float(np.linalg.norm(corners[a] - corners[b]))
            if not math.isfinite(col_len) or col_len <= 1e-12:
                continue
            rows.append({
                "marker_id": marker_id,
                "segment": kind,
                "corner_a": a,
                "corner_b": b,
                "length_colmap_units": col_len,
                "length_metric_m": metric_len,
                "scale_m_per_colmap_unit": metric_len / col_len,
            })

    return rows


def robust_scale(scale_rows: list[dict]) -> tuple[float, list[dict], dict]:
    vals = np.array([float(r["scale_m_per_colmap_unit"]) for r in scale_rows], dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if len(vals) < 4:
        raise RuntimeError(f"Too few marker-size scale observations: {len(vals)}")

    med = float(np.median(vals))
    abs_dev = np.abs(vals - med)
    mad = float(np.median(abs_dev))
    rel_mad = mad / med if med > 0 else float("inf")

    # Robust but not too aggressive: if MAD is near zero, still allow 10%.
    threshold = max(3.0 * mad, 0.10 * med)
    keep = []
    used_vals = []
    for r in scale_rows:
        v = float(r["scale_m_per_colmap_unit"])
        used = math.isfinite(v) and v > 0.0 and abs(v - med) <= threshold
        rr = dict(r)
        rr["used_for_scale"] = "yes" if used else "no"
        rr["scale_abs_dev_from_median"] = abs(v - med) if math.isfinite(v) else float("nan")
        keep.append(rr)
        if used:
            used_vals.append(v)

    if len(used_vals) < 4:
        used_vals = vals.tolist()
        for rr in keep:
            rr["used_for_scale"] = "yes_fallback_all"

    used_vals_np = np.array(used_vals, dtype=np.float64)
    scale = float(np.median(used_vals_np))
    total_by_marker = Counter(
        int(row["marker_id"]) for row in scale_rows
    )
    used_by_marker = Counter(
        int(row["marker_id"])
        for row in keep
        if str(row["used_for_scale"]).startswith("yes")
    )
    meta = {
        "scale_m_per_colmap_unit": scale,
        "num_scale_observations_total": int(len(scale_rows)),
        "num_scale_observations_used": int(len(used_vals)),
        "raw_median_scale": med,
        "raw_mad_scale": mad,
        "raw_rel_mad_scale": rel_mad,
        "used_mean_scale": float(np.mean(used_vals_np)),
        "used_std_scale": float(np.std(used_vals_np)),
        "used_rel_std_scale": float(np.std(used_vals_np) / scale) if scale > 0 else float("inf"),
        "scale_candidates_per_marker": dict(sorted(total_by_marker.items())),
        "scale_inliers_per_marker": dict(sorted(used_by_marker.items())),
        "scale_outliers_per_marker": {
            marker: total_by_marker[marker] - used_by_marker[marker]
            for marker in sorted(total_by_marker)
        },
    }
    return scale, keep, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--marker-ids", default="0-14")
    ap.add_argument("--marker-length-m", type=float, default=MARKER_LENGTH_M)
    ap.add_argument("--reproj-thresh-px", type=float, default=5.0)
    ap.add_argument("--ransac-iters", type=int, default=1000)
    ap.add_argument("--min-inliers", type=int, default=4)
    ap.add_argument("--max-rel-scale-std-warn", type=float, default=0.10)
    ap.add_argument("--dictionary", default=ARUCO_DICT_NAME)
    ap.add_argument(
        "--detection-mode",
        choices=("baseline", "subpixel_refined", "high_sensitivity"),
        default="baseline",
    )
    ap.add_argument("--txt-root", type=Path)
    ap.add_argument("--image-dir", type=Path)
    ap.add_argument("--inspect-summary", type=Path)
    ap.add_argument("--accepted-observations", type=Path)
    ap.add_argument("--maximum-observations-per-marker", type=int)
    ap.add_argument("--min-area-px2", type=float, default=100.0)
    ap.add_argument(
        "--scale-input-policy",
        choices=(
            "registered_image_redetection_v1",
            "wizard_filtered_observations_v1",
        ),
        default="registered_image_redetection_v1",
    )
    ap.add_argument("--static-cameras", default=",".join(STATIC_CAMERAS))
    args = ap.parse_args()
    static_cameras = [
        value.strip()
        for value in args.static_cameras.split(",")
        if value.strip()
    ]
    txt_root = (
        args.txt_root.resolve()
        if args.txt_root is not None
        else AP03_ROOT / "02_colmap_sparse" / "sparse_txt"
    )
    image_dir = (
        args.image_dir.resolve()
        if args.image_dir is not None
        else AP03_ROOT / "01_colmap_dataset" / "images"
    )
    inspect_summary = (
        args.inspect_summary.resolve()
        if args.inspect_summary is not None
        else AP03_ROOT
        / "03_reconstruction_inspection"
        / "colmap_model_summary.csv"
    )

    ensure_dir(args.out_dir)
    marker_ids = set(parse_marker_ids(args.marker_ids))

    best_model = None
    model_dir = None
    cameras = {}
    images = {}
    points3d = {}
    static_poses = {}

    obs = []
    corners_3d = {}
    tri_rows = []
    scale_rows = []
    observation_selection_rows = []
    detected_corner_observation_count = 0
    scaled_pose_rows = []

    scale = None
    failure_reason = ""
    status = "FAILED_COLMAP"

    scale_meta = {
        "scale_m_per_colmap_unit": None,
        "num_scale_observations_total": 0,
        "num_scale_observations_used": 0,
        "raw_median_scale": None,
        "raw_mad_scale": None,
        "raw_rel_mad_scale": None,
        "used_mean_scale": None,
        "used_std_scale": None,
        "used_rel_std_scale": None,
    }

    try:
        best_model, model_dir, cameras, images, points3d = (
            load_best_colmap_model(txt_root, inspect_summary)
        )

        for image_name, payload in sorted(images.items()):
            if not image_name.startswith("static_"):
                continue
            cam = source_id_from_image_name(image_name)
            if cam in static_cameras:
                static_poses[cam] = payload["T_col_cam"]

        try:
            obs = detect_marker_observations(
                images,
                marker_ids,
                args.dictionary,
                image_dir,
                args.detection_mode,
                args.min_area_px2,
            )
            if args.accepted_observations is not None:
                allowed: dict[tuple[str, int], dict[str, str]] = {}
                for row in read_csv(args.accepted_observations):
                    marker = int(float(row["marker_id"]))
                    if row.get("observer_type") == "static":
                        observer = (
                            row.get("observer_id")
                            or row.get("camera_name")
                            or ""
                        )
                        name = f"static_{observer}.png"
                    else:
                        source = Path(
                            str(row.get("image_path", ""))
                        ).name
                        if not source:
                            source = (
                                f"frame_{int(float(row['frame_id'])):06d}.png"
                            )
                        name = f"moving_{source}"
                    key = (name, marker)
                    previous = allowed.get(key)
                    if (
                        previous is None
                        or float(row.get("selection_score") or 0.0)
                        > float(previous.get("selection_score") or 0.0)
                    ):
                        allowed[key] = row
                obs = [
                    {
                        **row,
                        "selection_score": allowed[
                            (str(row["image_name"]), int(row["marker_id"]))
                        ].get("selection_score", 0.0),
                    }
                    for row in obs
                    if (str(row["image_name"]), int(row["marker_id"]))
                    in allowed
                ]
            detected_corner_observation_count = len(obs)
            if args.scale_input_policy == "wizard_filtered_observations_v1":
                obs, observation_selection_rows = (
                    select_observations_per_marker(
                        obs, args.maximum_observations_per_marker
                    )
                )
            corners_3d, tri_rows = triangulate_marker_corners(
                obs, images, cameras, args
            )
            scale_rows_raw = compute_scale_observations(
                corners_3d, args.marker_length_m
            )
            scale, scale_rows, scale_meta = robust_scale(scale_rows_raw)
        except Exception as exc:
            status = "FAILED_SCALE"
            failure_reason = f"{type(exc).__name__}: {exc}"

        if scale is not None:
            for cam in sorted(static_poses):
                T = np.array(static_poses[cam], dtype=np.float64).copy()
                T[:3, 3] *= scale
                scaled_pose_rows.append(
                    pose_row(
                        "static_camera",
                        cam,
                        T,
                        "ap03_colmap_marker_size_scale_only",
                    )
                )

            n = len(static_poses)
            weak = (
                scale_meta.get("used_rel_std_scale") is not None
                and scale_meta["used_rel_std_scale"]
                > args.max_rel_scale_std_warn
            )

            expected = len(static_cameras)
            if n == expected:
                status = (
                    "SCALE_WEAK_CHECK_REQUIRED"
                    if weak
                    else "OK_FULL"
                )
            elif n >= 2:
                status = f"PARTIAL_{n}_OF_{expected}"
                if weak:
                    status += "_SCALE_WEAK"
            elif n == 1:
                status = f"FAILED_NO_PAIR_1_OF_{expected}"
            else:
                status = "FAILED_NO_STATIC_CAMERAS"

    except Exception as exc:
        status = "FAILED_COLMAP"
        failure_reason = f"{type(exc).__name__}: {exc}"

    available_cameras = sorted(static_poses)
    missing_cameras = sorted(
        set(static_cameras) - set(available_cameras)
    )

    write_csv(
        args.out_dir / "AP03_OBSERVATION_SELECTION.csv",
        observation_selection_rows,
        [
            "marker_id",
            "image_name",
            "quality_rank",
            "selected",
            "selection_score",
            "area_px2",
            "maximum_observations_per_marker",
            "tie_breaker",
        ],
    )

    write_csv(
        args.out_dir / "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv",
        scaled_pose_rows,
        pose_fields(),
    )

    write_csv(
        args.out_dir / "AP03_MARKER_SIZE_SCALE_ONLY_SCALE_OBSERVATIONS.csv",
        scale_rows,
        [
            "marker_id", "segment", "corner_a", "corner_b",
            "length_colmap_units", "length_metric_m",
            "scale_m_per_colmap_unit", "used_for_scale",
            "scale_abs_dev_from_median",
        ],
    )

    write_csv(
        args.out_dir / "AP03_MARKER_SIZE_SCALE_ONLY_TRIANGULATED_CORNERS.csv",
        tri_rows,
        [
            "marker_id", "corner_idx", "status",
            "x_colmap", "y_colmap", "z_colmap",
            "obs_count", "inlier_count",
            "mean_reproj_px", "median_reproj_px", "max_reproj_px",
            "note",
        ],
    )

    meta = {
        "approach": "AP03_targetless_colmap_marker_size_scale_only",
        "status": status,
        "failure_reason": failure_reason,
        "important_rule": (
            "No SDF marker map, no GT marker pose, no GT camera pose "
            "used by this method output."
        ),
        "best_model": best_model,
        "model_dir": str(model_dir) if model_dir is not None else "",
        "registered_images": len(images),
        "registered_static_cameras": len(static_poses),
        "registered_moving_frames": len(
            [n for n in images if n.startswith("moving_")]
        ),
        "available_static_cameras": available_cameras,
        "missing_static_cameras": missing_cameras,
        "num_sparse_points3d": len(points3d),
        "marker_ids_requested": sorted(marker_ids),
        "marker_length_m": args.marker_length_m,
        "minimum_marker_area_px2": args.min_area_px2,
        "scale_input_policy": args.scale_input_policy,
        "detection_mode": args.detection_mode,
        "maximum_observations_per_marker": (
            args.maximum_observations_per_marker
        ),
        "detected_corner_observations_before_selection": (
            detected_corner_observation_count
        ),
        "detected_corner_observations": len(obs),
        "selected_image_marker_observations": sum(
            bool(row["selected"]) for row in observation_selection_rows
        ),
        "triangulated_marker_corners": len(corners_3d),
        **scale_meta,
    }

    metadata_path = (
        args.out_dir / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
    )
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n")

    scale_text = (
        f"{float(scale):.12f}"
        if scale is not None
        else "UNAVAILABLE"
    )
    rel_std = scale_meta.get("used_rel_std_scale")
    rel_std_text = (
        f"{float(rel_std):.6f}"
        if rel_std is not None
        else "UNAVAILABLE"
    )

    report = [
        "AP03 MARKER-SIZE-ONLY SCALE / PARTIAL COVERAGE",
        "================================================",
        "",
        f"status: {status}",
        f"failure_reason: {failure_reason or '-'}",
        f"best_model: {best_model}",
        (
            "registered_static_cameras: "
            f"{len(static_poses)} / {len(static_cameras)}"
        ),
        f"available_static_cameras: {available_cameras}",
        f"missing_static_cameras: {missing_cameras}",
        f"registered_images: {len(images)}",
        f"detected_corner_observations: {len(obs)}",
        f"triangulated_marker_corners: {len(corners_3d)}",
        f"scale_m_per_colmap_unit: {scale_text}",
        (
            "scale_observations_used/total: "
            f"{scale_meta.get('num_scale_observations_used', 0)} / "
            f"{scale_meta.get('num_scale_observations_total', 0)}"
        ),
        f"used_rel_std_scale: {rel_std_text}",
        "",
        (
            "Partial static-camera poses are exported when metric scale "
            "is available. Missing cameras remain explicitly missing."
        ),
        (
            "Rule: scale is estimated only from known physical marker "
            "size, not from known marker-map positions."
        ),
    ]

    report_path = args.out_dir / "AP03_MARKER_SIZE_SCALE_ONLY_REPORT.txt"
    report_path.write_text("\n".join(report) + "\n")

    print("\n".join(report))
    print("[OK] wrote:", metadata_path)
    print("[OK] wrote:", report_path)


if __name__ == "__main__":
    main()

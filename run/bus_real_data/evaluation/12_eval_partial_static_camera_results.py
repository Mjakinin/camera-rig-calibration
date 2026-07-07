#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.constants import (
    STATIC_CAMERAS,
    WORLD_SDF_MOVING_CAMERA,
    REF_MARKER_ENTITY,
)
from _shared.common.geometry import (
    make_T,
    invT,
    rpy_to_R,
    rvec_to_R,
    R_to_rpy_deg,
    R_to_rvec,
    rot_error_deg,
)
from _shared.common.sdf_utils import gt_static_camera_poses_ref_aruco


FINAL = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT")

AP01_SOURCE = Path(
    "results/bus_real_data/"
    "01_marker_direct_relay_multimarker_multichain/"
    "07_final_extrinsics_cam3_reference/final_extrinsics_summary.csv"
)
AP01_META = AP01_SOURCE.parent / "AP01_PARTIAL_STATUS.json"

AP02_SOURCE = Path(
    "results/bus_real_data/"
    "02_ref_marker_graph_ba/08_final_results/"
    "ap02_with_moving_static_camera_poses_ref_marker.csv"
)

AP03_SOURCE = Path(
    "results/bus_real_data/"
    "03_targetless_colmap_aruco_scale/07_final_results/"
    "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
)
AP03_META = AP03_SOURCE.parent / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"

SHORT = {
    "cam_edge_0": "cam0",
    "cam_edge_1": "cam1",
    "cam_edge_3": "cam3",
    "cam_edge_5": "cam5",
}

PAIRS = [
    ("cam_edge_0", "cam_edge_1"),
    ("cam_edge_0", "cam_edge_3"),
    ("cam_edge_0", "cam_edge_5"),
    ("cam_edge_1", "cam_edge_3"),
    ("cam_edge_1", "cam_edge_5"),
    ("cam_edge_3", "cam_edge_5"),
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], preferred: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = set(preferred)
    for row in rows:
        keys.update(row.keys())
    fields = preferred + sorted(keys - set(preferred))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def norm_cam(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    text = text.replace("static_", "").replace(".png", "").replace(".jpg", "")
    for cam in STATIC_CAMERAS:
        if cam in text:
            return cam
    return None


def as_float(row: dict, keys: list[str]):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def parse_generic_pose(row: dict) -> np.ndarray:
    x = as_float(row, ["x_m", "x", "tx", "t_x", "est_x_m"])
    y = as_float(row, ["y_m", "y", "ty", "t_y", "est_y_m"])
    z = as_float(row, ["z_m", "z", "tz", "t_z", "est_z_m"])
    if None in (x, y, z):
        raise RuntimeError("missing translation columns")

    rx = as_float(row, ["rvec_x", "rx", "rotvec_x"])
    ry = as_float(row, ["rvec_y", "ry", "rotvec_y"])
    rz = as_float(row, ["rvec_z", "rz", "rotvec_z"])

    if None not in (rx, ry, rz):
        R = rvec_to_R(np.array([rx, ry, rz], dtype=np.float64))
    else:
        roll = as_float(row, ["est_roll_deg", "roll_deg", "roll"])
        pitch = as_float(row, ["est_pitch_deg", "pitch_deg", "pitch"])
        yaw = as_float(row, ["est_yaw_deg", "yaw_deg", "yaw"])
        if None in (roll, pitch, yaw):
            raise RuntimeError("missing rotation columns")
        R = rpy_to_R(
            math.radians(roll),
            math.radians(pitch),
            math.radians(yaw),
        )

    return make_T(R, [x, y, z])


def load_ap01():
    poses = {}
    reason = ""
    meta = {}

    if AP01_META.exists():
        try:
            meta = json.loads(AP01_META.read_text())
            reason = "; ".join(
                f"{x.get('target_camera')}: {x.get('reason')}"
                for x in meta.get("failures", [])
            )
        except Exception:
            pass

    rows = read_csv(AP01_SOURCE)
    main = [r for r in rows if r.get("category") == "main_no_gt"]

    if rows or AP01_META.exists():
        poses["cam_edge_3"] = np.eye(4, dtype=np.float64)

    for row in main:
        cam = norm_cam(row.get("target_camera"))
        if cam:
            try:
                poses[cam] = parse_generic_pose(row)
            except Exception as exc:
                reason += f"; {cam}: {exc}"

    return poses, str(AP01_SOURCE), reason, meta


def load_pose_csv(path: Path, metadata_path: Path | None = None):
    poses = {}
    reason = ""
    meta = {}

    if metadata_path and metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            reason = str(meta.get("failure_reason", "") or "")
        except Exception as exc:
            reason = f"metadata parse failed: {exc}"

    for row in read_csv(path):
        cam = None
        for key in [
            "entity_id", "camera", "camera_name", "cam",
            "target_camera", "name",
        ]:
            cam = norm_cam(row.get(key))
            if cam:
                break
        if not cam:
            continue
        try:
            poses[cam] = parse_generic_pose(row)
        except Exception as exc:
            reason += f"; {cam}: {exc}"

    return poses, str(path), reason.strip("; "), meta


def pair_label(a, b):
    return f"{SHORT[a]}-{SHORT[b]}"


def angle_between_deg(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    value = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def transform_fields(prefix: str, T: np.ndarray) -> dict:
    rpy = R_to_rpy_deg(T[:3, :3])
    rvec = R_to_rvec(T[:3, :3])
    return {
        f"{prefix}_tx_m": T[0, 3],
        f"{prefix}_ty_m": T[1, 3],
        f"{prefix}_tz_m": T[2, 3],
        f"{prefix}_roll_deg": rpy[0],
        f"{prefix}_pitch_deg": rpy[1],
        f"{prefix}_yaw_deg": rpy[2],
        f"{prefix}_rvec_x": rvec[0],
        f"{prefix}_rvec_y": rvec[1],
        f"{prefix}_rvec_z": rvec[2],
    }


def evaluate_method(method, poses, gt, source, reason):
    available = sorted(poses)
    missing = sorted(set(STATIC_CAMERAS) - set(available))
    rows = []

    for a, b in PAIRS:
        T_gt = invT(gt[a]) @ gt[b]
        base = {
            "method": method,
            "pair": pair_label(a, b),
            "from_camera": a,
            "to_camera": b,
            "gt_baseline_m": float(np.linalg.norm(T_gt[:3, 3])),
            "available_cameras": ";".join(available),
            "missing_cameras": ";".join(missing),
            "source": source,
            "failure_reason": reason,
            **transform_fields("gt", T_gt),
        }

        unavailable = [cam for cam in (a, b) if cam not in poses]
        if unavailable:
            rows.append({
                **base,
                "status": "MISSING_CAMERA",
                "translation_error_cm": "",
                "rotation_error_deg": "",
                "baseline_error_cm": "",
                "direction_error_deg": "",
                "est_baseline_m": "",
                "note": "missing " + ",".join(unavailable),
            })
            continue

        T_est = invT(poses[a]) @ poses[b]
        est_baseline = float(np.linalg.norm(T_est[:3, 3]))
        gt_baseline = float(np.linalg.norm(T_gt[:3, 3]))

        rows.append({
            **base,
            "status": "OK",
            "translation_error_cm": (
                100.0 * float(
                    np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
                )
            ),
            "rotation_error_deg": float(rot_error_deg(T_est, T_gt)),
            "baseline_error_cm": (
                100.0 * abs(est_baseline - gt_baseline)
            ),
            "direction_error_deg": angle_between_deg(
                T_est[:3, 3], T_gt[:3, 3]
            ),
            "est_baseline_m": est_baseline,
            "note": "",
            **transform_fields("est", T_est),
        })

    return rows


def summarize(method, rows, poses, reason, meta):
    ok = [r for r in rows if r["status"] == "OK"]
    available = sorted(poses)
    missing = sorted(set(STATIC_CAMERAS) - set(available))
    n_pairs = len(ok)

    if n_pairs == len(PAIRS):
        status = "OK_FULL"
    elif n_pairs > 0:
        status = f"PARTIAL_{len(available)}_OF_4"
    else:
        status = "FAILED_NO_PAIR"

    if method == "AP03":
        meta_status = str(meta.get("status", ""))
        if "SCALE_WEAK" in meta_status:
            status = (
                "SCALE_WEAK_CHECK_REQUIRED"
                if n_pairs == len(PAIRS)
                else status + "_SCALE_WEAK"
            )
        if not reason:
            reason = str(meta.get("failure_reason", "") or "")

    result = {
        "method": method,
        "status": status,
        "camera_count": len(available),
        "available_cameras": ";".join(available),
        "missing_cameras": ";".join(missing),
        "pair_count_ok": n_pairs,
        "pair_count_total": len(PAIRS),
        "pair_coverage": n_pairs / len(PAIRS),
        "failure_reason": reason,
        "source": rows[0].get("source", "") if rows else "",
        "note": (
            "Metrics are computed over available pairs only. "
            "Partial means are not directly comparable with full six-pair means."
            if 0 < n_pairs < len(PAIRS)
            else reason
        ),
    }

    if not ok:
        for key in [
            "mean_pair_t_cm", "median_pair_t_cm", "max_pair_t_cm",
            "mean_pair_r_deg", "median_pair_r_deg", "max_pair_r_deg",
            "mean_baseline_error_cm", "mean_direction_error_deg",
            "worst_pair", "worst_pair_t_cm", "worst_pair_r_deg",
        ]:
            result[key] = ""
        return result

    ts = np.array(
        [float(r["translation_error_cm"]) for r in ok],
        dtype=float,
    )
    rs = np.array(
        [float(r["rotation_error_deg"]) for r in ok],
        dtype=float,
    )
    bs = np.array(
        [float(r["baseline_error_cm"]) for r in ok],
        dtype=float,
    )
    ds = np.array(
        [
            float(r["direction_error_deg"])
            for r in ok
            if math.isfinite(float(r["direction_error_deg"]))
        ],
        dtype=float,
    )
    worst = max(ok, key=lambda r: float(r["translation_error_cm"]))

    result.update({
        "mean_pair_t_cm": float(np.mean(ts)),
        "median_pair_t_cm": float(np.median(ts)),
        "max_pair_t_cm": float(np.max(ts)),
        "mean_pair_r_deg": float(np.mean(rs)),
        "median_pair_r_deg": float(np.median(rs)),
        "max_pair_r_deg": float(np.max(rs)),
        "mean_baseline_error_cm": float(np.mean(bs)),
        "mean_direction_error_deg": (
            float(np.mean(ds)) if len(ds) else ""
        ),
        "worst_pair": worst["pair"],
        "worst_pair_t_cm": worst["translation_error_cm"],
        "worst_pair_r_deg": worst["rotation_error_deg"],
    })
    return result


def fmt(value):
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def estimate_alignment(est, gt, cams):
    M = np.zeros((3, 3), dtype=float)
    for cam in cams:
        M += gt[cam][:3, :3] @ est[cam][:3, :3].T

    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = np.mean(
        np.vstack([
            gt[cam][:3, 3] - R @ est[cam][:3, 3]
            for cam in cams
        ]),
        axis=0,
    )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_alignment(A, T):
    out = np.eye(4)
    out[:3, :3] = A[:3, :3] @ T[:3, :3]
    out[:3, 3] = A[:3, :3] @ T[:3, 3] + A[:3, 3]
    return out


def main():
    global FINAL

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-root",
        type=Path,
        default=FINAL,
        help="Output directory for primary and secondary evaluation files.",
    )
    args = parser.parse_args()

    FINAL = args.final_root
    FINAL.mkdir(parents=True, exist_ok=True)

    gt = gt_static_camera_poses_ref_aruco(
        WORLD_SDF_MOVING_CAMERA,
        STATIC_CAMERAS,
        REF_MARKER_ENTITY,
    )

    method_data = {
        "AP01": load_ap01(),
        "AP02": load_pose_csv(AP02_SOURCE),
        "AP03": load_pose_csv(AP03_SOURCE, AP03_META),
    }

    all_rows = []
    summaries = []

    for method, (poses, source, reason, meta) in method_data.items():
        rows = evaluate_method(method, poses, gt, source, reason)
        all_rows.extend(rows)
        summaries.append(
            summarize(method, rows, poses, reason, meta)
        )

    detail_fields = [
        "method", "status", "pair", "from_camera", "to_camera",
        "translation_error_cm", "rotation_error_deg",
        "baseline_error_cm", "direction_error_deg",
        "gt_baseline_m", "est_baseline_m",
        "available_cameras", "missing_cameras",
        "source", "failure_reason", "note",
    ]
    summary_fields = [
        "method", "status", "camera_count",
        "available_cameras", "missing_cameras",
        "pair_count_ok", "pair_count_total", "pair_coverage",
        "mean_pair_t_cm", "median_pair_t_cm", "max_pair_t_cm",
        "mean_pair_r_deg", "median_pair_r_deg", "max_pair_r_deg",
        "mean_baseline_error_cm", "mean_direction_error_deg",
        "worst_pair", "worst_pair_t_cm", "worst_pair_r_deg",
        "source", "failure_reason", "note",
    ]

    write_csv(
        FINAL / "BASELINE_FINAL_PAIRWISE_DETAIL.csv",
        all_rows,
        detail_fields,
    )
    write_csv(
        FINAL / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv",
        summaries,
        summary_fields,
    )

    for method in ["AP01", "AP02", "AP03"]:
        method_root = FINAL / method
        method_rows = [r for r in all_rows if r["method"] == method]
        method_summary = [
            r for r in summaries if r["method"] == method
        ]
        write_csv(
            method_root / f"{method}_PAIRWISE_RESULT.csv",
            method_rows,
            detail_fields,
        )
        write_csv(
            method_root / f"{method}_PAIRWISE_SUMMARY.csv",
            method_summary,
            summary_fields,
        )

    lines = [
        "PARTIAL-AWARE PAIRWISE STATIC-CAMERA EVALUATION",
        "================================================",
        "",
        (
            "OK_FULL requires all four cameras and all six camera pairs. "
            "PARTIAL results retain every evaluable camera pair."
        ),
        "",
        "Summary:",
    ]

    for row in summaries:
        lines.append(
            f"- {row['method']}: {row['status']} | "
            f"cameras={row['camera_count']}/4 | "
            f"pairs={row['pair_count_ok']}/6 | "
            f"mean={fmt(row['mean_pair_t_cm'])} cm / "
            f"{fmt(row['mean_pair_r_deg'])} deg | "
            f"missing={row['missing_cameras'] or '-'}"
        )
        if row.get("failure_reason"):
            lines.append(f"  reason: {row['failure_reason']}")

    lines += ["", "Pairwise detail:"]
    for method in ["AP01", "AP02", "AP03"]:
        lines.append("")
        lines.append(method)
        for row in [r for r in all_rows if r["method"] == method]:
            if row["status"] == "OK":
                lines.append(
                    f"  {row['pair']}: "
                    f"{fmt(row['translation_error_cm'])} cm / "
                    f"{fmt(row['rotation_error_deg'])} deg"
                )
            else:
                lines.append(
                    f"  {row['pair']}: {row['status']} "
                    f"({row.get('note', '')})"
                )

    primary_report = FINAL / "BASELINE_FINAL_CLEAN_COMPARISON.txt"
    primary_report.write_text("\n".join(lines) + "\n")

    # Secondary partial-aware evaluation.
    secondary_detail = []
    secondary_summary = []
    secondary_report = [
        "PARTIAL-AWARE SECONDARY REF14/WORLD CAMERA-MAP EVALUATION",
        "==========================================================",
        "",
        (
            "SE(3) alignment is evaluated only when at least three "
            "static cameras are available."
        ),
        (
            "Two-camera results remain available in the primary "
            "pairwise evaluation but are not independently aligned here."
        ),
    ]

    for method, (poses, source, reason, meta) in method_data.items():
        cams = sorted(poses)
        missing = sorted(set(STATIC_CAMERAS) - set(cams))
        secondary_report += ["", method, "-" * len(method)]
        secondary_report.append(f"available cameras: {cams}")
        secondary_report.append(f"missing cameras: {missing}")

        if len(cams) < 3:
            status = (
                f"PARTIAL_{len(cams)}_OF_4_NO_SE3_ALIGNMENT"
                if cams
                else "FAILED_NO_STATIC_CAMERAS"
            )
            secondary_summary.append({
                "method": method,
                "status": status,
                "alignment": "NOT_RUN_INSUFFICIENT_CAMERAS",
                "source_file": source,
                "camera_count": len(cams),
                "available_cameras": ";".join(cams),
                "missing_cameras": ";".join(missing),
                "failure_reason": reason,
                "mean_translation_error_cm": "",
                "median_translation_error_cm": "",
                "max_translation_error_cm": "",
                "mean_rotation_error_deg": "",
                "median_rotation_error_deg": "",
                "max_rotation_error_deg": "",
            })
            for cam in cams:
                secondary_detail.append({
                    "method": method,
                    "camera": cam,
                    "status": "AVAILABLE_NOT_ALIGNED",
                    "alignment": "NOT_RUN_INSUFFICIENT_CAMERAS",
                    "source_file": source,
                    "translation_error_cm": "",
                    "rotation_error_deg": "",
                })
            secondary_report.append(f"status: {status}")
            secondary_report.append(
                "Use the retained primary pairwise result for this subset."
            )
            continue

        A = estimate_alignment(poses, gt, cams)
        t_errors = []
        r_errors = []
        status = "OK_FULL" if len(cams) == 4 else f"PARTIAL_{len(cams)}_OF_4"

        for cam in cams:
            aligned = apply_alignment(A, poses[cam])
            t_error = 100.0 * float(
                np.linalg.norm(
                    aligned[:3, 3] - gt[cam][:3, 3]
                )
            )
            r_error = float(
                rot_error_deg(aligned, gt[cam])
            )
            t_errors.append(t_error)
            r_errors.append(r_error)

            secondary_detail.append({
                "method": method,
                "camera": cam,
                "status": "OK",
                "alignment": "SE3_available_static_cameras_no_scale",
                "source_file": source,
                "translation_error_cm": t_error,
                "rotation_error_deg": r_error,
                "aligned_est_x_m": aligned[0, 3],
                "aligned_est_y_m": aligned[1, 3],
                "aligned_est_z_m": aligned[2, 3],
                "gt_x_m": gt[cam][0, 3],
                "gt_y_m": gt[cam][1, 3],
                "gt_z_m": gt[cam][2, 3],
            })

            secondary_report.append(
                f"- {cam}: {t_error:.3f} cm / "
                f"{r_error:.3f} deg"
            )

        secondary_summary.append({
            "method": method,
            "status": status,
            "alignment": "SE3_available_static_cameras_no_scale",
            "source_file": source,
            "camera_count": len(cams),
            "available_cameras": ";".join(cams),
            "missing_cameras": ";".join(missing),
            "failure_reason": reason,
            "mean_translation_error_cm": float(np.mean(t_errors)),
            "median_translation_error_cm": float(np.median(t_errors)),
            "max_translation_error_cm": float(np.max(t_errors)),
            "mean_rotation_error_deg": float(np.mean(r_errors)),
            "median_rotation_error_deg": float(np.median(r_errors)),
            "max_rotation_error_deg": float(np.max(r_errors)),
        })

        secondary_report.append(
            f"summary: {status} | mean "
            f"{np.mean(t_errors):.3f} cm / "
            f"{np.mean(r_errors):.3f} deg"
        )

    write_csv(
        FINAL / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv",
        secondary_detail,
        [
            "method", "camera", "status", "alignment",
            "source_file", "translation_error_cm",
            "rotation_error_deg",
        ],
    )
    write_csv(
        FINAL / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv",
        secondary_summary,
        [
            "method", "status", "alignment", "source_file",
            "camera_count", "available_cameras", "missing_cameras",
            "failure_reason",
            "mean_translation_error_cm",
            "median_translation_error_cm",
            "max_translation_error_cm",
            "mean_rotation_error_deg",
            "median_rotation_error_deg",
            "max_rotation_error_deg",
        ],
    )

    (
        FINAL / "SECONDARY_REF14_WORLD_CAMERA_MAP_EVALUATION.txt"
    ).write_text("\n".join(secondary_report) + "\n")

    (
        FINAL / "SECONDARY_REF14_WORLD_CAMERA_MAP_METADATA.json"
    ).write_text(json.dumps({
        "evaluation": "partial_aware_secondary_ref14_world_map",
        "minimum_cameras_for_se3_alignment": 3,
        "primary_metric": "pairwise_static_camera_extrinsics",
        "summary_rows": secondary_summary,
    }, indent=2) + "\n")

    (
        FINAL / "_PARTIAL_AWARE_EVALUATION_MANIFEST.json"
    ).write_text(json.dumps({
        "primary_summary": summaries,
        "secondary_summary": secondary_summary,
    }, indent=2) + "\n")

    print(primary_report.read_text())
    print(
        (
            FINAL / "SECONDARY_REF14_WORLD_CAMERA_MAP_EVALUATION.txt"
        ).read_text()
    )


if __name__ == "__main__":
    main()

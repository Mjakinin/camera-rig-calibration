#!/usr/bin/env python3
"""
Common pairwise static-camera extrinsic evaluator.

Main metric:
For each static camera pair i->j:
    T_i_j = inv(T_frame_cam_i) @ T_frame_cam_j

Errors:
    translation_error_cm = || t_est - t_gt || * 100
    rotation_error_deg   = angle(R_gt^T R_est)
    baseline_error_cm    = abs(||t_est|| - ||t_gt||) * 100
    direction_error_deg  = angle(t_est, t_gt)

No GT alignment is used. Global frame cancels in the pairwise transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.constants import STATIC_CAMERAS, WORLD_SDF_MOVING_CAMERA, REF_MARKER_ENTITY
from _shared.common.geometry import make_T, invT, rpy_to_R, rvec_to_R, R_to_rpy_deg, R_to_rvec, rot_error_deg
from _shared.common.sdf_utils import gt_static_camera_poses_ref_aruco


AP01_SOURCE = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference/final_extrinsics_summary.csv")
AP02_SOURCE = Path("results/bus_real_data/02_ref_marker_graph_ba/08_final_results/ap02_with_moving_static_camera_poses_ref_marker.csv")
AP03_SOURCE = Path("results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv")
AP03_META = Path("results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json")

FINAL_ROOT = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT")

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


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def f3(x) -> str:
    if x == "" or x is None:
        return "-"
    try:
        if not math.isfinite(float(x)):
            return "-"
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def f2(x) -> str:
    if x == "" or x is None:
        return "-"
    try:
        if not math.isfinite(float(x)):
            return "-"
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    vals = sorted(float(x) for x in xs)
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def T_from_rvec_pose_row(row: dict) -> np.ndarray:
    rvec = np.array([float(row["rvec_x"]), float(row["rvec_y"]), float(row["rvec_z"])], dtype=np.float64)
    t = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], dtype=np.float64)
    return make_T(rvec_to_R(rvec), t)


def T_from_ap01_est_row(row: dict) -> np.ndarray:
    return make_T(
        rpy_to_R(
            math.radians(float(row["est_roll_deg"])),
            math.radians(float(row["est_pitch_deg"])),
            math.radians(float(row["est_yaw_deg"])),
        ),
        [float(row["est_x_m"]), float(row["est_y_m"]), float(row["est_z_m"])],
    )


def load_ap01_camera_map() -> tuple[dict[str, np.ndarray], dict]:
    if not AP01_SOURCE.exists():
        raise RuntimeError(f"Missing AP01 source: {AP01_SOURCE}")

    rows = read_csv(AP01_SOURCE)
    main = [r for r in rows if r.get("category") == "main_no_gt"]
    by_cam = {r.get("target_camera"): r for r in main if r.get("target_camera")}

    poses = {"cam_edge_3": np.eye(4, dtype=np.float64)}
    for cam in STATIC_CAMERAS:
        if cam == "cam_edge_3":
            continue
        if cam not in by_cam:
            raise RuntimeError(f"AP01 missing main_no_gt target row for {cam}")
        poses[cam] = T_from_ap01_est_row(by_cam[cam])

    return poses, {
        "method": "AP01",
        "source": str(AP01_SOURCE),
        "pose_frame": "cam_edge_3_local",
        "note": "AP01 pairwise uses cam3-local camera map; no Ref14/world GT anchor and no map alignment.",
    }


def load_pose_csv_camera_map(path: Path, method: str) -> tuple[dict[str, np.ndarray], dict]:
    if not path.exists():
        raise RuntimeError(f"Missing {method} source: {path}")

    rows = read_csv(path)
    poses = {}
    for r in rows:
        cam = r.get("entity_id") or r.get("camera") or r.get("cam") or ""
        if cam in STATIC_CAMERAS:
            poses[cam] = T_from_rvec_pose_row(r)

    missing = [cam for cam in STATIC_CAMERAS if cam not in poses]
    if missing:
        raise RuntimeError(f"{method} missing static cameras: {missing}")

    return poses, {
        "method": method,
        "source": str(path),
        "pose_frame": "method_local_or_metric_frame",
        "note": "",
    }


def load_gt_camera_map() -> tuple[dict[str, np.ndarray], dict]:
    poses = gt_static_camera_poses_ref_aruco(
        WORLD_SDF_MOVING_CAMERA,
        STATIC_CAMERAS,
        REF_MARKER_ENTITY,
    )
    return poses, {
        "method": "GT",
        "source": str(WORLD_SDF_MOVING_CAMERA),
        "pose_frame": "GT_ref14",
    }


def pair_label(a: str, b: str) -> str:
    return f"{SHORT[a]}-{SHORT[b]}"


def pose_compact(T: np.ndarray) -> str:
    t = T[:3, 3]
    r = R_to_rpy_deg(T[:3, :3])
    return f"t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}] rpy=[{r[0]:.2f},{r[1]:.2f},{r[2]:.2f}]"


def eval_method_pairwise(method: str, poses: dict[str, np.ndarray], gt: dict[str, np.ndarray], source: str, note: str) -> list[dict]:
    rows = []
    for a, b in PAIRS:
        T_est = invT(poses[a]) @ poses[b]
        T_gt = invT(gt[a]) @ gt[b]

        te_cm = 100.0 * float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
        re_deg = float(rot_error_deg(T_est, T_gt))
        baseline_est = float(np.linalg.norm(T_est[:3, 3]))
        baseline_gt = float(np.linalg.norm(T_gt[:3, 3]))
        baseline_err_cm = 100.0 * abs(baseline_est - baseline_gt)
        direction_err_deg = angle_between_deg(T_est[:3, 3], T_gt[:3, 3])

        est_rpy = R_to_rpy_deg(T_est[:3, :3])
        gt_rpy = R_to_rpy_deg(T_gt[:3, :3])
        est_rvec = R_to_rvec(T_est[:3, :3])
        gt_rvec = R_to_rvec(T_gt[:3, :3])

        rows.append({
            "method": method,
            "status": "OK",
            "pair": pair_label(a, b),
            "from_camera": a,
            "to_camera": b,
            "translation_error_cm": te_cm,
            "rotation_error_deg": re_deg,
            "baseline_error_cm": baseline_err_cm,
            "direction_error_deg": direction_err_deg,
            "gt_baseline_m": baseline_gt,
            "est_baseline_m": baseline_est,
            "gt_tx_m": T_gt[0, 3],
            "gt_ty_m": T_gt[1, 3],
            "gt_tz_m": T_gt[2, 3],
            "gt_roll_deg": gt_rpy[0],
            "gt_pitch_deg": gt_rpy[1],
            "gt_yaw_deg": gt_rpy[2],
            "gt_rvec_x": gt_rvec[0],
            "gt_rvec_y": gt_rvec[1],
            "gt_rvec_z": gt_rvec[2],
            "est_tx_m": T_est[0, 3],
            "est_ty_m": T_est[1, 3],
            "est_tz_m": T_est[2, 3],
            "est_roll_deg": est_rpy[0],
            "est_pitch_deg": est_rpy[1],
            "est_yaw_deg": est_rpy[2],
            "est_rvec_x": est_rvec[0],
            "est_rvec_y": est_rvec[1],
            "est_rvec_z": est_rvec[2],
            "source": source,
            "note": note,
        })
    return rows


def failed_rows(method: str, reason: str) -> list[dict]:
    return [{
        "method": method,
        "status": "FAILED",
        "pair": pair_label(a, b),
        "from_camera": a,
        "to_camera": b,
        "translation_error_cm": "",
        "rotation_error_deg": "",
        "baseline_error_cm": "",
        "direction_error_deg": "",
        "gt_baseline_m": "",
        "est_baseline_m": "",
        "gt_tx_m": "", "gt_ty_m": "", "gt_tz_m": "",
        "gt_roll_deg": "", "gt_pitch_deg": "", "gt_yaw_deg": "",
        "gt_rvec_x": "", "gt_rvec_y": "", "gt_rvec_z": "",
        "est_tx_m": "", "est_ty_m": "", "est_tz_m": "",
        "est_roll_deg": "", "est_pitch_deg": "", "est_yaw_deg": "",
        "est_rvec_x": "", "est_rvec_y": "", "est_rvec_z": "",
        "source": "",
        "note": reason,
    } for a, b in PAIRS]


def summarize(method: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "OK"]
    if len(ok) != len(PAIRS):
        return {
            "method": method,
            "status": "FAILED",
            "mean_pair_t_cm": "",
            "median_pair_t_cm": "",
            "max_pair_t_cm": "",
            "mean_pair_r_deg": "",
            "median_pair_r_deg": "",
            "max_pair_r_deg": "",
            "mean_baseline_error_cm": "",
            "mean_direction_error_deg": "",
            "worst_pair": "",
            "worst_pair_t_cm": "",
            "worst_pair_r_deg": "",
            "source": rows[0].get("source", "") if rows else "",
            "note": rows[0].get("note", "missing rows") if rows else "missing rows",
        }

    ts = [float(r["translation_error_cm"]) for r in ok]
    rs = [float(r["rotation_error_deg"]) for r in ok]
    bs = [float(r["baseline_error_cm"]) for r in ok]
    ds = [float(r["direction_error_deg"]) for r in ok if math.isfinite(float(r["direction_error_deg"]))]
    worst = max(ok, key=lambda r: float(r["translation_error_cm"]))

    return {
        "method": method,
        "status": "OK",
        "mean_pair_t_cm": mean(ts),
        "median_pair_t_cm": median(ts),
        "max_pair_t_cm": max(ts),
        "mean_pair_r_deg": mean(rs),
        "median_pair_r_deg": median(rs),
        "max_pair_r_deg": max(rs),
        "mean_baseline_error_cm": mean(bs),
        "mean_direction_error_deg": mean(ds),
        "worst_pair": worst["pair"],
        "worst_pair_t_cm": worst["translation_error_cm"],
        "worst_pair_r_deg": worst["rotation_error_deg"],
        "source": ok[0].get("source", ""),
        "note": ok[0].get("note", ""),
    }


def text_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = [" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    out.append("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out)


def method_cell(rows_by_method_pair: dict, method: str, pair: str) -> str:
    r = rows_by_method_pair.get((method, pair))
    if not r or r.get("status") != "OK":
        return "FAILED"
    return (
        f"{f2(r['translation_error_cm'])}cm/"
        f"{f2(r['rotation_error_deg'])}deg "
        f"(b {f2(r['baseline_error_cm'])}cm, d {f2(r['direction_error_deg'])}deg)"
    )


def write_reports(final_root: Path, all_rows: list[dict], summaries: list[dict], manifest: dict) -> None:
    if final_root.exists():
        shutil.rmtree(final_root)
    ensure_dir(final_root)

    detail_fields = [
        "method", "status", "pair", "from_camera", "to_camera",
        "translation_error_cm", "rotation_error_deg",
        "baseline_error_cm", "direction_error_deg",
        "gt_baseline_m", "est_baseline_m",
        "gt_tx_m", "gt_ty_m", "gt_tz_m",
        "gt_roll_deg", "gt_pitch_deg", "gt_yaw_deg",
        "gt_rvec_x", "gt_rvec_y", "gt_rvec_z",
        "est_tx_m", "est_ty_m", "est_tz_m",
        "est_roll_deg", "est_pitch_deg", "est_yaw_deg",
        "est_rvec_x", "est_rvec_y", "est_rvec_z",
        "source", "note",
    ]
    summary_fields = [
        "method", "status",
        "mean_pair_t_cm", "median_pair_t_cm", "max_pair_t_cm",
        "mean_pair_r_deg", "median_pair_r_deg", "max_pair_r_deg",
        "mean_baseline_error_cm", "mean_direction_error_deg",
        "worst_pair", "worst_pair_t_cm", "worst_pair_r_deg",
        "source", "note",
    ]

    write_csv(final_root / "BASELINE_FINAL_PAIRWISE_DETAIL.csv", all_rows, detail_fields)
    write_csv(final_root / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv", summaries, summary_fields)

    for method in ["AP01", "AP02", "AP03"]:
        sub = ensure_dir(final_root / method)
        mrows = [r for r in all_rows if r["method"] == method]
        write_csv(sub / f"{method}_PAIRWISE_RESULT.csv", mrows, detail_fields)
        ms = [s for s in summaries if s["method"] == method]
        write_csv(sub / f"{method}_PAIRWISE_SUMMARY.csv", ms, summary_fields)

    gt_by_pair = {}
    rows_by_method_pair = {}
    for r in all_rows:
        rows_by_method_pair[(r["method"], r["pair"])] = r
        if r["status"] == "OK" and r["pair"] not in gt_by_pair:
            gt_by_pair[r["pair"]] = r

    summary_table_rows = []
    for s in summaries:
        summary_table_rows.append([
            s["method"],
            s["status"],
            f2(s["mean_pair_t_cm"]),
            f2(s["mean_pair_r_deg"]),
            s["worst_pair"],
            f2(s["worst_pair_t_cm"]),
            f2(s["worst_pair_r_deg"]),
        ])

    detail_table_rows = []
    for a, b in PAIRS:
        p = pair_label(a, b)
        gr = gt_by_pair.get(p)
        gt_cell = "-"
        if gr:
            gt_cell = f"baseline {f3(gr['gt_baseline_m'])} m"
        detail_table_rows.append([
            p,
            gt_cell,
            method_cell(rows_by_method_pair, "AP01", p),
            method_cell(rows_by_method_pair, "AP02", p),
            method_cell(rows_by_method_pair, "AP03", p),
        ])

    txt = []
    txt.append("BASELINE FINAL RESULTS — PAIRWISE STATIC-CAMERA EXTRINSICS")
    txt.append("==========================================================")
    txt.append("")
    txt.append("Metric:")
    txt.append("T_i_j = inv(T_frame_cam_i) @ T_frame_cam_j.")
    txt.append("t_error_cm = ||t_est - t_gt|| * 100.")
    txt.append("r_error_deg = angle(R_gt^T R_est).")
    txt.append("baseline_error_cm = abs(||t_est|| - ||t_gt||) * 100.")
    txt.append("direction_error_deg = angle(t_est, t_gt).")
    txt.append("No GT map alignment is used for the main metric.")
    txt.append("")
    txt.append("Summary:")
    txt.append(text_table(
        ["method", "status", "mean_t_cm", "mean_r_deg", "worst_pair", "worst_t_cm", "worst_r_deg"],
        summary_table_rows,
    ))
    txt.append("")
    txt.append("Pairwise detail: t_error/r_error, plus baseline and direction error.")
    txt.append(text_table(
        ["pair", "GT", "AP01", "AP02", "AP03"],
        detail_table_rows,
    ))
    txt.append("")
    txt.append("Full estimated and GT pair transforms are in BASELINE_FINAL_PAIRWISE_DETAIL.csv.")
    txt.append("AP03 uses marker-size-only scale; no SDF marker map is used as method input.")
    txt.append("")

    (final_root / "BASELINE_FINAL_CLEAN_COMPARISON.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    (final_root / "_RECALCULATED_FROM_GLOBAL_RERUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (final_root / "README.txt").write_text(
        "Use BASELINE_FINAL_CLEAN_COMPARISON.txt for the concise report.\n"
        "Use BASELINE_FINAL_PAIRWISE_DETAIL.csv for full pair transforms and errors.\n",
        encoding="utf-8",
    )

    print((final_root / "BASELINE_FINAL_CLEAN_COMPARISON.txt").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-root", type=Path, default=FINAL_ROOT)
    args = ap.parse_args()

    gt, gt_meta = load_gt_camera_map()

    methods = []
    all_rows = []
    manifest = {
        "mode": "PAIRWISE_STATIC_CAMERA_EXTRINSIC_EVALUATION",
        "gt_source": gt_meta,
        "methods": {},
        "forbidden_method_inputs": [
            "GT camera poses during method estimation",
            "GT marker poses during method estimation",
            "SDF marker map during AP03 method estimation",
            "ablation/res snapshots for baseline final",
        ],
    }

    loaders = [
        ("AP01", load_ap01_camera_map),
        ("AP02", lambda: load_pose_csv_camera_map(AP02_SOURCE, "AP02")),
        ("AP03", lambda: load_pose_csv_camera_map(AP03_SOURCE, "AP03")),
    ]

    for method, loader in loaders:
        try:
            poses, meta = loader()
            rows = eval_method_pairwise(method, poses, gt, meta.get("source", ""), meta.get("note", ""))
            manifest["methods"][method] = {"status": "OK", **meta}
        except Exception as exc:
            rows = failed_rows(method, str(exc))
            manifest["methods"][method] = {"status": "FAILED", "reason": str(exc)}
        all_rows.extend(rows)

    summaries = [summarize(m, [r for r in all_rows if r["method"] == m]) for m in ["AP01", "AP02", "AP03"]]

    if AP03_META.exists():
        try:
            ap03_meta = json.loads(AP03_META.read_text())
            manifest["methods"]["AP03"]["scale_metadata"] = ap03_meta

            ap03_scale_status = str(ap03_meta.get("status", ""))
            if ap03_scale_status and ap03_scale_status != "OK":
                for r in all_rows:
                    if r.get("method") == "AP03":
                        r["status"] = ap03_scale_status
                        r["note"] = "AP03 marker-size scale is unstable; see AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"

                summaries = [
                    summarize(m, [r for r in all_rows if r["method"] == m])
                    for m in ["AP01", "AP02", "AP03"]
                ]
                for ss in summaries:
                    if ss["method"] == "AP03":
                        ss["status"] = ap03_scale_status
                        ss["note"] = "AP03 marker-size scale is unstable; AP03 values are diagnostic only."
        except Exception:
            pass

    write_reports(args.final_root, all_rows, summaries, manifest)


if __name__ == "__main__":
    main()

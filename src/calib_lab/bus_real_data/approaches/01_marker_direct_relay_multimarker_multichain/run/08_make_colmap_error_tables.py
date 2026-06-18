#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median


def f6(x):
    if x == "" or x is None:
        return ""
    return f"{float(x):.6f}"


def f3(x):
    if x == "" or x is None:
        return ""
    return f"{float(x):.3f}"


def rmse(vals):
    vals = list(vals)
    if not vals:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("[OK] wrote:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-dir",
        default="results/bus_real_data/04_colmap_moving_sequence/sim3_eval_vs_gt",
    )
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    src = eval_dir / "sim3_aligned_trajectory_errors.csv"
    out_dir = eval_dir / "readable_tables"

    if not src.exists():
        raise RuntimeError(f"Missing source CSV: {src}")

    rows = []
    with src.open() as f:
        for r in csv.DictReader(f):
            frame = int(r["frame"])

            ax = float(r["aligned_x"])
            ay = float(r["aligned_y"])
            az = float(r["aligned_z"])

            gx = float(r["gt_x"])
            gy = float(r["gt_y"])
            gz = float(r["gt_z"])

            ex = ax - gx
            ey = ay - gy
            ez = az - gz
            pos_err = float(r["position_error_m"])

            rows.append({
                "frame": frame,
                "image_name": r["image_name"],

                "aligned_x": ax,
                "gt_x": gx,
                "err_x_m": ex,
                "abs_err_x_m": abs(ex),

                "aligned_y": ay,
                "gt_y": gy,
                "err_y_m": ey,
                "abs_err_y_m": abs(ey),

                "aligned_z": az,
                "gt_z": gz,
                "err_z_m": ez,
                "abs_err_z_m": abs(ez),

                "position_error_m": pos_err,
                "position_error_cm": pos_err * 100.0,

                "gt_roll": float(r["gt_roll"]),
                "gt_pitch": float(r["gt_pitch"]),
                "gt_yaw": float(r["gt_yaw"]),

                "raw_colmap_x": float(r["colmap_x"]),
                "raw_colmap_y": float(r["colmap_y"]),
                "raw_colmap_z": float(r["colmap_z"]),
            })

    rows = sorted(rows, key=lambda r: r["frame"])

    # 1) Main side-by-side comparison.
    side_rows = []
    for r in rows:
        side_rows.append({
            "frame": r["frame"],
            "image_name": r["image_name"],

            "colmap_aligned_x": f6(r["aligned_x"]),
            "gt_x": f6(r["gt_x"]),
            "err_x_cm": f3(r["err_x_m"] * 100.0),

            "colmap_aligned_y": f6(r["aligned_y"]),
            "gt_y": f6(r["gt_y"]),
            "err_y_cm": f3(r["err_y_m"] * 100.0),

            "colmap_aligned_z": f6(r["aligned_z"]),
            "gt_z": f6(r["gt_z"]),
            "err_z_cm": f3(r["err_z_m"] * 100.0),

            "total_pos_err_cm": f3(r["position_error_cm"]),
            "gt_roll": f6(r["gt_roll"]),
            "gt_pitch": f6(r["gt_pitch"]),
            "gt_yaw": f6(r["gt_yaw"]),
        })

    side_fields = [
        "frame", "image_name",
        "colmap_aligned_x", "gt_x", "err_x_cm",
        "colmap_aligned_y", "gt_y", "err_y_cm",
        "colmap_aligned_z", "gt_z", "err_z_cm",
        "total_pos_err_cm",
        "gt_roll", "gt_pitch", "gt_yaw",
    ]

    write_csv(out_dir / "01_side_by_side_xyz_errors.csv", side_fields, side_rows)

    # 2) X-only table.
    x_rows = []
    for r in rows:
        x_rows.append({
            "frame": r["frame"],
            "image_name": r["image_name"],
            "colmap_aligned_x": f6(r["aligned_x"]),
            "gt_x": f6(r["gt_x"]),
            "err_x_m": f6(r["err_x_m"]),
            "err_x_cm": f3(r["err_x_m"] * 100.0),
            "abs_err_x_cm": f3(r["abs_err_x_m"] * 100.0),
            "total_pos_err_cm": f3(r["position_error_cm"]),
        })

    x_fields = [
        "frame", "image_name",
        "colmap_aligned_x", "gt_x",
        "err_x_m", "err_x_cm", "abs_err_x_cm",
        "total_pos_err_cm",
    ]

    write_csv(out_dir / "02_x_axis_comparison.csv", x_fields, x_rows)

    # 3) Y-only table.
    y_rows = []
    for r in rows:
        y_rows.append({
            "frame": r["frame"],
            "image_name": r["image_name"],
            "colmap_aligned_y": f6(r["aligned_y"]),
            "gt_y": f6(r["gt_y"]),
            "err_y_m": f6(r["err_y_m"]),
            "err_y_cm": f3(r["err_y_m"] * 100.0),
            "abs_err_y_cm": f3(r["abs_err_y_m"] * 100.0),
            "total_pos_err_cm": f3(r["position_error_cm"]),
        })

    y_fields = [
        "frame", "image_name",
        "colmap_aligned_y", "gt_y",
        "err_y_m", "err_y_cm", "abs_err_y_cm",
        "total_pos_err_cm",
    ]

    write_csv(out_dir / "03_y_axis_comparison.csv", y_fields, y_rows)

    # 4) Z-only table.
    z_rows = []
    for r in rows:
        z_rows.append({
            "frame": r["frame"],
            "image_name": r["image_name"],
            "colmap_aligned_z": f6(r["aligned_z"]),
            "gt_z": f6(r["gt_z"]),
            "err_z_m": f6(r["err_z_m"]),
            "err_z_cm": f3(r["err_z_m"] * 100.0),
            "abs_err_z_cm": f3(r["abs_err_z_m"] * 100.0),
            "total_pos_err_cm": f3(r["position_error_cm"]),
        })

    z_fields = [
        "frame", "image_name",
        "colmap_aligned_z", "gt_z",
        "err_z_m", "err_z_cm", "abs_err_z_cm",
        "total_pos_err_cm",
    ]

    write_csv(out_dir / "04_z_axis_comparison.csv", z_fields, z_rows)

    # 5) Sorted by total error.
    sorted_rows = sorted(rows, key=lambda r: r["position_error_m"], reverse=True)
    err_rows = []
    for r in sorted_rows:
        err_rows.append({
            "rank": len(err_rows) + 1,
            "frame": r["frame"],
            "image_name": r["image_name"],
            "total_pos_err_cm": f3(r["position_error_cm"]),
            "err_x_cm": f3(r["err_x_m"] * 100.0),
            "err_y_cm": f3(r["err_y_m"] * 100.0),
            "err_z_cm": f3(r["err_z_m"] * 100.0),
            "colmap_aligned_xyz": f"{f6(r['aligned_x'])} {f6(r['aligned_y'])} {f6(r['aligned_z'])}",
            "gt_xyz": f"{f6(r['gt_x'])} {f6(r['gt_y'])} {f6(r['gt_z'])}",
        })

    err_fields = [
        "rank", "frame", "image_name",
        "total_pos_err_cm",
        "err_x_cm", "err_y_cm", "err_z_cm",
        "colmap_aligned_xyz", "gt_xyz",
    ]

    write_csv(out_dir / "05_errors_sorted_worst_first.csv", err_fields, err_rows)

    # 6) Raw COLMAP reference, clearly marked as not directly comparable.
    raw_rows = []
    for r in rows:
        raw_rows.append({
            "frame": r["frame"],
            "image_name": r["image_name"],
            "raw_colmap_x_not_directly_comparable": f6(r["raw_colmap_x"]),
            "raw_colmap_y_not_directly_comparable": f6(r["raw_colmap_y"]),
            "raw_colmap_z_not_directly_comparable": f6(r["raw_colmap_z"]),
            "aligned_x": f6(r["aligned_x"]),
            "aligned_y": f6(r["aligned_y"]),
            "aligned_z": f6(r["aligned_z"]),
            "gt_x": f6(r["gt_x"]),
            "gt_y": f6(r["gt_y"]),
            "gt_z": f6(r["gt_z"]),
        })

    raw_fields = [
        "frame", "image_name",
        "raw_colmap_x_not_directly_comparable",
        "raw_colmap_y_not_directly_comparable",
        "raw_colmap_z_not_directly_comparable",
        "aligned_x", "aligned_y", "aligned_z",
        "gt_x", "gt_y", "gt_z",
    ]

    write_csv(out_dir / "06_raw_colmap_reference.csv", raw_fields, raw_rows)

    # 7) Summary.
    ex = [r["err_x_m"] for r in rows]
    ey = [r["err_y_m"] for r in rows]
    ez = [r["err_z_m"] for r in rows]
    pe = [r["position_error_m"] for r in rows]

    summary_rows = [
        {
            "metric": "num_evaluated_registered_frames",
            "value_m": len(rows),
            "value_cm": "",
        },
        {
            "metric": "x_rmse",
            "value_m": f6(rmse(ex)),
            "value_cm": f3(rmse(ex) * 100.0),
        },
        {
            "metric": "x_mean_abs",
            "value_m": f6(mean(abs(v) for v in ex)),
            "value_cm": f3(mean(abs(v) for v in ex) * 100.0),
        },
        {
            "metric": "x_median_abs",
            "value_m": f6(median(abs(v) for v in ex)),
            "value_cm": f3(median(abs(v) for v in ex) * 100.0),
        },
        {
            "metric": "x_max_abs",
            "value_m": f6(max(abs(v) for v in ex)),
            "value_cm": f3(max(abs(v) for v in ex) * 100.0),
        },
        {
            "metric": "y_rmse",
            "value_m": f6(rmse(ey)),
            "value_cm": f3(rmse(ey) * 100.0),
        },
        {
            "metric": "y_mean_abs",
            "value_m": f6(mean(abs(v) for v in ey)),
            "value_cm": f3(mean(abs(v) for v in ey) * 100.0),
        },
        {
            "metric": "y_median_abs",
            "value_m": f6(median(abs(v) for v in ey)),
            "value_cm": f3(median(abs(v) for v in ey) * 100.0),
        },
        {
            "metric": "y_max_abs",
            "value_m": f6(max(abs(v) for v in ey)),
            "value_cm": f3(max(abs(v) for v in ey) * 100.0),
        },
        {
            "metric": "z_rmse",
            "value_m": f6(rmse(ez)),
            "value_cm": f3(rmse(ez) * 100.0),
        },
        {
            "metric": "z_mean_abs",
            "value_m": f6(mean(abs(v) for v in ez)),
            "value_cm": f3(mean(abs(v) for v in ez) * 100.0),
        },
        {
            "metric": "z_median_abs",
            "value_m": f6(median(abs(v) for v in ez)),
            "value_cm": f3(median(abs(v) for v in ez) * 100.0),
        },
        {
            "metric": "z_max_abs",
            "value_m": f6(max(abs(v) for v in ez)),
            "value_cm": f3(max(abs(v) for v in ez) * 100.0),
        },
        {
            "metric": "total_position_rmse",
            "value_m": f6(rmse(pe)),
            "value_cm": f3(rmse(pe) * 100.0),
        },
        {
            "metric": "total_position_mean",
            "value_m": f6(mean(pe)),
            "value_cm": f3(mean(pe) * 100.0),
        },
        {
            "metric": "total_position_median",
            "value_m": f6(median(pe)),
            "value_cm": f3(median(pe) * 100.0),
        },
        {
            "metric": "total_position_max",
            "value_m": f6(max(pe)),
            "value_cm": f3(max(pe) * 100.0),
        },
    ]

    summary_fields = ["metric", "value_m", "value_cm"]
    write_csv(out_dir / "07_error_summary.csv", summary_fields, summary_rows)

    # 8) Human-readable TXT.
    txt = out_dir / "README_readable_tables.txt"
    txt.write_text(
        "Readable Sim(3) evaluation tables\n"
        "=================================\n\n"
        "Important:\n"
        "- Compare colmap_aligned_x/y/z with gt_x/y/z.\n"
        "- Raw COLMAP coordinates are not directly comparable before Sim(3) alignment.\n"
        "- Rotation is not evaluated here. gt_roll/pitch/yaw are only the commanded GT route angles.\n\n"
        "Files:\n"
        "01_side_by_side_xyz_errors.csv      Main readable table: aligned x next to gt x, etc.\n"
        "02_x_axis_comparison.csv            X-axis only.\n"
        "03_y_axis_comparison.csv            Y-axis only.\n"
        "04_z_axis_comparison.csv            Z-axis only.\n"
        "05_errors_sorted_worst_first.csv    Worst frames first.\n"
        "06_raw_colmap_reference.csv         Raw COLMAP coordinates for reference only.\n"
        "07_error_summary.csv                RMSE/mean/median/max summary.\n"
    )
    print("[OK] wrote:", txt)

    print()
    print("=== SUMMARY ===")
    for r in summary_rows:
        print(f"{r['metric']:32s} {r['value_m']} m   {r['value_cm']} cm")


if __name__ == "__main__":
    main()

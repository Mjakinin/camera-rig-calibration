#!/usr/bin/env python3

import csv
from pathlib import Path


IN_CSV = Path("results/bus_real_data/05_direct_static_pair_cam3_cam1/02_all_shared_marker_estimates_best_convention.csv")
OUT_DIR = Path("results/bus_real_data/05_direct_static_pair_cam3_cam1")
OUT_TXT = OUT_DIR / "05_pretty_direct_static_cam3_cam1_report.txt"
OUT_CSV = OUT_DIR / "05_pretty_direct_static_cam3_cam1_table.csv"


def as_float(row, key):
    return float(row[key])


def status_for(row):
    t_cm = as_float(row, "translation_error_cm")
    r_deg = as_float(row, "rotation_error_deg")
    root_d = as_float(row, "root_distance_m")
    target_d = as_float(row, "target_distance_m")

    if t_cm < 20.0 and r_deg < 3.0:
        return "GOOD_INLIER"
    if t_cm < 50.0 and r_deg < 8.0:
        return "WEAK_INLIER"
    if root_d > 5.0 or target_d > 5.0:
        return "REJECT_FAR_MARKER"
    return "REJECT_OUTLIER"


def main():
    rows = []
    with IN_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)

    rows = sorted(
        rows,
        key=lambda r: as_float(r, "translation_error_m") + 0.02 * as_float(r, "rotation_error_deg")
    )

    table_rows = []

    for r in rows:
        status = status_for(r)

        table_rows.append({
            "marker_id": r["marker_id"],
            "status": status,

            "estimated_cam1_in_cam3_x": f"{as_float(r, 'estimated_target_in_root_x'):.4f}",
            "gt_cam1_in_cam3_x": f"{as_float(r, 'gt_target_in_root_x'):.4f}",
            "err_x_cm": f"{(as_float(r, 'estimated_target_in_root_x') - as_float(r, 'gt_target_in_root_x')) * 100.0:.2f}",

            "estimated_cam1_in_cam3_y": f"{as_float(r, 'estimated_target_in_root_y'):.4f}",
            "gt_cam1_in_cam3_y": f"{as_float(r, 'gt_target_in_root_y'):.4f}",
            "err_y_cm": f"{(as_float(r, 'estimated_target_in_root_y') - as_float(r, 'gt_target_in_root_y')) * 100.0:.2f}",

            "estimated_cam1_in_cam3_z": f"{as_float(r, 'estimated_target_in_root_z'):.4f}",
            "gt_cam1_in_cam3_z": f"{as_float(r, 'gt_target_in_root_z'):.4f}",
            "err_z_cm": f"{(as_float(r, 'estimated_target_in_root_z') - as_float(r, 'gt_target_in_root_z')) * 100.0:.2f}",

            "estimated_roll_deg": f"{as_float(r, 'estimated_target_in_root_roll_deg'):.3f}",
            "gt_roll_deg": f"{as_float(r, 'gt_target_in_root_roll_deg'):.3f}",

            "estimated_pitch_deg": f"{as_float(r, 'estimated_target_in_root_pitch_deg'):.3f}",
            "gt_pitch_deg": f"{as_float(r, 'gt_target_in_root_pitch_deg'):.3f}",

            "estimated_yaw_deg": f"{as_float(r, 'estimated_target_in_root_yaw_deg'):.3f}",
            "gt_yaw_deg": f"{as_float(r, 'gt_target_in_root_yaw_deg'):.3f}",

            "translation_error_cm": f"{as_float(r, 'translation_error_cm'):.2f}",
            "rotation_error_deg": f"{as_float(r, 'rotation_error_deg'):.3f}",
            "root_distance_m": f"{as_float(r, 'root_distance_m'):.3f}",
            "target_distance_m": f"{as_float(r, 'target_distance_m'):.3f}",
        })

    with OUT_CSV.open("w", newline="") as f:
        fields = list(table_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)

    lines = []
    lines.append("Pretty direct static-to-static report")
    lines.append("====================================")
    lines.append("")
    lines.append("Pair: cam_edge_3 -> cam_edge_1")
    lines.append("")
    lines.append("Meaning:")
    lines.append("- estimated cam1 in cam3 = estimated relative pose of cam_edge_1 expressed in cam_edge_3 frame.")
    lines.append("- GT cam1 in cam3 = ground-truth relative pose from simulation, converted into optical-frame convention.")
    lines.append("- Errors are only for simulation evaluation.")
    lines.append("")

    for row in table_rows:
        lines.append(f"Marker {row['marker_id']} [{row['status']}]")
        lines.append("-" * 60)
        lines.append("Estimated pose: cam_edge_1 in cam_edge_3")
        lines.append(f"  position [m]: x={row['estimated_cam1_in_cam3_x']}, y={row['estimated_cam1_in_cam3_y']}, z={row['estimated_cam1_in_cam3_z']}")
        lines.append(f"  rotation [deg]: roll={row['estimated_roll_deg']}, pitch={row['estimated_pitch_deg']}, yaw={row['estimated_yaw_deg']}")
        lines.append("")
        lines.append("Ground Truth pose: cam_edge_1 in cam_edge_3")
        lines.append(f"  position [m]: x={row['gt_cam1_in_cam3_x']}, y={row['gt_cam1_in_cam3_y']}, z={row['gt_cam1_in_cam3_z']}")
        lines.append(f"  rotation [deg]: roll={row['gt_roll_deg']}, pitch={row['gt_pitch_deg']}, yaw={row['gt_yaw_deg']}")
        lines.append("")
        lines.append("Errors")
        lines.append(f"  position error: {row['translation_error_cm']} cm")
        lines.append(f"  rotation error: {row['rotation_error_deg']} deg")
        lines.append(f"  coordinate errors: dx={row['err_x_cm']} cm, dy={row['err_y_cm']} cm, dz={row['err_z_cm']} cm")
        lines.append(f"  marker distances: root={row['root_distance_m']} m, target={row['target_distance_m']} m")
        lines.append("")

    good = [r for r in table_rows if r["status"] in ["GOOD_INLIER", "WEAK_INLIER"]]
    rejected = [r for r in table_rows if r["status"].startswith("REJECT")]

    lines.append("Summary")
    lines.append("-------")
    lines.append(f"Total shared markers: {len(table_rows)}")
    lines.append(f"Kept as inlier candidates: {len(good)}")
    lines.append(f"Rejected: {len(rejected)}")
    lines.append("")
    lines.append("Recommended direct-static decision for now:")
    lines.append("- Use marker 1 as best single-marker sanity baseline.")
    lines.append("- Use marker 1 and marker 2 as weak inlier set for robust development.")
    lines.append("- Reject marker 5, 6, 7 for direct static aggregation.")
    lines.append("- Marker 8 is borderline and should not dominate the result.")
    lines.append("")
    lines.append("Why:")
    lines.append("- The far markers have large depth/pose instability although their IDs are visible.")
    lines.append("- Single planar ArUco marker pose is not reliable enough at long range for final extrinsics.")
    lines.append("- This motivates using nearby markers, multiple markers, CharUco/multi-marker boards, or robust outlier rejection.")

    OUT_TXT.write_text("\n".join(lines) + "\n")

    print("[OK] wrote:", OUT_TXT)
    print("[OK] wrote:", OUT_CSV)


if __name__ == "__main__":
    main()

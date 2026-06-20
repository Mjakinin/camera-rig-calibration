#!/usr/bin/env python3

import csv
import math
from pathlib import Path


ROOT = Path("results/bus_real_data/90_approach_comparison_ref_aruco/02_ref_marker_graph_ba")
COMBINED = Path("results/bus_real_data/90_approach_comparison_ref_aruco/combined")

CAM_CSV = ROOT / "ap02_static_cameras_ref_aruco_vs_gt.csv"
MARKER_CSV = ROOT / "ap02_markers_ref_aruco_vs_gt.csv"

OUT_TXT = ROOT / "AP02_FINAL_READABLE_REF_ARUCO_REPORT.txt"
OUT_MD = ROOT / "AP02_FINAL_READABLE_REF_ARUCO_REPORT.md"
OUT_CAM_SIMPLE = ROOT / "ap02_final_static_cameras_readable.csv"
OUT_MARKER_SIMPLE = ROOT / "ap02_final_marker_map_readable.csv"
OUT_FINAL_SUMMARY = COMBINED / "ap02_final_readable_summary.csv"

REF_MARKER_NAME = "aruco_ref_floor_14"


def read_csv(path):
    if not path.exists():
        raise RuntimeError(f"Missing input CSV: {path}")
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def f(row, key):
    return float(row[key])


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted([float(x) for x in xs])
    if not xs:
        return float("nan")
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def xyz(row, prefix):
    return (
        f(row, f"{prefix}_x_m"),
        f(row, f"{prefix}_y_m"),
        f(row, f"{prefix}_z_m"),
    )


def rpy(row, prefix):
    return (
        f(row, f"{prefix}_roll_deg"),
        f(row, f"{prefix}_pitch_deg"),
        f(row, f"{prefix}_yaw_deg"),
    )


def fmt_xyz(vals):
    return f"({vals[0]:+7.3f}, {vals[1]:+7.3f}, {vals[2]:+7.3f})"


def fmt_rpy(vals):
    return f"({vals[0]:+8.3f}, {vals[1]:+8.3f}, {vals[2]:+8.3f})"


def simplify_camera_rows(rows):
    out = []
    for r in rows:
        out.append({
            "camera": r["entity_id"],
            "trans_err_cm": fmt(r["translation_error_cm"], 3),
            "rot_err_deg": fmt(r["rotation_error_deg"], 3),
            "dx_cm": fmt(r["delta_x_cm"], 3),
            "dy_cm": fmt(r["delta_y_cm"], 3),
            "dz_cm": fmt(r["delta_z_cm"], 3),
            "est_x_m": fmt(r["est_ref_aruco_x_m"], 6),
            "est_y_m": fmt(r["est_ref_aruco_y_m"], 6),
            "est_z_m": fmt(r["est_ref_aruco_z_m"], 6),
            "gt_x_m": fmt(r["gt_ref_aruco_x_m"], 6),
            "gt_y_m": fmt(r["gt_ref_aruco_y_m"], 6),
            "gt_z_m": fmt(r["gt_ref_aruco_z_m"], 6),
            "est_roll_deg": fmt(r["est_ref_aruco_roll_deg"], 3),
            "est_pitch_deg": fmt(r["est_ref_aruco_pitch_deg"], 3),
            "est_yaw_deg": fmt(r["est_ref_aruco_yaw_deg"], 3),
            "gt_roll_deg": fmt(r["gt_ref_aruco_roll_deg"], 3),
            "gt_pitch_deg": fmt(r["gt_ref_aruco_pitch_deg"], 3),
            "gt_yaw_deg": fmt(r["gt_ref_aruco_yaw_deg"], 3),
        })
    return out


def simplify_marker_rows(rows):
    out = []
    for r in rows:
        out.append({
            "marker": r["entity_id"],
            "marker_id": r["marker_id"],
            "trans_err_cm": fmt(r["translation_error_cm"], 3),
            "rot_err_deg": fmt(r["rotation_error_deg"], 3),
            "dx_cm": fmt(r["delta_x_cm"], 3),
            "dy_cm": fmt(r["delta_y_cm"], 3),
            "dz_cm": fmt(r["delta_z_cm"], 3),
            "est_x_m": fmt(r["est_ref_aruco_x_m"], 6),
            "est_y_m": fmt(r["est_ref_aruco_y_m"], 6),
            "est_z_m": fmt(r["est_ref_aruco_z_m"], 6),
            "gt_x_m": fmt(r["gt_ref_aruco_x_m"], 6),
            "gt_y_m": fmt(r["gt_ref_aruco_y_m"], 6),
            "gt_z_m": fmt(r["gt_ref_aruco_z_m"], 6),
            "est_roll_deg": fmt(r["est_ref_aruco_roll_deg"], 3),
            "est_pitch_deg": fmt(r["est_ref_aruco_pitch_deg"], 3),
            "est_yaw_deg": fmt(r["est_ref_aruco_yaw_deg"], 3),
            "gt_roll_deg": fmt(r["gt_ref_aruco_roll_deg"], 3),
            "gt_pitch_deg": fmt(r["gt_ref_aruco_pitch_deg"], 3),
            "gt_yaw_deg": fmt(r["gt_ref_aruco_yaw_deg"], 3),
        })
    return out


def make_text_table(rows, headers, keys):
    data = []
    data.append(headers)

    for r in rows:
        data.append([str(r.get(k, "")) for k in keys])

    widths = [max(len(row[i]) for row in data) for i in range(len(headers))]

    lines = []
    header = " | ".join(data[0][i].ljust(widths[i]) for i in range(len(headers)))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header)
    lines.append(sep)

    for row in data[1:]:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    return "\n".join(lines)


def make_markdown_table(rows, headers, keys):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    COMBINED.mkdir(parents=True, exist_ok=True)

    cam_rows_raw = read_csv(CAM_CSV)
    marker_rows_raw = read_csv(MARKER_CSV)

    cam_rows = simplify_camera_rows(cam_rows_raw)
    marker_rows = simplify_marker_rows(marker_rows_raw)

    marker_rows_no_ref_raw = [r for r in marker_rows_raw if r["entity_id"] != REF_MARKER_NAME]
    marker_rows_no_ref = [r for r in marker_rows if r["marker"] != REF_MARKER_NAME]

    cam_t = [f(r, "translation_error_cm") for r in cam_rows_raw]
    cam_r = [f(r, "rotation_error_deg") for r in cam_rows_raw]

    marker_t_all = [f(r, "translation_error_cm") for r in marker_rows_raw]
    marker_r_all = [f(r, "rotation_error_deg") for r in marker_rows_raw]
    marker_t_no_ref = [f(r, "translation_error_cm") for r in marker_rows_no_ref_raw]
    marker_r_no_ref = [f(r, "rotation_error_deg") for r in marker_rows_no_ref_raw]

    worst_cams = sorted(cam_rows_raw, key=lambda r: f(r, "translation_error_cm"), reverse=True)
    worst_markers = sorted(marker_rows_no_ref_raw, key=lambda r: f(r, "translation_error_cm"), reverse=True)

    camera_fields = list(cam_rows[0].keys())
    marker_fields = list(marker_rows[0].keys())

    write_csv(OUT_CAM_SIMPLE, cam_rows, camera_fields)
    write_csv(OUT_MARKER_SIMPLE, marker_rows, marker_fields)

    final_summary = [{
        "approach": "AP02_ref_marker_graph_ba",
        "reference_frame": "aruco_marker_14",
        "final_variant": "graph_ba_with_moving_sparse",
        "camera_count": len(cam_rows),
        "camera_mean_translation_error_cm": fmt(mean(cam_t), 3),
        "camera_median_translation_error_cm": fmt(median(cam_t), 3),
        "camera_mean_rotation_error_deg": fmt(mean(cam_r), 3),
        "camera_median_rotation_error_deg": fmt(median(cam_r), 3),
        "marker_count_including_ref": len(marker_rows),
        "marker_count_excluding_ref": len(marker_rows_no_ref),
        "marker_mean_translation_error_cm_excluding_ref": fmt(mean(marker_t_no_ref), 3),
        "marker_median_translation_error_cm_excluding_ref": fmt(median(marker_t_no_ref), 3),
        "marker_mean_rotation_error_deg_excluding_ref": fmt(mean(marker_r_no_ref), 3),
        "marker_median_rotation_error_deg_excluding_ref": fmt(median(marker_r_no_ref), 3),
        "worst_camera_by_translation": worst_cams[0]["entity_id"],
        "worst_camera_translation_error_cm": fmt(worst_cams[0]["translation_error_cm"], 3),
        "worst_marker_by_translation": worst_markers[0]["entity_id"],
        "worst_marker_translation_error_cm": fmt(worst_markers[0]["translation_error_cm"], 3),
    }]
    write_csv(OUT_FINAL_SUMMARY, final_summary, list(final_summary[0].keys()))

    cam_table = make_text_table(
        cam_rows,
        headers=[
            "camera", "t_err_cm", "r_err_deg", "dX_cm", "dY_cm", "dZ_cm",
            "est_xyz_m", "gt_xyz_m", "est_rpy_deg", "gt_rpy_deg"
        ],
        keys=[
            "camera", "trans_err_cm", "rot_err_deg", "dx_cm", "dy_cm", "dz_cm",
            "est_xyz", "gt_xyz", "est_rpy", "gt_rpy"
        ],
    )

    # Add compact tuple columns only for text/markdown readability.
    for r in cam_rows:
        r["est_xyz"] = fmt_xyz((float(r["est_x_m"]), float(r["est_y_m"]), float(r["est_z_m"])))
        r["gt_xyz"] = fmt_xyz((float(r["gt_x_m"]), float(r["gt_y_m"]), float(r["gt_z_m"])))
        r["est_rpy"] = fmt_rpy((float(r["est_roll_deg"]), float(r["est_pitch_deg"]), float(r["est_yaw_deg"])))
        r["gt_rpy"] = fmt_rpy((float(r["gt_roll_deg"]), float(r["gt_pitch_deg"]), float(r["gt_yaw_deg"])))

    for r in marker_rows:
        r["est_xyz"] = fmt_xyz((float(r["est_x_m"]), float(r["est_y_m"]), float(r["est_z_m"])))
        r["gt_xyz"] = fmt_xyz((float(r["gt_x_m"]), float(r["gt_y_m"]), float(r["gt_z_m"])))
        r["est_rpy"] = fmt_rpy((float(r["est_roll_deg"]), float(r["est_pitch_deg"]), float(r["est_yaw_deg"])))
        r["gt_rpy"] = fmt_rpy((float(r["gt_roll_deg"]), float(r["gt_pitch_deg"]), float(r["gt_yaw_deg"])))

    cam_table = make_text_table(
        cam_rows,
        headers=["camera", "t_err_cm", "r_err_deg", "dX", "dY", "dZ", "est xyz [m]", "gt xyz [m]", "est rpy [deg]", "gt rpy [deg]"],
        keys=["camera", "trans_err_cm", "rot_err_deg", "dx_cm", "dy_cm", "dz_cm", "est_xyz", "gt_xyz", "est_rpy", "gt_rpy"],
    )

    marker_table = make_text_table(
        marker_rows,
        headers=["marker", "id", "t_err_cm", "r_err_deg", "dX", "dY", "dZ", "est xyz [m]", "gt xyz [m]"],
        keys=["marker", "marker_id", "trans_err_cm", "rot_err_deg", "dx_cm", "dy_cm", "dz_cm", "est_xyz", "gt_xyz"],
    )

    worst_cam_lines = []
    for r in worst_cams:
        worst_cam_lines.append(
            f"- {r['entity_id']}: {fmt(r['translation_error_cm'], 3)} cm, {fmt(r['rotation_error_deg'], 3)} deg"
        )

    worst_marker_lines = []
    for r in worst_markers[:5]:
        worst_marker_lines.append(
            f"- {r['entity_id']}: {fmt(r['translation_error_cm'], 3)} cm, {fmt(r['rotation_error_deg'], 3)} deg"
        )

    text = f"""AP02 FINAL EXTRINSICS / MARKER MAP REPORT
=========================================

Approach:
- AP02: Reference-ArUco marker-map graph bundle adjustment
- Selected variant: graph_ba_with_moving_sparse
- Reference frame: aruco_marker_14 / aruco_ref_floor_14
- All poses below are expressed relative to the reference ArUco marker.
- GT is used only for evaluation, not for estimation.

HIGH-LEVEL RESULT
-----------------
Static cameras:
- count: {len(cam_rows)}
- mean translation error:   {mean(cam_t):.3f} cm
- median translation error: {median(cam_t):.3f} cm
- mean rotation error:      {mean(cam_r):.3f} deg
- median rotation error:    {median(cam_r):.3f} deg

Marker map:
- markers including reference: {len(marker_rows)}
- markers excluding reference: {len(marker_rows_no_ref)}
- mean translation error excluding reference:   {mean(marker_t_no_ref):.3f} cm
- median translation error excluding reference: {median(marker_t_no_ref):.3f} cm
- mean rotation error excluding reference:      {mean(marker_r_no_ref):.3f} deg
- median rotation error excluding reference:    {median(marker_r_no_ref):.3f} deg

Worst camera by translation error:
{worst_cam_lines[0]}

Worst marker by translation error, excluding reference:
{worst_marker_lines[0]}

STATIC CAMERA EXTRINSICS VS GT, REF-ARUCO FRAME
-----------------------------------------------
{cam_table}

MARKER MAP VS GT, REF-ARUCO FRAME
---------------------------------
{marker_table}

CAMERA ERROR RANKING
--------------------
{chr(10).join(worst_cam_lines)}

TOP 5 MARKER ERROR RANKING, EXCLUDING REF MARKER
------------------------------------------------
{chr(10).join(worst_marker_lines)}

FILES
-----
Readable report:
- {OUT_TXT}

Readable CSV exports:
- {OUT_CAM_SIMPLE}
- {OUT_MARKER_SIMPLE}
- {OUT_FINAL_SUMMARY}

Raw full evaluation CSVs:
- {CAM_CSV}
- {MARKER_CSV}
"""

    OUT_TXT.write_text(text)

    md_cam_table = make_markdown_table(
        cam_rows,
        headers=["camera", "t err [cm]", "r err [deg]", "dX [cm]", "dY [cm]", "dZ [cm]", "est xyz [m]", "gt xyz [m]"],
        keys=["camera", "trans_err_cm", "rot_err_deg", "dx_cm", "dy_cm", "dz_cm", "est_xyz", "gt_xyz"],
    )

    md_marker_table = make_markdown_table(
        marker_rows,
        headers=["marker", "id", "t err [cm]", "r err [deg]", "dX [cm]", "dY [cm]", "dZ [cm]", "est xyz [m]", "gt xyz [m]"],
        keys=["marker", "marker_id", "trans_err_cm", "rot_err_deg", "dx_cm", "dy_cm", "dz_cm", "est_xyz", "gt_xyz"],
    )

    md = f"""# AP02 Final Ref-ArUco Evaluation

**Approach:** AP02 reference-marker graph BA  
**Selected variant:** `graph_ba_with_moving_sparse`  
**Reference frame:** `aruco_marker_14 / aruco_ref_floor_14`

## Summary

| metric | value |
|---|---:|
| camera count | {len(cam_rows)} |
| camera mean translation error | {mean(cam_t):.3f} cm |
| camera median translation error | {median(cam_t):.3f} cm |
| camera mean rotation error | {mean(cam_r):.3f} deg |
| camera median rotation error | {median(cam_r):.3f} deg |
| markers excluding ref | {len(marker_rows_no_ref)} |
| marker mean translation error excluding ref | {mean(marker_t_no_ref):.3f} cm |
| marker median translation error excluding ref | {median(marker_t_no_ref):.3f} cm |
| marker mean rotation error excluding ref | {mean(marker_r_no_ref):.3f} deg |
| marker median rotation error excluding ref | {median(marker_r_no_ref):.3f} deg |

## Static camera extrinsics vs GT

{md_cam_table}

## Marker map vs GT

{md_marker_table}
"""

    OUT_MD.write_text(md)

    print("[OK] wrote readable AP02 report:")
    print(f"- {OUT_TXT}")
    print(f"- {OUT_MD}")
    print(f"- {OUT_CAM_SIMPLE}")
    print(f"- {OUT_MARKER_SIMPLE}")
    print(f"- {OUT_FINAL_SUMMARY}")
    print()
    print(text)


if __name__ == "__main__":
    main()

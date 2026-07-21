#!/usr/bin/env python3

from pathlib import Path
import csv
import json

from ap02_common import AP02_ROOT, ensure_dir, read_csv, write_csv


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text()


def parse_summary_value(text, prefix):
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def copy_csv_rows(src, dst):
    rows = read_csv(src)
    if rows:
        fields = list(rows[0].keys())
    else:
        fields = []
    write_csv(dst, rows, fields)
    return rows


def main():
    out = ensure_dir(AP02_ROOT / "08_final_results")

    single_ref_report = read_text(AP02_ROOT / "04_single_ref_marker_pnp" / "single_ref_marker_report.txt")
    static_graph_report = read_text(AP02_ROOT / "05_graph_initialization" / "static_only" / "graph_connectivity_report.txt")
    moving_graph_report = read_text(AP02_ROOT / "05_graph_initialization" / "with_moving" / "graph_connectivity_report.txt")
    static_ba_summary = read_text(AP02_ROOT / "07_graph_ba" / "static_only" / "ba_summary.txt")
    moving_ba_summary = read_text(AP02_ROOT / "07_graph_ba" / "with_moving" / "ba_summary.txt")

    static_final_src = AP02_ROOT / "07_graph_ba" / "static_only" / "optimized_static_camera_poses_ref_marker.csv"
    moving_final_src = AP02_ROOT / "07_graph_ba" / "with_moving" / "optimized_static_camera_poses_ref_marker.csv"

    if not moving_final_src.exists():
        raise RuntimeError(f"Missing AP02 with_moving BA static-camera output: {moving_final_src}")

    static_rows = copy_csv_rows(
        static_final_src,
        out / "ap02_static_only_static_camera_poses_ref_marker.csv",
    ) if static_final_src.exists() else []

    moving_rows = copy_csv_rows(
        moving_final_src,
        out / "ap02_with_moving_static_camera_poses_ref_marker.csv",
    )

    marker_rows = copy_csv_rows(
        AP02_ROOT / "07_graph_ba" / "with_moving" / "optimized_marker_poses_ref_marker.csv",
        out / "ap02_with_moving_marker_poses_ref_marker.csv",
    )

    moving_frame_rows = copy_csv_rows(
        AP02_ROOT / "07_graph_ba" / "with_moving" / "optimized_moving_frame_poses_ref_marker.csv",
        out / "ap02_with_moving_moving_frame_poses_ref_marker.csv",
    )

    summary_rows = [
        {
            "method": "single_ref_marker_pnp",
            "static_cameras_estimated": "2",
            "notes": "Only cameras directly seeing reference marker 14 can be estimated.",
        },
        {
            "method": "graph_initialization_static_only",
            "static_cameras_estimated": "3",
            "notes": "Static-only graph misses cam_edge_5.",
        },
        {
            "method": "graph_initialization_with_moving",
            "static_cameras_estimated": "4",
            "notes": "Moving-camera observations connect all static cameras and all markers.",
        },
        {
            "method": "graph_ba_static_only",
            "static_cameras_estimated": str(len(static_rows)),
            "notes": "Bundle adjustment on static-only connected component.",
        },
        {
            "method": "graph_ba_with_moving_sparse",
            "static_cameras_estimated": str(len(moving_rows)),
            "notes": "Sparse moving-frame BA. Main AP02 result.",
        },
    ]

    write_csv(
        out / "ap02_method_stage_summary.csv",
        summary_rows,
        ["method", "static_cameras_estimated", "notes"],
    )

    metadata = {
        "ap02_root": str(AP02_ROOT),
        "reference_marker_id": 14,
        "final_selected_result": "graph_ba_with_moving_sparse",
        "final_static_camera_pose_file": "ap02_with_moving_static_camera_poses_ref_marker.csv",
        "transform_convention": "T_ref_marker_cam maps camera coordinates into the reference-marker coordinate frame.",
        "static_camera_count_final": len(moving_rows),
        "marker_count_final": len(marker_rows),
        "moving_frame_count_final": len(moving_frame_rows),
    }

    (out / "ap02_final_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    report = [
        "AP02 Final Results Report",
        "=========================",
        "",
        "Selected AP02 result:",
        "- graph_ba_with_moving_sparse",
        "",
        "Why this result is selected:",
        "- Single-reference-marker PnP only estimates cameras that directly see marker 14.",
        "- Static-only graph improves connectivity but still misses cam_edge_5.",
        "- With-moving graph connects all 4 static cameras, all 15 markers, and all selected moving frames.",
        "- Bundle adjustment reduces the reprojection error to subpixel-level mean/median values.",
        "",
        "Connectivity summary:",
        "",
        "Single Reference Marker PnP:",
        single_ref_report,
        "",
        "Static-only graph initialization:",
        static_graph_report,
        "",
        "With-moving graph initialization:",
        moving_graph_report,
        "",
        "Static-only BA summary:",
        static_ba_summary,
        "",
        "With-moving BA summary:",
        moving_ba_summary,
        "",
        "Final exported files:",
        "- ap02_with_moving_static_camera_poses_ref_marker.csv",
        "- ap02_with_moving_marker_poses_ref_marker.csv",
        "- ap02_with_moving_moving_frame_poses_ref_marker.csv",
        "- ap02_method_stage_summary.csv",
        "- ap02_final_metadata.json",
        "",
        "Important interpretation:",
        "AP02 is not just another local ArUco chain method. It shows that a global reference-marker pose graph becomes fully connected only when moving-camera observations are included. The final BA result is therefore the AP02 with_moving variant.",
        "",
    ]

    (out / "ap02_final_results_report.txt").write_text("\n".join(report) + "\n")

    print("[OK] wrote AP02 final results:", out)
    print("[OK] final static cameras:", len(moving_rows))
    print("[OK] final markers:", len(marker_rows))
    print("[OK] final moving frames:", len(moving_frame_rows))
    print()
    print((out / "ap02_final_results_report.txt").read_text())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from ap02_common import AP02_ROOT, ensure_dir, read_csv, write_csv


def count_rows(path):
    if not path.exists():
        return 0
    return len(read_csv(path))


def read_report_value(report_path, prefix):
    if not report_path.exists():
        return ""
    for line in report_path.read_text().splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def main():
    out = ensure_dir(AP02_ROOT / "06_graph_initialization_comparison")

    rows = []
    for mode in ["static_only", "with_moving"]:
        root = AP02_ROOT / "05_graph_initialization" / mode
        report = root / "graph_connectivity_report.txt"

        rows.append({
            "mode": mode,
            "static_camera_pose_count": count_rows(root / "initial_static_camera_poses_ref_marker.csv"),
            "moving_frame_pose_count": count_rows(root / "initial_moving_frame_poses_ref_marker.csv"),
            "marker_pose_count": count_rows(root / "initial_marker_poses_ref_marker.csv"),
            "initialized_static_cameras": read_report_value(report, "Initialized static cameras"),
            "missing_static_cameras": read_report_value(report, "Missing static cameras"),
            "initialized_markers": read_report_value(report, "Initialized markers"),
            "missing_markers": read_report_value(report, "Missing markers"),
        })

    fields = [
        "mode",
        "static_camera_pose_count",
        "moving_frame_pose_count",
        "marker_pose_count",
        "initialized_static_cameras",
        "missing_static_cameras",
        "initialized_markers",
        "missing_markers",
    ]

    write_csv(out / "graph_initialization_static_vs_moving.csv", rows, fields)

    report_lines = [
        "AP02 graph initialization comparison",
        "====================================",
        "",
        "This compares the AP02 reference-marker pose graph before bundle adjustment.",
        "",
    ]

    for r in rows:
        report_lines += [
            f"Mode: {r['mode']}",
            f"- static camera poses: {r['static_camera_pose_count']}",
            f"- moving frame poses: {r['moving_frame_pose_count']}",
            f"- marker poses: {r['marker_pose_count']}",
            f"- initialized static cameras: {r['initialized_static_cameras']}",
            f"- missing static cameras: {r['missing_static_cameras']}",
            "",
        ]

    report_lines += [
        "Main question:",
        "Does adding moving-camera observations connect more static cameras and markers to the reference ArUco?",
        "",
    ]

    (out / "graph_initialization_comparison_report.txt").write_text("\n".join(report_lines) + "\n")

    print("[OK] wrote", out)
    print((out / "graph_initialization_comparison_report.txt").read_text())


if __name__ == "__main__":
    main()

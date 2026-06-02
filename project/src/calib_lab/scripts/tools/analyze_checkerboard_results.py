#!/usr/bin/env python3

import csv
import argparse
from pathlib import Path


def sweep_group(scenario: str) -> str:
    if scenario.startswith("dist_"):
        return "distance"
    if scenario.startswith("yaw_"):
        return "yaw"
    if scenario.startswith("shift_"):
        return "shift"
    if scenario.startswith("height_"):
        return "height"
    if "yaw" in scenario or "shift" in scenario or "far" in scenario or "close" in scenario:
        return "mixed"
    return "other"


def classify_row(row, max_error_cm: float, max_rot_deg: float):
    detection_success = row.get("success", "").lower() == "true"

    if not detection_success:
        return "failure"

    try:
        err_cm = float(row["baseline_error_m"]) * 100.0
        rot_deg = float(row["rotation_error_deg"])
    except Exception:
        return "failure"

    if err_cm <= max_error_cm and rot_deg <= max_rot_deg:
        return "valid"

    return "pose_outlier"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_error_cm", type=float, default=10.0)
    parser.add_argument("--max_rot_deg", type=float, default=10.0)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows = []

    for r in rows:
        cls = classify_row(r, args.max_error_cm, args.max_rot_deg)
        group = sweep_group(r["scenario"])

        err_cm = ""
        rot_deg = ""
        baseline = ""

        if r.get("baseline_error_m"):
            err_cm = float(r["baseline_error_m"]) * 100.0
        if r.get("rotation_error_deg"):
            rot_deg = float(r["rotation_error_deg"])
        if r.get("estimated_baseline_m"):
            baseline = float(r["estimated_baseline_m"])

        out_rows.append({
            "scenario": r["scenario"],
            "group": group,
            "method": r.get("method", "checkerboard"),
            "detection_success": r.get("success", ""),
            "pose_class": cls,
            "camera_1_status": r.get("camera_1_status", ""),
            "camera_2_status": r.get("camera_2_status", ""),
            "camera_1_points": r.get("camera_1_points", ""),
            "camera_2_points": r.get("camera_2_points", ""),
            "estimated_baseline_m": f"{baseline:.6f}" if baseline != "" else "",
            "baseline_error_cm": f"{err_cm:.3f}" if err_cm != "" else "",
            "rotation_error_deg": f"{rot_deg:.3f}" if rot_deg != "" else "",
            "failure_reason": r.get("failure_reason", ""),
            "camera_1_image": r.get("camera_1_image", ""),
            "camera_2_image": r.get("camera_2_image", ""),
        })

    fieldnames = [
        "scenario",
        "group",
        "method",
        "detection_success",
        "pose_class",
        "camera_1_status",
        "camera_2_status",
        "camera_1_points",
        "camera_2_points",
        "estimated_baseline_m",
        "baseline_error_cm",
        "rotation_error_deg",
        "failure_reason",
        "camera_1_image",
        "camera_2_image",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print("scenario, group, pose_class, cam1, cam2, error_cm, rot_deg")
    for r in out_rows:
        print(
            f"{r['scenario']}, {r['group']}, {r['pose_class']}, "
            f"{r['camera_1_status']}, {r['camera_2_status']}, "
            f"{r['baseline_error_cm']}, {r['rotation_error_deg']}"
        )

    counts = {}
    for r in out_rows:
        key = (r["group"], r["pose_class"])
        counts[key] = counts.get(key, 0) + 1

    print("\nSummary counts:")
    for key in sorted(counts):
        print(f"{key[0]} | {key[1]}: {counts[key]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import csv
import argparse
import re
from pathlib import Path
from collections import defaultdict


GROUPS = ["distance", "yaw", "shift", "height", "mixed"]


def parse_target_geometry(target_name: str):
    """
    Example:
      target_9x6_square0_12
    becomes:
      pattern = 9x6
      square_size_m = 0.12
    """
    result = {
        "target_id": target_name,
        "target_pattern": "",
        "square_size_m": "",
    }

    pattern_match = re.search(r"target_(\d+x\d+)", target_name)
    if pattern_match:
        result["target_pattern"] = pattern_match.group(1)

    square_match = re.search(r"square(\d+)_(\d+)", target_name)
    if square_match:
        whole = square_match.group(1)
        frac = square_match.group(2)
        result["square_size_m"] = f"{int(whole)}.{frac}"

    return result


def infer_method_from_target_dir(target_dir: Path):
    parts = list(target_dir.parts)
    if "results" in parts:
        idx = parts.index("results")
        if len(parts) > idx + 1:
            return parts[idx + 1]
    return ""


def read_csv(path: Path):
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_row(row, source_csv: Path, method: str, target_info: dict, resolution: str, group: str):
    """
    Supports checkerboard summary.csv primarily.
    Also tolerates raw_results.csv if later needed.
    """
    scenario = row.get("scenario", "")

    pose_class = row.get("pose_class", "")
    if not pose_class:
        success = row.get("success", row.get("detection_success", "")).lower()
        if success == "true":
            pose_class = "unclassified_success"
        elif success == "false":
            pose_class = "failure"

    detection_success = row.get("detection_success", "")
    if not detection_success:
        detection_success = row.get("success", "")

    return {
        "method": method,
        "target_id": target_info["target_id"],
        "target_pattern": target_info["target_pattern"],
        "square_size_m": target_info["square_size_m"],
        "resolution": resolution,
        "group": group,
        "scenario": scenario,
        "detection_success": detection_success,
        "pose_class": pose_class,
        "camera_1_status": row.get("camera_1_status", ""),
        "camera_2_status": row.get("camera_2_status", ""),
        "camera_1_points": row.get("camera_1_points", ""),
        "camera_2_points": row.get("camera_2_points", ""),
        "estimated_baseline_m": row.get("estimated_baseline_m", ""),
        "baseline_error_cm": row.get("baseline_error_cm", ""),
        "rotation_error_deg": row.get("rotation_error_deg", ""),
        "failure_reason": row.get("failure_reason", ""),
        "camera_1_image": row.get("camera_1_image", ""),
        "camera_2_image": row.get("camera_2_image", ""),
        "source_csv": str(source_csv),
    }


def collect_long_rows(target_dir: Path):
    method = infer_method_from_target_dir(target_dir)
    target_info = parse_target_geometry(target_dir.name)

    long_rows = []

    resolution_dirs = sorted(
        p for p in target_dir.iterdir()
        if p.is_dir() and p.name.startswith("res")
    )

    for res_dir in resolution_dirs:
        resolution = res_dir.name

        for group in GROUPS:
            group_dir = res_dir / group
            if not group_dir.exists():
                continue

            summary_csv = group_dir / "summary.csv"
            raw_csv = group_dir / "raw_results.csv"

            if summary_csv.exists():
                source_csv = summary_csv
            elif raw_csv.exists():
                source_csv = raw_csv
            else:
                continue

            rows = read_csv(source_csv)

            for row in rows:
                long_rows.append(
                    normalize_row(
                        row=row,
                        source_csv=source_csv,
                        method=method,
                        target_info=target_info,
                        resolution=resolution,
                        group=group,
                    )
                )

    return long_rows


def write_long_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "target_id",
        "target_pattern",
        "square_size_m",
        "resolution",
        "group",
        "scenario",
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
        "source_csv",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_counts_csv(rows, out_path: Path):
    counts = defaultdict(int)

    for r in rows:
        key = (
            r["method"],
            r["target_id"],
            r["resolution"],
            r["group"],
            r["pose_class"],
        )
        counts[key] += 1

    out_rows = []

    for key, count in sorted(counts.items()):
        method, target_id, resolution, group, pose_class = key
        out_rows.append({
            "method": method,
            "target_id": target_id,
            "resolution": resolution,
            "group": group,
            "pose_class": pose_class,
            "count": count,
        })

    fieldnames = [
        "method",
        "target_id",
        "resolution",
        "group",
        "pose_class",
        "count",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def write_wide_comparison_csv(rows, out_path: Path):
    """
    One row per group+scenario.
    Resolution-specific columns are created dynamically.
    """
    resolutions = sorted(set(r["resolution"] for r in rows))
    grouped = defaultdict(dict)

    base_info = {}

    for r in rows:
        key = (r["method"], r["target_id"], r["group"], r["scenario"])
        grouped[key][r["resolution"]] = r

        base_info[key] = {
            "method": r["method"],
            "target_id": r["target_id"],
            "target_pattern": r["target_pattern"],
            "square_size_m": r["square_size_m"],
            "group": r["group"],
            "scenario": r["scenario"],
        }

    fieldnames = [
        "method",
        "target_id",
        "target_pattern",
        "square_size_m",
        "group",
        "scenario",
    ]

    for res in resolutions:
        fieldnames.extend([
            f"{res}_pose_class",
            f"{res}_detection_success",
            f"{res}_cam1",
            f"{res}_cam2",
            f"{res}_baseline_error_cm",
            f"{res}_rotation_error_deg",
        ])

    fieldnames.extend([
        "class_change",
        "best_resolution_by_error",
        "best_error_cm",
    ])

    out_rows = []

    for key in sorted(grouped.keys()):
        row = dict(base_info[key])

        classes = []
        best_res = ""
        best_error = None

        for res in resolutions:
            r = grouped[key].get(res)

            if r is None:
                row[f"{res}_pose_class"] = "missing"
                row[f"{res}_detection_success"] = ""
                row[f"{res}_cam1"] = ""
                row[f"{res}_cam2"] = ""
                row[f"{res}_baseline_error_cm"] = ""
                row[f"{res}_rotation_error_deg"] = ""
                classes.append("missing")
                continue

            pose_class = r["pose_class"]
            classes.append(pose_class)

            row[f"{res}_pose_class"] = pose_class
            row[f"{res}_detection_success"] = r["detection_success"]
            row[f"{res}_cam1"] = r["camera_1_status"]
            row[f"{res}_cam2"] = r["camera_2_status"]
            row[f"{res}_baseline_error_cm"] = r["baseline_error_cm"]
            row[f"{res}_rotation_error_deg"] = r["rotation_error_deg"]

            if pose_class == "valid" and r["baseline_error_cm"] != "":
                try:
                    err = float(r["baseline_error_cm"])
                    if best_error is None or err < best_error:
                        best_error = err
                        best_res = res
                except ValueError:
                    pass

        unique_classes = list(dict.fromkeys(classes))
        row["class_change"] = "same" if len(unique_classes) == 1 else " -> ".join(unique_classes)
        row["best_resolution_by_error"] = best_res
        row["best_error_cm"] = f"{best_error:.3f}" if best_error is not None else ""

        out_rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        required=True,
        help="Example: results/checkerboard/target_9x6_square0_12",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)

    if not target_dir.exists():
        raise FileNotFoundError(target_dir)

    comparison_dir = target_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    long_csv = comparison_dir / "all_results_long.csv"
    wide_csv = comparison_dir / "resolution_comparison_wide.csv"
    counts_csv = comparison_dir / "counts_by_resolution.csv"

    rows = collect_long_rows(target_dir)

    if not rows:
        print(f"No rows found under {target_dir}")
        return

    write_long_csv(rows, long_csv)
    write_wide_comparison_csv(rows, wide_csv)
    write_counts_csv(rows, counts_csv)

    print(f"Collected rows: {len(rows)}")
    print(f"Wrote: {long_csv}")
    print(f"Wrote: {wide_csv}")
    print(f"Wrote: {counts_csv}")

    print("")
    print("Counts overview:")
    counts = defaultdict(int)
    for r in rows:
        counts[(r["resolution"], r["group"], r["pose_class"])] += 1

    for key, count in sorted(counts.items()):
        resolution, group, pose_class = key
        print(f"{resolution:10s} | {group:8s} | {pose_class:14s} | {count}")


if __name__ == "__main__":
    main()

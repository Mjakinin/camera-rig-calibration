#!/usr/bin/env python3
"""Export AP01 calibration outputs without legacy GT-anchored reports.

The AP01 estimator is cam_edge_3-rooted. Scientific comparison is performed
later by the shared Primary camera-to-camera and Secondary aligned-map
evaluators. This wrapper keeps the estimator output needed by those evaluators
and removes the old Ref14 GT-anchor/readable report that displayed cam3 with
zero error by construction.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "15_export_final_extrinsics_cam3_reference.py"
OUT = Path(
    "results/bus_real_data/"
    "01_marker_direct_relay_multimarker_multichain/"
    "07_final_extrinsics_cam3_reference"
)


def load_legacy():
    spec = importlib.util.spec_from_file_location("ap01_legacy_export", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remove_legacy_outputs() -> None:
    for name in [
        "FINAL_READABLE_REPORT.txt",
        "final_camera_poses_ref14_gt_eval.csv",
    ]:
        path = OUT / name
        if path.exists():
            path.unlink()
            print(f"[CLEAN] removed legacy GT-anchored output: {path}")


def filter_summary_to_deployable_rows() -> None:
    path = OUT / "final_extrinsics_summary.csv"
    if not path.is_file():
        raise RuntimeError(f"Missing AP01 summary after export: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader if row.get("category") == "main_no_gt"]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] canonical AP01 deployable rows: {len(rows)}")


def main() -> None:
    module = load_legacy()

    def no_ref14_rows(_module, _entries):
        return [], {
            "enabled": False,
            "reason": (
                "Legacy Ref14 GT anchoring is disabled. Use the shared "
                "Secondary SE(3)-aligned map evaluation instead."
            ),
        }

    def no_ref14_csv(_rows):
        return None

    def no_legacy_report(_entries, _rows):
        return None

    def write_primary_only_json(entries, _rows, _meta):
        path = OUT / "final_extrinsics_cam3_reference.json"
        main_entries = [
            entry for entry in entries
            if entry.get("category") == "main_no_gt"
        ]
        payload = {
            "reference_camera": module.ROOT_CAM,
            "output_role": "calibration_estimate_only",
            "evaluation_contract": {
                "primary": "shared static camera-to-camera evaluation",
                "secondary": "shared SE(3)-aligned camera-map evaluation",
                "legacy_ref14_gt_anchor": "disabled",
            },
            "extrinsics_cam3_reference": [
                {
                    "name": entry["name"],
                    "target_camera": entry["target_camera"],
                    "method": entry["method"],
                    "category": entry["category"],
                    "estimated_transform_cam3_to_target": module.pose_payload(
                        entry["T_est"]
                    ),
                    "num_candidates": entry.get("num_candidates", ""),
                    "num_inliers": entry.get("num_inliers", ""),
                    "num_outliers": entry.get("num_outliers", ""),
                    "notes": entry.get("notes", ""),
                }
                for entry in main_entries
            ],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[OK] wrote deployable AP01 JSON: {path}")

    def write_primary_only_readme(entries):
        path = OUT / "README.txt"
        main_entries = [
            entry for entry in entries
            if entry.get("category") == "main_no_gt"
        ]
        lines = [
            "AP01 calibration estimate output",
            "================================",
            "",
            "Reference camera: cam_edge_3",
            "",
            "This directory contains estimator outputs, not the canonical",
            "scientific comparison report.",
            "",
            "Canonical evaluation:",
            "- Primary: static camera-to-camera extrinsics for all six pairs.",
            "- Secondary: SE(3)-aligned full static-camera map.",
            "",
            "The former Ref14 GT-anchor report is intentionally disabled",
            "because it assigned zero error to the anchor camera by construction.",
            "",
            "Available AP01 target estimates:",
        ]
        for entry in main_entries:
            lines.append(
                f"- {entry['target_camera']} via {entry['method']}"
            )
        path.write_text("\n".join(lines) + "\n")
        print(f"[OK] wrote AP01 output contract: {path}")

    module.build_ref14_rows = no_ref14_rows
    module.write_ref14_csv = no_ref14_csv
    module.write_readable_report = no_legacy_report
    module.write_json = write_primary_only_json
    module.write_readme = write_primary_only_readme

    remove_legacy_outputs()
    module.main()
    filter_summary_to_deployable_rows()
    remove_legacy_outputs()

    print("[OK] AP01 estimator export complete.")
    print("[INFO] View results only through shared Primary/Secondary reports.")


if __name__ == "__main__":
    main()

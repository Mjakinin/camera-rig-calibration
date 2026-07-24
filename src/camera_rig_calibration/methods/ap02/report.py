from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from camera_rig_calibration.pipeline import StageResult, run_stage

from .common import read_csv, write_csv


def _position(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
        dtype=np.float64,
    )


def run(
    *,
    output_root: Path,
    camera_ids: tuple[str, ...],
    reference_marker_id: int,
) -> StageResult:
    stage_root = output_root / "08_final_results"

    def action() -> dict[str, Path | str | int]:
        combined_path = (
            output_root
            / "07_graph_ba"
            / "with_moving"
            / "optimized_static_camera_poses_ref_marker.csv"
        )
        combined = {
            row["entity_id"]: row
            for row in read_csv(combined_path)
            if row.get("entity_id") in camera_ids
        }
        missing = sorted(set(camera_ids) - set(combined))
        rows = []
        available_camera_ids = tuple(
            camera for camera in camera_ids if camera in combined
        )
        for first, second in combinations(available_camera_ids, 2):
            rows.append(
                {
                    "camera_a": first,
                    "camera_b": second,
                    "distance_m": float(
                        np.linalg.norm(
                            _position(combined[first])
                            - _position(combined[second])
                        )
                    ),
                }
            )
        pairwise = stage_root / "AP02_PAIRWISE_DISTANCES.csv"
        write_csv(
            pairwise,
            rows,
            ["camera_a", "camera_b", "distance_m"],
        )
        static_manifest = (
            output_root
            / "07_graph_ba"
            / "static_only"
            / "stage_manifest.json"
        )
        static_status = "NOT_AVAILABLE"
        if static_manifest.is_file():
            static_status = json.loads(
                static_manifest.read_text(encoding="utf-8")
            ).get("status", "UNKNOWN")
        combined_optimizer = (
            output_root
            / "07_graph_ba"
            / "with_moving"
            / "optimizer_report.json"
        )
        complete = not missing
        status = (
            "OK"
            if complete and static_status == "COMPLETED"
            else "PARTIAL"
            if complete
            else f"PARTIAL_{len(combined)}_OF_{len(camera_ids)}"
        )
        report = {
            "schema_version": 5,
            "method": "AP02",
            "status": status,
            "primary_result": "combined" if complete else None,
            "primary_result_available": complete,
            "comparison_eligible": complete,
            "diagnostic_partial": not complete,
            "reference_marker_id": reference_marker_id,
            "static_only_diagnostic_status": static_status,
            "combined_optimizer": json.loads(
                combined_optimizer.read_text(encoding="utf-8")
            ),
            "combined_static_camera_coverage": {
                "available": sorted(combined),
                "expected": list(camera_ids),
                "missing": missing,
            },
        }
        report_path = stage_root / "AP02_REPORT.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        status_path = output_root / "METHOD_STATUS.json"
        status_path.write_text(
            json.dumps(
                {
                    "method": "AP02",
                    "status": report["status"],
                    "success": True,
                    "primary_result": report["primary_result"],
                    "primary_result_available": complete,
                    "comparison_eligible": complete,
                    "diagnostic_partial": not complete,
                    "available_static_cameras": sorted(combined),
                    "missing_static_cameras": missing,
                    "static_only_diagnostic_status": static_status,
                    "reference_marker_id": reference_marker_id,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "method_status": status_path,
            "report": report_path,
            "pairwise": pairwise,
            "status": str(report["status"]),
            "combined_camera_count": len(combined),
        }

    return run_stage(
        "ap02.report",
        stage_root,
        action,
        inputs={
            "combined_ba": output_root / "07_graph_ba/with_moving",
            "static_only_ba": output_root / "07_graph_ba/static_only",
        },
        parameters={"reference_marker_id": reference_marker_id},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--ref-marker-id", type=int, required=True)
    args = parser.parse_args()
    run(
        output_root=args.out.resolve(),
        camera_ids=tuple(
            item.strip() for item in args.cameras.split(",") if item.strip()
        ),
        reference_marker_id=args.ref_marker_id,
    )


if __name__ == "__main__":
    main()

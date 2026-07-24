from __future__ import annotations

import csv
import json
from pathlib import Path

from camera_rig_calibration.methods.ap02.report import run


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["entity_id", "x_m", "y_m", "z_m"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_ap02_partial_report_is_diagnostic_and_not_comparison_eligible(
    tmp_path: Path,
) -> None:
    output = tmp_path / "03_AP02"
    _write_csv(
        output
        / "07_graph_ba/with_moving"
        / "optimized_static_camera_poses_ref_marker.csv",
        [
            {"entity_id": "cam_1", "x_m": 0, "y_m": 0, "z_m": 0},
            {"entity_id": "cam_2", "x_m": 1, "y_m": 0, "z_m": 0},
            {"entity_id": "cam_3", "x_m": 2, "y_m": 0, "z_m": 0},
        ],
    )
    optimizer = output / "07_graph_ba/with_moving/optimizer_report.json"
    optimizer.write_text(json.dumps({"success": True, "nfev": 12}))

    run(
        output_root=output,
        camera_ids=("cam_1", "cam_2", "cam_3", "cam_4"),
        reference_marker_id=3,
    )

    status = json.loads((output / "METHOD_STATUS.json").read_text())
    report = json.loads(
        (output / "08_final_results/AP02_REPORT.json").read_text()
    )
    assert status["status"] == "PARTIAL_3_OF_4"
    assert status["success"] is True
    assert status["primary_result_available"] is False
    assert status["comparison_eligible"] is False
    assert status["missing_static_cameras"] == ["cam_4"]
    assert report["diagnostic_partial"] is True
    assert report["combined_static_camera_coverage"]["available"] == [
        "cam_1",
        "cam_2",
        "cam_3",
    ]

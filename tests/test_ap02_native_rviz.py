from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.visualization.ap02_native import (
    discover_ap02_native_scenes,
    ensure_ap02_native_scene,
)


def _csv(rows: list[tuple[str, float]]) -> str:
    lines = ["entity_id,x_m,y_m,z_m,rvec_x,rvec_y,rvec_z"]
    for camera, x in rows:
        lines.append(f"{camera},{x},0,1,0,0,0")
    return "\n".join(lines) + "\n"


def test_partial_ap02_exposes_separate_native_component_scenes(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    result = experiment / "methods/ap02/partial"
    component_root = result / "diagnostics/method/component_diagnostics"
    component_root.mkdir(parents=True)
    result_payload = {
        "method": "ap02",
        "label": "partial",
        "config_summary": {
            "reference_marker_id": 8,
            "resolved_reference_marker_id": 8,
        },
    }
    (result / "RESULT.json").write_text(
        json.dumps(result_payload), encoding="utf-8"
    )
    (result / "camera_extrinsics.csv").write_text(
        _csv([("camera_a", 0.0), ("camera_b", 1.0)]), encoding="utf-8"
    )
    components = {
        "primary_component_id": "component_01",
        "components": [
            {
                "component_id": "component_01",
                "execution_status": "primary_component",
                "anchor_marker_id": 2,
                "static_cameras": ["camera_a", "camera_b"],
            },
            {
                "component_id": "component_02",
                "execution_status": "available",
                "local_reference_marker_id": 7,
                "static_cameras": ["camera_c", "camera_d"],
            },
            {
                "component_id": "component_03",
                "execution_status": "not_calibratable",
                "anchor_marker_id": 0,
                "static_cameras": ["camera_e"],
            },
        ],
    }
    (component_root / "AP02_COMPONENT_RESULTS.json").write_text(
        json.dumps(components), encoding="utf-8"
    )
    diagnostic = component_root / "component_02"
    diagnostic.mkdir()
    (diagnostic / "camera_extrinsics.csv").write_text(
        _csv([("camera_c", 0.0), ("camera_d", 0.3)]), encoding="utf-8"
    )

    scenes = discover_ap02_native_scenes(experiment)
    assert len(scenes) == 2
    assert scenes[0]["component_id"] == "component_01"
    assert scenes[0]["reference_marker_id"] == 8
    assert scenes[1]["component_id"] == "component_02"
    assert scenes[1]["reference_marker_id"] == 7

    output = ensure_ap02_native_scene(experiment, scenes[0])
    manifest = json.loads(
        (output / "visualization_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["available"] is True
    assert manifest["fixed_frame"] == "ap02_marker_8"
    assert manifest["point_count"] == 0
    assert manifest["ground_truth_used"] is False
    rviz = (output / "rigcal_result.rviz").read_text(encoding="utf-8")
    assert "ap02_marker_8" in rviz
    assert "ap02/" in rviz
    assert "Enabled: true" in rviz

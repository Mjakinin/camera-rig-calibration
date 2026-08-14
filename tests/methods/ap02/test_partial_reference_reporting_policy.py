from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.policies import real_partial_evaluation_policy as partial
from camera_rig_calibration.policies.ap02_partial_reference_reporting_policy import (
    install_ap02_partial_reference_reporting_policy,
)


def _payload() -> dict:
    return {
        "method": "ap02",
        "config_summary": {
            "reference_marker_id": 8,
            "resolved_reference_marker_id": 8,
        },
        "metrics": {
            "ap02_component_results": {
                "status": "partial_coverage",
                "primary_component_id": "component_01",
                "cross_component_extrinsics": "not_observable",
                "camera_pair_observability": [],
                "components": [
                    {
                        "component_id": "component_01",
                        "execution_status": "primary_component",
                        "anchor_marker_id": 2,
                        "static_cameras": ["camera_a", "camera_b"],
                        "marker_ids": [2, 8],
                    },
                    {
                        "component_id": "component_02",
                        "execution_status": "available",
                        "local_reference_marker_id": 7,
                        "static_cameras": ["camera_c", "camera_d"],
                        "marker_ids": [7, 9],
                    },
                ],
            }
        },
    }


def test_primary_component_summary_uses_frozen_ap02_reference() -> None:
    install_ap02_partial_reference_reporting_policy()
    text = partial._component_summary_text(_payload())
    primary_line = next(line for line in text.splitlines() if line.startswith("component_01"))
    secondary_line = next(line for line in text.splitlines() if line.startswith("component_02"))
    assert "8" in primary_line
    assert "7" in secondary_line


def test_primary_component_pose_header_uses_frozen_ap02_reference(tmp_path: Path) -> None:
    install_ap02_partial_reference_reporting_policy()
    root = tmp_path / "result"
    component_root = root / "diagnostics/method/component_diagnostics"
    component_root.mkdir(parents=True)
    (root / "RESULT.json").write_text(json.dumps(_payload()), encoding="utf-8")
    (component_root / "AP02_COMPONENT_RESULTS.json").write_text(
        json.dumps(_payload()["metrics"]["ap02_component_results"]),
        encoding="utf-8",
    )
    (root / "camera_extrinsics.csv").write_text(
        "entity_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
        "camera_a,0,0,0,0,0,0\n"
        "camera_b,1,0,0,0,0,0\n",
        encoding="utf-8",
    )
    diagnostic = component_root / "component_02"
    diagnostic.mkdir()
    (diagnostic / "camera_extrinsics.csv").write_text(
        "entity_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
        "camera_c,0,0,0,0,0,0\n"
        "camera_d,0,1,0,0,0,0\n",
        encoding="utf-8",
    )

    text = partial._component_pose_detail(root)
    assert "component_01 | execution=primary_component | local frame=marker_8" in text
    assert "component_02 | execution=available | local frame=marker_7" in text

from __future__ import annotations

import json
from pathlib import Path

import yaml

from camera_rig_calibration.anchor_export.compact import (
    COMPACT_ANCHOR_YAML,
    compact_anchor_payload,
    write_compact_method_anchor_yaml,
)


def _canonical_payload() -> dict:
    return {
        "parent_frame": "evaluation_anchor_marker_14",
        "method": "ap01",
        "label": "baseline",
        "cameras": [
            {
                "camera_id": "cam_edge_0",
                "x_m": 0.55,
                "y_m": -1.70,
                "z_m": 2.14,
                "roll_rad": -1.98,
                "pitch_rad": -0.02,
                "yaw_rad": -0.04,
                "qx": 0.1,
                "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
            {
                "camera_id": "cam_edge_3",
                "x_m": -3.28,
                "y_m": -1.56,
                "z_m": 2.08,
                "roll_rad": -1.91,
                "pitch_rad": 0.02,
                "yaw_rad": -1.05,
                "quality_status": "gauge_identity",
            },
        ],
    }


def test_compact_anchor_payload_is_exact_six_dof_view() -> None:
    payload = compact_anchor_payload(_canonical_payload())

    assert payload == {
        "evaluation_anchor_marker_14": {
            "cam_edge_0": {
                "x": 0.55,
                "y": -1.70,
                "z": 2.14,
                "roll": -1.98,
                "pitch": -0.02,
                "yaw": -0.04,
            },
            "cam_edge_3": {
                "x": -3.28,
                "y": -1.56,
                "z": 2.08,
                "roll": -1.91,
                "pitch": 0.02,
                "yaw": -1.05,
            },
        }
    }


def test_compact_yaml_is_written_beside_canonical_export(tmp_path: Path) -> None:
    source = tmp_path / "camera_extrinsics_anchor.json"
    source.write_text(json.dumps(_canonical_payload()), encoding="utf-8")

    target = write_compact_method_anchor_yaml(tmp_path)

    assert target == tmp_path / COMPACT_ANCHOR_YAML
    assert yaml.safe_load(target.read_text(encoding="utf-8")) == (
        compact_anchor_payload(_canonical_payload())
    )

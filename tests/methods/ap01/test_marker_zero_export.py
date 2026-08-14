from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from camera_rig_calibration.policies.ap01_common_anchor_policy import (
    install_ap01_common_anchor_policy,
)
from camera_rig_calibration.anchor_export import adapters
from camera_rig_calibration.anchor_export.geometry import make_transform
from camera_rig_calibration.visualization.ros_scene import _color


install_ap01_common_anchor_policy()


FIELDS = [
    "observer_type",
    "observer_id",
    "camera_name",
    "frame_id",
    "marker_id",
    "selection_score",
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_m",
    "tvec_y_m",
    "tvec_z_m",
]


def _row(
    observer_type: str,
    observer_id: str,
    marker_id: int,
    x: float,
    *,
    frame: int | None = None,
    score: float = 10.0,
) -> dict[str, object]:
    return {
        "observer_type": observer_type,
        "observer_id": observer_id,
        "camera_name": observer_id,
        "frame_id": "" if frame is None else str(frame),
        "marker_id": marker_id,
        "selection_score": score,
        "rvec_x": 0.0,
        "rvec_y": 0.0,
        "rvec_z": 0.0,
        "tvec_x_m": x,
        "tvec_y_m": 0.0,
        "tvec_z_m": 1.0,
    }


def _pose(x: float) -> np.ndarray:
    return make_transform(np.eye(3), np.asarray([x, 0.0, 0.0]))


def _write_fixture(root: Path) -> None:
    preflight = root / "diagnostics" / "preflight"
    method = root / "diagnostics" / "method"
    preflight.mkdir(parents=True)
    (method / "static_extrinsics").mkdir(parents=True)
    (method / "moving_colmap" / "sparse_txt_best").mkdir(parents=True)

    # Native AP01 frame: cam_a at x=0, cam_b at x=2. Marker 0 is truly at x=1,
    # but the only static marker-0 PnP row is deliberately wrong (x=5). The
    # moving trajectory plus independent bridge markers 5 and 6 still determines
    # the correct global marker-0 frame.
    rows = [
        _row("static", "cam_a", 0, 5.0, score=50.0),
        _row("static", "cam_b", 5, 1.0, score=20.0),
        _row("static", "cam_b", 6, 2.0, score=20.0),
    ]
    moving_x = {0: 0.0, 1: 1.0, 2: 2.0}
    marker_x = {0: 1.0, 5: 3.0, 6: 4.0}
    for frame, camera_x in moving_x.items():
        for marker_id, world_x in marker_x.items():
            rows.append(
                _row(
                    "moving",
                    "moving_calib_camera",
                    marker_id,
                    world_x - camera_x,
                    frame=frame,
                    score=30.0,
                )
            )
    with (preflight / "accepted_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    (method / "static_extrinsics" / "AP01_DIAGNOSTICS.json").write_text(
        json.dumps({"metric_scale": {"scale_m_per_colmap_unit": 1.0}}),
        encoding="utf-8",
    )

    # COLMAP stores world->camera. Identity rotations with camera centers at
    # x={0,1,2} therefore have translations {0,-1,-2}.
    image_lines = []
    for image_id, frame in enumerate((0, 1, 2), start=1):
        tx = -moving_x[frame]
        image_lines.extend(
            [
                f"{image_id} 1 0 0 0 {tx} 0 0 1 frame_{frame:06d}.png",
                "0 0 -1",
            ]
        )
    (method / "moving_colmap" / "sparse_txt_best" / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="utf-8"
    )


def test_marker_zero_uses_moving_bridge_consensus_not_single_static_pnp(
    tmp_path: Path,
) -> None:
    method_root = tmp_path / "ap01"
    _write_fixture(method_root)
    config = SimpleNamespace(
        dataset=SimpleNamespace(category="real_vehicle")
    )
    camera_poses = {
        "cam_a": _pose(0.0),
        "cam_b": _pose(2.0),
    }

    resolution = adapters._ap01(method_root, config, 0, camera_poses)

    assert resolution.available is True
    assert resolution.code == "OK_MOVING_BRIDGE_CONSENSUS"
    assert resolution.transform_method_anchor is not None
    assert np.isclose(resolution.transform_method_anchor[0, 3], 1.0, atol=1e-9)
    assert resolution.diagnostics["independent_bridge_marker_count"] >= 2
    direct = resolution.diagnostics["direct_static_cross_check"]
    assert direct["translation_disagreement_m"] > 1.0


def test_rviz_method_colors_are_stable_and_distinct() -> None:
    colors = {
        "ap01": _color("ap01/baseline"),
        "ap02": _color("ap02/baseline"),
        "ap03_multi": _color("ap03_multi/baseline"),
        "ap03_single": _color("ap03_single/baseline"),
    }
    assert len(set(colors.values())) == len(colors)
    assert _color("ap01/other_variant") == colors["ap01"]
    assert _color("ap02/other_variant") == colors["ap02"]

from __future__ import annotations

from camera_rig_calibration.visualization.ros_scene import (
    _ground_truth_anchor_segment,
)


def test_ground_truth_anchor_segment_starts_at_common_marker_origin() -> None:
    camera = {
        "matrix": [
            [1.0, 0.0, 0.0, 1.25],
            [0.0, 1.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    start, end = _ground_truth_anchor_segment(camera)

    assert start == (0.0, 0.0, 0.0)
    assert end == (1.25, -0.5, 2.0)

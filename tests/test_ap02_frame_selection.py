from __future__ import annotations

import pytest

from camera_rig_calibration.methods.ap02.frame_selection import (
    AP02FrameSelectionError,
    select_ap02_frames,
)


def _row(
    observer_type: str,
    observer_id: str,
    marker_id: int,
    *,
    score: float,
) -> dict[str, str]:
    return {
        "observer_type": observer_type,
        "observer_id": observer_id,
        "camera_name": (
            observer_id if observer_type == "static" else "moving"
        ),
        "frame_id": observer_id,
        "marker_id": str(marker_id),
        "selection_score": str(score),
        "score_area": str(score),
        "score_reprojection": "1",
        "score_border": "1",
        "score_distance": "1",
    }


def _fixture_rows() -> list[dict[str, str]]:
    return [
        _row("static", "cam_1", 1, score=1.0),
        _row("static", "cam_2", 2, score=1.0),
        _row("moving", "moving_frame_1", 1, score=0.8),
        _row("moving", "moving_frame_1", 2, score=0.8),
        _row("moving", "moving_frame_2", 1, score=0.9),
        _row("moving", "moving_frame_3", 2, score=0.7),
    ]


def test_ap02_frame_selection_is_deterministic_and_preserves_graph() -> None:
    selection = select_ap02_frames(
        _fixture_rows(),
        camera_ids=("cam_1", "cam_2"),
        reference_marker_id=1,
        reference_marker_maximum_frames=None,
        top_per_marker=1,
        top_per_marker_pair=1,
        maximum_total_frames=1,
    )

    assert selection.selected_frame_ids == ("moving_frame_1",)
    assert selection.summary["minimum_graph_preserving_frames"] == 1
    bridge = next(
        row
        for row in selection.diagnostics
        if row["frame_id"] == "moving_frame_1"
    )
    assert "graph_preservation" in bridge["selection_reasons"]
    assert bridge["selected"] is True


def test_ap02_cap_smaller_than_minimum_is_a_preflight_error() -> None:
    with pytest.raises(
        AP02FrameSelectionError,
        match="minimum graph-preserving set \\(1 moving frames\\)",
    ):
        select_ap02_frames(
            _fixture_rows(),
            camera_ids=("cam_1", "cam_2"),
            reference_marker_id=1,
            reference_marker_maximum_frames=None,
            top_per_marker=1,
            top_per_marker_pair=1,
            maximum_total_frames=0,
        )


def test_ap02_frame_tie_breaker_uses_ascending_string_id() -> None:
    rows = [
        _row("static", "cam_1", 1, score=1.0),
        _row("static", "cam_2", 2, score=1.0),
        _row("moving", "frame_alpha", 1, score=0.5),
        _row("moving", "frame_alpha", 2, score=0.5),
        _row("moving", "frame_zulu", 1, score=0.5),
        _row("moving", "frame_zulu", 2, score=0.5),
    ]

    selection = select_ap02_frames(
        rows,
        camera_ids=("cam_1", "cam_2"),
        reference_marker_id=1,
        reference_marker_maximum_frames=None,
        top_per_marker=1,
        top_per_marker_pair=1,
        maximum_total_frames=1,
    )

    assert selection.selected_frame_ids == ("frame_alpha",)

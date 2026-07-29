from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from camera_rig_calibration.config.models import (
    AP02Settings,
)
from camera_rig_calibration.methods.ap01.solve_extrinsics import (
    compare_paths,
    evaluate_path_gate,
)
from camera_rig_calibration.methods.ap02.initialize import (
    best_observations,
    build_graph,
    deterministic_breadth_first_tree,
    marker_node,
    maximum_bottleneck_tree,
    observer_node,
)
from camera_rig_calibration.methods.ap03.inspect import (
    reconstruction_diagnostics,
)


def _observation(
    observer: str,
    marker: int,
    score: float,
    *,
    frame: str,
    rmse: float = 1.0,
    area: float = 0.01,
) -> dict[str, str]:
    return {
        "observer_type": (
            "moving" if observer.startswith("moving_frame_") else "static"
        ),
        "observer_id": observer,
        "marker_id": str(marker),
        "frame_id": frame,
        "image_path": f"{frame}.png",
        "pnp_success": "true",
        "selection_score": str(score),
        "pnp_reprojection_rmse_px": str(rmse),
        "marker_area_ratio": str(area),
    }


def test_ap02_maximum_bottleneck_beats_short_weak_bfs_path() -> None:
    rows = best_observations(
        [
            _observation("cam_edge_5", 14, 0.20, frame="weak_direct"),
            _observation("moving_frame_0001", 14, 0.90, frame="bridge_a"),
            _observation("moving_frame_0001", 7, 0.90, frame="bridge_b"),
            _observation("cam_edge_5", 7, 0.90, frame="strong_direct"),
        ]
    )
    graph = build_graph(rows)
    start = marker_node(14)

    bfs = deterministic_breadth_first_tree(graph, start)
    productive, metrics = maximum_bottleneck_tree(graph, start)

    assert bfs[observer_node("cam_edge_5")][1]["frame_id"] == "weak_direct"
    assert (
        productive[observer_node("cam_edge_5")][1]["frame_id"]
        == "strong_direct"
    )
    assert metrics[observer_node("cam_edge_5")]["path_length"] == 3
    assert metrics[observer_node("cam_edge_5")]["bottleneck_score"] == pytest.approx(
        0.90
    )


def test_ap02_edge_selection_uses_documented_tie_breakers() -> None:
    selected = best_observations(
        [
            _observation(
                "cam_edge_5", 14, 0.8, frame="later", rmse=2.0, area=0.02
            ),
            _observation(
                "cam_edge_5", 14, 0.8, frame="better_rmse", rmse=1.0, area=0.01
            ),
            _observation(
                "cam_edge_5", 14, 0.8, frame="larger_area", rmse=1.0, area=0.03
            ),
        ]
    )
    assert len(selected) == 1
    assert selected[0]["frame_id"] == "larger_area"


def test_ap01_direct_gate_counts_true_inlier_markers() -> None:
    candidates = [
        {"root_marker": marker, "inlier": inlier}
        for marker, inlier in ((2, True), (3, True), (4, True), (5, False))
    ]
    result = evaluate_path_gate(
        candidates,
        {
            "maximum_inlier_translation_dispersion_m": 0.11,
            "maximum_inlier_rotation_dispersion_deg": 3.9,
            "pose_fallback_used": False,
        },
        minimum_inlier_ratio=0.70,
        maximum_translation_dispersion_m=0.12,
        maximum_rotation_dispersion_deg=4.0,
        minimum_independent_markers=3,
    )
    assert result["stable"] is True
    assert result["inlier_ratio"] == pytest.approx(0.75)
    assert result["inlier_marker_ids"] == [2, 3, 4]


def test_ap01_path_disagreement_is_independent_from_stability() -> None:
    direct = np.eye(4)
    relay = np.eye(4)
    relay[0, 3] = 0.13
    result = compare_paths(
        direct,
        relay,
        maximum_translation_disagreement_m=0.12,
        maximum_rotation_disagreement_deg=4.0,
    )
    assert result["available"] is True
    assert result["consistent"] is False


def test_ap02_legacy_marker_modes_and_baseline_contract() -> None:
    assert AP02Settings(reference_marker_id=7).reference_marker_selection_mode == (
        "explicit"
    )
    assert AP02Settings(
        reference_marker_selection_mode="baseline"
    ).reference_marker_id == 14
    with pytest.raises(ValidationError, match="marker ID"):
        AP02Settings(
            reference_marker_selection_mode="explicit",
            reference_marker_id="auto",
        )


def test_ap03_reconstruction_diagnostics_report_tracks_and_groups(
    tmp_path,
) -> None:
    model = tmp_path / "0"
    model.mkdir()
    (model / "images.txt").write_text(
        "\n".join(
            [
                "1 1 0 0 0 0 0 0 1 static_cam_edge_0.png",
                "10 10 1 20 20 2",
                "2 1 0 0 0 0 0 0 2 static_cam_edge_1.png",
                "10 10 2",
                "3 1 0 0 0 0 0 0 3 moving_frame_0001.png",
                "10 10 1 20 20 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (model / "points3D.txt").write_text(
        "\n".join(
            [
                "1 0 0 1 255 0 0 1.0 1 0 3 0",
                "2 0 0 2 0 255 0 2.0 1 1 2 0 3 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics = reconstruction_diagnostics(
        model,
        [
            {
                "image_name": "static_cam_edge_0.png",
                "source_type": "static",
                "source_id": "cam_edge_0",
            },
            {
                "image_name": "static_cam_edge_1.png",
                "source_type": "static",
                "source_id": "cam_edge_1",
            },
            {
                "image_name": "moving_frame_0001.png",
                "source_type": "moving",
                "source_id": "moving_calib_camera",
            },
        ],
        ("static_cam_edge_0.png", "static_cam_edge_1.png"),
    )
    assert diagnostics["ground_truth_used"] is False
    assert diagnostics["sparse_point_count"] == 2
    assert diagnostics["registered_static_camera_count"] == 2
    assert diagnostics["registered_moving_frame_count"] == 1
    assert diagnostics["static_cameras"][0]["track_support"] == 2
    assert diagnostics["static_cameras"][0]["shared_tracks_with_moving"] == 2
    assert diagnostics["camera_groups"][
        "one_camera_id_per_physical_camera"
    ] is True
    assert diagnostics["camera_groups"][
        "physical_camera_ids_are_distinct"
    ] is True

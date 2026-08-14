from __future__ import annotations

import csv
from pathlib import Path

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.config.models import (
    EvaluationSettings,
    MethodSettings,
    RigConfig,
    SelectionSettings,
    StaticCameraSettings,
)
from camera_rig_calibration.contracts import RunContext
from camera_rig_calibration.observations import (
    freeze_selections,
    resolve_selections,
)
from camera_rig_calibration.registry import calibration_methods


def _row(
    observer_type: str,
    observer_id: str,
    marker: int,
    area: float = 100.0,
    frame: str = "",
) -> dict[str, object]:
    return {
        "observer_type": observer_type,
        "observer_id": observer_id,
        "frame_id": frame,
        "marker_id": marker,
        "area_px2": area,
        "pnp_reprojection_rmse_px": 0.5,
        "pnp_success": True,
    }


def _write(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir()
    with (root / "shared_all_aruco_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_ap01_root_uses_observation_reachability_not_intrinsics(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    root = tmp_path / "observations"
    rows = [
        _row("static", "front-left", 1),
        _row("static", "front-left", 2),
        _row("static", "roof.camera", 2),
        _row("static", "roof.camera", 3),
        _row("moving", "moving_frame_1", 1, frame="1"),
        _row("moving", "moving_frame_2", 1, frame="2"),
        _row("moving", "moving_frame_3", 2, frame="3"),
        _row("moving", "moving_frame_4", 2, frame="4"),
    ]
    _write(root, rows)

    resolved = resolve_selections(prepared_config, root)

    assert resolved.root_camera == "front-left"
    candidates = {
        row["id"]: row
        for row in resolved.payload["ap01_root_camera"]["candidates"]
    }
    assert candidates["front-left"]["moving_bridges"] == ["roof.camera"]
    assert candidates["front-left"]["reachable_cameras"] == [
        "front-left",
        "roof.camera",
    ]


def test_ap01_root_reachability_is_transitive(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    root = tmp_path / "observations"
    rows = [
        _row("static", "front-left", 1),
        _row("static", "roof.camera", 1),
        _row("static", "roof.camera", 2),
        _row("static", "third-camera", 2),
        _row("moving", "moving_frame_1", 1, frame="1"),
        _row("moving", "moving_frame_2", 1, frame="2"),
    ]
    _write(root, rows)
    config = prepared_config.model_copy(
        update={
            "static_cameras": [
                *prepared_config.static_cameras,
                StaticCameraSettings(id="third-camera"),
            ]
        },
        deep=True,
    )

    resolved = resolve_selections(config, root)

    candidates = {
        row["id"]: row
        for row in resolved.payload["ap01_root_camera"]["candidates"]
    }
    assert candidates["front-left"]["reachable_cameras"] == [
        "front-left",
        "roof.camera",
        "third-camera",
    ]


def test_review_freezes_distinct_method_selections(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    root = tmp_path / "observations"
    rows = [
        _row("static", "front-left", 7),
        _row("static", "roof.camera", 7),
        _row("static", "front-left", 9),
        _row("moving", "moving_frame_1", 7, frame="1"),
        _row("moving", "moving_frame_2", 7, frame="2"),
        _row("moving", "moving_frame_3", 9, frame="3"),
        _row("moving", "moving_frame_4", 9, frame="4"),
    ]
    _write(root, rows)
    config = prepared_config.model_copy(
        update={
            "methods": prepared_config.methods.model_copy(
                update={"enabled": ["ap03"]}, deep=True
            ),
            "selection": SelectionSettings(mode="review_once"),
            "evaluation": EvaluationSettings(anchor_marker_id="auto"),
        },
        deep=True,
    )
    resolved = resolve_selections(config, root)

    frozen = freeze_selections(
        config,
        resolved,
        {
            "root_camera": "roof.camera",
            "ap02_reference_marker_id": 7,
            "ap03_single_scale_marker_id": 9,
            "ap03_multi_marker_ids": [7, 9],
        },
    )

    assert frozen.selection.mode == "explicit"
    assert frozen.methods.ap01.root_camera == "roof.camera"
    assert frozen.methods.ap02.reference_marker_id == 7
    assert frozen.methods.ap03.single.scale_marker_id == 9
    assert frozen.evaluation.anchor_marker_id == 7


def test_ap01_contract_has_no_reference_marker_requirement(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    register_builtin_components()
    config = prepared_config.model_copy(
        update={"methods": MethodSettings(enabled=["ap01"])}, deep=True
    )
    context = RunContext(
        repository_root=Path(__file__).resolve().parents[1],
        config=config,
        dataset_root=config.dataset.prepared_root,
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run",
        resolved_root_camera="front-left",
    )

    assert calibration_methods.get("ap01").requirements(context).compatible


def test_disabled_methods_do_not_promote_singleton_marker_recommendations(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    root = tmp_path / "observations"
    _write(root, [_row("static", "front-left", 7)])
    config = prepared_config.model_copy(
        update={
            "methods": prepared_config.methods.model_copy(
                update={
                    "enabled": ["diagnostic_extension"],
                    "extensions": {"diagnostic_extension": {}},
                },
                deep=True,
            ),
            "evaluation": EvaluationSettings(enabled=False),
        },
        deep=True,
    )

    resolved = resolve_selections(config, root)

    assert resolved.ap02_reference_marker_id == 7
    assert resolved.ap03_single_scale_marker_id == 7
    assert (
        resolved.payload["automatic_recommendations"][
            "ap02_reference_marker_id"
        ]
        is None
    )
    assert (
        resolved.payload["automatic_recommendations"][
            "ap03_single_scale_marker_id"
        ]
        is None
    )

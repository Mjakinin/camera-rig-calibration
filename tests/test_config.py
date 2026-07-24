from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from camera_rig_calibration.config import config_fingerprint, load_config, save_config
from camera_rig_calibration.config.models import (
    AP02Settings,
    DatasetSettings,
    MethodSettings,
    MovingCameraSettings,
    ObservationQualitySettings,
    ProjectSettings,
    RigConfig,
    SceneType,
    SimulationSettings,
    StaticCameraSettings,
    McapSettings,
)


def test_ap02_recommended_defaults_are_explicit() -> None:
    assert AP02Settings().model_dump() == {
        "reference_marker_id": "auto",
        "static_only_ba_max_function_evaluations": 100,
        "combined_ba_max_function_evaluations": 120,
        "ba_robust_loss": "soft_l1",
        "ba_robust_loss_scale_px": 3.0,
    }


def test_observation_quality_v5_defaults_are_explicit() -> None:
    assert ObservationQualitySettings().model_dump() == {
        "maximum_pnp_reprojection_error_px": 25.0,
        "minimum_marker_area_px2": 0.0,
        "maximum_marker_distance_m": "disabled",
    }


def test_unknown_configuration_fields_are_rejected(prepared_config: RigConfig) -> None:
    payload = prepared_config.model_dump(mode="python")
    payload["dataset"]["bus_layout"] = "interior"
    with pytest.raises(ValidationError, match="bus_layout"):
        RigConfig.model_validate(payload)


def test_scene_type_is_metadata_only(prepared_config: RigConfig) -> None:
    exterior = prepared_config
    interior = exterior.model_copy(
        update={
            "dataset": exterior.dataset.model_copy(
                update={"scene_type": SceneType.INTERIOR}
            )
        },
        deep=True,
    )
    assert exterior.methods == interior.methods
    assert exterior.colmap == interior.colmap
    assert exterior.markers == interior.markers


def test_config_round_trip_is_reproducible(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    path = save_config(prepared_config, tmp_path / "run.yaml")
    loaded = load_config(path)
    assert loaded == prepared_config
    assert config_fingerprint(loaded) == config_fingerprint(prepared_config)


def test_schema_v1_references_migrate_to_method_owned_v5_fields(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    payload = prepared_config.model_dump(mode="json")
    payload["schema_version"] = 1
    payload["references"] = {
        "root_camera": "front-left",
        "ap02_pose_reference_marker_id": 7,
        "evaluation_scale_anchor_marker_id": 9,
    }
    payload["methods"]["ap01"].pop("root_camera")
    payload["methods"]["ap02"].pop("reference_marker_id")
    ap03 = payload["methods"].pop("ap03")
    single = ap03["single"]
    single["marker_id"] = single.pop("scale_marker_id")
    payload["methods"]["ap03_single"] = single
    payload["methods"]["ap03_multi"] = ap03["multi"]
    payload["methods"]["enabled"] = ["ap01", "ap02", "ap03_single", "ap03_multi"]
    payload["evaluation"].pop("anchor_marker_id")
    source = tmp_path / "legacy.yaml"
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")

    migrated = load_config(source, resolve_paths=False)
    destination = save_config(migrated, tmp_path / "v5.yaml")
    saved = yaml.safe_load(destination.read_text(encoding="utf-8"))

    assert migrated.schema_version == 5
    assert migrated.methods.ap01.root_camera == "front-left"
    assert migrated.methods.ap02.reference_marker_id == 7
    assert migrated.methods.ap03.single.scale_marker_id == "auto"
    assert migrated.methods.enabled == ["ap01", "ap02", "ap03"]
    assert migrated.evaluation.anchor_marker_id == 9
    assert "references" not in saved


def test_schema_v5_rejects_removed_selection_fields(
    prepared_config: RigConfig,
) -> None:
    payload = prepared_config.model_dump(mode="python")
    payload["frame_selection"] = {"policy": "baseline"}
    with pytest.raises(ValidationError, match="frame_selection"):
        RigConfig.model_validate(payload)

    removed_nested_fields = (
        ("ap01", "top_moving_per_marker", 8),
        ("ap02", "moving_selection", "smart"),
        ("ap02", "top_per_marker", 8),
        ("ap02", "max_moving_frames", 100),
    )
    for method, field, value in removed_nested_fields:
        payload = prepared_config.model_dump(mode="python")
        payload["methods"][method][field] = value
        with pytest.raises(ValidationError, match=field):
            RigConfig.model_validate(payload)

    payload = prepared_config.model_dump(mode="python")
    payload["methods"]["ap03"]["single"]["minimum_area_px2"] = 100.0
    with pytest.raises(ValidationError, match="minimum_area_px2"):
        RigConfig.model_validate(payload)

    payload = prepared_config.model_dump(mode="python")
    payload["methods"]["ap02"]["reference_marker_id"] = "sweep"
    with pytest.raises(ValidationError, match="reference_marker_id"):
        RigConfig.model_validate(payload)


def test_schema_v1_ap03_legacy_marker_fields_are_migrated(
    prepared_config: RigConfig,
) -> None:
    payload = prepared_config.model_dump(
        mode="python", exclude_none=True
    )
    payload["schema_version"] = 1
    payload["methods"]["ap03_single"] = {"marker_id": 9}
    payload["methods"]["ap03_multi"] = {
        "marker_id": "auto",
        "marker_ids": [7, 9],
    }

    migrated = RigConfig.model_validate(payload)

    assert migrated.methods.ap03.single.scale_marker_id == 9
    assert migrated.methods.ap03.multi.marker_ids == [7, 9]


def test_simulation_capture_has_an_explicit_generic_contract(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    route = tmp_path / "route.json"
    world.write_text('<sdf version="1.8"><world name="fixture"/></sdf>')
    route.write_text('{"frames": []}')
    config = RigConfig(
        project=ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "datasets",
            output_root=tmp_path / "results",
        ),
        dataset=DatasetSettings(id="simulation_fixture", scene_type="simulation"),
        static_cameras=[
            StaticCameraSettings(
                id="left",
                image_topic="/left/image",
                camera_info_topic="/left/camera_info",
            )
        ],
        moving_camera=MovingCameraSettings(
            id="wand",
            image_topic="/wand/image",
            camera_info_topic="/wand/camera_info",
        ),
        simulation=SimulationSettings(enabled=True, world=world, route=route),
        methods=MethodSettings(enabled=["ap02"]),
    )
    assert config.simulation.enabled
    assert config.dataset.scene_type is SceneType.SIMULATION
    payload = config.model_dump(mode="python")
    payload["dataset"]["scene_type"] = "exterior"
    changed_metadata = RigConfig.model_validate(payload)
    assert changed_metadata.dataset.category.value == "simulation"
    assert changed_metadata.dataset.scene_type.value == "exterior"


def test_ros_recording_can_supply_both_static_and_moving_camera_topics(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "all_cameras.mcap"
    recording.write_bytes(b"fixture")
    config = RigConfig(
        project=ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "datasets",
            output_root=tmp_path / "results",
        ),
        dataset=DatasetSettings(id="bag_all_cameras", input_root=tmp_path),
        static_cameras=[
            StaticCameraSettings(
                id="front", image_topic="/front/image", camera_info_topic="/front/info"
            )
        ],
        moving_camera=MovingCameraSettings(
            id="wand", image_topic="/wand/image", camera_info_topic="/wand/info"
        ),
        mcap=McapSettings(path=recording),
        methods=MethodSettings(enabled=["ap02"]),
    )
    assert config.moving_camera.video is None
    assert config.moving_camera.frames is None
    assert config.moving_camera.image_topic == "/wand/image"

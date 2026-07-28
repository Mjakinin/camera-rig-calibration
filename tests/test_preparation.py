from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from camera_rig_calibration.config.models import (
    DatasetSettings,
    MethodSettings,
    MovingCameraSettings,
    ProjectSettings,
    RigConfig,
    SimulationSettings,
    StaticCameraSettings,
)
from camera_rig_calibration.input.preparation import (
    build_preparation_plan,
    finalize_dataset,
)
from camera_rig_calibration.experiments import input_fingerprint

from conftest import write_intrinsics


def test_input_fingerprint_uses_portable_relative_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prepared"
    frame = root / "raw_images" / "moving" / "frame_000000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    payload = {
        "static_camera_ids": [],
        "moving_camera_id": None,
        "sampling_hz": None,
        "files": [
            {
                "role": "prepared:raw_images/moving/frame_000000.png",
                "sha256": hashlib.sha256(b"frame").hexdigest(),
                "size_bytes": 5,
            }
        ],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    expected = (
        "input_"
        + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    )

    assert input_fingerprint(None, root) == expected


def _direct_config(tmp_path: Path) -> RigConfig:
    source = tmp_path / "source"
    source.mkdir()
    static_image = source / "outside-left.png"
    static_image.write_bytes(b"static-v1")
    static_info = source / "outside-left.json"
    write_intrinsics(static_info, "outside-left")
    frames = source / "moving"
    frames.mkdir()
    (frames / "capture-0042.png").write_bytes(b"moving-v1")
    moving_info = source / "moving.json"
    write_intrinsics(moving_info, "wand-camera")
    return RigConfig(
        project=ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "cache",
            output_root=tmp_path / "results",
        ),
        dataset=DatasetSettings(id="outside_day", input_root=source),
        static_cameras=[
            StaticCameraSettings(
                id="outside-left",
                images=[static_image],
                intrinsics=static_info,
            )
        ],
        moving_camera=MovingCameraSettings(
            id="wand-camera", frames=frames, intrinsics=moving_info
        ),
        methods=MethodSettings(enabled=["ap02"]),
    )


def test_input_content_gets_an_immutable_reusable_cache(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = _direct_config(tmp_path)
    first = build_preparation_plan(config, repository)
    manifest = finalize_dataset(config, first)
    assert manifest.files
    assert (first.dataset_root / "raw_images/static/outside-left.png").is_file()
    assert (first.dataset_root / "raw_images/moving/frame_000000.png").is_file()

    experiment = config.model_copy(
        update={
            "project": config.project.model_copy(update={"run_label": "experiment"}),
            "methods": config.methods.model_copy(update={"enabled": ["ap01"]}),
        },
        deep=True,
    )
    reused = build_preparation_plan(experiment, repository)
    assert reused.dataset_root == first.dataset_root
    assert reused.existing_manifest is not None
    assert reused.commands == []

    config.static_cameras[0].images[0].write_bytes(b"static-v2")
    changed = build_preparation_plan(config, repository)
    assert changed.dataset_root != first.dataset_root
    assert (
        first.dataset_root / "raw_images/static/outside-left.png"
    ).read_bytes() == b"static-v1"


def test_authoritative_prepared_dataset_reuses_original_manifest_fingerprint(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_config = _direct_config(tmp_path)
    source_plan = build_preparation_plan(source_config, repository)
    source_manifest = finalize_dataset(source_config, source_plan)
    exact = source_config.model_copy(
        update={
            "dataset": source_config.dataset.model_copy(
                update={"prepared_root": source_plan.dataset_root}
            ),
            "moving_camera": source_config.moving_camera.model_copy(
                update={
                    "intrinsics": None,
                    "intrinsics_profile": None,
                    "intrinsic_calibration_video": None,
                    "intrinsic_calibration_images": None,
                },
                deep=True,
            ),
        },
        deep=True,
    )

    root_manifest = source_plan.dataset_root / "dataset_manifest.json"
    metadata_manifest = (
        source_plan.dataset_root
        / "metadata"
        / "dataset_manifest.json"
    )
    metadata_manifest.parent.mkdir(parents=True, exist_ok=True)
    root_manifest.replace(metadata_manifest)
    reused = build_preparation_plan(exact, repository)

    assert reused.existing_manifest is not None
    assert input_fingerprint(
        reused.existing_manifest, reused.dataset_root
    ) == input_fingerprint(source_manifest, source_plan.dataset_root)


def test_checkerboard_image_folder_is_a_reproducible_intrinsics_source(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = _direct_config(tmp_path)
    checkerboard = tmp_path / "source/intrinsics_images"
    checkerboard.mkdir()
    (checkerboard / "view_01.png").write_bytes(b"checkerboard-1")
    (checkerboard / "view_02.png").write_bytes(b"checkerboard-2")
    moving = MovingCameraSettings(
        id=config.moving_camera.id,
        frames=config.moving_camera.frames,
        intrinsic_calibration_images=checkerboard,
        intrinsics_profile="phone_checkerboard",
    )
    config = config.model_copy(
        update={"moving_camera": moving},
        deep=True,
    )

    plan = build_preparation_plan(config, repository)

    command = next(
        item for item in plan.commands if item.stage_id == "prepare_moving_intrinsics"
    )
    assert "--images" in command.argv
    assert str(checkerboard.resolve()) in command.argv
    assert {
        role for role, _ in plan.source_files
        if role == "intrinsic_calibration_image"
    } == {"intrinsic_calibration_image"}


def test_static_video_materializes_deterministic_middle_frame(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = _direct_config(tmp_path)
    video = tmp_path / "source/static_v2/outside-left.avi"
    video.parent.mkdir()
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in (20, 80, 140, 200):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    static_camera = StaticCameraSettings(
        id="outside-left",
        video=video,
        intrinsics=config.static_cameras[0].intrinsics,
    )
    config = config.model_copy(
        update={"static_cameras": [static_camera]},
        deep=True,
    )

    plan = build_preparation_plan(config, repository)

    extracted = plan.acquisition_root / "raw_images/static/outside-left.png"
    metadata = (
        plan.acquisition_root
        / "metadata/static_video_extraction/outside-left.json"
    )
    assert extracted.is_file()
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["selection_policy"] == "middle_frame_with_first_frame_fallback"
    assert payload["selected_frame_index"] == 2


def test_alternate_moving_intrinsics_reuse_the_same_acquisition_frames(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    first_config = _direct_config(tmp_path)
    first = build_preparation_plan(first_config, repository)
    first_manifest = finalize_dataset(first_config, first)
    alternate_info = tmp_path / "source" / "moving_alternate.json"
    write_intrinsics(alternate_info, "wand-camera")
    payload = json.loads(alternate_info.read_text(encoding="utf-8"))
    payload["K"][0] = 510.0
    alternate_info.write_text(json.dumps(payload), encoding="utf-8")
    alternate_config = first_config.model_copy(
        update={
            "moving_camera": first_config.moving_camera.model_copy(
                update={"intrinsics": alternate_info}
            )
        },
        deep=True,
    )

    alternate = build_preparation_plan(alternate_config, repository)
    alternate_manifest = finalize_dataset(alternate_config, alternate)

    assert alternate.acquisition_root == first.acquisition_root
    assert alternate.dataset_root != first.dataset_root
    first_frame = first.dataset_root / "raw_images/moving/frame_000000.png"
    alternate_frame = (
        alternate.dataset_root / "raw_images/moving/frame_000000.png"
    )
    assert first_frame.read_bytes() == alternate_frame.read_bytes()
    assert first_frame.stat().st_ino == alternate_frame.stat().st_ino
    assert (
        first.dataset_root
        / "raw_images/camera_info/wand-camera.json"
    ).read_text() != (
        alternate.dataset_root
        / "raw_images/camera_info/wand-camera.json"
    ).read_text()
    assert input_fingerprint(
        first_manifest, first.dataset_root
    ) != input_fingerprint(alternate_manifest, alternate.dataset_root)


def test_prepared_frames_can_be_composed_with_new_intrinsics_without_overwrite(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_config = _direct_config(tmp_path)
    source_plan = build_preparation_plan(source_config, repository)
    finalize_dataset(source_config, source_plan)
    original_info = (
        source_plan.dataset_root
        / "raw_images/camera_info/wand-camera.json"
    )
    original_bytes = original_info.read_bytes()
    replacement = tmp_path / "replacement.json"
    write_intrinsics(replacement, "wand-camera")
    payload = json.loads(replacement.read_text(encoding="utf-8"))
    payload["K"][0] = 525.0
    replacement.write_text(json.dumps(payload), encoding="utf-8")
    prepared_config = source_config.model_copy(
        update={
            "dataset": source_config.dataset.model_copy(
                update={
                    "prepared_root": source_plan.dataset_root,
                    "input_root": source_plan.dataset_root,
                }
            ),
            "moving_camera": source_config.moving_camera.model_copy(
                update={
                    "video": None,
                    "frames": None,
                    "intrinsics": replacement,
                }
            ),
        },
        deep=True,
    )

    composed = build_preparation_plan(prepared_config, repository)
    finalize_dataset(prepared_config, composed)

    assert composed.dataset_root != source_plan.dataset_root
    assert original_info.read_bytes() == original_bytes
    assert (
        composed.dataset_root
        / "raw_images/moving/frame_000000.png"
    ).stat().st_ino == (
        source_plan.dataset_root
        / "raw_images/moving/frame_000000.png"
    ).stat().st_ino
    assert (
        composed.dataset_root
        / "raw_images/camera_info/wand-camera.json"
    ).read_bytes() != original_bytes


def test_simulation_preparation_plans_a_new_isolated_capture(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    world = tmp_path / "fixture.sdf"
    route = tmp_path / "route.json"
    world.write_text('<sdf version="1.8"><world name="fixture"/></sdf>')
    route.write_text('{"frames": [{"frame": 0}]}')
    config = RigConfig(
        project=ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "cache",
            output_root=tmp_path / "results",
        ),
        dataset=DatasetSettings(
            id="simulation_capture",
            category="simulation",
            scene_type="simulation",
        ),
        static_cameras=[
            StaticCameraSettings(
                id="outside-left",
                image_topic="/outside-left/image",
                camera_info_topic="/outside-left/camera_info",
            ),
            StaticCameraSettings(
                id="roof.camera",
                image_topic="/roof/camera/image",
                camera_info_topic="/roof/camera/camera_info",
            ),
        ],
        moving_camera=MovingCameraSettings(
            id="calibration-camera",
            image_topic="/calibration/image",
            camera_info_topic="/calibration/camera_info",
        ),
        simulation=SimulationSettings(
            enabled=True,
            preset="fixture",
            world=world,
            route=route,
        ),
        methods=MethodSettings(enabled=["ap02"]),
    )
    plan = build_preparation_plan(config, repository)
    assert plan.dataset_root != world.parent
    assert [command.stage_id for command in plan.commands] == ["prepare_simulation"]
    command = plan.commands[0]
    assert "camera_rig_calibration.input.simulation" in command.argv
    assert any(role == "simulation_world" for role, _ in plan.source_files)
    assert any(role == "simulation_route" for role, _ in plan.source_files)
    assert not (plan.dataset_root / "dataset_manifest.json").exists()


def test_simulation_mapping_preserves_explicit_camera_intrinsics(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    world = tmp_path / "fixture.sdf"
    route = tmp_path / "route.json"
    static_info = tmp_path / "static.json"
    moving_info = tmp_path / "moving.json"
    world.write_text(
        '<sdf version="1.8"><world name="fixture"/></sdf>',
        encoding="utf-8",
    )
    route.write_text(
        '{"frames": [{"frame": 0}, {"frame": 1}]}',
        encoding="utf-8",
    )
    write_intrinsics(static_info, "static_camera")
    write_intrinsics(moving_info, "moving_camera")
    config = RigConfig(
        project=ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "cache",
            output_root=tmp_path / "results",
        ),
        dataset=DatasetSettings(
            id="explicit_intrinsics",
            category="simulation",
            scene_type="simulation",
        ),
        static_cameras=[
            StaticCameraSettings(
                id="static_camera",
                intrinsics=static_info,
                image_topic="/static/image",
                camera_info_topic="/static/camera_info",
            )
        ],
        moving_camera=MovingCameraSettings(
            id="moving_camera",
            intrinsics=moving_info,
            image_topic="/moving/image",
            camera_info_topic="/moving/camera_info",
        ),
        simulation=SimulationSettings(
            enabled=True,
            world=world,
            route=route,
            moving_width=640,
            moving_height=360,
            moving_hfov_deg=100.0,
        ),
        methods=MethodSettings(enabled=["ap02"]),
    )

    plan = build_preparation_plan(config, repository)
    mapping = json.loads(
        (plan.dataset_root / "metadata/simulation_capture.json").read_text(
            encoding="utf-8"
        )
    )

    assert mapping["static_cameras"][0]["intrinsics_source"] == "provided"
    assert mapping["moving_camera"]["intrinsics_source"] == "provided"
    assert (
        plan.dataset_root / "raw_images/camera_info/static_camera.json"
    ).is_file()
    assert (
        plan.dataset_root / "raw_images/camera_info/moving_camera.json"
    ).is_file()
    assert mapping["moving_width"] == 640
    assert mapping["moving_height"] == 360
    assert mapping["moving_hfov_deg"] == 100.0


def test_simulation_capture_id_forces_a_distinct_input_cache(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    world = tmp_path / "fixture.sdf"
    route = tmp_path / "route.json"
    world.write_text(
        '<sdf version="1.8"><world name="fixture"/></sdf>',
        encoding="utf-8",
    )
    route.write_text(
        '{"frames": [{"frame": 0}, {"frame": 1}]}',
        encoding="utf-8",
    )
    common = {
        "project": ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "cache",
            output_root=tmp_path / "results",
        ),
        "dataset": DatasetSettings(
            id="simulation_capture",
            category="simulation",
            scene_type="simulation",
        ),
        "static_cameras": [
            StaticCameraSettings(
                id="static_camera",
                image_topic="/static/image",
                camera_info_topic="/static/camera_info",
            )
        ],
        "moving_camera": MovingCameraSettings(
            id="moving_camera",
            image_topic="/moving/image",
            camera_info_topic="/moving/camera_info",
        ),
        "simulation": SimulationSettings(
            enabled=True,
            preset="fixture",
            capture_id="capture_one",
            world=world,
            route=route,
        ),
        "methods": MethodSettings(enabled=["ap02"]),
    }
    first = RigConfig(**common)
    second = first.model_copy(
        update={
            "simulation": first.simulation.model_copy(
                update={"capture_id": "capture_two"}
            )
        },
        deep=True,
    )

    assert (
        build_preparation_plan(first, repository).dataset_root
        != build_preparation_plan(second, repository).dataset_root
    )

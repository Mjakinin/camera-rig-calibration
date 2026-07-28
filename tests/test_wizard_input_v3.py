from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
import typer
from rich.console import Console

import camera_rig_calibration.wizard as wizard_module
from camera_rig_calibration.config.models import MovingCameraSettings
from camera_rig_calibration.input.video_geometry import VideoGeometry
from camera_rig_calibration.inventory import PreparedDatasetSummary
from camera_rig_calibration.input.topics import McapTopic
from camera_rig_calibration.wizard import (
    _aruco_experiment_id,
    _camera_id_from_ros_topic,
    _checkerboard_sources,
    _data_local_input_root,
    _detected_static_camera_groups,
    _detected_static_pairs,
    _mcap_camera_sources,
    _moving_source,
    _prepared_input,
    _stored_prepared_marker_settings,
    _prompt_enum_choice,
    _real_data_input,
    _related_camera_info_topics,
)


def _intrinsics(path: Path, camera_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "camera_name": camera_id,
                "width": 640,
                "height": 480,
                "K": [500, 0, 320, 0, 500, 240, 0, 0, 1],
                "D": [0, 0, 0, 0, 0],
            }
        ),
        encoding="utf-8",
    )


def _mock_video_geometry(monkeypatch) -> None:
    monkeypatch.setattr(
        wizard_module,
        "probe_video_geometry",
        lambda path: VideoGeometry(
            encoded_width=1920,
            encoded_height=1080,
            display_rotation_degrees=-90,
            output_width=1080,
            output_height=1920,
        ),
    )


def test_prepared_manifest_camera_binding_needs_no_identity_or_hz_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "prepared"
    raw = root / "raw_images"
    for camera in ("left", "right"):
        (raw / "static").mkdir(parents=True, exist_ok=True)
        (raw / "static" / f"{camera}.png").write_bytes(b"fixture")
        _intrinsics(raw / "camera_info" / f"{camera}.json", camera)
    (raw / "moving").mkdir(parents=True)
    (raw / "moving/frame_000001.png").write_bytes(b"fixture")
    _intrinsics(raw / "camera_info/wand.json", "wand")
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "prepared",
                "sampling_hz": 1.0,
                "static_cameras": [{"id": "left"}, {"id": "right"}],
                "moving_camera": {"id": "wand"},
            }
        ),
        encoding="utf-8",
    )
    prompts: list[str] = []

    def prompt(text, *args, **kwargs):
        prompts.append(text)
        return 1

    monkeypatch.setattr(typer, "prompt", prompt)
    summary = PreparedDatasetSummary(
        id="prepared",
        display_name="prepared",
        category="real_vehicle",
        description="fixture",
        path=root,
        static_camera_ids=("left", "right"),
        moving_frames=1,
        has_results=False,
    )

    _, cameras, moving, _ = _prepared_input(
        Console(file=StringIO(), force_terminal=False),
        tmp_path,
        [summary],
    )

    assert [camera.id for camera in cameras] == ["left", "right"]
    assert moving.id == "wand"
    assert prompts == ["Prepared dataset number (0 = back)"]


def test_prepared_dataset_reuses_its_versioned_detector_contract(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    observations = prepared / "observations"
    observations.mkdir(parents=True)
    (observations / "detection_config.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "markers": {
                    "dictionary": "DICT_4X4_50",
                    "length_m": 0.17,
                    "accepted_ids": "all_detected",
                    "detection_mode": "high_sensitivity",
                },
            }
        ),
        encoding="utf-8",
    )

    settings = _stored_prepared_marker_settings(prepared)

    assert settings.detection_mode == "high_sensitivity"
    assert settings.dictionary == "DICT_4X4_50"


def test_aruco_experiment_id_replaces_an_existing_detector_suffix() -> None:
    sensitive = "capture__aruco_high_sensitivity"

    assert _aruco_experiment_id(sensitive, "high_sensitivity") == sensitive
    assert _aruco_experiment_id(sensitive, "baseline") == "capture"
    assert (
        _aruco_experiment_id(sensitive, "subpixel_refined")
        == "capture__aruco_subpixel_refined"
    )


def test_data_local_root_is_used_without_a_folder_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    landing = tmp_path / "data_local"
    (landing / "nested").mkdir(parents=True)
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: pytest.fail(
            "selecting the canonical data_local root must not prompt"
        ),
    )

    assert _data_local_input_root(tmp_path) == landing.resolve()


def test_missing_data_local_fails_without_creating_or_prompting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: pytest.fail(
            "missing data_local must not open a manual path prompt"
        ),
    )

    with pytest.raises(RuntimeError, match="data_local"):
        _data_local_input_root(tmp_path)

    assert not (tmp_path / "data_local").exists()


def test_empty_data_local_fails_before_any_configuration_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "data_local").mkdir()
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: pytest.fail(
            "empty data_local must fail before configuration prompts"
        ),
    )

    with pytest.raises(RuntimeError, match="No moving-camera"):
        _real_data_input(
            tmp_path,
            Console(file=StringIO(), force_terminal=False),
        )


def test_target_hz_is_prompted_for_new_video_but_not_prepared_frames(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_video_geometry(monkeypatch)
    video_root = tmp_path / "video"
    video_root.mkdir()
    (video_root / "moving.mp4").write_bytes(b"fixture")
    _intrinsics(video_root / "moving_intrinsics.json", "wand")
    video_responses = iter(["1", 1, 1, 5.0])
    video_prompts: list[str] = []

    def video_prompt(text, *args, **kwargs):
        video_prompts.append(text)
        value = next(video_responses)
        requested_type = kwargs.get("type")
        return requested_type(value) if requested_type in {int, float} else value

    monkeypatch.setattr(typer, "prompt", video_prompt)
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)
    _, video_sampling, _ = _moving_source(
        Console(file=StringIO(), force_terminal=False),
        video_root,
    )

    frame_root = tmp_path / "frames"
    (frame_root / "moving_frames").mkdir(parents=True)
    (frame_root / "moving_frames/frame_000001.png").write_bytes(b"fixture")
    _intrinsics(frame_root / "moving_intrinsics.json", "wand")
    frame_responses = iter(["2", 1, 1])
    frame_prompts: list[str] = []

    def frame_prompt(text, *args, **kwargs):
        frame_prompts.append(text)
        value = next(frame_responses)
        requested_type = kwargs.get("type")
        return requested_type(value) if requested_type in {int, float} else value

    monkeypatch.setattr(typer, "prompt", frame_prompt)
    _, frame_sampling, _ = _moving_source(
        Console(file=StringIO(), force_terminal=False),
        frame_root,
    )

    assert video_sampling.target_hz == 5.0
    assert any("sampling rate" in text for text in video_prompts)
    assert frame_sampling.target_hz is None
    assert not any("sampling rate" in text for text in frame_prompts)


def test_declining_advanced_video_settings_skips_checkerboard_prompts(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_video_geometry(monkeypatch)
    input_root = tmp_path / "videos"
    input_root.mkdir()
    (input_root / "moving.mp4").write_bytes(b"fixture")
    (input_root / "checkerboard.mp4").write_bytes(b"fixture")
    responses = iter(["1", 2, 1, 1, "checkerboard_4k", 2.0, 1])
    prompts: list[str] = []
    confirmations: list[str] = []
    stream = StringIO()

    def prompt(text, *args, **kwargs):
        prompts.append(text)
        value = next(responses)
        requested_type = kwargs.get("type")
        return requested_type(value) if requested_type in {int, float} else value

    def confirm(text, *args, **kwargs):
        confirmations.append(text)
        return False

    monkeypatch.setattr(typer, "prompt", prompt)
    monkeypatch.setattr(typer, "confirm", confirm)

    moving, sampling, _ = _moving_source(
        Console(file=stream, force_terminal=False),
        input_root,
    )

    assert confirmations == [
        "Open advanced video sampling and checkerboard settings?"
    ]
    assert not any("Inner corner" in text for text in prompts)
    assert not any("Maximum selected views" in text for text in prompts)
    assert moving.checkerboard_columns == 8
    assert moving.checkerboard_rows == 6
    assert moving.intrinsic_maximum_views == 80
    assert moving.intrinsic_minimum_frame_gap == 5
    assert moving.intrinsic_minimum_detections == 20
    assert sampling.target_hz == 2.0
    assert "recommended defaults" in stream.getvalue()


def test_moving_and_checkerboard_image_folders_are_selected_independently(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "data_local/recording"
    moving_frames = root / "moving_frames"
    checkerboard = root / "intrinsics_images"
    moving_frames.mkdir(parents=True)
    checkerboard.mkdir()
    (moving_frames / "frame_0001.png").write_bytes(b"moving")
    (checkerboard / "view_0001.png").write_bytes(b"checkerboard")
    responses = iter(["2", 1, 1, "phone_images", 1])

    def prompt(text, *args, **kwargs):
        value = next(responses)
        requested_type = kwargs.get("type")
        return requested_type(value) if requested_type in {int, float} else value

    monkeypatch.setattr(typer, "prompt", prompt)
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)

    moving, sampling, _ = _moving_source(
        Console(file=StringIO(), force_terminal=False),
        root,
    )

    assert moving.frames == moving_frames.resolve()
    assert moving.intrinsic_calibration_video is None
    assert moving.intrinsic_calibration_images == checkerboard.resolve()
    assert moving.intrinsics_profile == "phone_images"
    assert moving.intrinsic_minimum_frame_gap == 0
    assert sampling.target_hz is None


def test_matcher_choice_is_shown_before_prompt_and_invalid_input_reprompts(
    monkeypatch, capsys
) -> None:
    responses = iter(["unknown", "2"])
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: next(responses),
    )

    selected = _prompt_enum_choice(
        "Matcher",
        "exhaustive",
        (
            ("exhaustive", "compare every image pair"),
            ("sequential", "compare temporal neighbors"),
        ),
    )

    output = capsys.readouterr().out
    assert "1. exhaustive (current)" in output
    assert "2. sequential" in output
    assert "Choose 1-2" in output
    assert selected == "sequential"


def test_ros_color_image_is_paired_with_its_camera_info_and_generic_id() -> None:
    image = McapTopic(
        "/edge_5/camera/color/image_raw/compressed",
        "sensor_msgs/msg/CompressedImage",
    )
    infos = [
        McapTopic(
            "/edge_5/camera/color/camera_info",
            "sensor_msgs/msg/CameraInfo",
        ),
        McapTopic(
            "/edge_5/camera/depth/camera_info",
            "sensor_msgs/msg/CameraInfo",
        ),
    ]

    assert _related_camera_info_topics(image, infos) == [
        "/edge_5/camera/color/camera_info"
    ]
    assert _camera_id_from_ros_topic(image.name) == "cam_edge_5"


def test_ros_bag_defaults_to_color_streams_and_keeps_camera_info_automatic(
    tmp_path: Path, monkeypatch
) -> None:
    topics: list[McapTopic] = []
    for edge in ("edge_0", "edge_1"):
        topics.extend(
            [
                McapTopic(
                    f"/{edge}/camera/depth/image_rect_raw/compressed",
                    "sensor_msgs/msg/CompressedImage",
                ),
                McapTopic(
                    f"/{edge}/camera/depth/camera_info",
                    "sensor_msgs/msg/CameraInfo",
                ),
                McapTopic(
                    f"/{edge}/camera/color/image_raw/compressed",
                    "sensor_msgs/msg/CompressedImage",
                ),
                McapTopic(
                    f"/{edge}/camera/color/camera_info",
                    "sensor_msgs/msg/CameraInfo",
                ),
            ]
        )
    prompts: list[str] = []

    def prompt(text, *args, **kwargs):
        prompts.append(text)
        return kwargs.get("default", "")

    monkeypatch.setattr(
        "camera_rig_calibration.wizard.list_mcap_topics",
        lambda path: topics,
    )
    monkeypatch.setattr(typer, "prompt", prompt)

    cameras, moving = _mcap_camera_sources(
        Console(file=StringIO(), force_terminal=False),
        tmp_path / "recording.mcap",
        moving_from_recording=False,
    )

    assert moving is None
    assert [camera.id for camera in cameras] == ["cam_edge_0", "cam_edge_1"]
    assert [camera.image_topic for camera in cameras] == [
        "/edge_0/camera/color/image_raw/compressed",
        "/edge_1/camera/color/image_raw/compressed",
    ]
    assert [camera.camera_info_topic for camera in cameras] == [
        "/edge_0/camera/color/camera_info",
        "/edge_1/camera/color/camera_info",
    ]
    assert prompts == ["Static image topic numbers (comma-separated)"]


def test_recommended_direct_frame_layout_is_detected_without_hardcoded_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data_local" / "outside_day"
    (root / "moving_frames").mkdir(parents=True)
    (root / "static").mkdir()
    (root / "intrinsics").mkdir()
    (root / "moving_frames/frame_000001.png").write_bytes(b"moving")
    (root / "static/front_left.png").write_bytes(b"static")
    _intrinsics(root / "intrinsics/front_left.yaml", "front_left")
    _intrinsics(
        root / "intrinsics/moving_calib_camera.yaml",
        "moving_calib_camera",
    )
    moving = MovingCameraSettings(
        id="moving_calib_camera",
        frames=root / "moving_frames",
        intrinsics=root / "intrinsics/moving_calib_camera.yaml",
    )

    pairs = _detected_static_pairs(root, moving)

    assert pairs == [
        (
            (root / "static/front_left.png").resolve(),
            (root / "intrinsics/front_left.yaml").resolve(),
        )
    ]


def test_nested_static_camera_image_folders_are_grouped_with_intrinsics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data_local" / "outside_day"
    camera_images = root / "static_v2/front_left/images"
    camera_images.mkdir(parents=True)
    for name in ("capture_01.png", "capture_02.jpg"):
        (camera_images / name).write_bytes(b"static")
    _intrinsics(root / "intrinsics/front_left.yaml", "front_left")
    moving_frames = root / "moving_frames"
    moving_frames.mkdir()
    (moving_frames / "frame_0001.png").write_bytes(b"moving")

    groups = _detected_static_camera_groups(
        root,
        MovingCameraSettings(id="moving", frames=moving_frames),
    )

    assert len(groups) == 1
    camera_id, images, video, intrinsics = groups[0]
    assert camera_id == "front_left"
    assert video is None
    assert images == sorted(
        [
            (camera_images / "capture_01.png").resolve(),
            (camera_images / "capture_02.jpg").resolve(),
        ]
    )
    assert intrinsics == (root / "intrinsics/front_left.yaml").resolve()


def test_static_video_in_versioned_folder_is_bound_to_camera_intrinsics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data_local/recording"
    video = root / "static_v2/front_left/IMG_1001.mov"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _intrinsics(root / "intrinsics/front_left.yaml", "front_left")

    groups = _detected_static_camera_groups(
        root,
        MovingCameraSettings(id="moving"),
    )

    assert groups == [
        (
            "front_left",
            [],
            video.resolve(),
            (root / "intrinsics/front_left.yaml").resolve(),
        )
    ]


def test_generic_video_inside_intrinsics_variant_folder_is_checkerboard_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data_local/recording"
    video = root / "intrinsics_iphone_v2/IMG_1001.mov"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    sources = _checkerboard_sources(root)

    assert sources == [
        ("video", video.resolve(), f"video: {video.resolve()}")
    ]

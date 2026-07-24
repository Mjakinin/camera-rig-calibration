from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config.models import RigConfig, StaticCameraSettings
from ..contracts import CommandSpec
from ..dataset.manifest import (
    CameraManifest,
    DatasetManifest,
    FileProvenance,
    load_dataset_manifest,
    save_dataset_manifest,
)
from ..intrinsics_profiles import resolve_intrinsic_profile


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class PreparationPlan:
    dataset_root: Path
    commands: list[CommandSpec] = field(default_factory=list)
    source_files: list[tuple[str, Path]] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    prepared_input: bool = False
    existing_manifest: DatasetManifest | None = None
    acquisition_root: Path | None = None
    acquisition_fingerprint: str | None = None
    moving_intrinsics_override: bool = False


def _image_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.resolve().iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.samefile(destination):
            return
        return
    shutil.copy2(source, destination)


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _materialize_tree(
    source: Path,
    destination: Path,
    *,
    excluded: set[Path] | None = None,
) -> None:
    excluded_resolved = {path.resolve() for path in (excluded or set())}
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file() and path.resolve() not in excluded_resolved:
            _link_or_copy_file(path, target)


def _copy_image_as_png(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".png":
        _copy_file(source, destination)
        return
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            f"OpenCV is required to convert {source.suffix} images to PNG"
        ) from exc
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None or not cv2.imwrite(str(destination), image):
        raise RuntimeError(f"Could not convert image to PNG: {source}")


def _extract_static_video_frame(
    source: Path,
    destination: Path,
    metadata_path: Path,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to extract a static-camera video frame"
        ) from exc
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open static-camera video: {source}")
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_index = max(0, reported_count // 2)
    if selected_index:
        capture.set(cv2.CAP_PROP_POS_FRAMES, selected_index)
    ok, frame = capture.read()
    if not ok and selected_index:
        selected_index = 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"Could not decode a representative frame from static video: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Could not write extracted static frame: {destination}")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "source_video": str(source),
                "reported_frame_count": reported_count,
                "selection_policy": "middle_frame_with_first_frame_fallback",
                "selected_frame_index": selected_index,
                "output": str(destination),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _normalize_intrinsics(source: Path, destination: Path, camera_id: str) -> None:
    if destination.exists():
        return
    if not source.is_file():
        raise FileNotFoundError(f"Intrinsics not found for '{camera_id}': {source}")
    text = source.read_text(encoding="utf-8")
    payload = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Intrinsic file must contain a mapping: {source}")
    matrix = payload.get("K", payload.get("k", payload.get("camera_matrix")))
    if isinstance(matrix, dict):
        matrix = matrix.get("data")
    distortion = payload.get(
        "D", payload.get("d", payload.get("distortion_coefficients", []))
    )
    if isinstance(distortion, dict):
        distortion = distortion.get("data", [])
    width = payload.get("width", payload.get("image_width", 0))
    height = payload.get("height", payload.get("image_height", 0))
    if not isinstance(matrix, list) or len(matrix) != 9:
        raise ValueError(f"Camera matrix K must contain nine values: {source}")
    if not isinstance(distortion, list):
        raise ValueError(f"Distortion coefficients must be a list: {source}")
    normalized = dict(payload)
    normalized.update(
        {
            "camera_name": camera_id,
            "width": int(width),
            "height": int(height),
            "image_width": int(width),
            "image_height": int(height),
            "K": [float(value) for value in matrix],
            "k": [float(value) for value in matrix],
            "D": [float(value) for value in distortion],
            "d": [float(value) for value in distortion],
            "distortion_model": str(payload.get("distortion_model", "plumb_bob")),
            "rigcal_source": str(source.resolve()),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


def _copy_static_camera(
    camera: StaticCameraSettings,
    dataset_root: Path,
    source_files: list[tuple[str, Path]],
) -> None:
    if not camera.images:
        if camera.video is not None:
            source = camera.video.resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Static video not found: {source}")
            _extract_static_video_frame(
                source,
                dataset_root
                / "raw_images"
                / "static"
                / f"{camera.id}.png",
                dataset_root
                / "metadata"
                / "static_video_extraction"
                / f"{camera.id}.json",
            )
            source_files.append((f"static_video:{camera.id}", source))
            if camera.intrinsics:
                _normalize_intrinsics(
                    camera.intrinsics.resolve(),
                    dataset_root
                    / "raw_images"
                    / "camera_info"
                    / f"{camera.id}.json",
                    camera.id,
                )
                source_files.append(
                    (f"static_intrinsics:{camera.id}", camera.intrinsics.resolve())
                )
            return
        if camera.image_topic:
            if camera.intrinsics is not None:
                _normalize_intrinsics(
                    camera.intrinsics.resolve(),
                    dataset_root
                    / "raw_images"
                    / "camera_info"
                    / f"{camera.id}.json",
                    camera.id,
                )
                source_files.append(
                    (f"static_intrinsics:{camera.id}", camera.intrinsics.resolve())
                )
            return
        raise ValueError(f"Static camera '{camera.id}' has no image or MCAP topic")
    images = [path.resolve() for path in camera.images]
    for source in images:
        if not source.is_file():
            raise FileNotFoundError(f"Static image not found: {source}")
    _copy_image_as_png(
        images[0], dataset_root / "raw_images" / "static" / f"{camera.id}.png"
    )
    source_files.append((f"static_image:{camera.id}", images[0]))
    if len(images) > 1:
        for index, source in enumerate(images):
            destination = (
                dataset_root
                / "raw_images"
                / "static_multi"
                / camera.id
                / f"frame_{index:06d}.png"
            )
            _copy_image_as_png(source, destination)
            source_files.append((f"static_candidate:{camera.id}", source))
    if camera.intrinsics:
        _normalize_intrinsics(
            camera.intrinsics.resolve(),
            dataset_root / "raw_images" / "camera_info" / f"{camera.id}.json",
            camera.id,
        )
        source_files.append((f"static_intrinsics:{camera.id}", camera.intrinsics.resolve()))


def _copy_moving_frames(config: RigConfig, dataset_root: Path, files: list[tuple[str, Path]]) -> None:
    source_root = config.moving_camera.frames
    if source_root is None:
        return
    candidates = [
        path
        for path in sorted(source_root.resolve().iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not candidates:
        raise RuntimeError(f"No moving frames found in {source_root}")
    maximum = config.sampling.maximum_frames or len(candidates)
    for index, source in enumerate(candidates[:maximum]):
        destination = dataset_root / "raw_images" / "moving" / f"frame_{index:06d}.png"
        _copy_image_as_png(source, destination)
        files.append(("moving_frame", source))


def _configured_source_files(config: RigConfig) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for camera in config.static_cameras:
        for index, image in enumerate(camera.images):
            role = f"static_image:{camera.id}" if index == 0 else f"static_candidate:{camera.id}"
            sources.append((role, image.resolve()))
        if camera.intrinsics is not None:
            sources.append((f"static_intrinsics:{camera.id}", camera.intrinsics.resolve()))
        if camera.video is not None:
            sources.append((f"static_video:{camera.id}", camera.video.resolve()))
    moving = config.moving_camera
    if moving.video is not None:
        sources.append(("moving_video", moving.video.resolve()))
    if moving.frames is not None:
        frames = [
            path
            for path in sorted(moving.frames.resolve().iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        maximum = config.sampling.maximum_frames or len(frames)
        sources.extend(("moving_frame", path) for path in frames[:maximum])
    if moving.intrinsics is not None:
        sources.append(("moving_intrinsics", moving.intrinsics.resolve()))
    if moving.intrinsic_calibration_video is not None:
        sources.append(
            ("intrinsic_calibration_video", moving.intrinsic_calibration_video.resolve())
        )
    if moving.intrinsic_calibration_images is not None:
        sources.extend(
            ("intrinsic_calibration_image", path)
            for path in _image_files(moving.intrinsic_calibration_images)
        )
    if config.mcap.path is not None:
        sources.append(("mcap", config.mcap.path.resolve()))
    if config.simulation.enabled:
        if config.simulation.world is not None:
            sources.append(("simulation_world", config.simulation.world.resolve()))
        if config.simulation.route is not None:
            sources.append(("simulation_route", config.simulation.route.resolve()))
    for _, path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"Configured input file not found: {path}")
    return sources


def _prepared_source_files(config: RigConfig, root: Path) -> list[tuple[str, Path]]:
    raw = root / "raw_images"
    sources = []
    for camera in config.static_cameras:
        sources.extend(
            (f"prepared_static_image:{camera.id}", path)
            for path in sorted((raw / "static").glob(f"{camera.id}.*"))
        )
        info = raw / "camera_info" / f"{camera.id}.json"
        if info.is_file():
            sources.append((f"prepared_static_intrinsics:{camera.id}", info))
    sources.extend(
        ("prepared_moving_frame", path)
        for path in sorted((raw / "moving").glob("frame_*.*"))
    )
    moving_info = raw / "camera_info" / f"{config.moving_camera.id}.json"
    if moving_info.is_file():
        sources.append(("prepared_moving_intrinsics", moving_info))
    return sources


def _hash_sources(sources: list[tuple[str, Path]]) -> dict[str, str]:
    return {
        f"{role}\0{path}": _sha256(path)
        for role, path in sources
        if path.is_file()
    }


def _preparation_fingerprint(config: RigConfig) -> str:
    payload = {
        "dataset": config.dataset.model_dump(mode="json", exclude={"input_root"}),
        "static_cameras": [
            camera.model_dump(mode="json") for camera in config.static_cameras
        ],
        "moving_camera": config.moving_camera.model_dump(mode="json"),
        "mcap": config.mcap.model_dump(mode="json"),
        "simulation": config.simulation.model_dump(mode="json"),
        "sampling": config.sampling.model_dump(mode="json"),
        "marker_dictionary": config.markers.dictionary,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fingerprint_sources(
    payload: dict[str, object],
    sources: list[tuple[str, Path]],
) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for role, path in sorted(sources, key=lambda item: (item[0], str(item[1]))):
        if path.is_file():
            digest.update(role.encode("utf-8"))
            digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _resolved_moving_intrinsics(
    config: RigConfig, repository_root: Path
) -> tuple[RigConfig, Path | None]:
    moving = config.moving_camera
    if (
        moving.intrinsics_profile
        and moving.intrinsics is None
        and moving.intrinsic_calibration_video is None
        and moving.intrinsic_calibration_images is None
    ):
        profile = resolve_intrinsic_profile(
            repository_root, moving.intrinsics_profile
        )
        moving = moving.model_copy(update={"intrinsics": profile.intrinsics})
        return (
            config.model_copy(update={"moving_camera": moving}, deep=True),
            profile.intrinsics,
        )
    return config, moving.intrinsics


def _real_acquisition_payload(config: RigConfig) -> dict[str, object]:
    moving = config.moving_camera.model_dump(
        mode="json",
        exclude={
            "intrinsics",
            "intrinsics_profile",
            "intrinsic_calibration_video",
            "intrinsic_calibration_images",
            "intrinsic_scan",
            "checkerboard_columns",
            "checkerboard_rows",
            "intrinsic_maximum_views",
            "intrinsic_minimum_frame_gap",
            "intrinsic_minimum_detections",
        },
    )
    return {
        "static_cameras": [
            camera.model_dump(mode="json") for camera in config.static_cameras
        ],
        "moving_camera": moving,
        "mcap": config.mcap.model_dump(mode="json"),
        "sampling": config.sampling.model_dump(mode="json"),
        "marker_dictionary": config.markers.dictionary,
        "contract": "rigcal_real_acquisition_v1",
    }


def _real_acquisition_sources(
    config: RigConfig, prepared_root: Path | None
) -> list[tuple[str, Path]]:
    if prepared_root is not None:
        sources = _prepared_source_files(config, prepared_root)
        moving_info = (
            prepared_root
            / "raw_images"
            / "camera_info"
            / f"{config.moving_camera.id}.json"
        ).resolve()
        return [
            item
            for item in sources
            if not (
                item[0] == "prepared_moving_intrinsics"
                and item[1].resolve() == moving_info
            )
        ]
    return [
        item
        for item in _configured_source_files(config)
        if item[0]
        not in {
            "moving_intrinsics",
            "intrinsic_calibration_video",
            "intrinsic_calibration_image",
        }
    ]


def _build_real_preparation_plan(
    config: RigConfig,
    repository_root: Path,
) -> PreparationPlan:
    config, selected_intrinsics = _resolved_moving_intrinsics(
        config, repository_root
    )
    moving = config.moving_camera
    prepared_root = config.dataset.prepared_root
    if prepared_root is not None:
        prepared_root = prepared_root.resolve()
        if prepared_root.name == "raw_images":
            prepared_root = prepared_root.parent
    has_override = (
        selected_intrinsics is not None
        or moving.intrinsic_calibration_video is not None
        or moving.intrinsic_calibration_images is not None
        or moving.intrinsics_profile is not None
    )
    if prepared_root is not None and not has_override:
        sources = _prepared_source_files(config, prepared_root)
        return PreparationPlan(
            prepared_root,
            source_files=sources,
            source_hashes=_hash_sources(sources),
            prepared_input=True,
        )

    acquisition_sources = _real_acquisition_sources(config, prepared_root)
    acquisition_fingerprint = _fingerprint_sources(
        _real_acquisition_payload(config), acquisition_sources
    )
    acquisition_root = (
        prepared_root
        if prepared_root is not None
        else config.project.dataset_cache_root.resolve()
        / "_acquisitions"
        / acquisition_fingerprint[:12]
    )
    intrinsic_sources: list[tuple[str, Path]] = []
    if selected_intrinsics is not None:
        intrinsic_sources.append(
            ("moving_intrinsics", selected_intrinsics.resolve())
        )
    if moving.intrinsic_calibration_video is not None:
        intrinsic_sources.append(
            (
                "intrinsic_calibration_video",
                moving.intrinsic_calibration_video.resolve(),
            )
        )
    if moving.intrinsic_calibration_images is not None:
        intrinsic_sources.extend(
            ("intrinsic_calibration_image", path)
            for path in _image_files(moving.intrinsic_calibration_images)
        )
    if not intrinsic_sources and prepared_root is None:
        # A ROS recording may provide the moving CameraInfo together with its
        # moving image topic. Its MCAP hash is already part of the acquisition.
        intrinsic_identity: dict[str, object] = {
            "source": "acquisition_camera_info"
        }
    else:
        intrinsic_identity = {
            "sources": [
                {"role": role, "sha256": _sha256(path)}
                for role, path in intrinsic_sources
            ],
            "profile": moving.intrinsics_profile,
            "scan": moving.intrinsic_scan.model_dump(mode="json"),
            "checkerboard": {
                "columns": moving.checkerboard_columns,
                "rows": moving.checkerboard_rows,
                "maximum_views": moving.intrinsic_maximum_views,
                "minimum_frame_gap": moving.intrinsic_minimum_frame_gap,
                "minimum_detections": moving.intrinsic_minimum_detections,
            },
        }
    composition_payload = {
        "acquisition_fingerprint": acquisition_fingerprint,
        "moving_camera_id": moving.id,
        "intrinsics": intrinsic_identity,
        "contract": "rigcal_real_composition_v1",
    }
    composition_fingerprint = hashlib.sha256(
        json.dumps(
            composition_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    dataset_root = (
        config.project.dataset_cache_root.resolve()
        / config.dataset.id
        / composition_fingerprint[:12]
    )
    all_sources = [*acquisition_sources, *intrinsic_sources]
    all_hashes = _hash_sources(all_sources)
    existing_path = dataset_root / "dataset_manifest.json"
    if existing_path.is_file():
        return PreparationPlan(
            dataset_root,
            source_files=all_sources,
            source_hashes=all_hashes,
            existing_manifest=load_dataset_manifest(existing_path),
            acquisition_root=acquisition_root,
            acquisition_fingerprint=acquisition_fingerprint,
            moving_intrinsics_override=has_override,
        )

    for directory in (
        dataset_root / "raw_images" / "static",
        dataset_root / "raw_images" / "moving",
        dataset_root / "raw_images" / "camera_info",
        dataset_root / "metadata",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    acquisition_ready = prepared_root is not None or (
        acquisition_root / "ACQUISITION_COMPLETE.json"
    ).is_file()
    commands: list[CommandSpec] = []
    if not acquisition_ready:
        for directory in (
            acquisition_root / "raw_images" / "static",
            acquisition_root / "raw_images" / "moving",
            acquisition_root / "raw_images" / "camera_info",
            acquisition_root / "metadata",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for camera in config.static_cameras:
            _copy_static_camera(camera, acquisition_root, [])
        _copy_moving_frames(config, acquisition_root, [])
        if config.mcap.path is not None:
            mapping_path = (
                acquisition_root / "metadata" / "mcap_topic_mapping.json"
            )
            mapping = {
                "marker_dictionary": config.markers.dictionary,
                "save_all_candidates": config.mcap.save_all_candidates,
                "moving_sampling_hz": config.sampling.target_hz,
                "cameras": [
                    {
                        "id": camera.id,
                        "image_topic": camera.image_topic,
                        "camera_info_topic": camera.camera_info_topic,
                    }
                    for camera in config.static_cameras
                ],
                "moving_camera": (
                    {
                        "id": moving.id,
                        "image_topic": moving.image_topic,
                        "camera_info_topic": moving.camera_info_topic,
                    }
                    if moving.video is None
                    and moving.frames is None
                    and moving.image_topic is not None
                    else None
                ),
            }
            mapping_path.write_text(
                json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
            )
            commands.append(
                CommandSpec(
                    "prepare_mcap",
                    "Extract selected camera inputs from MCAP",
                    (
                        sys.executable,
                        "-m",
                        "camera_rig_calibration.input.mcap",
                        "--mcap",
                        str(config.mcap.path.resolve()),
                        "--dataset",
                        str(acquisition_root),
                        "--mapping",
                        str(mapping_path),
                    ),
                    repository_root,
                    acquisition_root,
                )
            )
        if moving.video is not None:
            argv = [
                sys.executable,
                str(
                    repository_root
                    / "run/real_vehicle_data/01_extract_moving_video_3hz.py"
                ),
                "--video",
                str(moving.video.resolve()),
                "--dataset",
                str(acquisition_root),
                "--target-fps",
                str(config.sampling.target_hz),
                "--start-s",
                str(config.sampling.start_seconds),
            ]
            if config.sampling.end_seconds is not None:
                argv += ["--end-s", str(config.sampling.end_seconds)]
            if config.sampling.maximum_frames is not None:
                argv += ["--max-frames", str(config.sampling.maximum_frames)]
            argv.append("--overwrite")
            commands.append(
                CommandSpec(
                    "prepare_moving_frames",
                    "Extract moving-camera video frames",
                    tuple(argv),
                    repository_root,
                    acquisition_root / "raw_images" / "moving",
                )
            )

    info_destination = (
        dataset_root / "raw_images" / "camera_info" / f"{moving.id}.json"
    )
    if selected_intrinsics is not None:
        _normalize_intrinsics(
            selected_intrinsics.resolve(), info_destination, moving.id
        )
    elif (
        moving.intrinsic_calibration_video is not None
        or moving.intrinsic_calibration_images is not None
    ):
        calibration_source = (
            moving.intrinsic_calibration_video
            or moving.intrinsic_calibration_images
        )
        assert calibration_source is not None
        source_option = (
            "--video"
            if moving.intrinsic_calibration_video is not None
            else "--images"
        )
        profile_id = (
            moving.intrinsics_profile
            or calibration_source.stem
        ).split("@", 1)[0]
        commands.append(
            CommandSpec(
                "prepare_moving_intrinsics",
                "Calibrate moving-camera intrinsics",
                (
                    sys.executable,
                    "-m",
                    "camera_rig_calibration.input.intrinsics",
                    "--script",
                    str(
                        repository_root
                        / "run/real_vehicle_data/02_calibrate_intrinsics_from_video.py"
                    ),
                    source_option,
                    str(calibration_source.resolve()),
                    "--work-directory",
                    str(dataset_root / "metadata" / "intrinsic_calibration"),
                    "--destination",
                    str(info_destination),
                    "--camera-id",
                    moving.id,
                    "--repository",
                    str(repository_root),
                    "--profile-id",
                    profile_id,
                    "--cols",
                    str(moving.checkerboard_columns),
                    "--rows",
                    str(moving.checkerboard_rows),
                    "--max-views",
                    str(moving.intrinsic_maximum_views),
                    "--minimum-frame-gap",
                    str(moving.intrinsic_minimum_frame_gap),
                    "--minimum-detections",
                    str(moving.intrinsic_minimum_detections),
                    "--scan-mode",
                    moving.intrinsic_scan.mode,
                    "--scan-target-hz",
                    str(moving.intrinsic_scan.target_hz),
                    "--preview-max-dimension",
                    str(moving.intrinsic_scan.preview_max_dimension),
                ),
                repository_root,
                info_destination.parent,
            )
        )
    return PreparationPlan(
        dataset_root,
        commands,
        all_sources,
        all_hashes,
        prepared_input=prepared_root is not None,
        acquisition_root=acquisition_root,
        acquisition_fingerprint=acquisition_fingerprint,
        moving_intrinsics_override=has_override,
    )


def build_preparation_plan(config: RigConfig, repository_root: Path) -> PreparationPlan:
    if not config.simulation.enabled:
        return _build_real_preparation_plan(config, repository_root)
    config, _ = _resolved_moving_intrinsics(config, repository_root)
    if config.dataset.prepared_root is not None:
        root = config.dataset.prepared_root.resolve()
        if root.name == "raw_images":
            root = root.parent
        sources = _prepared_source_files(config, root)
        return PreparationPlan(
            root,
            source_files=sources,
            source_hashes=_hash_sources(sources),
            prepared_input=True,
        )

    sources = _configured_source_files(config)
    source_hashes = _hash_sources(sources)
    digest = hashlib.sha256(_preparation_fingerprint(config).encode("ascii"))
    for key, value in sorted(source_hashes.items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.encode("ascii"))
    fingerprint = digest.hexdigest()[:12]
    dataset_root = (
        config.project.dataset_cache_root.resolve() / config.dataset.id / fingerprint
    )
    existing_path = dataset_root / "dataset_manifest.json"
    if existing_path.is_file():
        return PreparationPlan(
            dataset_root,
            source_files=sources,
            source_hashes=source_hashes,
            existing_manifest=load_dataset_manifest(existing_path),
        )
    for directory in (
        dataset_root / "raw_images" / "static",
        dataset_root / "raw_images" / "moving",
        dataset_root / "raw_images" / "camera_info",
        dataset_root / "metadata",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for camera in config.static_cameras:
        _copy_static_camera(camera, dataset_root, [])
    _copy_moving_frames(config, dataset_root, [])

    commands: list[CommandSpec] = []
    if config.simulation.enabled:
        mapping_path = dataset_root / "metadata" / "simulation_capture.json"
        mapping = {
            "preset": config.simulation.preset,
            "capture_id": config.simulation.capture_id,
            "world": str(config.simulation.world.resolve()),
            "route": str(config.simulation.route.resolve()),
            "moving_model_name": config.simulation.moving_model_name,
            "moving_sensor_name": config.simulation.moving_sensor_name,
            "world_id": config.simulation.world_id,
            "world_baseline": config.simulation.world_baseline,
            "resource_paths": [
                str(path)
                for path in config.simulation.resource_paths
            ],
            "settle_seconds": config.simulation.settle_seconds,
            "post_pose_skip": config.simulation.post_pose_skip,
            "frame_timeout_seconds": config.simulation.frame_timeout_seconds,
            "startup_timeout_seconds": config.simulation.startup_timeout_seconds,
            "route_name": config.simulation.route_name,
            "moving_width": config.simulation.moving_width,
            "moving_height": config.simulation.moving_height,
            "moving_hfov_deg": config.simulation.moving_hfov_deg,
            "lighting": config.simulation.lighting,
            "lighting_scale": config.simulation.lighting_scale,
            "motion_blur_kernel": config.simulation.motion_blur_kernel,
            "motion_blur_angle_deg": config.simulation.motion_blur_angle_deg,
            "target_route_frames": config.simulation.target_route_frames,
            "route_sampling_strategy": config.simulation.route_sampling_strategy,
            "static_cameras": [
                {
                    "id": camera.id,
                    "image_topic": camera.image_topic,
                    "camera_info_topic": camera.camera_info_topic,
                    "intrinsics_source": (
                        "provided"
                        if camera.intrinsics is not None
                        else "gazebo_camera_info"
                    ),
                }
                for camera in config.static_cameras
            ],
            "moving_camera": {
                "id": config.moving_camera.id,
                "image_topic": config.moving_camera.image_topic,
                "camera_info_topic": config.moving_camera.camera_info_topic,
                "intrinsics_source": (
                    "provided"
                    if config.moving_camera.intrinsics is not None
                    else "gazebo_camera_info"
                ),
            },
        }
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        commands.append(
            CommandSpec(
                "prepare_simulation",
                "Launch Gazebo and capture simulation dataset",
                (
                    sys.executable,
                    "-m",
                    "camera_rig_calibration.input.simulation",
                    "--repository",
                    str(repository_root),
                    "--dataset",
                    str(dataset_root),
                    "--mapping",
                    str(mapping_path),
                ),
                repository_root,
                dataset_root,
            )
        )
    if config.mcap.path is not None:
        mapping_path = dataset_root / "metadata" / "mcap_topic_mapping.json"
        mapping = {
            "marker_dictionary": config.markers.dictionary,
            "save_all_candidates": config.mcap.save_all_candidates,
            "moving_sampling_hz": config.sampling.target_hz,
            "cameras": [
                {
                    "id": camera.id,
                    "image_topic": camera.image_topic,
                    "camera_info_topic": camera.camera_info_topic,
                }
                for camera in config.static_cameras
            ],
            "moving_camera": (
                {
                    "id": config.moving_camera.id,
                    "image_topic": config.moving_camera.image_topic,
                    "camera_info_topic": config.moving_camera.camera_info_topic,
                }
                if config.moving_camera.video is None
                and config.moving_camera.frames is None
                and config.moving_camera.image_topic is not None
                else None
            ),
        }
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        commands.append(
            CommandSpec(
                "prepare_mcap",
                "Extract selected static images from MCAP",
                (
                    sys.executable,
                    "-m",
                    "camera_rig_calibration.input.mcap",
                    "--mcap",
                    str(config.mcap.path.resolve()),
                    "--dataset",
                    str(dataset_root),
                    "--mapping",
                    str(mapping_path),
                ),
                repository_root,
                dataset_root,
            )
        )

    moving = config.moving_camera
    if moving.video is not None:
        argv = [
            sys.executable,
            str(repository_root / "run/real_vehicle_data/01_extract_moving_video_3hz.py"),
            "--video",
            str(moving.video.resolve()),
            "--dataset",
            str(dataset_root),
            "--target-fps",
            str(config.sampling.target_hz),
            "--start-s",
            str(config.sampling.start_seconds),
        ]
        if config.sampling.end_seconds is not None:
            argv += ["--end-s", str(config.sampling.end_seconds)]
        if config.sampling.maximum_frames is not None:
            argv += ["--max-frames", str(config.sampling.maximum_frames)]
        argv.append("--overwrite")
        commands.append(
            CommandSpec(
                "prepare_moving_frames",
                "Extract moving-camera video frames",
                tuple(argv),
                repository_root,
                dataset_root / "raw_images" / "moving",
            )
        )

    info_destination = dataset_root / "raw_images" / "camera_info" / f"{moving.id}.json"
    if moving.intrinsics is not None:
        _normalize_intrinsics(moving.intrinsics.resolve(), info_destination, moving.id)
    elif (
        moving.intrinsic_calibration_video is not None
        or moving.intrinsic_calibration_images is not None
    ):
        calibration_source = (
            moving.intrinsic_calibration_video
            or moving.intrinsic_calibration_images
        )
        assert calibration_source is not None
        source_option = (
            "--video"
            if moving.intrinsic_calibration_video is not None
            else "--images"
        )
        commands.append(
            CommandSpec(
                "prepare_moving_intrinsics",
                "Calibrate moving-camera intrinsics",
                (
                    sys.executable,
                    "-m",
                    "camera_rig_calibration.input.intrinsics",
                    "--script",
                    str(repository_root / "run/real_vehicle_data/02_calibrate_intrinsics_from_video.py"),
                    source_option,
                    str(calibration_source.resolve()),
                    "--work-directory",
                    str(dataset_root / "metadata" / "intrinsic_calibration"),
                    "--destination",
                    str(info_destination),
                    "--camera-id",
                    moving.id,
                    "--cols",
                    str(moving.checkerboard_columns),
                    "--rows",
                    str(moving.checkerboard_rows),
                    "--max-views",
                    str(moving.intrinsic_maximum_views),
                    "--minimum-frame-gap",
                    str(moving.intrinsic_minimum_frame_gap),
                    "--minimum-detections",
                    str(moving.intrinsic_minimum_detections),
                ),
                repository_root,
                info_destination.parent,
            )
        )

    return PreparationPlan(
        dataset_root,
        commands,
        sources,
        source_hashes,
        prepared_input=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_dataset(config: RigConfig, plan: PreparationPlan) -> DatasetManifest:
    if config.dataset.scene_type.value == "simulation":
        from .simulation import capture_frame_diversity

        simulation_frames = sorted(
            (plan.dataset_root.resolve() / "raw_images" / "moving").glob(
                "frame_*.*"
            )
        )
        capture_frame_diversity(simulation_frames)
    if plan.existing_manifest is not None:
        return plan.existing_manifest.model_copy(
            update={
                "scene_type": config.dataset.scene_type,
                "sampling_hz": config.sampling.target_hz,
                "marker_dictionary": config.markers.dictionary,
                "marker_length_m": config.markers.length_m,
            },
            deep=True,
        )
    root = plan.dataset_root.resolve()
    if plan.acquisition_root is not None:
        acquisition = plan.acquisition_root.resolve()
        marker = acquisition / "ACQUISITION_COMPLETE.json"
        if not marker.is_file() and acquisition != root:
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "acquisition_fingerprint": plan.acquisition_fingerprint,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        moving_info = (
            acquisition
            / "raw_images"
            / "camera_info"
            / f"{config.moving_camera.id}.json"
        )
        _materialize_tree(
            acquisition / "raw_images",
            root / "raw_images",
            excluded=(
                {moving_info}
                if plan.moving_intrinsics_override
                else set()
            ),
        )
        _materialize_tree(
            acquisition / "metadata",
            root / "metadata" / "acquisition",
        )
    raw = root / "raw_images"
    static_manifests = []
    for camera in config.static_cameras:
        images = sorted((raw / "static_multi" / camera.id).glob("*.png"))
        if not images:
            images = sorted((raw / "static").glob(f"{camera.id}.*"))
        static_manifests.append(
            CameraManifest(
                id=camera.id,
                kind="static",
                image_count=len(images),
                images=[str(path.relative_to(root)) for path in images],
                intrinsics=str(Path("raw_images/camera_info") / f"{camera.id}.json"),
                source_topic=camera.image_topic,
            )
        )
    moving_frames = sorted((raw / "moving").glob("frame_*.*"))
    provenance = []
    for role, path in plan.source_files:
        if path.is_file():
            key = f"{role}\0{path}"
            provenance.append(
                FileProvenance(
                    role=role,
                    path=str(path),
                    sha256=plan.source_hashes.get(key) or _sha256(path),
                    size_bytes=path.stat().st_size,
                )
            )
    resolved_moving_info = (
        raw
        / "camera_info"
        / f"{config.moving_camera.id}.json"
    )
    if resolved_moving_info.is_file():
        provenance.append(
            FileProvenance(
                role="resolved_moving_intrinsics",
                path=str(resolved_moving_info),
                sha256=_sha256(resolved_moving_info),
                size_bytes=resolved_moving_info.stat().st_size,
            )
        )
    manifest = DatasetManifest(
        dataset_id=config.dataset.id,
        scene_type=config.dataset.scene_type,
        prepared_root=str(root),
        static_cameras=static_manifests,
        moving_camera=CameraManifest(
            id=config.moving_camera.id,
            kind="moving",
            image_count=len(moving_frames),
            images=[str(path.relative_to(root)) for path in moving_frames],
            intrinsics=str(
                Path("raw_images/camera_info") / f"{config.moving_camera.id}.json"
            ),
            source_topic=config.moving_camera.image_topic,
        ),
        sampling_hz=config.sampling.target_hz,
        marker_dictionary=config.markers.dictionary,
        marker_length_m=config.markers.length_m,
        simulation_parameters=(
            {
                "route": config.simulation.route_name,
                "moving_width": config.simulation.moving_width,
                "moving_height": config.simulation.moving_height,
                "moving_hfov_deg": config.simulation.moving_hfov_deg,
                "lighting": config.simulation.lighting,
                "lighting_scale": config.simulation.lighting_scale,
                "motion_blur_kernel": config.simulation.motion_blur_kernel,
                "motion_blur_angle_deg": config.simulation.motion_blur_angle_deg,
                "target_route_frames": config.simulation.target_route_frames,
                "route_sampling_strategy": config.simulation.route_sampling_strategy,
            }
            if config.dataset.scene_type.value == "simulation"
            else {}
        ),
        files=provenance,
        notes=[
            "scene_type is descriptive metadata and does not change method mathematics",
            *(
                [
                    "Simulation inputs were captured into a new immutable dataset cache; historical results were not overwritten."
                ]
                if config.simulation.enabled
                else []
            ),
        ],
    )
    if not plan.prepared_input:
        save_dataset_manifest(manifest, root / "dataset_manifest.json")
    return manifest

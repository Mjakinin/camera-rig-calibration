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
from .video_geometry import open_oriented_video


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
    capture = open_oriented_video(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open static-camera video: {source}")
    geometry = capture.geometry
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
                "video_geometry": geometry.as_dict(),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



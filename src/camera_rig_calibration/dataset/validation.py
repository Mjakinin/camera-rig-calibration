from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config.models import RigConfig


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class DatasetValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    static_camera_count: int = 0
    moving_frame_count: int = 0

    def require_valid(self) -> None:
        if not self.valid:
            raise RuntimeError("Dataset validation failed:\n- " + "\n- ".join(self.errors))


def _validate_intrinsics(
    path: Path, camera_id: str, errors: list[str]
) -> tuple[int, int] | None:
    if not path.is_file():
        errors.append(f"Missing intrinsics for '{camera_id}': {path}")
        return None
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        matrix = payload.get("K", payload.get("k"))
        if matrix is None:
            matrix = [
                payload["fx"], 0.0, payload["cx"],
                0.0, payload.get("fy", payload["fx"]), payload["cy"],
                0.0, 0.0, 1.0,
            ]
        if len(matrix) != 9 or float(matrix[0]) <= 0 or float(matrix[4]) <= 0:
            raise ValueError("invalid camera matrix K")
        return (
            int(payload.get("width", payload.get("image_width", 0)) or 0),
            int(payload.get("height", payload.get("image_height", 0)) or 0),
        )
    except Exception as exc:
        errors.append(f"Invalid intrinsics for '{camera_id}' ({path}): {exc}")
        return None


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        import cv2
    except ImportError:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def validate_dataset(config: RigConfig, dataset_root: Path) -> DatasetValidation:
    root = dataset_root.resolve()
    raw = root / "raw_images"
    errors: list[str] = []
    warnings: list[str] = []

    if not raw.is_dir():
        return DatasetValidation(False, [f"Missing canonical input directory: {raw}"])

    static_dir = raw / "static"
    info_dir = raw / "camera_info"
    moving_dir = raw / "moving"
    for camera in config.static_cameras:
        images = [
            path
            for path in static_dir.glob(f"{camera.id}.*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not images:
            message = f"Missing static image for '{camera.id}' in {static_dir}"
            (errors if camera.required else warnings).append(message)
        _validate_intrinsics(info_dir / f"{camera.id}.json", camera.id, errors)

    moving_frames = [
        path
        for path in moving_dir.glob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    ] if moving_dir.is_dir() else []
    if not moving_frames:
        errors.append(f"No moving-camera frames in {moving_dir}")
    moving_intrinsic_dimensions = _validate_intrinsics(
        info_dir / f"{config.moving_camera.id}.json",
        config.moving_camera.id,
        errors,
    )
    if moving_frames and moving_intrinsic_dimensions is not None:
        frame_dimensions = _image_dimensions(moving_frames[0])
        if (
            frame_dimensions is not None
            and all(value > 0 for value in moving_intrinsic_dimensions)
            and frame_dimensions != moving_intrinsic_dimensions
        ):
            errors.append(
                "Moving-camera intrinsic resolution does not match the "
                f"prepared frames: intrinsics={moving_intrinsic_dimensions[0]}x"
                f"{moving_intrinsic_dimensions[1]}, frames={frame_dimensions[0]}x"
                f"{frame_dimensions[1]}. rigcal never scales K silently."
            )

    return DatasetValidation(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        static_camera_count=len(config.static_cameras),
        moving_frame_count=len(moving_frames),
    )

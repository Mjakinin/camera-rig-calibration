"""Canonical handling of encoded video geometry and display rotation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIDEO_GEOMETRY_CONTRACT = "rigcal_video_geometry_v1"
DISPLAY_ORIENTATION_POLICY = "apply_ffprobe_display_rotation"


def normalize_rotation_degrees(value: float | int | None) -> int:
    """Return the nearest supported display rotation as -90/0/90/180."""
    if value is None:
        return 0
    quarter_turns = int(round(float(value) / 90.0))
    normalized = (quarter_turns * 90) % 360
    if normalized == 270:
        return -90
    return normalized


@dataclass(frozen=True)
class VideoGeometry:
    encoded_width: int
    encoded_height: int
    display_rotation_degrees: int
    output_width: int
    output_height: int
    orientation_policy: str = DISPLAY_ORIENTATION_POLICY
    contract: str = VIDEO_GEOMETRY_CONTRACT

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rotation_from_stream(stream: dict[str, Any]) -> float | int | None:
    for side_data in stream.get("side_data_list", []) or []:
        if not isinstance(side_data, dict):
            continue
        if side_data.get("rotation") is not None:
            return side_data["rotation"]
    tags = stream.get("tags") or {}
    if isinstance(tags, dict) and tags.get("rotate") is not None:
        return tags["rotate"]
    return None


def probe_video_geometry(
    path: Path | str,
    *,
    ffprobe_executable: str = "ffprobe",
) -> VideoGeometry:
    """Read encoded dimensions and display rotation from the first video stream."""
    video = Path(path).resolve()
    try:
        completed = subprocess.run(
            [
                ffprobe_executable,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe is required to apply video display rotation. "
            "Install FFmpeg and ensure ffprobe is available in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"ffprobe could not inspect video {video}: {detail or exc}"
        ) from exc

    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        encoded_width = int(stream["width"])
        encoded_height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"ffprobe returned no usable video geometry for {video}"
        ) from exc
    if encoded_width <= 0 or encoded_height <= 0:
        raise RuntimeError(
            f"ffprobe returned invalid video dimensions for {video}: "
            f"{encoded_width}x{encoded_height}"
        )

    rotation = normalize_rotation_degrees(_rotation_from_stream(stream))
    swaps_axes = abs(rotation) == 90
    return VideoGeometry(
        encoded_width=encoded_width,
        encoded_height=encoded_height,
        display_rotation_degrees=rotation,
        output_width=encoded_height if swaps_axes else encoded_width,
        output_height=encoded_width if swaps_axes else encoded_height,
    )


def apply_display_rotation(
    frame: np.ndarray,
    rotation_degrees: int | float,
) -> np.ndarray:
    """Apply the FFmpeg display transform without relying on OpenCV autorotation."""
    rotation = normalize_rotation_degrees(rotation_degrees)
    if rotation == 0:
        return frame
    if rotation == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError(f"Unsupported display rotation: {rotation_degrees}")


class OrientedVideoCapture:
    """Small VideoCapture facade that applies one explicit display transform."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self.geometry = probe_video_geometry(self.path)
        self._capture = cv2.VideoCapture(str(self.path))
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            self._capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return bool(self._capture.isOpened())

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return False, None
        return True, apply_display_rotation(
            frame, self.geometry.display_rotation_degrees
        )

    def grab(self) -> bool:
        return bool(self._capture.grab())

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._capture.retrieve()
        if not ok or frame is None:
            return False, None
        return True, apply_display_rotation(
            frame, self.geometry.display_rotation_degrees
        )

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.geometry.output_width)
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.geometry.output_height)
        return float(self._capture.get(property_id))

    def set(self, property_id: int, value: float) -> bool:
        return bool(self._capture.set(property_id, value))

    def release(self) -> None:
        self._capture.release()


def open_oriented_video(path: Path | str) -> OrientedVideoCapture:
    return OrientedVideoCapture(path)

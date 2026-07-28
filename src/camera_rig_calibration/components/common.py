"""Small helpers shared by the three calibration pipeline adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import RequirementResult, RunContext


def canonical_dataset_files(context: RunContext) -> dict[str, Path]:
    """Return the canonical input folders used by every calibration method."""

    raw = context.dataset_root / "raw_images"
    return {
        "raw": raw,
        "static": raw / "static",
        "moving": raw / "moving",
        "camera_info": raw / "camera_info",
    }


def calibration_requirements(context: RunContext) -> RequirementResult:
    """Validate the input contract shared by AP01, AP02 and AP03."""

    files = canonical_dataset_files(context)
    reasons = []
    if len(context.config.static_cameras) < 2:
        reasons.append("at least two static cameras are required")
    for camera in context.config.static_cameras:
        if not any(files["static"].glob(f"{camera.id}.*")):
            reasons.append(f"static image is missing for '{camera.id}'")
        if not (files["camera_info"] / f"{camera.id}.json").is_file():
            reasons.append(f"intrinsics are missing for '{camera.id}'")
    if not any(files["moving"].glob("frame_*.*")):
        reasons.append("moving-camera frames are missing")
    if not (
        files["camera_info"] / f"{context.config.moving_camera.id}.json"
    ).is_file():
        reasons.append(
            f"intrinsics are missing for '{context.config.moving_camera.id}'"
        )
    return RequirementResult.unavailable(*reasons) if reasons else RequirementResult.ok()


def read_method_status(directory: Path) -> dict[str, Any]:
    """Read the normalized status emitted by a completed method pipeline."""

    path = directory / "METHOD_STATUS.json"
    if not path.is_file():
        return {"status": "MISSING", "success": False, "directory": str(directory)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "INVALID_STATUS",
            "success": False,
            "directory": str(directory),
            "error": str(exc),
        }
    payload["directory"] = str(directory)
    return payload

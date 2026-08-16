from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


COMPACT_ANCHOR_YAML = "camera_extrinsics_anchor_compact.yaml"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def compact_anchor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deployment-friendly six-DOF view of one anchor export.

    The detailed canonical export remains authoritative.  This representation
    intentionally keeps only x/y/z and roll/pitch/yaw and preserves the real
    parent frame instead of relabelling anchor-relative poses as ``base_link``.
    """
    parent_frame = str(payload.get("parent_frame") or "unavailable_parent_frame")
    cameras: dict[str, dict[str, Any]] = {}
    for camera in payload.get("cameras", []):
        if not isinstance(camera, dict):
            continue
        camera_id = str(camera.get("camera_id") or "").strip()
        if not camera_id:
            continue
        cameras[camera_id] = {
            "x": camera.get("x_m"),
            "y": camera.get("y_m"),
            "z": camera.get("z_m"),
            "roll": camera.get("roll_rad"),
            "pitch": camera.get("pitch_rad"),
            "yaw": camera.get("yaw_rad"),
        }
    return {parent_frame: cameras}


def write_compact_method_anchor_yaml(method_root: Path) -> Path | None:
    """Write the compact YAML beside an existing canonical anchor export."""
    method_root = Path(method_root)
    source = method_root / "camera_extrinsics_anchor.json"
    payload = _read_json(source)
    if not payload or not isinstance(payload.get("cameras"), list):
        return None
    target = method_root / COMPACT_ANCHOR_YAML
    text = yaml.safe_dump(
        compact_anchor_payload(payload),
        sort_keys=False,
        allow_unicode=True,
    )
    try:
        if target.read_text(encoding="utf-8") == text:
            return target
    except OSError:
        pass
    target.write_text(text, encoding="utf-8")
    return target


__all__ = [
    "COMPACT_ANCHOR_YAML",
    "compact_anchor_payload",
    "write_compact_method_anchor_yaml",
]

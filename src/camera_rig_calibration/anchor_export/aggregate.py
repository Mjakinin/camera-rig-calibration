from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


AGGREGATE_CONTRACT = "rigcal_experiment_anchor_aggregate_v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_text(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _legacy_row(camera: dict[str, Any]) -> dict[str, Any]:
    """Keep the historic top-level export schema while using canonical inputs."""
    return {
        "method": camera.get("method"),
        "label": camera.get("label"),
        "anchor_marker_id": camera.get("anchor_marker_id"),
        "parent_frame": camera.get("parent_frame"),
        "camera_id": camera.get("camera_id"),
        "quality_status": camera.get("quality_status"),
        "deployment_eligible": bool(camera.get("deployment_eligible", True)),
        "x": camera.get("x_m"),
        "y": camera.get("y_m"),
        "z": camera.get("z_m"),
        "roll": camera.get("roll_rad"),
        "pitch": camera.get("pitch_rad"),
        "yaw": camera.get("yaw_rad"),
        "qx": camera.get("qx"),
        "qy": camera.get("qy"),
        "qz": camera.get("qz"),
        "qw": camera.get("qw"),
    }


def build_experiment_anchor_aggregate(experiment_root: Path) -> dict[str, Any]:
    """Rebuild the experiment-wide compatibility export from every variant.

    Per-method ``camera_extrinsics_anchor.*`` files are authoritative.  The
    experiment-level files are a deterministic aggregate/front door and must
    never retain a stale subset from an earlier reporting pass.
    """
    rows: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    methods_root = experiment_root / "methods"
    if methods_root.is_dir():
        for path in sorted(methods_root.glob("*/*/camera_extrinsics_anchor.json")):
            payload = _read_json(path)
            method = str(payload.get("method") or path.parents[1].name)
            label = str(payload.get("label") or path.parent.name)
            cameras = [
                item for item in payload.get("cameras", [])
                if isinstance(item, dict)
            ]
            status = payload.get("anchor_export_status", {})
            variants.append(
                {
                    "method": method,
                    "label": label,
                    "anchor_marker_id": payload.get("anchor_marker_id"),
                    "available": bool(status.get("available", cameras)),
                    "camera_count": len(cameras),
                    "source": str(path.relative_to(experiment_root)),
                }
            )
            for camera in cameras:
                normalized = dict(camera)
                normalized.setdefault("method", method)
                normalized.setdefault("label", label)
                normalized.setdefault(
                    "anchor_marker_id", payload.get("anchor_marker_id")
                )
                normalized.setdefault("parent_frame", payload.get("parent_frame"))
                rows.append(_legacy_row(normalized))

    rows.sort(
        key=lambda row: (
            str(row.get("method", "")),
            str(row.get("label", "")),
            str(row.get("camera_id", "")),
        )
    )
    variants.sort(key=lambda item: (item["method"], item["label"]))
    anchor_ids = sorted(
        {
            int(row["anchor_marker_id"])
            for row in rows
            if row.get("anchor_marker_id") is not None
        }
    )
    anchor_marker_id: int | None = anchor_ids[0] if len(anchor_ids) == 1 else None
    parent_frame = (
        f"evaluation_anchor_marker_{anchor_marker_id}"
        if anchor_marker_id is not None
        else None
    )
    payload = {
        "schema_version": 1,
        "layout_version": 2,
        "contract": AGGREGATE_CONTRACT,
        "anchor_marker_id": anchor_marker_id,
        "anchor_marker_ids": anchor_ids,
        "parent_frame": parent_frame,
        "transform_convention": (
            "T_anchor_camera; p_anchor = T_anchor_camera @ p_camera"
        ),
        "translation_unit": "m",
        "rotation_unit": "rad",
        "rpy_convention": "R = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        "variants": variants,
        "rows": rows,
        "all_published_variant_rows": rows,
    }
    _atomic_text(
        experiment_root / "CAMERA_EXTRINSICS_COMMON_ANCHOR.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        experiment_root / "CAMERA_EXTRINSICS_COMMON_ANCHOR.yaml",
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )
    _write_csv(experiment_root / "CAMERA_EXTRINSICS_COMMON_ANCHOR.csv", rows)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method", "label", "anchor_marker_id", "parent_frame", "camera_id",
        "quality_status", "deployment_eligible", "x", "y", "z", "roll",
        "pitch", "yaw", "qx", "qy", "qz", "qw",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    temporary.replace(path)


def experiment_anchor_aggregate_text(payload: dict[str, Any]) -> str:
    lines = [
        "COMMON-ANCHOR STATIC-CAMERA 6DOF EXPORTS",
        "-" * 138,
        f"Reference frame: {payload.get('parent_frame') or 'multiple/unavailable'}",
        "Translation: metres; roll/pitch/yaw: radians; optical-camera convention x right, y down, z forward.",
        "These are method outputs expressed in the common evaluation/export anchor frame; no GT alignment is applied.",
        "",
    ]
    rows = payload.get("rows", [])
    keys = sorted(
        {
            (str(row.get("method", "")), str(row.get("label", "")))
            for row in rows
            if isinstance(row, dict)
        }
    )
    for method, label in keys:
        lines.append(f"{method}/{label}:")
        selected = [
            row for row in rows
            if str(row.get("method")) == method and str(row.get("label")) == label
        ]
        for row in selected:
            lines.extend(
                [
                    f"  {row.get('camera_id', '-')}:",
                    f"    x: {row.get('x')}",
                    f"    y: {row.get('y')}",
                    f"    z: {row.get('z')}",
                    f"    roll: {row.get('roll')}",
                    f"    pitch: {row.get('pitch')}",
                    f"    yaw: {row.get('yaw')}",
                ]
            )
        lines.append("")
    if not rows:
        lines.extend(["Unavailable: no common-anchor camera poses are published.", ""])
    lines.extend(
        [
            "Machine-readable exports:",
            "- CAMERA_EXTRINSICS_COMMON_ANCHOR.yaml",
            "- CAMERA_EXTRINSICS_COMMON_ANCHOR.json",
            "- CAMERA_EXTRINSICS_COMMON_ANCHOR.csv",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AGGREGATE_CONTRACT",
    "build_experiment_anchor_aggregate",
    "experiment_anchor_aggregate_text",
]

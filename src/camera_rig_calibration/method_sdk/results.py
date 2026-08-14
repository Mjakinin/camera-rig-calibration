"""Canonical, method-independent static-camera 6DOF result models."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..anchor_export.geometry import (
    rotation_to_quaternion,
    rotation_to_rpy,
    validate_transform,
)


CANONICAL_RESULT_CONTRACT = "rigcal_canonical_method_result_v1"
TRANSFORM_CONVENTION = (
    "T_reference_camera; p_reference = T_reference_camera @ p_camera"
)


class CanonicalCameraPose(BaseModel):
    """One finite rigid camera pose with explicit frame semantics."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    reference_frame: str = Field(min_length=1)
    child_frame: str = Field(min_length=1)
    transform_convention: str = TRANSFORM_CONVENTION
    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    roll_pitch_yaw_rad: tuple[float, float, float]
    matrix: list[list[float]]
    source: str | None = None

    @model_validator(mode="after")
    def validate_pose(self) -> "CanonicalCameraPose":
        transform = validate_transform(np.asarray(self.matrix, dtype=np.float64))
        translation = np.asarray(self.translation_m, dtype=np.float64)
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
        rpy = np.asarray(self.roll_pitch_yaw_rad, dtype=np.float64)
        if not (
            np.all(np.isfinite(translation))
            and np.all(np.isfinite(quaternion))
            and np.all(np.isfinite(rpy))
        ):
            raise ValueError("canonical 6DOF values must be finite")
        if not np.allclose(transform[:3, 3], translation, atol=1e-9):
            raise ValueError("translation_m does not match the SE(3) matrix")
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-8):
            raise ValueError("quaternion_xyzw must have unit length")
        expected_quaternion = np.asarray(
            rotation_to_quaternion(transform[:3, :3]), dtype=np.float64
        )
        if not (
            np.allclose(quaternion, expected_quaternion, atol=1e-8)
            or np.allclose(quaternion, -expected_quaternion, atol=1e-8)
        ):
            raise ValueError("quaternion_xyzw does not match the SE(3) matrix")
        return self

    @classmethod
    def from_transform(
        cls,
        *,
        camera_id: str,
        reference_frame: str,
        transform: Any,
        source: str | None = None,
    ) -> "CanonicalCameraPose":
        matrix = validate_transform(np.asarray(transform, dtype=np.float64))
        return cls(
            camera_id=camera_id,
            reference_frame=reference_frame,
            child_frame=f"{camera_id}_optical",
            translation_m=tuple(float(value) for value in matrix[:3, 3]),
            quaternion_xyzw=rotation_to_quaternion(matrix[:3, :3]),
            roll_pitch_yaw_rad=rotation_to_rpy(matrix[:3, :3]),
            matrix=matrix.tolist(),
            source=source,
        )


class CanonicalMethodResult(BaseModel):
    """Stable result consumed by publication, UI and visualization."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    contract: Literal["rigcal_canonical_method_result_v1"] = (
        CANONICAL_RESULT_CONTRACT
    )
    method_id: str = Field(min_length=1)
    algorithm_version: str = Field(min_length=1)
    status: Literal["available", "incomplete", "diagnostic"]
    reference_frame: str = Field(min_length=1)
    transform_convention: str = TRANSFORM_CONVENTION
    camera_poses: list[CanonicalCameraPose] = Field(default_factory=list)
    quality_status: str = "unknown"
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    native_artifacts: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result(self) -> "CanonicalMethodResult":
        camera_ids = [pose.camera_id for pose in self.camera_poses]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("canonical result contains duplicate camera IDs")
        if self.status == "available" and not self.camera_poses:
            raise ValueError("an available calibration result needs 6DOF poses")
        if any(
            pose.reference_frame != self.reference_frame
            for pose in self.camera_poses
        ):
            raise ValueError("all camera poses must use the result reference frame")
        return self


def write_canonical_result(
    root: Path, result: CanonicalMethodResult
) -> tuple[Path, Path]:
    """Atomically write the canonical JSON and a flat interoperable CSV."""

    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "canonical_method_result.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(json_path)
    csv_path = root / "camera_poses_6dof.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    fields = [
        "camera_id",
        "reference_frame",
        "child_frame",
        "transform_convention",
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "qx",
        "qy",
        "qz",
        "qw",
        "matrix_json",
        "source",
    ]
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pose in result.camera_poses:
            writer.writerow(
                {
                    "camera_id": pose.camera_id,
                    "reference_frame": pose.reference_frame,
                    "child_frame": pose.child_frame,
                    "transform_convention": pose.transform_convention,
                    "x_m": pose.translation_m[0],
                    "y_m": pose.translation_m[1],
                    "z_m": pose.translation_m[2],
                    "roll_rad": pose.roll_pitch_yaw_rad[0],
                    "pitch_rad": pose.roll_pitch_yaw_rad[1],
                    "yaw_rad": pose.roll_pitch_yaw_rad[2],
                    "qx": pose.quaternion_xyzw[0],
                    "qy": pose.quaternion_xyzw[1],
                    "qz": pose.quaternion_xyzw[2],
                    "qw": pose.quaternion_xyzw[3],
                    "matrix_json": json.dumps(pose.matrix, separators=(",", ":")),
                    "source": pose.source or "",
                }
            )
    csv_temporary.replace(csv_path)
    return json_path, csv_path


def load_canonical_result(path: Path) -> CanonicalMethodResult:
    return CanonicalMethodResult.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def write_native_camera_extrinsics(
    path: Path, result: CanonicalMethodResult
) -> Path:
    """Bridge canonical poses to the established public pose CSV format."""

    fields = [
        "reference_frame",
        "transform_convention",
        "entity_type",
        "entity_id",
        "source",
        "x_m",
        "y_m",
        "z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "rvec_x",
        "rvec_y",
        "rvec_z",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pose in result.camera_poses:
            qx, qy, qz, qw = pose.quaternion_xyzw
            vector_norm = math.sqrt(qx * qx + qy * qy + qz * qz)
            if vector_norm <= 1e-15:
                rvec = (0.0, 0.0, 0.0)
            else:
                angle = 2.0 * math.atan2(vector_norm, qw)
                rvec = tuple(
                    angle * value / vector_norm for value in (qx, qy, qz)
                )
            writer.writerow(
                {
                    "reference_frame": pose.reference_frame,
                    "transform_convention": (
                        "T_reference_camera (camera pose expressed in reference frame)"
                    ),
                    "entity_type": "static_camera",
                    "entity_id": pose.camera_id,
                    "source": pose.source or result.method_id,
                    "x_m": pose.translation_m[0],
                    "y_m": pose.translation_m[1],
                    "z_m": pose.translation_m[2],
                    "roll_deg": math.degrees(pose.roll_pitch_yaw_rad[0]),
                    "pitch_deg": math.degrees(pose.roll_pitch_yaw_rad[1]),
                    "yaw_deg": math.degrees(pose.roll_pitch_yaw_rad[2]),
                    "rvec_x": rvec[0],
                    "rvec_y": rvec[1],
                    "rvec_z": rvec[2],
                }
            )
    return path


__all__ = [
    "CANONICAL_RESULT_CONTRACT",
    "TRANSFORM_CONVENTION",
    "CanonicalCameraPose",
    "CanonicalMethodResult",
    "load_canonical_result",
    "write_canonical_result",
    "write_native_camera_extrinsics",
]

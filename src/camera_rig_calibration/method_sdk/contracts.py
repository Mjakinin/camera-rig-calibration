"""Stable metadata contracts for calibration-method extensions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from ..contracts import RunContext


@dataclass(frozen=True)
class MethodInputRequirements:
    """Declare which canonical inputs a method consumes.

    The shared preparation pipeline owns all file-format specific ingestion.
    Methods therefore request canonical inputs instead of implementing separate
    image, video, frame-folder or ROS-bag loaders.
    """

    static_images: bool = True
    moving_frames: bool = True
    camera_intrinsics: bool = True
    marker_observations: bool = True


class MethodConfigEditor(Protocol):
    """Optional terminal-UI hook for configurations too complex for auto-UI."""

    def edit(self, console: Any, config: BaseModel) -> BaseModel: ...


class CanonicalResultBuilder(Protocol):
    """Convert native method artifacts into the shared result contract."""

    def __call__(
        self,
        context: RunContext,
        status: dict[str, Any],
    ) -> "CanonicalMethodResult": ...


@dataclass(frozen=True)
class MethodMetadata:
    """Normalized SDK metadata, including safe built-in defaults."""

    algorithm_version: str
    run_manifest_algorithm_version: str
    artifact_directory: Path
    primary_pose_path: Path | None
    input_requirements: MethodInputRequirements
    result_contract_required: bool
    config_editor: MethodConfigEditor | None


def method_metadata(method: Any) -> MethodMetadata:
    """Resolve SDK metadata without breaking pre-SDK extension objects."""

    method_id = str(getattr(method, "id", "")).strip()
    if not method_id:
        raise ValueError("calibration method has no ID")
    artifact = Path(getattr(method, "artifact_directory", method_id))
    if artifact.is_absolute() or ".." in artifact.parts:
        raise ValueError(
            f"Method '{method_id}' artifact_directory must be relative"
        )
    primary_value = getattr(method, "primary_pose_path", None)
    primary = Path(primary_value) if primary_value is not None else None
    if primary is not None and (
        primary.is_absolute() or ".." in primary.parts
    ):
        raise ValueError(
            f"Method '{method_id}' primary_pose_path must be relative"
        )
    inputs = getattr(method, "input_requirements", MethodInputRequirements())
    if not isinstance(inputs, MethodInputRequirements):
        raise TypeError(
            f"Method '{method_id}' input_requirements has the wrong type"
        )
    algorithm_version = str(
        getattr(method, "algorithm_version", "extension_v1")
    )
    return MethodMetadata(
        algorithm_version=algorithm_version,
        run_manifest_algorithm_version=str(
            getattr(
                method,
                "run_manifest_algorithm_version",
                algorithm_version,
            )
        ),
        artifact_directory=artifact,
        primary_pose_path=primary,
        input_requirements=inputs,
        result_contract_required=bool(
            getattr(method, "result_contract_required", False)
        ),
        config_editor=getattr(method, "config_editor", None),
    )


# Imported only for type-checking at runtime by postponed annotations.
from .results import CanonicalMethodResult  # noqa: E402


__all__ = [
    "CanonicalResultBuilder",
    "MethodConfigEditor",
    "MethodInputRequirements",
    "MethodMetadata",
    "method_metadata",
]

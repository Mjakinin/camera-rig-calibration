from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..config.models import SceneType


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileProvenance(ManifestModel):
    role: str
    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    source: str | None = None


class CameraManifest(ManifestModel):
    id: str
    kind: Literal["static", "moving"]
    image_count: int = 0
    images: list[str] = Field(default_factory=list)
    intrinsics: str | None = None
    source_topic: str | None = None


class AutoSelection(ManifestModel):
    kind: str
    selected: str | int
    candidates: list[str | int] = Field(default_factory=list)
    reason: str


class DatasetManifest(ManifestModel):
    schema_version: Literal[1] = 1
    dataset_id: str
    scene_type: SceneType
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    prepared_root: str
    static_cameras: list[CameraManifest]
    moving_camera: CameraManifest
    sampling_hz: float | None = None
    marker_dictionary: str
    marker_length_m: float
    simulation_parameters: dict[str, Any] = Field(default_factory=dict)
    files: list[FileProvenance] = Field(default_factory=list)
    automatic_selections: list[AutoSelection] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def save_dataset_manifest(manifest: DatasetManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_dataset_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))

"""Validated discovery and identity for user-provided simulation routes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..dataset.discovery import safe_id


ROUTE_CONTRACT = "rigcal_simulation_route_v1"


class SimulationRouteFrame(BaseModel):
    model_config = ConfigDict(extra="allow")

    frame: int = Field(ge=0)
    segment: str = "custom"
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    @field_validator("segment", mode="before")
    @classmethod
    def validate_segment(cls, value: Any) -> str:
        if value is None:
            return "custom"
        value = str(value).strip()
        return value or "custom"

    @field_validator("x", "y", "z", "roll", "pitch", "yaw")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("route pose values must be finite")
        return value


class SimulationRouteDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    contract: Literal["rigcal_simulation_route_v1"] = ROUTE_CONTRACT
    pose_format: str = "x y z roll pitch yaw"
    frames: list[SimulationRouteFrame] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_frames(self) -> "SimulationRouteDocument":
        frame_ids = [item.frame for item in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("route frame numbers must be unique")
        if frame_ids != sorted(frame_ids):
            raise ValueError("route frames must be ordered by frame number")
        return self


@dataclass(frozen=True)
class SimulationRouteAsset:
    id: str
    path: Path
    sha256: str
    frame_count: int
    source: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_simulation_route(path: Path) -> SimulationRouteDocument:
    path = path.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read simulation route {path}: {exc}") from exc
    try:
        return SimulationRouteDocument.model_validate(payload)
    except ValueError as exc:
        raise ValueError(f"Invalid simulation route {path}: {exc}") from exc


def simulation_route_asset(
    path: Path, *, route_id: str | None = None, source: str = "explicit"
) -> SimulationRouteAsset:
    path = path.expanduser().resolve()
    document = load_simulation_route(path)
    return SimulationRouteAsset(
        id=safe_id(route_id or path.stem),
        path=path,
        sha256=_sha256(path),
        frame_count=len(document.frames),
        source=source,
    )


def discover_local_simulation_routes(
    repository_root: Path,
) -> tuple[SimulationRouteAsset, ...]:
    root = repository_root.resolve() / "data_local" / "simulation_routes"
    if not root.is_dir():
        return ()
    routes: list[SimulationRouteAsset] = []
    builtin_identifiers = {"route1", "route2"}
    identifiers = set(builtin_identifiers)
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).with_suffix("")
        identifier = safe_id("__".join(relative.parts))
        if identifier in builtin_identifiers:
            identifier = f"local_{identifier}"
        if identifier in identifiers:
            raise ValueError(
                f"Duplicate local simulation route ID '{identifier}': {path}"
            )
        identifiers.add(identifier)
        routes.append(
            simulation_route_asset(
                path, route_id=identifier, source="data_local"
            )
        )
    return tuple(routes)


def simulation_route_manifest(path: Path) -> dict[str, Any]:
    asset = simulation_route_asset(path)
    document = load_simulation_route(path)
    return {
        "contract": ROUTE_CONTRACT,
        "path": str(asset.path),
        "sha256": asset.sha256,
        "pose_format": document.pose_format,
        "frame_count": asset.frame_count,
        "frames": [
            frame.model_dump(mode="json") for frame in document.frames
        ],
    }


__all__ = [
    "ROUTE_CONTRACT",
    "SimulationRouteAsset",
    "SimulationRouteDocument",
    "SimulationRouteFrame",
    "discover_local_simulation_routes",
    "load_simulation_route",
    "simulation_route_asset",
    "simulation_route_manifest",
]

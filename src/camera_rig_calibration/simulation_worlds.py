from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator

from .config.models import ID_PATTERN, StrictModel


class WorldCamera(StrictModel):
    id: str
    model_name: str
    sensor_name: str | None = None
    image_topic: str
    camera_info_topic: str

    @field_validator(
        "id",
        "model_name",
        "image_topic",
        "camera_info_topic",
    )
    @classmethod
    def non_empty_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("id")
    @classmethod
    def safe_camera_id(cls, value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError("camera id is not safe for a directory name")
        return value


class WorldRoute(StrictModel):
    id: str
    path: Path
    baseline: bool = False


class SimulationWorldManifest(StrictModel):
    schema_version: int = 1
    id: str
    display_name: str
    sdf: Path
    resource_paths: list[Path] = Field(default_factory=list)
    static_cameras: list[WorldCamera]
    moving_camera: WorldCamera
    routes: list[WorldRoute]
    baseline: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(
        default_factory=lambda: [
            "route",
            "density",
            "resolution",
            "fov",
            "motion_blur",
            "capture",
        ]
    )
    lighting_profiles: dict[str, Path | None] = Field(default_factory=dict)
    manifest_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_contract(self) -> "SimulationWorldManifest":
        if not ID_PATTERN.fullmatch(self.id):
            raise ValueError("world id is not safe for a directory name")
        if not self.static_cameras:
            raise ValueError("at least one static camera is required")
        camera_ids = [
            *(camera.id for camera in self.static_cameras),
            self.moving_camera.id,
        ]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera IDs must be unique")
        if not self.routes:
            raise ValueError("at least one route is required")
        if sum(route.baseline for route in self.routes) != 1:
            raise ValueError("exactly one route must have baseline: true")
        supported = {
            "route",
            "density",
            "resolution",
            "fov",
            "lighting",
            "motion_blur",
            "capture",
        }
        unknown = sorted(set(self.capabilities) - supported)
        if unknown:
            raise ValueError(
                "unknown mutable simulation capabilities: "
                + ", ".join(unknown)
            )
        if "lighting" in self.capabilities and not self.lighting_profiles:
            raise ValueError(
                "lighting capability requires at least one lighting profile"
            )
        return self

    @property
    def baseline_route(self) -> WorldRoute:
        return next(route for route in self.routes if route.baseline)


def _resolve(path: Path, base: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _camera_element(
    root: ET.Element, model_name: str, sensor_name: str | None
) -> ET.Element:
    matches: list[ET.Element] = []
    for model in root.iter("model"):
        if model.get("name") != model_name:
            continue
        for sensor in model.iter("sensor"):
            if sensor_name is not None and sensor.get("name") != sensor_name:
                continue
            camera = sensor.find("camera")
            if camera is not None:
                matches.append(camera)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one camera sensor for model '{model_name}'"
            + (f" and sensor '{sensor_name}'" if sensor_name else "")
            + f", found {len(matches)}"
        )
    return matches[0]


def _moving_baseline(manifest: SimulationWorldManifest) -> dict[str, Any]:
    root = ET.parse(manifest.sdf).getroot()
    world = root if root.tag == "world" else root.find("world")
    if world is None or not world.get("name"):
        raise ValueError("SDF must contain a named <world>")
    camera = _camera_element(
        root,
        manifest.moving_camera.model_name,
        manifest.moving_camera.sensor_name,
    )
    width = camera.findtext("image/width")
    height = camera.findtext("image/height")
    hfov = camera.findtext("horizontal_fov")
    if width is None or height is None or hfov is None:
        raise ValueError(
            "Moving camera must define image width, height and horizontal_fov"
        )
    route_payload = json.loads(
        manifest.baseline_route.path.read_text(encoding="utf-8")
    )
    frames = route_payload.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError(
            f"Baseline route needs at least two frames: "
            f"{manifest.baseline_route.path}"
        )
    defaults = {
        "route_name": manifest.baseline_route.id,
        "moving_width": int(width),
        "moving_height": int(height),
        "moving_hfov_deg": math.degrees(float(hfov)),
        "lighting": "baseline",
        "lighting_scale": 1.0,
        "motion_blur_kernel": 0,
        "motion_blur_angle_deg": 0.0,
        "target_route_frames": len(frames),
        "route_sampling_strategy": "original_route_poses",
        "settle_seconds": 0.35,
        "post_pose_skip": 5,
        "frame_timeout_seconds": 3.0,
        "startup_timeout_seconds": 60.0,
    }
    return {**defaults, **manifest.baseline}


def validate_world_manifest(
    manifest: SimulationWorldManifest,
) -> SimulationWorldManifest:
    if not manifest.sdf.is_file():
        raise FileNotFoundError(f"Gazebo SDF does not exist: {manifest.sdf}")
    missing_resources = [
        path for path in manifest.resource_paths if not path.exists()
    ]
    if missing_resources:
        raise FileNotFoundError(
            "Gazebo resource paths do not exist: "
            + ", ".join(str(path) for path in missing_resources)
        )
    missing_lighting = [
        path
        for path in manifest.lighting_profiles.values()
        if path is not None and not path.is_file()
    ]
    if missing_lighting:
        raise FileNotFoundError(
            "Gazebo lighting-profile SDFs do not exist: "
            + ", ".join(str(path) for path in missing_lighting)
        )
    root = ET.parse(manifest.sdf).getroot()
    _camera_element(
        root,
        manifest.moving_camera.model_name,
        manifest.moving_camera.sensor_name,
    )
    for camera in manifest.static_cameras:
        _camera_element(root, camera.model_name, camera.sensor_name)
    static_contract = {
        camera.id: ET.tostring(
            _camera_element(root, camera.model_name, camera.sensor_name),
            encoding="utf-8",
        )
        for camera in manifest.static_cameras
    }
    for profile, profile_path in manifest.lighting_profiles.items():
        if profile_path is None:
            continue
        profile_root = ET.parse(profile_path).getroot()
        _camera_element(
            profile_root,
            manifest.moving_camera.model_name,
            manifest.moving_camera.sensor_name,
        )
        profile_contract = {
            camera.id: ET.tostring(
                _camera_element(
                    profile_root,
                    camera.model_name,
                    camera.sensor_name,
                ),
                encoding="utf-8",
            )
            for camera in manifest.static_cameras
        }
        if profile_contract != static_contract:
            raise ValueError(
                f"Lighting profile '{profile}' changes a static camera "
                "sensor contract"
            )
    for route in manifest.routes:
        if not route.path.is_file():
            raise FileNotFoundError(f"Route does not exist: {route.path}")
        payload = json.loads(route.path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("frames"), list) or len(payload["frames"]) < 2:
            raise ValueError(f"Route needs at least two frames: {route.path}")
    baseline = _moving_baseline(manifest)
    route_frame_counts: dict[str, int] = {}
    for route in manifest.routes:
        payload = json.loads(route.path.read_text(encoding="utf-8"))
        route_frame_counts[route.id] = len(payload["frames"])
    baseline["route_frame_counts"] = route_frame_counts
    return manifest.model_copy(update={"baseline": baseline}, deep=True)


def load_world_manifest(path: Path) -> SimulationWorldManifest:
    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"World manifest must contain a mapping: {source}")
    base = source.parent
    payload = dict(payload)
    payload["sdf"] = _resolve(Path(payload["sdf"]), base)
    payload["resource_paths"] = [
        _resolve(Path(value), base)
        for value in payload.get("resource_paths", [])
    ]
    payload["lighting_profiles"] = {
        str(name): (
            _resolve(Path(value), base) if value is not None else None
        )
        for name, value in payload.get("lighting_profiles", {}).items()
    }
    payload["routes"] = [
        {**route, "path": _resolve(Path(route["path"]), base)}
        for route in payload.get("routes", [])
    ]
    manifest = SimulationWorldManifest.model_validate(
        {**payload, "manifest_path": source}
    )
    return validate_world_manifest(manifest)


def discover_world_manifests(
    repository_root: Path,
) -> tuple[SimulationWorldManifest, ...]:
    root = repository_root.resolve() / "config" / "simulation_worlds"
    if not root.is_dir():
        return ()
    manifests: list[SimulationWorldManifest] = []
    errors: list[str] = []
    for path in sorted(root.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            manifests.append(load_world_manifest(path))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        raise RuntimeError(
            "Invalid registered Gazebo world manifest(s): " + "; ".join(errors)
        )
    return tuple(manifests)


def install_world_manifest(
    repository_root: Path, source: Path
) -> SimulationWorldManifest:
    manifest = load_world_manifest(source)
    destination = (
        repository_root.resolve()
        / "config"
        / "simulation_worlds"
        / f"{manifest.id}.yaml"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.resolve() != source.resolve():
        raise FileExistsError(
            f"A world manifest with ID '{manifest.id}' already exists: "
            f"{destination}"
        )
    if destination.resolve() != source.resolve():
        payload = manifest.model_dump(
            mode="json",
            exclude={"manifest_path"},
            exclude_none=True,
        )
        temporary = destination.with_suffix(".yaml.tmp")
        temporary.write_text(
            yaml.safe_dump(
                payload, sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)
    return load_world_manifest(destination)

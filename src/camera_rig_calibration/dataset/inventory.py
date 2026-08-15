from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset.discovery import (
    IMAGE_SUFFIXES,
    MCAP_SUFFIXES,
    VIDEO_SUFFIXES,
    classify_path,
    image_directory_role,
)


@dataclass(frozen=True)
class PreparedDatasetSummary:
    id: str
    display_name: str
    category: str
    description: str
    path: Path
    static_camera_ids: tuple[str, ...]
    moving_frames: int
    has_results: bool


@dataclass(frozen=True)
class RawInputSummary:
    path: Path
    videos: int
    images: int
    intrinsics: int
    recordings: int


@dataclass(frozen=True)
class SimulationExperimentSummary:
    variant: str
    factor: str
    value: str
    moving_frames: int | None
    has_results: bool
    dataset_root: Path | None
    parameters: dict[str, Any]


BASELINE_SIMULATION_PARAMETERS: dict[str, Any] = {
    "route": "route2",
    "moving_width": 1280,
    "moving_height": 720,
    "moving_hfov_deg": 69.1,
    "lighting": "baseline",
    "lighting_scale": 1.0,
    "motion_blur_kernel": 0,
    "motion_blur_angle_deg": 0.0,
    "target_route_frames": 189,
    "route_sampling_strategy": "original_route_poses",
    "settle_seconds": 0.35,
    "post_pose_skip": 5,
    "frame_timeout_seconds": 3.0,
    "startup_timeout_seconds": 60.0,
}


def format_simulation_parameters(parameters: dict[str, Any]) -> str:
    sampling = str(
        parameters.get("route_sampling_strategy", "?")
    ).replace("_", " ")
    return (
        f"route={parameters.get('route', '?')}, "
        f"{parameters.get('moving_width', '?')}x"
        f"{parameters.get('moving_height', '?')}, "
        f"FOV={parameters.get('moving_hfov_deg', '?')} deg, "
        f"light={parameters.get('lighting', '?')}"
        f"@{parameters.get('lighting_scale', '?')}x, "
        f"blur={parameters.get('motion_blur_kernel', '?')}"
        f"@{parameters.get('motion_blur_angle_deg', '?')} deg, "
        f"frames={parameters.get('target_route_frames', '?')}, "
        f"sampling={sampling}, "
        f"settle={parameters.get('settle_seconds', '?')} s, "
        f"skip={parameters.get('post_pose_skip', '?')}, "
        f"timeouts={parameters.get('frame_timeout_seconds', '?')}/"
        f"{parameters.get('startup_timeout_seconds', '?')} s"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _valid_dataset_root(path: Path) -> bool:
    raw = path / "raw_images"
    return all(
        (raw / name).is_dir()
        for name in ("static", "moving", "camera_info")
    )


def _has_authoritative_results(path: Path) -> bool:
    """Return true only when at least one calibration method is available."""

    summary = _read_json(path / "SUMMARY.json")
    if not summary:
        return False
    available = summary.get("available_methods")
    if isinstance(available, int):
        return available > 0
    methods = summary.get("methods", [])
    return isinstance(methods, list) and any(
        isinstance(row, dict)
        and (
            row.get("artifact_status") == "available"
            or row.get("status") == "available"
        )
        for row in methods
    )


def _image_count(path: Path) -> int:
    return sum(
        item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        for item in path.iterdir()
    ) if path.is_dir() else 0


def discover_prepared_datasets(
    repository_root: Path,
) -> list[PreparedDatasetSummary]:
    """Discover only local layout-v2 datasets and prepared data_local inputs."""
    root = repository_root.resolve()
    candidates: set[Path] = set()
    for base in (root / "results", root / "data_local"):
        if not base.is_dir():
            continue
        for raw in base.rglob("raw_images"):
            if _valid_dataset_root(raw.parent):
                candidates.add(raw.parent.resolve())
    summaries: list[PreparedDatasetSummary] = []
    for path in candidates:
        descriptor = _read_json(path / "dataset.json")
        raw = path / "raw_images"
        category = str(descriptor.get("category") or "other")
        dataset_id = str(descriptor.get("id") or path.name)
        storage = descriptor.get("storage", {})
        display_name = (
            "BASELINE — Route 2"
            if category == "simulation"
            and isinstance(storage, dict)
            and storage.get("factor") == "baseline"
            else dataset_id
        )
        has_results = _has_authoritative_results(path)
        summaries.append(
            PreparedDatasetSummary(
                id=dataset_id,
                display_name=display_name,
                category=category,
                description=(
                    f"{category.replace('_', ' ')}; immutable layout-v2 dataset"
                    if descriptor
                    else "prepared local camera-rig dataset"
                ),
                path=path,
                static_camera_ids=tuple(
                    sorted(
                        item.stem
                        for item in (raw / "static").iterdir()
                        if item.is_file()
                        and item.suffix.lower() in IMAGE_SUFFIXES
                    )
                ),
                moving_frames=_image_count(raw / "moving"),
                has_results=has_results,
            )
        )
    unique: dict[tuple[str, str], PreparedDatasetSummary] = {}
    for item in summaries:
        key = item.category, item.id
        previous = unique.get(key)
        if previous is None or (
            item.has_results,
            item.path.is_relative_to(root / "results"),
            str(item.path),
        ) > (
            previous.has_results,
            previous.path.is_relative_to(root / "results"),
            str(previous.path),
        ):
            unique[key] = item
    order = {"real_vehicle": 0, "simulation": 1, "other": 2}
    return sorted(
        unique.values(),
        key=lambda item: (
            order.get(item.category, 9),
            item.display_name != "BASELINE — Route 2",
            item.display_name.lower(),
        ),
    )


def discover_raw_input_folders(
    repository_root: Path,
) -> list[RawInputSummary]:
    root = repository_root.resolve() / "data_local"
    if not root.is_dir():
        return []
    loose = [
        path
        for path in root.iterdir()
        if path.is_file() and classify_path(path) is not None
    ]
    children = [path for path in root.iterdir() if path.is_dir()]
    role_children = [
        path
        for path in children
        if image_directory_role(path, root) is not None
    ]
    auxiliary = ("rosbag", "mcap", "bag")
    one_acquisition = bool(role_children) and all(
        image_directory_role(path, root) is not None
        or any(token in path.name.lower() for token in auxiliary)
        for path in children
    )
    directories = [root] if loose or one_acquisition else children
    result: list[RawInputSummary] = []
    for directory in directories:
        files = [path for path in directory.rglob("*") if path.is_file()]
        item = RawInputSummary(
            path=directory.resolve(),
            videos=sum(path.suffix.lower() in VIDEO_SUFFIXES for path in files),
            images=sum(path.suffix.lower() in IMAGE_SUFFIXES for path in files),
            intrinsics=sum(classify_path(path) == "intrinsics" for path in files),
            recordings=sum(path.suffix.lower() in MCAP_SUFFIXES for path in files),
        )
        if item.videos or item.images or item.intrinsics or item.recordings:
            result.append(item)
    return sorted(
        result,
        key=lambda item: (
            item.path != root,
            -item.path.stat().st_mtime,
            str(item.path).lower(),
        ),
    )


def discover_simulation_experiments(
    repository_root: Path, *, include_v2: bool = True
) -> list[SimulationExperimentSummary]:
    """Read simulation experiments only from layout-v2 dataset descriptors."""
    del include_v2
    root = repository_root.resolve()
    entries: list[SimulationExperimentSummary] = []
    simulation_root = root / "results" / "simulation"
    if simulation_root.is_dir():
        for descriptor_path in sorted(simulation_root.rglob("dataset.json")):
            descriptor = _read_json(descriptor_path)
            if descriptor.get("layout_version") != 2:
                continue
            dataset_root = descriptor_path.parent
            parameters = {
                **BASELINE_SIMULATION_PARAMETERS,
                **(descriptor.get("simulation_parameters") or {}),
            }
            storage = descriptor.get("storage", {})
            factor = (
                str(storage.get("factor", "mixed"))
                if isinstance(storage, dict)
                else "mixed"
            )
            value = (
                str(storage.get("value", dataset_root.name))
                if isinstance(storage, dict)
                else dataset_root.name
            )
            entries.append(
                SimulationExperimentSummary(
                    variant=str(descriptor.get("id") or dataset_root.name),
                    factor=factor,
                    value=value,
                    moving_frames=_image_count(
                        dataset_root / "raw_images" / "moving"
                    ),
                    has_results=_has_authoritative_results(dataset_root),
                    dataset_root=(
                        dataset_root if _valid_dataset_root(dataset_root) else None
                    ),
                    parameters=parameters,
                )
            )
    if not any(
        all(
            item.parameters.get(key) == value
            for key, value in BASELINE_SIMULATION_PARAMETERS.items()
        )
        for item in entries
    ):
        entries.append(
            SimulationExperimentSummary(
                variant="route2",
                factor="all baseline parameters",
                value="Route 2 baseline parameter vector",
                moving_frames=None,
                has_results=False,
                dataset_root=None,
                parameters=dict(BASELINE_SIMULATION_PARAMETERS),
            )
        )
    return sorted(
        entries,
        key=lambda item: (
            item.variant != "route2",
            item.factor,
            item.variant,
        ),
    )


def find_matching_simulation(
    entries: list[SimulationExperimentSummary],
    parameters: dict[str, Any],
) -> SimulationExperimentSummary | None:
    matches: list[SimulationExperimentSummary] = []
    for entry in entries:
        equal = True
        for key in BASELINE_SIMULATION_PARAMETERS:
            first = entry.parameters.get(key)
            second = parameters.get(key)
            if isinstance(first, (int, float)) and isinstance(
                second, (int, float)
            ):
                if not math.isclose(float(first), float(second), abs_tol=1e-9):
                    equal = False
                    break
            elif first != second:
                equal = False
                break
        if equal:
            matches.append(entry)
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda item: (
            item.dataset_root is None,
            not item.has_results,
            item.variant != "route2",
            item.variant,
        ),
    )[0]

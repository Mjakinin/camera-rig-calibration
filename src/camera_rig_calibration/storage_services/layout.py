from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config.models import (
    DatasetCategory,
    RigConfig,
)
from .dataset.discovery import safe_id


BUS_BASELINE: dict[str, Any] = {
    "route_name": "route2",
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
    "route_frame_counts": {"route2": 189, "route1": 352},
}


@dataclass(frozen=True)
class StorageKey:
    category: str
    relative: Path
    factor: str
    value: str
    canonical_id: str


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any, length: int = 10) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:length]


def _equal(first: Any, second: Any) -> bool:
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(
            float(first), float(second), rel_tol=1e-9, abs_tol=1e-6
        )
    return first == second


def _number(value: float | int) -> str:
    return f"{float(value):g}"


def _sampling_bucket(config: RigConfig) -> str:
    value = config.sampling.target_hz
    if value is not None:
        return f"{_number(value)}Hz"
    return "native_rate"


def _real_storage_key(config: RigConfig) -> StorageKey:
    rate = _sampling_bucket(config)
    relative = Path(rate) / safe_id(
        config.project.experiment_id or config.dataset.id
    )
    return StorageKey(
        category=DatasetCategory.REAL_VEHICLE.value,
        relative=relative,
        factor="sampling_rate",
        value=rate,
        canonical_id=config.project.experiment_id or config.dataset.id,
    )


def _simulation_parameters(config: RigConfig) -> dict[str, Any]:
    simulation = config.simulation
    return {
        "route_name": simulation.route_name,
        "moving_width": simulation.moving_width,
        "moving_height": simulation.moving_height,
        "moving_hfov_deg": simulation.moving_hfov_deg,
        "lighting": simulation.lighting,
        "lighting_scale": simulation.lighting_scale,
        "motion_blur_kernel": simulation.motion_blur_kernel,
        "motion_blur_angle_deg": simulation.motion_blur_angle_deg,
        "target_route_frames": simulation.target_route_frames,
        "route_sampling_strategy": simulation.route_sampling_strategy,
        "settle_seconds": simulation.settle_seconds,
        "post_pose_skip": simulation.post_pose_skip,
        "frame_timeout_seconds": simulation.frame_timeout_seconds,
        "startup_timeout_seconds": simulation.startup_timeout_seconds,
    }


def _changed_groups(
    parameters: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, tuple[str, ...]]:
    groups = {
        "route": ("route_name",),
        "density": ("target_route_frames", "route_sampling_strategy"),
        "resolution": ("moving_width", "moving_height"),
        "fov": ("moving_hfov_deg",),
        "lighting": ("lighting", "lighting_scale"),
        "motion_blur": (
            "motion_blur_kernel",
            "motion_blur_angle_deg",
        ),
        "capture": (
            "settle_seconds",
            "post_pose_skip",
            "frame_timeout_seconds",
            "startup_timeout_seconds",
        ),
    }
    changed = {
        group: fields
        for group, fields in groups.items()
        if any(
            not _equal(parameters.get(field), baseline.get(field))
            for field in fields
        )
    }
    route_counts = baseline.get("route_frame_counts", {})
    expected_route_frames = (
        route_counts.get(str(parameters.get("route_name")))
        if isinstance(route_counts, dict)
        else None
    )
    if (
        expected_route_frames is not None
        and parameters.get("route_sampling_strategy")
        == "original_route_poses"
        and _equal(
            parameters.get("target_route_frames"),
            expected_route_frames,
        )
    ):
        changed.pop("density", None)
    return changed


def _density_value(parameters: dict[str, Any]) -> str:
    strategy = str(parameters["route_sampling_strategy"])
    stride = re.search(r"stride_(\d+)", strategy)
    offset = re.search(r"offset_(\d+)", strategy)
    if stride:
        value = f"stride_{stride.group(1)}"
        if offset:
            value += f"_offset_{offset.group(1)}"
        return value
    frames = parameters.get("target_route_frames")
    return safe_id(f"frames_{frames}_{strategy}")


def _capture_value(
    parameters: dict[str, Any], baseline: dict[str, Any]
) -> str:
    labels = {
        "settle_seconds": "settle",
        "post_pose_skip": "skip",
        "frame_timeout_seconds": "frame_timeout",
        "startup_timeout_seconds": "startup_timeout",
    }
    units = {
        "settle_seconds": "s",
        "frame_timeout_seconds": "s",
        "startup_timeout_seconds": "s",
    }
    values = []
    for field, label in labels.items():
        value = parameters[field]
        if _equal(value, baseline[field]):
            continue
        values.append(f"{label}_{_number(value)}{units.get(field, '')}")
    return safe_id("__".join(values))


def _factor_value(
    factor: str, parameters: dict[str, Any], baseline: dict[str, Any]
) -> str:
    if factor == "route":
        return safe_id(str(parameters["route_name"]))
    if factor == "density":
        return _density_value(parameters)
    if factor == "resolution":
        return f"{parameters['moving_width']}x{parameters['moving_height']}"
    if factor == "fov":
        return f"{_number(parameters['moving_hfov_deg'])}deg"
    if factor == "lighting":
        value = str(parameters["lighting"])
        if not _equal(parameters["lighting_scale"], 1.0):
            value += f"_{_number(parameters['lighting_scale'])}x"
        return safe_id(value)
    if factor == "motion_blur":
        value = f"kernel_{parameters['motion_blur_kernel']}"
        if not _equal(parameters["motion_blur_angle_deg"], 0.0):
            value += f"_angle_{_number(parameters['motion_blur_angle_deg'])}deg"
        return safe_id(value)
    if factor == "capture":
        return _capture_value(parameters, baseline)
    raise ValueError(f"Unknown simulation factor: {factor}")


def _simulation_storage_key(config: RigConfig) -> StorageKey:
    parameters = _simulation_parameters(config)
    baseline = {**BUS_BASELINE, **config.simulation.world_baseline}
    world_id = safe_id(config.simulation.world_id)
    relative = classify_simulation_parameters(
        parameters,
        experiment_id=config.project.experiment_id or config.dataset.id,
        baseline_experiment_id=config.project.experiment_id,
        world_id=world_id,
        baseline=baseline,
    )
    factor = (
        relative.parts[-2]
        if len(relative.parts) >= 2
        else "mixed"
    )
    value = relative.name
    return StorageKey(
        category=DatasetCategory.SIMULATION.value,
        relative=relative,
        factor=factor,
        value=value,
        canonical_id=config.project.experiment_id or config.dataset.id,
    )


def classify_simulation_parameters(
    parameters: dict[str, Any],
    *,
    experiment_id: str,
    baseline_experiment_id: str | None = None,
    world_id: str = "bus",
    baseline: dict[str, Any] | None = None,
) -> Path:
    """Return the canonical schema-v5 relative path for an inventory row."""
    if world_id != "bus":
        raise ValueError(
            "only the built-in bus Gazebo world is supported"
        )
    normalized_baseline = {**BUS_BASELINE, **(baseline or {})}
    normalized = dict(normalized_baseline)
    supplied = dict(parameters)
    if "route" in supplied and "route_name" not in supplied:
        supplied["route_name"] = supplied.pop("route")
    normalized.update(
        {
            key: value
            for key, value in supplied.items()
            if key in normalized_baseline
        }
    )
    changed = _changed_groups(normalized, normalized_baseline)
    prefix = Path()
    if not changed:
        baseline_id = safe_id(
            baseline_experiment_id
            or str(normalized_baseline["route_name"])
        )
        return (
            prefix
            / "baseline"
            / baseline_id
        )
    if len(changed) == 1:
        factor = next(iter(changed))
        return (
            prefix
            / factor
            / _factor_value(factor, normalized, normalized_baseline)
        )
    label = safe_id(experiment_id)
    value = f"{label}_{_digest({'world': world_id, 'parameters': normalized})}"
    return prefix / "mixed" / value


def storage_key(config: RigConfig) -> StorageKey:
    if config.dataset.category is DatasetCategory.SIMULATION:
        return _simulation_storage_key(config)
    return _real_storage_key(config)


def canonical_result_root(config: RigConfig) -> Path:
    key = storage_key(config)
    return config.project.output_root.resolve() / key.category / key.relative


def canonical_dataset_root(config: RigConfig) -> Path:
    """Return the experiment root that owns both input and result artifacts.

    ``dataset_cache_root`` remains an internal preparation-cache setting.  A
    published dataset is part of its experiment and therefore lives beside
    ``methods/`` and ``RESULTS.*`` instead of in a second public tree.
    """
    return canonical_result_root(config)


def queue_temporary_root(config: RigConfig, queue_id: str) -> Path:
    return (
        config.project.workspace_root.resolve()
        / "temporary_runs"
        / safe_id(queue_id)
    )


def storage_manifest(config: RigConfig) -> dict[str, Any]:
    key = storage_key(config)
    return {
        "schema_version": 5,
        "layout_version": 2,
        "category": key.category,
        "relative_path": key.relative.as_posix(),
        "factor": key.factor,
        "value": key.value,
        "canonical_id": key.canonical_id,
        "result_root": str(canonical_result_root(config)),
        "dataset_root": str(canonical_dataset_root(config)),
    }

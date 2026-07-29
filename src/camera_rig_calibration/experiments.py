from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config.models import (
    AP01Settings,
    AP02Settings,
    AP03Settings,
    ColmapSettings,
    EvaluationSettings,
    MarkerSettings,
    ObservationQualitySettings,
    RigConfig,
    effective_observation_quality,
)
from .dataset.manifest import DatasetManifest
from .dataset.discovery import safe_id
from .observations import ResolvedSelections
from .methods.common.aruco_utils import effective_detector_config
from .storage_layout import (
    canonical_dataset_root,
    canonical_result_root,
    storage_key,
    storage_manifest,
)


STAGE_ORDER = (
    "capture_import",
    "input_preparation",
    "marker_detection_pnp",
    "observation_quality",
    "colmap",
    "method_estimation",
    "evaluation",
    "comparison",
    "report",
)


PARAMETER_INVALIDATION: dict[str, str] = {
    "simulation.route_name": "capture_import",
    "simulation.moving_width": "capture_import",
    "simulation.moving_height": "capture_import",
    "simulation.moving_hfov_deg": "capture_import",
    "simulation.lighting": "capture_import",
    "simulation.lighting_scale": "capture_import",
    "simulation.motion_blur_kernel": "capture_import",
    "simulation.motion_blur_angle_deg": "capture_import",
    "sampling.target_hz": "input_preparation",
    "markers.dictionary": "marker_detection_pnp",
    "markers.detection_mode": "marker_detection_pnp",
    "markers.accepted_ids": "observation_quality",
    "observation_quality": "observation_quality",
    "markers.length_m": "method_estimation",
    "methods.ap01.root_camera": "method_estimation",
    "methods.ap02.reference_marker_id": "method_estimation",
    "methods.ap03.single.scale_marker_id": "method_estimation",
    "methods.ap03.multi.marker_ids": "method_estimation",
    "colmap": "colmap",
    "evaluation": "evaluation",
    "reporting": "report",
}


@dataclass(frozen=True)
class ExperimentPaths:
    category: str
    experiment_id: str
    root: Path
    dataset_root: Path
    datasets: Path
    methods: Path
    evaluations: Path
    comparisons: Path
    attempts: Path
    artifacts: Path
    staging: Path

    @property
    def inputs(self) -> Path:
        """The layout-v2 experiment has one dataset and no input-id layer."""
        return self.datasets


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any, length: int = 12) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:length]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(
    manifest: DatasetManifest | None, dataset_root: Path
) -> str:
    """Fingerprint normalized input content, independent from timestamps/paths."""
    provenance: list[dict[str, Any]] = [
        {
            "role": f"source:{item.role}",
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in (manifest.files if manifest is not None else [])
        if item.sha256
    ]
    # Source provenance alone is insufficient for captures: the same world and
    # route can produce different frames. Always include the normalized camera
    # inputs so recaptures and post-processing variants receive distinct IDs.
    raw_images = dataset_root / "raw_images"
    if raw_images.is_dir():
        for path in sorted(item for item in raw_images.rglob("*") if item.is_file()):
            provenance.append(
                {
                    "role": (
                        "prepared:"
                        f"{path.relative_to(dataset_root).as_posix()}"
                    ),
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not provenance:
        metadata = dataset_root / "metadata"
        if metadata.is_dir():
            for path in sorted(item for item in metadata.rglob("*") if item.is_file()):
                provenance.append(
                    {
                        "role": (
                            "prepared:"
                            f"{path.relative_to(dataset_root).as_posix()}"
                        ),
                        "sha256": _file_sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    payload = {
        "static_camera_ids": (
            [camera.id for camera in manifest.static_cameras] if manifest else []
        ),
        "moving_camera_id": manifest.moving_camera.id if manifest else None,
        "sampling_hz": manifest.sampling_hz if manifest else None,
        "files": sorted(
            provenance,
            key=lambda item: (
                str(item["role"]),
                str(item["sha256"]),
                int(item["size_bytes"] or 0),
            ),
        ),
    }
    return f"input_{_digest(payload)}"


def result_category(config: RigConfig) -> str:
    return config.dataset.category.value


def experiment_paths(config: RigConfig) -> ExperimentPaths:
    key = storage_key(config)
    category = key.category
    experiment_id = config.project.experiment_id or config.dataset.id
    root = canonical_result_root(config)
    dataset_root = canonical_dataset_root(config)
    staging = (
        config.project.workspace_root.resolve()
        / "temporary_runs"
        / f"standalone_{key.canonical_id}"
        / "jobs"
    )
    return ExperimentPaths(
        category=category,
        experiment_id=experiment_id,
        root=root,
        dataset_root=dataset_root,
        datasets=dataset_root,
        methods=root / "methods",
        evaluations=root / "evaluations",
        comparisons=root,
        attempts=root / "attempts",
        artifacts=(
            config.project.workspace_root.resolve()
            / "cache"
            / "colmap"
            / key.category
            / key.relative
        ),
        staging=staging,
    )


def colmap_artifact_fingerprint(
    config: RigConfig, method_id: str, input_id: str
) -> str:
    """Identify a reusable COLMAP stage without method-scale selections.

    AP01 reconstructs only the moving sequence; AP03 reconstructs the grouped
    static and moving image set. Root-camera and scale-marker choices are
    deliberately excluded because they are consumed after COLMAP.
    """
    if method_id == "ap01":
        family = "ap01_moving"
        cameras: list[str] = []
    elif method_id == "ap03":
        family = "ap03_grouped"
        cameras = [camera.id for camera in config.static_cameras]
    else:
        raise ValueError(f"Method '{method_id}' has no COLMAP artifact")
    colmap = config.colmap.model_dump(mode="json")
    colmap.pop("reuse", None)
    return _digest(
        {
            "family": family,
            "input_id": input_id,
            "static_cameras": cameras,
            "moving_camera": config.moving_camera.id,
            "colmap": colmap,
        },
        64,
    )


def experiment_fingerprint(config: RigConfig) -> str:
    simulation = {
        "world": str(config.simulation.world)
        if config.simulation.world is not None
        else None,
        "route": str(config.simulation.route)
        if config.simulation.route is not None
        else None,
        "route_name": config.simulation.route_name,
        "moving_model_name": config.simulation.moving_model_name,
        "moving_width": config.simulation.moving_width,
        "moving_height": config.simulation.moving_height,
        "moving_hfov_deg": config.simulation.moving_hfov_deg,
        "lighting": config.simulation.lighting,
        "lighting_scale": config.simulation.lighting_scale,
        "motion_blur_kernel": config.simulation.motion_blur_kernel,
        "motion_blur_angle_deg": config.simulation.motion_blur_angle_deg,
        "target_route_frames": config.simulation.target_route_frames,
        "route_sampling_strategy": (
            config.simulation.route_sampling_strategy
        ),
        "world_id": config.simulation.world_id,
        "world_baseline": config.simulation.world_baseline,
    }
    payload = {
        "category": result_category(config),
        "experiment_id": config.project.experiment_id or config.dataset.id,
        "scene_type": config.dataset.scene_type.value,
        "static_cameras": [
            {"id": camera.id, "label": camera.label}
            for camera in config.static_cameras
        ],
        "moving_camera": config.moving_camera.id,
        "simulation": (
            simulation
            if result_category(config) == "simulation"
            else None
        ),
        "sampling": config.sampling.model_dump(mode="json"),
    }
    return _digest(payload, 64)


def _method_payload(
    config: RigConfig, method_id: str, selections: ResolvedSelections
) -> dict[str, Any]:
    method_settings = (
        getattr(config.methods, method_id).model_dump(mode="json")
        if method_id in {"ap01", "ap02", "ap03"}
        else config.methods.extensions.get(method_id, {})
    )
    effective_quality, quality_sources = effective_observation_quality(
        config, method_id
    )
    payload: dict[str, Any] = {
        "method_id": method_id,
        "settings": method_settings,
        "marker_detection": {
            "settings": config.markers.model_dump(mode="json"),
            "effective_detector": effective_detector_config(
                config.markers.detection_mode,
                config.markers.dictionary,
            ),
        },
        "observation_quality": {
            "effective": effective_quality.model_dump(mode="json"),
            "sources": quality_sources,
        },
    }
    if method_id in {"ap01", "ap03"}:
        payload["colmap"] = config.colmap.model_dump(mode="json")
    if method_id == "ap01":
        payload["resolved_root_camera"] = selections.root_camera
    elif method_id == "ap02":
        payload["resolved_ap02_reference_marker_id"] = (
            selections.ap02_reference_marker_id
        )
    elif method_id == "ap03":
        payload["resolved_single_scale_marker_id"] = (
            selections.ap03_single_scale_marker_id
        )
        payload["resolved_multi_marker_ids"] = list(
            selections.ap03_multi_marker_ids
        )
    return payload


def method_fingerprint(
    config: RigConfig, method_id: str, selections: ResolvedSelections
) -> str:
    return _digest(_method_payload(config, method_id, selections), 64)


def evaluation_fingerprint(config: RigConfig, anchor_marker_id: int) -> str:
    return _digest(
        {
            "evaluation": config.evaluation.model_dump(mode="json"),
            "anchor_marker_id": anchor_marker_id,
            "marker_length_m": config.markers.length_m,
        },
        64,
    )


def _token(value: Any) -> str:
    text = str(value).lower().strip()
    text = text.replace(".", "p")
    return re.sub(r"[^a-z0-9_-]+", "_", text).strip("_") or "value"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def _method_defaults(method_id: str) -> dict[str, Any]:
    models = {
        "ap01": AP01Settings,
        "ap02": AP02Settings,
        "ap03": AP03Settings,
    }
    return models[method_id]().model_dump(mode="json")


def method_config_diff(
    config: RigConfig,
    method_id: str,
    selections: ResolvedSelections | None = None,
) -> dict[str, dict[str, Any]]:
    del selections
    if method_id in {"ap01", "ap02", "ap03"}:
        current = getattr(config.methods, method_id).model_dump(mode="json")
        defaults = _method_defaults(method_id)
        current.pop("observation_quality", None)
        defaults.pop("observation_quality", None)
    else:
        current = dict(config.methods.extensions.get(method_id, {}))
        defaults = {}
    if method_id in {"ap01", "ap03"}:
        current["colmap"] = config.colmap.model_dump(mode="json")
        defaults["colmap"] = ColmapSettings().model_dump(mode="json")
    current["markers"] = config.markers.model_dump(mode="json")
    defaults["markers"] = MarkerSettings().model_dump(mode="json")
    effective_quality, quality_sources = effective_observation_quality(
        config, method_id
    )
    current["observation_quality"] = effective_quality.model_dump(mode="json")
    current["observation_quality_source"] = quality_sources
    defaults["observation_quality"] = (
        type(config.observation_quality)().model_dump(mode="json")
    )
    defaults["observation_quality_source"] = {
        field_name: "global"
        for field_name in defaults["observation_quality"]
    }
    flat_current = _flatten(current)
    flat_defaults = _flatten(defaults)
    return {
        key: {"baseline": flat_defaults.get(key), "value": value}
        for key, value in flat_current.items()
        if value != flat_defaults.get(key)
    }


_LABEL_FIELDS = {
    "root_camera": "root",
    "top_moving_per_marker": "moving_top_per_marker",
    "scale_top_per_marker": "scale_top_per_marker",
    "reference_marker_id": "ref_marker",
    "reference_marker_maximum_frames": "ref_frames",
    "top_per_marker": "frame_top_per_marker",
    "top_per_marker_pair": "frame_top_per_pair",
    "maximum_total_frames": "frame_cap",
    "static_only_ba_max_function_evaluations": "static_nfev",
    "combined_ba_max_function_evaluations": "combined_nfev",
    "ba_robust_loss": "loss",
    "ba_robust_loss_scale_px": "loss_scale_px",
    "single.scale_marker_id": "single_marker",
    "multi.marker_ids": "multi_markers",
    "scale.reprojection_threshold_px": "scale_reproj_px",
    "scale.ransac_iterations": "scale_ransac",
    "scale.minimum_inliers": "scale_inliers",
    "scale.maximum_observations_per_marker": "scale_obs_per_marker",
    "colmap.matcher": "matcher",
    "colmap.gpu_mode": "gpu",
    "colmap.maximum_image_size": "image_size",
    "colmap.maximum_features": "features",
    "colmap.sequential_overlap": "overlap",
    "colmap.loop_detection": "loop",
    "colmap.mapper_minimum_matches": "mapper_matches",
    "colmap.ap03_maximum_image_size": "ap03_image_size",
    "colmap.ap03_maximum_features": "ap03_features",
    "colmap.ap03_loop_detection": "ap03_loop",
    "markers.dictionary": "aruco",
    "markers.length_m": "marker_m",
    "markers.accepted_ids": "marker_ids",
    "markers.detection_mode": "aruco_mode",
    "observation_quality.maximum_pnp_reprojection_error_px": "pnp_reproj_px",
    "observation_quality.minimum_marker_area_ratio": "marker_area_ratio",
    "observation_quality.require_positive_depth": "positive_depth",
    "observation_quality.maximum_marker_distance_m": "marker_distance_m",
    "quality_override_fields": "quality_override",
}


def _label_value(value: Any) -> str:
    if isinstance(value, list):
        return "-".join(_token(item) for item in value)
    return _token(value)


def automatic_method_label(
    method_id: str,
    *,
    methods: Any,
    markers: MarkerSettings,
    observation_quality: ObservationQualitySettings,
    colmap: ColmapSettings,
) -> str:
    """Name a result only from calibration-affecting deviations to baseline."""
    overrides: dict[str, Any] = {}
    if method_id in {"ap01", "ap02", "ap03"}:
        current = getattr(methods, method_id).model_dump(mode="json")
        defaults = _method_defaults(method_id)
        overrides = current.pop("observation_quality", {})
        defaults.pop("observation_quality", None)
    else:
        current = dict(methods.extensions.get(method_id, {}))
        defaults = {}
    if method_id in {"ap01", "ap03"}:
        current["colmap"] = colmap.model_dump(mode="json")
        defaults["colmap"] = ColmapSettings().model_dump(mode="json")
    current["markers"] = markers.model_dump(mode="json")
    defaults["markers"] = MarkerSettings().model_dump(mode="json")
    effective_quality = observation_quality.model_dump(mode="json")
    for field_name, value in overrides.items():
        if value is not None:
            effective_quality[field_name] = value
    current["observation_quality"] = effective_quality
    current["quality_override_fields"] = sorted(
        field_name
        for field_name, value in overrides.items()
        if value is not None
    )
    defaults["observation_quality"] = ObservationQualitySettings().model_dump(
        mode="json"
    )
    defaults["quality_override_fields"] = []
    flat_current = _flatten(current)
    flat_defaults = _flatten(defaults)
    differences = [
        (path, value)
        for path, value in sorted(flat_current.items())
        if value != flat_defaults.get(path)
        and path not in {"colmap.executable", "colmap.reuse"}
    ]
    if not differences:
        return "baseline"
    tokens = [
        f"{_LABEL_FIELDS.get(path, path.replace('.', '_'))}_{_label_value(value)}"
        for path, value in differences
    ]
    readable = "__".join(tokens)
    if len(readable) <= 88:
        return safe_id(readable)
    digest = _digest(
        {path: value for path, value in differences},
        8,
    )
    return safe_id(f"{readable[:79].rstrip('_')}__{digest}")


def method_variant_name(
    config: RigConfig, method_id: str, selections: ResolvedSelections
) -> str:
    if method_id == "ap01":
        primary = f"root_{_token(selections.root_camera)}"
    elif method_id == "ap02":
        primary = f"ref_marker_{selections.ap02_reference_marker_id}"
    elif method_id == "ap03":
        markers = selections.ap03_multi_marker_ids
        marker_set_hash = _digest(list(markers), 6)
        if config.methods.ap03_multi.marker_ids == "auto":
            multi = f"multi_all_{len(markers)}_{marker_set_hash}"
        elif len(markers) <= 4:
            multi = "multi_" + "-".join(
                str(marker) for marker in markers
            )
        else:
            multi = f"multi_set_{len(markers)}_{marker_set_hash}"
        primary = (
            f"single_marker_{selections.ap03_single_scale_marker_id}__{multi}"
        )
    else:
        primary = _token(method_id)

    tokens = [primary]
    if method_id in {"ap01", "ap03"}:
        tokens.append(f"matcher_{config.colmap.matcher}")
    diff = method_config_diff(config, method_id, selections)
    ignored = {
        "root_camera",
        "reference_marker_id",
        "scale_marker_id",
        "marker_ids",
        "single.scale_marker_id",
        "multi.marker_ids",
        "colmap.matcher",
    }
    additions: list[str] = []
    for path, values in sorted(diff.items()):
        if path in ignored:
            continue
        short = {
            "reprojection_threshold_px": "reproj",
            "minimum_inliers": "inliers",
            "markers.length_m": "marker_size_m",
            "observation_quality.minimum_marker_area_ratio": "area_ratio",
            "observation_quality.maximum_pnp_reprojection_error_px": "pnp_reproj",
            "observation_quality.maximum_marker_distance_m": "distance_max",
        }.get(path, path.split(".")[-1])
        additions.append(f"{_token(short)}_{_token(values['value'])}")
    if additions:
        tokens.extend(additions[:4])
        if len(additions) > 4:
            tokens.append(f"plus_{len(additions) - 4}")
    elif method_id == "ap02":
        tokens.append("baseline")
    fingerprint = method_fingerprint(config, method_id, selections)[:8]
    readable = "__".join(tokens)
    maximum_readable = 80 - len(fingerprint) - 1
    readable = readable[:maximum_readable].rstrip("_")
    return f"{readable}_{fingerprint}"


def method_result_label(config: RigConfig, method_id: str) -> str:
    """Return the deterministic public label for the effective configuration."""
    if (
        method_id in {"ap01", "ap03"}
        and config.colmap.executable != "auto"
        and Path(config.colmap.executable).is_absolute()
    ):
        return safe_id(config.project.run_label or "baseline")
    return automatic_method_label(
        method_id,
        methods=config.methods,
        markers=config.markers,
        observation_quality=config.observation_quality,
        colmap=config.colmap,
    )


def experiment_manifest_payload(
    config: RigConfig,
    paths: ExperimentPaths,
    input_id: str,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the canonical dataset identity from the final experiment config."""

    return {
        "schema_version": 5,
        "layout_version": 2,
        "id": paths.experiment_id,
        "category": paths.category,
        "scene_type": config.dataset.scene_type.value,
        "source_kind": config.dataset.source_kind.value,
        "storage": storage_manifest(config),
        "experiment_fingerprint": experiment_fingerprint(config),
        "input_fingerprint": input_id,
        "created_at": (
            created_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds")
        ),
        "static_cameras": [
            {"id": camera.id, "label": camera.label}
            for camera in config.static_cameras
        ],
        "moving_camera": {"id": config.moving_camera.id},
        "simulation_parameters": (
            {
                "route": config.simulation.route_name,
                "moving_width": config.simulation.moving_width,
                "moving_height": config.simulation.moving_height,
                "moving_hfov_deg": config.simulation.moving_hfov_deg,
                "lighting": config.simulation.lighting,
                "lighting_scale": config.simulation.lighting_scale,
                "motion_blur_kernel": config.simulation.motion_blur_kernel,
                "motion_blur_angle_deg": (
                    config.simulation.motion_blur_angle_deg
                ),
                "target_route_frames": (
                    config.simulation.target_route_frames
                ),
                "route_sampling_strategy": (
                    config.simulation.route_sampling_strategy
                ),
                "settle_seconds": config.simulation.settle_seconds,
                "post_pose_skip": config.simulation.post_pose_skip,
                "frame_timeout_seconds": (
                    config.simulation.frame_timeout_seconds
                ),
                "startup_timeout_seconds": (
                    config.simulation.startup_timeout_seconds
                ),
            }
            if paths.category == "simulation"
            else None
        ),
    }


def write_experiment_manifest(
    config: RigConfig, paths: ExperimentPaths, input_id: str
) -> Path:
    """Write and validate the single immutable layout-v2 dataset descriptor."""
    paths.dataset_root.mkdir(parents=True, exist_ok=True)
    destination = paths.dataset_root / "dataset.json"
    payload = experiment_manifest_payload(config, paths, input_id)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        previous_fingerprint = existing.get(
            "experiment_fingerprint"
        )
        if (
            previous_fingerprint
            and previous_fingerprint
            != payload["experiment_fingerprint"]
        ):
            raise RuntimeError(
                f"Experiment ID '{paths.experiment_id}' already belongs to a "
                "different rig/capture parameter contract. Choose a new "
                "dataset/experiment ID instead of mixing inputs."
            )
        previous_input = existing.get("input_fingerprint")
        if previous_input and previous_input != input_id:
            raise RuntimeError(
                f"Experiment ID '{paths.experiment_id}' already contains a "
                "different immutable dataset. Choose a new experiment ID."
            )
        return destination
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def first_invalidated_stage(changed_paths: list[str]) -> str:
    if not changed_paths:
        return "report"
    stages: list[str] = []
    for changed in changed_paths:
        match = next(
            (
                stage
                for prefix, stage in PARAMETER_INVALIDATION.items()
                if changed == prefix or changed.startswith(prefix + ".")
            ),
            "method_estimation",
        )
        stages.append(match)
    return min(stages, key=STAGE_ORDER.index)

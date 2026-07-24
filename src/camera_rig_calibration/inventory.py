from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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
    """Return one compact, complete parameter vector for the terminal catalogue."""
    sampling = str(parameters.get("route_sampling_strategy", "?")).replace("_", " ")
    return (
        f"route={parameters.get('route', '?')}, "
        f"{parameters.get('moving_width', '?')}x{parameters.get('moving_height', '?')}, "
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
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _experiment_manifest_for(path: Path, repository_root: Path) -> tuple[Path, dict]:
    root = repository_root.resolve()
    for candidate in (path.resolve(), *path.resolve().parents):
        if candidate == root:
            break
        manifest = candidate / "experiment.yaml"
        if not manifest.is_file():
            continue
        try:
            payload = yaml.safe_load(
                manifest.read_text(encoding="utf-8")
            ) or {}
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            return candidate, payload
    return path, {}


def discover_prepared_datasets(repository_root: Path) -> list[PreparedDatasetSummary]:
    candidates: set[Path] = set()
    for root_name in ("results", "datasets", "data_local"):
        root = repository_root / root_name
        if not root.is_dir():
            continue
        for raw in root.rglob("raw_images"):
            if (
                raw.is_dir()
                and (raw / "static").is_dir()
                and (raw / "moving").is_dir()
                and (raw / "camera_info").is_dir()
            ):
                candidates.add(raw.parent.resolve())
    summaries = []
    for path in candidates:
        raw = path / "raw_images"
        camera_ids = tuple(
            sorted(
                item.stem
                for item in (raw / "static").iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
        )
        moving_frames = sum(
            1
            for item in (raw / "moving").iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        manifest = _read_json(path / "dataset_manifest.json")
        try:
            relative = path.relative_to(repository_root.resolve())
        except ValueError:
            relative = path
        parts = relative.parts
        experiment_root, experiment_manifest = _experiment_manifest_for(
            path, repository_root
        )
        manifest_category = str(experiment_manifest.get("category", ""))
        if manifest_category in {"real_vehicle", "simulation"}:
            dataset_id = str(
                experiment_manifest.get("id") or path.name
            )
            display_name = (
                "BASELINE — Route 2"
                if (
                    manifest_category == "simulation"
                    and experiment_manifest.get("storage", {}).get(
                        "factor"
                    )
                    == "baseline"
                )
                else dataset_id
            )
            category = manifest_category
            description = (
                f"{manifest_category.replace('_', ' ')}; "
                "schema-v5 canonical input"
            )
        elif len(parts) >= 4 and parts[:2] == ("results", "real_vehicle"):
            dataset_id = parts[2]
            display_name = dataset_id
            category = "real_vehicle"
            description = "real vehicle; rigcal v2 content-addressed input"
        elif len(parts) >= 4 and parts[:2] == ("results", "simulation"):
            dataset_id = parts[2]
            display_name = (
                "BASELINE — Route 2" if dataset_id == "route2" else dataset_id
            )
            category = "simulation"
            description = "Gazebo simulation; rigcal v2 content-addressed input"
        elif len(parts) >= 3 and parts[:2] == ("results", "real_vehicle_data"):
            dataset_id = parts[2] if path.name == "00_shared_input" else path.name
            display_name = dataset_id
            category = "real_vehicle"
            description = "real vehicle; prepared frames and camera intrinsics"
        elif (
            manifest.get("scene_type") == "simulation"
            or len(parts) >= 2
            and parts[:2] == ("results", "bus_real_data")
        ):
            dataset_id = str(manifest.get("dataset_id") or path.name)
            display_name = (
                "BASELINE — Route 2"
                if path.name == "route2"
                else "Legacy shared simulation baseline"
                if path.name == "bus_real_data_ref_marker_v1"
                else path.name
            )
            category = "simulation"
            description = "Gazebo simulation; prepared frames and camera intrinsics"
        else:
            dataset_id = str(manifest.get("dataset_id") or path.name)
            display_name = dataset_id
            category = "other"
            description = "prepared canonical camera-rig dataset"
        has_results = (
            (path / "FINAL_RESULTS").is_dir()
            or (path / "99_FINAL_RESULTS").is_dir()
            or (path.parent / "99_FINAL_RESULTS").is_dir()
            or any(path.rglob("METHOD_STATUS.json"))
            or (
                manifest_category in {"real_vehicle", "simulation"}
                and (
                    repository_root
                    / "results"
                    / experiment_root.relative_to(
                        repository_root / "datasets"
                    )
                    / "PUBLISHED.json"
                ).is_file()
                if experiment_root.is_relative_to(
                    repository_root / "datasets"
                )
                else False
            )
            or (
                len(parts) >= 4
                and parts[0] == "results"
                and (
                    repository_root
                    / "results"
                    / parts[1]
                    / parts[2]
                    / "legacy_results"
                ).is_dir()
            )
        )
        summaries.append(
            PreparedDatasetSummary(
                id=dataset_id,
                display_name=display_name,
                category=category,
                description=description,
                path=path,
                static_camera_ids=camera_ids,
                moving_frames=moving_frames,
                has_results=has_results,
            )
        )
    # Some legacy FOV inputs exist once as an intermediate prepared directory and
    # once beside their final results. They are the same logical dataset. Show the
    # result-bearing location once instead of asking users to distinguish aliases.
    unique: dict[tuple[str, str], PreparedDatasetSummary] = {}
    for item in summaries:
        key = (item.category, item.id)
        current = unique.get(key)
        if current is None or (
            item.has_results,
            "00_prepared_datasets" not in item.path.parts,
            str(item.path),
        ) > (
            current.has_results,
            "00_prepared_datasets" not in current.path.parts,
            str(current.path),
        ):
            unique[key] = item
    category_order = {"real_vehicle": 0, "simulation": 1, "other": 2}
    return sorted(
        unique.values(),
        key=lambda item: (
            category_order.get(item.category, 9),
            item.display_name != "BASELINE — Route 2",
            item.display_name.lower(),
        ),
    )


def discover_raw_input_folders(repository_root: Path) -> list[RawInputSummary]:
    root = repository_root / "data_local"
    if not root.is_dir():
        return []
    loose_inputs = [
        path
        for path in root.iterdir()
        if path.is_file() and classify_path(path) is not None
    ]
    # Loose files make data_local itself one acquisition. Its nested ROS bag
    # directory belongs to that acquisition and must not appear as a duplicate.
    child_directories = [
        path for path in root.iterdir() if path.is_dir()
    ]
    role_children = [
        path
        for path in child_directories
        if image_directory_role(path, root) is not None
    ]
    auxiliary_names = ("rosbag", "mcap", "bag")
    all_children_belong_to_loose_acquisition = bool(role_children) and all(
        image_directory_role(path, root) is not None
        or any(token in path.name.lower() for token in auxiliary_names)
        for path in child_directories
    )
    directories = (
        [root]
        if loose_inputs or all_children_belong_to_loose_acquisition
        else child_directories
    )
    result = []
    for directory in directories:
        files = [path for path in directory.rglob("*") if path.is_file()]
        videos = sum(path.suffix.lower() in VIDEO_SUFFIXES for path in files)
        images = sum(path.suffix.lower() in IMAGE_SUFFIXES for path in files)
        intrinsics = sum(classify_path(path) == "intrinsics" for path in files)
        recordings = sum(path.suffix.lower() in MCAP_SUFFIXES for path in files)
        if videos or images or intrinsics or recordings:
            result.append(
                RawInputSummary(directory.resolve(), videos, images, intrinsics, recordings)
            )
    return sorted(
        result,
        key=lambda item: (
            0 if item.path == root.resolve() else 1,
            -item.path.stat().st_mtime,
            str(item.path).lower(),
        ),
    )


def _experiment_parameters(group: str, variant: str, metadata: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    parameters = dict(BASELINE_SIMULATION_PARAMETERS)
    factor = str(metadata.get("parameter") or metadata.get("ablation") or group)
    value = variant
    if group == "moving_cam/fov":
        match = re.search(r"fov_(\d+(?:\.\d+)?)", variant)
        fov = (
            69.1
            if "baseline" in variant
            else float(match.group(1)) if match else 69.1
        )
        parameters["moving_hfov_deg"] = fov
        factor, value = "moving-camera horizontal FOV", f"{fov:g} deg"
    elif group == "moving_cam/res":
        width = int(metadata.get("moving_width", 1280))
        height = int(metadata.get("moving_height", 720))
        parameters.update({"moving_width": width, "moving_height": height})
        factor, value = "moving-camera resolution", f"{width}x{height}"
    elif group == "moving_cam/motion_blur":
        kernel = int(metadata.get("kernel_size", 0))
        angle = float(metadata.get("angle_deg", 0.0))
        parameters.update(
            {"motion_blur_kernel": kernel, "motion_blur_angle_deg": angle}
        )
        factor, value = "moving-camera motion blur", f"kernel={kernel}, angle={angle:g} deg"
    elif group == "world/lighting":
        level = str(metadata.get("light_level", variant.removeprefix("ceiling_")))
        parameters["lighting"] = level
        factor, value = "physical ceiling lighting", level
    elif group == "world/route":
        route = str(metadata.get("variant", variant))
        frames = int(metadata.get("num_route_frames", 189 if route == "route2" else 352))
        parameters.update({"route": route, "target_route_frames": frames})
        factor, value = (
            ("BASELINE — all defaults", "Route 2 baseline parameter vector")
            if route == "route2"
            else ("moving-camera route", f"{route}, {frames} frames")
        )
    elif group == "moving_cam/density":
        frames = int(metadata.get("selected_frame_count", metadata.get("route_frame_count", 189)))
        parameters["target_route_frames"] = frames
        stride = int(metadata.get("stride", 1))
        offset = int(metadata.get("offset", 0))
        if frames == 189 and stride == 1:
            strategy = "original_route_poses"
        elif metadata.get("images_newly_rendered_in_gazebo"):
            strategy = "resampled_route_poses"
        else:
            strategy = f"existing_subset_stride_{stride}_offset_{offset}"
        parameters["route_sampling_strategy"] = strategy
        factor, value = "route sampling density", f"{frames} frames"
    return factor, value, parameters


def discover_simulation_experiments(
    repository_root: Path, *, include_v2: bool = True
) -> list[SimulationExperimentSummary]:
    ablation_root = repository_root / "results/bus_real_data/ablation"
    entries: list[SimulationExperimentSummary] = []
    if ablation_root.is_dir():
        for path in sorted(ablation_root.glob("*/*/*/VARIANT_METADATA.json")):
            relative = path.relative_to(ablation_root)
            group = "/".join(relative.parts[:2])
            variant_root = path.parent
            metadata = _read_json(path)
            variant = str(metadata.get("variant") or variant_root.name)
            factor, value, parameters = _experiment_parameters(group, variant, metadata)
            frame_value = metadata.get(
                "selected_frame_count",
                metadata.get("num_route_frames", metadata.get("route_frame_count")),
            )
            raw = variant_root / "raw_images"
            dataset_root = (
                variant_root
                if all((raw / name).is_dir() for name in ("static", "moving", "camera_info"))
                else None
            )
            frames = (
                int(frame_value)
                if frame_value is not None
                else sum(
                    1
                    for item in (raw / "moving").iterdir()
                    if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
                )
                if (raw / "moving").is_dir()
                else None
            )
            entries.append(
                SimulationExperimentSummary(
                    variant,
                    factor,
                    value,
                    frames,
                    (variant_root / "FINAL_RESULTS").is_dir()
                    and any((variant_root / "FINAL_RESULTS").iterdir()),
                    dataset_root,
                    parameters,
                )
            )
    baseline_root = ablation_root / "world/route/route2"
    if not any(entry.variant == "route2" for entry in entries) and baseline_root.is_dir():
        entries.append(
            SimulationExperimentSummary(
                "route2",
                "all baseline parameters",
                "Route 2, 1280x720, 69.1 deg, no blur, baseline light",
                189,
                (baseline_root / "FINAL_RESULTS").is_dir(),
                baseline_root
                if all(
                    (baseline_root / "raw_images" / name).is_dir()
                    for name in ("static", "moving", "camera_info")
                )
                else None,
                dict(BASELINE_SIMULATION_PARAMETERS),
            )
        )

    # Canonical rigcal manifests are authoritative and may be nested below
    # factor/world folders.
    v2_root = repository_root / "results" / "simulation"
    if include_v2 and v2_root.is_dir():
        experiment_roots = {
            path.parent for path in v2_root.rglob("experiment.yaml")
        }
        experiment_roots.update(
            path.parent for path in v2_root.rglob("legacy_manifest.json")
        )
        for experiment_root in sorted(experiment_roots):
            manifest_path = experiment_root / "experiment.yaml"
            if experiment_root.is_symlink():
                continue
            try:
                relative = experiment_root.relative_to(v2_root)
            except ValueError:
                continue
            legacy_manifest = _read_json(
                experiment_root / "legacy_manifest.json"
            )
            dataset_experiment = (
                repository_root / "datasets" / "simulation" / relative
            )
            input_roots = sorted(
                path
                for path in (
                    dataset_experiment / "inputs"
                    if (dataset_experiment / "inputs").is_dir()
                    else experiment_root / "inputs"
                ).glob("*")
                if path.is_dir()
            )
            prepared = next(
                (
                    path
                    for path in input_roots
                    if all(
                        (path / "raw_images" / name).is_dir()
                        for name in ("static", "moving", "camera_info")
                    )
                ),
                None,
            )
            if prepared is None:
                continue
            metadata_candidates = list(
                (prepared / "metadata").glob("*METADATA.json")
            )
            metadata = (
                _read_json(metadata_candidates[0])
                if metadata_candidates
                else {}
            )
            if manifest_path.is_file():
                try:
                    experiment_payload = yaml.safe_load(
                        manifest_path.read_text(encoding="utf-8")
                    ) or {}
                except Exception:
                    experiment_payload = {}
            else:
                experiment_payload = {}
            variant = str(
                experiment_payload.get("id") or experiment_root.name
            )
            discovered_parameters = dict(
                legacy_manifest.get("parameters")
                or experiment_payload.get("simulation_parameters")
                or metadata.get("simulation_parameters")
                or {}
            )
            if discovered_parameters:
                parameters = {
                    **BASELINE_SIMULATION_PARAMETERS,
                    **discovered_parameters,
                }
                factor = str(
                    legacy_manifest.get("factor")
                    or "migrated simulation experiment"
                )
                value = str(
                    legacy_manifest.get("value")
                    or format_simulation_parameters(parameters)
                )
            else:
                factor, value, parameters = _experiment_parameters(
                    str(metadata.get("parameter") or "rigcal_v2"),
                    variant,
                    metadata,
                )
            moving_frames = sum(
                1
                for path in (prepared / "raw_images" / "moving").glob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            entries.append(
                SimulationExperimentSummary(
                    variant=variant,
                    factor=factor,
                    value=value,
                    moving_frames=moving_frames,
                    has_results=(
                        (experiment_root / "PUBLISHED.json").is_file()
                        or
                        (experiment_root / "methods").is_dir()
                        or bool(legacy_manifest.get("has_results"))
                        or (experiment_root / "legacy_results").is_dir()
                    ),
                    dataset_root=prepared,
                    parameters=parameters,
                )
            )

    # Include simulation captures created by rigcal itself. Their resolved config is
    # authoritative, so freely combined parameters can also be detected on the next run.
    results_root = repository_root / "results"
    config_paths = (
        sorted(results_root.rglob("resolved_config.yaml"))
        if include_v2
        else []
    )
    for config_path in config_paths:
        if "run_history" in config_path.parts:
            continue
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        dataset = payload.get("dataset", {})
        simulation = payload.get("simulation", {})
        if dataset.get("scene_type") != "simulation" or not simulation:
            continue
        parameters = {
            "route": simulation.get("route_name", "route2"),
            "moving_width": simulation.get("moving_width", 1280),
            "moving_height": simulation.get("moving_height", 720),
            "moving_hfov_deg": simulation.get("moving_hfov_deg", 69.1),
            "lighting": simulation.get("lighting", "baseline"),
            "lighting_scale": simulation.get("lighting_scale", 1.0),
            "motion_blur_kernel": simulation.get("motion_blur_kernel", 0),
            "motion_blur_angle_deg": simulation.get("motion_blur_angle_deg", 0.0),
            "target_route_frames": simulation.get("target_route_frames"),
            "route_sampling_strategy": simulation.get(
                "route_sampling_strategy", "original_route_poses"
            ),
            "settle_seconds": simulation.get("settle_seconds", 0.35),
            "post_pose_skip": simulation.get("post_pose_skip", 5),
            "frame_timeout_seconds": simulation.get(
                "frame_timeout_seconds", 3.0
            ),
            "startup_timeout_seconds": simulation.get(
                "startup_timeout_seconds", 60.0
            ),
        }
        run_root = config_path.parent
        run_manifest = _read_json(run_root / "run_manifest.json")
        dataset_manifest = _read_json(run_root / "00_INPUT/dataset_manifest.json")
        prepared_text = dataset_manifest.get("prepared_root")
        prepared = Path(prepared_text).expanduser() if prepared_text else None
        if prepared is not None and not prepared.is_absolute():
            prepared = (repository_root / prepared).resolve()
        raw = prepared / "raw_images" if prepared is not None else None
        reusable = (
            prepared
            if raw is not None
            and all((raw / name).is_dir() for name in ("static", "moving", "camera_info"))
            else None
        )
        moving_frames = dataset_manifest.get("moving_camera", {}).get("image_count")
        entries.append(
            SimulationExperimentSummary(
                f"{dataset.get('id', config_path.parents[2].name)}/{run_root.name}",
                "rigcal combined parameters",
                format_simulation_parameters(parameters),
                int(moving_frames) if moving_frames is not None else None,
                run_manifest.get("status") == "completed",
                reusable,
                parameters,
            )
        )
    parameter_keys = tuple(BASELINE_SIMULATION_PARAMETERS)
    grouped: dict[tuple[Any, ...], SimulationExperimentSummary] = {}
    for entry in entries:
        signature = tuple(entry.parameters.get(key) for key in parameter_keys)
        current = grouped.get(signature)
        if current is None or (
            (
                entry.dataset_root is not None
                and (
                    "/results/simulation/" in str(entry.dataset_root)
                    or "/datasets/simulation/" in str(entry.dataset_root)
                )
            ),
            entry.dataset_root is not None,
            entry.has_results,
            entry.variant == "route2",
            "baseline" not in entry.variant,
        ) > (
            (
                current.dataset_root is not None
                and (
                    "/results/simulation/" in str(current.dataset_root)
                    or "/datasets/simulation/" in str(current.dataset_root)
                )
            ),
            current.dataset_root is not None,
            current.has_results,
            current.variant == "route2",
            "baseline" not in current.variant,
        ):
            grouped[signature] = entry
    baseline_signature = tuple(
        BASELINE_SIMULATION_PARAMETERS.get(key) for key in parameter_keys
    )
    return sorted(
        grouped.values(),
        key=lambda item: (
            tuple(item.parameters.get(key) for key in parameter_keys)
            != baseline_signature,
            item.factor,
            item.variant,
        ),
    )


def find_matching_simulation(
    entries: list[SimulationExperimentSummary], parameters: dict[str, Any]
) -> SimulationExperimentSummary | None:
    keys = BASELINE_SIMULATION_PARAMETERS.keys()
    matches = []
    for entry in entries:
        equal = True
        for key in keys:
            first = entry.parameters.get(key)
            second = parameters.get(key)
            if isinstance(first, (int, float)) and isinstance(second, (int, float)):
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

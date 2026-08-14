from __future__ import annotations

import hashlib
import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .models import RigConfig


PATH_FIELDS = {
    "workspace_root",
    "dataset_cache_root",
    "output_root",
    "prepared_root",
    "input_root",
    "video",
    "frames",
    "intrinsics",
    "intrinsic_calibration_video",
    "intrinsic_calibration_images",
    "path",
    "images",
    "world",
    "route",
    "resource_paths",
}

OPTIONAL_LIMIT_PATHS = (
    ("methods", "ap01", "top_moving_per_marker"),
    ("methods", "ap01", "scale_top_per_marker"),
    ("methods", "ap02", "reference_marker_maximum_frames"),
    ("methods", "ap02", "top_per_marker"),
    ("methods", "ap02", "top_per_marker_pair"),
    ("methods", "ap02", "maximum_total_frames"),
    ("methods", "ap03", "scale", "maximum_observations_per_marker"),
)
QUALITY_OVERRIDE_PATHS = tuple(
    ("methods", method_id, "observation_quality", field_name)
    for method_id in ("ap01", "ap02", "ap03")
    for field_name in (
        "minimum_marker_area_ratio",
        "maximum_pnp_reprojection_error_px",
        "require_positive_depth",
        "maximum_marker_distance_m",
    )
)
EXPLICIT_NULL_PATHS = OPTIONAL_LIMIT_PATHS + QUALITY_OVERRIDE_PATHS


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _nested_set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = payload
    for part in path[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            nested = {}
            current[part] = nested
        current = nested
    current[path[-1]] = value


def _migrate_schema_v5(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    """Apply only unambiguous migrations within the supported v5 contract."""

    migrated = deepcopy(payload)
    colmap = migrated.get("colmap")
    if isinstance(colmap, dict) and "gpu_mode" in colmap:
        legacy_gpu_mode = str(colmap.pop("gpu_mode")).strip().lower()
        compute_mode = {
            "false": "cpu_baseline",
            "true": "gpu",
            "auto": "auto",
        }.get(legacy_gpu_mode)
        if compute_mode is None:
            raise ValueError(
                f"colmap.gpu_mode has an unsupported value in {source}: "
                f"{legacy_gpu_mode!r}"
            )
        configured = colmap.get("compute_mode")
        if configured is not None and configured != compute_mode:
            raise ValueError(
                "colmap.compute_mode conflicts with the deprecated "
                f"colmap.gpu_mode in {source}"
            )
        colmap["compute_mode"] = compute_mode
        warnings.warn(
            "Migrated colmap.gpu_mode to colmap.compute_mode.",
            DeprecationWarning,
            stacklevel=3,
        )

    quality = migrated.setdefault("observation_quality", {})
    if not isinstance(quality, dict):
        return migrated
    if "minimum_marker_area_px2" in quality:
        old_area = quality.pop("minimum_marker_area_px2")
        if old_area not in (0, 0.0, None):
            raise ValueError(
                "observation_quality.minimum_marker_area_px2 uses an "
                "absolute pixel-area threshold that cannot be converted to an "
                "image-resolution-neutral ratio. Remove it and configure "
                "minimum_marker_area_ratio explicitly in "
                f"{source}."
            )
        quality.setdefault("minimum_marker_area_ratio", 0.000008)
        warnings.warn(
            "Migrated observation_quality.minimum_marker_area_px2=0 to "
            "minimum_marker_area_ratio=0.000008.",
            DeprecationWarning,
            stacklevel=3,
        )

    evaluation = migrated.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("anchor_marker_id") == "auto_common":
        evaluation["anchor_marker_id"] = "auto"
        warnings.warn(
            "evaluation.anchor_marker_id=auto_common is deprecated; migrated "
            "to auto.",
            DeprecationWarning,
            stacklevel=3,
        )

    for path in OPTIONAL_LIMIT_PATHS:
        present, value = _nested_get(migrated, path)
        if present and value == 0:
            _nested_set(migrated, path, None)
            warnings.warn(
                f"Migrated {'.'.join(path)}=0 to null (unlimited).",
                DeprecationWarning,
                stacklevel=3,
            )
    return migrated


def _serialized_config_payload(config: RigConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json", exclude_none=True, by_alias=True)
    complete = config.model_dump(mode="json", exclude_none=False, by_alias=True)
    for path in EXPLICIT_NULL_PATHS:
        _present, value = _nested_get(complete, path)
        _nested_set(payload, path, value)
    return payload


def _user_config_payload(config: RigConfig) -> dict[str, Any]:
    """Omit dormant compatibility defaults from Wizard-generated YAML.

    Loading the compact file restores the same model defaults. Internal
    provenance continues to use :func:`save_config` and therefore retains the
    complete schema payload.
    """

    payload = deepcopy(_serialized_config_payload(config))
    methods = payload.get("methods", {})
    enabled = methods.get("enabled", []) if isinstance(methods, dict) else []
    if isinstance(methods, dict):
        ap01 = methods.get("ap01")
        if isinstance(ap01, dict) and ap01.get("method_contract") == "baseline_v1":
            for key in (
                "top_moving_per_marker",
                "scale_top_per_marker",
                "direct_quality_gate",
                "relay_quality_gate",
                "direct_relay_consistency",
            ):
                ap01.pop(key, None)
        ap02 = methods.get("ap02")
        if isinstance(ap02, dict) and ap02.get("method_contract") == "baseline_v1":
            for key, default in (
                ("frame_selection_strategy", "smart_v1"),
                ("initialization_strategy", "maximum_frontier_v1"),
                ("graph_edge_weight_strategy", "geometric_observation_quality_v1"),
                ("reprojection_model", "pinhole_v1"),
            ):
                if ap02.get(key) == default:
                    ap02.pop(key, None)
        ap03 = methods.get("ap03")
        if isinstance(ap03, dict) and ap03.get("method_contract") == "baseline_v1":
            baseline_scale_input = (
                ap03.get(
                    "scale_input_policy",
                    "registered_image_redetection_v1",
                )
                == "registered_image_redetection_v1"
            )
            for key, default in (
                ("feature_limit_policy", "colmap_defaults_v1"),
                (
                    "scale_input_policy",
                    "registered_image_redetection_v1",
                ),
            ):
                if ap03.get(key) == default:
                    ap03.pop(key, None)
            if baseline_scale_input and isinstance(ap03.get("scale"), dict):
                ap03["scale"].pop(
                    "maximum_observations_per_marker", None
                )

    colmap = payload.get("colmap")
    if isinstance(colmap, dict) and len(enabled) == 1:
        method_id = enabled[0]
        if method_id == "ap02":
            payload.pop("colmap", None)
        elif method_id == "ap01":
            ap01 = methods.get("ap01", {})
            if ap01.get("method_contract", "baseline_v1") == "baseline_v1":
                for key in tuple(colmap):
                    if key not in {"executable", "reuse"}:
                        colmap.pop(key, None)
        elif method_id == "ap03":
            ap03 = methods.get("ap03", {})
            if ap03.get(
                "feature_limit_policy", "colmap_defaults_v1"
            ) == "colmap_defaults_v1":
                for key in (
                    "maximum_image_size",
                    "maximum_features",
                    "loop_detection",
                    "ap03_maximum_image_size",
                    "ap03_maximum_features",
                    "ap03_loop_detection",
                ):
                    colmap.pop(key, None)
    return payload


def _resolve_path(path: Path, base: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_config_paths(config: RigConfig, config_path: Path) -> RigConfig:
    base = config_path.resolve().parent
    payload = config.model_dump(mode="python")

    def visit(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            # This dictionary contains scientific baseline values. A key named
            # `route` is a route identifier here, not a filesystem path.
            if key == "world_baseline":
                return value
            return {name: visit(item, name) for name, item in value.items()}
        if isinstance(value, list):
            if key in {"images", "resource_paths"}:
                return [_resolve_path(Path(item), base) for item in value]
            return [visit(item, key) for item in value]
        if key in PATH_FIELDS and value is not None:
            return _resolve_path(Path(value), base)
        return value

    return RigConfig.model_validate(visit(payload))


def load_config(path: str | Path, *, resolve_paths: bool = True) -> RigConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    if payload.get("schema_version") != 5:
        raise ValueError(
            f"Only schema_version 5 is supported: {source}. Recreate the "
            "configuration with the current rigcal wizard."
        )
    payload = _migrate_schema_v5(payload, source)
    config = RigConfig.model_validate(payload)
    return resolve_config_paths(config, source) if resolve_paths else config


def save_config(config: RigConfig, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialized_config_payload(config)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def save_user_config(config: RigConfig, path: str | Path) -> Path:
    """Save a concise user-facing config with canonical defaults implicit."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        yaml.safe_dump(
            _user_config_payload(config),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def config_fingerprint(config: RigConfig) -> str:
    canonical = json.dumps(
        _serialized_config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

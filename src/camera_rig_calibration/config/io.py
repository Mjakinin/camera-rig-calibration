from __future__ import annotations

import hashlib
import json
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


def _resolve_path(path: Path, base: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_config_paths(config: RigConfig, config_path: Path) -> RigConfig:
    base = config_path.resolve().parent
    payload = config.model_dump(mode="python")

    def visit(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
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
    config = RigConfig.model_validate(payload)
    return resolve_config_paths(config, source) if resolve_paths else config


def save_config(config: RigConfig, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json", exclude_none=True, by_alias=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def config_fingerprint(config: RigConfig) -> str:
    canonical = json.dumps(
        config.model_dump(mode="json", exclude_none=True, by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

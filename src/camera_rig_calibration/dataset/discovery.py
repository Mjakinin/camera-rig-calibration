from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
INTRINSIC_SUFFIXES = {".json", ".yaml", ".yml"}
MCAP_SUFFIXES = {".mcap", ".db3"}
CHECKERBOARD_DIRECTORY_TOKENS = {
    "checker",
    "checkerboard",
    "chess",
    "chessboard",
    "intrinsic",
    "intrinsics",
    "intrinsic_images",
    "intrinsics_images",
    "calibration",
    "calibration_images",
}
MOVING_DIRECTORY_TOKENS = {
    "moving",
    "moving_camera",
    "moving_frames",
    "frames",
    "extracted_frames",
}
STATIC_DIRECTORY_TOKENS = {
    "static",
    "static_camera",
    "static_cameras",
    "static_images",
}


@dataclass(frozen=True)
class DiscoveredInput:
    path: Path
    kind: str
    suggested_id: str


def safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.-")
    return normalized or "dataset"


def classify_path(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in MCAP_SUFFIXES:
        return "mcap"
    if suffix in INTRINSIC_SUFFIXES:
        searchable = "/".join(part.lower() for part in path.parts[-4:])
        if any(
            token in searchable
            for token in ("camera_info", "intrinsic", "calibration", "meta.yaml", "meta.yml")
        ):
            return "intrinsics"
    return None


def discover_inputs(root: Path, *, recursive: bool = False) -> list[DiscoveredInput]:
    root = root.expanduser().resolve()
    if root.is_file():
        kind = classify_path(root)
        return [DiscoveredInput(root, kind, safe_id(root.stem))] if kind else []
    if not root.is_dir():
        return []
    candidates = root.rglob("*") if recursive else root.iterdir()
    result = []
    for path in candidates:
        if not path.is_file():
            continue
        kind = classify_path(path)
        if kind:
            result.append(DiscoveredInput(path.resolve(), kind, safe_id(path.stem)))
    return sorted(result, key=lambda item: (item.kind, item.path.name.lower()))


def _contains_role_token(tokens: set[str], keywords: set[str]) -> bool:
    return any(
        keyword in token
        for token in tokens
        for keyword in keywords
    )


def image_directory_role(directory: Path, root: Path) -> str | None:
    """Infer a media-folder role from keyword-containing directory names."""

    directory = directory.resolve()
    root = root.resolve()
    try:
        relative = directory.relative_to(root)
    except ValueError:
        relative = directory
    tokens = {
        safe_id(part).lower()
        for part in relative.parts
        if part not in {"", "."}
    }
    if _contains_role_token(tokens, CHECKERBOARD_DIRECTORY_TOKENS):
        return "checkerboard"
    # "static_frames_v2" must remain static even though it contains "frames".
    if _contains_role_token(tokens, STATIC_DIRECTORY_TOKENS):
        return "static"
    if _contains_role_token(tokens, MOVING_DIRECTORY_TOKENS):
        return "moving"
    return None


def media_path_role(path: Path, root: Path) -> str | None:
    """Infer image/video role from ancestors first, then from its filename."""

    role = image_directory_role(path.parent, root)
    if role is not None:
        return role
    stem = safe_id(path.stem).lower()
    tokens = {stem}
    if _contains_role_token(tokens, CHECKERBOARD_DIRECTORY_TOKENS):
        return "checkerboard"
    if _contains_role_token(tokens, STATIC_DIRECTORY_TOKENS):
        return "static"
    if _contains_role_token(tokens, {"moving"}):
        return "moving"
    return None


def discover_image_directories(root: Path) -> dict[str, list[Path]]:
    """Return non-empty image directories grouped by an explicit role hint."""

    root = root.expanduser().resolve()
    grouped: dict[str, set[Path]] = {
        "static": set(),
        "moving": set(),
        "checkerboard": set(),
    }
    if not root.is_dir():
        return {name: [] for name in grouped}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        role = image_directory_role(path.parent, root)
        if role is not None:
            grouped[role].add(path.parent.resolve())
    return {
        name: sorted(paths, key=lambda path: str(path).lower())
        for name, paths in grouped.items()
    }


def inspect_prepared_dataset(root: Path) -> dict[str, object]:
    dataset_root = root.expanduser().resolve()
    raw = dataset_root
    if (raw / "raw_images").is_dir():
        raw = raw / "raw_images"
    static_root = raw / "static"
    info_root = raw / "camera_info"
    moving_root = raw / "moving"

    static_images = [
        path
        for path in sorted(static_root.glob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    static_ids = [path.stem for path in static_images]
    moving_frames = [
        path
        for path in sorted(moving_root.glob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    intrinsic_ids = {
        path.stem
        for path in info_root.glob("*")
        if path.is_file() and path.suffix.lower() in INTRINSIC_SUFFIXES
    }
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except (OSError, json.JSONDecodeError):
            manifest = {}
    manifest_static_ids = [
        str(item.get("id"))
        for item in manifest.get("static_cameras", [])
        if isinstance(item, dict) and item.get("id")
    ]
    moving_value = manifest.get("moving_camera")
    manifest_moving_id = (
        str(moving_value.get("id"))
        if isinstance(moving_value, dict) and moving_value.get("id")
        else None
    )
    return {
        "raw_root": raw,
        "static_camera_ids": static_ids,
        "static_images": static_images,
        "moving_frames": moving_frames,
        "intrinsic_ids": intrinsic_ids,
        "manifest_static_camera_ids": manifest_static_ids,
        "manifest_moving_camera_id": manifest_moving_id,
        "manifest": manifest,
    }

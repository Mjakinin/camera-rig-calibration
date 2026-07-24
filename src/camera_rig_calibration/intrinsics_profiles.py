from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .dataset.discovery import safe_id


@dataclass(frozen=True)
class IntrinsicProfile:
    profile_id: str
    fingerprint: str
    root: Path
    intrinsics: Path
    width: int
    height: int
    distortion_model: str
    source_video: str | None
    created_at: str | None
    legacy: bool = False
    display_name: str | None = None

    @property
    def key(self) -> str:
        return f"{self.profile_id}@{self.fingerprint}"

    @property
    def label(self) -> str:
        return self.display_name or self.profile_id

    @property
    def size_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source(path: Path) -> str:
    source = path.resolve()
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(f"Intrinsic source does not exist: {source}")
    digest = hashlib.sha256()
    images = [
        item
        for item in sorted(source.iterdir())
        if item.is_file()
        and item.suffix.lower()
        in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    ]
    if not images:
        raise ValueError(f"Intrinsic image folder contains no images: {source}")
    for image in images:
        digest.update(image.name.encode("utf-8"))
        digest.update(sha256_file(image).encode("ascii"))
    return digest.hexdigest()


def read_intrinsics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = (
        json.loads(text)
        if path.suffix.lower() == ".json"
        else yaml.safe_load(text)
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Intrinsic file must contain a mapping: {path}")
    return payload


def intrinsic_dimensions(path: Path) -> tuple[int, int]:
    payload = read_intrinsics(path)
    return (
        int(payload.get("width", payload.get("image_width", 0)) or 0),
        int(payload.get("height", payload.get("image_height", 0)) or 0),
    )


def profile_fingerprint(
    video: Path,
    *,
    columns: int,
    rows: int,
    maximum_views: int,
    minimum_frame_gap: int,
    minimum_detections: int,
    scan_mode: str,
    scan_target_hz: float,
    preview_max_dimension: int,
) -> str:
    payload = {
        "source_sha256": sha256_source(video.resolve()),
        "columns": columns,
        "rows": rows,
        "maximum_views": maximum_views,
        "minimum_frame_gap": minimum_frame_gap,
        "minimum_detections": minimum_detections,
        "scan_mode": scan_mode,
        "scan_target_hz": scan_target_hz,
        "preview_max_dimension": preview_max_dimension,
        "engine_contract": "rigcal_intrinsics_v3_1",
    }
    if video.resolve().is_dir():
        payload["source_kind"] = "image_directory"
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def profile_directory(
    repository_root: Path, profile_id: str, fingerprint: str
) -> Path:
    return (
        repository_root.resolve()
        / "results"
        / "real_vehicle"
        / "_intrinsics"
        / safe_id(profile_id)
        / fingerprint[:12]
    )


def _profile_from_manifest(path: Path) -> IntrinsicProfile | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    intrinsic_value = payload.get("intrinsics", "intrinsics.json")
    intrinsic_path = (path.parent / str(intrinsic_value)).resolve()
    if not intrinsic_path.is_file():
        return None
    try:
        intrinsic_payload = read_intrinsics(intrinsic_path)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return None
    fingerprint = str(
        payload.get("fingerprint") or sha256_file(intrinsic_path)
    )
    return IntrinsicProfile(
        profile_id=safe_id(str(payload.get("profile_id") or path.parent.parent.name)),
        fingerprint=fingerprint,
        root=path.parent.resolve(),
        intrinsics=intrinsic_path,
        width=int(
            intrinsic_payload.get(
                "width", intrinsic_payload.get("image_width", 0)
            )
            or 0
        ),
        height=int(
            intrinsic_payload.get(
                "height", intrinsic_payload.get("image_height", 0)
            )
            or 0
        ),
        distortion_model=str(
            intrinsic_payload.get("distortion_model", "unknown")
        ),
        source_video=(
            str(payload.get("source_video"))
            if payload.get("source_video")
            else str(payload.get("source_images"))
            if payload.get("source_images")
            else intrinsic_payload.get(
                "source_video", intrinsic_payload.get("source_images")
            )
        ),
        created_at=(
            str(payload.get("created_at"))
            if payload.get("created_at")
            else None
        ),
        display_name=(
            str(payload.get("display_name"))
            if payload.get("display_name")
            else None
        ),
    )


def discover_intrinsic_profiles(repository_root: Path) -> list[IntrinsicProfile]:
    root = (
        repository_root.resolve()
        / "results"
        / "real_vehicle"
        / "_intrinsics"
    )
    if not root.is_dir():
        return []
    profiles: list[IntrinsicProfile] = []
    covered: set[Path] = set()
    covered_roots: set[Path] = set()
    for manifest in sorted(root.rglob("profile.yaml")):
        profile = _profile_from_manifest(manifest)
        if profile is not None:
            profiles.append(profile)
            covered.add(profile.intrinsics.resolve())
            covered_roots.add(profile.root.resolve())
    for path in sorted(root.rglob("*.json")):
        if (
            path.name in {"timings.json", "profile.json"}
            or path.resolve() in covered
            or any(path.resolve().is_relative_to(item) for item in covered_roots)
        ):
            continue
        try:
            payload = read_intrinsics(path)
            width = int(payload.get("width", payload.get("image_width", 0)) or 0)
            height = int(payload.get("height", payload.get("image_height", 0)) or 0)
            matrix = payload.get("K", payload.get("k"))
            if width <= 0 or height <= 0 or not isinstance(matrix, list):
                continue
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            continue
        profile_id = safe_id(
            path.parent.name
            if path.parent != root
            else path.stem.removesuffix("_moving_calib_camera")
        )
        profiles.append(
            IntrinsicProfile(
                profile_id=profile_id,
                fingerprint=sha256_file(path),
                root=path.parent.resolve(),
                intrinsics=path.resolve(),
                width=width,
                height=height,
                distortion_model=str(payload.get("distortion_model", "unknown")),
                source_video=payload.get("source_video"),
                created_at=None,
                legacy=True,
            )
        )
    unique: dict[tuple[str, str], IntrinsicProfile] = {}
    for profile in profiles:
        unique[(profile.profile_id, profile.fingerprint)] = profile
    return sorted(
        unique.values(),
        key=lambda item: (
            item.label.lower(),
            item.created_at or "",
            item.fingerprint,
        ),
    )


def resolve_intrinsic_profile(
    repository_root: Path, key: str
) -> IntrinsicProfile:
    profiles = discover_intrinsic_profiles(repository_root)
    if "@" in key:
        profile_id, prefix = key.rsplit("@", 1)
        matches = [
            item
            for item in profiles
            if item.profile_id == profile_id
            and item.fingerprint.startswith(prefix)
        ]
    else:
        matches = [item for item in profiles if item.profile_id == key]
    if not matches:
        raise FileNotFoundError(f"Unknown moving-camera intrinsics profile: {key}")
    if len(matches) > 1:
        keys = ", ".join(item.key for item in matches)
        raise ValueError(
            f"Intrinsics profile '{key}' is ambiguous; use one of: {keys}"
        )
    return matches[0]


def profile_manifest(
    *,
    profile_id: str,
    fingerprint: str,
    video: Path,
    intrinsics: Path,
    settings: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload = read_intrinsics(intrinsics)
    source = video.resolve()
    result = {
        "schema_version": 1,
        "profile_id": safe_id(profile_id),
        "display_name": profile_id.strip(),
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "intrinsics": "intrinsics.json",
        "image_width": int(payload.get("width", payload.get("image_width", 0)) or 0),
        "image_height": int(
            payload.get("height", payload.get("image_height", 0)) or 0
        ),
        "distortion_model": str(payload.get("distortion_model", "unknown")),
        "settings": settings,
        "elapsed_seconds": elapsed_seconds,
    }
    if source.is_dir():
        result.update(
            {
                "source_images": str(source),
                "source_images_sha256": sha256_source(source),
            }
        )
    else:
        result.update(
            {
                "source_video": str(source),
                "source_video_sha256": sha256_file(source),
            }
        )
    return result


def intrinsic_profile_references(
    repository_root: Path, profile: IntrinsicProfile
) -> tuple[Path, ...]:
    """Find saved configs that refer to the immutable profile key."""
    root = repository_root.resolve()
    candidates: list[Path] = []
    if (root / "workspace").is_dir():
        candidates.extend((root / "workspace").rglob("*.yaml"))
    if (root / "results").is_dir():
        candidates.extend((root / "results").rglob("*config*.yaml"))
    references: list[Path] = []
    keys = {profile.key, profile.profile_id}
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(key in text for key in keys):
            references.append(path.resolve())
    return tuple(sorted(set(references)))


def update_profile_alias(
    profile: IntrinsicProfile, display_name: str | None
) -> IntrinsicProfile:
    manifest = profile.root / "profile.yaml"
    if not manifest.is_file():
        raise RuntimeError(
            "Legacy intrinsics without profile.yaml cannot be renamed; "
            "create a managed profile version first"
        )
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    if display_name:
        payload["display_name"] = display_name.strip()
    else:
        payload.pop("display_name", None)
    temporary = manifest.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(manifest)
    updated = _profile_from_manifest(manifest)
    if updated is None:
        raise RuntimeError(f"Updated profile manifest is invalid: {manifest}")
    return updated


def active_intrinsic_profile_references(
    repository_root: Path, profile: IntrinsicProfile
) -> tuple[Path, ...]:
    root = repository_root.resolve() / "workspace" / "temporary_runs"
    if not root.is_dir():
        return ()
    keys = {profile.key, profile.profile_id}
    references: list[Path] = []
    for path in root.rglob("*.yaml"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(key in text for key in keys):
            references.append(path.resolve())
    return tuple(sorted(set(references)))


def delete_profile(
    repository_root: Path, profile: IntrinsicProfile
) -> None:
    active_references = active_intrinsic_profile_references(
        repository_root, profile
    )
    if active_references:
        raise RuntimeError(
            f"Profile {profile.key} is used by {len(active_references)} active "
            "temporary queue configuration(s); finish or delete those runs first"
        )
    profiles_root = (
        repository_root.resolve()
        / "results"
        / "real_vehicle"
        / "_intrinsics"
    )
    if not profile.root.is_relative_to(profiles_root):
        raise RuntimeError("Refusing to delete a profile outside _intrinsics")
    if profile.legacy and not (profile.root / "profile.yaml").is_file():
        profile.intrinsics.unlink()
        try:
            profile.root.rmdir()
        except OSError:
            pass
        return
    shutil.rmtree(profile.root)

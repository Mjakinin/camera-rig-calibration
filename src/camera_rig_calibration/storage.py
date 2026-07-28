from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CleanupTarget:
    path: Path
    kind: str
    experiment_root: Path | None = None


@dataclass(frozen=True)
class CleanupPlan:
    targets: tuple[CleanupTarget, ...]
    protected_paths: tuple[Path, ...]
    file_count: int
    logical_bytes: int
    reclaimable_bytes: int


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _files(targets: Iterable[CleanupTarget]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        path = target.path
        candidates = (
            [path]
            if path.is_file() or path.is_symlink()
            else (
                sorted(item for item in path.rglob("*") if item.is_file())
                if path.is_dir()
                else []
            )
        )
        for item in candidates:
            identity = item.absolute()
            if identity not in seen:
                seen.add(identity)
                files.append(item)
    return files


def _sizes(files: Iterable[Path]) -> tuple[int, int, int]:
    logical = 0
    inode_targets: dict[tuple[int, int], dict[str, int]] = {}
    count = 0
    for path in files:
        if path.is_symlink():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        logical += stat.st_size
        key = (stat.st_dev, stat.st_ino)
        item = inode_targets.setdefault(
            key,
            {"size": stat.st_size, "links": stat.st_nlink, "targets": 0},
        )
        item["targets"] += 1
    reclaimable = sum(
        item["size"]
        for item in inode_targets.values()
        if item["targets"] >= item["links"]
    )
    return count, logical, reclaimable


def protected_storage_paths(repository_root: Path) -> tuple[Path, ...]:
    """Return active transaction storage that cleanup must never touch."""
    temporary = repository_root.resolve() / "workspace" / "temporary_runs"
    return (temporary.resolve(),) if temporary.exists() else ()


def build_cleanup_plan(
    repository_root: Path, *, include_data_local: bool = False
) -> CleanupPlan:
    """Plan removal of reproducible caches, never canonical science data.

    Layout-v2 datasets and results are intentionally absent from this plan:
    raw images, observations, debug galleries, diagnostics, logs and
    provenance are all part of the experiment record.
    """
    root = repository_root.resolve()
    protected = protected_storage_paths(root)
    candidates: list[CleanupTarget] = []

    def add(path: Path, kind: str) -> None:
        if not (path.exists() or path.is_symlink()):
            return
        absolute = path.absolute()
        if any(
            _is_within(absolute, item) or _is_within(item, absolute)
            for item in protected
        ):
            return
        candidates.append(CleanupTarget(absolute, kind))

    workspace = root / "workspace"
    for relative in ("cache", "artifacts"):
        add(workspace / relative, "reproducible workspace cache")

    if include_data_local:
        local = root / "data_local"
        if local.is_dir():
            for child in sorted(local.iterdir()):
                add(child, "user data_local input")

    ordered = sorted(candidates, key=lambda item: len(item.path.parts))
    targets: list[CleanupTarget] = []
    for candidate in ordered:
        if any(_is_within(candidate.path, kept.path) for kept in targets):
            continue
        targets.append(candidate)
    files = _files(targets)
    count, logical, reclaimable = _sizes(files)
    return CleanupPlan(
        targets=tuple(targets),
        protected_paths=protected,
        file_count=count,
        logical_bytes=logical,
        reclaimable_bytes=reclaimable,
    )


def build_data_local_cleanup_plan(repository_root: Path) -> CleanupPlan:
    combined = build_cleanup_plan(repository_root, include_data_local=True)
    targets = tuple(
        target
        for target in combined.targets
        if target.kind == "user data_local input"
    )
    files = _files(targets)
    count, logical, reclaimable = _sizes(files)
    return CleanupPlan(
        targets=targets,
        protected_paths=combined.protected_paths,
        file_count=count,
        logical_bytes=logical,
        reclaimable_bytes=reclaimable,
    )


def _target_digest(targets: Iterable[CleanupTarget]) -> str:
    digest = hashlib.sha256()
    for path in sorted(_files(targets)):
        if path.is_symlink():
            digest.update(str(path).encode("utf-8"))
            digest.update(os.readlink(path).encode("utf-8"))
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def execute_cleanup(plan: CleanupPlan) -> dict[str, object]:
    """Delete exactly the paths previously returned in a reviewed plan."""
    digest = _target_digest(plan.targets)
    removed_targets: list[str] = []
    for target in sorted(
        plan.targets, key=lambda item: len(item.path.parts), reverse=True
    ):
        path = target.path
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)
        removed_targets.append(str(path))
    return {
        "removed_targets": removed_targets,
        "file_count": plan.file_count,
        "logical_bytes": plan.logical_bytes,
        "reclaimable_bytes_estimate": plan.reclaimable_bytes,
        "content_manifest_sha256": digest,
    }

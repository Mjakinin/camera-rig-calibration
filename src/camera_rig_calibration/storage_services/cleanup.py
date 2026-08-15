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
    scope_roots: tuple[Path, ...]
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


def _lexically_within(path: Path, parent: Path) -> bool:
    try:
        path.absolute().relative_to(parent.absolute())
        return True
    except ValueError:
        return False


def _build_plan(
    targets: Iterable[CleanupTarget],
    *,
    scope_roots: Iterable[Path],
    protected_paths: Iterable[Path] = (),
) -> CleanupPlan:
    ordered = sorted(
        targets, key=lambda item: (len(item.path.parts), str(item.path))
    )
    selected: list[CleanupTarget] = []
    for candidate in ordered:
        if any(
            _lexically_within(candidate.path, existing.path)
            for existing in selected
        ):
            continue
        selected.append(candidate)
    files = _files(selected)
    count, logical, reclaimable = _sizes(files)
    return CleanupPlan(
        targets=tuple(selected),
        protected_paths=tuple(
            path.absolute() for path in protected_paths
        ),
        scope_roots=tuple(path.absolute() for path in scope_roots),
        file_count=count,
        logical_bytes=logical,
        reclaimable_bytes=reclaimable,
    )


def _children(
    directory: Path,
    kind: str,
    *,
    excluded_names: frozenset[str] = frozenset(),
) -> list[CleanupTarget]:
    if not directory.is_dir():
        return []
    return [
        CleanupTarget(child.absolute(), kind)
        for child in sorted(directory.iterdir(), key=lambda item: item.name)
        if child.name not in excluded_names
    ]


def build_results_cleanup_plan(repository_root: Path) -> CleanupPlan:
    """Select every published result while retaining the empty root itself."""
    root = repository_root.resolve()
    results = root / "results"
    return _build_plan(
        _children(results, "published result and embedded dataset"),
        scope_roots=(results,),
    )


def build_preparation_cache_cleanup_plan(repository_root: Path) -> CleanupPlan:
    """Select reusable preparation caches, never published or local input."""
    root = repository_root.resolve()
    workspace = root / "workspace"
    candidates: list[CleanupTarget] = []
    path = workspace / "preparation_cache"
    if path.exists() or path.is_symlink():
        candidates.append(
            CleanupTarget(path.absolute(), "prepared dataset cache")
        )
    return _build_plan(
        candidates,
        scope_roots=(workspace,),
    )


def build_temporary_cleanup_plan(repository_root: Path) -> CleanupPlan:
    """Select all generated workspace state except dataset caches."""
    root = repository_root.resolve()
    workspace = root / "workspace"
    return _build_plan(
        _children(
            workspace,
            "temporary run, queue, batch or reusable artifact",
            excluded_names=frozenset({"preparation_cache", "README.md"}),
        ),
        scope_roots=(workspace,),
    )


def combine_cleanup_plans(*plans: CleanupPlan) -> CleanupPlan:
    """Combine independently reviewed plans and recalculate hardlink sizes."""
    return _build_plan(
        (
            target
            for plan in plans
            for target in plan.targets
        ),
        scope_roots=(
            root
            for plan in plans
            for root in plan.scope_roots
        ),
        protected_paths=(
            path
            for plan in plans
            for path in plan.protected_paths
        ),
    )


def build_cleanup_plan(repository_root: Path) -> CleanupPlan:
    """Build one plan for all non-result generated storage."""
    return combine_cleanup_plans(
        build_preparation_cache_cleanup_plan(repository_root),
        build_temporary_cleanup_plan(repository_root),
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
    for target in plan.targets:
        if not any(
            _lexically_within(target.path, root)
            and target.path.absolute() != root.absolute()
            for root in plan.scope_roots
        ):
            raise RuntimeError(
                "Refusing cleanup target outside its reviewed storage roots: "
                f"{target.path}"
            )
        if any(
            _is_within(target.path, protected)
            or _is_within(protected, target.path)
            for protected in plan.protected_paths
        ):
            raise RuntimeError(
                f"Refusing protected cleanup target: {target.path}"
            )
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
    remaining_targets = [
        str(target.path)
        for target in plan.targets
        if target.path.exists() or target.path.is_symlink()
    ]
    if remaining_targets:
        raise RuntimeError(
            "Cleanup verification failed; these selected targets remain: "
            + ", ".join(remaining_targets)
        )
    return {
        "removed_targets": removed_targets,
        "verified_removed": True,
        "remaining_targets": remaining_targets,
        "file_count": plan.file_count,
        "logical_bytes": plan.logical_bytes,
        "reclaimable_bytes_estimate": plan.reclaimable_bytes,
        "content_manifest_sha256": digest,
    }

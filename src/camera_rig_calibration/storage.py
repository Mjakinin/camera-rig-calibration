from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROTECTED_RUN_STATUSES = {
    "running",
    "failed",
    "interrupted",
    "waiting_for_selection",
    "failed_preflight",
}


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
        candidates = [path] if path.is_file() or path.is_symlink() else (
            sorted(item for item in path.rglob("*") if item.is_file())
            if path.is_dir()
            else []
        )
        for item in candidates:
            resolved = item.absolute()
            if resolved not in seen:
                seen.add(resolved)
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
        entry = inode_targets.setdefault(
            key,
            {"size": stat.st_size, "links": stat.st_nlink, "targets": 0},
        )
        entry["targets"] += 1
    reclaimable = sum(
        item["size"]
        for item in inode_targets.values()
        if item["targets"] >= item["links"]
    )
    return count, logical, reclaimable


def _run_payloads(results_root: Path) -> list[tuple[Path, dict]]:
    payloads: list[tuple[Path, dict]] = []
    if not results_root.is_dir():
        return payloads
    for manifest in results_root.rglob("run_manifest.json"):
        if "_views" in manifest.parts or "run_history" in manifest.parts:
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payloads.append((manifest.parent.resolve(), payload))
    return payloads


def protected_storage_paths(repository_root: Path) -> tuple[Path, ...]:
    root = repository_root.resolve()
    protected: set[Path] = set()
    temporary_runs = root / "workspace" / "temporary_runs"
    if temporary_runs.exists():
        protected.add(temporary_runs.resolve())
    for run, payload in _run_payloads(root / "results"):
        if str(payload.get("status", "")) not in PROTECTED_RUN_STATUSES:
            continue
        protected.add(run)
        experiment = payload.get("experiment_root")
        if experiment:
            protected.add(Path(str(experiment)).resolve())
        pointer = run / "00_INPUT" / "dataset_pointer.json"
        if pointer.is_file():
            try:
                dataset = json.loads(pointer.read_text(encoding="utf-8"))[
                    "dataset_root"
                ]
                protected.add(Path(str(dataset)).resolve())
            except (OSError, KeyError, json.JSONDecodeError):
                pass
    workspace = root / "workspace"
    if workspace.is_dir():
        for state_path in workspace.rglob("*.state.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not state.get("queue_id"):
                continue
            statuses = {
                str(row.get("status", ""))
                for row in state.get("entries", {}).values()
                if isinstance(row, dict)
            }
            if statuses and statuses.issubset(
                {"completed", "duplicate_skipped"}
            ):
                continue
            preparation_value = state.get("preflight_preparation")
            if not preparation_value:
                continue
            preparation = Path(str(preparation_value)).resolve()
            protected.add(preparation)
            pointer = preparation / "00_INPUT" / "dataset_pointer.json"
            if pointer.is_file():
                try:
                    dataset = json.loads(
                        pointer.read_text(encoding="utf-8")
                    )["dataset_root"]
                    protected.add(Path(str(dataset)).resolve())
                except (OSError, KeyError, json.JSONDecodeError):
                    pass
            try:
                relative = preparation.relative_to(root / "results")
            except ValueError:
                continue
            if (
                len(relative.parts) >= 2
                and relative.parts[0] in {"simulation", "real_vehicle"}
            ):
                protected.add(
                    root / "results" / relative.parts[0] / relative.parts[1]
                )
    return tuple(sorted(protected))


def _experiment_root(
    path: Path, results_root: Path, datasets_root: Path
) -> Path | None:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if not (
            (candidate / "PUBLISHED.json").is_file()
            or (candidate / "experiment.yaml").is_file()
        ):
            continue
        try:
            relative = candidate.relative_to(results_root.resolve())
            if relative.parts and relative.parts[0] in {
                "simulation",
                "real_vehicle",
            }:
                return candidate
        except ValueError:
            pass
        try:
            relative = candidate.relative_to(datasets_root.resolve())
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in {
            "simulation",
            "real_vehicle",
        }:
            return results_root / relative
    for candidate in (resolved, *resolved.parents):
        try:
            relative = candidate.relative_to(results_root.resolve())
        except ValueError:
            continue
        if (
            len(relative.parts) >= 2
            and relative.parts[0] in {"simulation", "real_vehicle"}
            and (candidate / "methods").is_dir()
        ):
            return candidate
    return None


def _published_experiments(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    experiments = {
        path.parent.resolve()
        for path in root.rglob("PUBLISHED.json")
        if not any(parent.is_symlink() for parent in path.parents)
    }
    return tuple(sorted(experiments))


def build_cleanup_plan(
    repository_root: Path, *, include_data_local: bool = False
) -> CleanupPlan:
    root = repository_root.resolve()
    results = root / "results"
    protected = protected_storage_paths(root)
    candidates: list[CleanupTarget] = []

    def add(path: Path, kind: str) -> None:
        if not (path.exists() or path.is_symlink()):
            return
        if any(
            _is_within(path, item) or _is_within(item, path)
            for item in protected
        ):
            return
        candidates.append(
            CleanupTarget(
                path.absolute(),
                kind,
                _experiment_root(path, results, root / "datasets"),
            )
        )

    datasets = root / "datasets"
    if datasets.is_dir():
        for child in sorted(datasets.iterdir()):
            if child.name not in {"simulation", "real_vehicle"}:
                add(child, "legacy generated dataset cache")
    for experiment in _published_experiments(datasets):
        inputs = experiment / "inputs"
        if not inputs.is_dir():
            continue
        for input_root in sorted(
            path for path in inputs.iterdir() if path.is_dir()
        ):
            add(
                input_root / "raw_images",
                "experiment raw static/moving frames",
            )
            for pattern in (
                "metadata/**/images",
                "metadata/**/selected_frames",
                "metadata/**/diagnostics",
            ):
                for images in input_root.glob(pattern):
                    add(images, "generated input preparation images")
            observations = input_root / "observations"
            if observations.is_dir():
                for debug in observations.glob("*/debug_images"):
                    add(debug, "ArUco debug images")
                for gallery in observations.glob("*/debug_gallery"):
                    add(gallery, "ArUco debug gallery")

    if results.is_dir():
        result_experiments = set(_published_experiments(results))
        for methods in results.rglob("methods"):
            if methods.is_dir() and not methods.is_symlink():
                result_experiments.add(methods.parent.resolve())
        for experiment in sorted(result_experiments):
            for input_root_name in ("datasets", "inputs"):
                inputs = experiment / input_root_name
                if not inputs.is_dir():
                    continue
                for raw in inputs.glob("*/raw_images"):
                    add(raw, "legacy experiment raw static/moving frames")
                for debug in inputs.glob(
                    "*/observations/*/debug_images"
                ):
                    add(debug, "ArUco debug images")
                for gallery in inputs.glob(
                    "*/observations/*/debug_gallery"
                ):
                    add(gallery, "ArUco debug gallery")
            methods = experiment / "methods"
            if methods.is_dir():
                for pattern in (
                    "**/01_colmap_dataset/images",
                    "**/01_moving_colmap/images",
                    "**/colmap/images",
                ):
                    for images in methods.glob(pattern):
                        add(images, "COLMAP working-image copies")

        profiles = results / "real_vehicle" / "_intrinsics"
        if profiles.is_dir():
            for pattern in (
                "**/selected_frames",
                "**/diagnostics",
                "**/.work",
            ):
                for directory in profiles.glob(pattern):
                    add(directory, "intrinsics selected/debug images")

    if include_data_local:
        local = root / "data_local"
        if local.is_dir():
            for child in sorted(local.iterdir()):
                add(child, "user data_local input")

    # Keep the smallest non-overlapping target set.
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
    combined = build_cleanup_plan(
        repository_root, include_data_local=True
    )
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
    """Delete exactly the reviewed plan and record affected experiments."""
    before_files = _files(plan.targets)
    digest = _target_digest(plan.targets)
    by_experiment: dict[Path, list[CleanupTarget]] = {}
    for target in plan.targets:
        if target.experiment_root is not None:
            by_experiment.setdefault(target.experiment_root, []).append(target)
    experiment_metrics: dict[Path, tuple[int, int, int]] = {}
    for experiment, targets in by_experiment.items():
        experiment_files = [
            path
            for path in before_files
            if any(_is_within(path, target.path) for target in targets)
        ]
        experiment_metrics[experiment] = _sizes(experiment_files)

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

    for experiment, targets in by_experiment.items():
        file_count, logical, reclaimable = experiment_metrics[experiment]
        _write_json(
            experiment / "INPUT_REMOVED.json",
            {
                "schema_version": 1,
                "removed_at": _now(),
                "rerunnable": False,
                "file_count": file_count,
                "logical_bytes": logical,
                "reclaimable_bytes_estimate": reclaimable,
                "deleted_kinds": sorted({target.kind for target in targets}),
                "deleted_paths": [str(target.path) for target in targets],
                "content_manifest_sha256": digest,
                "results_preserved": True,
            },
        )
    return {
        "removed_targets": removed_targets,
        "file_count": plan.file_count,
        "logical_bytes": plan.logical_bytes,
        "reclaimable_bytes_estimate": plan.reclaimable_bytes,
        "content_manifest_sha256": digest,
    }

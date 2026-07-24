from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .results import create_simulation_factor_views
from .storage_layout import (
    BUS_BASELINE,
    classify_simulation_parameters,
)


@dataclass(frozen=True)
class TreeInventory:
    file_count: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SimulationMigration:
    experiment_id: str
    source_result: Path
    target_result: Path
    target_dataset: Path
    source_cache: Path | None
    parameters: dict[str, Any]
    before: TreeInventory


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def inventory_tree(path: Path) -> TreeInventory:
    digest = hashlib.sha256()
    count = 0
    size = 0
    if not path.is_dir():
        return TreeInventory(0, 0, digest.hexdigest())
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path)
        file_hash = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                file_hash.update(chunk)
        stat = item.stat()
        digest.update(str(relative).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(file_hash.digest())
        count += 1
        size += stat.st_size
    return TreeInventory(count, size, digest.hexdigest())


def _combined_inventory(paths: list[Path]) -> TreeInventory:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for index, path in enumerate(paths):
        inventory = inventory_tree(path)
        digest.update(str(index).encode("ascii"))
        digest.update(inventory.sha256.encode("ascii"))
        count += inventory.file_count
        size += inventory.size_bytes
    return TreeInventory(count, size, digest.hexdigest())


def _content_counts(
    paths: list[Path], *, exclude_rewritten_metadata: bool = False
) -> Counter[tuple[int, str]]:
    counts: Counter[tuple[int, str]] = Counter()
    for root in paths:
        if not root.is_dir():
            continue
        for item in sorted(
            candidate for candidate in root.rglob("*") if candidate.is_file()
        ):
            if (
                exclude_rewritten_metadata
                and item.suffix.lower()
                in {".json", ".yaml", ".yml", ".txt"}
            ):
                continue
            file_hash = hashlib.sha256()
            with item.open("rb") as handle:
                for chunk in iter(
                    lambda: handle.read(8 * 1024 * 1024), b""
                ):
                    file_hash.update(chunk)
            counts[(item.stat().st_size, file_hash.hexdigest())] += 1
    return counts


def _legacy_payload(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    legacy = root / "legacy_manifest.json"
    experiment = root / "experiment.yaml"
    if legacy.is_file():
        payload = json.loads(legacy.read_text(encoding="utf-8"))
        return payload, dict(payload.get("parameters") or {})
    payload = yaml.safe_load(experiment.read_text(encoding="utf-8")) or {}
    return payload, dict(payload.get("simulation_parameters") or {})


def plan_simulation_migration(
    repository_root: Path,
) -> tuple[SimulationMigration, ...]:
    root = repository_root.resolve()
    simulation = root / "results" / "simulation"
    if not simulation.is_dir():
        return ()
    structured = {
        "baseline",
        "fov",
        "resolution",
        "lighting",
        "motion_blur",
        "density",
        "route",
        "capture",
        "mixed",
        "worlds",
        "_views",
    }
    migrations: list[SimulationMigration] = []
    for source in sorted(
        path
        for path in simulation.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and path.name not in structured
    ):
        if not (
            (source / "legacy_manifest.json").is_file()
            or (source / "experiment.yaml").is_file()
        ):
            continue
        payload, parameters = _legacy_payload(source)
        if not (source / "legacy_manifest.json").is_file():
            # Unpublished schema-v4 work is handled as a temporary queue, not
            # presented as a successful historical experiment.
            continue
        experiment_id = str(
            payload.get("experiment_id")
            or payload.get("id")
            or source.name
        )
        relative = classify_simulation_parameters(
            parameters,
            experiment_id=experiment_id,
        )
        target_result = simulation / relative
        target_dataset = root / "datasets" / "simulation" / relative
        cache = root / "datasets" / source.name
        sources = [source]
        if cache.is_dir():
            sources.append(cache)
        migrations.append(
            SimulationMigration(
                experiment_id=experiment_id,
                source_result=source,
                target_result=target_result,
                target_dataset=target_dataset,
                source_cache=cache if cache.is_dir() else None,
                parameters=parameters,
                before=_combined_inventory(sources),
            )
        )
    return tuple(migrations)


def _active_process(root: Path) -> bool:
    for manifest in root.rglob("run_manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            pid = int(payload.get("runner_pid") or 0)
        except Exception:
            continue
        if pid <= 0 or pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        return True
    return False


def _relative_after_named(path: Path, name: str) -> Path:
    parts = path.parts
    index = parts.index(name)
    return Path(*parts[index + 2 :])


def _publish_receipt(
    migration: SimulationMigration,
    inventory: TreeInventory,
) -> dict[str, Any]:
    return {
        "schema_version": 5,
        "status": "published",
        "publication_kind": "verified_legacy_migration",
        "published_at": _now(),
        "experiment_id": migration.experiment_id,
        "source_result": str(migration.source_result),
        "target_result": str(migration.target_result),
        "target_dataset": str(migration.target_dataset),
        "pre_migration_inventory": asdict(migration.before),
        "post_migration_inventory": asdict(inventory),
    }


def _image_files(directory: Path) -> list[Path]:
    extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _legacy_dataset_manifest(
    input_root: Path,
    *,
    experiment_id: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Describe one verified legacy input without changing its source files."""
    raw_images = input_root / "raw_images"
    static_root = raw_images / "static"
    moving_root = raw_images / "moving"
    camera_info = raw_images / "camera_info"
    static_images = _image_files(static_root)
    moving_images = _image_files(moving_root)
    static_ids = sorted({path.stem for path in static_images})
    moving_id = "moving_calib_camera"
    intrinsic_ids = (
        {path.stem for path in camera_info.glob("*.json")}
        if camera_info.is_dir()
        else set()
    )
    remaining = sorted(intrinsic_ids.difference(static_ids))
    if len(remaining) == 1:
        moving_id = remaining[0]

    def relative(path: Path) -> str:
        return str(path.relative_to(input_root))

    static_cameras = []
    for camera_id in static_ids:
        images = [path for path in static_images if path.stem == camera_id]
        intrinsic = camera_info / f"{camera_id}.json"
        static_cameras.append(
            {
                "id": camera_id,
                "kind": "static",
                "image_count": len(images),
                "images": [relative(path) for path in images],
                "intrinsics": (
                    relative(intrinsic) if intrinsic.is_file() else None
                ),
                "source_topic": None,
            }
        )
    moving_intrinsics = camera_info / f"{moving_id}.json"
    return {
        "schema_version": 1,
        "dataset_id": experiment_id,
        "scene_type": "simulation",
        "created_at": _now(),
        "prepared_root": str(input_root.resolve()),
        "static_cameras": static_cameras,
        "moving_camera": {
            "id": moving_id,
            "kind": "moving",
            "image_count": len(moving_images),
            "images": [relative(path) for path in moving_images],
            "intrinsics": (
                relative(moving_intrinsics)
                if moving_intrinsics.is_file()
                else None
            ),
            "source_topic": None,
        },
        "sampling_hz": None,
        "marker_dictionary": "DICT_4X4_50",
        "marker_length_m": 0.17,
        "simulation_parameters": parameters,
        "files": [],
        "automatic_selections": [],
        "notes": [
            "Generated during the verified schema-v5 legacy migration.",
            "Original scientific files remain byte-identical; the migration "
            "journal records their SHA-256 inventory.",
        ],
    }


def _write_legacy_input_manifests(
    dataset_root: Path,
    *,
    experiment_id: str,
    parameters: dict[str, Any],
) -> list[Path]:
    written: list[Path] = []
    inputs = dataset_root / "inputs"
    if not inputs.is_dir():
        return written
    for input_root in sorted(path for path in inputs.iterdir() if path.is_dir()):
        destination = input_root / "dataset_manifest.json"
        if destination.is_file():
            continue
        _write_json(
            destination,
            _legacy_dataset_manifest(
                input_root,
                experiment_id=experiment_id,
                parameters=parameters,
            ),
        )
        written.append(destination)
    return written


def apply_simulation_migration(
    repository_root: Path,
    migrations: tuple[SimulationMigration, ...] | None = None,
) -> Path:
    root = repository_root.resolve()
    planned = migrations or plan_simulation_migration(root)
    journal = (
        root
        / "workspace"
        / "migrations"
        / f"rigcal_v43_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    _write_json(
        journal,
        {
            "schema_version": 5,
            "status": "planned",
            "created_at": _now(),
            "moves": [
                {
                    **asdict(item),
                    "source_result": str(item.source_result),
                    "target_result": str(item.target_result),
                    "target_dataset": str(item.target_dataset),
                    "source_cache": (
                        str(item.source_cache)
                        if item.source_cache is not None
                        else None
                    ),
                }
                for item in planned
            ],
        },
    )
    completed: list[dict[str, Any]] = []
    for migration in planned:
        if _active_process(migration.source_result):
            raise RuntimeError(
                f"Active process protects migration source: "
                f"{migration.source_result}"
            )
        if migration.target_result.exists() or migration.target_dataset.exists():
            raise FileExistsError(
                "Migration target already exists: "
                f"{migration.target_result} or {migration.target_dataset}"
            )
        result_incoming = migration.target_result.with_name(
            f".incoming_{migration.target_result.name}_v43"
        )
        dataset_incoming = migration.target_dataset.with_name(
            f".incoming_{migration.target_dataset.name}_v43"
        )
        if result_incoming.exists() or dataset_incoming.exists():
            raise FileExistsError("A previous v4.3 migration is incomplete")
        result_incoming.parent.mkdir(parents=True, exist_ok=True)
        dataset_incoming.mkdir(parents=True, exist_ok=False)
        source_paths = [migration.source_result]
        if migration.source_cache is not None:
            source_paths.append(migration.source_cache)
        before_content = _content_counts(source_paths)
        migration.source_result.rename(result_incoming)
        try:
            inputs = result_incoming / "inputs"
            if inputs.is_dir():
                inputs.rename(dataset_incoming / "inputs")
            else:
                (dataset_incoming / "inputs").mkdir()
            observations = result_incoming / "observations"
            if observations.is_dir():
                for input_observations in sorted(observations.iterdir()):
                    destination = (
                        dataset_incoming
                        / "inputs"
                        / input_observations.name
                        / "observations"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise FileExistsError(
                            f"Observation destination exists: {destination}"
                        )
                    input_observations.rename(destination)
                observations.rmdir()
            if migration.source_cache is not None:
                cache_target = dataset_incoming / "_legacy_preparation_cache"
                migration.source_cache.rename(cache_target)
            input_ids = sorted(
                path.name
                for path in (dataset_incoming / "inputs").iterdir()
                if path.is_dir()
            )
            old_experiment = result_incoming / "experiment.yaml"
            if old_experiment.is_file():
                old_experiment.rename(
                    result_incoming / "legacy_experiment_schema4.yaml"
                )
            manifest = {
                "schema_version": 5,
                "id": migration.experiment_id,
                "category": "simulation",
                "source_kind": "prepared",
                "storage": {
                    "category": "simulation",
                    "relative": str(
                        _relative_after_named(
                            migration.target_result, "results"
                        )
                    ),
                },
                "input_ids": input_ids,
                "simulation_parameters": migration.parameters,
                "migration": "rigcal_v43_verified_move",
                "migrated_at": _now(),
            }
            (result_incoming / "experiment.yaml").write_text(
                yaml.safe_dump(
                    manifest, sort_keys=False, allow_unicode=True
                ),
                encoding="utf-8",
            )
            shutil.copy2(
                result_incoming / "experiment.yaml",
                dataset_incoming / "experiment.yaml",
            )
            _write_legacy_input_manifests(
                dataset_incoming,
                experiment_id=migration.experiment_id,
                parameters=migration.parameters,
            )
            after_content = _combined_inventory(
                [result_incoming, dataset_incoming]
            )
            after_counts = _content_counts(
                [result_incoming, dataset_incoming]
            )
            # New manifests increase count/bytes. Every original file must
            # still be present; exact SHA identities are recorded in the
            # pre-migration journal.
            if (
                after_content.file_count < migration.before.file_count
                or after_content.size_bytes < migration.before.size_bytes
            ):
                raise RuntimeError(
                    f"Migration verification lost files for "
                    f"{migration.experiment_id}"
                )
            missing_hashes = {
                key: count - after_counts[key]
                for key, count in before_content.items()
                if after_counts[key] < count
            }
            if missing_hashes:
                raise RuntimeError(
                    "Migration SHA-256 verification lost or changed "
                    f"{sum(missing_hashes.values())} file(s) for "
                    f"{migration.experiment_id}"
                )
            receipt = _publish_receipt(migration, after_content)
            _write_json(result_incoming / "PUBLISHED.json", receipt)
            _write_json(dataset_incoming / "PUBLISHED.json", receipt)
            dataset_incoming.rename(migration.target_dataset)
            try:
                result_incoming.rename(migration.target_result)
            except Exception:
                migration.target_dataset.rename(dataset_incoming)
                raise
            completed.append(receipt)
        except Exception:
            if result_incoming.exists() and not migration.source_result.exists():
                inputs = dataset_incoming / "inputs"
                if inputs.exists() and not (result_incoming / "inputs").exists():
                    inputs.rename(result_incoming / "inputs")
                cache = dataset_incoming / "_legacy_preparation_cache"
                if (
                    cache.exists()
                    and migration.source_cache is not None
                    and not migration.source_cache.exists()
                ):
                    migration.source_cache.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    cache.rename(migration.source_cache)
                result_incoming.rename(migration.source_result)
            if dataset_incoming.exists():
                shutil.rmtree(dataset_incoming)
            raise
    views = root / "results" / "simulation" / "_views"
    if views.is_dir():
        shutil.rmtree(views)
    baseline = next(
        (
            item
            for item in planned
            if item.target_result.relative_to(
                root / "results" / "simulation"
            )
            == Path("baseline/route2")
        ),
        None,
    )
    if baseline is not None:
        create_simulation_factor_views(
            root / "results",
            baseline.target_result,
            experiment_id=baseline.experiment_id,
            parameters=baseline.parameters,
            baseline=BUS_BASELINE,
        )
        create_simulation_factor_views(
            root / "datasets",
            baseline.target_dataset,
            experiment_id=baseline.experiment_id,
            parameters=baseline.parameters,
            baseline=BUS_BASELINE,
        )
    _write_json(
        journal,
        {
            "schema_version": 5,
            "status": "completed",
            "completed_at": _now(),
            "moves": completed,
        },
    )
    return journal


def backfill_migrated_dataset_manifests(repository_root: Path) -> Path | None:
    """Add missing input manifests to already verified schema-v5 migrations."""
    root = repository_root.resolve()
    datasets = root / "datasets" / "simulation"
    if not datasets.is_dir():
        return None
    changes: list[dict[str, Any]] = []
    for experiment_manifest in sorted(datasets.rglob("experiment.yaml")):
        experiment_root = experiment_manifest.parent
        if experiment_root.is_symlink():
            continue
        try:
            payload = yaml.safe_load(
                experiment_manifest.read_text(encoding="utf-8")
            ) or {}
        except (OSError, yaml.YAMLError):
            continue
        if payload.get("migration") != "rigcal_v43_verified_move":
            continue
        before = inventory_tree(experiment_root)
        written = _write_legacy_input_manifests(
            experiment_root,
            experiment_id=str(payload.get("id") or experiment_root.name),
            parameters=dict(payload.get("simulation_parameters") or {}),
        )
        if not written:
            continue
        after = inventory_tree(experiment_root)
        changes.append(
            {
                "experiment": str(experiment_root),
                "created": [str(path) for path in written],
                "before": asdict(before),
                "after": asdict(after),
            }
        )
    if not changes:
        return None
    journal = (
        root
        / "workspace"
        / "migrations"
        / (
            "rigcal_v43_dataset_manifest_backfill_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    )
    _write_json(
        journal,
        {
            "schema_version": 5,
            "status": "completed",
            "completed_at": _now(),
            "operation": "additive_dataset_manifest_backfill",
            "changes": changes,
        },
    )
    return journal


def migrate_unpublished_flat_experiments(
    repository_root: Path,
) -> tuple[Path, ...]:
    """Relocate pre-v5 unpublished experiment trees into temporary queues."""
    root = repository_root.resolve()
    temporary_root = root / "workspace" / "temporary_runs"
    moved: list[Path] = []
    structured = {
        "baseline",
        "fov",
        "resolution",
        "lighting",
        "motion_blur",
        "density",
        "route",
        "capture",
        "mixed",
        "worlds",
        "_views",
        "_intrinsics",
    }
    for category in ("simulation", "real_vehicle"):
        category_root = root / "results" / category
        if not category_root.is_dir():
            continue
        for source in sorted(
            path
            for path in category_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and path.name not in structured
        ):
            if (
                (source / "PUBLISHED.json").is_file()
                or (source / "legacy_manifest.json").is_file()
            ):
                continue
            cache = root / "datasets" / source.name
            if not any(source.iterdir()) and not cache.is_dir():
                source.rmdir()
                continue
            if _active_process(source):
                raise RuntimeError(
                    f"Active process protects unpublished experiment: {source}"
                )
            queue_id = (
                f"legacy_unpublished_{category}_{source.name}"
            )
            destination = temporary_root / queue_id
            suffix = 2
            while destination.exists():
                destination = temporary_root / f"{queue_id}_{suffix}"
                suffix += 1
            source_paths = [source]
            if cache.is_dir():
                source_paths.append(cache)
            before = _combined_inventory(source_paths)
            before_counts = _content_counts(
                source_paths, exclude_rewritten_metadata=True
            )
            destination.mkdir(parents=True, exist_ok=False)
            result_destination = destination / "legacy_experiment"
            source.rename(result_destination)
            cache_destination: Path | None = None
            if cache.is_dir():
                cache_destination = destination / "legacy_dataset_cache"
                cache.rename(cache_destination)
            merged_transactions = destination / "legacy_staging_transactions"
            for existing in sorted(
                path
                for path in temporary_root.iterdir()
                if path.is_dir() and path != destination
            ):
                payload_path = existing / "queue_transaction.json"
                try:
                    payload = json.loads(
                        payload_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue
                migrated_from = str(payload.get("migrated_from", ""))
                if not migrated_from.startswith(str(source / ".staging")):
                    continue
                merged_transactions.mkdir(parents=True, exist_ok=True)
                existing.rename(merged_transactions / existing.name)
            replacements = {
                str(source): str(result_destination),
            }
            if cache_destination is not None:
                replacements[str(cache)] = str(cache_destination)
            for path in destination.rglob("*"):
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.suffix.lower()
                    not in {".json", ".yaml", ".yml", ".txt"}
                ):
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                updated = text
                for old, new in replacements.items():
                    updated = updated.replace(old, new)
                if updated != text:
                    path.write_text(updated, encoding="utf-8")
            after = _combined_inventory([destination])
            # Metadata paths may be rewritten for resumability, so verify all
            # non-text scientific/input payload hashes plus total file count.
            after_counts = _content_counts(
                [destination], exclude_rewritten_metadata=True
            )
            if after.file_count < before.file_count:
                raise RuntimeError(
                    f"Unpublished migration lost files: {source}"
                )
            missing_binary = {
                key: count - after_counts[key]
                for key, count in before_counts.items()
                if after_counts[key] < count
            }
            if missing_binary:
                raise RuntimeError(
                    "Unpublished migration changed or lost "
                    f"{sum(missing_binary.values())} binary payload file(s): "
                    f"{source}"
                )
            transaction = {
                "schema_version": 5,
                "queue_id": destination.name,
                "status": "incomplete",
                "legacy_unpublished_experiment": True,
                "migrated_at": _now(),
                "migrated_from": str(source),
                "legacy_cache": (
                    str(cache_destination)
                    if cache_destination is not None
                    else None
                ),
                "pre_migration_inventory": asdict(before),
                "post_migration_inventory": asdict(after),
                "content_hash_entries_before": sum(
                    before_counts.values()
                ),
                "content_hash_entries_after": sum(after_counts.values()),
            }
            _write_json(
                destination / "queue_transaction.json", transaction
            )
            moved.append(destination)
    datasets_root = root / "datasets"
    if datasets_root.is_dir():
        for cache in sorted(
            path
            for path in datasets_root.iterdir()
            if path.is_dir()
            and path.name not in {"simulation", "real_vehicle"}
        ):
            destination = (
                temporary_root / f"legacy_orphan_cache_{cache.name}"
            )
            suffix = 2
            while destination.exists():
                destination = (
                    temporary_root
                    / f"legacy_orphan_cache_{cache.name}_{suffix}"
                )
                suffix += 1
            before = inventory_tree(cache)
            destination.mkdir(parents=True, exist_ok=False)
            cache.rename(destination / "legacy_dataset_cache")
            after = inventory_tree(destination / "legacy_dataset_cache")
            if before != after:
                raise RuntimeError(
                    f"Orphan cache verification failed: {cache}"
                )
            _write_json(
                destination / "queue_transaction.json",
                {
                    "schema_version": 5,
                    "queue_id": destination.name,
                    "status": "incomplete",
                    "legacy_orphan_cache": True,
                    "migrated_at": _now(),
                    "migrated_from": str(cache),
                    "inventory": asdict(before),
                },
            )
            moved.append(destination)
    return tuple(moved)

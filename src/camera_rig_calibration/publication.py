from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config, save_config
from .experiments import experiment_paths, write_experiment_manifest
from .results import create_simulation_factor_views


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_tree(source: Path, destination: Path) -> dict[str, int]:
    counts = {"hardlinked": 0, "copied": 0, "existing": 0}
    if not source.is_dir():
        return counts
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_symlink():
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if (
                target.stat().st_size != item.stat().st_size
                or _sha256(target) != _sha256(item)
            ):
                raise RuntimeError(
                    f"Publication conflicts with an existing file: {target}"
                )
            counts["existing"] += 1
            continue
        try:
            os.link(item, target)
            counts["hardlinked"] += 1
        except OSError:
            shutil.copy2(item, target)
            counts["copied"] += 1
    return counts


def _snapshot_tree(source: Path, destination: Path) -> None:
    """Create a cheap independent directory snapshot for an atomic swap."""
    if not source.is_dir():
        destination.mkdir(parents=True, exist_ok=False)
        return

    def link_or_copy(first: str, second: str) -> str:
        try:
            os.link(first, second)
            return second
        except OSError:
            return shutil.copy2(first, second)

    shutil.copytree(
        source,
        destination,
        symlinks=True,
        copy_function=link_or_copy,
    )


def _recover_swap(target: Path, queue_id: str) -> None:
    backup = target.with_name(f".previous_{target.name}_{queue_id}")
    if target.exists() and backup.exists():
        shutil.rmtree(backup)
    elif not target.exists() and backup.exists():
        backup.rename(target)


def _atomic_swap(incoming: Path, target: Path, queue_id: str) -> None:
    backup = target.with_name(f".previous_{target.name}_{queue_id}")
    _recover_swap(target, queue_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.rename(backup)
    try:
        incoming.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _archive_current(current: Path) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = current.parent / "run_history" / stamp
    suffix = 2
    while history.exists():
        history = current.parent / "run_history" / f"{stamp}_{suffix}"
        suffix += 1
    history.parent.mkdir(parents=True, exist_ok=True)
    current.rename(history)


def _rewrite_execution(
    incoming: Path,
    *,
    canonical_dataset_input: Path,
    canonical_experiment: Path,
    target: Path,
    queue_id: str,
    publication_source: Path,
) -> None:
    raw_link = incoming / "00_INPUT" / "raw_images"
    if raw_link.is_symlink() or raw_link.exists():
        if raw_link.is_dir() and not raw_link.is_symlink():
            shutil.rmtree(raw_link)
        else:
            raw_link.unlink()
    raw_link.symlink_to(
        (canonical_dataset_input / "raw_images").resolve(),
        target_is_directory=True,
    )
    manifest_path = incoming / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observation_id = str(manifest.get("observation_id", ""))
    observations = (
        canonical_dataset_input / "observations" / observation_id
    )
    observation_link = incoming / "01_OBSERVATIONS"
    if observation_link.is_symlink() or observation_link.exists():
        if observation_link.is_dir() and not observation_link.is_symlink():
            shutil.rmtree(observation_link)
        else:
            observation_link.unlink()
    if observation_id:
        observation_link.symlink_to(
            observations.resolve(), target_is_directory=True
        )
    pointer = incoming / "00_INPUT" / "dataset_pointer.json"
    if pointer.is_file():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        payload["dataset_root"] = str(canonical_dataset_input.resolve())
        _write_json(pointer, payload)
    for name in ("requested_config.yaml", "resolved_config.yaml"):
        path = incoming / name
        if not path.is_file():
            continue
        config = load_config(path)
        moving_intrinsics = (
            canonical_dataset_input
            / "raw_images"
            / "camera_info"
            / f"{config.moving_camera.id}.json"
        )
        config = config.model_copy(
            update={
                "dataset": config.dataset.model_copy(
                    update={"prepared_root": canonical_dataset_input}
                ),
                "moving_camera": config.moving_camera.model_copy(
                    update={
                        "intrinsics": (
                            moving_intrinsics
                            if moving_intrinsics.is_file()
                            else config.moving_camera.intrinsics
                        )
                    }
                ),
            },
            deep=True,
        )
        save_config(config, path)
    manifest.update(
        {
            "schema_version": 5,
            "experiment_root": str(canonical_experiment.resolve()),
            "published_result": str(target.resolve()),
            "observations_root": str(observations.resolve()),
            "status": "completed",
            "published_at": _now(),
            "publication_source": str(publication_source.resolve()),
            "publication_queue_id": queue_id,
        }
    )
    manifest.pop("transaction_root", None)
    _write_json(manifest_path, manifest)


def publish_queue_transaction(
    transaction_root: Path,
    *,
    queue_id: str,
    configs: list,
    results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Publish a fully terminated queue; leave every byte temporary on failure."""
    transaction = transaction_root.resolve()
    journal = transaction / "queue_transaction.json"
    publishable = {"completed", "duplicate_skipped"}
    non_terminal = {
        entry_id: row.get("status")
        for entry_id, row in results.items()
        if row.get("status") not in publishable
    }
    if non_terminal or len(results) != len(configs):
        _write_json(
            journal,
            {
                "schema_version": 5,
                "queue_id": queue_id,
                "status": "incomplete",
                "updated_at": _now(),
                "non_terminal_entries": non_terminal,
            },
        )
        return results
    first = configs[0]
    canonical = experiment_paths(first)
    published_receipt = canonical.root / "PUBLISHED.json"
    if published_receipt.is_file():
        try:
            previous = json.loads(
                published_receipt.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("queue_id") == queue_id:
            for entry_id, row in previous.get("entries", {}).items():
                if entry_id in results and isinstance(row, dict):
                    results[entry_id].update(row)
            _write_json(journal, previous)
            return results
    input_ids = {
        str(
            json.loads(
                (Path(str(row["result"])) / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            ).get("input_id")
        )
        for row in results.values()
        if row.get("status") == "completed"
    }
    input_ids.discard("None")
    _write_json(
        journal,
        {
            "schema_version": 5,
            "queue_id": queue_id,
            "status": "publishing",
            "started_at": _now(),
            "canonical_result_root": str(canonical.root),
            "canonical_dataset_root": str(canonical.dataset_root),
            "input_ids": sorted(input_ids),
        },
    )
    result_incoming = canonical.root.with_name(
        f".incoming_{canonical.root.name}_{queue_id}"
    )
    dataset_incoming = canonical.dataset_root.with_name(
        f".incoming_{canonical.dataset_root.name}_{queue_id}"
    )
    try:
        _recover_swap(canonical.root, queue_id)
        _recover_swap(canonical.dataset_root, queue_id)
        for incoming in (result_incoming, dataset_incoming):
            if incoming.exists():
                shutil.rmtree(incoming)
        _snapshot_tree(canonical.root, result_incoming)
        _snapshot_tree(canonical.dataset_root, dataset_incoming)
        (result_incoming / "PUBLISHED.json").unlink(missing_ok=True)
        (dataset_incoming / "PUBLISHED.json").unlink(missing_ok=True)
        for input_id in sorted(input_ids):
            source = transaction / "dataset" / "inputs" / input_id
            destination = dataset_incoming / "inputs" / input_id
            _materialize_tree(source, destination)
        for entry_id, row in results.items():
            if row.get("status") == "duplicate_skipped":
                continue
            source = Path(str(row["result"]))
            manifest = json.loads(
                (source / "run_manifest.json").read_text(encoding="utf-8")
            )
            canonical_target = Path(
                str(manifest["intended_result_target"])
            ).resolve()
            try:
                relative_target = canonical_target.relative_to(
                    canonical.root.resolve()
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Method publication target is outside its canonical "
                    f"experiment: {canonical_target}"
                ) from exc
            target = result_incoming / relative_target
            input_id = str(manifest["input_id"])
            execution_incoming = target.with_name(
                f".incoming_execution_{queue_id}_{time.time_ns()}"
            )
            shutil.copytree(source, execution_incoming, symlinks=True)
            _rewrite_execution(
                execution_incoming,
                canonical_dataset_input=canonical.datasets / input_id,
                canonical_experiment=canonical.root,
                target=canonical_target,
                queue_id=queue_id,
                publication_source=source,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                _archive_current(target)
            execution_incoming.rename(target)
            row["result"] = str(canonical_target)
            row["status"] = "completed"
        for name in ("evaluations", "comparisons"):
            _materialize_tree(
                transaction / "results" / name,
                result_incoming / name,
            )
        _materialize_tree(
            transaction / "artifacts",
            result_incoming / "artifacts",
        )
        incoming_paths = replace(
            canonical,
            root=result_incoming,
            dataset_root=dataset_incoming,
            datasets=dataset_incoming / "inputs",
            methods=result_incoming / "methods",
            evaluations=result_incoming / "evaluations",
            comparisons=result_incoming / "comparisons",
            artifacts=result_incoming / "artifacts",
        )
        for input_id in sorted(input_ids):
            write_experiment_manifest(first, incoming_paths, input_id)
        if (result_incoming / "experiment.yaml").is_file():
            shutil.copy2(
                result_incoming / "experiment.yaml",
                dataset_incoming / "experiment.yaml",
            )
        receipt = {
            "schema_version": 5,
            "queue_id": queue_id,
            "status": "published",
            "published_at": _now(),
            "input_ids": sorted(input_ids),
            "entries": {
                key: {
                    "status": value.get("status"),
                    "result": value.get("result"),
                }
                for key, value in results.items()
            },
        }
        _write_json(result_incoming / "PUBLISHED.json", receipt)
        _write_json(dataset_incoming / "PUBLISHED.json", receipt)
        _atomic_swap(dataset_incoming, canonical.dataset_root, queue_id)
        _atomic_swap(result_incoming, canonical.root, queue_id)
        alias_errors: list[str] = []
        if canonical.category == "simulation":
            parameters = {
                "route": first.simulation.route_name,
                "moving_width": first.simulation.moving_width,
                "moving_height": first.simulation.moving_height,
                "moving_hfov_deg": first.simulation.moving_hfov_deg,
                "lighting": first.simulation.lighting,
                "lighting_scale": first.simulation.lighting_scale,
                "motion_blur_kernel": first.simulation.motion_blur_kernel,
                "motion_blur_angle_deg": (
                    first.simulation.motion_blur_angle_deg
                ),
                "target_route_frames": (
                    first.simulation.target_route_frames
                ),
                "route_sampling_strategy": (
                    first.simulation.route_sampling_strategy
                ),
                "settle_seconds": first.simulation.settle_seconds,
                "post_pose_skip": first.simulation.post_pose_skip,
                "frame_timeout_seconds": (
                    first.simulation.frame_timeout_seconds
                ),
                "startup_timeout_seconds": (
                    first.simulation.startup_timeout_seconds
                ),
            }
            for output_root, experiment_root in (
                (first.project.output_root, canonical.root),
                (
                    first.project.dataset_cache_root,
                    canonical.dataset_root,
                ),
            ):
                try:
                    create_simulation_factor_views(
                        output_root,
                        experiment_root,
                        experiment_id=canonical.experiment_id,
                        parameters=parameters,
                        baseline=first.simulation.world_baseline,
                        world_id=first.simulation.world_id,
                    )
                except RuntimeError as exc:
                    alias_errors.append(str(exc))
        if alias_errors:
            receipt["alias_errors"] = alias_errors
            _write_json(canonical.root / "PUBLISHED.json", receipt)
            _write_json(
                canonical.dataset_root / "PUBLISHED.json", receipt
            )
        _write_json(journal, receipt)
    except Exception as exc:
        _write_json(
            journal,
            {
                "schema_version": 5,
                "queue_id": queue_id,
                "status": "publication_failed",
                "updated_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    return results

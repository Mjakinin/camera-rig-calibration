"""Focused atomic-publication responsibility."""

from __future__ import annotations

import hashlib
import csv
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import load_config
from ..anchor_export import export_method_anchor_poses
from ..dataset.discovery import safe_id
from ..dataset_identity import (
    build_dataset_identity,
    identities_match,
    write_dataset_identity,
)
from ..evaluation.reporting import write_scientific_experiment_reports
from ..experiments import (
    experiment_manifest_payload,
    experiment_paths,
    method_result_label,
)
from ..filesystem import rename_with_retry
from ..storage_layout import storage_manifest


from .core import (
    _now,
    _read_json,
    _refresh_dataset_descriptor,
    _write_json,
)
from .dataset import (
    _publish_dataset,
)
from .evaluation import publish_evaluation_tree
from .inventory import (
    _native_calibration_hashes,
    _write_inventory_reports,
    write_experiment_reports,
)
from .method import (
    _publish_failure,
    _publish_success,
)

def reconcile_existing_experiment(
    root: Path,
    *,
    dataset_root: Path,
    category: str,
) -> dict[str, Any]:
    """Rebuild a layout-v2 front door without rerunning calibration methods."""
    from ..evaluation.reporting import (
        complete_existing_dataset,
        run_real_marker_consistency,
    )
    from ..evaluation.simulation_ground_truth import (
        resolve_simulation_ground_truth,
    )

    native_before = _native_calibration_hashes(root)
    complete_existing_dataset(dataset_root, root)
    ground_truth_regenerated = False
    if category == "simulation":
        ground_truth_regenerated = resolve_simulation_ground_truth(
            dataset_root, backfilled=True
        ).regenerated
    if category == "real_vehicle":
        run_real_marker_consistency(root, dataset_root, force=False)
    payload = write_scientific_experiment_reports(
        root,
        dataset_root=dataset_root,
        category=category,
    )
    previous = _read_json(root / "SUMMARY.json")
    sampling_rate = str(previous.get("sampling_rate", "native_rate"))
    _write_inventory_reports(
        root,
        dataset_root=dataset_root,
        category=category,
        experiment=str(previous.get("experiment") or root.name),
        sampling_rate=sampling_rate,
        queue_id=str(previous.get("queue_id") or "reconciled"),
        queue_complete=True,
    )
    native_after = _native_calibration_hashes(root)
    if native_after != native_before:
        changed = sorted(
            key
            for key in set(native_before) | set(native_after)
            if native_before.get(key) != native_after.get(key)
        )
        raise RuntimeError(
            "Reconcile modified native calibration artifacts, which is "
            "forbidden: " + ", ".join(changed)
        )
    payload["reconcile"] = {
        "ground_truth_regenerated": ground_truth_regenerated,
        "method_rerun": False,
        "colmap_rerun": False,
        "native_artifacts_unchanged": True,
        "native_artifact_count": len(native_before),
    }
    return payload


def publish_preparation_transaction(
    transaction_root: Path,
    *,
    queue_id: str,
    config: Any,
    preparation: Path,
) -> Path:
    """Publish a complete layout-v2 dataset without a calibration result."""
    del preparation
    source = transaction_root.resolve() / "dataset"
    # A detector retry changes the experiment ID and observation contract
    # after the normalized capture was prepared. Method jobs reuse this
    # transaction dataset, so its descriptor must carry the final identity as
    # well as the canonical copy. Otherwise method startup sees the stale
    # baseline fingerprint and rejects its own prepared input.
    _refresh_dataset_descriptor(source, config)
    canonical = experiment_paths(config).dataset_root
    published = _publish_dataset(source, canonical, config=config)
    preparation_record = published / "metadata" / "preparation.json"
    if not preparation_record.is_file():
        _write_json(
            preparation_record,
            {
                "schema_version": 5,
                "layout_version": 2,
                "queue_id": queue_id,
                "status": "prepared",
                "published_at": _now(),
            },
        )
    return published


def publish_queue_transaction(
    transaction_root: Path,
    *,
    queue_id: str,
    configs: list[Any],
    results: dict[str, dict[str, Any]],
    finalize: bool = True,
) -> dict[str, dict[str, Any]]:
    """Publish one immutable dataset and independent layout-v2 method outcomes."""
    transaction = transaction_root.resolve()
    terminal = {
        "completed",
        "duplicate_skipped",
        "failed",
        "failed_preflight",
        "skipped_dependency",
        "published",
        "failed_published",
    }
    if finalize and (
        len(results) != len(configs)
        or any(str(row.get("status")) not in terminal for row in results.values())
    ):
        return results
    if not configs:
        return results

    first = configs[0]
    paths = experiment_paths(first)
    # This intentionally happens before any method result is made visible.
    _publish_dataset(
        transaction / "dataset",
        paths.dataset_root,
        config=first,
    )
    paths.root.mkdir(parents=True, exist_ok=True)

    config_by_label = {
        safe_id(config.project.run_label): config for config in configs
    }
    for entry_id, row in results.items():
        status = str(row.get("status", "unknown"))
        if status in {"duplicate_skipped", "published", "failed_published"} or (
            status == "completed" and row.get("published") is True
        ):
            continue
        source_text = str(row.get("result", "")).strip()
        source = (
            Path(source_text)
            if source_text
            else transaction / "__missing_method_result__"
        )
        config = config_by_label.get(safe_id(entry_id), first)
        row_config = Path(str(row.get("config", "")))
        if row_config.is_file():
            config = load_config(row_config)
        if source.is_dir() and (source / "resolved_config.yaml").is_file():
            config = load_config(source / "resolved_config.yaml")
        if status == "completed":
            target, outcome = _publish_success(
                source,
                config=config,
                canonical_root=paths.root,
                queue_id=queue_id,
            )
            row.update(
                {
                    "status": "completed" if outcome == "completed" else outcome,
                    "result": str(target.resolve()),
                    "published": True,
                }
            )
        else:
            target, failure = _publish_failure(
                source,
                config=config,
                canonical_root=paths.root,
                entry_id=entry_id,
                error=(
                    f"{status}: "
                    f"{row.get('error') or row.get('errors') or status}"
                ),
            )
            row.update(
                {
                    "status": "failed_published",
                    "result": str(target.resolve()),
                    "attempt": str(target.resolve()),
                    "failure": failure,
                }
            )

    publish_evaluation_tree(
        transaction / "results" / "evaluations", paths.root / "evaluations"
    )
    write_experiment_reports(
        paths.root,
        config=first,
        queue_id=queue_id,
        queue_complete=finalize,
    )
    _write_json(
        transaction / "queue_transaction.json",
        {
            "schema_version": 5,
            "layout_version": 2,
            "queue_id": queue_id,
            "status": "published" if finalize else "publishing_queue",
            "result_root": str(paths.root.resolve()),
            "dataset_root": str(paths.dataset_root.resolve()),
            "updated_at": _now(),
        },
    )
    return results

__all__ = [
    'reconcile_existing_experiment',
    'publish_preparation_transaction',
    'publish_queue_transaction',
]

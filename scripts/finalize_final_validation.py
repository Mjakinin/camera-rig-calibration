#!/usr/bin/env python3
"""Finalize a completed final-validation batch after evaluation publication conflict.

The expensive calibration methods may already be published successfully while the
queue fails only because ``evaluations/`` was treated as immutable. This helper
promotes the already-computed transaction evaluation files atomically and then
rebuilds only derived experiment reports.

It deliberately does not invoke AP01, AP02, AP03, COLMAP, ArUco detection, input
preparation, or bundle adjustment. ``reconcile_existing_experiment`` independently
hashes native calibration artifacts before/after reconciliation and raises if any
native method artifact changes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

from camera_rig_calibration.product_policy import install_product_policy
from camera_rig_calibration.reporting_authority_policy import (
    install_reporting_authority_policy,
)


# Reconciliation must use the same final publication/reporting policy as rigcal:
# derived evaluations are refreshable, the published common anchor is authoritative
# for final reporting/AP03 derived exports, and native calibration artifacts remain
# immutable.
install_product_policy()
install_reporting_authority_policy()

from camera_rig_calibration.publication import (  # noqa: E402
    reconcile_existing_experiment,
)


EXPERIMENTS = (
    "main_route2_reference",
    "route2",
)


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_refreshable_tree(source: Path, destination: Path) -> dict[str, int]:
    """Atomically refresh derived files while preserving unrelated files."""
    if not source.is_dir():
        raise FileNotFoundError(f"Derived evaluation source is missing: {source}")
    counts = {"new": 0, "replaced": 0, "unchanged": 0}
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_symlink():
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == item.stat().st_size:
            if sha256(target) == sha256(item):
                counts["unchanged"] += 1
                continue
        existed = target.exists()
        incoming = target.with_name(
            f".incoming_{target.name}_{os.getpid()}_{time.time_ns()}"
        )
        try:
            shutil.copy2(item, incoming)
            os.replace(incoming, target)
        finally:
            incoming.unlink(missing_ok=True)
        counts["replaced" if existed else "new"] += 1
    return counts


def finalize_experiment(repository: Path, experiment: str, stamp: str) -> None:
    transaction = (
        repository
        / "workspace"
        / "temporary_runs"
        / f"final_validation_{experiment}_{stamp}"
    )
    source_evaluations = transaction / "results" / "evaluations"
    canonical = (
        repository
        / "results"
        / "simulation"
        / "baseline"
        / experiment
    )
    if not canonical.is_dir():
        raise FileNotFoundError(f"Canonical experiment is missing: {canonical}")
    if not (canonical / "methods").is_dir():
        raise RuntimeError(
            f"Canonical experiment has no published methods: {canonical}"
        )

    print(f"[FINALIZE] {experiment}")
    print(f"  transaction: {transaction}")
    print(f"  evaluation source: {source_evaluations}")
    print(f"  canonical experiment: {canonical}")

    counts = promote_refreshable_tree(
        source_evaluations,
        canonical / "evaluations",
    )
    print(
        "  derived evaluation promotion: "
        f"new={counts['new']}, replaced={counts['replaced']}, "
        f"unchanged={counts['unchanged']}"
    )

    payload = reconcile_existing_experiment(
        canonical,
        dataset_root=canonical,
        category="simulation",
    )
    reconcile = payload.get("reconcile", {})
    if reconcile.get("native_artifacts_unchanged") is not True:
        raise RuntimeError(
            "Reconcile did not certify native calibration artifacts as unchanged"
        )
    print(
        "  [OK] reconciliation complete; "
        f"native artifacts unchanged ({reconcile.get('native_artifact_count', 0)} files), "
        f"method rerun={reconcile.get('method_rerun')}, "
        f"COLMAP rerun={reconcile.get('colmap_rerun')}"
    )
    print(f"  results: {canonical / 'RESULTS.txt'}")
    print(f"  comparison: {canonical / 'COMPARISON.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize already-computed final-validation evaluations without "
            "rerunning calibration methods."
        )
    )
    parser.add_argument(
        "--stamp",
        required=True,
        help="Validation batch timestamp, e.g. 20260808_003644.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=EXPERIMENTS,
        help="Finalize one experiment only; repeat as needed. Default: both.",
    )
    args = parser.parse_args()

    repository = repository_root()
    selected = args.experiment or list(EXPERIMENTS)
    for experiment in selected:
        finalize_experiment(repository, experiment, args.stamp)

    print("[OK] Final validation publication repaired without method reruns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

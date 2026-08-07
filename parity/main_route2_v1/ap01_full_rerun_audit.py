"""Record the controlled AP01 production-rerun preflight and safe rejection.

This audit is intentionally read-only with respect to the experiment.  It may
write only to parity/main_route2_v1/ap01/full_rerun.  No method, evaluator,
publisher, reconciler, detector, solver, or COLMAP process is invoked here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from camera_rig_calibration import __file__ as package_file
from camera_rig_calibration.config import config_fingerprint
from camera_rig_calibration.dataset_identity import build_dataset_identity
from camera_rig_calibration.experiments import method_fingerprint
from camera_rig_calibration.methods.ap01.contracts import (
    resolve_ap01_method_contract,
)
from camera_rig_calibration.observations import resolve_selections
from camera_rig_calibration.rerun import (
    _frozen_observation_contract,
    _method_config_path,
    _resolved_rerun_config,
)


CONTRACT = "main_route2_parity_v1"
EXPECTED_ACCEPTED_OBSERVATIONS = 554
INTENDED_COMMAND = (
    "rigcal rerun-method --experiment "
    "results/simulation/baseline/route2_cpu_ref14_50x50 "
    "--method ap01 --variant baseline --reuse-prepared-input "
    "--reuse-matching-intermediates --reconcile-after "
    "--ap01-method-contract main_route2_parity_v1"
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    fields = (
        "status",
        "classification",
        "camera_id",
        "field",
        "production_value",
        "locked_value",
        "reason",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return _sha256(path)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.longpaths=true", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    return sorted(path for path in root.rglob("*") if path.is_file())


def _tree_snapshot(root: Path, *, relative_to: Path) -> dict[str, Any]:
    entries = [
        {
            "path": path.relative_to(relative_to).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _files(root)
    ]
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "root": root.relative_to(relative_to).as_posix(),
        "present": root.is_dir(),
        "file_count": len(entries),
        "tree_sha256": digest,
        "files": entries,
    }


def _selected_files_snapshot(
    roots: Iterable[Path], *, relative_to: Path
) -> dict[str, Any]:
    unique = sorted(
        {
            path.resolve()
            for root in roots
            for path in (
                [root] if root.is_file() else list(_files(root))
            )
            if path.is_file()
        }
    )
    entries = [
        {
            "path": path.relative_to(relative_to.resolve()).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in unique
    ]
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "file_count": len(entries),
        "tree_sha256": digest,
        "files": entries,
    }


def _observation_counts(experiment: Path) -> dict[str, int]:
    root = experiment / "observations"

    def rows(name: str) -> int:
        with (root / name).open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))

    return {
        "static": rows("shared_static_aruco_observations.csv"),
        "moving": rows("shared_moving_aruco_observations.csv"),
        "accepted_total": rows("shared_all_aruco_observations.csv"),
    }


def _historical_input_comparison(
    repository: Path, experiment: Path
) -> dict[str, Any]:
    inventory_path = repository / "parity/main_route2_v1/INPUT_FILE_HASHES.csv"
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        historical = [
            row
            for row in csv.DictReader(handle)
            if row["dataset_side"] == "main_historical"
        ]
    comparisons: list[dict[str, Any]] = []
    for row in historical:
        relative = str(row["path"])
        if row["category"] == "route_metadata":
            current = experiment / "metadata/simulation/route_commanded.csv"
        else:
            current = experiment / "raw_images" / relative
        current_hash = _sha256(current) if current.is_file() else None
        comparisons.append(
            {
                "category": row["category"],
                "historical_path": relative,
                "current_path": (
                    current.relative_to(experiment).as_posix()
                    if current.is_relative_to(experiment)
                    else str(current)
                ),
                "historical_sha256": row["sha256"],
                "current_sha256": current_hash,
                "equal": current_hash == row["sha256"],
            }
        )
    image_rows = [
        row
        for row in comparisons
        if row["category"] in {"raw_static_image", "raw_moving_image"}
    ]
    return {
        "reference": "origin/main@8f9dcea1e8b3189b3c195db2cafe65d5b0e5756b Route-2 inventory",
        "compared_file_count": len(comparisons),
        "exact_file_count": sum(bool(row["equal"]) for row in comparisons),
        "mismatch_count": sum(not bool(row["equal"]) for row in comparisons),
        "compared_image_count": len(image_rows),
        "equal_image_count": sum(bool(row["equal"]) for row in image_rows),
        "image_mismatch_count": sum(not bool(row["equal"]) for row in image_rows),
        "mismatches": [row for row in comparisons if not row["equal"]],
        "exact_match": all(bool(row["equal"]) for row in comparisons),
    }


def record_pre_run_rejection(repository: Path, experiment: Path) -> dict[str, Any]:
    repository = repository.resolve()
    experiment = experiment.resolve()
    evidence = repository / "parity/main_route2_v1/ap01/full_rerun"
    lock = json.loads(
        (repository / "parity/main_route2_v1/PARITY_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    original, config = _resolved_rerun_config(
        repository,
        experiment,
        "ap01",
        "baseline",
        ap01_method_contract=CONTRACT,
    )
    selections = resolve_selections(config, experiment / "observations")
    contract = resolve_ap01_method_contract(
        config.methods.ap01.method_contract,
        direct_target_camera=config.methods.ap01.direct_target_camera,
        top_moving_per_marker=config.methods.ap01.top_moving_per_marker,
    )
    observation_contract = _frozen_observation_contract(experiment, config)
    observations = _observation_counts(experiment)
    input_comparison = _historical_input_comparison(repository, experiment)
    dataset_identity = build_dataset_identity(experiment)

    method_roots = {
        method: experiment / "methods" / method
        for method in ("ap01", "ap02", "ap03")
    }
    before = {
        method: _tree_snapshot(root, relative_to=experiment)
        for method, root in method_roots.items()
    }
    attempts_before = _tree_snapshot(
        experiment / "attempts", relative_to=experiment
    )
    shared_before = _selected_files_snapshot(
        [
            path
            for path in experiment.iterdir()
            if path.is_file()
        ]
        + [experiment / "evaluations"],
        relative_to=experiment,
    )
    public_result = json.loads(
        (
            experiment / "methods/ap01/baseline/RESULT.json"
        ).read_text(encoding="utf-8")
    )
    current_camera_count = int(public_result.get("static_camera_count", 0))

    rejection_reasons = []
    if not input_comparison["exact_match"]:
        rejection_reasons.append(
            "target experiment input is not byte-identical to locked historical Main Route-2 input"
        )
    if observations["accepted_total"] != EXPECTED_ACCEPTED_OBSERVATIONS:
        rejection_reasons.append(
            f"published frozen observations contain {observations['accepted_total']} accepted rows; locked parity requires {EXPECTED_ACCEPTED_OBSERVATIONS}"
        )
    if not rejection_reasons:
        raise RuntimeError("Pre-run rejection requested but no rejecting condition exists")

    common = {
        "schema_version": 1,
        "status": "PRE_RUN_REJECTED",
        "rejection_reasons": rejection_reasons,
        "ground_truth_used": False,
        "method_execution_invoked": False,
        "colmap_invoked": False,
        "publication_invoked": False,
        "reconciliation_invoked": False,
        "ap02_invoked": False,
        "ap03_invoked": False,
    }
    pre_run = {
        **common,
        "repository": {
            "branch": _git(repository, "branch", "--show-current"),
            "head": _git(repository, "rev-parse", "HEAD"),
            "status_short_at_rejection": _git(repository, "status", "--short").splitlines(),
            "editable_package_file": str(Path(package_file).resolve()),
            "editable_package_matches_checkout": Path(package_file).resolve().is_relative_to(
                repository
            ),
        },
        "experiment": str(experiment),
        "input_id": observation_contract["input_id"],
        "runtime_dataset_identity": dataset_identity,
        "locked_dataset_fingerprint": lock["locks"]["dataset_fingerprint"],
        "historical_input_comparison": input_comparison,
        "observation_contract": observation_contract,
        "observation_counts": observations,
        "source_published_contract": original.methods.ap01.method_contract,
        "requested_contract": config.methods.ap01.method_contract,
        "contract_sha256": contract.scientific_fingerprint(),
        "config_fingerprint": config_fingerprint(config),
        "method_fingerprint": method_fingerprint(config, "ap01", selections),
        "resolved_root_camera": selections.root_camera,
        "enabled_methods": list(config.methods.enabled),
        "published_before": {
            "ap01": before["ap01"],
            "ap02": before["ap02"],
            "ap03": before["ap03"],
            "attempts": attempts_before,
            "shared_experiment_metadata": shared_before,
            "ap01_static_camera_count": current_camera_count,
            "ap01_method_fingerprint": public_result.get("method_fingerprint"),
        },
    }
    pre_run_hash = _write_json(
        evidence / "AP01_FULL_RERUN_PRE_RUN.json", pre_run
    )

    manifest = {
        **common,
        "requested_operation": "exactly one complete AP01 explicit method rerun",
        "intended_command": INTENDED_COMMAND,
        "command_executed": None,
        "command_execution_count": 0,
        "queue_id": None,
        "run_id": None,
        "input_id": observation_contract["input_id"],
        "immutable_dataset_fingerprint": dataset_identity.get("fingerprint"),
        "locked_historical_dataset_fingerprint": lock["locks"]["dataset_fingerprint"],
        "ap01_method_fingerprint": method_fingerprint(config, "ap01", selections),
        "ap01_contract_name": CONTRACT,
        "ap01_contract_sha256": contract.scientific_fingerprint(),
        "pre_run_snapshot": {
            "path": "AP01_FULL_RERUN_PRE_RUN.json",
            "sha256": pre_run_hash,
        },
        "old_public_ap01_survived_unchanged": True,
        "existing_attempts_preserved": True,
    }
    stage_counts = {
        **common,
        "observations": {
            "actual": observations,
            "expected_accepted_total": EXPECTED_ACCEPTED_OBSERVATIONS,
            "matches": observations["accepted_total"]
            == EXPECTED_ACCEPTED_OBSERVATIONS,
        },
        "candidates": {
            "actual_total": None,
            "actual_breakdown": None,
            "reason": "method execution was not started",
        },
        "selection": {
            "actual_root": None,
            "actual_per_camera": None,
            "reason": "method execution was not started",
        },
        "final_pose": {
            "actual_inventory": None,
            "actual_order": None,
            "reason": "method execution was not started",
        },
        "stage_execution": {
            "queue_preflight": "rejected_before_queue_creation",
            "observation_stage": "not_started",
            "candidate_construction": "not_started",
            "aggregate_selection": "not_started",
            "final_pose": "not_started",
            "scientific_validation": "pre_run_failed",
            "publication": "not_started",
            "reconciliation": "not_started",
        },
    }
    validation = {
        **common,
        "validation_status": "FAILED_PRE_RUN",
        "checks": {
            "editable_installation_matches_checkout": pre_run["repository"][
                "editable_package_matches_checkout"
            ],
            "requested_contract_is_main_route2_parity_v1": (
                config.methods.ap01.method_contract == CONTRACT
            ),
            "method_fingerprint_resolved": bool(
                method_fingerprint(config, "ap01", selections)
            ),
            "target_dataset_matches_locked_historical_input": input_comparison[
                "exact_match"
            ],
            "accepted_observation_count_is_554": observations[
                "accepted_total"
            ]
            == EXPECTED_ACCEPTED_OBSERVATIONS,
            "old_public_ap01_present": method_roots["ap01"].is_dir(),
            "ap02_present": method_roots["ap02"].is_dir(),
            "ap03_present": method_roots["ap03"].is_dir(),
            "existing_attempts_present": (experiment / "attempts").is_dir(),
        },
        "production_scientific_validation": "UNAVAILABLE",
    }
    after = {
        method: _tree_snapshot(root, relative_to=experiment)
        for method, root in method_roots.items()
    }
    attempts_after = _tree_snapshot(
        experiment / "attempts", relative_to=experiment
    )
    shared_after = _selected_files_snapshot(
        [path for path in experiment.iterdir() if path.is_file()]
        + [experiment / "evaluations"],
        relative_to=experiment,
    )
    publication = {
        **common,
        "publication_status": "NOT_ATTEMPTED",
        "superseded_prior_result_path": None,
        "reconciliation_status": "NOT_ATTEMPTED",
        "old_public_ap01": {
            "before_tree_sha256": before["ap01"]["tree_sha256"],
            "after_tree_sha256": after["ap01"]["tree_sha256"],
            "unchanged": before["ap01"]["tree_sha256"]
            == after["ap01"]["tree_sha256"],
            "still_present": method_roots["ap01"].is_dir(),
            "published_camera_count": current_camera_count,
        },
        "attempts": {
            "before_tree_sha256": attempts_before["tree_sha256"],
            "after_tree_sha256": attempts_after["tree_sha256"],
            "unchanged": attempts_before["tree_sha256"]
            == attempts_after["tree_sha256"],
        },
        "ap02": {
            "before_tree_sha256": before["ap02"]["tree_sha256"],
            "after_tree_sha256": after["ap02"]["tree_sha256"],
            "unchanged": before["ap02"]["tree_sha256"]
            == after["ap02"]["tree_sha256"],
        },
        "ap03": {
            "before_tree_sha256": before["ap03"]["tree_sha256"],
            "after_tree_sha256": after["ap03"]["tree_sha256"],
            "unchanged": before["ap03"]["tree_sha256"]
            == after["ap03"]["tree_sha256"],
        },
        "shared_experiment_metadata": {
            "before_tree_sha256": shared_before["tree_sha256"],
            "after_tree_sha256": shared_after["tree_sha256"],
            "unchanged": shared_before["tree_sha256"]
            == shared_after["tree_sha256"],
        },
    }
    production_parity = {
        **common,
        "status": "unavailable",
        "classification": "UNAVAILABLE",
        "reason": "production AP01 was not started because mandatory input/observation preflight failed",
        "comparison_method": "direct_transform_comparison_without_alignment",
        "locked_final_pose_sha256": lock["locks"][
            "ap01_final_pose_wizard_camera_poses_sha256"
        ],
        "production_result": None,
        "alignment_used": False,
        "best_fit_alignment_used": False,
    }

    hashes = {
        "manifest_sha256": _write_json(
            evidence / "AP01_FULL_RERUN_MANIFEST.json", manifest
        ),
        "stage_counts_sha256": _write_json(
            evidence / "AP01_FULL_RERUN_STAGE_COUNTS.json", stage_counts
        ),
        "validation_sha256": _write_json(
            evidence / "AP01_FULL_RERUN_VALIDATION.json", validation
        ),
        "publication_sha256": _write_json(
            evidence / "AP01_FULL_RERUN_PUBLICATION.json", publication
        ),
        "production_parity_sha256": _write_json(
            evidence / "AP01_PRODUCTION_PARITY.json", production_parity
        ),
        "production_diff_sha256": _write_csv(
            evidence / "AP01_PRODUCTION_DIFF.csv",
            [
                {
                    "status": "unavailable",
                    "classification": "UNAVAILABLE",
                    "camera_id": "",
                    "field": "pre_run_validation",
                    "production_value": "not_executed",
                    "locked_value": "historical_Main_input_and_554_observations",
                    "reason": "; ".join(rejection_reasons),
                }
            ],
        ),
    }
    return {
        "status": common["status"],
        "rejection_reasons": rejection_reasons,
        "intended_command": INTENDED_COMMAND,
        "command_execution_count": 0,
        "hashes": hashes,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(__doc__)
    result.add_argument("--repository", type=Path, default=Path.cwd())
    result.add_argument("--experiment", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    result = record_pre_run_rejection(
        args.repository.resolve(), args.experiment.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

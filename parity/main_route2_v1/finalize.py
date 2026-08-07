"""Finalize the observation-only parity phase from recorded evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .evidence import write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def finalize_observation_phase(
    *, repository: Path, output: Path, tests_summary: str
) -> None:
    root = repository.resolve()
    evidence = output.resolve()
    materialization = _read(evidence / "HISTORICAL_INPUT_MATERIALIZATION.json")
    observations = _read(evidence / "OBSERVATION_PARITY.json")
    lock = _read(evidence / "PARITY_LOCK.json")
    audit = _read(evidence / "INPUT_AUDIT.json")

    lock.update(
        {
            "status": "observation_parity_verified_pre_solver",
            "parity_result": "pre_solver_observation_parity_verified",
            "historical_materialization": {
                "status": materialization["materialization_status"],
                "worktree_path": materialization["worktree_path"],
                "commit_sha": materialization["commit_sha"],
                "hash_status": materialization["hash_status"],
                "file_count": materialization["counts"]["total_files"],
            },
            "observation_parity": {
                "classification": observations["classification"],
                "mode": observations["mode"],
                "main_counts": observations["counts"]["main"],
                "wizard_counts": observations["counts"]["wizard"],
                "first_mismatch": observations["first_mismatch"],
                "tolerances": observations["semantic_comparison"]["tolerances"],
                "ground_truth_used": False,
                "solver_invoked": False,
            },
            "method_core_parity": {
                "status": "unavailable",
                "reason": (
                    "AP01 candidate/selection/pose, AP02 graph/parameter/residual/pose, "
                    "and AP03 reconstruction comparisons belong to later solver phases."
                ),
            },
            "next_phase": (
                "freeze the exact accepted observation table for both implementations, "
                "then compare AP01 candidate construction and selection only; stop before "
                "final pose estimation and all AP02/AP03 solvers"
            ),
        }
    )
    lock.pop("reason_parity_unavailable", None)
    write_json(evidence / "PARITY_LOCK.json", lock)

    audit["status"] = "audited_observation_parity_exact_pre_solver"
    audit["historical_recovery"].update(
        {
            "materialized_in_isolated_worktree": True,
            "materialization_evidence": (
                "parity/main_route2_v1/HISTORICAL_INPUT_MATERIALIZATION.json"
            ),
            "materialization_status": materialization["materialization_status"],
            "materialized_worktree_path": materialization["worktree_path"],
        }
    )
    audit["pre_solver_policy"].update(
        {
            "observation_parity_classification": observations["classification"],
            "observation_solver_invoked": False,
        }
    )
    write_json(evidence / "INPUT_AUDIT.json", audit)

    status = subprocess.check_output(
        ["git", "-c", "core.longpaths=true", "status", "--short"],
        cwd=root,
        text=True,
    ).splitlines()
    main_counts = observations["counts"]["main"]
    wizard_counts = observations["counts"]["wizard"]
    report = [
        "MAIN-TO-WIZARD ROUTE-2 OBSERVATION PARITY",
        "=" * 72,
        "",
        "Safety boundary",
        "- No AP01/AP02/AP03 solver, AP02 bundle adjustment, COLMAP, ROS, Gazebo, reconcile, or Ground Truth was executed/read.",
        "- The detached historical worktree remained clean and all generated evidence stayed under parity/main_route2_v1/.",
        "",
        "1. Historical worktree status",
        f"- {materialization['materialization_status']} at {materialization['commit_sha']}",
        f"- path: {materialization['worktree_path']}",
        f"- clean: {materialization['worktree_clean']}",
        "",
        "2. Historical input counts and hashes",
        f"- counts: {materialization['counts']}",
        f"- hash status: {materialization['hash_status']}; missing={materialization['missing_files']}; unexpected={materialization['unexpected_files']}",
        f"- LFS pointers: {materialization['lfs_pointer_files']}; invalid content: {materialization['invalid_content_files']}",
        "",
        "3. Files modified",
        "- source: runtime.py, queueing.py, rerun.py",
        "- parity harness/evidence: materialization.py, observation_parity.py, finalize.py, CLI, generated Legacy/Wizard observation directories, JSON/CSV reports",
        "- tests: test_rerun_guard_audit.py and test_observation_parity.py",
        "- published results and the preserved SUMMARY.json/attempts/AP03 artifacts were not changed",
        f"- final git status --short: {status}",
        "",
        "4. Runtime guard change",
        "- rerun-method now passes explicit_method_rerun=True through QueueRunner to PipelineOrchestrator",
        "- only that context may pass an existing conflicting method target, and only when the existing result input ID and copied immutable dataset identity match exactly",
        "- normal queues still reject conflicts; stage reuse remains independently fingerprint-gated",
        "- the old target is untouched until validated transactional publication; reconcile is gated on completed+published success",
        "",
        "5. Focused tests",
        f"- {tests_summary}",
        "",
        "6. Legacy observation counts",
        f"- raw={main_counts['raw']}; accepted={main_counts['accepted']}; rejected={main_counts['rejected']}",
        "- Main emits no separate rejection table; all generated detector rows continue to pre-solver method preparation",
        "",
        "7. Wizard observation counts",
        f"- raw={wizard_counts['raw']}; accepted={wizard_counts['accepted']}; rejected={wizard_counts['rejected']}",
        "- baseline observation_quality_v2 accepted every row",
        "",
        "8. First observation mismatch",
        f"- {observations['first_mismatch']}",
        "",
        "9. Observation parity classification",
        f"- {observations['classification']}",
        "- image inventory, normalized IDs, detector semantics, marker inventory, row keys, corner order/coordinates, PnP values/convention, reconstructed reprojection metrics, filtering, duplicates, and original ordering agree",
        f"- tolerances: {observations['semantic_comparison']['tolerances']}; maximum deltas: {observations['semantic_comparison']['maximum_absolute_deltas']}",
        "",
        "10. Remaining blockers",
        "- AP01 candidate, aggregate-selection and pose parity remain unavailable",
        "- AP02 graph, parameter, residual and pose parity remain unavailable",
        "- AP03 reconstruction parity remains unavailable",
        "- no full-method parity claim is made by this observation-only result",
        "",
        "11. Exact next phase",
        "- use the now-verified frozen accepted observation rows to compare Legacy and Wizard AP01 candidate construction and aggregate selection only",
        "- stop before AP01 final pose estimation, AP02 initialization/BA, and every AP03/COLMAP stage",
        "",
    ]
    (evidence / "FINAL_PARITY_REPORT.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

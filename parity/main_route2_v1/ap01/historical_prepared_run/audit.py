"""Audit the historical prepared-input gate and protected experiment.

This module is evidence-only.  It never invokes a detector, calibration
method, COLMAP, publisher, reconciler, evaluator, ROS, or Gazebo process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from parity.main_route2_v1.ap01_candidate_parity import _canonical_row
from parity.main_route2_v1.observation_parity import (
    load_observation_csv,
    semantic_observation_rows,
)


HISTORICAL_COMMIT = "8f9dcea1e8b3189b3c195db2cafe65d5b0e5756b"
EXPECTED_COUNTS = {"total": 554, "static": 30, "moving": 524}
LOCKED_INPUT_COUNT = 199
METADATA_FILES = (
    "metadata/bus_real_data_moving_camera.sdf",
    "metadata/moving_camera_route2_interpolated_final.json",
    "metadata/route_commanded.csv",
)
DIFF_FIELDS = (
    "row_index",
    "observation_key",
    "field",
    "prepared_value",
    "locked_value",
    "reason",
)
PRODUCTION_DIFF_FIELDS = (
    "camera_id",
    "status",
    "selected_candidate_type",
    "rotation_matrix_max_abs_delta",
    "relative_rotation_angle_deg",
    "translation_x_abs_delta_m",
    "translation_y_abs_delta_m",
    "translation_z_abs_delta_m",
    "translation_norm_delta_m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    payload = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_diff(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIFF_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return _sha256(path)


def _files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    return sorted(path for path in root.rglob("*") if path.is_file())


def tree_snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
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
        "root": str(root),
        "present": root.is_dir(),
        "file_count": len(entries),
        "tree_sha256": digest,
        "files": entries,
    }


def snapshot_protected(
    *, repository: Path, protected_experiment: Path, output: Path
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "SNAPSHOT_COMPLETE",
        "scope": "entire protected current-dataset experiment",
        "repository_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "experiment": tree_snapshot(protected_experiment),
    }
    _write_json(output, payload)
    return payload


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], child))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            result.update(_flatten(item, child))
        return result
    return {prefix: value}


def _observation_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["source_kind"]) for row in rows)
    return {
        "total": len(rows),
        "static": counts["static"],
        "moving": counts["moving"],
    }


def _compare_observations(
    prepared_csv: Path, locked_jsonl: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared_source = load_observation_csv(prepared_csv)
    prepared_semantic = semantic_observation_rows(prepared_source)
    prepared = [
        _canonical_row(semantic, source)
        for semantic, source in zip(
            prepared_semantic, prepared_source, strict=True
        )
    ]
    with locked_jsonl.open(encoding="utf-8") as handle:
        locked = [json.loads(line) for line in handle if line.strip()]

    differences: list[dict[str, Any]] = []
    maximum_differences = 500
    for index in range(max(len(prepared), len(locked))):
        if len(differences) >= maximum_differences:
            break
        if index >= len(prepared) or index >= len(locked):
            differences.append(
                {
                    "row_index": index,
                    "observation_key": "",
                    "field": "row",
                    "prepared_value": "missing" if index >= len(prepared) else "present",
                    "locked_value": "missing" if index >= len(locked) else "present",
                    "reason": "row_count_mismatch",
                }
            )
            continue
        left = {key: value for key, value in prepared[index].items() if key != "image_path"}
        right = {key: value for key, value in locked[index].items() if key != "image_path"}
        left_flat = _flatten(left)
        right_flat = _flatten(right)
        for field in sorted(set(left_flat) | set(right_flat)):
            if left_flat.get(field) == right_flat.get(field):
                continue
            differences.append(
                {
                    "row_index": index,
                    "observation_key": str(
                        locked[index].get("observation_key", "")
                    ),
                    "field": field,
                    "prepared_value": json.dumps(
                        left_flat.get(field), sort_keys=True
                    ),
                    "locked_value": json.dumps(
                        right_flat.get(field), sort_keys=True
                    ),
                    "reason": "semantic_value_mismatch",
                }
            )
            if len(differences) >= maximum_differences:
                break

    prepared_keys = [
        (
            row["source_kind"],
            row["camera_id"],
            row["frame_id"],
            row["marker_id"],
            row["occurrence_index"],
        )
        for row in prepared
    ]
    locked_keys = [
        (
            row["source_kind"],
            row["camera_id"],
            row["frame_id"],
            row["marker_id"],
            row["occurrence_index"],
        )
        for row in locked
    ]
    image_paths_equal = len(prepared) == len(locked) and all(
        Path(str(left.get("image_path", ""))).name
        == Path(str(right.get("image_path", ""))).name
        for left, right in zip(prepared, locked, strict=True)
    )
    exact = (
        not differences
        and prepared_keys == locked_keys
        and _observation_counts(prepared) == EXPECTED_COUNTS
    )
    return (
        {
            "schema_version": 1,
            "status": "equal" if exact else "different",
            "classification": "EXACT" if exact else "DIFFERENT",
            "prepared_counts": _observation_counts(prepared),
            "locked_counts": _observation_counts(locked),
            "expected_counts": EXPECTED_COUNTS,
            "row_key_fields": [
                "source_kind",
                "camera_id",
                "frame_id",
                "marker_id",
                "occurrence_index",
            ],
            "row_keys_exact": prepared_keys == locked_keys,
            "original_order_exact": prepared_keys == locked_keys,
            "duplicate_base_key_count_prepared": sum(
                count - 1
                for count in Counter(key[:-1] for key in prepared_keys).values()
            ),
            "duplicate_base_key_count_locked": sum(
                count - 1
                for count in Counter(key[:-1] for key in locked_keys).values()
            ),
            "semantic_fields_exact": not differences,
            "image_filename_parity": image_paths_equal,
            "image_path_policy": (
                "absolute source roots are provenance-only; semantic comparison "
                "requires the same filename and excludes the absolute prefix"
            ),
            "compared_fields": (
                "camera/frame/marker keys, original order, corners and corner "
                "order, PnP rotation/translation, reprojection, camera model, "
                "geometry, quality/filter decisions, detector metadata, and "
                "duplicate occurrences"
            ),
            "difference_count": len(differences),
            "first_causal_divergence": differences[0] if differences else None,
            "ground_truth_used": False,
            "solver_invoked": False,
            "colmap_invoked": False,
        },
        differences,
    )


def verify_prepared(
    *,
    repository: Path,
    experiment: Path,
    historical_source: Path,
    command: str,
) -> dict[str, Any]:
    evidence = repository / "parity/main_route2_v1/ap01/historical_prepared_run"
    materialization = json.loads(
        (repository / "parity/main_route2_v1/HISTORICAL_INPUT_MATERIALIZATION.json").read_text(
            encoding="utf-8"
        )
    )
    with (repository / "parity/main_route2_v1/INPUT_FILE_HASHES.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        locked_rows = [
            row
            for row in csv.DictReader(handle)
            if row["dataset_side"] == "main_historical"
        ]
    comparisons: list[dict[str, Any]] = []
    for row in locked_rows:
        source = historical_source / "raw_images" / row["path"]
        prepared = experiment / "raw_images" / row["path"]
        source_hash = _sha256(source) if source.is_file() else None
        prepared_hash = _sha256(prepared) if prepared.is_file() else None
        comparisons.append(
            {
                "category": row["category"],
                "path": row["path"],
                "locked_sha256": row["sha256"],
                "source_sha256": source_hash,
                "prepared_sha256": prepared_hash,
                "source_matches_lock": source_hash == row["sha256"],
                "prepared_matches_lock": prepared_hash == row["sha256"],
                "byte_identity_preserved": source_hash == prepared_hash,
            }
        )
    metadata_comparisons = []
    for relative in METADATA_FILES:
        source = historical_source / relative
        prepared = experiment / relative
        source_hash = _sha256(source) if source.is_file() else None
        prepared_hash = _sha256(prepared) if prepared.is_file() else None
        metadata_comparisons.append(
            {
                "path": relative,
                "source_sha256": source_hash,
                "prepared_sha256": prepared_hash,
                "byte_identity_preserved": bool(source_hash)
                and source_hash == prepared_hash,
            }
        )
    source_scope_status = subprocess.run(
        [
            "git",
            "-c",
            "core.longpaths=true",
            "status",
            "--short",
            "--",
            "results/bus_real_data/ablation/world/route/route2",
        ],
        cwd=materialization["worktree_path"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    identity = json.loads(
        (experiment / "metadata/dataset_identity.json").read_text(
            encoding="utf-8"
        )
    )
    input_exact = (
        len(comparisons) == LOCKED_INPUT_COUNT
        and all(row["source_matches_lock"] for row in comparisons)
        and all(row["prepared_matches_lock"] for row in comparisons)
        and all(row["byte_identity_preserved"] for row in comparisons)
        and all(row["byte_identity_preserved"] for row in metadata_comparisons)
        and not source_scope_status
    )
    input_payload = {
        "schema_version": 1,
        "status": "equal" if input_exact else "different",
        "classification": "EXACT_HASH_MATCH" if input_exact else "DIFFERENT",
        "historical_commit": HISTORICAL_COMMIT,
        "historical_materialization_status": materialization[
            "materialization_status"
        ],
        "historical_materialization_was_clean": materialization[
            "worktree_clean"
        ],
        "current_source_scope_status": source_scope_status,
        "locked_file_count": len(locked_rows),
        "expected_locked_file_count": LOCKED_INPUT_COUNT,
        "matched_file_count": sum(
            row["prepared_matches_lock"] for row in comparisons
        ),
        "mismatch_count": sum(
            not row["prepared_matches_lock"] for row in comparisons
        ),
        "comparisons": comparisons,
        "simulation_acquisition_metadata": metadata_comparisons,
        "prepared_input_fingerprint": identity["fingerprint"],
        "prepared_input_file_count": identity["file_count"],
        "ground_truth_used": False,
    }
    _write_json(evidence / "HISTORICAL_PREPARED_INPUT_PARITY.json", input_payload)

    observation_payload, differences = _compare_observations(
        experiment / "observations/quality/accepted_observations.csv",
        repository
        / "parity/main_route2_v1/frozen/ap01_accepted_observations.jsonl",
    )
    _write_json(
        evidence / "HISTORICAL_PREPARED_OBSERVATION_PARITY.json",
        observation_payload,
    )
    _write_diff(
        evidence / "HISTORICAL_PREPARED_OBSERVATION_DIFF.csv",
        differences,
    )

    preparation = json.loads(
        (experiment / "metadata/preparation.json").read_text(encoding="utf-8")
    )
    experiment_payload = {
        "schema_version": 1,
        "status": (
            "READY_FOR_AP01"
            if input_exact and observation_payload["classification"] == "EXACT"
            else "STOP_BEFORE_AP01"
        ),
        "experiment": str(experiment.resolve()),
        "historical_source": str(historical_source.resolve()),
        "historical_commit": HISTORICAL_COMMIT,
        "preparation_command": command,
        "queue_id": preparation.get("queue_id"),
        "execution_mode": "prepare_only",
        "calibration_methods_executed": [],
        "prepared_input_fingerprint": identity["fingerprint"],
        "input_parity": input_payload["classification"],
        "observation_parity": observation_payload["classification"],
        "observation_counts": observation_payload["prepared_counts"],
        "ground_truth_used": False,
        "ap01_invoked": False,
        "ap02_invoked": False,
        "ap03_invoked": False,
        "colmap_invoked": False,
    }
    _write_json(
        evidence / "HISTORICAL_PREPARED_EXPERIMENT.json",
        experiment_payload,
    )
    if experiment_payload["status"] != "READY_FOR_AP01":
        divergence = (
            observation_payload["first_causal_divergence"]
            if observation_payload["classification"] != "EXACT"
            else next(
                (
                    row
                    for row in comparisons
                    if not row["prepared_matches_lock"]
                ),
                metadata_comparisons,
            )
        )
        raise RuntimeError(
            "Historical prepared-input gate failed; AP01 must not run. "
            f"First divergence: {divergence}"
        )
    return experiment_payload


def verify_protected(
    *, before: Path, protected_experiment: Path, output: Path
) -> dict[str, Any]:
    locked = json.loads(before.read_text(encoding="utf-8"))["experiment"]
    after = tree_snapshot(protected_experiment)
    payload = {
        "schema_version": 1,
        "status": (
            "UNCHANGED"
            if locked["tree_sha256"] == after["tree_sha256"]
            else "CHANGED"
        ),
        "before_tree_sha256": locked["tree_sha256"],
        "after_tree_sha256": after["tree_sha256"],
        "before_file_count": locked["file_count"],
        "after_file_count": after["file_count"],
        "unchanged": locked["tree_sha256"] == after["tree_sha256"],
        "experiment": after,
    }
    _write_json(output, payload)
    if not payload["unchanged"]:
        raise RuntimeError("Protected current experiment changed")
    return payload


def _rpy_transform(row: Mapping[str, str]) -> np.ndarray:
    roll, pitch, yaw = (
        math.radians(float(row[field]))
        for field in ("roll_deg", "pitch_deg", "yaw_deg")
    )
    rx = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(roll), -math.sin(roll)),
            (0.0, math.sin(roll), math.cos(roll)),
        )
    )
    ry = np.asarray(
        (
            (math.cos(pitch), 0.0, math.sin(pitch)),
            (0.0, 1.0, 0.0),
            (-math.sin(pitch), 0.0, math.cos(pitch)),
        )
    )
    rz = np.asarray(
        (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = [
        float(row["x_m"]),
        float(row["y_m"]),
        float(row["z_m"]),
    ]
    return transform


def _selected_type(source: str) -> str:
    if source == "gauge_identity":
        return "root"
    if source.startswith("direct_"):
        return "direct"
    if source.startswith("moving_relay_"):
        return "relay"
    return "unknown"


def _pose_differences(
    pose_csv: Path, locked_pose_json: Path, tolerances: Mapping[str, float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with pose_csv.open(newline="", encoding="utf-8") as handle:
        produced_rows = list(csv.DictReader(handle))
    produced = {
        str(row["entity_id"]): (_rpy_transform(row), row)
        for row in produced_rows
    }
    locked_payload = json.loads(locked_pose_json.read_text(encoding="utf-8"))
    locked_records = {
        str(row["camera_id"]): row
        for row in locked_payload["camera_records"]
    }
    locked_order = list(locked_payload["camera_order"])
    rows: list[dict[str, Any]] = []
    for camera_id in locked_order:
        locked = np.asarray(
            locked_records[camera_id]["homogeneous_transform_4x4"],
            dtype=np.float64,
        )
        if camera_id not in produced:
            rows.append(
                {
                    "camera_id": camera_id,
                    "status": "MISSING",
                    "selected_candidate_type": "",
                    **{
                        field: ""
                        for field in PRODUCTION_DIFF_FIELDS[3:]
                    },
                }
            )
            continue
        actual, source_row = produced[camera_id]
        relative = actual[:3, :3] @ locked[:3, :3].T
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        rotation_angle = math.degrees(math.acos(cosine))
        rotation_max = float(
            np.max(np.abs(actual[:3, :3] - locked[:3, :3]))
        )
        translation_delta = np.abs(actual[:3, 3] - locked[:3, 3])
        translation_norm = float(
            np.linalg.norm(actual[:3, 3] - locked[:3, 3])
        )
        exact = (
            rotation_max <= float(tolerances["rotation_matrix_max_abs"])
            and rotation_angle
            <= float(tolerances["relative_rotation_angle_deg"])
            and translation_norm <= float(tolerances["translation_norm_m"])
        )
        rows.append(
            {
                "camera_id": camera_id,
                "status": "EXACT" if exact else "DIFFERENT",
                "selected_candidate_type": _selected_type(source_row["source"]),
                "rotation_matrix_max_abs_delta": rotation_max,
                "relative_rotation_angle_deg": rotation_angle,
                "translation_x_abs_delta_m": float(translation_delta[0]),
                "translation_y_abs_delta_m": float(translation_delta[1]),
                "translation_z_abs_delta_m": float(translation_delta[2]),
                "translation_norm_delta_m": translation_norm,
            }
        )
    inventory_exact = set(produced) == set(locked_records)
    pose_exact = inventory_exact and all(row["status"] == "EXACT" for row in rows)
    numeric_rows = [row for row in rows if row["status"] != "MISSING"]
    return (
        {
            "classification": "EXACT" if pose_exact else "DIFFERENT_FINAL_POSES",
            "inventory_exact": inventory_exact,
            "locked_camera_order": locked_order,
            "produced_file_order": [str(row["entity_id"]) for row in produced_rows],
            "produced_camera_inventory": list(produced),
            "maximum_rotation_matrix_element_abs_delta": max(
                float(row["rotation_matrix_max_abs_delta"])
                for row in numeric_rows
            ),
            "maximum_relative_rotation_angle_deg": max(
                float(row["relative_rotation_angle_deg"])
                for row in numeric_rows
            ),
            "maximum_translation_vector_norm_delta_m": max(
                float(row["translation_norm_delta_m"])
                for row in numeric_rows
            ),
            "first_divergence": next(
                (row for row in rows if row["status"] != "EXACT"), None
            ),
            "comparison_method": "direct_transform_comparison_without_alignment",
            "alignment_used": False,
            "ground_truth_used": False,
            "tolerances": dict(tolerances),
        },
        rows,
    )


def verify_production_attempt(
    *,
    repository: Path,
    experiment: Path,
    attempt: Path,
    command: str,
) -> dict[str, Any]:
    """Audit one retained production attempt without executing any method."""

    evidence = repository / "parity/main_route2_v1/ap01/historical_prepared_run"
    run_manifest = json.loads(
        (attempt / "diagnostics/run_manifest.json").read_text(encoding="utf-8")
    )
    failure = json.loads((attempt / "FAILURE.json").read_text(encoding="utf-8"))
    candidate_root = attempt / "diagnostics/02_AP01/03_candidates"
    candidate_manifest = json.loads(
        (candidate_root / "stage_manifest.json").read_text(encoding="utf-8")
    )
    candidates = json.loads(
        (candidate_root / "transform_candidates.json").read_text(encoding="utf-8")
    )
    locked_candidate = json.loads(
        (
            repository
            / "parity/main_route2_v1/ap01/post_fix/AP01_CANDIDATE_PARITY.json"
        ).read_text(encoding="utf-8")
    )
    locked_breakdown = locked_candidate["candidate_counts"]["wizard"]
    breakdown: dict[str, dict[str, int]] = {}
    for row in candidates:
        camera = str(row["target_camera"])
        mode = str(row["mode"])
        breakdown.setdefault(camera, {})[mode] = (
            breakdown.setdefault(camera, {}).get(mode, 0) + 1
        )
    root_camera = str(run_manifest["resolved_selections"]["ap01_root_camera"])
    breakdown.setdefault(root_camera, {})["root"] = 1
    candidate_total = len(candidates) + 1

    pose_root = attempt / "diagnostics/02_AP01/03_static_extrinsics"
    locked_final_parity = json.loads(
        (
            repository
            / "parity/main_route2_v1/ap01/final_pose/AP01_FINAL_POSE_PARITY.json"
        ).read_text(encoding="utf-8")
    )
    tolerances = locked_final_parity["tolerances"]
    pose_summary, pose_rows = _pose_differences(
        pose_root / "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv",
        repository
        / "parity/main_route2_v1/ap01/final_pose/wizard/AP01_FINAL_CAMERA_POSES.json",
        tolerances,
    )
    with (
        attempt / "diagnostics/preflight/accepted_observations.csv"
    ).open(newline="", encoding="utf-8") as handle:
        observation_rows = list(csv.DictReader(handle))
    observation_counts = Counter(row["observer_type"] for row in observation_rows)
    selected = {
        row["camera_id"]: row["selected_candidate_type"]
        for row in pose_rows
    }
    locked_selected = {
        row["camera_id"]: row["source_selected_candidate_type"]
        for row in json.loads(
            (
                repository
                / "parity/main_route2_v1/ap01/final_pose/wizard/AP01_FINAL_CAMERA_POSES.json"
            ).read_text(encoding="utf-8")
        )["camera_records"]
    }
    pairwise_count = sum(
        1
        for _ in csv.DictReader(
            (pose_root / "AP01_PAIRWISE_DISTANCES.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    published = experiment / "methods/ap01/baseline"
    summary = json.loads((experiment / "SUMMARY.json").read_text(encoding="utf-8"))
    contract = candidate_manifest["parameters"]["method_contract"]
    candidate_exact = (
        candidate_total == locked_candidate["candidate_multiplicity"]["wizard_count"]
        and breakdown == locked_breakdown
    )
    selection_exact = selected == locked_selected
    completed = run_manifest["status"] == "completed"
    authoritative_classification = (
        pose_summary["classification"] if completed else "UNAVAILABLE"
    )
    production_payload = {
        "schema_version": 1,
        "status": run_manifest["status"].upper(),
        "production_command": command,
        "queue_id": summary.get("queue_id"),
        "run_id": run_manifest["run_id"],
        "input_id": run_manifest["input_id"],
        "prepared_input_fingerprint": run_manifest["dataset_identity"]["fingerprint"],
        "method_contract": contract["name"],
        "method_contract_sha256": candidate_manifest["parameters"][
            "method_contract_sha256"
        ],
        "method_fingerprint": run_manifest["method_fingerprint"],
        "enabled_methods": run_manifest["enabled_methods"],
        "observation_counts": {
            "total": len(observation_rows),
            "static": observation_counts["static"],
            "moving": observation_counts["moving"],
        },
        "stored_non_root_candidate_count": len(candidates),
        "candidate_count_including_root_gauge": candidate_total,
        "candidate_breakdown": breakdown,
        "candidate_count_and_breakdown_match_lock": candidate_exact,
        "selection": selected,
        "selection_matches_lock": selection_exact,
        "partial_final_pose": pose_summary,
        "partial_pairwise_count": pairwise_count,
        "published_result_present": published.is_dir(),
        "published_camera_count": 0,
        "published_pairwise_count": 0,
        "evaluation_ran": False,
        "ap02_ran": False,
        "ap03_ran": False,
        "ground_truth_used": False,
        "failure": failure,
        "serializer_fix_required": (
            "quality_filtered_weighted_mean_diagnostic ndarray must be "
            "converted to a JSON list"
        ),
    }
    parity_payload = {
        "schema_version": 1,
        "status": "unavailable_incomplete_production_attempt",
        "classification": authoritative_classification,
        "partial_scientific_classification": pose_summary["classification"],
        "observations_exact": production_payload["observation_counts"]
        == EXPECTED_COUNTS,
        "candidate_count_and_breakdown_exact": candidate_exact,
        "selection_exact": selection_exact,
        "final_pose": pose_summary,
        "publication_succeeded": published.is_dir(),
        "reconciliation_performed": False,
        "first_runtime_failure": failure["evidence"],
        "first_scientific_divergence": pose_summary["first_divergence"],
        "ground_truth_used": False,
        "alignment_used": False,
    }
    _write_json(evidence / "AP01_PRODUCTION_RUN.json", production_payload)
    _write_json(evidence / "AP01_PRODUCTION_PARITY.json", parity_payload)
    with (evidence / "AP01_PRODUCTION_DIFF.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PRODUCTION_DIFF_FIELDS)
        writer.writeheader()
        writer.writerows(pose_rows)
    _write_json(
        evidence / "AP01_PRODUCTION_PUBLICATION.json",
        {
            "schema_version": 1,
            "status": "NOT_PUBLISHED_METHOD_FAILED",
            "attempt": str(attempt.resolve()),
            "authoritative_method_result": str(published.resolve()),
            "authoritative_method_result_present": published.is_dir(),
            "failed_attempt_preserved": attempt.is_dir(),
            "reconciliation_performed": False,
        },
    )
    return parity_payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(__doc__)
    sub = result.add_subparsers(dest="action", required=True)
    start = sub.add_parser("snapshot-protected")
    start.add_argument("--repository", type=Path, default=Path.cwd())
    start.add_argument("--experiment", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)
    gate = sub.add_parser("verify-prepared")
    gate.add_argument("--repository", type=Path, default=Path.cwd())
    gate.add_argument("--experiment", type=Path, required=True)
    gate.add_argument("--historical-source", type=Path, required=True)
    gate.add_argument("--command", required=True)
    finish = sub.add_parser("verify-protected")
    finish.add_argument("--before", type=Path, required=True)
    finish.add_argument("--experiment", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    production = sub.add_parser("verify-production-attempt")
    production.add_argument("--repository", type=Path, default=Path.cwd())
    production.add_argument("--experiment", type=Path, required=True)
    production.add_argument("--attempt", type=Path, required=True)
    production.add_argument("--command", required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    if arguments.action == "snapshot-protected":
        result = snapshot_protected(
            repository=arguments.repository.resolve(),
            protected_experiment=arguments.experiment.resolve(),
            output=arguments.output.resolve(),
        )
    elif arguments.action == "verify-prepared":
        result = verify_prepared(
            repository=arguments.repository.resolve(),
            experiment=arguments.experiment.resolve(),
            historical_source=arguments.historical_source.resolve(),
            command=arguments.command,
        )
    elif arguments.action == "verify-protected":
        result = verify_protected(
            before=arguments.before.resolve(),
            protected_experiment=arguments.experiment.resolve(),
            output=arguments.output.resolve(),
        )
    else:
        result = verify_production_attempt(
            repository=arguments.repository.resolve(),
            experiment=arguments.experiment.resolve(),
            attempt=arguments.attempt.resolve(),
            command=arguments.command,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the single published historical AP02 reproduction without GT."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from camera_rig_calibration.methods.ap02.optimize_core import T_from_pose_row


REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "ap02"
LEGACY_COMMIT = "8f9dcea1e8b3189b3c195db2cafe65d5b0e5756b"
LEGACY_ROOT = "results/bus_real_data/02_ref_marker_graph_ba"
PUBLIC = (
    REPOSITORY
    / "results/simulation/baseline/route2_main_parity_v1/methods/ap02/baseline"
)
CURRENT_BA = PUBLIC / "diagnostics/method/graph_ba/with_moving"
POSE_TOLERANCE = 1e-6
PRODUCTION_COMMAND = (
    'wsl.exe --cd "$PWD" -e env PYTHONPATH=src '
    "workspace/.venv-wsl/bin/python -m camera_rig_calibration.cli "
    "rerun-method --experiment "
    "results/simulation/baseline/route2_main_parity_v1 --method ap02 "
    "--variant baseline --reuse-prepared-input "
    "--ap02-historical-reproduction --reconcile-after"
)


def git_text(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{LEGACY_COMMIT}:{path}"],
        cwd=REPOSITORY,
        text=True,
        encoding="utf-8",
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def summary_number(text: str, section: str, label: str) -> float:
    match = re.search(
        rf"{re.escape(section)}.*?{re.escape(label)}:\s*([0-9.]+)",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"Missing summary value: {section} / {label}")
    return float(match.group(1))


def rotation_delta_degrees(left: np.ndarray, right: np.ndarray) -> float:
    delta = left[:3, :3].T @ right[:3, :3]
    cosine = min(1.0, max(-1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def main() -> None:
    current_rows = rows(
        (CURRENT_BA / "optimized_static_camera_poses_ref_marker.csv")
        .read_text(encoding="utf-8")
    )
    historical_rows = rows(
        git_text(
            f"{LEGACY_ROOT}/07_graph_ba/with_moving/"
            "optimized_static_camera_poses_ref_marker.csv"
        )
    )
    current = {row["entity_id"]: T_from_pose_row(row) for row in current_rows}
    historical = {
        row["entity_id"]: T_from_pose_row(row) for row in historical_rows
    }
    inventory_exact = set(current) == set(historical)
    differences: list[dict[str, object]] = []
    maximum_element_delta = 0.0
    maximum_translation_delta = 0.0
    maximum_rotation_delta = 0.0
    for camera in sorted(set(current) & set(historical)):
        element = float(np.max(np.abs(current[camera] - historical[camera])))
        translation = float(
            np.linalg.norm(current[camera][:3, 3] - historical[camera][:3, 3])
        )
        rotation = rotation_delta_degrees(current[camera], historical[camera])
        maximum_element_delta = max(maximum_element_delta, element)
        maximum_translation_delta = max(maximum_translation_delta, translation)
        maximum_rotation_delta = max(maximum_rotation_delta, rotation)
        differences.append(
            {
                "camera_id": camera,
                "maximum_transform_element_delta": element,
                "translation_delta_m": translation,
                "rotation_delta_deg": rotation,
                "within_tolerance": element <= POSE_TOLERANCE,
            }
        )
    if inventory_exact and maximum_element_delta == 0.0:
        classification = "EXACT"
    elif inventory_exact and maximum_element_delta <= POSE_TOLERANCE:
        classification = "NUMERICALLY_EQUIVALENT_WITHIN_TOLERANCE"
    else:
        classification = "DIFFERENT_FINAL_POSES"

    current_summary = (CURRENT_BA / "ba_summary.txt").read_text(encoding="utf-8")
    historical_summary = git_text(
        f"{LEGACY_ROOT}/07_graph_ba/with_moving/ba_summary.txt"
    )
    optimizer = json.loads(
        (CURRENT_BA / "optimizer_report.json").read_text(encoding="utf-8")
    )
    result = json.loads((PUBLIC / "RESULT.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (
            PUBLIC
            / "diagnostics/method/aruco_observations/"
            "HISTORICAL_REPRODUCTION_PROVENANCE.json"
        ).read_text(encoding="utf-8")
    )
    pre_solver = json.loads(
        (CURRENT_BA / "HISTORICAL_PRE_SOLVER_INVARIANTS.json").read_text(
            encoding="utf-8"
        )
    )
    with (PUBLIC / "pairwise_camera_extrinsics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        pair_rows = list(csv.DictReader(handle))
    pairs = {
        tuple(sorted((row["from_camera"], row["to_camera"])))
        for row in pair_rows
    }
    comparison = json.loads(
        (
            PUBLIC.parents[2] / "COMPARISON.json"
        ).read_text(encoding="utf-8")
    )
    reconciled_ap02 = any(
        item.get("method") == "ap02"
        and item.get("label") == "baseline"
        and item.get("status") == "available"
        for item in comparison.get("methods", [])
    )
    payload = {
        "schema_version": 1,
        "classification": classification,
        "ground_truth_used": False,
        "legacy_commit": LEGACY_COMMIT,
        "production_command": PRODUCTION_COMMAND,
        "frozen_observation_artifact": provenance["source_artifact"],
        "frozen_observation_sha256": provenance["source_artifact_sha256"],
        "method_contract": provenance["method_contract"],
        "method_contract_sha256": provenance["method_contract_sha256"],
        "pre_solver": pre_solver["actual"],
        "pre_solver_status": pre_solver["status"],
        "final_mean_reprojection_px": summary_number(
            current_summary, "Final reprojection error [px]", "- mean"
        ),
        "historical_final_mean_reprojection_px": summary_number(
            historical_summary, "Final reprojection error [px]", "- mean"
        ),
        "nfev": optimizer["nfev"],
        "historical_nfev": int(
            summary_number(historical_summary, "Optimizer", "- nfev")
        ),
        "static_camera_count": result["static_camera_count"],
        "unordered_camera_pair_count": len(pairs),
        "reference_marker_id": result["reference_marker_id"],
        "camera_inventory_exact": inventory_exact,
        "maximum_transform_element_delta": maximum_element_delta,
        "maximum_translation_delta_m": maximum_translation_delta,
        "maximum_rotation_delta_deg": maximum_rotation_delta,
        "pose_tolerance": POSE_TOLERANCE,
        "per_camera_differences": differences,
        "publication_status": (
            "published"
            if result.get("status") == "available"
            and result.get("execution_status") == "completed"
            else "not_published"
        ),
        "published_result": str(PUBLIC.relative_to(REPOSITORY)),
        "reconciliation_status": (
            "reconciled" if reconciled_ap02 else "not_reconciled"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "AP02_FINAL_POSE_PARITY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT / "AP02_FINAL_POSE_DIFF.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

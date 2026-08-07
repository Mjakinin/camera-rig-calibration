"""Generate concise AP02 graph, initialization, and pre-solver parity evidence."""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from camera_rig_calibration.methods.ap02 import optimize_core
from camera_rig_calibration.methods.ap02.contracts import (
    resolve_ap02_method_contract,
)
from camera_rig_calibration.methods.ap02.frame_selection import (
    select_legacy_smart_moving_observations,
)
from camera_rig_calibration.methods.ap02.initialize import (
    best_observations,
    build_graph,
    initialize_from_tree,
    main_compat_widest_path_tree,
    main_observation_score,
    marker_node,
)


REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "ap02"
LEGACY_COMMIT = "8f9dcea1e8b3189b3c195db2cafe65d5b0e5756b"
LEGACY_ROOT = "results/bus_real_data/02_ref_marker_graph_ba"
COMMON_INPUT = (
    REPOSITORY
    / "parity/main_route2_v1/generated/main_legacy_observations"
    / "shared_all_aruco_observations.csv"
)
POSE_FILES = {
    "static": "initial_static_camera_poses_ref_marker.csv",
    "moving": "initial_moving_frame_poses_ref_marker.csv",
    "marker": "initial_marker_poses_ref_marker.csv",
}
POSE_TOLERANCE = 1e-12
RESIDUAL_TOLERANCE = 1e-10


def git_text(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{LEGACY_COMMIT}:{path}"],
        cwd=REPOSITORY,
        text=True,
        encoding="utf-8",
    )


def git_rows(path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(git_text(path))))


def write_json(name: str, payload: dict[str, object]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_diff(name: str, rows: list[dict[str, object]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    fields = list(rows[0]) if rows else ["category", "key", "difference"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def observation_key(row: dict[str, str]) -> tuple[str, int]:
    return str(row["observer_id"]), int(float(row["marker_id"]))


def state(rows: list[dict[str, str]]) -> dict[str, object]:
    selected = best_observations(
        rows, edge_weight_policy="legacy_observation_quality_v1"
    )
    adjacency = build_graph(selected, preserve_input_order=True)
    parent, metrics = main_compat_widest_path_tree(
        adjacency,
        marker_node(14),
        edge_weight_policy="legacy_observation_quality_v1",
    )
    marker_poses, observer_poses, initialization_log, used_edges = (
        initialize_from_tree(
            parent,
            14,
            path_metrics=metrics,
            algorithm="legacy_maximum_bottleneck_v1",
            edge_weight_policy="legacy_observation_quality_v1",
        )
    )
    return {
        "selected": selected,
        "adjacency": adjacency,
        "parent": parent,
        "metrics": metrics,
        "marker_poses": marker_poses,
        "observer_poses": observer_poses,
        "initialization_log": initialization_log,
        "used_edges": used_edges,
    }


def pre_solver(
    rows: list[dict[str, str]],
    marker_poses: dict[int, np.ndarray],
    observer_poses: dict[str, np.ndarray],
) -> dict[str, object]:
    available = optimize_core.filter_observations(
        rows, "with_moving", marker_poses, observer_poses
    )
    static_rows = [
        row for row in available if row.get("observer_type") == "static"
    ]
    moving_rows = [
        row for row in available if row.get("observer_type") == "moving"
    ]
    selection = select_legacy_smart_moving_observations(
        moving_rows,
        reference_marker_id=14,
        reference_marker_maximum_frames=None,
        top_per_marker=8,
        top_per_marker_pair=4,
        maximum_total_frames=None,
        observation_score=main_observation_score,
    )
    observations = [*static_rows, *selection.selected_rows]
    used_observers = {row["observer_id"] for row in observations}
    selected_observers = {
        key: value
        for key, value in observer_poses.items()
        if key in used_observers
    }
    used_markers = {
        int(float(row["marker_id"])) for row in observations
    }
    selected_markers = {
        key: value
        for key, value in marker_poses.items()
        if key in used_markers or key == 14
    }
    x0, names = optimize_core.pack_params(
        selected_markers, selected_observers, 14
    )
    residual = optimize_core.make_residual_function(
        observations, names, 14
    )(x0)
    norms = np.linalg.norm(residual.reshape(-1, 2), axis=1)
    return {
        "observations": observations,
        "selected_frame_ids": selection.selected_frame_ids,
        "names": names,
        "x0": x0,
        "residual": residual,
        "initial_mean": float(np.mean(norms)),
        "initial_median": float(np.median(norms)),
        "initial_maximum": float(np.max(norms)),
    }


def legacy_pose_maps() -> dict[str, dict[str | int, np.ndarray]]:
    result: dict[str, dict[str | int, np.ndarray]] = {}
    for kind, name in POSE_FILES.items():
        rows = git_rows(
            f"{LEGACY_ROOT}/05_graph_initialization/with_moving/{name}"
        )
        result[kind] = {
            (
                int(row["entity_id"])
                if kind == "marker"
                else row["entity_id"]
            ): optimize_core.T_from_pose_row(row)
            for row in rows
        }
    return result


def summary_number(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s*([0-9.]+)", text)
    if match is None:
        raise RuntimeError(f"Missing historical AP02 summary field: {label}")
    return float(match.group(1))


def main() -> None:
    contract = resolve_ap02_method_contract()
    historical_rows = git_rows(
        f"{LEGACY_ROOT}/02_aruco_observations/ap02_all_aruco_observations.csv"
    )
    baseline = state(historical_rows)

    historical_init_log = git_rows(
        f"{LEGACY_ROOT}/05_graph_initialization/with_moving/initialization_log.csv"
    )
    current_order = [
        (str(row["initialized_type"]), str(row["initialized_id"]))
        for row in baseline["initialization_log"]
    ]
    historical_order = [
        (str(row["initialized_type"]), str(row["initialized_id"]))
        for row in historical_init_log
    ]
    historical_tree_edges = git_rows(
        f"{LEGACY_ROOT}/05_graph_initialization/with_moving/used_initialization_edges.csv"
    )
    current_tree_keys = [
        observation_key(row) for row in baseline["used_edges"]
    ]
    historical_tree_keys = [
        observation_key(row) for row in historical_tree_edges
    ]
    graph_exact = (
        len(baseline["selected"]) == len(historical_rows)
        and current_tree_keys == historical_tree_keys
        and current_order == historical_order
    )

    common_rows = list(csv.DictReader(COMMON_INPUT.open(encoding="utf-8")))
    common = state(common_rows)
    historical_keys = {observation_key(row) for row in historical_rows}
    common_keys = {observation_key(row) for row in common_rows}
    common_moving_frames = {
        row["observer_id"]
        for row in common_rows
        if row.get("observer_type") == "moving"
    }
    historical_moving_frames = {
        row["observer_id"]
        for row in historical_rows
        if row.get("observer_type") == "moving"
    }
    graph_payload = {
        "schema_version": 1,
        "classification": "EXACT" if graph_exact else "DIFFERENT",
        "ground_truth_used": False,
        "legacy_commit": LEGACY_COMMIT,
        "method_contract": contract.fingerprint_payload(),
        "method_contract_sha256": contract.scientific_fingerprint(),
        "locked_historical_ap02_stream": {
            "observation_count": len(historical_rows),
            "moving_frame_count": len(historical_moving_frames),
            "observer_marker_edge_count": len(baseline["selected"]),
            "tree_edge_count": len(baseline["used_edges"]),
            "node_count": len(baseline["adjacency"]),
            "tree_inventory_exact": current_tree_keys == historical_tree_keys,
            "construction_order_exact": current_order == historical_order,
            "edge_weight_policy": "legacy_observation_quality_v1",
        },
        "verified_common_554_stream": {
            "observation_count": len(common_rows),
            "moving_frame_count": len(common_moving_frames),
            "observer_marker_edge_count": len(common["selected"]),
            "historical_stream_common_keys": len(historical_keys & common_keys),
            "historical_only_keys": len(historical_keys - common_keys),
            "common_only_keys": len(common_keys - historical_keys),
            "classification": "DIFFERENT_INPUT_INVENTORY",
            "reason": (
                "The verified 554-row detector stream is exact between the "
                "reconstructed Main and Wizard detectors, but is not the "
                "committed 513-row AP02-specific stream that produced the "
                "historical 170/458/960 result."
            ),
        },
    }
    write_json("AP02_GRAPH_PARITY.json", graph_payload)
    write_diff("AP02_GRAPH_DIFF.csv", [] if graph_exact else [{
        "category": "tree",
        "key": "initialization_order",
        "difference": "Legacy and baseline tree order differ",
    }])

    legacy_poses = legacy_pose_maps()
    current_poses = {
        "static": {
            key: value
            for key, value in baseline["observer_poses"].items()
            if str(key).startswith("cam_edge_")
        },
        "moving": {
            key: value
            for key, value in baseline["observer_poses"].items()
            if str(key).startswith("moving_frame_")
        },
        "marker": baseline["marker_poses"],
    }
    pose_diffs: list[dict[str, object]] = []
    maximum_pose_delta = 0.0
    inventories_exact = True
    for kind in ("static", "moving", "marker"):
        inventories_exact &= set(current_poses[kind]) == set(legacy_poses[kind])
        for key in sorted(
            set(current_poses[kind]) & set(legacy_poses[kind]), key=str
        ):
            delta = float(
                np.max(
                    np.abs(current_poses[kind][key] - legacy_poses[kind][key])
                )
            )
            maximum_pose_delta = max(maximum_pose_delta, delta)
            if delta > POSE_TOLERANCE:
                pose_diffs.append({
                    "category": kind,
                    "key": key,
                    "difference": delta,
                })
    initialization_exact = (
        inventories_exact
        and current_order == historical_order
        and maximum_pose_delta <= POSE_TOLERANCE
    )
    write_json("AP02_INITIALIZATION_PARITY.json", {
        "schema_version": 1,
        "classification": (
            "NUMERICALLY_EQUIVALENT_WITHIN_TOLERANCE"
            if initialization_exact
            else "DIFFERENT"
        ),
        "ground_truth_used": False,
        "initialized_static_cameras": len(current_poses["static"]),
        "initialized_moving_frames": len(current_poses["moving"]),
        "initialized_markers": len(current_poses["marker"]),
        "inventory_exact": inventories_exact,
        "order_exact": current_order == historical_order,
        "maximum_transform_element_delta": maximum_pose_delta,
        "tolerance": POSE_TOLERANCE,
    })
    write_diff("AP02_INITIALIZATION_DIFF.csv", pose_diffs)

    current_pre = pre_solver(
        historical_rows,
        baseline["marker_poses"],
        baseline["observer_poses"],
    )
    reference_pre = pre_solver(
        historical_rows,
        legacy_poses["marker"],
        {**legacy_poses["static"], **legacy_poses["moving"]},
    )
    x0_delta = float(np.max(np.abs(current_pre["x0"] - reference_pre["x0"])))
    residual_delta = float(
        np.max(np.abs(current_pre["residual"] - reference_pre["residual"]))
    )
    historical_selection = git_rows(
        f"{LEGACY_ROOT}/07_graph_ba/with_moving/moving_frame_selection.csv"
    )
    selected_frames_exact = list(current_pre["selected_frame_ids"]) == [
        row["observer_id"] for row in historical_selection
    ]
    names_exact = current_pre["names"] == reference_pre["names"]
    summary = git_text(f"{LEGACY_ROOT}/07_graph_ba/with_moving/ba_summary.txt")
    reported_initial_mean = summary_number(summary, "- mean")
    mean_matches_report = abs(
        float(current_pre["initial_mean"]) - reported_initial_mean
    ) <= 5e-7
    pre_solver_exact = (
        selected_frames_exact
        and names_exact
        and x0_delta <= POSE_TOLERANCE
        and residual_delta <= RESIDUAL_TOLERANCE
        and mean_matches_report
    )
    pre_diffs: list[dict[str, object]] = []
    if not names_exact:
        pre_diffs.append({
            "category": "parameters",
            "key": "ordering",
            "difference": "parameter block ordering differs",
        })
    if not selected_frames_exact:
        pre_diffs.append({
            "category": "selection",
            "key": "moving_frames",
            "difference": "smart moving-frame inventory/order differs",
        })
    if residual_delta > RESIDUAL_TOLERANCE:
        pre_diffs.append({
            "category": "residual",
            "key": "x0",
            "difference": residual_delta,
        })
    write_json("AP02_PRE_SOLVER_PARITY.json", {
        "schema_version": 1,
        "classification": (
            "NUMERICALLY_EQUIVALENT_WITHIN_TOLERANCE"
            if pre_solver_exact
            else "DIFFERENT"
        ),
        "ground_truth_used": False,
        "ba_observations": len(current_pre["observations"]),
        "selected_moving_frames": len(current_pre["selected_frame_ids"]),
        "variable_poses": len(current_pre["names"]),
        "parameter_count": len(current_pre["x0"]),
        "scalar_residual_count": len(current_pre["residual"]),
        "parameter_order_exact": names_exact,
        "selected_frame_inventory_and_order_exact": selected_frames_exact,
        "maximum_x0_delta": x0_delta,
        "maximum_residual_delta_at_x0": residual_delta,
        "initial_mean_reprojection_px": current_pre["initial_mean"],
        "historical_reported_initial_mean_reprojection_px": (
            reported_initial_mean
        ),
        "loss": contract.robust_loss,
        "f_scale": contract.robust_loss_scale_px,
        "solver_method": contract.solver_method,
        "solver_bounds_policy": contract.solver_bounds_policy,
        "tolerances": {
            "pose_and_x0": POSE_TOLERANCE,
            "residual": RESIDUAL_TOLERANCE,
            "reported_mean_rounding": 5e-7,
        },
    })
    write_diff("AP02_PRE_SOLVER_DIFF.csv", pre_diffs)

    common_pre = pre_solver(
        common_rows,
        common["marker_poses"],
        common["observer_poses"],
    )
    write_json("AP02_COMMON_INPUT_READINESS.json", {
        "schema_version": 1,
        "classification": "BLOCKED_BY_DIFFERENT_AP02_OBSERVATION_INVENTORY",
        "ground_truth_used": False,
        "common_observation_count": len(common_rows),
        "common_initialized_moving_frames": len([
            key
            for key in common["observer_poses"]
            if str(key).startswith("moving_frame_")
        ]),
        "common_ba_observations": len(common_pre["observations"]),
        "common_variable_poses": len(common_pre["names"]),
        "common_parameter_count": len(common_pre["x0"]),
        "common_initial_mean_reprojection_px": common_pre["initial_mean"],
        "production_run_authorized": False,
        "reason": (
            "The requested 170/458/960 historical pre-solver invariants are "
            "exact on the committed AP02 stream, but the prepared experiment's "
            "verified 554-row stream has only 169 observed moving frames. A "
            "production run must not silently substitute the historical stream."
        ),
    })


if __name__ == "__main__":
    main()

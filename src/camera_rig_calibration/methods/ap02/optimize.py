"""Distortion-aware entry point for the AP02 graph bundle adjustment.

The underlying parameterization and output contract remain unchanged. Only the
reprojection model is replaced so that real ROS camera_info distortion is not
silently discarded.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np


from . import optimize_core as core


HISTORICAL_PRE_SOLVER_TARGETS = {
    "reference_marker_id": 14,
    "initialized_moving_frames": 170,
    "ba_observations": 458,
    "variable_poses": 160,
    "parameter_count": 960,
    "initial_mean_reprojection_px": 10.247645,
    "loss": "soft_l1",
    "loss_scale_px": 3.0,
    "maximum_function_evaluations": 80,
}


def validate_historical_pre_solver(
    *,
    ap02_root: Path,
    initialization_root: Path,
    reference_marker_id: int,
    maximum_function_evaluations: int,
    robust_loss: str,
    robust_loss_scale_px: float,
    initial_parameters: np.ndarray,
    initial_residuals: np.ndarray,
) -> dict[str, object]:
    """Authorize the historical solver only after invariant reproduction."""

    provenance_path = (
        ap02_root
        / "02_aruco_observations"
        / "HISTORICAL_REPRODUCTION_PROVENANCE.json"
    )
    if not provenance_path.is_file():
        raise RuntimeError(
            "Historical AP02 pre-solver guard has no frozen-input provenance"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    moving_initialization = (
        initialization_root
        / "with_moving"
        / "initial_moving_frame_poses_ref_marker.csv"
    )
    if not moving_initialization.is_file():
        raise RuntimeError(
            "Historical AP02 moving-frame initialization is missing"
        )
    with moving_initialization.open(newline="", encoding="utf-8") as handle:
        initialized_moving_frames = sum(1 for _ in csv.DictReader(handle))
    residuals = np.asarray(initial_residuals, dtype=np.float64)
    parameters = np.asarray(initial_parameters, dtype=np.float64)
    if len(residuals) % 8:
        raise RuntimeError(
            "Historical AP02 residual vector is not four marker corners"
        )
    pairwise = residuals.reshape(-1, 2)
    actual = {
        "reference_marker_id": reference_marker_id,
        "initialized_moving_frames": initialized_moving_frames,
        "ba_observations": len(residuals) // 8,
        "variable_poses": len(parameters) // 6,
        "parameter_count": len(parameters),
        "initial_mean_reprojection_px": float(
            np.mean(np.linalg.norm(pairwise, axis=1))
        ),
        "loss": robust_loss,
        "loss_scale_px": robust_loss_scale_px,
        "maximum_function_evaluations": maximum_function_evaluations,
    }
    tolerances = {
        "initial_mean_reprojection_px": 5e-6,
        "loss_scale_px": 1e-12,
    }
    checks = {
        key: (
            abs(float(actual[key]) - float(expected))
            <= tolerances.get(key, 0.0)
            if isinstance(expected, float)
            else actual[key] == expected
        )
        for key, expected in HISTORICAL_PRE_SOLVER_TARGETS.items()
    }
    checks["frozen_input_provenance"] = (
        provenance.get("historical_reproduction") is True
        and provenance.get("ground_truth_used") is False
        and provenance.get("validation_status") == "passed"
        and provenance.get("reference_marker_id") == 14
    )
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "schema_version": 1,
        "historical_reproduction": True,
        "ground_truth_used": False,
        "actual": actual,
        "validation_targets": HISTORICAL_PRE_SOLVER_TARGETS,
        "tolerances": tolerances,
        "checks": checks,
        "status": "passed" if not failed else "failed",
        "solver_authorized": not failed,
    }
    evidence = (
        ap02_root
        / "07_graph_ba"
        / "with_moving"
        / "HISTORICAL_PRE_SOLVER_INVARIANTS.json"
    )
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(
            "Historical AP02 pre-solver invariant mismatch: "
            + ", ".join(failed)
        )
    return payload


def distortion_from_row(row: dict) -> tuple[str, np.ndarray]:
    model = str(row.get("distortion_model", "plumb_bob") or "plumb_bob")
    model = model.strip().lower()
    values = []
    for index in range(8):
        value = core.safe_float(row, f"d{index}", 0.0)
        values.append(value if np.isfinite(value) else 0.0)

    if model in {"equidistant", "fisheye"}:
        return "equidistant", np.asarray(values[:4], dtype=np.float64)
    if model in {"", "none", "plumb_bob", "rational_polynomial"}:
        # cv2.projectPoints accepts 4, 5, 8, 12 or 14 coefficients. Eight
        # coefficients preserve ROS plumb_bob/rational-polynomial values.
        return "plumb_bob", np.asarray(values, dtype=np.float64)
    raise RuntimeError(f"Unsupported AP02 distortion model: {model!r}")


def project_camera_point(
    point_camera: np.ndarray,
    K: np.ndarray,
    distortion_model: str,
    distortion: np.ndarray,
) -> np.ndarray | None:
    point_camera = np.asarray(point_camera, dtype=np.float64).reshape(3)
    if point_camera[2] <= 1e-9:
        return None

    if distortion_model == "equidistant":
        projected, _ = cv2.fisheye.projectPoints(
            point_camera.reshape(1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            K,
            distortion.reshape(4, 1),
        )
    else:
        projected, _ = cv2.projectPoints(
            point_camera.reshape(1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            K,
            distortion.reshape(-1, 1),
        )
    return projected.reshape(2).astype(np.float64)


def observation_residuals(row, marker_poses, observer_poses):
    marker_id = int(float(row["marker_id"]))
    observer_id = row["observer_id"]

    T_ref_marker = marker_poses[marker_id]
    T_ref_observer = observer_poses[observer_id]
    T_observer_ref = core.invT(T_ref_observer)

    fx = core.safe_float(row, "fx")
    fy = core.safe_float(row, "fy")
    cx = core.safe_float(row, "cx")
    cy = core.safe_float(row, "cy")
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(K)):
        raise RuntimeError(f"Invalid intrinsics in AP02 observation: {row}")

    distortion_model, distortion = distortion_from_row(row)
    marker_length_m = core.safe_float(row, "marker_length_m", 0.170)
    object_points = core.marker_object_points(marker_length_m)

    residuals = []
    for index, point_marker in enumerate(object_points):
        observed = np.array(
            [
                core.safe_float(row, f"corner{index}_u"),
                core.safe_float(row, f"corner{index}_v"),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(observed)):
            continue

        point_marker_h = np.array(
            [point_marker[0], point_marker[1], point_marker[2], 1.0],
            dtype=np.float64,
        )
        point_ref = T_ref_marker @ point_marker_h
        point_observer = T_observer_ref @ point_ref
        projected = project_camera_point(
            point_observer[:3], K, distortion_model, distortion
        )
        if projected is None:
            residuals.extend([1000.0, 1000.0])
        else:
            residuals.extend((projected - observed).tolist())

    return residuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["static_only", "with_moving"], required=True
    )
    parser.add_argument(
        "--ref-marker-id", type=int, default=core.DEFAULT_REF_MARKER_ID
    )
    parser.add_argument("--max-nfev", type=int, default=50)
    parser.add_argument("--ap02-root", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--initialization-root", type=Path, required=True)
    parser.add_argument(
        "--robust-loss",
        choices=["soft_l1", "huber", "linear"],
        default="soft_l1",
    )
    parser.add_argument("--robust-loss-scale-px", type=float, default=3.0)
    parser.add_argument(
        "--reprojection-model",
        choices=["legacy_pinhole_v1", "distortion_aware_v1"],
        default="legacy_pinhole_v1",
    )
    parser.add_argument(
        "--moving-frame-selection-policy",
        choices=[
            "legacy_smart_at_ba_boundary_v1",
            "all_graph_preselected_frames_v1",
        ],
        default="legacy_smart_at_ba_boundary_v1",
    )
    parser.add_argument("--reference-marker-maximum-frames", type=int)
    parser.add_argument("--top-per-marker", type=int, default=8)
    parser.add_argument("--top-per-marker-pair", type=int, default=4)
    parser.add_argument("--maximum-total-frames", type=int)
    parser.add_argument("--historical-reproduction", action="store_true")
    args = parser.parse_args()
    if args.robust_loss_scale_px <= 0:
        parser.error("--robust-loss-scale-px must be greater than zero")
    ap02_root = args.ap02_root.resolve()
    observations = args.observations.resolve()
    initialization_root = args.initialization_root.resolve()

    scipy_least_squares = core.least_squares
    core_observation_residuals = core.observation_residuals
    if args.reprojection_model == "distortion_aware_v1":
        core.observation_residuals = observation_residuals
    optimizer_report: dict[str, object] = {}
    optimization_summary: dict[str, object] = {}
    optimization_history: list[dict[str, object]] = []

    def robust_cost(residuals: np.ndarray) -> float:
        residuals = np.asarray(residuals, dtype=np.float64)
        if args.robust_loss == "linear":
            return 0.5 * float(np.sum(residuals * residuals))
        scale = float(args.robust_loss_scale_px)
        z = (residuals / scale) ** 2
        if args.robust_loss == "soft_l1":
            rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
        else:  # huber
            rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
        return 0.5 * scale * scale * float(np.sum(rho))

    def configured_least_squares(function, initial, **kwargs):
        kwargs["loss"] = args.robust_loss
        kwargs["f_scale"] = args.robust_loss_scale_px
        started = time.monotonic()
        evaluation_index = 0
        previous_parameters: np.ndarray | None = None

        def evaluate(
            parameters: np.ndarray,
            *,
            source: str,
        ) -> np.ndarray:
            nonlocal evaluation_index, previous_parameters
            residuals = np.asarray(
                function(parameters), dtype=np.float64
            )
            evaluation_index += 1
            pairwise = residuals.reshape(-1, 2)
            norms = np.linalg.norm(pairwise, axis=1)
            step_norm = (
                float(
                    np.linalg.norm(
                        np.asarray(parameters) - previous_parameters
                    )
                )
                if previous_parameters is not None
                else None
            )
            optimization_history.append(
                {
                    "residual_evaluation_index": evaluation_index,
                    "source": source,
                    "solver_iteration": "",
                    "solver_nfev": "",
                    "elapsed_seconds": time.monotonic() - started,
                    "robust_cost": robust_cost(residuals),
                    "reprojection_rmse_px": (
                        float(np.sqrt(np.mean(norms * norms)))
                        if len(norms)
                        else None
                    ),
                    "mean_reprojection_error_px": (
                        float(np.mean(norms)) if len(norms) else None
                    ),
                    "maximum_reprojection_error_px": (
                        float(np.max(norms)) if len(norms) else None
                    ),
                    "parameter_step_norm": step_norm,
                }
            )
            previous_parameters = np.asarray(
                parameters, dtype=np.float64
            ).copy()
            return residuals

        initial_residuals = evaluate(
            np.asarray(initial, dtype=np.float64),
            source="initial_diagnostic",
        )
        if args.historical_reproduction and args.mode == "with_moving":
            validate_historical_pre_solver(
                ap02_root=ap02_root,
                initialization_root=initialization_root,
                reference_marker_id=args.ref_marker_id,
                maximum_function_evaluations=args.max_nfev,
                robust_loss=args.robust_loss,
                robust_loss_scale_px=args.robust_loss_scale_px,
                initial_parameters=np.asarray(initial, dtype=np.float64),
                initial_residuals=initial_residuals,
            )
        result = scipy_least_squares(
            lambda parameters: evaluate(
                parameters, source="solver_residual_call"
            ),
            initial,
            **kwargs,
        )
        initial_cost = robust_cost(initial_residuals)
        final_cost = float(result.cost)
        initial_norms = np.linalg.norm(
            initial_residuals.reshape(-1, 2), axis=1
        )
        final_norms = np.linalg.norm(
            np.asarray(result.fun).reshape(-1, 2), axis=1
        )
        runtime_seconds = time.monotonic() - started
        optimization_summary.update(
            {
                "schema_version": 5,
                "mode": args.mode,
                "solver_success": bool(result.success),
                "solver_status": int(result.status),
                "solver_message": str(result.message),
                "nfev": int(result.nfev),
                "njev": (
                    int(result.njev)
                    if getattr(result, "njev", None) is not None
                    else None
                ),
                "maximum_function_evaluations": int(args.max_nfev),
                "residual_evaluation_calls_recorded": evaluation_index,
                "initial_cost": initial_cost,
                "final_cost": final_cost,
                "absolute_cost_improvement": initial_cost - final_cost,
                "relative_cost_improvement": (
                    (initial_cost - final_cost) / initial_cost
                    if initial_cost > 0
                    else None
                ),
                "initial_reprojection_rmse_px": (
                    float(np.sqrt(np.mean(initial_norms**2)))
                    if len(initial_norms)
                    else None
                ),
                "final_reprojection_rmse_px": (
                    float(np.sqrt(np.mean(final_norms**2)))
                    if len(final_norms)
                    else None
                ),
                "optimality": float(result.optimality),
                "runtime_seconds": runtime_seconds,
                "variable_count": int(len(initial)),
                "scalar_residual_count": int(len(initial_residuals)),
                "corner_residual_count": int(len(initial_residuals) // 2),
                "loss": args.robust_loss,
                "loss_scale_px": float(args.robust_loss_scale_px),
                "reprojection_model": args.reprojection_model,
                "moving_frame_selection_policy": (
                    args.moving_frame_selection_policy
                ),
            }
        )
        optimizer_report.update(
            {
                "mode": args.mode,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "maximum_function_evaluations": int(args.max_nfev),
                "initial_cost": initial_cost,
                "final_cost": final_cost,
                "loss": args.robust_loss,
                "loss_scale_px": float(args.robust_loss_scale_px),
                "runtime_seconds": runtime_seconds,
                "final_reprojection_rmse_px": (
                    optimization_summary[
                        "final_reprojection_rmse_px"
                    ]
                ),
                "optimality": optimization_summary["optimality"],
            }
        )
        return result

    # The numerical core owns residuals, pose parameterization and optimizer
    # invocation. This adapter changes only the two explicitly configurable
    # SciPy loss arguments; defaults remain soft_l1 and 3.0 px.
    core.least_squares = configured_least_squares

    try:
        core.run_ba(
            args.mode,
            args.ref_marker_id,
            args.max_nfev,
            ap02_root=ap02_root,
            observations_csv=observations,
            initialization_root=initialization_root,
            moving_frame_selection_policy=(
                args.moving_frame_selection_policy
            ),
            reference_marker_maximum_frames=(
                args.reference_marker_maximum_frames
            ),
            top_per_marker=args.top_per_marker,
            top_per_marker_pair=args.top_per_marker_pair,
            maximum_total_frames=args.maximum_total_frames,
        )
        report_path = (
            ap02_root
            / "07_graph_ba"
            / args.mode
            / "optimizer_report.json"
        )
        report_path.write_text(
            json.dumps(optimizer_report, indent=2) + "\n",
            encoding="utf-8",
        )
        (report_path.parent / "ap02_optimization_summary.json").write_text(
            json.dumps(
                optimization_summary, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        history_path = (
            report_path.parent / "ap02_optimization_history.csv"
        )
        with history_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(optimization_history[0]),
            )
            writer.writeheader()
            writer.writerows(optimization_history)
    finally:
        core.least_squares = scipy_least_squares
        core.observation_residuals = core_observation_residuals


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


CAMERAS = [
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run existing AP02 graph initialization and distortion-aware BA in an isolated real-data root."
    )
    parser.add_argument("--observations-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref-marker-id", type=int, required=True)
    parser.add_argument("--max-nfev-static", type=int, default=100)
    parser.add_argument("--max-nfev-moving", type=int, default=160)
    parser.add_argument("--top-per-marker", type=int, default=8)
    parser.add_argument("--top-per-pair", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_status(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "METHOD_STATUS.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_main(module, argv: list[str]) -> None:
    previous = sys.argv[:]
    try:
        sys.argv = [str(getattr(module, "__file__", "module")), *argv]
        module.main()
    finally:
        sys.argv = previous


def pose_matrix(row: dict[str, str]) -> np.ndarray:
    rvec = np.asarray([
        float(row["rvec_x"]),
        float(row["rvec_y"]),
        float(row["rvec_z"]),
    ], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [
        float(row["x_m"]),
        float(row["y_m"]),
        float(row["z_m"]),
    ]
    return T


def invT(T: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return result


def marker_points(length_m: float) -> np.ndarray:
    half = length_m / 2.0
    return np.asarray([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def positive_depth_fraction(
    observation_rows: list[dict[str, str]],
    marker_poses: dict[int, np.ndarray],
    observer_poses: dict[str, np.ndarray],
    selected_moving: set[str],
) -> tuple[float, int, int]:
    positive = 0
    total = 0

    for row in observation_rows:
        if str(row.get("pnp_success", "")).strip().lower() not in {"true", "1", "yes"}:
            continue

        observer_id = row["observer_id"]
        if row.get("observer_type") == "moving" and observer_id not in selected_moving:
            continue

        try:
            marker_id = int(float(row["marker_id"]))
        except Exception:
            continue

        if marker_id not in marker_poses or observer_id not in observer_poses:
            continue

        length = float(row.get("marker_length_m", 0.17) or 0.17)
        points = marker_points(length)
        T_ref_marker = marker_poses[marker_id]
        T_observer_ref = invT(observer_poses[observer_id])

        for point in points:
            homogeneous = np.asarray([point[0], point[1], point[2], 1.0])
            camera_point = T_observer_ref @ T_ref_marker @ homogeneous
            positive += int(camera_point[2] > 0)
            total += 1

    fraction = positive / total if total else 0.0
    return fraction, positive, total


def pairwise_rows(poses: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for first, second in combinations(CAMERAS, 2):
        if first not in poses or second not in poses:
            continue
        distance = float(np.linalg.norm(poses[first][:3, 3] - poses[second][:3, 3]))
        rows.append({
            "camera_a": first,
            "camera_b": second,
            "distance_m": distance,
        })
    return rows


def summarize_reprojection(path: Path) -> dict:
    rows = read_csv(path)
    values = []
    maximums = []
    for row in rows:
        try:
            values.append(float(row["mean_reproj_px"]))
            maximums.append(float(row["max_reproj_px"]))
        except Exception:
            pass
    if not values:
        return {
            "observation_count": 0,
            "mean_reprojection_px": None,
            "median_reprojection_px": None,
            "maximum_reprojection_px": None,
        }
    return {
        "observation_count": len(values),
        "mean_reprojection_px": float(np.mean(values)),
        "median_reprojection_px": float(np.median(values)),
        "maximum_reprojection_px": float(np.max(maximums)),
    }


def main() -> None:
    args = parse_args()
    started = time.time()

    observations_root = Path(args.observations_root).resolve()
    out = Path(args.out).resolve()
    repo = Path(__file__).resolve().parents[2]
    ap02_dir = repo / "run" / "bus_real_data" / "approach2_ref_marker_graph_ba"

    try:
        if not ap02_dir.is_dir():
            raise RuntimeError(f"Missing existing AP02 implementation: {ap02_dir}")

        final_pose = (
            out
            / "07_graph_ba"
            / "with_moving"
            / "optimized_static_camera_poses_ref_marker.csv"
        )

        if args.reuse and final_pose.is_file():
            print("[REUSE] AP02 result:", final_pose)
        else:
            shutil.rmtree(out, ignore_errors=True)
            observation_out = out / "02_aruco_observations"
            observation_out.mkdir(parents=True)

            mapping = {
                "shared_static_aruco_observations.csv": "ap02_static_aruco_observations.csv",
                "shared_moving_aruco_observations.csv": "ap02_moving_aruco_observations.csv",
                "shared_all_aruco_observations.csv": "ap02_all_aruco_observations.csv",
            }
            for source_name, destination_name in mapping.items():
                source = observations_root / source_name
                if not source.is_file():
                    raise RuntimeError(f"Missing AP02 source observations: {source}")
                shutil.copy2(source, observation_out / destination_name)

            if str(ap02_dir) not in sys.path:
                sys.path.insert(0, str(ap02_dir))
            bus_run = repo / "run" / "bus_real_data"
            if str(bus_run) not in sys.path:
                sys.path.insert(0, str(bus_run))

            ap02_common = importlib.import_module("ap02_common")
            ap02_common.AP02_ROOT = out
            ap02_common.DEFAULT_REF_MARKER_ID = int(args.ref_marker_id)

            initializer = load_module(
                "ap02_real_initializer",
                ap02_dir / "05_initialize_ref_marker_pose_graph_v2.py",
            )

            for mode in ("static_only", "with_moving"):
                run_main(initializer, [
                    "--mode", mode,
                    "--ref-marker-id", str(args.ref_marker_id),
                    "--out-root", str(out / "05_graph_initialization"),
                ])

            ba_module = load_module(
                "ap02_real_ba_distortion",
                ap02_dir / "07_run_ref_marker_graph_ba_distortion.py",
            )

            run_main(ba_module, [
                "--mode", "static_only",
                "--ref-marker-id", str(args.ref_marker_id),
                "--max-nfev", str(args.max_nfev_static),
                "--moving-selection", "smart",
                "--top-per-marker", str(args.top_per_marker),
                "--top-per-pair", str(args.top_per_pair),
            ])

            run_main(ba_module, [
                "--mode", "with_moving",
                "--ref-marker-id", str(args.ref_marker_id),
                "--max-nfev", str(args.max_nfev_moving),
                "--moving-selection", "smart",
                "--top-per-marker", str(args.top_per_marker),
                "--top-per-pair", str(args.top_per_pair),
                "--max-moving-frames", "0",
            ])

        pose_rows = read_csv(final_pose)
        poses = {
            row["entity_id"]: pose_matrix(row)
            for row in pose_rows
            if row.get("entity_id") in CAMERAS
        }

        marker_pose_path = (
            out / "07_graph_ba" / "with_moving"
            / "optimized_marker_poses_ref_marker.csv"
        )
        moving_pose_path = (
            out / "07_graph_ba" / "with_moving"
            / "optimized_moving_frame_poses_ref_marker.csv"
        )
        marker_poses = {
            int(row["entity_id"]): pose_matrix(row)
            for row in read_csv(marker_pose_path)
        }
        observer_poses = dict(poses)
        observer_poses.update({
            row["entity_id"]: pose_matrix(row)
            for row in read_csv(moving_pose_path)
        })

        selection_path = (
            out / "07_graph_ba" / "with_moving"
            / "moving_frame_selection.csv"
        )
        selected_moving = {
            row["observer_id"]
            for row in read_csv(selection_path)
        }

        all_observations = read_csv(
            out / "02_aruco_observations" / "ap02_all_aruco_observations.csv"
        )
        depth_fraction, positive, total = positive_depth_fraction(
            all_observations,
            marker_poses,
            observer_poses,
            selected_moving,
        )

        reprojection = summarize_reprojection(
            out / "07_graph_ba" / "with_moving"
            / "reprojection_errors_by_observation.csv"
        )

        final_dir = out / "08_final_results"
        final_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            final_dir / "AP02_PAIRWISE_DISTANCES.csv",
            pairwise_rows(poses),
            ["camera_a", "camera_b", "distance_m"],
        )

        positions = np.asarray([T[:3, 3] for T in poses.values()], dtype=np.float64)
        diameter = 0.0
        if len(positions) >= 2:
            diameter = max(
                float(np.linalg.norm(first - second))
                for first, second in combinations(positions, 2)
            )

        diagnostics = {
            "approach": "AP02_reference_marker_graph_distortion_aware_BA",
            "reference_marker_id": args.ref_marker_id,
            "available_static_cameras": sorted(poses),
            "missing_static_cameras": sorted(set(CAMERAS) - set(poses)),
            "optimized_markers": len(marker_poses),
            "optimized_moving_frames": len(observer_poses) - len(poses),
            "selected_moving_frames": len(selected_moving),
            "reprojection": reprojection,
            "positive_depth_fraction": depth_fraction,
            "positive_depth_corners": positive,
            "evaluated_depth_corners": total,
            "estimated_rig_diameter_m": diameter,
            "runtime_seconds": time.time() - started,
            "ground_truth_used": False,
        }
        diagnostics_path = final_dir / "AP02_DIAGNOSTICS.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")

        status = "OK_FULL" if len(poses) == 4 else f"PARTIAL_{len(poses)}_OF_4"
        success = (
            len(poses) == 4
            and depth_fraction >= 0.95
            and reprojection["median_reprojection_px"] is not None
        )

        write_status(out, {
            "method": "AP02",
            "status": status,
            "success": success,
            "available_static_cameras": sorted(poses),
            "runtime_seconds": time.time() - started,
            "pose_file": str(final_pose),
            "diagnostics_file": str(diagnostics_path),
        })

        print("\nAP02 REAL-DATA RESULT")
        print("=" * 72)
        print("status:", status)
        print("static cameras:", sorted(poses))
        print("selected moving frames:", len(selected_moving))
        print("median reprojection [px]:", reprojection["median_reprojection_px"])
        print("positive depth fraction:", depth_fraction)
        print("pose file:", final_pose)

        if len(poses) < 4:
            raise RuntimeError(f"AP02 produced only {len(poses)}/4 static camera poses")

    except Exception as exc:
        failure = {
            "method": "AP02",
            "status": "FAILED",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.time() - started,
        }
        write_status(out, failure)
        print(failure["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

"""AP01 scientific core.

The functions in this module preserve the established marker-direct and
moving-COLMAP-relay mathematics.  The v4 stage modules import these functions
directly; no path mutation or simulated command-line invocation is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
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
ROOT_CAMERA = "cam_edge_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AP01 marker-direct / COLMAP-relay on real data without GT."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--observations-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--use-gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max-image-size", type=int, default=2400)
    parser.add_argument("--reuse-colmap", action="store_true")
    parser.add_argument("--marker-length-m", type=float, default=0.17)
    parser.add_argument("--cameras", default=",".join(CAMERAS))
    parser.add_argument("--root-camera", default=ROOT_CAMERA)
    parser.add_argument("--moving-camera-id", default="moving_calib_camera")
    parser.add_argument("--colmap-executable", default="colmap")
    parser.add_argument("--max-features", type=int, default=8192)
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--loop-detection", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mapper-min-matches", type=int, default=8)
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


def status_path(out: Path) -> Path:
    return out / "METHOD_STATUS.json"


def write_status(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    status_path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def is_success(row: dict[str, str]) -> bool:
    return str(row.get("pnp_success", "")).strip().lower() in {"true", "1", "yes"}


def safe_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except Exception:
        return default


def frame_number(row: dict[str, str]) -> int:
    for key in ("frame_id", "observer_id", "image_path"):
        value = str(row.get(key, ""))
        matches = re.findall(r"(\d+)", value)
        if matches:
            return int(matches[-1])
    raise RuntimeError(f"Cannot infer moving-frame number from row: {row}")


def load_camera_info(path: Path) -> dict:
    data = json.loads(path.read_text())
    flat = data.get("K", data.get("k"))
    if flat is None:
        flat = [
            float(data["fx"]), 0.0, float(data["cx"]),
            0.0, float(data.get("fy", data["fx"])), float(data["cy"]),
            0.0, 0.0, 1.0,
        ]
    K = np.asarray(flat, dtype=np.float64).reshape(3, 3)
    D = np.asarray(data.get("D", data.get("d", [])), dtype=np.float64).reshape(-1)
    return {
        "K": K,
        "D": D,
        "width": int(data.get("width", data.get("image_width", 0)) or 0),
        "height": int(data.get("height", data.get("image_height", 0)) or 0),
        "distortion_model": str(data.get("distortion_model", "plumb_bob")),
        "source": str(path),
    }


def colmap_camera_model(info: dict) -> tuple[str, str]:
    K = info["K"]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    model = info["distortion_model"].strip().lower()
    d = list(float(v) for v in info["D"])
    if model in {"equidistant", "fisheye"}:
        d = (d + [0.0] * 4)[:4]
        params = [fx, fy, cx, cy, *d]
        return "OPENCV_FISHEYE", ",".join(f"{v:.17g}" for v in params)
    if model not in {"", "none", "plumb_bob", "rational_polynomial"}:
        raise RuntimeError(f"Unsupported distortion model: {info['distortion_model']}")
    if not d or max(abs(v) for v in d) <= 1e-15:
        return "PINHOLE", ",".join(f"{v:.17g}" for v in [fx, fy, cx, cy])
    d = (d + [0.0] * 8)[:8]
    params = [fx, fy, cx, cy, *d]
    return "FULL_OPENCV", ",".join(f"{v:.17g}" for v in params)


def run_command(command: list[str]) -> None:
    print("\n[CMD]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def count_colmap_images(images_txt: Path) -> int:
    return len(parse_colmap_poses(images_txt))


def run_colmap(
    image_dir: Path,
    camera_info: dict,
    out_dir: Path,
    matcher: str,
    use_gpu: int,
    max_image_size: int,
    max_features: int,
    sequential_overlap: int,
    loop_detection: int,
    mapper_min_matches: int,
    colmap_executable: str,
    reuse: bool,
) -> Path:
    best = out_dir / "sparse_txt_best" / "images.txt"
    if reuse and best.is_file():
        print("[REUSE] AP01 moving COLMAP reconstruction:", best)
        return best

    executable = shutil.which(colmap_executable)
    if executable is None:
        raise RuntimeError(f"COLMAP executable not found: {colmap_executable}")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    database = out_dir / "database.db"
    sparse = out_dir / "sparse"
    sparse_txt = out_dir / "sparse_txt"
    sparse.mkdir()
    sparse_txt.mkdir()

    model, params = colmap_camera_model(camera_info)

    run_command([
        executable, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", model,
        "--ImageReader.camera_params", params,
        "--SiftExtraction.use_gpu", str(use_gpu),
        "--SiftExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.max_num_features", str(max_features),
    ])

    if matcher == "exhaustive":
        run_command([
            executable, "exhaustive_matcher",
            "--database_path", str(database),
            "--SiftMatching.use_gpu", str(use_gpu),
        ])
    else:
        run_command([
            executable, "sequential_matcher",
            "--database_path", str(database),
            "--SiftMatching.use_gpu", str(use_gpu),
            "--SequentialMatching.overlap", str(sequential_overlap),
            "--SequentialMatching.loop_detection", str(loop_detection),
        ])

    run_command([
        executable, "mapper",
        "--database_path", str(database),
        "--image_path", str(image_dir),
        "--output_path", str(sparse),
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.min_num_matches", str(mapper_min_matches),
    ])

    model_dirs = sorted(path for path in sparse.iterdir() if path.is_dir())
    if not model_dirs:
        raise RuntimeError("AP01 COLMAP mapper produced no sparse model")

    best_dir = None
    best_count = -1
    model_counts = []

    for model_dir in model_dirs:
        text_dir = sparse_txt / model_dir.name
        text_dir.mkdir(parents=True, exist_ok=True)
        run_command([
            executable, "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(text_dir),
            "--output_type", "TXT",
        ])
        count = count_colmap_images(text_dir / "images.txt")
        model_counts.append({"model": model_dir.name, "registered_images": count})
        if count > best_count:
            best_count = count
            best_dir = text_dir

    if best_dir is None:
        raise RuntimeError("Could not select AP01 COLMAP model")

    destination = out_dir / "sparse_txt_best"
    shutil.copytree(best_dir, destination)

    write_csv(out_dir / "model_counts.csv", model_counts)
    (out_dir / "COLMAP_REPORT.txt").write_text(
        "\n".join([
            "AP01 MOVING-CAMERA COLMAP",
            "=" * 72,
            "",
            f"Image directory: {image_dir}",
            f"Matcher: {matcher}",
            f"Camera model: {model}",
            f"Camera parameters: {params}",
            f"Feature max image size: {max_image_size}",
            f"Maximum features: {max_features}",
            f"Sequential overlap: {sequential_overlap}",
            f"Loop detection: {loop_detection}",
            f"Mapper minimum matches: {mapper_min_matches}",
            f"Registered images in best model: {best_count}",
            f"Best model: {best_dir}",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    return destination / "images.txt"


def qvec_to_R(values: list[float]) -> np.ndarray:
    qw, qx, qy, qz = values
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-15)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return result


def T_from_observation(row: dict[str, str]) -> np.ndarray:
    rvec = np.asarray([
        safe_float(row, "rvec_x"),
        safe_float(row, "rvec_y"),
        safe_float(row, "rvec_z"),
    ], dtype=np.float64)
    tvec = np.asarray([
        safe_float(row, "tvec_x_m"),
        safe_float(row, "tvec_y_m"),
        safe_float(row, "tvec_z_m"),
    ], dtype=np.float64)
    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        raise RuntimeError("Non-finite ArUco PnP pose")
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec)


def parse_colmap_poses(images_txt: Path) -> dict[int, np.ndarray]:
    if not images_txt.is_file():
        raise RuntimeError(f"Missing COLMAP images.txt: {images_txt}")
    result = {}
    for raw in images_txt.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10 or not parts[9].lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        matches = re.findall(r"(\d+)", Path(parts[9]).stem)
        if not matches:
            continue
        frame = int(matches[-1])
        R = qvec_to_R([float(v) for v in parts[1:5]])
        t = np.asarray([float(v) for v in parts[5:8]], dtype=np.float64)
        result[frame] = make_T(R, t)  # world -> camera
    if not result:
        raise RuntimeError(f"No AP01 moving poses parsed from {images_txt}")
    return result


def observation_quality(row: dict[str, str], width: float, height: float) -> float:
    area = max(safe_float(row, "area_px2", 0.0), 1.0)
    distance = max(safe_float(row, "distance_m", 99.0), 0.1)
    center_u = safe_float(row, "center_u", width / 2.0)
    center_v = safe_float(row, "center_v", height / 2.0)
    center_norm = math.hypot(center_u - width / 2.0, center_v - height / 2.0)
    center_norm /= max(math.hypot(width / 2.0, height / 2.0), 1.0)
    return math.sqrt(area) / (distance * (1.0 + center_norm))


def prepare_observations(
    static_rows: list[dict[str, str]],
    moving_rows: list[dict[str, str]],
    static_size: tuple[int, int],
    moving_size: tuple[int, int],
) -> tuple[list[dict], list[dict]]:
    prepared_static = []
    for row in static_rows:
        if not is_success(row):
            continue
        item = dict(row)
        item["_marker"] = int(float(row["marker_id"]))
        item["_camera"] = row["camera_name"]
        item["_quality"] = safe_float(
            row,
            "selection_score",
            observation_quality(row, *static_size),
        )
        item["_T_cam_marker"] = T_from_observation(row)
        prepared_static.append(item)

    prepared_moving = []
    for row in moving_rows:
        if not is_success(row):
            continue
        item = dict(row)
        item["_marker"] = int(float(row["marker_id"]))
        item["_frame"] = frame_number(row)
        item["_quality"] = safe_float(
            row,
            "selection_score",
            observation_quality(row, *moving_size),
        )
        item["_T_cam_marker"] = T_from_observation(row)
        prepared_moving.append(item)

    return prepared_static, prepared_moving


def robust_scale(
    moving_rows: list[dict],
    colmap_poses: dict[int, np.ndarray],
    maximum_observations_per_marker: int | None = None,
) -> tuple[float, dict, list[dict]]:
    by_marker = defaultdict(list)
    for row in moving_rows:
        if row["_frame"] in colmap_poses:
            by_marker[row["_marker"]].append(row)

    registered_counts = {
        int(marker): len(rows) for marker, rows in sorted(by_marker.items())
    }
    for marker, rows in by_marker.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                -float(row["_quality"]),
                int(row["_frame"]),
            ),
        )
        if maximum_observations_per_marker is not None:
            ranked = ranked[:maximum_observations_per_marker]
        by_marker[marker] = ranked
    selected_counts = {
        int(marker): len(rows) for marker, rows in sorted(by_marker.items())
    }

    pairs = []
    for marker, rows in by_marker.items():
        rows = sorted(rows, key=lambda r: r["_frame"])
        for first, second in combinations(rows, 2):
            gap = abs(first["_frame"] - second["_frame"])
            if gap < 2 or gap > 80:
                continue

            T_metric = first["_T_cam_marker"] @ invT(second["_T_cam_marker"])
            metric_distance = float(np.linalg.norm(T_metric[:3, 3]))
            if not (0.05 <= metric_distance <= 6.0):
                continue

            T_colmap = colmap_poses[first["_frame"]] @ invT(colmap_poses[second["_frame"]])
            colmap_distance = float(np.linalg.norm(T_colmap[:3, 3]))
            if colmap_distance <= 1e-10:
                continue

            ratio = metric_distance / colmap_distance
            if not math.isfinite(ratio) or ratio <= 0:
                continue

            pairs.append({
                "marker_id": marker,
                "frame_i": first["_frame"],
                "frame_j": second["_frame"],
                "frame_gap": gap,
                "metric_translation_m": metric_distance,
                "colmap_translation_units": colmap_distance,
                "scale_m_per_colmap_unit": ratio,
                "quality": math.sqrt(first["_quality"] * second["_quality"]),
            })

    if len(pairs) < 10:
        raise RuntimeError(f"Too few AP01 metric-scale pairs: {len(pairs)}")

    values = np.asarray([row["scale_m_per_colmap_unit"] for row in pairs], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    threshold = max(3.0 * sigma, 0.10 * median)
    kept = [row for row in pairs if abs(row["scale_m_per_colmap_unit"] - median) <= threshold]
    if len(kept) < max(10, int(0.25 * len(pairs))):
        kept = pairs
    kept_values = np.asarray([row["scale_m_per_colmap_unit"] for row in kept], dtype=np.float64)
    scale = float(np.median(kept_values))
    kept_ids = {id(row) for row in kept}
    for row in pairs:
        row["used_for_scale"] = id(row) in kept_ids

    marker_pair_counts = Counter(
        int(row["marker_id"]) for row in pairs
    )
    marker_inlier_counts = Counter(
        int(row["marker_id"]) for row in kept
    )
    stats = {
        "scale_m_per_colmap_unit": scale,
        "raw_pairs": len(pairs),
        "used_pairs": len(kept),
        "raw_median": median,
        "raw_mad": mad,
        "used_mean": float(np.mean(kept_values)),
        "used_std": float(np.std(kept_values)),
        "used_relative_std": float(np.std(kept_values) / scale),
        "markers_with_registered_observations": sorted(by_marker),
        "maximum_observations_per_marker": maximum_observations_per_marker,
        "registered_observations_per_marker": registered_counts,
        "selected_observations_per_marker": selected_counts,
        "candidate_pairs_per_marker": dict(sorted(marker_pair_counts.items())),
        "inlier_pairs_per_marker": dict(sorted(marker_inlier_counts.items())),
        "outlier_pairs_per_marker": {
            marker: marker_pair_counts[marker] - marker_inlier_counts[marker]
            for marker in sorted(marker_pair_counts)
        },
    }
    return scale, stats, pairs


def R_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1e-15, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        q = np.array([
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1e-15, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        q = np.array([
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        ])
    else:
        s = math.sqrt(max(1e-15, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        q = np.array([
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ])
    q /= max(float(np.linalg.norm(q)), 1e-15)
    return q


def quat_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-15)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def weighted_rotation_mean(rotations: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    quaternions = [R_to_quat(R) for R in rotations]
    reference = quaternions[0]
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for quaternion, weight in zip(quaternions, weights):
        if float(np.dot(quaternion, reference)) < 0:
            quaternion = -quaternion
        accumulator += float(weight) * np.outer(quaternion, quaternion)
    _, vectors = np.linalg.eigh(accumulator)
    quaternion = vectors[:, -1]
    if quaternion[0] < 0:
        quaternion = -quaternion
    return quat_to_R(quaternion)


def rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    value = max(-1.0, min(1.0, float((np.trace(relative) - 1.0) / 2.0)))
    return math.degrees(math.acos(value))


def aggregate_candidates(candidates: list[dict], translation_floor: float = 0.20, rotation_floor: float = 5.0) -> tuple[np.ndarray, dict]:
    if not candidates:
        raise RuntimeError("No AP01 transform candidates")

    translations = np.asarray([row["T"][:3, 3] for row in candidates], dtype=np.float64)
    weights = np.asarray([max(float(row["quality"]), 1e-12) for row in candidates], dtype=np.float64)
    weights /= weights.sum()

    initial_translation = np.median(translations, axis=0)
    initial_rotation = weighted_rotation_mean([row["T"][:3, :3] for row in candidates], weights)

    translation_deviation = np.linalg.norm(translations - initial_translation[None, :], axis=1)
    rotation_deviation = np.asarray(
        [rotation_difference_deg(initial_rotation, row["T"][:3, :3]) for row in candidates],
        dtype=np.float64,
    )

    t_median = float(np.median(translation_deviation))
    r_median = float(np.median(rotation_deviation))
    t_mad = 1.4826 * float(np.median(np.abs(translation_deviation - t_median)))
    r_mad = 1.4826 * float(np.median(np.abs(rotation_deviation - r_median)))

    t_threshold = max(translation_floor, t_median + 3.0 * t_mad)
    r_threshold = max(rotation_floor, r_median + 3.0 * r_mad)

    robust_inlier_indices = [
        index
        for index, (t_dev, r_dev) in enumerate(zip(translation_deviation, rotation_deviation))
        if t_dev <= t_threshold and r_dev <= r_threshold
    ]
    inlier_indices = list(robust_inlier_indices)
    pose_fallback_used = False
    if len(inlier_indices) < min(3, len(candidates)):
        pose_fallback_used = True
        inlier_indices = list(np.argsort(weights)[::-1][:max(1, min(3, len(candidates)))])

    inlier_weights = weights[inlier_indices]
    inlier_weights /= inlier_weights.sum()
    translation = np.sum(translations[inlier_indices] * inlier_weights[:, None], axis=0)
    rotation = weighted_rotation_mean(
        [candidates[index]["T"][:3, :3] for index in inlier_indices],
        inlier_weights,
    )
    transform = make_T(rotation, translation)

    robust_inlier_set = set(robust_inlier_indices)
    pose_support_set = set(inlier_indices)
    final_translation_deviation = np.linalg.norm(
        translations - translation[None, :], axis=1
    )
    final_rotation_deviation = np.asarray(
        [
            rotation_difference_deg(rotation, row["T"][:3, :3])
            for row in candidates
        ],
        dtype=np.float64,
    )
    robust_translation_deviation = np.asarray([], dtype=np.float64)
    robust_rotation_deviation = np.asarray([], dtype=np.float64)
    if robust_inlier_indices:
        robust_weights = weights[robust_inlier_indices]
        robust_weights /= robust_weights.sum()
        robust_translation = np.sum(
            translations[robust_inlier_indices]
            * robust_weights[:, None],
            axis=0,
        )
        robust_rotation = weighted_rotation_mean(
            [
                candidates[index]["T"][:3, :3]
                for index in robust_inlier_indices
            ],
            robust_weights,
        )
        robust_translation_deviation = np.linalg.norm(
            translations[robust_inlier_indices]
            - robust_translation[None, :],
            axis=1,
        )
        robust_rotation_deviation = np.asarray(
            [
                rotation_difference_deg(
                    robust_rotation,
                    candidates[index]["T"][:3, :3],
                )
                for index in robust_inlier_indices
            ],
            dtype=np.float64,
        )
    for index, row in enumerate(candidates):
        row["translation_deviation_m"] = float(
            final_translation_deviation[index]
        )
        row["rotation_deviation_deg"] = float(
            final_rotation_deviation[index]
        )
        row["inlier"] = index in robust_inlier_set
        row["pose_support"] = index in pose_support_set

    stats = {
        "candidates": len(candidates),
        "inliers": len(robust_inlier_indices),
        "robust_inliers": len(robust_inlier_indices),
        "pose_support_count": len(inlier_indices),
        "pose_fallback_used": pose_fallback_used,
        "inlier_ratio": (
            len(robust_inlier_indices) / len(candidates)
            if candidates
            else 0.0
        ),
        "maximum_inlier_translation_dispersion_m": (
            float(np.max(robust_translation_deviation))
            if robust_inlier_indices
            else None
        ),
        "maximum_inlier_rotation_dispersion_deg": (
            float(np.max(robust_rotation_deviation))
            if robust_inlier_indices
            else None
        ),
        "translation_deviation_median_m": t_median,
        "rotation_deviation_median_deg": r_median,
        "translation_threshold_m": t_threshold,
        "rotation_threshold_deg": r_threshold,
    }
    return transform, stats


def best_static_by_camera_marker(rows: list[dict]) -> dict[tuple[str, int], dict]:
    result = {}
    for row in rows:
        key = (row["_camera"], row["_marker"])
        if key not in result or row["_quality"] > result[key]["_quality"]:
            result[key] = row
    return result


def moving_by_marker(
    rows: list[dict],
    registered_frames: set[int],
    top_per_marker: int | None = None,
) -> dict[int, list[dict]]:
    """Rank registered moving observations and apply the AP01 relay cap."""
    grouped = defaultdict(list)
    for row in rows:
        if row["_frame"] in registered_frames:
            grouped[row["_marker"]].append(row)
    for marker in grouped:
        ranked = sorted(
            grouped[marker],
            key=lambda row: (
                -float(row["_quality"]),
                int(row["_frame"]),
            ),
        )
        if top_per_marker is not None:
            ranked = ranked[:top_per_marker]
        grouped[marker] = sorted(
            ranked, key=lambda row: int(row["_frame"])
        )
    return grouped


def direct_candidates(
    root: str,
    target: str,
    static_best: dict[tuple[str, int], dict],
) -> list[dict]:
    root_markers = {marker for camera, marker in static_best if camera == root}
    target_markers = {marker for camera, marker in static_best if camera == target}
    result = []
    for marker in sorted(root_markers & target_markers):
        root_row = static_best[(root, marker)]
        target_row = static_best[(target, marker)]
        transform = root_row["_T_cam_marker"] @ invT(target_row["_T_cam_marker"])
        result.append({
            "mode": "direct",
            "root_camera": root,
            "target_camera": target,
            "root_marker": marker,
            "target_marker": marker,
            "root_frame": "",
            "target_frame": "",
            "quality": math.sqrt(root_row["_quality"] * target_row["_quality"]),
            "T": transform,
        })
    return result


def relay_candidates(
    root: str,
    target: str,
    static_best: dict[tuple[str, int], dict],
    moving_by_marker: dict[int, list[dict]],
    colmap_poses: dict[int, np.ndarray],
    scale: float,
) -> list[dict]:
    root_markers = sorted(
        marker
        for camera, marker in static_best
        if camera == root and marker in moving_by_marker
    )
    target_markers = sorted(
        marker
        for camera, marker in static_best
        if camera == target and marker in moving_by_marker
    )

    result = []
    for root_marker in root_markers:
        root_static = static_best[(root, root_marker)]
        T_root_marker = root_static["_T_cam_marker"]
        for target_marker in target_markers:
            target_static = static_best[(target, target_marker)]
            T_target_marker = target_static["_T_cam_marker"]

            for root_moving in moving_by_marker[root_marker]:
                frame_i = root_moving["_frame"]
                T_root_moving_i = T_root_marker @ invT(root_moving["_T_cam_marker"])

                for target_moving in moving_by_marker[target_marker]:
                    frame_j = target_moving["_frame"]
                    if frame_i == frame_j and root_marker == target_marker:
                        continue

                    T_target_moving_j = T_target_marker @ invT(target_moving["_T_cam_marker"])
                    T_moving_i_moving_j = colmap_poses[frame_i] @ invT(colmap_poses[frame_j])
                    T_moving_i_moving_j = T_moving_i_moving_j.copy()
                    T_moving_i_moving_j[:3, 3] *= scale

                    transform = (
                        T_root_moving_i
                        @ T_moving_i_moving_j
                        @ invT(T_target_moving_j)
                    )
                    quality = (
                        root_static["_quality"]
                        * target_static["_quality"]
                        * root_moving["_quality"]
                        * target_moving["_quality"]
                    ) ** 0.25
                    result.append({
                        "mode": "relay",
                        "root_camera": root,
                        "target_camera": target,
                        "root_marker": root_marker,
                        "target_marker": target_marker,
                        "root_frame": frame_i,
                        "target_frame": frame_j,
                        "quality": quality,
                        "T": transform,
                    })
    return result


def R_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def pose_row(camera: str, T: np.ndarray, source: str) -> dict:
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    roll, pitch, yaw = R_to_rpy_deg(T[:3, :3])
    return {
        "entity_type": "static_camera",
        "entity_id": camera,
        "source": source,
        "x_m": float(T[0, 3]),
        "y_m": float(T[1, 3]),
        "z_m": float(T[2, 3]),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "rvec_x": float(rvec[0]),
        "rvec_y": float(rvec[1]),
        "rvec_z": float(rvec[2]),
    }


def serializable_candidate(row: dict) -> dict:
    result = {key: value for key, value in row.items() if key != "T"}
    T = row["T"]
    result.update({
        "x_m": float(T[0, 3]),
        "y_m": float(T[1, 3]),
        "z_m": float(T[2, 3]),
    })
    return result


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


def main() -> None:
    global CAMERAS, ROOT_CAMERA
    args = parse_args()
    CAMERAS = [value.strip() for value in args.cameras.split(",") if value.strip()]
    ROOT_CAMERA = args.root_camera.strip()
    if not CAMERAS:
        raise RuntimeError("--cameras must contain at least one camera ID")
    if ROOT_CAMERA not in CAMERAS:
        raise RuntimeError(f"Root camera '{ROOT_CAMERA}' is not in --cameras")
    started = time.time()

    dataset = Path(args.dataset).resolve()
    observations_root = Path(args.observations_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        moving_dir = dataset / "raw_images" / "moving"
        moving_info_path = (
            dataset
            / "raw_images"
            / "camera_info"
            / f"{args.moving_camera_id}.json"
        )
        static_csv = observations_root / "shared_static_aruco_observations.csv"
        moving_csv = observations_root / "shared_moving_aruco_observations.csv"

        if not moving_dir.is_dir():
            raise RuntimeError(f"Missing moving images: {moving_dir}")
        if not moving_info_path.is_file():
            raise RuntimeError(f"Missing moving camera info: {moving_info_path}")

        moving_info = load_camera_info(moving_info_path)
        static_info = load_camera_info(
            dataset / "raw_images" / "camera_info" / f"{ROOT_CAMERA}.json"
        )

        images_txt = run_colmap(
            image_dir=moving_dir,
            camera_info=moving_info,
            out_dir=out / "01_moving_colmap",
            matcher=args.matcher,
            use_gpu=args.use_gpu,
            max_image_size=args.max_image_size,
            max_features=args.max_features,
            sequential_overlap=args.sequential_overlap,
            loop_detection=args.loop_detection,
            mapper_min_matches=args.mapper_min_matches,
            colmap_executable=args.colmap_executable,
            reuse=args.reuse_colmap,
        )

        colmap_poses = parse_colmap_poses(images_txt)
        static_rows_raw = read_csv(static_csv)
        moving_rows_raw = read_csv(moving_csv)

        static_rows, moving_rows = prepare_observations(
            static_rows_raw,
            moving_rows_raw,
            (static_info["width"], static_info["height"]),
            (moving_info["width"], moving_info["height"]),
        )

        scale, scale_stats, scale_pairs = robust_scale(
            moving_rows,
            colmap_poses,
        )

        scale_dir = out / "02_metric_scale"
        scale_dir.mkdir(parents=True, exist_ok=True)
        (scale_dir / "metric_scale.txt").write_text(f"{scale:.12g}\n")
        (scale_dir / "SCALE_DIAGNOSTICS.json").write_text(
            json.dumps(scale_stats, indent=2) + "\n"
        )
        write_csv(scale_dir / "scale_pairs.csv", scale_pairs)

        static_best = best_static_by_camera_marker(static_rows)
        moving_by_marker_rows = moving_by_marker(
            moving_rows,
            set(colmap_poses),
        )

        poses = {ROOT_CAMERA: np.eye(4, dtype=np.float64)}
        method_by_camera = {ROOT_CAMERA: "gauge_identity"}
        method_diagnostics = {}
        all_candidate_rows = []

        for target in CAMERAS:
            if target == ROOT_CAMERA:
                continue

            direct = direct_candidates(ROOT_CAMERA, target, static_best)
            relay = relay_candidates(
                ROOT_CAMERA,
                target,
                static_best,
                moving_by_marker_rows,
                colmap_poses,
                scale,
            )

            direct_transform = None
            direct_stats = None
            if direct:
                direct_transform, direct_stats = aggregate_candidates(
                    direct,
                    translation_floor=0.12,
                    rotation_floor=4.0,
                )

            relay_transform = None
            relay_stats = None
            if relay:
                relay_transform, relay_stats = aggregate_candidates(
                    relay,
                    translation_floor=0.30,
                    rotation_floor=7.0,
                )

            if len(direct) >= 2:
                selected = direct_transform
                selected_method = "direct_multimarker"
            elif relay_transform is not None:
                selected = relay_transform
                selected_method = "moving_colmap_relay"
            elif direct_transform is not None:
                selected = direct_transform
                selected_method = "direct_single_marker"
            else:
                selected = None
                selected_method = "unavailable"

            if selected is not None:
                poses[target] = selected
                method_by_camera[target] = selected_method

            method_diagnostics[target] = {
                "selected_method": selected_method,
                "direct_common_markers": sorted(
                    {int(row["root_marker"]) for row in direct}
                ),
                "direct": direct_stats,
                "relay": relay_stats,
                "relay_candidates": len(relay),
            }

            all_candidate_rows.extend(serializable_candidate(row) for row in direct)
            all_candidate_rows.extend(serializable_candidate(row) for row in relay)

        final_dir = out / "03_static_extrinsics"
        final_dir.mkdir(parents=True, exist_ok=True)

        pose_rows = [
            pose_row(camera, transform, method_by_camera[camera])
            for camera, transform in sorted(poses.items())
        ]
        pose_fields = [
            "entity_type", "entity_id", "source",
            "x_m", "y_m", "z_m",
            "roll_deg", "pitch_deg", "yaw_deg",
            "rvec_x", "rvec_y", "rvec_z",
        ]
        generic_pose_file = (
            final_dir / "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
        )
        write_csv(generic_pose_file, pose_rows, pose_fields)
        if ROOT_CAMERA == "cam_edge_3":
            write_csv(
                final_dir / "AP01_STATIC_CAMERA_POSES_CAM3_REFERENCE.csv",
                pose_rows,
                pose_fields,
            )
        write_csv(
            final_dir / "AP01_PAIRWISE_DISTANCES.csv",
            pairwise_rows(poses),
            ["camera_a", "camera_b", "distance_m"],
        )
        write_csv(final_dir / "AP01_TRANSFORM_CANDIDATES.csv", all_candidate_rows)

        diagnostics = {
            "approach": "AP01_marker_direct_and_moving_colmap_relay",
            "root_camera": ROOT_CAMERA,
            "registered_moving_frames": len(colmap_poses),
            "input_moving_frames": len(list(moving_dir.glob("frame_*.png"))),
            "metric_scale": scale_stats,
            "static_camera_methods": method_by_camera,
            "per_target_diagnostics": method_diagnostics,
            "available_static_cameras": sorted(poses),
            "missing_static_cameras": sorted(set(CAMERAS) - set(poses)),
            "runtime_seconds": time.time() - started,
            "ground_truth_used": False,
        }
        (final_dir / "AP01_DIAGNOSTICS.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )

        expected_count = len(CAMERAS)
        status = (
            "OK_FULL"
            if len(poses) == expected_count
            else f"PARTIAL_{len(poses)}_OF_{expected_count}"
        )
        write_status(out, {
            "method": "AP01",
            "status": status,
            "success": len(poses) == expected_count,
            "available_static_cameras": sorted(poses),
            "runtime_seconds": time.time() - started,
            "pose_file": str(generic_pose_file),
            "diagnostics_file": str(final_dir / "AP01_DIAGNOSTICS.json"),
        })

        print("\nAP01 REAL-DATA RESULT")
        print("=" * 72)
        print("status:", status)
        print("registered moving frames:", len(colmap_poses))
        print("metric scale:", scale)
        print("camera methods:", method_by_camera)
        print("pose file:", generic_pose_file)

        if len(poses) < expected_count:
            raise RuntimeError(
                f"AP01 produced only {len(poses)}/{expected_count} static camera poses"
            )

    except Exception as exc:
        failure = {
            "method": "AP01",
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

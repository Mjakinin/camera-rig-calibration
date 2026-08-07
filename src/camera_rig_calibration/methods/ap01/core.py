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

from .contracts import AP01MethodContract, resolve_ap01_method_contract


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


def colmap_camera_model(
    info: dict,
    contract: AP01MethodContract | None = None,
) -> tuple[str, str]:
    contract = contract or resolve_ap01_method_contract()
    K = info["K"]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    model = info["distortion_model"].strip().lower()
    d = list(float(v) for v in info["D"])
    if contract.colmap_camera_model_policy == "legacy_shared_pinhole_v1":
        values = [fx, fy, cx, cy]
        return "PINHOLE", ",".join(
            f"{value:.{contract.colmap_intrinsics_precision}f}"
            for value in values
        )
    if contract.colmap_camera_model_policy != "camera_info_distortion_model_v1":
        raise ValueError(
            "Unknown AP01 COLMAP camera-model policy: "
            f"{contract.colmap_camera_model_policy}"
        )

    def serialize(values: list[float]) -> str:
        if contract.colmap_intrinsics_serialization == "significant_digits":
            return ",".join(
                f"{value:.{contract.colmap_intrinsics_precision}g}"
                for value in values
            )
        if contract.colmap_intrinsics_serialization == "fixed_decimal_places":
            return ",".join(
                f"{value:.{contract.colmap_intrinsics_precision}f}"
                for value in values
            )
        raise ValueError(
            "Unknown AP01 intrinsics serialization: "
            f"{contract.colmap_intrinsics_serialization}"
        )

    if model in {"equidistant", "fisheye"}:
        d = (d + [0.0] * 4)[:4]
        params = [fx, fy, cx, cy, *d]
        return "OPENCV_FISHEYE", serialize(params)
    if model not in {"", "none", "plumb_bob", "rational_polynomial"}:
        raise RuntimeError(f"Unsupported distortion model: {info['distortion_model']}")
    if not d or max(abs(v) for v in d) <= 1e-15:
        return "PINHOLE", serialize([fx, fy, cx, cy])
    d = (d + [0.0] * 8)[:8]
    params = [fx, fy, cx, cy, *d]
    return "FULL_OPENCV", serialize(params)


def run_command(command: list[str]) -> None:
    print("\n[CMD]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def colmap_feature_extractor_command(
    *,
    executable: str,
    database: Path,
    image_dir: Path,
    camera_model: str,
    camera_parameters: str,
    contract: AP01MethodContract,
) -> list[str]:
    command = [
        executable,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(image_dir),
        "--ImageReader.single_camera",
        "1" if contract.colmap_single_shared_camera else "0",
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.camera_params",
        camera_parameters,
        "--SiftExtraction.use_gpu",
        "1" if contract.colmap_matcher_use_gpu else "0",
        "--SiftExtraction.max_image_size",
        str(contract.colmap_sift_maximum_image_size),
        "--SiftExtraction.max_num_features",
        str(contract.colmap_sift_max_features),
    ]
    if contract.colmap_sift_extraction_threads is not None:
        command.extend(
            [
                "--SiftExtraction.num_threads",
                str(contract.colmap_sift_extraction_threads),
            ]
        )
    return command


def colmap_mapper_command(
    *,
    executable: str,
    database: Path,
    image_dir: Path,
    sparse: Path,
    contract: AP01MethodContract,
) -> list[str]:
    return [
        executable,
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse),
        "--Mapper.ba_refine_focal_length",
        "1" if contract.colmap_refine_focal_length else "0",
        "--Mapper.ba_refine_principal_point",
        "1" if contract.colmap_refine_principal_point else "0",
        "--Mapper.ba_refine_extra_params",
        "1" if contract.colmap_refine_extra_parameters else "0",
        "--Mapper.min_num_matches",
        str(contract.colmap_mapper_minimum_matches),
    ]


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
    contract: AP01MethodContract | None = None,
) -> Path:
    contract = contract or resolve_ap01_method_contract(
        colmap_matcher=matcher,
        colmap_use_gpu=bool(use_gpu),
        colmap_maximum_image_size=max_image_size,
        colmap_maximum_features=max_features,
        colmap_sequential_overlap=sequential_overlap,
        colmap_loop_detection=bool(loop_detection),
        colmap_mapper_minimum_matches=mapper_min_matches,
    )
    matcher = contract.colmap_matching_mode
    use_gpu = int(contract.colmap_matcher_use_gpu)
    max_image_size = contract.colmap_sift_maximum_image_size
    max_features = contract.colmap_sift_max_features
    sequential_overlap = contract.colmap_sequential_overlap
    loop_detection = int(contract.colmap_loop_detection)
    mapper_min_matches = contract.colmap_mapper_minimum_matches
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

    model, params = colmap_camera_model(camera_info, contract)

    run_command(
        colmap_feature_extractor_command(
            executable=executable,
            database=database,
            image_dir=image_dir,
            camera_model=model,
            camera_parameters=params,
            contract=contract,
        )
    )

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

    run_command(
        colmap_mapper_command(
            executable=executable,
            database=database,
            image_dir=image_dir,
            sparse=sparse,
            contract=contract,
        )
    )

    if contract.colmap_sparse_model_selection_policy != (
        "maximum_registered_images_first_lexicographic_tie"
    ):
        raise ValueError(
            "Unknown AP01 sparse-model selection policy: "
            f"{contract.colmap_sparse_model_selection_policy}"
        )
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
            "SIFT extraction threads: "
            f"{contract.colmap_sift_extraction_threads}",
            f"Sequential overlap: {sequential_overlap}",
            f"Loop detection: {loop_detection}",
            f"Mapper minimum matches: {mapper_min_matches}",
            f"Registered images in best model: {best_count}",
            f"Best model: {best_dir}",
            f"Method contract: {contract.name}",
            f"Method contract SHA-256: {contract.scientific_fingerprint()}",
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


def marker_area_from_corners(row: dict[str, str]) -> float:
    points = np.asarray(
        [
            [
                safe_float(row, f"corner{index}_u"),
                safe_float(row, f"corner{index}_v"),
            ]
            for index in range(4)
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(points)):
        return float("nan")
    x = points[:, 0]
    y = points[:, 1]
    return float(
        0.5
        * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def legacy_detection_quality(
    row: dict[str, str], width: float = 1280.0, height: float = 720.0
) -> tuple[float, dict[str, float]]:
    """Legacy Main's exact area/(distance^2*(1+center_norm)) score."""

    distance = safe_float(row, "distance_m", 99.0)
    area = marker_area_from_corners(row)
    center_u = safe_float(row, "center_u")
    center_v = safe_float(row, "center_v")
    if math.isfinite(center_u) and math.isfinite(center_v):
        center_norm = math.hypot(
            center_u - width / 2.0, center_v - height / 2.0
        ) / max(math.hypot(width / 2.0, height / 2.0), 1.0)
    else:
        center_norm = 1.0
    if not math.isfinite(distance) or distance <= 0.0:
        distance = 99.0
    if not math.isfinite(area) or area <= 0.0:
        area = 1.0
    if not math.isfinite(center_norm):
        center_norm = 1.0
    return area / (distance * distance * (1.0 + center_norm)), {
        "area_px2_from_corners": area,
        "distance_m": distance,
        "center_norm": center_norm,
        "quality_image_width_px": float(width),
        "quality_image_height_px": float(height),
    }


def prepare_observations(
    static_rows: list[dict[str, str]],
    moving_rows: list[dict[str, str]],
    static_size: tuple[int, int],
    moving_size: tuple[int, int],
    *,
    contract: AP01MethodContract | None = None,
) -> tuple[list[dict], list[dict]]:
    contract = contract or resolve_ap01_method_contract()

    def prepare_quality(
        row: dict[str, str], image_size: tuple[int, int]
    ) -> tuple[float, dict[str, float | str]]:
        if contract.quality_model == "legacy_area_over_distance_squared_center_v1":
            return legacy_detection_quality(
                row,
                float(contract.quality_image_width_px or 1280),
                float(contract.quality_image_height_px or 720),
            )
        if contract.quality_model == "observation_quality_v2_selection_score":
            score = safe_float(
                row,
                "selection_score",
                observation_quality(row, *image_size),
            )
            return score, {
                "quality_model": contract.quality_model,
                "selection_score": score,
            }
        raise ValueError(f"Unknown AP01 quality model: {contract.quality_model}")

    prepared_static = []
    for row in static_rows:
        if not is_success(row):
            continue
        item = dict(row)
        item["_marker"] = int(float(row["marker_id"]))
        item["_camera"] = row["camera_name"]
        item["_quality"], item["_quality_components"] = prepare_quality(
            row, static_size
        )
        item["_area_px2"] = marker_area_from_corners(row)
        item["_distance_m"] = safe_float(row, "distance_m", 99.0)
        item["_T_cam_marker"] = T_from_observation(row)
        prepared_static.append(item)

    prepared_moving = []
    for row in moving_rows:
        if not is_success(row):
            continue
        item = dict(row)
        item["_marker"] = int(float(row["marker_id"]))
        item["_frame"] = frame_number(row)
        item["_quality"], item["_quality_components"] = prepare_quality(
            row, moving_size
        )
        item["_area_px2"] = marker_area_from_corners(row)
        item["_distance_m"] = safe_float(row, "distance_m", 99.0)
        item["_T_cam_marker"] = T_from_observation(row)
        prepared_moving.append(item)

    return prepared_static, prepared_moving


def robust_scale(
    moving_rows: list[dict],
    colmap_poses: dict[int, np.ndarray],
    maximum_observations_per_marker: int | None = None,
    contract: AP01MethodContract | None = None,
) -> tuple[float, dict, list[dict]]:
    contract = contract or resolve_ap01_method_contract(
        scale_top_per_marker=maximum_observations_per_marker,
    )
    by_marker = defaultdict(list)
    rejected_observations: Counter[str] = Counter()
    for row in moving_rows:
        if contract.scale_pnp_success_only and not is_success(row):
            rejected_observations["pnp_unsuccessful"] += 1
            continue
        if contract.scale_registered_frames_only and row["_frame"] not in colmap_poses:
            rejected_observations["unregistered_frame"] += 1
            continue
        if (
            contract.scale_minimum_marker_area_px2 is not None
            and float(row["_area_px2"])
            < contract.scale_minimum_marker_area_px2
        ):
            rejected_observations["marker_area_below_minimum"] += 1
            continue
        if (
            contract.scale_maximum_marker_distance_m is not None
            and float(row["_distance_m"])
            > contract.scale_maximum_marker_distance_m
        ):
            rejected_observations["marker_distance_above_maximum"] += 1
            continue
        if contract.scale_maximum_center_norm is not None:
            width = float(contract.quality_image_width_px or 1280)
            height = float(contract.quality_image_height_px or 720)
            center_u = safe_float(row, "center_u")
            center_v = safe_float(row, "center_v")
            center_norm = math.hypot(
                center_u - width / 2.0,
                center_v - height / 2.0,
            ) / math.hypot(width / 2.0, height / 2.0)
            if center_norm > contract.scale_maximum_center_norm:
                rejected_observations["marker_center_norm_above_maximum"] += 1
                continue
        by_marker[row["_marker"]].append(row)

    registered_counts = {
        int(marker): len(rows) for marker, rows in sorted(by_marker.items())
    }
    for marker, rows in by_marker.items():
        if contract.scale_observation_construction_policy == (
            "quality_ranked_per_marker_before_pairing_v1"
        ):
            selected = sorted(
                rows,
                key=lambda row: (
                    -float(row["_quality"]),
                    int(row["_frame"]),
                ),
            )
        elif contract.scale_observation_construction_policy == (
            "legacy_registered_quality_filters_then_all_pairs_v1"
        ):
            selected = list(rows)
        else:
            raise ValueError(
                "Unknown AP01 scale observation policy: "
                f"{contract.scale_observation_construction_policy}"
            )
        if contract.scale_observation_limit_per_marker is not None:
            truncated = max(
                0,
                len(selected) - contract.scale_observation_limit_per_marker,
            )
            rejected_observations["per_marker_limit"] += truncated
            selected = selected[: contract.scale_observation_limit_per_marker]
        by_marker[marker] = selected
    selected_counts = {
        int(marker): len(rows) for marker, rows in sorted(by_marker.items())
    }

    pairs = []
    if contract.scale_sample_multiplicity_policy != (
        "all_within_marker_unordered_frame_pairs"
    ):
        raise ValueError(
            "Unknown AP01 scale sample multiplicity policy: "
            f"{contract.scale_sample_multiplicity_policy}"
        )
    for marker, rows in by_marker.items():
        rows = sorted(rows, key=lambda r: r["_frame"])
        for first, second in combinations(rows, 2):
            gap = abs(first["_frame"] - second["_frame"])
            if (
                gap < contract.scale_frame_gap_minimum
                or gap > contract.scale_frame_gap_maximum
            ):
                continue

            if contract.scale_pnp_quantity_policy != (
                "relative_camera_translation_norm_from_T_cam_marker_v1"
            ):
                raise ValueError(
                    "Unknown AP01 scale PnP quantity policy: "
                    f"{contract.scale_pnp_quantity_policy}"
                )
            T_metric = first["_T_cam_marker"] @ invT(second["_T_cam_marker"])
            metric_distance = float(np.linalg.norm(T_metric[:3, 3]))
            if not (
                contract.scale_metric_translation_minimum_m
                <= metric_distance
                <= contract.scale_metric_translation_maximum_m
            ):
                continue

            T_colmap = colmap_poses[first["_frame"]] @ invT(colmap_poses[second["_frame"]])
            colmap_distance = float(np.linalg.norm(T_colmap[:3, 3]))
            if contract.scale_colmap_translation_rejection_policy == "less_than":
                colmap_rejected = (
                    colmap_distance
                    < contract.scale_colmap_translation_minimum_units
                )
            elif contract.scale_colmap_translation_rejection_policy == (
                "less_than_or_equal"
            ):
                colmap_rejected = (
                    colmap_distance
                    <= contract.scale_colmap_translation_minimum_units
                )
            else:
                raise ValueError(
                    "Unknown COLMAP scale displacement rejection policy: "
                    f"{contract.scale_colmap_translation_rejection_policy}"
                )
            if colmap_rejected:
                continue

            ratio = metric_distance / colmap_distance
            if not math.isfinite(ratio) or ratio <= 0:
                continue

            if contract.scale_pair_quality_policy == "sqrt_marker_area_product":
                pair_quality = math.sqrt(
                    float(first["_area_px2"]) * float(second["_area_px2"])
                )
            elif contract.scale_pair_quality_policy == (
                "sqrt_observation_quality_product"
            ):
                pair_quality = math.sqrt(
                    float(first["_quality"]) * float(second["_quality"])
                )
            else:
                raise ValueError(
                    "Unknown AP01 scale pair-quality policy: "
                    f"{contract.scale_pair_quality_policy}"
                )
            pairs.append({
                "marker_id": marker,
                "frame_i": first["_frame"],
                "frame_j": second["_frame"],
                "frame_gap": gap,
                "metric_translation_m": metric_distance,
                "colmap_translation_units": colmap_distance,
                "scale_m_per_colmap_unit": ratio,
                "quality": pair_quality,
            })

    if len(pairs) < contract.scale_minimum_pair_count:
        raise RuntimeError(f"Too few AP01 metric-scale pairs: {len(pairs)}")

    values = np.asarray([row["scale_m_per_colmap_unit"] for row in pairs], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = contract.scale_mad_sigma_factor * mad
    threshold = contract.scale_mad_multiplier * sigma
    if contract.scale_relative_deviation_floor_fraction is not None:
        threshold = max(
            threshold,
            contract.scale_relative_deviation_floor_fraction * median,
        )
    if (
        contract.scale_aggregation_policy
        == "legacy_median_three_sigma_mad_v1"
        and mad <= 1e-12
    ):
        kept = pairs
    elif contract.scale_aggregation_policy in {
        "legacy_median_three_sigma_mad_v1",
        "wizard_median_mad_relative_floor_v1",
    }:
        kept = [
            row
            for row in pairs
            if abs(row["scale_m_per_colmap_unit"] - median) <= threshold
        ]
    else:
        raise ValueError(
            "Unknown AP01 scale aggregation policy: "
            f"{contract.scale_aggregation_policy}"
        )
    fallback_threshold: float | int
    if contract.scale_aggregation_policy == "legacy_median_three_sigma_mad_v1":
        fallback_threshold = max(
            contract.scale_fallback_minimum_count,
            contract.scale_fallback_fraction * len(pairs),
        )
    else:
        fallback_threshold = max(
            contract.scale_fallback_minimum_count,
            int(contract.scale_fallback_fraction * len(pairs)),
        )
    fallback_used = len(kept) < fallback_threshold
    if fallback_used:
        kept = pairs
    kept_values = np.asarray([row["scale_m_per_colmap_unit"] for row in kept], dtype=np.float64)
    if contract.scale_final_statistic != "median":
        raise ValueError(
            "Unknown AP01 final scale statistic: "
            f"{contract.scale_final_statistic}"
        )
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
        "maximum_observations_per_marker": (
            contract.scale_observation_limit_per_marker
        ),
        "rejected_observations_by_reason": dict(
            sorted(rejected_observations.items())
        ),
        "scale_contract": contract.fingerprint_payload(),
        "scale_contract_sha256": contract.scientific_fingerprint(),
        "aggregation_threshold": threshold,
        "aggregation_fallback_threshold": fallback_threshold,
        "aggregation_fallback_used": fallback_used,
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
        "translation_deviation_p90_m": (
            float(np.percentile(robust_translation_deviation, 90))
            if robust_inlier_indices
            else None
        ),
        "rotation_deviation_p90_deg": (
            float(np.percentile(robust_rotation_deviation, 90))
            if robust_inlier_indices
            else None
        ),
        "translation_robust_rms_m": (
            float(np.sqrt(np.mean(robust_translation_deviation**2)))
            if robust_inlier_indices
            else None
        ),
        "rotation_robust_rms_deg": (
            float(np.sqrt(np.mean(robust_rotation_deviation**2)))
            if robust_inlier_indices
            else None
        ),
        "translation_threshold_m": t_threshold,
        "rotation_threshold_deg": r_threshold,
    }
    return transform, stats


def aggregate_direct_marker_estimates(
    candidates: list[dict],
) -> tuple[np.ndarray, dict]:
    """Aggregate one GT-free relation per independent shared marker."""

    transform, stats = aggregate_candidates(
        candidates, translation_floor=0.12, rotation_floor=4.0
    )
    return transform, {
        **stats,
        "aggregate_type": (
            "quality_filtered_weighted_mean_of_mad_inliers_"
            "no_gt_selection"
        ),
        "raw_candidate_count": len(candidates),
        "independent_marker_count": len(
            {
                int(item["root_marker"])
                for item in candidates
                if item.get("root_marker") is not None
            }
        ),
        "ground_truth_used": False,
    }


def aggregate_relay_marker_chains(
    candidates: list[dict],
) -> tuple[np.ndarray, dict, list[dict]]:
    """Aggregate correlated relay samples in two GT-free hierarchy levels.

    Samples sharing a root/target marker pair form one correlated chain.
    Stage one estimates each chain.  Stage two robustly combines only those
    independent chain estimates, so thousands of Cartesian frame pairs can no
    longer masquerade as thousands of independent observations.
    """

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in candidates:
        grouped[
            (int(item["root_marker"]), int(item["target_marker"]))
        ].append(item)
    chain_candidates: list[dict] = []
    chain_reports: list[dict] = []
    for (root_marker, target_marker), group in sorted(grouped.items()):
        pose, stats = aggregate_candidates(
            group, translation_floor=0.30, rotation_floor=7.0
        )
        inlier_quality = [
            max(float(item.get("quality", 0.0)), 1e-12)
            for item in group
            if item.get("inlier")
        ]
        quality = (
            float(np.mean(inlier_quality))
            if inlier_quality
            else max(float(item.get("quality", 0.0)) for item in group)
        )
        chain_id = f"{root_marker}->{target_marker}"
        chain_candidate = {
            "mode": "relay_chain",
            "chain_id": chain_id,
            "root_marker": root_marker,
            "target_marker": target_marker,
            "quality": quality,
            "T": pose,
            "raw_candidate_count": len(group),
        }
        chain_candidates.append(chain_candidate)
        chain_reports.append(
            {
                "chain_id": chain_id,
                "root_marker": root_marker,
                "target_marker": target_marker,
                "raw_candidate_count": len(group),
                "robust_inlier_count": stats.get("robust_inliers", 0),
                "quality_weight": quality,
                "translation_dispersion_m": stats.get(
                    "maximum_inlier_translation_dispersion_m"
                ),
                "rotation_dispersion_deg": stats.get(
                    "maximum_inlier_rotation_dispersion_deg"
                ),
                "translation_robust_rms_m": stats.get(
                    "translation_robust_rms_m"
                ),
                "rotation_robust_rms_deg": stats.get(
                    "rotation_robust_rms_deg"
                ),
                "estimate": pose.tolist(),
            }
        )
    if not chain_candidates:
        raise RuntimeError("No AP01 relay marker-chain estimate")
    transform, final_stats = aggregate_candidates(
        chain_candidates, translation_floor=0.30, rotation_floor=7.0
    )
    final_stats.update(
        {
            "aggregate_type": (
                "hierarchical_weighted_mean_of_mad_inliers_"
                "no_gt_selection"
            ),
            "raw_candidate_count": len(candidates),
            "chain_count": len(chain_candidates),
            "independent_marker_chain_count": len(chain_candidates),
            "effective_support": int(
                final_stats.get("robust_inliers", 0)
            ),
            "chain_reports": chain_reports,
            "ground_truth_used": False,
        }
    )
    return transform, final_stats, chain_candidates


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


def moving_by_contract(
    rows: list[dict],
    registered_frames: set[int],
    contract: AP01MethodContract,
) -> dict[int, list[dict]]:
    """Select moving supports using the resolved AP01 scientific contract."""

    if contract.moving_support_policy == (
        "quality_ranked_registered_then_frame_ascending"
    ):
        return moving_by_marker(
            rows,
            registered_frames,
            top_per_marker=contract.relay_input_limit,
        )
    if contract.moving_support_policy == (
        "best_quality_per_frame_marker_first_tie_registered_only_frame_ascending"
    ):
        best: dict[tuple[int, int], dict] = {}
        for row in rows:
            frame = int(row["_frame"])
            marker = int(row["_marker"])
            if frame not in registered_frames:
                continue
            key = (frame, marker)
            if key not in best or float(row["_quality"]) > float(
                best[key]["_quality"]
            ):
                best[key] = row
        grouped: defaultdict[int, list[dict]] = defaultdict(list)
        for (_, marker), row in best.items():
            grouped[marker].append(row)
        for marker in grouped:
            grouped[marker].sort(key=lambda row: int(row["_frame"]))
        return dict(grouped)
    raise ValueError(
        f"Unknown AP01 moving-support policy: {contract.moving_support_policy}"
    )


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
            "root_area_px2": float(root_row.get("_area_px2", float("nan"))),
            "target_area_px2": float(target_row.get("_area_px2", float("nan"))),
            "root_distance_m": float(root_row.get("_distance_m", float("nan"))),
            "target_distance_m": float(target_row.get("_distance_m", float("nan"))),
            "root_support_key": root_row.get("observation_key"),
            "target_support_key": target_row.get("observation_key"),
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
                        "support_keys": [
                            root_static.get("observation_key"),
                            target_static.get("observation_key"),
                            root_moving.get("observation_key"),
                            target_moving.get("observation_key"),
                        ],
                        "T": transform,
                    })
    return result


def weighted_transform_mean(
    candidates: list[dict], indices: list[int]
) -> np.ndarray:
    if not indices:
        raise RuntimeError("No AP01 transforms to average")
    weights = np.asarray(
        [max(1e-12, float(candidates[index]["quality"])) for index in indices],
        dtype=np.float64,
    )
    weights /= weights.sum()
    translation = np.sum(
        np.asarray(
            [candidates[index]["T"][:3, 3] for index in indices],
            dtype=np.float64,
        )
        * weights[:, None],
        axis=0,
    )
    rotation = weighted_rotation_mean(
        [candidates[index]["T"][:3, :3] for index in indices], weights
    )
    return make_T(rotation, translation)


def legacy_se3_medoid(candidates: list[dict]) -> tuple[int, float]:
    """Legacy first-on-equal SE(3) medoid with t + 0.02*r distance."""

    if not candidates:
        raise RuntimeError("No AP01 direct candidates")
    best_index = 0
    best_score: float | None = None
    for index, candidate in enumerate(candidates):
        distances = []
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            translation = float(
                np.linalg.norm(candidate["T"][:3, 3] - other["T"][:3, 3])
            )
            rotation = rotation_difference_deg(
                candidate["T"][:3, :3], other["T"][:3, :3]
            )
            distances.append(translation + 0.02 * rotation)
        score = float(np.median(distances)) if distances else 0.0
        if best_score is None or score < best_score:
            best_index = index
            best_score = score
    return best_index, float(best_score or 0.0)


def legacy_medoid_inliers(
    candidates: list[dict],
    medoid_index: int,
    *,
    translation_floor: float,
    rotation_floor: float,
) -> tuple[list[int], dict]:
    center = candidates[medoid_index]["T"]
    translation = np.asarray(
        [
            np.linalg.norm(candidate["T"][:3, 3] - center[:3, 3])
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [
            rotation_difference_deg(
                candidate["T"][:3, :3], center[:3, :3]
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    t_median = float(np.median(translation))
    r_median = float(np.median(rotation))
    t_mad = 1.4826 * float(np.median(np.abs(translation - t_median)))
    r_mad = 1.4826 * float(np.median(np.abs(rotation - r_median)))
    t_threshold = max(translation_floor, t_median + 3.0 * t_mad)
    r_threshold = max(rotation_floor, r_median + 3.0 * r_mad)
    indices = [
        index
        for index, (t_value, r_value) in enumerate(zip(translation, rotation))
        if t_value <= t_threshold and r_value <= r_threshold
    ]
    return indices, {
        "translation_deviation_median_m": t_median,
        "translation_deviation_mad_scaled_m": t_mad,
        "translation_inlier_threshold_m": t_threshold,
        "rotation_deviation_median_deg": r_median,
        "rotation_deviation_mad_scaled_deg": r_mad,
        "rotation_inlier_threshold_deg": r_threshold,
    }


def aggregate_legacy_direct_candidates(
    candidates: list[dict], contract: AP01MethodContract
) -> tuple[np.ndarray, dict]:
    """Reproduce Main's quality-filtered direct aggregate and priority."""

    if not candidates:
        raise RuntimeError("No AP01 direct candidates")
    quality_indices = [
        index
        for index, candidate in enumerate(candidates)
        if float(candidate.get("root_area_px2", float("nan")))
        >= float(contract.direct_minimum_area_px2 or 0.0)
        and float(candidate.get("target_area_px2", float("nan")))
        >= float(contract.direct_minimum_area_px2 or 0.0)
        and float(candidate.get("root_distance_m", float("inf")))
        <= float(contract.direct_maximum_distance_m or float("inf"))
        and float(candidate.get("target_distance_m", float("inf")))
        <= float(contract.direct_maximum_distance_m or float("inf"))
        and float(candidate["quality"])
        >= float(contract.direct_minimum_combined_quality or 0.0)
    ]
    fallback = False
    if not quality_indices:
        fallback = True
        ranked = sorted(
            range(len(candidates)),
            key=lambda index: float(candidates[index]["quality"]),
            reverse=True,
        )
        fallback_count = int(contract.direct_quality_fallback_count or 1)
        quality_indices = ranked[: max(1, min(fallback_count, len(ranked)))]
    quality_candidates = [candidates[index] for index in quality_indices]
    medoid_index, medoid_score = legacy_se3_medoid(quality_candidates)
    inlier_local, inlier_stats = legacy_medoid_inliers(
        quality_candidates,
        medoid_index,
        translation_floor=contract.direct_translation_mad_floor_m,
        rotation_floor=contract.direct_rotation_mad_floor_deg,
    )
    if not inlier_local:
        inlier_local = list(range(len(quality_candidates)))
    inlier_indices = {quality_indices[index] for index in inlier_local}
    quality_set = set(quality_indices)
    preferred_index = next(
        (
            index
            for index in quality_indices
            if int(candidates[index]["root_marker"])
            == contract.preferred_direct_marker_id
        ),
        None,
    )
    if preferred_index is None:
        selected_index = quality_indices[medoid_index]
        selection_note = "quality_filtered_se3_medoid_fallback"
    else:
        selected_index = preferred_index
        selection_note = "marker14_visible_and_passed_no_gt_quality_filter"
    weighted_diagnostic = weighted_transform_mean(
        candidates, sorted(inlier_indices)
    )
    for index, candidate in enumerate(candidates):
        candidate["quality_filter_eligible"] = index in quality_set
        candidate["quality_filter_fallback_used"] = fallback
        candidate["inlier"] = index in inlier_indices
        candidate["pose_support"] = index in inlier_indices
        candidate["preferred_marker_selected"] = index == selected_index
    return candidates[selected_index]["T"], {
        "selected_aggregate_type": (
            "quality_filtered_preferred_marker_no_gt_selection"
        ),
        "aggregate_priority": [
            "quality_filtered_preferred_marker_no_gt_selection",
            "quality_filtered_weighted_mean_no_gt_selection",
            "weighted_mean_of_mad_inliers_no_gt_selection",
            "se3_medoid_no_gt_selection",
        ],
        "selected_marker_id": candidates[selected_index]["root_marker"],
        "selected_candidate_index": selected_index,
        "selection_note": selection_note,
        "quality_filter_fallback_used": fallback,
        "num_candidates": len(candidates),
        "num_quality_candidates": len(quality_indices),
        "num_quality_mad_inliers": len(inlier_indices),
        "quality_subset_medoid_score": medoid_score,
        "quality_subset_mad": inlier_stats,
        "quality_filtered_weighted_mean_diagnostic": (
            weighted_diagnostic.tolist()
        ),
        "ground_truth_used": False,
    }


def aggregate_legacy_relay_candidates(
    candidates: list[dict], contract: AP01MethodContract
) -> tuple[np.ndarray, dict]:
    """Reproduce Main's one-level flat MAD aggregate over every relay row."""

    if not candidates:
        raise RuntimeError("No AP01 relay candidates")
    translations = np.asarray(
        [candidate["T"][:3, 3] for candidate in candidates], dtype=np.float64
    )
    all_indices = list(range(len(candidates)))
    weighted = weighted_transform_mean(candidates, all_indices)
    initial = make_T(weighted[:3, :3], np.median(translations, axis=0))
    translation_deviation = np.asarray(
        [
            np.linalg.norm(candidate["T"][:3, 3] - initial[:3, 3])
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    rotation_deviation = np.asarray(
        [
            rotation_difference_deg(
                candidate["T"][:3, :3], initial[:3, :3]
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    t_median = float(np.median(translation_deviation))
    r_median = float(np.median(rotation_deviation))
    t_mad = 1.4826 * float(
        np.median(np.abs(translation_deviation - t_median))
    )
    r_mad = 1.4826 * float(
        np.median(np.abs(rotation_deviation - r_median))
    )
    t_threshold = max(
        contract.relay_translation_mad_floor_m, t_median + 3.0 * t_mad
    )
    r_threshold = max(
        contract.relay_rotation_mad_floor_deg, r_median + 3.0 * r_mad
    )
    inlier_indices = [
        index
        for index, (t_value, r_value) in enumerate(
            zip(translation_deviation, rotation_deviation)
        )
        if t_value <= t_threshold and r_value <= r_threshold
    ]
    fallback = False
    minimum = int(contract.relay_fallback_minimum_count or 0)
    if len(inlier_indices) < minimum:
        fallback = True
        ranked = sorted(
            all_indices,
            key=lambda index: float(candidates[index]["quality"]),
            reverse=True,
        )
        fraction = float(contract.relay_fallback_fraction or 0.5)
        keep = max(minimum, int(len(ranked) * fraction))
        inlier_indices = ranked[:keep]
    inlier_set = set(inlier_indices)
    for index, candidate in enumerate(candidates):
        candidate["inlier"] = index in inlier_set
        candidate["pose_support"] = index in inlier_set
        candidate["translation_deviation_m"] = float(
            translation_deviation[index]
        )
        candidate["rotation_deviation_deg"] = float(rotation_deviation[index])
    transform = weighted_transform_mean(candidates, inlier_indices)
    return transform, {
        "aggregate_type": "weighted_mean_of_mad_inliers_no_gt_selection",
        "num_candidates": len(candidates),
        "num_inliers": len(inlier_indices),
        "num_outliers": len(candidates) - len(inlier_indices),
        "translation_deviation_median_m": t_median,
        "translation_deviation_mad_scaled_m": t_mad,
        "translation_inlier_threshold_m": t_threshold,
        "rotation_deviation_median_deg": r_median,
        "rotation_deviation_mad_scaled_deg": r_mad,
        "rotation_inlier_threshold_deg": r_threshold,
        "fallback_top_half_by_quality": fallback,
        "ground_truth_used": False,
    }


def R_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def rpy_deg_to_R(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> np.ndarray:
    """Reconstruct the ZYX rotation used by the Legacy aggregate CSV."""

    roll, pitch, yaw = (
        math.radians(float(value))
        for value in (roll_deg, pitch_deg, yaw_deg)
    )
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def serialize_final_pose(
    transform: np.ndarray, contract: AP01MethodContract
) -> np.ndarray:
    """Apply the contract-scoped final numeric serialization adapter."""

    if contract.final_pose_serialization_policy == "native_full_precision_v1":
        return np.asarray(transform, dtype=np.float64).copy()
    if (
        contract.final_pose_serialization_policy
        != "legacy_aggregate_csv_rpy_roundtrip_v1"
    ):
        raise ValueError(
            "Unknown AP01 final-pose serialization policy: "
            f"{contract.final_pose_serialization_policy}"
        )
    places = contract.final_pose_serialization_decimal_places
    if places is None:
        raise ValueError("Legacy AP01 final-pose serialization needs precision")
    result = np.eye(4, dtype=np.float64)
    rpy = R_to_rpy_deg(np.asarray(transform, dtype=np.float64)[:3, :3])
    serialized_rpy = tuple(float(f"{value:.{places}f}") for value in rpy)
    result[:3, :3] = rpy_deg_to_R(*serialized_rpy)
    result[:3, 3] = [
        float(f"{float(value):.{places}f}")
        for value in np.asarray(transform, dtype=np.float64)[:3, 3]
    ]
    return result


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
        "rvec_x": float(rvec[0, 0]),
        "rvec_y": float(rvec[1, 0]),
        "rvec_z": float(rvec[2, 0]),
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

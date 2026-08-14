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



from .core_geometry import parse_colmap_poses
from .core_io import write_csv
def colmap_camera_model(
    info: dict,
    contract: AP01MethodContract | None = None,
) -> tuple[str, str]:
    contract = contract or resolve_ap01_method_contract()
    K = info["K"]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    model = info["distortion_model"].strip().lower()
    d = list(float(v) for v in info["D"])
    if contract.colmap_camera_model_policy == "baseline_shared_pinhole_v1":
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

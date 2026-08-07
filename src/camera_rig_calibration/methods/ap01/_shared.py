from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from camera_rig_calibration.pipeline.stage_cli import camera_ids, method_parser

from . import core
from .contracts import (
    AP01_CONTRACT_NAMES,
    DEFAULT_AP01_CONTRACT,
    resolve_ap01_method_contract,
)


def parser(description: str) -> argparse.ArgumentParser:
    result = method_parser(description)
    result.add_argument("--root-camera", required=True)
    result.add_argument("--moving-camera-id", required=True)
    result.add_argument(
        "--method-contract",
        choices=AP01_CONTRACT_NAMES,
        default=DEFAULT_AP01_CONTRACT,
    )
    result.add_argument("--direct-target-camera", default="cam_edge_1")
    result.add_argument("--top-moving-per-marker", type=int, default=None)
    result.add_argument("--scale-top-per-marker", type=int, default=None)
    result.add_argument(
        "--direct-minimum-independent-markers", type=int, default=3
    )
    result.add_argument(
        "--direct-minimum-inlier-ratio", type=float, default=0.70
    )
    result.add_argument(
        "--direct-maximum-translation-dispersion-m",
        type=float,
        default=0.12,
    )
    result.add_argument(
        "--direct-maximum-rotation-dispersion-deg",
        type=float,
        default=4.0,
    )
    result.add_argument(
        "--relay-minimum-inlier-ratio", type=float, default=0.70
    )
    result.add_argument(
        "--relay-maximum-translation-dispersion-m",
        type=float,
        default=0.30,
    )
    result.add_argument(
        "--relay-maximum-rotation-dispersion-deg",
        type=float,
        default=7.0,
    )
    result.add_argument(
        "--maximum-path-translation-disagreement-m",
        type=float,
        default=0.12,
    )
    result.add_argument(
        "--maximum-path-rotation-disagreement-deg",
        type=float,
        default=4.0,
    )
    return result


def cameras(arguments: argparse.Namespace) -> tuple[str, ...]:
    values = camera_ids(arguments.cameras)
    if arguments.root_camera not in values:
        raise RuntimeError(
            f"Root camera '{arguments.root_camera}' is not in --cameras"
        )
    return values


def colmap_images(output: Path) -> Path:
    preferred = output / "01_moving_colmap/sparse_txt_best/images.txt"
    if preferred.is_file():
        return preferred
    candidates = sorted((output / "01_moving_colmap").rglob("images.txt"))
    if not candidates:
        raise RuntimeError(
            f"AP01 COLMAP images.txt is missing under {output}"
        )
    return candidates[0]


def prepared_observations(
    arguments: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict[int, np.ndarray]]:
    contract = resolve_ap01_method_contract(
        getattr(arguments, "method_contract", DEFAULT_AP01_CONTRACT),
        direct_target_camera=getattr(
            arguments, "direct_target_camera", "cam_edge_1"
        ),
        top_moving_per_marker=getattr(arguments, "top_moving_per_marker", None),
        scale_top_per_marker=getattr(
            arguments, "scale_top_per_marker", None
        ),
    )
    moving_info = core.load_camera_info(
        arguments.dataset
        / "raw_images"
        / "camera_info"
        / f"{arguments.moving_camera_id}.json"
    )
    static_info = core.load_camera_info(
        arguments.dataset
        / "raw_images"
        / "camera_info"
        / f"{arguments.root_camera}.json"
    )
    static_rows, moving_rows = core.prepare_observations(
        core.read_csv(
            arguments.observations_root
            / "shared_static_aruco_observations.csv"
        ),
        core.read_csv(
            arguments.observations_root
            / "shared_moving_aruco_observations.csv"
        ),
        (static_info["width"], static_info["height"]),
        (moving_info["width"], moving_info["height"]),
        contract=contract,
    )
    return (
        static_rows,
        moving_rows,
        core.parse_colmap_poses(colmap_images(arguments.out)),
    )


def encode_candidate(row: dict) -> dict:
    return {
        **{key: value for key, value in row.items() if key != "T"},
        "transform": np.asarray(row["T"], dtype=np.float64).tolist(),
    }


def decode_candidate(row: dict) -> dict:
    return {
        **{key: value for key, value in row.items() if key != "transform"},
        "T": np.asarray(row["transform"], dtype=np.float64),
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

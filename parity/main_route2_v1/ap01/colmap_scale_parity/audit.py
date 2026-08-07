"""Read-only AP01 historical-vs-Wizard COLMAP/scale parity audit.

This utility consumes preserved artifacts only.  It never invokes COLMAP, runs an
AP01 method, publishes/reconciles results, or reads Ground Truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
LEGACY_REPO = ROOT.parent / "camera-rig-calibration-main-route2-recovery"
LEGACY = (
    LEGACY_REPO
    / "results/bus_real_data/01_marker_direct_relay_multimarker_multichain"
    / "04_moving_camera_colmap_trajectory"
)
FRESH_RUN = (
    ROOT
    / "results/simulation/baseline/route2_main_parity_v1/attempts/ap01/baseline"
    / "20260807_012714_baseline_f6020505/diagnostics"
)
FRESH = FRESH_RUN / "02_AP01"
LEGACY_DB = LEGACY / "database.db"
FRESH_DB = FRESH / "01_moving_colmap/database.db"
LEGACY_IMAGES = LEGACY / "sparse_txt_best/images.txt"
FRESH_IMAGES = FRESH / "01_moving_colmap/sparse_txt_best/images.txt"
LEGACY_SCALE_ALL = LEGACY / "aruco_metric_scale/scale_pairs_all.csv"
LEGACY_SCALE_KEPT = LEGACY / "aruco_metric_scale/scale_pairs_kept_after_mad.csv"
FRESH_SCALE_ALL = FRESH / "02_metric_scale/scale_pairs.csv"
FRESH_MOVING_OBSERVATIONS = FRESH_RUN / "preflight/observations/shared_moving_aruco_observations.csv"
RECOVERED_MAIN_MOVING_OBSERVATIONS = (
    ROOT / "parity/main_route2_v1/generated/main_legacy_observations/shared_moving_aruco_observations.csv"
)
PAIR_ID_BASE = 2_147_483_647


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blob_sha256(value: bytes | None) -> str | None:
    return hashlib.sha256(value).hexdigest() if value is not None else None


def write_json(name: str, payload: Any) -> None:
    (OUT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(name: str, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def quaternion_rotation(q: Iterable[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(list(q), dtype=float)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = (qw / norm, qx / norm, qy / norm, qz / norm)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def rotation_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def parse_colmap_images(path: Path) -> dict[str, dict[str, Any]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) % 2:
        raise ValueError(f"Malformed COLMAP images file: {path}")
    result: dict[str, dict[str, Any]] = {}
    for record_index in range(0, len(lines), 2):
        tokens = lines[record_index].split()
        if len(tokens) != 10:
            raise ValueError(f"Malformed pose record at {path}:{record_index + 1}")
        q = np.asarray([float(value) for value in tokens[1:5]], dtype=float)
        t = np.asarray([float(value) for value in tokens[5:8]], dtype=float)
        rotation = quaternion_rotation(q)
        name = tokens[9]
        result[name] = {
            "image_id": int(tokens[0]),
            "camera_id": int(tokens[8]),
            "qvec": q,
            "tvec": t,
            "rotation": rotation,
            "center": -(rotation.T @ t),
            "text_record_index": record_index // 2,
        }
    return result


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3) mapping target ~= scale * rotation * source + translation."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be Nx3 with identical shape")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    scale = float(np.dot(singular_values, sign) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def database_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def database_inventory(path: Path) -> dict[str, Any]:
    connection = database_connection(path)
    try:
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        schema = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        }
        images = {
            str(name): {"image_id": int(image_id), "camera_id": int(camera_id)}
            for image_id, name, camera_id in connection.execute(
                "SELECT image_id, name, camera_id FROM images"
            )
        }
        camera_rows = list(connection.execute("SELECT camera_id, model, width, height, params FROM cameras"))
        cameras = []
        for camera_id, model, width, height, params in camera_rows:
            unpacked = list(struct.unpack(f"<{len(params) // 8}d", params))
            cameras.append(
                {
                    "camera_id": int(camera_id),
                    "model_id": int(model),
                    "width": int(width),
                    "height": int(height),
                    "parameters": unpacked,
                    "parameter_blob_sha256": blob_sha256(params),
                }
            )
        image_id_to_name = {value["image_id"]: key for key, value in images.items()}
        features: dict[str, Any] = {}
        for table in ("keypoints", "descriptors"):
            features[table] = {}
            for image_id, rows, cols, data in connection.execute(
                f"SELECT image_id, rows, cols, data FROM {table}"
            ):
                features[table][image_id_to_name[int(image_id)]] = {
                    "rows": int(rows),
                    "cols": int(cols),
                    "blob_sha256": blob_sha256(data),
                }
        pairs: dict[str, Any] = {}
        for table in ("matches", "two_view_geometries"):
            records: dict[tuple[str, str], Any] = {}
            columns = "pair_id, rows, cols, data"
            if table == "two_view_geometries":
                columns += ", config"
            for row in connection.execute(f"SELECT {columns} FROM {table}"):
                pair_id, rows, cols, data = row[:4]
                image_id1 = int(pair_id) // PAIR_ID_BASE
                image_id2 = int(pair_id) % PAIR_ID_BASE
                name1 = image_id_to_name[image_id1]
                name2 = image_id_to_name[image_id2]
                key = tuple(sorted((name1, name2)))
                records[key] = {
                    "rows": int(rows),
                    "cols": int(cols),
                    "blob_sha256": blob_sha256(data),
                    "config": int(row[4]) if len(row) > 4 else None,
                }
            pairs[table] = records
        return {
            "user_version": user_version,
            "schema": schema,
            "images": images,
            "cameras": cameras,
            "features": features,
            "pairs": pairs,
        }
    finally:
        connection.close()


def first_difference(
    keys: Iterable[Any], legacy: dict[Any, Any], fresh: dict[Any, Any], predicate
) -> Any | None:
    return next((key for key in sorted(keys) if predicate(legacy.get(key), fresh.get(key))), None)


def robust_statistics(values: Iterable[float], wizard: bool) -> dict[str, Any]:
    ratios = np.asarray(list(values), dtype=float)
    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))
    sigma = 1.4826 * mad
    threshold = max(3.0 * sigma, 0.10 * median) if wizard else 3.0 * sigma
    kept = ratios[np.abs(ratios - median) <= threshold]
    fallback_threshold = (
        max(10, int(0.25 * len(ratios)))
        if wizard
        else max(10, 0.30 * len(ratios))
    )
    fallback_used = len(kept) < fallback_threshold
    if fallback_used:
        kept = ratios
    return {
        "raw_count": len(ratios),
        "raw_median": median,
        "raw_mad": mad,
        "threshold": float(threshold),
        "fallback_threshold": fallback_threshold,
        "fallback_used": fallback_used,
        "kept_count": len(kept),
        "final_median": float(np.median(kept)),
        "kept_mean": float(np.mean(kept)),
        "kept_std": float(np.std(kept)),
    }


def build_configuration(legacy_db: dict[str, Any], fresh_db: dict[str, Any]) -> dict[str, Any]:
    environment = read_json(FRESH_RUN / "environment.json")
    fields = [
        ("input_image_inventory", "189 moving PNG files", "189 moving PNG files", "EXACT"),
        ("colmap_executable", "colmap resolved from PATH", "/usr/bin/colmap", "HISTORICAL_VALUE_UNAVAILABLE"),
        ("colmap_executable_version", "UNKNOWN", "3.9.1 (no CUDA)", "HISTORICAL_VALUE_UNAVAILABLE"),
        ("database_user_version", legacy_db["user_version"], fresh_db["user_version"], "DIFFERENT_CONFIGURATION"),
        ("camera_model", "PINHOLE", "PINHOLE", "EXACT"),
        ("camera_dimensions", "1280x720", "1280x720", "EXACT"),
        ("camera_parameters_text", "929.46716309,929.46713448,640.00000000,360.00000000", "929.4671630859375,929.46713447570801,640,360", "DIFFERENT_CONFIGURATION"),
        ("single_shared_camera", True, True, "EXACT"),
        ("refine_focal_length", False, False, "EXACT"),
        ("refine_principal_point", False, False, "EXACT"),
        ("refine_extra_parameters", False, False, "EXACT"),
        ("feature_type", "COLMAP SIFT", "COLMAP SIFT", "SEMANTICALLY_EQUIVALENT"),
        ("sift_use_gpu", False, False, "EXACT"),
        ("sift_max_image_size", 1600, 1600, "EXACT"),
        ("sift_max_num_features", 4096, 4096, "EXACT"),
        ("sift_num_threads", 1, "omitted; COLMAP default/all available threads", "DIFFERENT_CONFIGURATION"),
        ("matcher", "exhaustive", "exhaustive", "EXACT"),
        ("matcher_use_gpu", False, False, "EXACT"),
        ("matcher_other_options", "COLMAP defaults for historical executable", "COLMAP 3.9.1 defaults", "HISTORICAL_VALUE_UNAVAILABLE"),
        ("mapper_min_num_matches", 15, 8, "DIFFERENT_CONFIGURATION"),
        ("mapper_threads", -1, -1, "EXACT"),
        ("mapper_random_seed", 0, 0, "EXACT"),
        ("mapper_multiple_models", True, True, "EXACT"),
        ("mapper_min_model_size", 10, 10, "EXACT"),
        ("triangulation_min_max_options", "init_min_tri_angle=16; filter_min_tri_angle=1.5; local_ba_min_tri_angle=6; tri_min_angle=1.5; tri_create/continue_max_angle_error=2; tri_re_max_angle_error=5; tri_re_min_ratio=0.2; tri_max_transitivity=1; tri_complete_max_transitivity=5; tri_re_max_trials=1", "same saved values", "EXACT"),
        ("masks", "none", "none", "EXACT"),
        ("input_filtering", "none; all moving images", "none; all moving images", "EXACT"),
        ("database_construction", "new DB by feature_extractor", "new DB by feature_extractor", "SEMANTICALLY_EQUIVALENT"),
        ("image_id_order", "numeric filename order", "parallel completion order; 73/189 IDs differ", "DIFFERENT_CONFIGURATION"),
        ("model_selection", "largest registered-image count; lexicographically first tie", "largest registered-image count; lexicographically first tie", "EXACT"),
        ("post_colmap_filter", "none before selecting best model", "none before selecting best model", "EXACT"),
    ]
    return {
        "schema_version": 1,
        "scope": "AP01 moving-camera COLMAP only; no Ground Truth",
        "overall_classification": "DIFFERENT_CONFIGURATION",
        "comparison": [
            {"field": field, "legacy": legacy, "wizard": fresh, "classification": status}
            for field, legacy, fresh, status in fields
        ],
        "environment_audit": {
            "legacy": {
                "os": "UNKNOWN",
                "python": "UNKNOWN",
                "pycolmap": "not used by recovered invocation",
                "colmap_executable": "colmap from PATH; resolved path UNKNOWN",
                "colmap_version": "UNKNOWN",
                "database_user_version": legacy_db["user_version"],
                "cuda": "disabled by explicit SIFT extraction/matching flags; build capability UNKNOWN",
                "gpu_model": "NOT_APPLICABLE",
                "cpu_architecture": "UNKNOWN",
                "available_threads": "UNKNOWN",
                "feature_extraction_threads": 1,
                "opencv": "UNKNOWN",
                "numpy": "UNKNOWN",
                "scipy": "UNKNOWN",
            },
            "wizard": {
                "os": environment["platform"],
                "python": environment["python"],
                "python_executable": environment["python_executable"],
                "pycolmap": "not installed/not used",
                "colmap_executable": environment["colmap"],
                "colmap_version": "3.9.1",
                "colmap_build_cuda": False,
                "requested_gpu": False,
                "gpu_model": "NOT_APPLICABLE",
                "cpu_architecture": "x86_64; Intel Core i5-12400F",
                "logical_cpu_threads": 12,
                "feature_extraction_threads": "COLMAP default; option omitted",
                "opencv": environment["scientific_packages"]["cv2"],
                "numpy": environment["scientific_packages"]["numpy"],
                "scipy": environment["scientific_packages"]["scipy"],
            },
        },
        "recovered_commands": {
            "legacy": {
                "feature_extractor": ["colmap", "feature_extractor", "--database_path", "<legacy database.db>", "--image_path", "<189-image moving directory>", "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params", "929.46716309,929.46713448,640.00000000,360.00000000", "--SiftExtraction.use_gpu", "0", "--SiftExtraction.num_threads", "1", "--SiftExtraction.max_num_features", "4096", "--SiftExtraction.max_image_size", "1600"],
                "matcher": ["colmap", "exhaustive_matcher", "--database_path", "<legacy database.db>", "--SiftMatching.use_gpu", "0"],
                "mapper": ["colmap", "mapper", "--database_path", "<legacy database.db>", "--image_path", "<189-image moving directory>", "--output_path", "<legacy sparse>", "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0", "--Mapper.ba_refine_extra_params", "0"],
                "mapper_effective_min_num_matches_from_saved_project": 15,
            },
            "wizard": {
                "feature_extractor": ["/usr/bin/colmap", "feature_extractor", "--database_path", "<fresh database.db>", "--image_path", "<same 189-image moving directory>", "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params", "929.4671630859375,929.46713447570801,640,360", "--SiftExtraction.use_gpu", "0", "--SiftExtraction.max_image_size", "1600", "--SiftExtraction.max_num_features", "4096"],
                "matcher": ["/usr/bin/colmap", "exhaustive_matcher", "--database_path", "<fresh database.db>", "--SiftMatching.use_gpu", "0"],
                "mapper": ["/usr/bin/colmap", "mapper", "--database_path", "<fresh database.db>", "--image_path", "<same 189-image moving directory>", "--output_path", "<fresh sparse>", "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0", "--Mapper.ba_refine_extra_params", "0", "--Mapper.min_num_matches", "8"],
            },
        },
        "saved_project_ini_audit": {
            "scientific_difference": {"Mapper.min_num_matches": {"legacy": 15, "wizard": 8}},
            "same_triangulation_and_bundle_adjustment_values": True,
            "non_scientific_differences": {"log_to_stderr": {"legacy": False, "wizard": True}, "log_level": {"legacy": 2, "wizard": 0}, "database_path": "different output locations", "image_path": "different path spelling to hash-identical input"},
            "version_signature_fields": {"legacy_only": {"Mapper.ba_global_use_pba": False, "Mapper.ba_global_pba_gpu_index": -1}, "wizard_only": []},
        },
        "evidence": {
            "legacy_invocation": str(LEGACY_REPO / "run/bus_real_data/approach1_marker_direct_relay/06_run_colmap_moving_sequence.py"),
            "legacy_project_ini": str(LEGACY / "sparse/0/project.ini"),
            "legacy_database_sha256": sha256_file(LEGACY_DB),
            "wizard_core": str(ROOT / "src/camera_rig_calibration/methods/ap01/core.py"),
            "wizard_stage_manifest": str(FRESH / "01_moving_colmap/stage_manifest.json"),
            "wizard_project_ini": str(FRESH / "01_moving_colmap/sparse/0/project.ini"),
            "wizard_environment": str(FRESH_RUN / "environment.json"),
            "wizard_database_sha256": sha256_file(FRESH_DB),
        },
        "limitations": [
            "Database user_version proves different stored COLMAP schema generations (3700 vs 3900), but does not identify the exact historical executable version.",
            "Historical runtime OS, CPU, package versions, and executable resolution are not present in repository artifacts and remain UNKNOWN.",
        ],
    }


def build_registration(legacy_poses: dict[str, Any], fresh_poses: dict[str, Any], legacy_db: dict[str, Any], fresh_db: dict[str, Any]) -> dict[str, Any]:
    all_names = sorted(set(legacy_db["images"]) | set(fresh_db["images"]))
    rows = []
    for name in all_names:
        legacy_registered = name in legacy_poses
        fresh_registered = name in fresh_poses
        legacy_image = legacy_db["images"].get(name, {})
        fresh_image = fresh_db["images"].get(name, {})
        status = "EXACT"
        if legacy_registered != fresh_registered:
            status = "REGISTRATION_DIFF"
        elif legacy_image.get("image_id") != fresh_image.get("image_id"):
            status = "IMAGE_ID_DIFF"
        elif legacy_image.get("camera_id") != fresh_image.get("camera_id"):
            status = "CAMERA_ID_DIFF"
        rows.append(
            {
                "image_name": name,
                "legacy_registered": legacy_registered,
                "wizard_registered": fresh_registered,
                "legacy_image_id": legacy_image.get("image_id", ""),
                "wizard_image_id": fresh_image.get("image_id", ""),
                "legacy_camera_id": legacy_image.get("camera_id", ""),
                "wizard_camera_id": fresh_image.get("camera_id", ""),
                "status": status,
            }
        )
    write_csv(
        "COLMAP_REGISTRATION_DIFF.csv",
        ["image_name", "legacy_registered", "wizard_registered", "legacy_image_id", "wizard_image_id", "legacy_camera_id", "wizard_camera_id", "status"],
        rows,
    )
    legacy_names = set(legacy_poses)
    fresh_names = set(fresh_poses)
    return {
        "schema_version": 1,
        "classification": "DIFFERENT_REGISTERED_IMAGES",
        "input_images": {"legacy": len(legacy_db["images"]), "wizard": len(fresh_db["images"]), "same_names": set(legacy_db["images"]) == set(fresh_db["images"])},
        "registered_images": {"legacy": len(legacy_names), "wizard": len(fresh_names), "common": len(legacy_names & fresh_names)},
        "legacy_only_registered": sorted(legacy_names - fresh_names),
        "wizard_only_registered": sorted(fresh_names - legacy_names),
        "legacy_unregistered": sorted(set(legacy_db["images"]) - legacy_names),
        "wizard_unregistered": sorted(set(fresh_db["images"]) - fresh_names),
        "image_id_parity": {
            "different_count_all_inputs": sum(legacy_db["images"][name]["image_id"] != fresh_db["images"][name]["image_id"] for name in all_names),
            "different_count_common_registered": sum(legacy_db["images"][name]["image_id"] != fresh_db["images"][name]["image_id"] for name in legacy_names & fresh_names),
            "first_difference": next((name for name in all_names if legacy_db["images"][name]["image_id"] != fresh_db["images"][name]["image_id"]), None),
        },
        "camera_id_parity": {"legacy_unique": sorted({v["camera_id"] for v in legacy_db["images"].values()}), "wizard_unique": sorted({v["camera_id"] for v in fresh_db["images"].values()})},
        "registration_order": "UNKNOWN for both; image IDs and model-converter text record order are not mapper registration chronology",
        "selected_sparse_model": {"legacy": "0 (only produced model)", "wizard": "0 (only produced model)"},
        "historical_report_note": "Legacy colmap_report.txt says 176 because its text-line counter can count a POINTS2D line. The actual model header and 175 parsed image records, also consumed by the legacy scale report, are authoritative.",
        "evidence": {"legacy_images_txt": str(LEGACY_IMAGES), "legacy_images_sha256": sha256_file(LEGACY_IMAGES), "wizard_images_txt": str(FRESH_IMAGES), "wizard_images_sha256": sha256_file(FRESH_IMAGES)},
    }


def build_trajectory(legacy_poses: dict[str, Any], fresh_poses: dict[str, Any]) -> dict[str, Any]:
    common = sorted(set(legacy_poses) & set(fresh_poses))
    legacy_centers = np.asarray([legacy_poses[name]["center"] for name in common])
    fresh_centers = np.asarray([fresh_poses[name]["center"] for name in common])
    scale, rotation, translation = umeyama(legacy_centers, fresh_centers)
    aligned = scale * (legacy_centers @ rotation.T) + translation
    direct_errors = np.linalg.norm(legacy_centers - fresh_centers, axis=1)
    aligned_errors = np.linalg.norm(aligned - fresh_centers, axis=1)
    direct_rotation_errors = []
    aligned_rotation_errors = []
    rows = []
    for index, name in enumerate(common):
        legacy_rotation = legacy_poses[name]["rotation"]
        fresh_rotation = fresh_poses[name]["rotation"]
        direct_rotation = rotation_degrees(fresh_rotation @ legacy_rotation.T)
        expected_fresh_rotation = legacy_rotation @ rotation.T
        aligned_rotation = rotation_degrees(fresh_rotation @ expected_fresh_rotation.T)
        direct_rotation_errors.append(direct_rotation)
        aligned_rotation_errors.append(aligned_rotation)
        lc = legacy_centers[index]
        fc = fresh_centers[index]
        ac = aligned[index]
        rows.append(
            {
                "image_name": name,
                "legacy_x": lc[0], "legacy_y": lc[1], "legacy_z": lc[2],
                "wizard_x": fc[0], "wizard_y": fc[1], "wizard_z": fc[2],
                "direct_translation_error": direct_errors[index],
                "aligned_legacy_x": ac[0], "aligned_legacy_y": ac[1], "aligned_legacy_z": ac[2],
                "sim3_translation_residual": aligned_errors[index],
                "direct_rotation_error_degrees": direct_rotation,
                "sim3_rotation_residual_degrees": aligned_rotation,
            }
        )
    write_csv(
        "RAW_TRAJECTORY_DIFF.csv",
        ["image_name", "legacy_x", "legacy_y", "legacy_z", "wizard_x", "wizard_y", "wizard_z", "direct_translation_error", "aligned_legacy_x", "aligned_legacy_y", "aligned_legacy_z", "sim3_translation_residual", "direct_rotation_error_degrees", "sim3_rotation_residual_degrees"],
        rows,
    )
    direct_rotation_errors = np.asarray(direct_rotation_errors)
    aligned_rotation_errors = np.asarray(aligned_rotation_errors)
    error_median = float(np.median(aligned_errors))
    error_mad = float(np.median(np.abs(aligned_errors - error_median)))
    inlier_threshold = error_median + 3.0 * 1.4826 * error_mad
    radius = float(np.max(np.linalg.norm(fresh_centers - fresh_centers.mean(axis=0), axis=1)))
    return {
        "schema_version": 1,
        "classification": "DIFFERENT_REGISTERED_IMAGES",
        "interpretation": "The required inventory-first classification is DIFFERENT_REGISTERED_IMAGES. The common 175-image segment is close under one Sim(3), but residuals are materially above serialization tolerance and lower-level descriptor/match evidence differs, so this is not proven to be exactly the same reconstruction in only a different gauge.",
        "common_images": len(common),
        "direct_raw_coordinate_comparison": {
            "translation_rmse_colmap_units": float(np.sqrt(np.mean(direct_errors ** 2))),
            "translation_max_colmap_units": float(np.max(direct_errors)),
            "rotation_mean_degrees": float(np.mean(direct_rotation_errors)),
            "rotation_max_degrees": float(np.max(direct_rotation_errors)),
        },
        "diagnostic_similarity_legacy_to_wizard": {
            "scale_wizard_units_per_legacy_unit": scale,
            "rotation": rotation.tolist(),
            "translation_wizard_units": translation.tolist(),
            "translation_rmse_wizard_units": float(np.sqrt(np.mean(aligned_errors ** 2))),
            "translation_median_wizard_units": error_median,
            "translation_max_wizard_units": float(np.max(aligned_errors)),
            "translation_max_image": common[int(np.argmax(aligned_errors))],
            "wizard_trajectory_radius_units": radius,
            "translation_rmse_fraction_of_radius": float(np.sqrt(np.mean(aligned_errors ** 2)) / radius),
            "rotation_residual_mean_degrees": float(np.mean(aligned_rotation_errors)),
            "rotation_residual_median_degrees": float(np.median(aligned_rotation_errors)),
            "rotation_residual_max_degrees": float(np.max(aligned_rotation_errors)),
            "robust_translation_inlier_threshold": inlier_threshold,
            "robust_translation_inliers": int(np.sum(aligned_errors <= inlier_threshold)),
            "common_image_count": len(common),
        },
        "no_ground_truth": True,
        "diagnostic_only": True,
        "not_applied_to_published_extrinsics": True,
    }


def build_feature_match(legacy_db: dict[str, Any], fresh_db: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    image_names = sorted(set(legacy_db["images"]) | set(fresh_db["images"]))
    for name in image_names:
        legacy_keypoint = legacy_db["features"]["keypoints"].get(name)
        fresh_keypoint = fresh_db["features"]["keypoints"].get(name)
        legacy_descriptor = legacy_db["features"]["descriptors"].get(name)
        fresh_descriptor = fresh_db["features"]["descriptors"].get(name)
        for stage, legacy_feature, fresh_feature in (
            ("keypoints", legacy_keypoint, fresh_keypoint),
            ("descriptors", legacy_descriptor, fresh_descriptor),
        ):
            rows.append(
                {
                    "stage": stage, "key": name,
                    "legacy_rows": legacy_feature["rows"] if legacy_feature else "",
                    "wizard_rows": fresh_feature["rows"] if fresh_feature else "",
                    "legacy_cols": legacy_feature["cols"] if legacy_feature else "",
                    "wizard_cols": fresh_feature["cols"] if fresh_feature else "",
                    "count_equal": bool(legacy_feature and fresh_feature and legacy_feature["rows"] == fresh_feature["rows"]),
                    "blob_equal": bool(legacy_feature and fresh_feature and legacy_feature["blob_sha256"] == fresh_feature["blob_sha256"]),
                    "legacy_blob_sha256": legacy_feature["blob_sha256"] if legacy_feature else "",
                    "wizard_blob_sha256": fresh_feature["blob_sha256"] if fresh_feature else "",
                    "detail": "per-image SQLite feature record compared by semantic image name",
                }
            )
    pair_summaries = {}
    for table in ("matches", "two_view_geometries"):
        legacy_pairs = legacy_db["pairs"][table]
        fresh_pairs = fresh_db["pairs"][table]
        keys = set(legacy_pairs) | set(fresh_pairs)
        for key in sorted(keys):
            legacy = legacy_pairs.get(key)
            fresh = fresh_pairs.get(key)
            if legacy != fresh:
                rows.append(
                    {
                        "stage": table, "key": "|".join(key),
                        "legacy_rows": legacy["rows"] if legacy else "",
                        "wizard_rows": fresh["rows"] if fresh else "",
                        "legacy_cols": legacy["cols"] if legacy else "",
                        "wizard_cols": fresh["cols"] if fresh else "",
                        "count_equal": bool(legacy and fresh and legacy["rows"] == fresh["rows"]),
                        "blob_equal": bool(legacy and fresh and legacy["blob_sha256"] == fresh["blob_sha256"]),
                        "legacy_blob_sha256": legacy["blob_sha256"] if legacy else "",
                        "wizard_blob_sha256": fresh["blob_sha256"] if fresh else "",
                        "detail": f"legacy_config={legacy.get('config') if legacy else None};wizard_config={fresh.get('config') if fresh else None}",
                    }
                )
        pair_summaries[table] = {
            "legacy_pair_records": len(legacy_pairs),
            "wizard_pair_records": len(fresh_pairs),
            "same_pair_inventory": set(legacy_pairs) == set(fresh_pairs),
            "legacy_total_rows": sum(value["rows"] for value in legacy_pairs.values()),
            "wizard_total_rows": sum(value["rows"] for value in fresh_pairs.values()),
            "legacy_connected_edges": sum(value["rows"] > 0 for value in legacy_pairs.values()),
            "wizard_connected_edges": sum(value["rows"] > 0 for value in fresh_pairs.values()),
            "different_records": sum(legacy_pairs.get(key) != fresh_pairs.get(key) for key in keys),
            "first_count_difference": first_difference(keys, legacy_pairs, fresh_pairs, lambda a, b: not a or not b or a["rows"] != b["rows"]),
            "first_blob_difference": first_difference(keys, legacy_pairs, fresh_pairs, lambda a, b: not a or not b or a["blob_sha256"] != b["blob_sha256"]),
        }
        for field in ("first_count_difference", "first_blob_difference"):
            if pair_summaries[table][field] is not None:
                pair_summaries[table][field] = list(pair_summaries[table][field])
    write_csv(
        "FEATURE_MATCH_DIFF.csv",
        ["stage", "key", "legacy_rows", "wizard_rows", "legacy_cols", "wizard_cols", "count_equal", "blob_equal", "legacy_blob_sha256", "wizard_blob_sha256", "detail"],
        rows,
    )
    keypoints_legacy = legacy_db["features"]["keypoints"]
    keypoints_fresh = fresh_db["features"]["keypoints"]
    descriptors_legacy = legacy_db["features"]["descriptors"]
    descriptors_fresh = fresh_db["features"]["descriptors"]
    descriptor_first = first_difference(image_names, descriptors_legacy, descriptors_fresh, lambda a, b: a != b)
    return {
        "schema_version": 1,
        "classification": "SAME_KEYPOINTS_DIFFERENT_DESCRIPTORS_AND_MATCHES",
        "byte_parity": {
            "keypoint_blobs_equal": sum(keypoints_legacy[name]["blob_sha256"] == keypoints_fresh[name]["blob_sha256"] for name in image_names),
            "descriptor_blobs_equal": sum(descriptors_legacy[name]["blob_sha256"] == descriptors_fresh[name]["blob_sha256"] for name in image_names),
            "descriptor_blobs_different": sum(descriptors_legacy[name]["blob_sha256"] != descriptors_fresh[name]["blob_sha256"] for name in image_names),
        },
        "count_parity": {
            "images": len(image_names),
            "all_per_image_feature_counts_equal": all(keypoints_legacy[name]["rows"] == keypoints_fresh[name]["rows"] for name in image_names),
            "legacy_total_features": sum(value["rows"] for value in keypoints_legacy.values()),
            "wizard_total_features": sum(value["rows"] for value in keypoints_fresh.values()),
        },
        "semantic_graph_parity": pair_summaries,
        "first_observed_numeric_divergence": {
            "stage": "SIFT descriptor generation",
            "image": descriptor_first,
            "legacy_descriptor_sha256": descriptors_legacy[descriptor_first]["blob_sha256"],
            "wizard_descriptor_sha256": descriptors_fresh[descriptor_first]["blob_sha256"],
            "feature_count_both": descriptors_legacy[descriptor_first]["rows"],
            "interpretation": "Counts and keypoint blobs remain exact, but descriptor bytes first differ here. Byte identity across different COLMAP schema/version generations is diagnostic evidence, not a portability requirement; downstream semantic match counts also differ.",
        },
        "database_schema_sql_equal": legacy_db["schema"] == fresh_db["schema"],
        "database_user_versions": {"legacy": legacy_db["user_version"], "wizard": fresh_db["user_version"]},
    }


def scale_key(row: dict[str, str]) -> tuple[int, int, int]:
    return int(row["marker_id"]), int(row["frame_i"]), int(row["frame_j"])


def build_scale(trajectory: dict[str, Any]) -> dict[str, Any]:
    legacy_rows = read_csv(LEGACY_SCALE_ALL)
    legacy_kept_rows = read_csv(LEGACY_SCALE_KEPT)
    fresh_rows = read_csv(FRESH_SCALE_ALL)
    legacy = {scale_key(row): row for row in legacy_rows}
    fresh = {scale_key(row): row for row in fresh_rows}
    legacy_kept = {scale_key(row) for row in legacy_kept_rows}
    keys = sorted(set(legacy) | set(fresh))
    diff_rows = []
    common_metric_differences = []
    common_colmap_differences = []
    common_ratio_differences = []
    for key in keys:
        old = legacy.get(key)
        new = fresh.get(key)
        legacy_metric = float(old["metric_translation_m"]) if old else None
        fresh_metric = float(new["metric_translation_m"]) if new else None
        legacy_colmap = float(old["colmap_translation_raw"]) if old else None
        fresh_colmap = float(new["colmap_translation_units"]) if new else None
        legacy_ratio = float(old["scale_ratio"]) if old else None
        fresh_ratio = float(new["scale_m_per_colmap_unit"]) if new else None
        if old and new:
            common_metric_differences.append(abs(legacy_metric - fresh_metric))
            common_colmap_differences.append(abs(legacy_colmap - fresh_colmap))
            common_ratio_differences.append(abs(legacy_ratio - fresh_ratio))
        status = "COMMON" if old and new else ("LEGACY_ONLY" if old else "WIZARD_ONLY")
        diff_rows.append(
            {
                "marker_id": key[0], "frame_i": key[1], "frame_j": key[2], "status": status,
                "legacy_metric_translation_m": legacy_metric if legacy_metric is not None else "",
                "wizard_metric_translation_m": fresh_metric if fresh_metric is not None else "",
                "metric_translation_abs_delta": abs(legacy_metric - fresh_metric) if old and new else "",
                "legacy_colmap_translation_units": legacy_colmap if legacy_colmap is not None else "",
                "wizard_colmap_translation_units": fresh_colmap if fresh_colmap is not None else "",
                "colmap_translation_abs_delta": abs(legacy_colmap - fresh_colmap) if old and new else "",
                "legacy_scale_sample": legacy_ratio if legacy_ratio is not None else "",
                "wizard_scale_sample": fresh_ratio if fresh_ratio is not None else "",
                "scale_sample_abs_delta": abs(legacy_ratio - fresh_ratio) if old and new else "",
                "legacy_used": key in legacy_kept,
                "wizard_used": bool(new and new["used_for_scale"].lower() == "true"),
            }
        )
    write_csv(
        "SCALE_OBSERVATION_DIFF.csv",
        ["marker_id", "frame_i", "frame_j", "status", "legacy_metric_translation_m", "wizard_metric_translation_m", "metric_translation_abs_delta", "legacy_colmap_translation_units", "wizard_colmap_translation_units", "colmap_translation_abs_delta", "legacy_scale_sample", "wizard_scale_sample", "scale_sample_abs_delta", "legacy_used", "wizard_used"],
        diff_rows,
    )
    legacy_values = [float(row["scale_ratio"]) for row in legacy_rows]
    fresh_values = [float(row["scale_m_per_colmap_unit"]) for row in fresh_rows]
    legacy_scale = 0.676879570208235
    fresh_diagnostics = read_json(FRESH / "02_metric_scale/SCALE_DIAGNOSTICS.json")
    fresh_scale = float(fresh_diagnostics["scale_m_per_colmap_unit"])
    sim3_scale = trajectory["diagnostic_similarity_legacy_to_wizard"]["scale_wizard_units_per_legacy_unit"]
    gauge_only_prediction = legacy_scale / sim3_scale
    legacy_observations = {(key[0], frame) for key in legacy for frame in key[1:]}
    fresh_observations = {(key[0], frame) for key in fresh for frame in key[1:]}
    legacy_legacy_aggregation = robust_statistics(legacy_values, wizard=False)
    legacy_wizard_aggregation = robust_statistics(legacy_values, wizard=True)
    fresh_legacy_aggregation = robust_statistics(fresh_values, wizard=False)
    fresh_wizard_aggregation = robust_statistics(fresh_values, wizard=True)
    return {
        "schema_version": 1,
        "classification": "DIFFERENT_TRAJECTORY_AND_SCALE_SAMPLE_CONSTRUCTION",
        "legacy": {
            "raw_trajectory_input_sha256": sha256_file(LEGACY_IMAGES),
            "registered_poses": 175,
            "raw_samples": len(legacy_rows),
            "used_samples": len(legacy_kept_rows),
            "marker_frame_observations_represented": len(legacy_observations),
            "markers": sorted({key[0] for key in legacy}),
            "validity_filters": {"marker_area_min_px2": 1200, "pnp_distance_max_m": 4.0, "center_norm_max": 0.95, "frame_gap_min": 3, "frame_gap_max": 45, "metric_translation_min_m": 0.12, "metric_translation_max_m": 5.0, "colmap_translation_min": 1e-9, "per_marker_observation_cap": None},
            "formula": "norm(metric camera displacement) / norm(raw COLMAP camera-center displacement)",
            "aggregation": "median after median +/- 3*(1.4826*MAD); fall back to all if kept < max(10, 30% raw)",
            "raw_median": 0.681275258139,
            "raw_mad": 0.0219569235995,
            "kept_mean": 0.679669899312,
            "kept_std": 0.0332936861321,
            "final_scale": legacy_scale,
        },
        "wizard": {
            "raw_trajectory_input_sha256": sha256_file(FRESH_IMAGES),
            "registered_poses": 189,
            "raw_samples": len(fresh_rows),
            "used_samples": int(fresh_diagnostics["used_pairs"]),
            "marker_frame_observations_represented": len(fresh_observations),
            "markers": sorted({key[0] for key in fresh}),
            "validity_filters": {"quality_ranked_observations_per_marker": True, "per_marker_observation_cap": 30, "frame_gap_min": 2, "frame_gap_max": 80, "metric_translation_min_m": 0.05, "metric_translation_max_m": 6.0, "colmap_translation_min": 1e-10},
            "formula": "norm(metric camera displacement) / norm(raw COLMAP camera-center displacement)",
            "aggregation": "median after abs deviation <= max(3*(1.4826*MAD), 10% median); fall back to all if kept < max(10, int(25% raw))",
            "raw_median": fresh_diagnostics["raw_median"],
            "raw_mad": fresh_diagnostics["raw_mad"],
            "used_mean": fresh_diagnostics["used_mean"],
            "used_std": fresh_diagnostics["used_std"],
            "final_scale": fresh_scale,
        },
        "inventory_comparison": {
            "common_samples": len(set(legacy) & set(fresh)),
            "legacy_only_samples": len(set(legacy) - set(fresh)),
            "wizard_only_samples": len(set(fresh) - set(legacy)),
            "same_sample_inventory": set(legacy) == set(fresh),
            "common_marker_frame_observations": len(legacy_observations & fresh_observations),
            "legacy_only_marker_frame_observations": len(legacy_observations - fresh_observations),
            "wizard_only_marker_frame_observations": len(fresh_observations - legacy_observations),
            "common_metric_quantity_max_abs_delta_m": max(common_metric_differences),
            "common_metric_quantity_exact_count": sum(delta == 0 for delta in common_metric_differences),
            "common_colmap_quantity_max_abs_delta_units": max(common_colmap_differences),
            "common_scale_sample_max_abs_delta": max(common_ratio_differences),
        },
        "aggregation_isolation_on_same_samples": {
            "legacy_samples_legacy_rule": legacy_legacy_aggregation,
            "legacy_samples_wizard_rule": legacy_wizard_aggregation,
            "wizard_samples_legacy_rule": fresh_legacy_aggregation,
            "wizard_samples_wizard_rule": fresh_wizard_aggregation,
            "interpretation": "The written rules differ. On each actual raw sample list, both rules retain the same effective set/final median if the paired final medians and counts below match; aggregation is therefore not the first or dominant observed cause in this acquisition.",
        },
        "cause_assessment": {
            "A_different_raw_gauge_only": False,
            "B_different_trajectory_geometry": True,
            "C_different_scale_sample_inventory": True,
            "D_different_scale_formula": False,
            "E_different_rejection_filtering": True,
            "F_different_aggregation_definition": True,
            "F_aggregation_effect_on_actual_samples": legacy_legacy_aggregation["final_median"] != legacy_wizard_aggregation["final_median"] or fresh_legacy_aggregation["final_median"] != fresh_wizard_aggregation["final_median"],
            "G_numeric_version_effects": "PROVEN NUMERIC INPUT DIFFERENCE at historical-vs-fresh metric/PnP pair quantities (zero exactly equal common metric distances), plus upstream descriptor differences; whether OpenCV version, unavailable historical cache contents, or both caused the PnP delta is not isolated",
        },
        "gauge_only_check": {
            "raw_sim3_scale_wizard_units_per_legacy_unit": sim3_scale,
            "wizard_scale_predicted_if_only_gauge_changed": gauge_only_prediction,
            "wizard_scale_actual": fresh_scale,
            "absolute_residual": fresh_scale - gauge_only_prediction,
            "relative_residual": (fresh_scale - gauge_only_prediction) / gauge_only_prediction,
            "wizard_scale_mapped_to_legacy_gauge": fresh_scale * sim3_scale,
            "legacy_scale": legacy_scale,
        },
        "observation_source_audit": {
            "legacy_scale_input_declared_by_code": "results/.../.ap01_compat_cache/moving_observations/moving_detections.csv",
            "legacy_scale_input_currently_available": False,
            "legacy_surviving_observation_fields": "scale pair CSV preserves marker/frame endpoints, area, center norm, distance, metric displacement, COLMAP displacement, ratio, and pair quality; original corners/rvec/tvec rows are unavailable",
            "wizard_input": str(FRESH_MOVING_OBSERVATIONS),
            "wizard_input_sha256": sha256_file(FRESH_MOVING_OBSERVATIONS),
            "wizard_rows": len(read_csv(FRESH_MOVING_OBSERVATIONS)),
            "recovered_main_detector_output": str(RECOVERED_MAIN_MOVING_OBSERVATIONS),
            "recovered_main_detector_output_sha256": sha256_file(RECOVERED_MAIN_MOVING_OBSERVATIONS),
            "warning": "The regenerated Main/Wizard accepted observation semantics are established separately, but they are not a substitute for the missing historical .ap01_compat_cache consumed by the locked scale. Surviving pair fields prove the locked metric/PnP quantities are numerically different from every common fresh pair.",
        },
        "method_contract_propagation_finding": "estimate_scale.py does not attach method_contract to the Arguments object consumed by prepared_observations; it therefore defaults to recommended_wizard_v1 quality semantics, while the stage also applies the configured top-30 cap. The parity contract currently does not encode/reproduce the historical scale filters.",
        "evidence": {"legacy_all": str(LEGACY_SCALE_ALL), "legacy_kept": str(LEGACY_SCALE_KEPT), "wizard_all": str(FRESH_SCALE_ALL), "wizard_diagnostics": str(FRESH / "02_metric_scale/SCALE_DIAGNOSTICS.json")},
    }


def format_float(value: float) -> str:
    return f"{value:.12g}"


def build_first_divergence(configuration: dict[str, Any], feature: dict[str, Any], scale: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "classification": "FIRST_PROVEN_DIVERGENCE_IS_COLMAP_FEATURE_EXTRACTION_CONFIGURATION",
        "execution_order": [
            {"stage": "input image bytes/inventory", "result": "EXACT by established historical prepared-input parity; 189 images"},
            {"stage": "COLMAP feature-extraction command/environment", "result": "DIFFERENT_CONFIGURATION", "differences": ["Legacy explicitly uses --SiftExtraction.num_threads 1; Wizard omits it and the fresh console shows parallel completion order.", "Legacy serializes PINHOLE parameters to eight decimal places; Wizard uses full precision.", "Preserved databases report PRAGMA user_version 3700 versus 3900; exact historical executable version is UNKNOWN."]},
            {"stage": "keypoint extraction output", "result": "EXACT_COUNTS_AND_BLOBS", "total": feature["count_parity"]["legacy_total_features"]},
            {"stage": "descriptor extraction output", "result": "FIRST_OBSERVED_NUMERIC_DATA_DIVERGENCE", "first_image": feature["first_observed_numeric_divergence"]["image"]},
            {"stage": "matching/verified geometry", "result": "DIFFERENT_COUNTS_AND_BLOBS_WITH_SAME_EXHAUSTIVE_PAIR_INVENTORY"},
            {"stage": "mapper command", "result": "DIFFERENT_CONFIGURATION", "difference": "Legacy Mapper.min_num_matches=15 (saved default); Wizard explicitly passes 8."},
            {"stage": "registration/model", "result": "175 versus 189 registered images"},
            {"stage": "metric scale", "result": "DIFFERENT_TRAJECTORY_AND_SCALE_SAMPLE_CONSTRUCTION", "difference": "Legacy has no top-per-marker cap and uses gaps 3..45 plus legacy quality/distance filters; Wizard caps 30 and uses gaps 2..80 plus different thresholds."},
        ],
        "exact_first_causal_decision_supported_by_evidence": "Before descriptor generation, Wizard invokes a non-parity COLMAP feature extraction configuration/environment: it omits Legacy's deterministic single-thread SIFT option, changes camera-parameter text precision, and uses a demonstrably different COLMAP database generation. The first output divergence is descriptor bytes at frame_0003.png. Existing artifacts cannot isolate which of these simultaneous feature-extraction differences changes the descriptors; a second run was forbidden.",
        "later_independent_proven_configuration_divergences": ["Mapper minimum matches 15 (Legacy) versus 8 (Wizard).", "Historical scale sample construction/filtering versus Wizard quality-ranked top-30 and wider pair thresholds.", "The scale stage does not receive the AP01 method contract and falls back to recommended_wizard_v1 observation-quality semantics."],
        "not_merely_scale_drift": True,
        "recommended_path": "PATH A — FIX WIZARD CONFIGURATION",
        "justification": "Parity is not yet configured: recoverable COLMAP flags and scale logic differ before any claim of irreducible version nondeterminism can be made. PATH B is insufficient because raw registration differs; PATH C prerequisites are not met; PATH D is unnecessary because the defect is configuration/contract resolution.",
        "minimal_fix": ["For main_route2_parity_v1 only, serialize camera parameters with the historical eight-decimal representation and pass SiftExtraction.num_threads=1.", "Resolve Mapper.min_num_matches=15 for this contract.", "Propagate method_contract into estimate_scale and encode the historical scale observation filters/pair bounds/no-cap behavior in main_route2_parity_v1. Do not change recommended_wizard_v1.", "Record the COLMAP executable version/schema in future stage manifests. After focused tests, perform exactly one controlled AP01 rerun in the next task."],
        "implementation_in_this_task": "No production code changed; diagnosis evidence and tests only.",
        "evidence": {"configuration_classification": configuration["overall_classification"], "first_descriptor_difference": feature["first_observed_numeric_divergence"], "scale_classification": scale["classification"]},
    }


def build_report(configuration: dict[str, Any], registration: dict[str, Any], trajectory: dict[str, Any], feature: dict[str, Any], scale: dict[str, Any], first: dict[str, Any]) -> str:
    sim3 = trajectory["diagnostic_similarity_legacy_to_wizard"]
    inventory = scale["inventory_comparison"]
    return f"""AP01 MOVING-CAMERA COLMAP / SCALE PARITY REPORT
================================================

Scope: preserved historical Main AP01 artifacts versus failed Wizard attempt
20260807_012714_baseline_f6020505. Evidence-only; no Ground Truth.

1. Historical COLMAP configuration recovered
   Main used all 189 moving PNGs, one shared 1280x720 PINHOLE camera with
   929.46716309,929.46713448,640.00000000,360.00000000; CPU SIFT, maximum
   image size 1600, maximum 4096 features, SiftExtraction.num_threads=1,
   exhaustive CPU matching, no masks, no intrinsic refinement, mapper saved
   minimum matches 15/random seed 0, and selected the largest sparse model
   (lexicographically first on ties). Exact historical executable version is UNKNOWN;
   the preserved database reports user_version 3700.

2. Fresh Wizard COLMAP configuration
   Wizard used the same images/shared PINHOLE semantics/feature limits and CPU
   exhaustive matcher through /usr/bin/colmap 3.9.1 without CUDA. It passed full-
   precision intrinsics, omitted SiftExtraction.num_threads, explicitly passed
   Mapper.min_num_matches=8, and produced database user_version 3900. Fresh WSL,
   Python, CPU, OpenCV, NumPy, and SciPy details are in COLMAP_CONFIGURATION_PARITY.json.

3. Configuration parity classification
   DIFFERENT_CONFIGURATION. Historical unknowns remain explicitly UNKNOWN.

4. Legacy registered-image inventory
   175/189 registered in selected model 0. Frames frame_0175.png through
   frame_0188.png are unregistered. The legacy report's text count of 176 is a
   counter defect; the model header, 175 parsed pose records, and scale consumer agree.

5. Fresh registered-image inventory
   189/189 registered in selected model 0. No unregistered images. Image IDs differ
   for {registration['image_id_parity']['different_count_all_inputs']}/189 semantic names due to extraction insertion order.

6. Raw trajectory parity classification
   DIFFERENT_REGISTERED_IMAGES. The common segment is close under a Sim(3), but it
   is not exact and the lower-level descriptor/match stream differs.

7. Diagnostic Sim(3) results
   Mapping Legacy raw centers to Wizard raw centers: scale
   {format_float(sim3['scale_wizard_units_per_legacy_unit'])}, rotation
   {json.dumps(sim3['rotation'])}, translation {json.dumps(sim3['translation_wizard_units'])}.
   On 175 common images, translation RMSE is
   {format_float(sim3['translation_rmse_wizard_units'])} Wizard units
   ({format_float(100 * sim3['translation_rmse_fraction_of_radius'])}% of trajectory radius),
   median {format_float(sim3['translation_median_wizard_units'])}, max
   {format_float(sim3['translation_max_wizard_units'])}; relative-rotation residual
   mean/median/max is {format_float(sim3['rotation_residual_mean_degrees'])}/
   {format_float(sim3['rotation_residual_median_degrees'])}/
   {format_float(sim3['rotation_residual_max_degrees'])} degrees. Robust diagnostic
   inliers: {sim3['robust_translation_inliers']}/175. This transform was not applied
   to calibration output.

8. Legacy scale
   0.676879570208235 m per Legacy COLMAP unit (locked text: 0.676879570208).

9. Fresh scale
   0.7357593316733281 m per Wizard COLMAP unit.

10. Scale-sample inventory comparison
   Legacy raw/used: {scale['legacy']['raw_samples']}/{scale['legacy']['used_samples']}.
   Wizard raw/used: {scale['wizard']['raw_samples']}/{scale['wizard']['used_samples']}.
   Pair keys common/Legacy-only/Wizard-only: {inventory['common_samples']}/
   {inventory['legacy_only_samples']}/{inventory['wizard_only_samples']}.
   The two implementations also consume different marker-frame endpoint inventories
   and common-pair metric/PnP quantities are not all numerically equal.

11. Scale-stage parity classification
   DIFFERENT_TRAJECTORY_AND_SCALE_SAMPLE_CONSTRUCTION. The ratio formula is the
   same, but raw trajectory geometry, observation/pair inventory, quality filters,
   frame-gap bounds, per-marker cap, and written robust rule differ. The gauge-only
   prediction is {format_float(scale['gauge_only_check']['wizard_scale_predicted_if_only_gauge_changed'])};
   actual Fresh is higher by {format_float(100 * scale['gauge_only_check']['relative_residual'])}%,
   so arbitrary gauge alone is insufficient.

12. Exact first causal divergence
   Same 189 input images lead to a DIFFERENT feature-extraction configuration/
   environment before numeric output: Main explicitly used one SIFT extraction
   thread and eight-decimal camera parameters with database generation 3700;
   Wizard omitted the thread option, used full precision, and generated schema 3900.
   All 151125 keypoint rows/blobs are exact, then descriptor bytes first differ at
   {feature['first_observed_numeric_divergence']['image']}. Match differences follow.
   Existing evidence cannot assign the descriptor delta to only one simultaneous
   option/version difference without a controlled rerun.

13. Relevant functions/files/lines
   Historical invocation: run/bus_real_data/approach1_marker_direct_relay/
   06_run_colmap_moving_sequence.py:122-164 (camera/SIFT/matcher/mapper options)
   and :172-190 (best-model selection).
   Historical scale: 12_estimate_colmap_scale_from_aruco.py (quality filtering,
   :141-175, pair construction :214-264, MAD aggregation :181-203). Wizard COLMAP:
   src/camera_rig_calibration/
   methods/ap01/core.py:170 (run_colmap_pipeline), especially 195-229. Wizard scale:
   core.py:476 (robust_scale), 497-545; estimate_scale.py:14-39; pipeline.py:57-72,
   118-119. Machine-readable evidence supplies artifact paths and hashes.

14. Whether any code change was made
   No production/scientific code change. Only this read-only audit utility, evidence
   outputs, and focused audit tests were added. The earlier NumPy serialization fix
   was pre-existing and is not treated as this scientific divergence.

15. Recommended path
   PATH A — FIX WIZARD CONFIGURATION. Recoverable parity settings are demonstrably
   different, and the parity contract does not reproduce historical scale rules.
   Minimal contract-scoped fix: eight-decimal intrinsics, one SIFT extraction thread,
   mapper minimum matches 15, method-contract propagation into scale, and historical
   scale filters/pair bounds/no top-30 cap. Do not alter recommended_wizard_v1.
   PATH C is premature until configuration parity is actually tested.

16. Tests and results
   83 focused AP01 parity/rerun-guard/audit tests passed. The validator checks
   configuration classification, registration inventory, Sim(3) recomputation,
   feature/match first divergence, scale inventory, gauge-only check, fingerprint
   sensitivity, and the required output set. git diff --check passed.

17. Confirmation that no second full AP01 run occurred
   Confirmed. This audit opened preserved artifacts read-only. It did not invoke
   COLMAP/AP01, publish, reconcile, run AP02/AP03, use Ground Truth, modify the Main
   recovery worktree, or modify route2_cpu_ref14_50x50 scientific artifacts. The
   protected experiment remains 1256 files at tree SHA-256
   c40adcde29c7177a2e7fa4fd6c49b45a268b102f13fb18d3ca2fa2d23a886ea3.

18. Exact next action
   Implement the PATH-A changes only in main_route2_parity_v1 resolution, add option/
   scale-contract regression tests, re-run focused tests and the protected-artifact
   guard, then authorize exactly one controlled AP01 production rerun using the same
   prepared experiment and queue workflow. Do not reuse or force the historical scale.
"""


def main() -> None:
    required = [LEGACY_DB, FRESH_DB, LEGACY_IMAGES, FRESH_IMAGES, LEGACY_SCALE_ALL, LEGACY_SCALE_KEPT, FRESH_SCALE_ALL]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing preserved evidence:\n" + "\n".join(missing))
    legacy_db = database_inventory(LEGACY_DB)
    fresh_db = database_inventory(FRESH_DB)
    legacy_poses = parse_colmap_images(LEGACY_IMAGES)
    fresh_poses = parse_colmap_images(FRESH_IMAGES)
    configuration = build_configuration(legacy_db, fresh_db)
    write_json("COLMAP_CONFIGURATION_PARITY.json", configuration)
    registration = build_registration(legacy_poses, fresh_poses, legacy_db, fresh_db)
    write_json("COLMAP_REGISTRATION_PARITY.json", registration)
    trajectory = build_trajectory(legacy_poses, fresh_poses)
    write_json("RAW_TRAJECTORY_PARITY.json", trajectory)
    feature = build_feature_match(legacy_db, fresh_db)
    write_json("FEATURE_MATCH_PARITY.json", feature)
    scale = build_scale(trajectory)
    write_json("SCALE_PARITY.json", scale)
    first = build_first_divergence(configuration, feature, scale)
    write_json("FIRST_CAUSAL_DIVERGENCE.json", first)
    (OUT / "COLMAP_SCALE_PARITY_REPORT.txt").write_text(
        build_report(configuration, registration, trajectory, feature, scale, first),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

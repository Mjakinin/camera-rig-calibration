"""GT-free AP01 candidate-construction and aggregate-selection parity evidence.

This module intentionally stops at candidate aggregation/selection.  It never
calls an AP01 stage runner, a solver, COLMAP, an evaluator, or a result
publisher.  The two adapters below consume the same immutable JSONL rows and
the same already-existing moving-camera poses/metric scale.
"""

from __future__ import annotations

import csv
import base64
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from camera_rig_calibration.methods.ap01 import core
from camera_rig_calibration.methods.ap01.build_candidates import (
    construct_candidates,
)
from camera_rig_calibration.methods.ap01.contracts import (
    AP01MethodContract,
    resolve_ap01_method_contract,
)
from camera_rig_calibration.methods.ap01.solve_extrinsics import (
    compare_paths,
    evaluate_path_gate,
    select_candidate_aggregates,
)

from .evidence import write_csv, write_json
from .inventory import assert_pre_solver_path, sha256_file
from .observation_parity import (
    compare_semantic_rows,
    load_observation_csv,
    semantic_observation_rows,
)


ROOT_CAMERA = "cam_edge_3"
CAMERA_ORDER = ("cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5")
LEGACY_DIRECT_TARGETS = ("cam_edge_1",)
LEGACY_RELAY_TARGETS = ("cam_edge_0", "cam_edge_1", "cam_edge_5")
WIZARD_TOP_MOVING_PER_MARKER = 8

NUMERIC_TOLERANCES = {
    "rotation_matrix_absolute": 1e-12,
    "translation_m_absolute": 1e-12,
    "quality_absolute": 1e-12,
    "score_absolute": 1e-12,
    "reprojection_px_absolute": 1e-12,
}

CANDIDATE_DIFF_FIELDS = (
    "phase",
    "semantic_candidate_key",
    "candidate_type",
    "target_camera",
    "root_marker",
    "target_marker",
    "root_frame",
    "target_frame",
    "field",
    "legacy_value",
    "wizard_value",
    "absolute_delta",
    "tolerance",
    "reason",
)

CANDIDATE_JSONL_FIELDS = (
    "schema_version",
    "implementation",
    "semantic_candidate_key",
    "candidate_type",
    "root_camera",
    "target_camera",
    "root_marker",
    "target_marker",
    "root_frame",
    "target_frame",
    "relay_path",
    "transform_chain",
    "composed_rotation_rodrigues",
    "composed_translation_m",
    "support_observation_keys",
    "support_count",
    "quality_values",
    "reprojection_values_px",
    "score_components",
    "aggregate_score",
    "acceptance_decision",
    "rejection_reason",
    "aggregate_decisions",
    "original_construction_index",
    "original_selection_ranking_index",
    "deterministic_tie_break_fields",
    "native_candidate",
)

SELECTION_DIFF_FIELDS = (
    "camera_id",
    "field",
    "legacy_value",
    "wizard_value",
    "reason",
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            payload = (_json(_json_safe(dict(row))) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
    return digest.hexdigest()


def _write_candidate_jsonl(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> str:
    """Write compact positional JSONL with one self-describing schema line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    camera_codes = {
        "0": "cam_edge_0",
        "1": "cam_edge_1",
        "3": "cam_edge_3",
        "5": "cam_edge_5",
    }
    implementation_codes = {"L": "legacy", "W": "wizard"}
    type_codes = {"g": "root", "d": "direct", "r": "relay"}
    acceptance_values = sorted(
        {str(record["acceptance_decision"]) for record in records}
    )
    rejection_values = sorted(
        {
            str(record["rejection_reason"])
            for record in records
            if record["rejection_reason"] is not None
        }
    )
    acceptance_codes = {
        value: index for index, value in enumerate(acceptance_values)
    }
    rejection_codes = {
        value: index for index, value in enumerate(rejection_values)
    }
    header = {
        "record_type": "schema",
        "schema_version": 1,
        "encoding": "each following line is a JSON array matching fields",
        "fields": list(CANDIDATE_JSONL_FIELDS),
        "rotation_representation": "Rodrigues vector, radians",
        "transform_direction": "target optical camera to cam_edge_3 root",
        "support_key_registry": "frozen/ap01_accepted_observations.jsonl",
        "compact_value_contract": {
            "implementation": implementation_codes,
            "camera": camera_codes,
            "candidate_type": type_codes,
            "semantic_candidate_key": "URL-safe base64 of the SHA-256 digest",
            "relay_path": "[root_moving_frame,target_moving_frame]; moving camera is moving_calib_camera",
            "transform_chain": {
                "g": "I_4x4; root->root",
                "d": "T_root_marker @ inverse(T_target_marker); target->root; supports=[root_static,target_static]",
                "r": (
                    "T_root_marker @ inverse(T_moving_i_marker) @ "
                    "scaled(Tcw_i @ inverse(Tcw_j)) @ "
                    "inverse(T_target_marker @ inverse(T_moving_j_marker)); "
                    "target->root; supports=[root_static,target_static,root_moving,target_moving]"
                ),
            },
            "support_observation_keys": (
                "integer N maps to frozen observation key ap01obs:{N:06d}"
            ),
            "quality_and_reprojection_array_order": {
                "g": [],
                "d": ["root_static", "target_static"],
                "r": [
                    "root_static",
                    "target_static",
                    "root_moving",
                    "target_moving",
                ],
            },
            "score_components": {
                "LD": "legacy sqrt(q_root_static*q_target_static)",
                "LR": "legacy fourth-root product of four ordered quality values",
                "WD": "Wizard sqrt(q_root_static*q_target_static)",
                "WR": "Wizard fourth-root product of four ordered quality values",
                "G": "root gauge; unavailable",
            },
            "acceptance_decision": {
                str(index): value for value, index in acceptance_codes.items()
            },
            "rejection_reason": {
                str(index): value for value, index in rejection_codes.items()
            },
            "aggregate_decisions": {
                "legacy_direct": [
                    "quality_filter_eligible",
                    "quality_filter_fallback_used",
                    "quality_filtered_mad_inlier",
                    "preferred_marker_selected",
                ],
                "legacy_relay": ["flat_mad_inlier"],
                "wizard_direct": [
                    "robust_inlier",
                    "pose_support",
                    "path_stable",
                    "path_selected_for_diagnostic_estimate",
                    "path_deployment_eligible",
                ],
                "wizard_relay": [
                    "within_chain_robust_inlier",
                    "within_chain_pose_support",
                    "independent_chain_robust_inlier",
                    "independent_chain_pose_support",
                    "path_stable",
                    "path_selected_for_diagnostic_estimate",
                    "path_deployment_eligible",
                ],
            },
            "tie_break_code": {
                "g": [],
                "d": ["shared_marker_asc"],
                "r": [
                    "root_marker_asc",
                    "target_marker_asc",
                    "root_frame_asc",
                    "target_frame_asc",
                ],
            },
        },
    }
    with path.open("wb") as handle:
        payload = (_json(header) + "\n").encode("utf-8")
        handle.write(payload)
        digest.update(payload)
        for record in records:
            rotation_vector, _ = cv2.Rodrigues(record["_T"][:3, :3])
            candidate_type = str(record["candidate_type"])
            type_code = next(
                code for code, value in type_codes.items() if value == candidate_type
            )
            implementation_code = next(
                code
                for code, value in implementation_codes.items()
                if value == record["implementation"]
            )
            root_camera_code = next(
                code
                for code, value in camera_codes.items()
                if value == record["root_camera"]
            )
            target_camera_code = next(
                code
                for code, value in camera_codes.items()
                if value == record["target_camera"]
            )
            digest_hex = str(record["semantic_candidate_key"]).split(":", 1)[1]
            compact_key = base64.urlsafe_b64encode(bytes.fromhex(digest_hex)).decode(
                "ascii"
            ).rstrip("=")
            order = header["compact_value_contract"][
                "quality_and_reprojection_array_order"
            ][type_code]
            aggregate = record.get("aggregate_decisions") or {}
            if candidate_type == "root":
                relay_path = None
                transform_chain = "g"
                score_code = "G"
                aggregate_values: list[Any] = []
                ranking: Any = None
                tie_code = "g"
            elif candidate_type == "direct":
                relay_path = None
                transform_chain = ["d", record["transform_chain"]["id"]]
                score_code = implementation_code + "D"
                decision_fields = header["compact_value_contract"][
                    "aggregate_decisions"
                ][
                    "legacy_direct"
                    if implementation_code == "L"
                    else "wizard_direct"
                ]
                aggregate_values = [bool(aggregate.get(field)) for field in decision_fields]
                ranking = record["original_selection_ranking_index"]
                tie_code = "d"
            else:
                relay_path = [record["root_frame"], record["target_frame"]]
                transform_chain = ["r", record["transform_chain"]["id"]]
                score_code = implementation_code + "R"
                decision_fields = header["compact_value_contract"][
                    "aggregate_decisions"
                ][
                    "legacy_relay"
                    if implementation_code == "L"
                    else "wizard_relay"
                ]
                aggregate_values = [bool(aggregate.get(field)) for field in decision_fields]
                source_ranking = record["original_selection_ranking_index"]
                ranking = (
                    [
                        source_ranking["root_moving_quality_rank"],
                        source_ranking["target_moving_quality_rank"],
                    ]
                    if source_ranking
                    else None
                )
                tie_code = "r"
            values = (
                record["schema_version"],
                implementation_code,
                compact_key,
                type_code,
                root_camera_code,
                target_camera_code,
                record["root_marker"],
                record["target_marker"],
                record["root_frame"],
                record["target_frame"],
                relay_path,
                transform_chain,
                rotation_vector.reshape(3).tolist(),
                record["_T"][:3, 3].tolist(),
                [
                    int(str(key).split(":", 1)[1])
                    for key in record["support_observation_keys"]
                ],
                record["support_count"],
                [record["quality_values"][field] for field in order],
                [record["reprojection_values_px"][field] for field in order],
                score_code,
                record["aggregate_score"],
                acceptance_codes[str(record["acceptance_decision"])],
                (
                    rejection_codes[str(record["rejection_reason"])]
                    if record["rejection_reason"] is not None
                    else None
                ),
                aggregate_values,
                record["original_construction_index"],
                ranking,
                tie_code,
                int(bool(record["native_candidate"])),
            )
            payload = (_json(_json_safe(values)) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert_pre_solver_path(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _observation_key(row: Mapping[str, Any]) -> str:
    return f"ap01obs:{int(row['original_index']):06d}"


def _canonical_row(
    semantic: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    corners = [
        [semantic[f"corner{index}_u"], semantic[f"corner{index}_v"]]
        for index in range(4)
    ]
    quality_fields = (
        "selection_score",
        "score_area",
        "score_reprojection",
        "score_border",
        "score_distance",
    )
    result = {
        "schema_version": 1,
        "original_index": semantic["original_index"],
        "source_kind": semantic["source_kind"],
        "camera_id": semantic["camera_id"],
        "observer_id": semantic["observer_id"],
        "frame_id": semantic["frame_id"],
        "marker_id": semantic["marker_id"],
        "occurrence_index": semantic["occurrence_index"],
        "image_path": str(source.get("image_path", "")),
        "marker_length_m": semantic["marker_length_m"],
        "corners_px": corners,
        "pnp": {
            "success": semantic["pnp_success"],
            "rotation_vector": [semantic[f"rvec_{axis}"] for axis in "xyz"],
            "translation_m": [
                semantic[f"tvec_{axis}_m"] for axis in "xyz"
            ],
            "reprojection_rmse_px": semantic["pnp_reprojection_rmse_px"],
            "convention": "marker_coordinates_to_observer_optical_camera",
        },
        "geometry": {
            "distance_m": semantic["distance_m"],
            "center_u": semantic["center_u"],
            "center_v": semantic["center_v"],
            "area_px2": semantic["area_px2"],
        },
        "camera_model": {
            "fx": semantic["fx"],
            "fy": semantic["fy"],
            "cx": semantic["cx"],
            "cy": semantic["cy"],
            "distortion_model": semantic["distortion_model"],
            "distortion_coefficients": list(
                semantic["distortion_coefficients"]
            ),
            "image_width_px": semantic["image_width_px"],
            "image_height_px": semantic["image_height_px"],
        },
        "wizard_observation_quality": {
            field: float(source[field])
            for field in quality_fields
            if str(source.get(field, "")).strip()
        },
        "filter": {
            "decision": str(source.get("decision", "accepted")),
            "reason": str(source.get("reason", "accepted")),
            "threshold": str(source.get("threshold", "")),
            "measured_value": str(source.get("measured_value", "")),
        },
        "detection_metadata": {
            field: source.get(field, "")
            for field in (
                "detection_mode",
                "detection_source",
                "detection_support",
                "detector_contract",
                "opencv_version",
                "detector_parameters_json",
            )
        },
    }
    result["observation_key"] = _observation_key(result)
    return result


def freeze_ap01_input(
    *,
    legacy_accepted_csv: Path,
    wizard_accepted_csv: Path,
    colmap_images_txt: Path,
    metric_scale_txt: Path,
    frozen_root: Path,
) -> dict[str, Any]:
    """Freeze exact accepted rows and common, already-existing relay inputs."""

    for path in (
        legacy_accepted_csv,
        wizard_accepted_csv,
        colmap_images_txt,
        metric_scale_txt,
        frozen_root,
    ):
        assert_pre_solver_path(path)
    legacy_rows = load_observation_csv(legacy_accepted_csv)
    wizard_rows = load_observation_csv(wizard_accepted_csv)
    comparison, differences = compare_semantic_rows(
        legacy_rows, wizard_rows, complete_diff=True
    )
    if differences or not comparison["original_order_parity"]:
        raise RuntimeError(
            "accepted observation sources are not exact and order-preserving"
        )

    semantic = semantic_observation_rows(wizard_rows)
    canonical = [
        _canonical_row(item, source)
        for item, source in zip(semantic, wizard_rows, strict=True)
    ]
    forbidden_fields = {
        key
        for row in canonical
        for key in row
        if "ground_truth" in key.lower() or key.lower() == "gt"
    }
    if forbidden_fields:
        raise RuntimeError(f"Ground Truth fields in frozen input: {forbidden_fields}")

    frozen_root.mkdir(parents=True, exist_ok=True)
    observations_path = frozen_root / "ap01_accepted_observations.jsonl"
    observation_hash = _write_jsonl(observations_path, canonical)
    frozen_colmap = frozen_root / "ap01_colmap_images.txt"
    frozen_scale = frozen_root / "ap01_metric_scale.txt"
    shutil.copyfile(colmap_images_txt, frozen_colmap)
    shutil.copyfile(metric_scale_txt, frozen_scale)

    duplicate_count = sum(
        count - 1
        for count in Counter(
            (
                row["source_kind"],
                row["camera_id"],
                row["frame_id"],
                row["marker_id"],
            )
            for row in canonical
        ).values()
    )
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_EXACT",
        "source_artifacts": {
            "legacy_accepted": {
                "path": str(legacy_accepted_csv.resolve()),
                "sha256": sha256_file(legacy_accepted_csv.resolve()),
            },
            "wizard_accepted": {
                "path": str(wizard_accepted_csv.resolve()),
                "sha256": sha256_file(wizard_accepted_csv.resolve()),
            },
        },
        "canonical_input": {
            "path": str(observations_path.resolve()),
            "sha256": observation_hash,
            "row_count": len(canonical),
        },
        "common_relay_inputs": {
            "moving_poses": {
                "path": str(frozen_colmap.resolve()),
                "sha256": sha256_file(frozen_colmap),
                "provenance": "preserved Main COLMAP images.txt; read only, COLMAP not invoked",
            },
            "metric_scale": {
                "path": str(frozen_scale.resolve()),
                "sha256": sha256_file(frozen_scale),
                "value": float(frozen_scale.read_text(encoding="utf-8").strip()),
                "provenance": "preserved no-GT ArUco metric scale; scale estimation not invoked",
            },
        },
        "schema": {
            "key": [
                "source_kind",
                "camera_id",
                "frame_id",
                "marker_id",
                "occurrence_index",
            ],
            "ap01_fields": [
                "corners_px",
                "pnp",
                "geometry",
                "camera_model",
                "wizard_observation_quality",
                "filter",
                "detection_metadata",
            ],
        },
        "ordering_contract": "original accepted CSV row order; no sorting",
        "duplicate_row_contract": (
            "duplicates are retained and disambiguated by zero-based occurrence_index"
        ),
        "duplicate_base_key_count": duplicate_count,
        "source_semantic_parity": "EXACT",
        "source_original_order_parity": True,
        "ground_truth_fields_present": False,
        "ground_truth_used": False,
        "marker_redetection_performed": False,
        "colmap_invoked": False,
        "solver_invoked": False,
    }
    write_json(frozen_root / "AP01_INPUT_MANIFEST.json", manifest)
    return manifest


def _manual_rodrigues(vector: Sequence[float]) -> np.ndarray:
    rvec = np.asarray(vector, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    unit = rvec / theta
    skew = np.asarray(
        (
            (0.0, -unit[2], unit[1]),
            (unit[2], 0.0, -unit[0]),
            (-unit[1], unit[0], 0.0),
        ),
        dtype=np.float64,
    )
    return (
        np.eye(3)
        + math.sin(theta) * skew
        + (1.0 - math.cos(theta)) * (skew @ skew)
    )


def _make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _observation_transform(row: Mapping[str, Any], *, legacy: bool) -> np.ndarray:
    rvec = row["pnp"]["rotation_vector"]
    if legacy:
        rotation = _manual_rodrigues(rvec)
    else:
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return _make_transform(rotation, row["pnp"]["translation_m"])


def _polygon_area(corners: Sequence[Sequence[float]]) -> float:
    points = np.asarray(corners, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    return float(
        0.5
        * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def _legacy_quality(row: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    distance = float(row["geometry"]["distance_m"])
    area = _polygon_area(row["corners_px"])
    center_u = float(row["geometry"]["center_u"])
    center_v = float(row["geometry"]["center_v"])
    center_norm = math.hypot(center_u - 640.0, center_v - 360.0)
    center_norm /= math.hypot(640.0, 360.0)
    if not math.isfinite(distance) or distance <= 0.0:
        distance = 99.0
    if not math.isfinite(area) or area <= 0.0:
        area = 1.0
    if not math.isfinite(center_norm):
        center_norm = 1.0
    quality = area / (distance * distance * (1.0 + center_norm))
    return quality, {
        "area_px2_from_corners": area,
        "distance_m": distance,
        "center_norm_1280x720": center_norm,
    }


def _prepared_rows(
    frozen: Sequence[Mapping[str, Any]], *, implementation: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for source in frozen:
        item = dict(source)
        item["_camera"] = source["camera_id"]
        item["_marker"] = int(source["marker_id"])
        item["_frame"] = (
            int(source["frame_id"])
            if source["source_kind"] == "moving"
            else ""
        )
        item["_T_cam_marker"] = _observation_transform(
            source, legacy=implementation == "legacy"
        )
        if implementation == "legacy":
            item["_quality"], item["_quality_components"] = _legacy_quality(
                source
            )
        else:
            item["_quality"] = float(
                source["wizard_observation_quality"]["selection_score"]
            )
            item["_quality_components"] = dict(
                source["wizard_observation_quality"]
            )
        item["_area_px2"] = _polygon_area(source["corners_px"])
        item["_distance_m"] = float(source["geometry"]["distance_m"])
        rows.append(item)
    return (
        [row for row in rows if row["source_kind"] == "static"],
        [row for row in rows if row["source_kind"] == "moving"],
    )


def _best_static(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["_camera"]), int(row["_marker"]))
        if key not in result or row["_quality"] > result[key]["_quality"]:
            result[key] = row
    return result


def _legacy_moving(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["_frame"]), int(row["_marker"]))
        if key not in best or row["_quality"] > best[key]["_quality"]:
            best[key] = row
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for (_, marker), row in best.items():
        grouped[marker].append(row)
    for marker in grouped:
        grouped[marker].sort(key=lambda row: int(row["_frame"]))
    return dict(grouped)


def _wizard_moving(
    rows: Sequence[dict[str, Any]], registered_frames: set[int]
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["_frame"]) in registered_frames:
            grouped[int(row["_marker"])].append(row)
    selected: dict[int, list[dict[str, Any]]] = {}
    ranking: list[dict[str, Any]] = []
    for marker, marker_rows in sorted(grouped.items()):
        ranked = sorted(
            marker_rows,
            key=lambda row: (-float(row["_quality"]), int(row["_frame"])),
        )
        kept = ranked[:WIZARD_TOP_MOVING_PER_MARKER]
        kept_keys = {row["observation_key"] for row in kept}
        for rank, row in enumerate(ranked, 1):
            ranking.append(
                {
                    "marker_id": marker,
                    "observation_key": row["observation_key"],
                    "frame_id": int(row["_frame"]),
                    "quality_rank": rank,
                    "selection_score": float(row["_quality"]),
                    "selected": row["observation_key"] in kept_keys,
                    "rejection_reason": (
                        None
                        if row["observation_key"] in kept_keys
                        else "quality_rank_exceeds_top_moving_per_marker_8"
                    ),
                    "tie_break_fields": [
                        "descending selection_score",
                        "ascending moving frame number",
                    ],
                }
            )
        selected[marker] = sorted(kept, key=lambda row: int(row["_frame"]))
    return selected, ranking


def _support_term(
    *,
    name: str,
    source_key: str,
    direction: str,
    operation: str = "as_observed",
) -> dict[str, Any]:
    return {
        "name": name,
        "source_observation_key": source_key,
        "operation": operation,
        "direction": direction,
    }


def _candidate_key(row: Mapping[str, Any]) -> str:
    fields = _json(
        {
            "target_camera": row["target_camera"],
            "candidate_type": row["candidate_type"],
            "root_marker": row.get("root_marker"),
            "target_marker": row.get("target_marker"),
            "root_frame": row.get("root_frame"),
            "target_frame": row.get("target_frame"),
            "support_observation_keys": row["support_observation_keys"],
            "transform_chain_identity": row["transform_chain"].get(
                "id", row["transform_chain"].get("identity")
            ),
        }
    )
    return "sha256:" + hashlib.sha256(fields.encode("utf-8")).hexdigest()


def _root_candidate(implementation: str) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "implementation": implementation,
        "candidate_type": "root",
        "root_camera": ROOT_CAMERA,
        "target_camera": ROOT_CAMERA,
        "root_marker": None,
        "target_marker": None,
        "root_frame": None,
        "target_frame": None,
        "relay_path": None,
        "transform_chain": {
            "id": "gauge_identity",
            "expression": "I_4x4",
            "terms": [],
            "direction": "root->root",
        },
        "rotation": np.eye(3),
        "translation_m": np.zeros(3),
        "support_observation_keys": [],
        "support_count": 0,
        "quality_values": {},
        "reprojection_values_px": {},
        "score_components": {},
        "aggregate_score": None,
        "acceptance_decision": "accepted_root_gauge",
        "rejection_reason": None,
        "original_construction_index": None,
        "original_selection_ranking_index": 0,
        "deterministic_tie_break_fields": [],
        "native_candidate": False,
        "_T": np.eye(4, dtype=np.float64),
        "_quality": None,
    }
    result["semantic_candidate_key"] = _candidate_key(result)
    return result


def _direct_record(
    *,
    implementation: str,
    target: str,
    marker: int,
    root_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    transform: np.ndarray,
    quality: float,
    index: int,
    quality_contract: str | None = None,
) -> dict[str, Any]:
    supports = [root_row["observation_key"], target_row["observation_key"]]
    result = {
        "schema_version": 1,
        "implementation": implementation,
        "candidate_type": "direct",
        "root_camera": ROOT_CAMERA,
        "target_camera": target,
        "root_marker": marker,
        "target_marker": marker,
        "root_frame": None,
        "target_frame": None,
        "relay_path": None,
        "transform_chain": {
            "id": f"d:m{marker}",
            "expression": "T_rm@inv(T_tm)",
            "terms": ["s0:marker->root", "inv(s1):target->marker"],
            "direction": "target->root",
        },
        "rotation": transform[:3, :3],
        "translation_m": transform[:3, 3],
        "support_observation_keys": supports,
        "support_count": 2,
        "quality_values": {
            "root_static": float(root_row["_quality"]),
            "target_static": float(target_row["_quality"]),
        },
        "reprojection_values_px": {
            "root_static": root_row["pnp"]["reprojection_rmse_px"],
            "target_static": target_row["pnp"]["reprojection_rmse_px"],
        },
        "score_components": {
            "formula": "sqrt(root_static_quality * target_static_quality)",
            "input_fields": ["root_static", "target_static"],
            "quality_contract": quality_contract or implementation,
        },
        "aggregate_score": float(quality),
        "acceptance_decision": "pending_aggregate_selection",
        "rejection_reason": None,
        "original_construction_index": index,
        "original_selection_ranking_index": marker,
        "deterministic_tie_break_fields": ["shared_marker_asc"],
        "native_candidate": True,
        "_T": transform,
        "_quality": float(quality),
        "_root_quality_components": root_row["_quality_components"],
        "_target_quality_components": target_row["_quality_components"],
        "_root_area_px2": root_row["_area_px2"],
        "_target_area_px2": target_row["_area_px2"],
        "_root_distance_m": root_row["_distance_m"],
        "_target_distance_m": target_row["_distance_m"],
    }
    result["semantic_candidate_key"] = _candidate_key(result)
    return result


def _relay_record(
    *,
    implementation: str,
    target: str,
    root_static: Mapping[str, Any],
    target_static: Mapping[str, Any],
    root_moving: Mapping[str, Any],
    target_moving: Mapping[str, Any],
    transform: np.ndarray,
    quality: float,
    index: int,
    quality_ranks: Mapping[str, int] | None = None,
    quality_contract: str | None = None,
) -> dict[str, Any]:
    root_marker = int(root_static["_marker"])
    target_marker = int(target_static["_marker"])
    root_frame = int(root_moving["_frame"])
    target_frame = int(target_moving["_frame"])
    supports = [
        root_static["observation_key"],
        target_static["observation_key"],
        root_moving["observation_key"],
        target_moving["observation_key"],
    ]
    chain_identity = (
        f"relay:m{root_marker}@f{root_frame}->"
        f"m{target_marker}@f{target_frame}"
    )
    result = {
        "schema_version": 1,
        "implementation": implementation,
        "candidate_type": "relay",
        "root_camera": ROOT_CAMERA,
        "target_camera": target,
        "root_marker": root_marker,
        "target_marker": target_marker,
        "root_frame": root_frame,
        "target_frame": target_frame,
        "relay_path": {
            "camera": "moving_calib_camera",
            "frames": [root_frame, target_frame],
        },
        "transform_chain": {
            "id": chain_identity,
            "expression": "T_rm@inv(T_imr)@T_ij@inv(T_tm@inv(T_jmt))",
            "terms": [
                "s0:root_marker->root",
                "inv(s2):moving_i->root_marker",
                "scaled(Tcw_i@inv(Tcw_j)):moving_j->moving_i",
                "inv(s1@inv(s3)):target->moving_j",
            ],
            "direction": "target->root",
        },
        "rotation": transform[:3, :3],
        "translation_m": transform[:3, 3],
        "support_observation_keys": supports,
        "support_count": 4,
        "quality_values": {
            "root_static": float(root_static["_quality"]),
            "target_static": float(target_static["_quality"]),
            "root_moving": float(root_moving["_quality"]),
            "target_moving": float(target_moving["_quality"]),
        },
        "reprojection_values_px": {
            "root_static": root_static["pnp"]["reprojection_rmse_px"],
            "target_static": target_static["pnp"]["reprojection_rmse_px"],
            "root_moving": root_moving["pnp"]["reprojection_rmse_px"],
            "target_moving": target_moving["pnp"]["reprojection_rmse_px"],
        },
        "score_components": {
            "formula": (
                "fourth_root(root_static * target_static * "
                "root_moving * target_moving)"
            ),
            "input_fields": [
                "root_static",
                "target_static",
                "root_moving",
                "target_moving",
            ],
            "quality_contract": quality_contract or implementation,
        },
        "aggregate_score": float(quality),
        "acceptance_decision": "pending_aggregate_selection",
        "rejection_reason": None,
        "original_construction_index": index,
        "original_selection_ranking_index": dict(quality_ranks or {}),
        "deterministic_tie_break_fields": [
            "root_marker_asc",
            "target_marker_asc",
            "root_frame_asc",
            "target_frame_asc",
        ],
        "native_candidate": True,
        "_T": transform,
        "_quality": float(quality),
    }
    result["semantic_candidate_key"] = _candidate_key(result)
    return result


def _candidate_core_row(record: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "mode": record["candidate_type"],
        "root_camera": record["root_camera"],
        "target_camera": record["target_camera"],
        "root_marker": record["root_marker"],
        "target_marker": record["target_marker"],
        "root_frame": record["root_frame"] or "",
        "target_frame": record["target_frame"] or "",
        "quality": record["_quality"],
        "T": record["_T"],
    }
    if record["candidate_type"] == "direct":
        result.update(
            {
                "root_area_px2": record["_root_area_px2"],
                "target_area_px2": record["_target_area_px2"],
                "root_distance_m": record["_root_distance_m"],
                "target_distance_m": record["_target_distance_m"],
            }
        )
    return result


def _build_direct(
    *,
    implementation: str,
    targets: Sequence[str],
    static: Mapping[tuple[str, int], dict[str, Any]],
    starting_index: int,
    product_contract: AP01MethodContract | None = None,
    quality_contract: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    index = starting_index
    for target in targets:
        common = sorted(
            {
                marker for camera, marker in static if camera == ROOT_CAMERA
            }
            & {marker for camera, marker in static if camera == target}
        )
        if implementation == "wizard" or product_contract is not None:
            native = core.direct_candidates(ROOT_CAMERA, target, dict(static))
            if [int(row["root_marker"]) for row in native] != common:
                raise AssertionError("Wizard direct-candidate order drifted")
        else:
            native = []
            for marker in common:
                root_row = static[(ROOT_CAMERA, marker)]
                target_row = static[(target, marker)]
                native.append(
                    {
                        "T": root_row["_T_cam_marker"]
                        @ core.invT(target_row["_T_cam_marker"]),
                        "quality": math.sqrt(
                            max(
                                1e-12,
                                root_row["_quality"] * target_row["_quality"],
                            )
                        ),
                    }
                )
        for marker, native_row in zip(common, native, strict=True):
            root_row = static[(ROOT_CAMERA, marker)]
            target_row = static[(target, marker)]
            records.append(
                _direct_record(
                    implementation=implementation,
                    target=target,
                    marker=marker,
                    root_row=root_row,
                    target_row=target_row,
                    transform=np.asarray(native_row["T"], dtype=np.float64),
                    quality=float(native_row["quality"]),
                    index=index,
                    quality_contract=quality_contract,
                )
            )
            index += 1
    return records, index


def _relay_support_sequence(
    *,
    target: str,
    static: Mapping[tuple[str, int], dict[str, Any]],
    moving: Mapping[int, list[dict[str, Any]]],
) -> Iterable[
    tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]
]:
    root_markers = sorted(
        marker
        for camera, marker in static
        if camera == ROOT_CAMERA and marker in moving
    )
    target_markers = sorted(
        marker
        for camera, marker in static
        if camera == target and marker in moving
    )
    for root_marker in root_markers:
        for target_marker in target_markers:
            root_static = static[(ROOT_CAMERA, root_marker)]
            target_static = static[(target, target_marker)]
            for root_moving in moving[root_marker]:
                for target_moving in moving[target_marker]:
                    if (
                        root_marker == target_marker
                        and root_moving["_frame"] == target_moving["_frame"]
                    ):
                        continue
                    yield (
                        root_static,
                        target_static,
                        root_moving,
                        target_moving,
                    )


def _build_relay(
    *,
    implementation: str,
    targets: Sequence[str],
    static: Mapping[tuple[str, int], dict[str, Any]],
    moving: Mapping[int, list[dict[str, Any]]],
    poses: Mapping[int, np.ndarray],
    scale: float,
    starting_index: int,
    quality_rank_by_key: Mapping[str, int] | None = None,
    product_contract: AP01MethodContract | None = None,
    quality_contract: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    index = starting_index
    for target in targets:
        supports = list(
            _relay_support_sequence(target=target, static=static, moving=moving)
        )
        if implementation == "wizard" or product_contract is not None:
            native = core.relay_candidates(
                ROOT_CAMERA,
                target,
                dict(static),
                dict(moving),
                dict(poses),
                scale,
            )
            if len(native) != len(supports):
                raise AssertionError("Wizard relay-candidate multiplicity drifted")
        else:
            native = []
            for root_static, target_static, root_moving, target_moving in supports:
                frame_i = int(root_moving["_frame"])
                frame_j = int(target_moving["_frame"])
                T_root_moving_i = root_static["_T_cam_marker"] @ core.invT(
                    root_moving["_T_cam_marker"]
                )
                T_target_moving_j = target_static["_T_cam_marker"] @ core.invT(
                    target_moving["_T_cam_marker"]
                )
                T_moving_i_moving_j = poses[frame_i] @ core.invT(poses[frame_j])
                T_moving_i_moving_j = T_moving_i_moving_j.copy()
                T_moving_i_moving_j[:3, 3] *= scale
                transform = (
                    T_root_moving_i
                    @ T_moving_i_moving_j
                    @ core.invT(T_target_moving_j)
                )
                qualities = (
                    root_static["_quality"],
                    target_static["_quality"],
                    root_moving["_quality"],
                    target_moving["_quality"],
                )
                native.append(
                    {
                        "T": transform,
                        "quality": float(
                            np.prod([max(1e-12, value) for value in qualities])
                            ** 0.25
                        ),
                    }
                )
        for support, native_row in zip(supports, native, strict=True):
            root_static, target_static, root_moving, target_moving = support
            ranks = None
            if quality_rank_by_key is not None:
                ranks = {
                    "root_moving_quality_rank": quality_rank_by_key[
                        root_moving["observation_key"]
                    ],
                    "target_moving_quality_rank": quality_rank_by_key[
                        target_moving["observation_key"]
                    ],
                }
            records.append(
                _relay_record(
                    implementation=implementation,
                    target=target,
                    root_static=root_static,
                    target_static=target_static,
                    root_moving=root_moving,
                    target_moving=target_moving,
                    transform=np.asarray(native_row["T"], dtype=np.float64),
                    quality=float(native_row["quality"]),
                    index=index,
                    quality_ranks=ranks,
                    quality_contract=quality_contract,
                )
            )
            index += 1
    return records, index


def _rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    return core.rotation_difference_deg(first[:3, :3], second[:3, :3])


def _legacy_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(
            max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        ) * 2.0
        values = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(
            max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        ) * 2.0
        values = (
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        )
    else:
        scale = math.sqrt(
            max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        ) * 2.0
        values = (
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        )
    quaternion = np.asarray(values, dtype=np.float64)
    quaternion /= max(1e-15, float(np.linalg.norm(quaternion)))
    return quaternion


def _legacy_quaternion_rotation(quaternion: Sequence[float]) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    values /= max(1e-15, float(np.linalg.norm(values)))
    qw, qx, qy, qz = values
    return np.asarray(
        (
            (
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ),
            (
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ),
            (
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ),
        ),
        dtype=np.float64,
    )


def _legacy_weighted_transform(
    transforms: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= max(1e-15, float(np.sum(normalized)))
    translation = np.zeros(3, dtype=np.float64)
    quaternions = [_legacy_quaternion(item[:3, :3]) for item in transforms]
    reference = quaternions[0]
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for transform, quaternion, weight in zip(
        transforms, quaternions, normalized, strict=True
    ):
        translation += float(weight) * transform[:3, 3]
        if float(np.dot(quaternion, reference)) < 0.0:
            quaternion = -quaternion
        accumulator += float(weight) * np.outer(quaternion, quaternion)
    values, vectors = np.linalg.eigh(accumulator)
    mean = vectors[:, int(np.argmax(values))]
    if mean[0] < 0.0:
        mean = -mean
    return _make_transform(_legacy_quaternion_rotation(mean), translation)


def _legacy_medoid(records: Sequence[Mapping[str, Any]]) -> tuple[int, float]:
    best_index: int | None = None
    best_score: float | None = None
    for index, candidate in enumerate(records):
        distances = []
        for other_index, other in enumerate(records):
            if index == other_index:
                continue
            translation = float(
                np.linalg.norm(candidate["_T"][:3, 3] - other["_T"][:3, 3])
            )
            rotation = _rotation_error(candidate["_T"], other["_T"])
            distances.append(translation + 0.02 * rotation)
        score = float(np.median(distances)) if distances else 0.0
        if best_score is None or score < best_score:
            best_index = index
            best_score = score
    if best_index is None or best_score is None:
        raise RuntimeError("No Legacy AP01 medoid candidate")
    return best_index, best_score


def _legacy_medoid_inliers(
    records: Sequence[dict[str, Any]],
    medoid_index: int,
    *,
    translation_floor: float,
    rotation_floor: float,
) -> tuple[list[int], dict[str, Any]]:
    center = records[medoid_index]["_T"]
    translation = np.asarray(
        [
            np.linalg.norm(record["_T"][:3, 3] - center[:3, 3])
            for record in records
        ]
    )
    rotation = np.asarray(
        [_rotation_error(record["_T"], center) for record in records]
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


def _legacy_relay_aggregate(
    records: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any], set[int]]:
    translations = np.asarray([record["_T"][:3, 3] for record in records])
    weights = np.asarray([max(1e-12, record["_quality"]) for record in records])
    initial = _make_transform(
        _legacy_weighted_transform(
            [record["_T"] for record in records], weights
        )[:3, :3],
        np.median(translations, axis=0),
    )
    translation_deviation = np.asarray(
        [
            np.linalg.norm(record["_T"][:3, 3] - initial[:3, 3])
            for record in records
        ]
    )
    rotation_deviation = np.asarray(
        [_rotation_error(record["_T"], initial) for record in records]
    )
    t_median = float(np.median(translation_deviation))
    r_median = float(np.median(rotation_deviation))
    t_mad = 1.4826 * float(
        np.median(np.abs(translation_deviation - t_median))
    )
    r_mad = 1.4826 * float(
        np.median(np.abs(rotation_deviation - r_median))
    )
    t_threshold = max(0.30, t_median + 3.0 * t_mad)
    r_threshold = max(7.0, r_median + 3.0 * r_mad)
    indices = [
        index
        for index, (t_value, r_value) in enumerate(
            zip(translation_deviation, rotation_deviation)
        )
        if t_value <= t_threshold and r_value <= r_threshold
    ]
    fallback = False
    if len(indices) < 3:
        fallback = True
        ranked = sorted(
            range(len(records)),
            key=lambda index: records[index]["_quality"],
            reverse=True,
        )
        indices = ranked[: max(3, len(ranked) // 2)]
    transform = _legacy_weighted_transform(
        [records[index]["_T"] for index in indices],
        [max(1e-12, records[index]["_quality"]) for index in indices],
    )
    return transform, {
        "aggregate_type": "weighted_mean_of_mad_inliers_no_gt_selection",
        "num_candidates": len(records),
        "num_inliers": len(indices),
        "num_outliers": len(records) - len(indices),
        "translation_deviation_median_m": t_median,
        "translation_deviation_mad_scaled_m": t_mad,
        "translation_inlier_threshold_m": t_threshold,
        "rotation_deviation_median_deg": r_median,
        "rotation_deviation_mad_scaled_deg": r_mad,
        "rotation_inlier_threshold_deg": r_threshold,
        "fallback_top_half_by_quality": fallback,
        "ground_truth_used": False,
    }, set(indices)


def _transform_payload(transform: np.ndarray | None) -> dict[str, Any] | None:
    if transform is None:
        return None
    return {
        "rotation": np.asarray(transform[:3, :3], dtype=np.float64).tolist(),
        "translation_m": np.asarray(transform[:3, 3], dtype=np.float64).tolist(),
        "direction": "target_camera_to_cam_edge_3",
    }


def _legacy_direct_selection(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    quality_indices = [
        index
        for index, record in enumerate(records)
        if record["_root_quality_components"][
            "area_px2_from_corners"
        ]
        >= 500.0
        and record["_target_quality_components"][
            "area_px2_from_corners"
        ]
        >= 500.0
        and record["_root_quality_components"][
            "distance_m"
        ]
        <= 5.5
        and record["_target_quality_components"][
            "distance_m"
        ]
        <= 5.5
        and record["aggregate_score"] >= 20.0
    ]
    fallback = False
    if not quality_indices:
        fallback = True
        quality_indices = sorted(
            range(len(records)),
            key=lambda index: records[index]["aggregate_score"],
            reverse=True,
        )[: max(1, min(2, len(records)))]
    quality_records = [records[index] for index in quality_indices]
    medoid_index, medoid_score = _legacy_medoid(quality_records)
    inlier_local, inlier_stats = _legacy_medoid_inliers(
        quality_records,
        medoid_index,
        translation_floor=0.08,
        rotation_floor=2.0,
    )
    if not inlier_local:
        inlier_local = list(range(len(quality_records)))
    inlier_indices = {quality_indices[index] for index in inlier_local}
    quality_set = set(quality_indices)
    marker14 = next(
        (
            index
            for index in quality_indices
            if int(records[index]["root_marker"]) == 14
        ),
        None,
    )
    if marker14 is not None:
        selected_index = marker14
        note = "marker14_visible_and_passed_no_gt_quality_filter"
    else:
        selected_index = quality_indices[medoid_index]
        note = "quality_filtered_se3_medoid_fallback"
    weighted_transform = _legacy_weighted_transform(
        [records[index]["_T"] for index in sorted(inlier_indices)],
        [records[index]["_quality"] for index in sorted(inlier_indices)],
    )
    for index, record in enumerate(records):
        record["aggregate_decisions"] = {
            "quality_filter_eligible": index in quality_set,
            "quality_filter_fallback_used": fallback,
            "quality_filtered_mad_inlier": index in inlier_indices,
            "preferred_marker_selected": index == selected_index,
        }
        if index == selected_index:
            record["acceptance_decision"] = (
                "selected_preferred_marker_for_legacy_direct_aggregate"
            )
        elif index in inlier_indices:
            record["acceptance_decision"] = (
                "accepted_as_support_but_not_preferred_marker"
            )
        elif index in quality_set:
            record["acceptance_decision"] = "rejected_by_quality_subset_mad"
            record["rejection_reason"] = "outside_quality_subset_mad_consensus"
        else:
            record["acceptance_decision"] = "rejected_by_direct_quality_filter"
            record["rejection_reason"] = (
                "area<500_px2 or distance>5.5_m or combined_quality<20"
            )
    return records[selected_index]["_T"], {
        "selected_aggregate_type": (
            "quality_filtered_preferred_marker_no_gt_selection"
        ),
        "aggregate_priority": [
            "quality_filtered_preferred_marker_no_gt_selection",
            "quality_filtered_weighted_mean_no_gt_selection",
            "weighted_mean_of_mad_inliers_no_gt_selection",
            "se3_medoid_no_gt_selection",
        ],
        "selected_marker_id": records[selected_index]["root_marker"],
        "selected_candidate_key": records[selected_index][
            "semantic_candidate_key"
        ],
        "selection_note": note,
        "quality_filter_fallback_used": fallback,
        "num_candidates": len(records),
        "num_quality_candidates": len(quality_indices),
        "num_quality_mad_inliers": len(inlier_indices),
        "quality_subset_medoid_score": medoid_score,
        "quality_subset_mad": inlier_stats,
        "quality_filtered_weighted_mean_diagnostic": _transform_payload(
            weighted_transform
        ),
        "ground_truth_used": False,
    }


def _parity_contract_aggregate_and_select(
    records: list[dict[str, Any]], contract: AP01MethodContract
) -> dict[str, Any]:
    """Format production parity-mode selection in the Legacy evidence schema."""

    grouped: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        if record["candidate_type"] != "root":
            grouped[record["target_camera"]][record["candidate_type"]].append(
                record
            )
    direct_targets = set(contract.direct_targets(CAMERA_ORDER, ROOT_CAMERA))
    relay_reports: dict[str, dict[str, Any]] = {}
    relay_poses: dict[str, np.ndarray] = {}
    direct_reports: dict[str, dict[str, Any]] = {}
    direct_poses: dict[str, np.ndarray] = {}
    for target in CAMERA_ORDER:
        if target == ROOT_CAMERA:
            continue
        direct = grouped[target]["direct"]
        relay = grouped[target]["relay"]
        if direct:
            direct_poses[target], direct_reports[target] = (
                _legacy_direct_selection(direct)
            )
        if relay:
            pose, report, inliers = _legacy_relay_aggregate(relay)
            relay_poses[target] = pose
            relay_reports[target] = report
            for candidate_index, record in enumerate(relay):
                record["aggregate_decisions"] = {
                    "flat_mad_inlier": candidate_index in inliers,
                    "selected_aggregate_type": report["aggregate_type"],
                }
                if candidate_index in inliers:
                    record["acceptance_decision"] = (
                        "accepted_into_flat_weighted_relay_aggregate"
                    )
                else:
                    record["acceptance_decision"] = (
                        "rejected_from_flat_weighted_relay_aggregate"
                    )
                    record["rejection_reason"] = "outside_flat_mad_consensus"

    per_camera: dict[str, Any] = {
        ROOT_CAMERA: {
            "selected_candidate_type": "root",
            "selected_method": "gauge_identity",
            "deployment_eligible": True,
            "omitted": False,
            "aggregate_transform": _transform_payload(np.eye(4)),
        }
    }
    for target in CAMERA_ORDER:
        if target == ROOT_CAMERA:
            continue
        fixed_type = "direct" if target in direct_targets else "relay"
        if fixed_type == "direct":
            pose = direct_poses.get(target)
            report = direct_reports.get(target)
            method = "direct_static_aruco_multimarker"
            missing_reason = "missing_legacy_direct_aggregate"
        else:
            pose = relay_poses.get(target)
            report = relay_reports.get(target)
            method = "moving_relay_multichain_colmap_motion_aruco_metric_scale"
            missing_reason = "missing_legacy_relay_aggregate"
        per_camera[target] = {
            "selected_candidate_type": fixed_type if pose is not None else None,
            "selected_method": method if pose is not None else "unavailable",
            "deployment_eligible": pose is not None,
            "omitted": pose is None,
            "omission_reason": None if pose is not None else missing_reason,
            "aggregate_statistics": _json_safe(report),
            "aggregate_transform": _transform_payload(pose),
        }
        if fixed_type == "relay" and report is not None:
            per_camera[target]["selected_aggregate_type"] = report[
                "aggregate_type"
            ]
    unselected = {}
    for target in sorted(direct_targets):
        if target in relay_reports:
            unselected[f"{target}_relay"] = {
                "aggregate_statistics": relay_reports[target],
                "aggregate_transform": _transform_payload(relay_poses[target]),
                "reason": (
                    "legacy exporter fixes cam_edge_1 to direct loader"
                ),
            }
    return {
        "schema_version": 1,
        "implementation": "wizard_main_route2_parity_v1",
        "root_camera": ROOT_CAMERA,
        "root_selection": {
            "method": "resolved_AP01_root_camera",
            "candidate_key": records[0]["semantic_candidate_key"],
            "eligible": True,
        },
        "camera_traversal_order": [
            camera for camera in CAMERA_ORDER if camera != ROOT_CAMERA
        ],
        "candidate_eligibility_before_ranking": (
            "aggregate row exists; no deployment stability gate"
        ),
        "candidate_sorting": "none across direct/relay paths",
        "tie_breaking": (
            "direct aggregate priority list then first matching row; relay exact aggregate type"
        ),
        "per_camera": per_camera,
        "unselected_diagnostics": unselected,
        "candidate_counts": _counts(records),
        "method_contract": contract.fingerprint_payload(),
        "ground_truth_used": False,
        "solver_invoked": False,
        "colmap_invoked": False,
        "final_pose_published": False,
    }


def _counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record["target_camera"])][
            str(record["candidate_type"])
        ] += 1
    return {
        camera: dict(sorted(values.items()))
        for camera, values in sorted(counts.items())
    }


def _public_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _json_safe(value)
        for key, value in record.items()
        if not key.startswith("_")
    }


def extract_legacy_candidates(
    *,
    frozen_observations: Path,
    frozen_colmap_images: Path,
    frozen_metric_scale: Path,
    output_root: Path,
    legacy_source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt only Main's GT-free candidate and aggregate-selection math."""

    for path in (
        frozen_observations,
        frozen_colmap_images,
        frozen_metric_scale,
        output_root,
        legacy_source_root,
    ):
        assert_pre_solver_path(path)
    frozen = _read_jsonl(frozen_observations)
    static_rows, moving_rows = _prepared_rows(frozen, implementation="legacy")
    static = _best_static(static_rows)
    poses = core.parse_colmap_poses(frozen_colmap_images)
    moving = _legacy_moving(moving_rows)
    moving = {
        marker: [row for row in rows if int(row["_frame"]) in poses]
        for marker, rows in moving.items()
    }
    moving = {marker: rows for marker, rows in moving.items() if rows}
    scale = float(frozen_metric_scale.read_text(encoding="utf-8").strip())

    root = _root_candidate("legacy")
    direct, index = _build_direct(
        implementation="legacy",
        targets=LEGACY_DIRECT_TARGETS,
        static=static,
        starting_index=0,
    )
    relay, _ = _build_relay(
        implementation="legacy",
        targets=LEGACY_RELAY_TARGETS,
        static=static,
        moving=moving,
        poses=poses,
        scale=scale,
        starting_index=index,
    )

    direct_by_target: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    relay_by_target: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in direct:
        direct_by_target[record["target_camera"]].append(record)
    for record in relay:
        relay_by_target[record["target_camera"]].append(record)

    direct_pose, direct_report = _legacy_direct_selection(
        direct_by_target["cam_edge_1"]
    )
    relay_reports: dict[str, dict[str, Any]] = {}
    relay_poses: dict[str, np.ndarray] = {}
    for target in LEGACY_RELAY_TARGETS:
        pose, report, inliers = _legacy_relay_aggregate(relay_by_target[target])
        relay_reports[target] = report
        relay_poses[target] = pose
        for candidate_index, record in enumerate(relay_by_target[target]):
            record["aggregate_decisions"] = {
                "flat_mad_inlier": candidate_index in inliers,
                "selected_aggregate_type": report["aggregate_type"],
            }
            if candidate_index in inliers:
                record["acceptance_decision"] = (
                    "accepted_into_flat_weighted_relay_aggregate"
                )
            else:
                record["acceptance_decision"] = (
                    "rejected_from_flat_weighted_relay_aggregate"
                )
                record["rejection_reason"] = "outside_flat_mad_consensus"

    selection = {
        "schema_version": 1,
        "implementation": "legacy_main",
        "root_camera": ROOT_CAMERA,
        "root_selection": {
            "method": "fixed_ROOT_CAM_constant",
            "candidate_key": root["semantic_candidate_key"],
            "eligible": True,
        },
        "camera_traversal_order": list(LEGACY_RELAY_TARGETS),
        "candidate_eligibility_before_ranking": (
            "aggregate row exists; no deployment stability gate"
        ),
        "candidate_sorting": "none across direct/relay paths",
        "tie_breaking": (
            "direct aggregate priority list then first matching row; relay exact aggregate type"
        ),
        "per_camera": {
            ROOT_CAMERA: {
                "selected_candidate_type": "root",
                "selected_method": "gauge_identity",
                "deployment_eligible": True,
                "omitted": False,
                "aggregate_transform": _transform_payload(np.eye(4)),
            },
            "cam_edge_0": {
                "selected_candidate_type": "relay",
                "selected_method": (
                    "moving_relay_multichain_colmap_motion_aruco_metric_scale"
                ),
                "selected_aggregate_type": (
                    "weighted_mean_of_mad_inliers_no_gt_selection"
                ),
                "deployment_eligible": True,
                "omitted": False,
                "omission_reason": None,
                "aggregate_statistics": relay_reports["cam_edge_0"],
                "aggregate_transform": _transform_payload(
                    relay_poses["cam_edge_0"]
                ),
            },
            "cam_edge_1": {
                "selected_candidate_type": "direct",
                "selected_method": "direct_static_aruco_multimarker",
                "deployment_eligible": True,
                "omitted": False,
                "omission_reason": None,
                "aggregate_statistics": direct_report,
                "aggregate_transform": _transform_payload(direct_pose),
            },
            "cam_edge_5": {
                "selected_candidate_type": "relay",
                "selected_method": (
                    "moving_relay_multichain_colmap_motion_aruco_metric_scale"
                ),
                "selected_aggregate_type": (
                    "weighted_mean_of_mad_inliers_no_gt_selection"
                ),
                "deployment_eligible": True,
                "omitted": False,
                "omission_reason": None,
                "aggregate_statistics": relay_reports["cam_edge_5"],
                "aggregate_transform": _transform_payload(
                    relay_poses["cam_edge_5"]
                ),
            },
        },
        "unselected_diagnostics": {
            "cam_edge_1_relay": {
                "aggregate_statistics": relay_reports["cam_edge_1"],
                "aggregate_transform": _transform_payload(
                    relay_poses["cam_edge_1"]
                ),
                "reason": "legacy exporter fixes cam_edge_1 to direct loader",
            }
        },
        "candidate_counts": _counts([root, *direct, *relay]),
        "ground_truth_used": False,
        "solver_invoked": False,
        "colmap_invoked": False,
        "final_pose_published": False,
    }
    records = [root, *direct, *relay]
    output_root.mkdir(parents=True, exist_ok=True)
    _write_candidate_jsonl(output_root / "AP01_CANDIDATES.jsonl", records)
    write_json(output_root / "AP01_SELECTION.json", selection)
    source_files = {
        "direct": legacy_source_root
        / "13_eval_direct_static_cam3_cam1_multimarker.py",
        "relay": legacy_source_root / "14_eval_moving_relay_chains.py",
        "export": legacy_source_root
        / "15_export_final_extrinsics_cam3_reference.py",
    }
    trace = "\n".join(
        (
            "AP01 LEGACY MAIN CANDIDATE/SELECTION TRACE",
            "boundary: candidate construction and aggregate selection only",
            f"frozen observations: {frozen_observations.resolve()}",
            f"frozen moving poses: {frozen_colmap_images.resolve()}",
            f"frozen no-GT scale: {frozen_metric_scale.resolve()}",
            "direct candidate math: 13_eval_direct_static_cam3_cam1_multimarker.py:256-303,477-508",
            "direct aggregate/priority: 13_eval_direct_static_cam3_cam1_multimarker.py:556-722; 15_export_final_extrinsics_cam3_reference.py:125-168",
            "relay candidate math: 14_eval_moving_relay_chains.py:289-336,630-730",
            "relay aggregate/selection: 14_eval_moving_relay_chains.py:550-627,771-811; 15_export_final_extrinsics_cam3_reference.py:172-218,585-605",
            *(
                f"source sha256 {name}: {sha256_file(path)}"
                for name, path in source_files.items()
            ),
            "adaptation: evaluation/GT statements were not executed; only the cited no-GT formulas were transcribed.",
            "candidate JSONL: first line is the decoding schema; each later positional row is one candidate in original construction order.",
            "COLMAP was not invoked; parse_colmap_poses read the frozen pre-existing images.txt.",
            "No AP01 stage runner, solver, evaluator, or result publisher was invoked.",
            "ground_truth_used=false",
            "",
        )
    )
    (output_root / "AP01_TRACE.txt").write_text(trace, encoding="utf-8")
    return records, selection


def _wizard_aggregate_and_select(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        if record["candidate_type"] != "root":
            grouped[record["target_camera"]][record["candidate_type"]].append(
                record
            )

    per_camera: dict[str, Any] = {
        ROOT_CAMERA: {
            "selected_candidate_type": "root",
            "selected_method": "gauge_identity",
            "deployment_eligible": True,
            "quality_status": "gauge_identity",
            "omitted": False,
            "omission_reason": None,
            "aggregate_transform": _transform_payload(np.eye(4)),
        }
    }
    for target in CAMERA_ORDER:
        if target == ROOT_CAMERA:
            continue
        direct_records = grouped[target]["direct"]
        relay_records = grouped[target]["relay"]
        direct_core = [_candidate_core_row(record) for record in direct_records]
        relay_core = [_candidate_core_row(record) for record in relay_records]
        direct_pose: np.ndarray | None = None
        relay_pose: np.ndarray | None = None
        direct_stats: dict[str, Any] | None = None
        relay_stats: dict[str, Any] | None = None
        relay_chains: list[dict[str, Any]] = []
        if direct_core:
            direct_pose, direct_stats = core.aggregate_direct_marker_estimates(
                direct_core
            )
        if relay_core:
            relay_pose, relay_stats, relay_chains = (
                core.aggregate_relay_marker_chains(relay_core)
            )
        direct_gate = evaluate_path_gate(
            direct_core,
            direct_stats,
            minimum_inlier_ratio=0.70,
            maximum_translation_dispersion_m=0.12,
            maximum_rotation_dispersion_deg=4.0,
            minimum_independent_markers=3,
        )
        relay_gate = evaluate_path_gate(
            relay_chains,
            relay_stats,
            minimum_inlier_ratio=0.70,
            maximum_translation_dispersion_m=0.30,
            maximum_rotation_dispersion_deg=7.0,
        )
        path_comparison = compare_paths(
            direct_pose,
            relay_pose,
            maximum_translation_disagreement_m=0.12,
            maximum_rotation_disagreement_deg=4.0,
        )
        direct_stable = bool(direct_gate["stable"])
        relay_stable = bool(relay_gate["stable"])
        deployment_eligible = False
        if (
            direct_stable
            and relay_stable
            and path_comparison["consistent"] is False
        ):
            selected_pose = direct_pose
            selected_type = "direct"
            selected_method = "direct_multimarker"
            quality_status = "rejected_direct_relay_disagreement"
            omission_reason = quality_status
        elif direct_stable:
            selected_pose = direct_pose
            selected_type = "direct"
            selected_method = "direct_multimarker"
            quality_status = "accepted"
            omission_reason = None
            deployment_eligible = True
        elif relay_stable:
            selected_pose = relay_pose
            selected_type = "relay"
            selected_method = "moving_colmap_relay"
            quality_status = "accepted"
            omission_reason = None
            deployment_eligible = True
        else:
            selected_pose = direct_pose if direct_pose is not None else relay_pose
            selected_type = (
                "direct"
                if direct_pose is not None
                else "relay"
                if relay_pose is not None
                else None
            )
            selected_method = (
                "direct_multimarker_diagnostic"
                if direct_pose is not None
                else "moving_colmap_relay_diagnostic"
                if relay_pose is not None
                else "unavailable"
            )
            quality_status = (
                "rejected_unstable_consensus"
                if selected_pose is not None
                else "unavailable_no_finite_estimate"
            )
            omission_reason = quality_status

        chain_by_id = {
            str(chain["chain_id"]): chain for chain in relay_chains
        }
        for record, candidate in zip(direct_records, direct_core, strict=True):
            record["aggregate_decisions"] = {
                "robust_inlier": bool(candidate.get("inlier")),
                "pose_support": bool(candidate.get("pose_support")),
                "path_stable": direct_stable,
                "path_selected_for_diagnostic_estimate": selected_type == "direct",
                "path_deployment_eligible": (
                    deployment_eligible and selected_type == "direct"
                ),
            }
            if not candidate.get("inlier"):
                record["acceptance_decision"] = "rejected_by_direct_mad"
                record["rejection_reason"] = "outside_direct_mad_consensus"
            elif not direct_stable:
                record["acceptance_decision"] = (
                    "accepted_into_diagnostic_direct_aggregate_only"
                )
                record["rejection_reason"] = "direct_path_failed_quality_gate"
            elif selected_type != "direct" or not deployment_eligible:
                record["acceptance_decision"] = (
                    "accepted_into_direct_aggregate_but_path_not_deployable"
                )
                record["rejection_reason"] = quality_status
            else:
                record["acceptance_decision"] = (
                    "accepted_into_selected_direct_aggregate"
                )
        for record, candidate in zip(relay_records, relay_core, strict=True):
            chain_id = f"{record['root_marker']}->{record['target_marker']}"
            chain = chain_by_id[chain_id]
            record["aggregate_decisions"] = {
                "within_chain_robust_inlier": bool(candidate.get("inlier")),
                "within_chain_pose_support": bool(candidate.get("pose_support")),
                "independent_chain_robust_inlier": bool(chain.get("inlier")),
                "independent_chain_pose_support": bool(chain.get("pose_support")),
                "path_stable": relay_stable,
                "path_selected_for_diagnostic_estimate": selected_type == "relay",
                "path_deployment_eligible": (
                    deployment_eligible and selected_type == "relay"
                ),
            }
            if not candidate.get("inlier"):
                record["acceptance_decision"] = "rejected_within_relay_chain_mad"
                record["rejection_reason"] = "outside_within_chain_mad_consensus"
            elif not chain.get("inlier"):
                record["acceptance_decision"] = (
                    "accepted_within_chain_but_chain_rejected"
                )
                record["rejection_reason"] = (
                    "independent_marker_chain_outside_final_mad_consensus"
                )
            elif not relay_stable:
                record["acceptance_decision"] = (
                    "accepted_into_diagnostic_relay_aggregate_only"
                )
                record["rejection_reason"] = "relay_path_failed_quality_gate"
            elif selected_type != "relay" or not deployment_eligible:
                record["acceptance_decision"] = (
                    "accepted_into_relay_aggregate_but_path_not_selected"
                )
                record["rejection_reason"] = quality_status
            else:
                record["acceptance_decision"] = (
                    "accepted_into_selected_relay_aggregate"
                )

        per_camera[target] = {
            "selected_candidate_type": selected_type,
            "selected_method": selected_method,
            "deployment_eligible": deployment_eligible,
            "quality_status": quality_status,
            "omitted": not deployment_eligible,
            "omission_reason": omission_reason,
            "direct": _json_safe(direct_gate),
            "relay": _json_safe(relay_gate),
            "direct_relay_consistency": _json_safe(path_comparison),
            "direct_raw_candidate_count": len(direct_records),
            "relay_raw_candidate_count": len(relay_records),
            "relay_independent_chain_count": len(relay_chains),
            "aggregate_transform": _transform_payload(selected_pose),
            "aggregate_transform_status": (
                "diagnostic_only_not_published"
                if not deployment_eligible and selected_pose is not None
                else "eligible_not_published"
                if deployment_eligible
                else "unavailable"
            ),
        }
    return {
        "schema_version": 1,
        "implementation": "wizard",
        "root_camera": ROOT_CAMERA,
        "root_selection": {
            "method": "resolved_AP01_root_camera",
            "candidate_key": records[0]["semantic_candidate_key"],
            "eligible": True,
        },
        "camera_traversal_order": list(CAMERA_ORDER),
        "candidate_eligibility_before_ranking": (
            "direct and relay aggregates must pass explicit GT-free stability gates"
        ),
        "candidate_sorting": (
            "direct preferred when stable; relay used only when direct is unstable and relay is stable"
        ),
        "tie_breaking": (
            "quality-ranked moving rows use descending score then ascending frame; marker/chain traversal is ascending"
        ),
        "per_camera": per_camera,
        "candidate_counts": _counts(records),
        "gates": {
            "direct": {
                "minimum_independent_markers": 3,
                "minimum_inlier_ratio": 0.70,
                "maximum_translation_dispersion_m": 0.12,
                "maximum_rotation_dispersion_deg": 4.0,
            },
            "relay": {
                "minimum_inlier_ratio": 0.70,
                "maximum_translation_dispersion_m": 0.30,
                "maximum_rotation_dispersion_deg": 7.0,
            },
            "path_consistency": {
                "maximum_translation_disagreement_m": 0.12,
                "maximum_rotation_disagreement_deg": 4.0,
            },
        },
        "ground_truth_used": False,
        "solver_invoked": False,
        "colmap_invoked": False,
        "final_pose_published": False,
    }


def extract_wizard_candidates(
    *,
    frozen_observations: Path,
    frozen_colmap_images: Path,
    frozen_metric_scale: Path,
    output_root: Path,
    wizard_source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Call Wizard's candidate/aggregate helpers without calling stage run()."""

    for path in (
        frozen_observations,
        frozen_colmap_images,
        frozen_metric_scale,
        output_root,
        wizard_source_root,
    ):
        assert_pre_solver_path(path)
    frozen = _read_jsonl(frozen_observations)
    static_rows, moving_rows = _prepared_rows(frozen, implementation="wizard")
    static = core.best_static_by_camera_marker(static_rows)
    poses = core.parse_colmap_poses(frozen_colmap_images)
    moving = core.moving_by_marker(
        moving_rows,
        set(poses),
        top_per_marker=WIZARD_TOP_MOVING_PER_MARKER,
    )
    independently_selected, ranking = _wizard_moving(moving_rows, set(poses))
    if {
        marker: [row["observation_key"] for row in rows]
        for marker, rows in moving.items()
    } != {
        marker: [row["observation_key"] for row in rows]
        for marker, rows in independently_selected.items()
    }:
        raise AssertionError("Wizard moving-observation selection drifted")
    quality_rank_by_key = {
        row["observation_key"]: int(row["quality_rank"]) for row in ranking
    }
    scale = float(frozen_metric_scale.read_text(encoding="utf-8").strip())

    records = [_root_candidate("wizard")]
    construction_index = 0
    for target in CAMERA_ORDER:
        if target == ROOT_CAMERA:
            continue
        direct, construction_index = _build_direct(
            implementation="wizard",
            targets=(target,),
            static=static,
            starting_index=construction_index,
        )
        relay, construction_index = _build_relay(
            implementation="wizard",
            targets=(target,),
            static=static,
            moving=moving,
            poses=poses,
            scale=scale,
            starting_index=construction_index,
            quality_rank_by_key=quality_rank_by_key,
        )
        records.extend(direct)
        records.extend(relay)

    selection = _wizard_aggregate_and_select(records)
    selection["moving_observation_ranking"] = ranking
    selection["top_moving_per_marker"] = WIZARD_TOP_MOVING_PER_MARKER
    output_root.mkdir(parents=True, exist_ok=True)
    _write_candidate_jsonl(output_root / "AP01_CANDIDATES.jsonl", records)
    write_json(output_root / "AP01_SELECTION.json", selection)
    source_files = {
        "build_candidates": wizard_source_root / "build_candidates.py",
        "core": wizard_source_root / "core.py",
        "solve_extrinsics": wizard_source_root / "solve_extrinsics.py",
    }
    trace = "\n".join(
        (
            "AP01 WIZARD CANDIDATE/SELECTION TRACE",
            "boundary: candidate construction and aggregate selection only",
            f"frozen observations: {frozen_observations.resolve()}",
            f"frozen moving poses: {frozen_colmap_images.resolve()}",
            f"frozen no-GT scale: {frozen_metric_scale.resolve()}",
            "observation preparation: core.py:309-395",
            "moving ranking/cap: core.py:849-872; build_candidates.py:41-119",
            "direct candidate math: core.py:875-898",
            "relay candidate math: core.py:901-964",
            "direct/relay aggregation: core.py:576-837",
            "eligibility and path selection: solve_extrinsics.py:15-122,173-295",
            *(
                f"source sha256 {name}: {sha256_file(path)}"
                for name, path in source_files.items()
            ),
            "execution: core candidate and aggregate helpers plus pure gate helpers only.",
            "build_candidates.run and solve_extrinsics.run were not invoked.",
            "candidate JSONL: first line is the decoding schema; each later positional row is one candidate in original construction order.",
            "COLMAP was not invoked; parse_colmap_poses read the frozen pre-existing images.txt.",
            "No AP01 stage runner, solver, evaluator, or result publisher was invoked.",
            "ground_truth_used=false",
            "",
        )
    )
    (output_root / "AP01_TRACE.txt").write_text(trace, encoding="utf-8")
    return records, selection


def extract_wizard_parity_candidates(
    *,
    frozen_observations: Path,
    frozen_colmap_images: Path,
    frozen_metric_scale: Path,
    output_root: Path,
    wizard_source_root: Path,
    direct_target_camera: str = "cam_edge_1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exercise production ``main_route2_parity_v1`` without a stage runner."""

    for path in (
        frozen_observations,
        frozen_colmap_images,
        frozen_metric_scale,
        output_root,
        wizard_source_root,
    ):
        assert_pre_solver_path(path)
    contract = resolve_ap01_method_contract(
        "main_route2_parity_v1",
        direct_target_camera=direct_target_camera,
        top_moving_per_marker=WIZARD_TOP_MOVING_PER_MARKER,
    )
    frozen = _read_jsonl(frozen_observations)
    static_rows, moving_rows = _prepared_rows(frozen, implementation="legacy")
    poses = core.parse_colmap_poses(frozen_colmap_images)
    scale = float(frozen_metric_scale.read_text(encoding="utf-8").strip())
    native_records, moving_selection = construct_candidates(
        static_rows=static_rows,
        moving_rows=moving_rows,
        poses=poses,
        scale=scale,
        camera_ids=CAMERA_ORDER,
        root_camera=ROOT_CAMERA,
        contract=contract,
    )
    static = core.best_static_by_camera_marker(static_rows)
    moving = core.moving_by_contract(moving_rows, set(poses), contract)
    direct_targets = contract.direct_targets(CAMERA_ORDER, ROOT_CAMERA)
    relay_targets = contract.relay_targets(CAMERA_ORDER, ROOT_CAMERA)
    records = [_root_candidate("wizard")]
    direct, index = _build_direct(
        implementation="wizard",
        targets=direct_targets,
        static=static,
        starting_index=0,
        product_contract=contract,
        quality_contract="legacy",
    )
    relay, _ = _build_relay(
        implementation="wizard",
        targets=relay_targets,
        static=static,
        moving=moving,
        poses=poses,
        scale=scale,
        starting_index=index,
        product_contract=contract,
        quality_contract="legacy",
    )
    records.extend(direct)
    records.extend(relay)
    if len(native_records) != len(records) - 1:
        raise AssertionError("Parity-mode candidate multiplicity drifted")
    for native, evidence in zip(native_records, records[1:], strict=True):
        if (
            native["mode"],
            native["target_camera"],
            native["root_marker"],
            native["target_marker"],
            native["root_frame"],
            native["target_frame"],
        ) != (
            evidence["candidate_type"],
            evidence["target_camera"],
            evidence["root_marker"],
            evidence["target_marker"],
            (
                "" if evidence["root_frame"] is None
                else evidence["root_frame"]
            ),
            (
                "" if evidence["target_frame"] is None
                else evidence["target_frame"]
            ),
        ):
            raise AssertionError("Parity-mode candidate construction order drifted")
        if not np.array_equal(native["T"], evidence["_T"]):
            raise AssertionError("Parity-mode candidate transform drifted")
        if float(native["quality"]) != float(evidence["_quality"]):
            raise AssertionError("Parity-mode candidate quality drifted")

    product_selection = select_candidate_aggregates(
        [_candidate_core_row(record) for record in records[1:]],
        camera_ids=CAMERA_ORDER,
        root_camera=ROOT_CAMERA,
        contract=contract,
        include_flattened=False,
    )
    selection = _parity_contract_aggregate_and_select(records, contract)
    for camera in CAMERA_ORDER:
        expected = selection["per_camera"][camera]
        actual = product_selection["camera_statuses"][camera]
        if bool(expected["deployment_eligible"]) != bool(
            actual["deployment_eligible"]
        ):
            raise AssertionError(
                f"Parity-mode production eligibility drifted for {camera}"
            )
        expected_transform = expected.get("aggregate_transform")
        if expected_transform is not None:
            actual_pose = product_selection["poses"][camera]
            expected_pose = np.eye(4, dtype=np.float64)
            expected_pose[:3, :3] = expected_transform["rotation"]
            expected_pose[:3, 3] = expected_transform["translation_m"]
            if not np.allclose(actual_pose, expected_pose, atol=1e-12, rtol=0.0):
                raise AssertionError(
                    f"Parity-mode production aggregate drifted for {camera}"
                )
    selection["moving_observation_selection"] = moving_selection
    selection["top_moving_per_marker"] = None
    output_root.mkdir(parents=True, exist_ok=True)
    _write_candidate_jsonl(output_root / "AP01_CANDIDATES.jsonl", records)
    write_json(output_root / "AP01_SELECTION.json", selection)
    source_files = {
        "contracts": wizard_source_root / "contracts.py",
        "build_candidates": wizard_source_root / "build_candidates.py",
        "core": wizard_source_root / "core.py",
        "solve_extrinsics": wizard_source_root / "solve_extrinsics.py",
    }
    trace = "\n".join(
        (
            "AP01 WIZARD MAIN-ROUTE2-PARITY CANDIDATE/SELECTION TRACE",
            "boundary: candidate construction and aggregate selection only",
            f"method contract: {contract.name}",
            f"method contract sha256: {contract.scientific_fingerprint()}",
            f"frozen observations: {frozen_observations.resolve()}",
            f"frozen moving poses: {frozen_colmap_images.resolve()}",
            f"frozen no-GT scale: {frozen_metric_scale.resolve()}",
            "execution: construct_candidates and select_candidate_aggregates pure boundaries.",
            *(
                f"source sha256 {name}: {sha256_file(path)}"
                for name, path in source_files.items()
            ),
            "build_candidates.run and solve_extrinsics.run were not invoked.",
            "COLMAP was not invoked; only the frozen images.txt was parsed.",
            "No final pose file, evaluator, result publisher, AP02, or AP03 was invoked.",
            "ground_truth_used=false",
            "",
        )
    )
    (output_root / "AP01_TRACE.txt").write_text(trace, encoding="utf-8")
    return records, selection


def _candidate_map(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        key = str(record["semantic_candidate_key"])
        if key in result:
            raise RuntimeError(f"non-unique semantic candidate key: {key}")
        result[key] = record
    return result


def _diff_row(
    *,
    phase: str,
    key: str,
    record: Mapping[str, Any] | None,
    field: str,
    legacy: Any,
    wizard: Any,
    reason: str,
    delta: float | str = "",
    tolerance: float | str = "",
) -> dict[str, Any]:
    return {
        "phase": phase,
        "semantic_candidate_key": key,
        "candidate_type": record.get("candidate_type", "") if record else "",
        "target_camera": record.get("target_camera", "") if record else "",
        "root_marker": record.get("root_marker", "") if record else "",
        "target_marker": record.get("target_marker", "") if record else "",
        "root_frame": record.get("root_frame", "") if record else "",
        "target_frame": record.get("target_frame", "") if record else "",
        "field": field,
        "legacy_value": _json_safe(legacy),
        "wizard_value": _json_safe(wizard),
        "absolute_delta": delta,
        "tolerance": tolerance,
        "reason": reason,
    }


def compare_candidates(
    legacy: Sequence[Mapping[str, Any]],
    wizard: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left = _candidate_map(legacy)
    right = _candidate_map(wizard)
    left_keys = list(left)
    right_keys = list(right)
    missing = [key for key in left_keys if key not in right]
    unexpected = [key for key in right_keys if key not in left]
    differences: list[dict[str, Any]] = []
    for key in missing:
        differences.append(
            _diff_row(
                phase="candidate_set",
                key=key,
                record=left[key],
                field="semantic_candidate_key",
                legacy="present",
                wizard="unavailable",
                reason="missing_wizard_candidate",
            )
        )
    for key in unexpected:
        differences.append(
            _diff_row(
                phase="candidate_set",
                key=key,
                record=right[key],
                field="semantic_candidate_key",
                legacy="unavailable",
                wizard="present",
                reason="unexpected_wizard_candidate",
            )
        )

    maximum = {
        "rotation_matrix_absolute": 0.0,
        "translation_m_absolute": 0.0,
        "quality_absolute": 0.0,
        "score_absolute": 0.0,
        "reprojection_px_absolute": 0.0,
    }
    common = [key for key in left_keys if key in right]
    for key in common:
        legacy_record = left[key]
        wizard_record = right[key]
        context = legacy_record
        for field in (
            "relay_path",
            "support_observation_keys",
            "support_count",
            "transform_chain",
            "score_components",
            "aggregate_decisions",
        ):
            if legacy_record.get(field) != wizard_record.get(field):
                differences.append(
                    _diff_row(
                        phase="candidate_content",
                        key=key,
                        record=context,
                        field=field,
                        legacy=legacy_record.get(field),
                        wizard=wizard_record.get(field),
                        reason="value_mismatch",
                    )
                )
        for field, maximum_field, tolerance_field in (
            ("rotation", "rotation_matrix_absolute", "rotation_matrix_absolute"),
            ("translation_m", "translation_m_absolute", "translation_m_absolute"),
        ):
            legacy_array = np.asarray(legacy_record[field], dtype=np.float64)
            wizard_array = np.asarray(wizard_record[field], dtype=np.float64)
            delta = float(np.max(np.abs(legacy_array - wizard_array)))
            maximum[maximum_field] = max(maximum[maximum_field], delta)
            tolerance = NUMERIC_TOLERANCES[tolerance_field]
            if delta > tolerance:
                differences.append(
                    _diff_row(
                        phase="composed_transform",
                        key=key,
                        record=context,
                        field=field,
                        legacy=legacy_record[field],
                        wizard=wizard_record[field],
                        delta=delta,
                        tolerance=tolerance,
                        reason="numeric_difference_exceeds_tolerance",
                    )
                )
        quality_fields = sorted(
            set(legacy_record["quality_values"])
            | set(wizard_record["quality_values"])
        )
        for quality_field in quality_fields:
            legacy_value = legacy_record["quality_values"].get(quality_field)
            wizard_value = wizard_record["quality_values"].get(quality_field)
            if legacy_value is None or wizard_value is None:
                differences.append(
                    _diff_row(
                        phase="quality",
                        key=key,
                        record=context,
                        field=f"quality_values.{quality_field}",
                        legacy=legacy_value,
                        wizard=wizard_value,
                        reason="field_unavailable_on_one_side",
                    )
                )
                continue
            delta = abs(float(legacy_value) - float(wizard_value))
            maximum["quality_absolute"] = max(
                maximum["quality_absolute"], delta
            )
            if delta > NUMERIC_TOLERANCES["quality_absolute"]:
                differences.append(
                    _diff_row(
                        phase="quality",
                        key=key,
                        record=context,
                        field=f"quality_values.{quality_field}",
                        legacy=legacy_value,
                        wizard=wizard_value,
                        delta=delta,
                        tolerance=NUMERIC_TOLERANCES["quality_absolute"],
                        reason="different_observation_quality_contract",
                    )
                )
        legacy_score = legacy_record["aggregate_score"]
        wizard_score = wizard_record["aggregate_score"]
        if legacy_score is None or wizard_score is None:
            if legacy_score != wizard_score:
                differences.append(
                    _diff_row(
                        phase="score",
                        key=key,
                        record=context,
                        field="aggregate_score",
                        legacy=legacy_score,
                        wizard=wizard_score,
                        reason="field_unavailable_on_one_side",
                    )
                )
        else:
            score_delta = abs(float(legacy_score) - float(wizard_score))
            maximum["score_absolute"] = max(
                maximum["score_absolute"], score_delta
            )
            if score_delta > NUMERIC_TOLERANCES["score_absolute"]:
                differences.append(
                    _diff_row(
                        phase="score",
                        key=key,
                        record=context,
                        field="aggregate_score",
                        legacy=legacy_score,
                        wizard=wizard_score,
                        delta=score_delta,
                        tolerance=NUMERIC_TOLERANCES["score_absolute"],
                        reason="different_observation_quality_contract",
                    )
                )
        reprojection_fields = sorted(
            set(legacy_record["reprojection_values_px"])
            | set(wizard_record["reprojection_values_px"])
        )
        for reprojection_field in reprojection_fields:
            legacy_value = legacy_record["reprojection_values_px"].get(
                reprojection_field
            )
            wizard_value = wizard_record["reprojection_values_px"].get(
                reprojection_field
            )
            if legacy_value is None or wizard_value is None:
                delta = math.inf
            else:
                delta = abs(float(legacy_value) - float(wizard_value))
                maximum["reprojection_px_absolute"] = max(
                    maximum["reprojection_px_absolute"], delta
                )
            if delta > NUMERIC_TOLERANCES["reprojection_px_absolute"]:
                differences.append(
                    _diff_row(
                        phase="reprojection",
                        key=key,
                        record=context,
                        field=f"reprojection_values_px.{reprojection_field}",
                        legacy=legacy_value,
                        wizard=wizard_value,
                        delta=delta,
                        tolerance=NUMERIC_TOLERANCES[
                            "reprojection_px_absolute"
                        ],
                        reason="numeric_difference_exceeds_tolerance",
                    )
                )
        for field in (
            "acceptance_decision",
            "rejection_reason",
            "original_construction_index",
            "original_selection_ranking_index",
            "deterministic_tie_break_fields",
        ):
            if legacy_record[field] != wizard_record[field]:
                differences.append(
                    _diff_row(
                        phase=(
                            "construction_order"
                            if field == "original_construction_index"
                            else "aggregate_decision"
                        ),
                        key=key,
                        record=context,
                        field=field,
                        legacy=legacy_record[field],
                        wizard=wizard_record[field],
                        reason="value_mismatch",
                    )
                )

    if missing or unexpected:
        classification = "DIFFERENT_CANDIDATE_SET"
    elif any(row["phase"] == "composed_transform" for row in differences):
        classification = "DIFFERENT_TRANSFORM_COMPOSITION"
    elif any(row["phase"] == "aggregate_decision" for row in differences):
        classification = "DIFFERENT_SUPPORT_AGGREGATION"
    elif any(row["phase"] in {"quality", "score"} for row in differences):
        classification = "DIFFERENT_SCORING"
    elif left_keys != right_keys:
        classification = "DIFFERENT_CONSTRUCTION_ORDER"
    elif maximum["rotation_matrix_absolute"] or maximum[
        "translation_m_absolute"
    ]:
        classification = "NUMERICALLY_EQUIVALENT_WITHIN_TOLERANCE"
    else:
        classification = "EXACT"

    first_common_score = next(
        (
            row
            for row in differences
            if row["phase"] == "score"
            and row["candidate_type"] in {"direct", "relay"}
        ),
        None,
    )
    first_missing = differences[0] if differences else None
    first_causal_divergence = {
        "phase": "observation_quality_preparation",
        "legacy_decision": (
            "derive area / (distance^2 * (1 + center_norm_1280x720))"
        ),
        "wizard_decision": (
            "consume frozen observation_quality_v2 selection_score"
        ),
        "effect": (
            "the first common native candidate already has different quality components and aggregate score"
        ),
        "first_common_candidate_score_difference": first_common_score,
        "source_locations": {
            "legacy_direct": (
                "13_eval_direct_static_cam3_cam1_multimarker.py:198-210,291,490"
            ),
            "legacy_relay": "14_eval_moving_relay_chains.py:241-253,300,324,698-704",
            "wizard": "core.py:359-395 (selection_score), 875-964",
        },
    }
    report = {
        "schema_version": 1,
        "status": "mismatch" if differences else "equal",
        "classification": classification,
        "candidate_counts": {
            "legacy": _counts(legacy),
            "wizard": _counts(wizard),
        },
        "candidate_set": {
            "legacy_unique_count": len(left),
            "wizard_unique_count": len(right),
            "common_count": len(common),
            "missing_wizard_count": len(missing),
            "unexpected_wizard_count": len(unexpected),
            "first_missing_wizard_key": missing[0] if missing else None,
            "first_unexpected_wizard_key": unexpected[0] if unexpected else None,
            "first_missing_wizard_candidate_context": (
                _public_candidate(left[missing[0]]) if missing else None
            ),
            "first_unexpected_wizard_candidate_context": (
                _public_candidate(right[unexpected[0]]) if unexpected else None
            ),
        },
        "candidate_multiplicity": {
            "semantic_keys_are_unique_with_support_observation_occurrences": True,
            "legacy_count": len(legacy),
            "wizard_count": len(wizard),
            "parity": Counter(left_keys) == Counter(right_keys),
        },
        "original_construction_order": {
            "parity": left_keys == right_keys,
            "legacy_first_keys": left_keys[:5],
            "wizard_first_keys": right_keys[:5],
        },
        "transform_chain_directions": {
            "common_candidate_parity": not any(
                row["field"] == "transform_chain" for row in differences
            )
        },
        "composed_transforms": {
            "common_candidate_parity_within_tolerance": not any(
                row["phase"] == "composed_transform" for row in differences
            ),
            "maximum_absolute_deltas": {
                "rotation_matrix": maximum["rotation_matrix_absolute"],
                "translation_m": maximum["translation_m_absolute"],
            },
        },
        "support_rows": {
            "common_candidate_parity": not any(
                row["field"] in {"support_observation_keys", "support_count"}
                for row in differences
            )
        },
        "quality_and_scoring": {
            "parity": not any(
                row["phase"] in {"quality", "score"} for row in differences
            ),
            "maximum_absolute_deltas": {
                "quality": maximum["quality_absolute"],
                "aggregate_score": maximum["score_absolute"],
                "reprojection_px": maximum["reprojection_px_absolute"],
            },
        },
        "aggregate_acceptance_rejection": {
            "parity": not any(
                row["phase"] == "aggregate_decision" for row in differences
            )
        },
        "numeric_tolerances": NUMERIC_TOLERANCES,
        "first_list_or_set_difference": first_missing,
        "first_causal_divergence": (
            first_causal_divergence if differences else None
        ),
        "secondary_contract_differences": ([
            {
                "contract": "direct_target_scope",
                "legacy": "direct candidates only for cam_edge_1",
                "wizard": "direct candidates for every non-root camera",
                "first_effect": unexpected[0] if unexpected else None,
                "source_locations": {
                    "legacy": "13_eval_direct_static_cam3_cam1_multimarker.py:12-13,449-508",
                    "wizard": "build_candidates.py:102-119",
                },
            },
            {
                "contract": "moving_observation_cap",
                "legacy": "all registered moving observations, ascending frame",
                "wizard": "top 8 per marker by quality, then ascending frame",
                "source_locations": {
                    "legacy": "14_eval_moving_relay_chains.py:307-336,630-730",
                    "wizard": "core.py:849-872; build_candidates.py:41-91",
                },
            },
            {
                "contract": "relay_support_aggregation",
                "legacy": "one flat robust aggregate over every relay candidate",
                "wizard": "within-marker-chain aggregates then robust aggregate across independent chains",
                "source_locations": {
                    "legacy": "14_eval_moving_relay_chains.py:550-627",
                    "wizard": "core.py:750-837",
                },
            },
        ] if differences else []),
        "ground_truth_used": False,
        "solver_invoked": False,
        "colmap_invoked": False,
        "final_pose_published": False,
    }
    return report, differences


def compare_selection(
    legacy: Mapping[str, Any], wizard: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    differences: list[dict[str, Any]] = []
    if legacy["root_camera"] != wizard["root_camera"]:
        differences.append(
            {
                "camera_id": "root",
                "field": "root_camera",
                "legacy_value": legacy["root_camera"],
                "wizard_value": wizard["root_camera"],
                "reason": "different_root",
            }
        )
    for camera in CAMERA_ORDER:
        left = legacy["per_camera"][camera]
        right = wizard["per_camera"][camera]
        for field in (
            "deployment_eligible",
            "omitted",
            "omission_reason",
            "selected_candidate_type",
            "selected_method",
        ):
            if left.get(field) != right.get(field):
                differences.append(
                    {
                        "camera_id": camera,
                        "field": field,
                        "legacy_value": _json_safe(left.get(field)),
                        "wizard_value": _json_safe(right.get(field)),
                        "reason": (
                            "different_eligibility_gate"
                            if field in {
                                "deployment_eligible",
                                "omitted",
                                "omission_reason",
                            }
                            else "different_direct_relay_selection"
                            if field == "selected_candidate_type"
                            else "different_selection_label"
                        ),
                    }
                )
        left_transform = left.get("aggregate_transform")
        right_transform = right.get("aggregate_transform")
        if (left_transform is None) != (right_transform is None):
            differences.append(
                {
                    "camera_id": camera,
                    "field": "aggregate_transform",
                    "legacy_value": _json_safe(left_transform),
                    "wizard_value": _json_safe(right_transform),
                    "reason": "different_aggregate_value",
                }
            )
        elif left_transform is not None and right_transform is not None:
            for field in ("rotation", "translation_m"):
                left_value = np.asarray(left_transform[field], dtype=np.float64)
                right_value = np.asarray(right_transform[field], dtype=np.float64)
                if not np.array_equal(left_value, right_value):
                    differences.append(
                        {
                            "camera_id": camera,
                            "field": f"aggregate_transform.{field}",
                            "legacy_value": _json_safe(left_transform[field]),
                            "wizard_value": _json_safe(right_transform[field]),
                            "reason": "different_aggregate_value",
                        }
                    )
        for field in ("aggregate_statistics", "selected_aggregate_type"):
            if _json_safe(left.get(field)) != _json_safe(right.get(field)):
                differences.append(
                    {
                        "camera_id": camera,
                        "field": field,
                        "legacy_value": _json_safe(left.get(field)),
                        "wizard_value": _json_safe(right.get(field)),
                        "reason": "different_aggregate_value",
                    }
                )
    for field in (
        "camera_traversal_order",
        "candidate_eligibility_before_ranking",
        "candidate_sorting",
        "tie_breaking",
        "candidate_counts",
    ):
        if _json_safe(legacy.get(field)) != _json_safe(wizard.get(field)):
            differences.append(
                {
                    "camera_id": "all",
                    "field": field,
                    "legacy_value": _json_safe(legacy.get(field)),
                    "wizard_value": _json_safe(wizard.get(field)),
                    "reason": "different_ranking_or_tie_break_contract",
                }
            )
    if any(row["reason"] == "different_root" for row in differences):
        classification = "DIFFERENT_ROOT"
    elif any(
        row["reason"] == "different_eligibility_gate" for row in differences
    ):
        classification = "DIFFERENT_ELIGIBILITY_GATE"
    elif any(
        row["reason"] == "different_direct_relay_selection"
        for row in differences
    ):
        classification = "DIFFERENT_DIRECT_RELAY_SELECTION"
    else:
        classification = "EXACT" if not differences else "DIFFERENT_RANKING"
    first = differences[0] if differences else None
    first_gate = next(
        (
            row
            for row in differences
            if row["reason"] == "different_eligibility_gate"
        ),
        first,
    )
    report = {
        "schema_version": 1,
        "status": "equal" if not differences else "mismatch",
        "classification": classification,
        "root_selection": {
            "legacy": legacy["root_camera"],
            "wizard": wizard["root_camera"],
            "parity": legacy["root_camera"] == wizard["root_camera"],
        },
        "legacy_per_camera": legacy["per_camera"],
        "wizard_per_camera": wizard["per_camera"],
        "candidate_eligibility_before_ranking": {
            "legacy": legacy["candidate_eligibility_before_ranking"],
            "wizard": wizard["candidate_eligibility_before_ranking"],
        },
        "candidate_sorting": {
            "legacy": legacy["candidate_sorting"],
            "wizard": wizard["candidate_sorting"],
        },
        "tie_breaking": {
            "legacy": legacy["tie_breaking"],
            "wizard": wizard["tie_breaking"],
        },
        "first_difference": first,
        "first_causal_divergence": ({
            "difference": first_gate,
            "legacy_decision": (
                "publish the fixed per-camera aggregate whenever its row exists"
            ),
            "wizard_decision": (
                "require direct/relay consensus stability and path consistency before deployment eligibility"
            ),
            "source_locations": {
                "legacy": (
                    "15_export_final_extrinsics_cam3_reference.py:125-218,585-618"
                ),
                "wizard": "solve_extrinsics.py:15-122,173-295",
            },
        } if differences else None),
        "ground_truth_used": False,
        "solver_invoked": False,
        "colmap_invoked": False,
        "final_pose_published": False,
    }
    return report, differences


def _csv_ready(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: _json_safe(value)
            if not isinstance(value, (dict, list, tuple))
            else _json(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def preserve_pre_fix_evidence(ap01_root: Path) -> dict[str, Any]:
    """Hash-reference the immutable mismatch evidence without copying/rewriting it."""

    ap01_root = ap01_root.resolve()
    required = [
        ap01_root / "AP01_CANDIDATE_PARITY.json",
        ap01_root / "AP01_SELECTION_PARITY.json",
        ap01_root / "AP01_CANDIDATE_DIFF.csv",
        ap01_root / "AP01_SELECTION_DIFF.csv",
    ]
    for directory in (ap01_root / "legacy", ap01_root / "wizard"):
        required.extend(path for path in sorted(directory.rglob("*")) if path.is_file())
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing pre-fix AP01 evidence: {missing[0]}")
    files = [
        {
            "path": path.relative_to(ap01_root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in required
    ]
    manifest = {
        "schema_version": 1,
        "evidence_role": "immutable_pre_fix_main_vs_recommended_wizard_mismatch",
        "storage": "hash references to unchanged sibling evidence files",
        "files": files,
        "file_count": len(files),
        "ground_truth_used": False,
        "final_pose_published": False,
    }
    destination = ap01_root / "pre_fix" / "PRE_FIX_MANIFEST.json"
    write_json(destination, manifest)
    return manifest


def generate_ap01_post_fix_parity(
    *, parity_root: Path, legacy_worktree: Path
) -> dict[str, Any]:
    """Generate only isolated post-fix candidate/selection parity evidence."""

    parity_root = parity_root.resolve()
    legacy_worktree = legacy_worktree.resolve()
    for path in (parity_root, legacy_worktree):
        assert_pre_solver_path(path)
    ap01_root = parity_root / "ap01"
    pre_fix = preserve_pre_fix_evidence(ap01_root)
    frozen_root = parity_root / "frozen"
    post_fix = ap01_root / "post_fix"
    legacy_records, legacy_selection = extract_legacy_candidates(
        frozen_observations=frozen_root / "ap01_accepted_observations.jsonl",
        frozen_colmap_images=frozen_root / "ap01_colmap_images.txt",
        frozen_metric_scale=frozen_root / "ap01_metric_scale.txt",
        output_root=post_fix / "legacy",
        legacy_source_root=(
            legacy_worktree / "run/bus_real_data/approach1_marker_direct_relay"
        ),
    )
    wizard_records, wizard_selection = extract_wizard_parity_candidates(
        frozen_observations=frozen_root / "ap01_accepted_observations.jsonl",
        frozen_colmap_images=frozen_root / "ap01_colmap_images.txt",
        frozen_metric_scale=frozen_root / "ap01_metric_scale.txt",
        output_root=post_fix / "wizard",
        wizard_source_root=(
            Path(__file__).resolve().parents[2]
            / "src/camera_rig_calibration/methods/ap01"
        ),
    )
    candidate_report, candidate_differences = compare_candidates(
        legacy_records, wizard_records
    )
    selection_report, selection_differences = compare_selection(
        legacy_selection, wizard_selection
    )
    write_json(post_fix / "AP01_CANDIDATE_PARITY.json", candidate_report)
    write_csv(
        post_fix / "AP01_CANDIDATE_DIFF.csv",
        _csv_ready(candidate_differences),
        list(CANDIDATE_DIFF_FIELDS),
    )
    write_json(post_fix / "AP01_SELECTION_PARITY.json", selection_report)
    write_csv(
        post_fix / "AP01_SELECTION_DIFF.csv",
        _csv_ready(selection_differences),
        list(SELECTION_DIFF_FIELDS),
    )
    return {
        "pre_fix_manifest": pre_fix,
        "candidate_parity": candidate_report,
        "selection_parity": selection_report,
        "post_fix": str(post_fix),
    }


def generate_ap01_candidate_parity(
    *,
    parity_root: Path,
    legacy_worktree: Path,
) -> dict[str, Any]:
    """Generate the requested evidence without entering any solver stage."""

    parity_root = parity_root.resolve()
    legacy_worktree = legacy_worktree.resolve()
    for path in (parity_root, legacy_worktree):
        assert_pre_solver_path(path)
    frozen_root = parity_root / "frozen"
    manifest = freeze_ap01_input(
        legacy_accepted_csv=(
            parity_root
            / "generated/main_legacy_observations/shared_all_aruco_observations.csv"
        ),
        wizard_accepted_csv=(
            parity_root
            / "generated/wizard_observations/filtered/accepted_observations.csv"
        ),
        colmap_images_txt=(
            legacy_worktree
            / "results/bus_real_data/01_marker_direct_relay_multimarker_multichain"
            / "04_moving_camera_colmap_trajectory/sparse_txt_best/images.txt"
        ),
        metric_scale_txt=(
            legacy_worktree
            / "results/bus_real_data/01_marker_direct_relay_multimarker_multichain"
            / "04_moving_camera_colmap_trajectory/aruco_metric_scale/metric_scale.txt"
        ),
        frozen_root=frozen_root,
    )
    ap01_root = parity_root / "ap01"
    legacy_records, legacy_selection = extract_legacy_candidates(
        frozen_observations=frozen_root / "ap01_accepted_observations.jsonl",
        frozen_colmap_images=frozen_root / "ap01_colmap_images.txt",
        frozen_metric_scale=frozen_root / "ap01_metric_scale.txt",
        output_root=ap01_root / "legacy",
        legacy_source_root=(
            legacy_worktree / "run/bus_real_data/approach1_marker_direct_relay"
        ),
    )
    wizard_records, wizard_selection = extract_wizard_candidates(
        frozen_observations=frozen_root / "ap01_accepted_observations.jsonl",
        frozen_colmap_images=frozen_root / "ap01_colmap_images.txt",
        frozen_metric_scale=frozen_root / "ap01_metric_scale.txt",
        output_root=ap01_root / "wizard",
        wizard_source_root=(
            Path(__file__).resolve().parents[2]
            / "src/camera_rig_calibration/methods/ap01"
        ),
    )
    candidate_report, candidate_differences = compare_candidates(
        legacy_records, wizard_records
    )
    selection_report, selection_differences = compare_selection(
        legacy_selection, wizard_selection
    )
    write_json(ap01_root / "AP01_CANDIDATE_PARITY.json", candidate_report)
    write_json(parity_root / "AP01_CANDIDATE_PARITY.json", candidate_report)
    write_csv(
        ap01_root / "AP01_CANDIDATE_DIFF.csv",
        _csv_ready(candidate_differences),
        list(CANDIDATE_DIFF_FIELDS),
    )
    write_json(ap01_root / "AP01_SELECTION_PARITY.json", selection_report)
    write_json(parity_root / "AP01_SELECTION_PARITY.json", selection_report)
    write_csv(
        ap01_root / "AP01_SELECTION_DIFF.csv",
        _csv_ready(selection_differences),
        list(SELECTION_DIFF_FIELDS),
    )
    return {
        "manifest": manifest,
        "candidate_parity": candidate_report,
        "selection_parity": selection_report,
        "paths": {
            "frozen": str(frozen_root.resolve()),
            "ap01": str(ap01_root.resolve()),
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate AP01 pre-solver candidate/selection parity evidence"
    )
    parser.add_argument("--parity-root", type=Path, required=True)
    parser.add_argument("--legacy-worktree", type=Path, required=True)
    arguments = parser.parse_args()
    result = generate_ap01_candidate_parity(
        parity_root=arguments.parity_root,
        legacy_worktree=arguments.legacy_worktree,
    )
    print(
        _json(
            {
                "frozen_row_count": result["manifest"]["canonical_input"][
                    "row_count"
                ],
                "frozen_sha256": result["manifest"]["canonical_input"][
                    "sha256"
                ],
                "candidate_classification": result["candidate_parity"][
                    "classification"
                ],
                "selection_classification": result["selection_parity"][
                    "classification"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()

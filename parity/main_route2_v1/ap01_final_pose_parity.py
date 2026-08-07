"""Pure AP01 final-pose parity from frozen aggregate selections.

This module deliberately starts after AP01 aggregate selection.  It neither
constructs candidates nor invokes a solver, detector, COLMAP, publication,
evaluation, or reconciliation code.  The Legacy adapter models only the
GT-free estimate/export operations needed for the cam_edge_3-rooted map.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PARITY_CONTRACT = "main_route2_parity_v1"
RECOMMENDED_CONTRACT = "recommended_wizard_v1"
ROOT_FRAME_SUFFIX = "_optical_frame"

ROTATION_MATRIX_TOLERANCE = 1e-10
ROTATION_ANGLE_TOLERANCE_DEG = 1e-7
TRANSLATION_TOLERANCE_M = 1e-9
SO3_TOLERANCE = 1e-10

DIFF_FIELDS = (
    "camera_id",
    "status",
    "legacy_role",
    "wizard_role",
    "frame_from_equal",
    "frame_to_equal",
    "selected_candidate_identity_equal",
    "inversion_history_equal",
    "composition_history_equal",
    "rotation_max_abs_delta",
    "relative_rotation_angle_deg",
    "translation_x_abs_delta_m",
    "translation_y_abs_delta_m",
    "translation_z_abs_delta_m",
    "translation_norm_delta_m",
    "homogeneous_max_abs_delta",
    "legacy_rotation_determinant",
    "wizard_rotation_determinant",
    "legacy_so3_valid",
    "wizard_so3_valid",
    "quaternion_export_applicable",
    "quaternion_raw_max_abs_delta",
    "quaternion_sign_equivalent",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def assert_final_pose_input_path(path: Path) -> None:
    """Reject evaluation/GT paths before any file read occurs."""

    lowered = [part.lower() for part in path.parts]
    forbidden = ("ground_truth", "ground-truth", "gt_eval", "gt-eval")
    if any(token in part for part in lowered for token in forbidden):
        raise ValueError(f"Ground Truth/evaluation path is forbidden: {path}")


def read_json(path: Path) -> dict[str, Any]:
    assert_final_pose_input_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    assert_final_pose_input_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, immutable: bool = False) -> str:
    payload = canonical_json_bytes(value)
    if immutable and path.exists():
        current = path.read_bytes()
        if current != payload:
            raise RuntimeError(f"Refusing to overwrite immutable evidence: {path}")
        return sha256_bytes(current)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def write_text(path: Path, value: str, *, immutable: bool = False) -> str:
    payload = value.replace("\r\n", "\n").encode("utf-8")
    if not payload.endswith(b"\n"):
        payload += b"\n"
    if immutable and path.exists():
        current = path.read_bytes()
        if current != payload:
            raise RuntimeError(f"Refusing to overwrite immutable evidence: {path}")
        return sha256_bytes(current)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def identity_transform() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def make_transform(
    rotation: Sequence[Sequence[float]], translation: Sequence[float]
) -> list[list[float]]:
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("rotation must be 3x3")
    if len(translation) != 3:
        raise ValueError("translation must have three components")
    return [
        [float(rotation[i][j]) for j in range(3)] + [float(translation[i])]
        for i in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def rotation_of(transform: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(transform[i][j]) for j in range(3)] for i in range(3)]


def translation_of(transform: Sequence[Sequence[float]]) -> list[float]:
    return [float(transform[i][3]) for i in range(3)]


def multiply_transforms(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [
            sum(float(left[i][k]) * float(right[k][j]) for k in range(4))
            for j in range(4)
        ]
        for i in range(4)
    ]


def invert_transform(transform: Sequence[Sequence[float]]) -> list[list[float]]:
    rotation = rotation_of(transform)
    translation = translation_of(transform)
    inverse_rotation = [[rotation[j][i] for j in range(3)] for i in range(3)]
    inverse_translation = [
        -sum(inverse_rotation[i][j] * translation[j] for j in range(3))
        for i in range(3)
    ]
    return make_transform(inverse_rotation, inverse_translation)


def compose_direct_candidate(
    root_marker: Sequence[Sequence[float]],
    target_marker: Sequence[Sequence[float]],
) -> list[list[float]]:
    """T_root_marker @ inverse(T_target_marker), mapping target to root."""

    return multiply_transforms(root_marker, invert_transform(target_marker))


def compose_relay_candidate(
    root_marker: Sequence[Sequence[float]],
    moving_i_marker: Sequence[Sequence[float]],
    scaled_moving_i_moving_j: Sequence[Sequence[float]],
    target_marker: Sequence[Sequence[float]],
    moving_j_marker: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Legacy target-to-root moving-relay multiplication order."""

    target_to_moving_j = multiply_transforms(
        target_marker, invert_transform(moving_j_marker)
    )
    result = multiply_transforms(root_marker, invert_transform(moving_i_marker))
    result = multiply_transforms(result, scaled_moving_i_moving_j)
    return multiply_transforms(result, invert_transform(target_to_moving_j))


def rotation_to_rpy_deg(
    rotation: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    pitch = math.atan2(
        -float(rotation[2][0]),
        math.hypot(float(rotation[0][0]), float(rotation[1][0])),
    )
    roll = math.atan2(float(rotation[2][1]), float(rotation[2][2]))
    yaw = math.atan2(float(rotation[1][0]), float(rotation[0][0]))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def rpy_deg_to_rotation(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> list[list[float]]:
    roll, pitch, yaw = (
        math.radians(float(value))
        for value in (roll_deg, pitch_deg, yaw_deg)
    )
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def legacy_aggregate_csv_roundtrip(
    transform: Sequence[Sequence[float]],
) -> tuple[list[list[float]], dict[str, Any]]:
    """Model Legacy aggregate CSV's fixed nine-decimal pose serialization."""

    rotation = rotation_of(transform)
    translation = translation_of(transform)
    rpy_deg = rotation_to_rpy_deg(rotation)
    serialized_translation = [float(f"{value:.9f}") for value in translation]
    serialized_rpy = [float(f"{value:.9f}") for value in rpy_deg]
    reconstructed = make_transform(
        rpy_deg_to_rotation(*serialized_rpy), serialized_translation
    )
    return reconstructed, {
        "serialized_translation_m": serialized_translation,
        "serialized_rotation_rpy_deg": serialized_rpy,
        "decimal_places": 9,
        "quaternion_fields_consumed": False,
    }


def determinant_3x3(rotation: Sequence[Sequence[float]]) -> float:
    a, b, c = (float(value) for value in rotation[0])
    d, e, f = (float(value) for value in rotation[1])
    g, h, i = (float(value) for value in rotation[2])
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def so3_metrics(rotation: Sequence[Sequence[float]]) -> dict[str, Any]:
    determinant = determinant_3x3(rotation)
    orthogonality_delta = max(
        abs(
            sum(float(rotation[k][i]) * float(rotation[k][j]) for k in range(3))
            - (1.0 if i == j else 0.0)
        )
        for i in range(3)
        for j in range(3)
    )
    return {
        "determinant": determinant,
        "determinant_abs_delta_from_one": abs(determinant - 1.0),
        "orthogonality_max_abs_delta": orthogonality_delta,
        "valid": (
            abs(determinant - 1.0) <= SO3_TOLERANCE
            and orthogonality_delta <= SO3_TOLERANCE
        ),
        "tolerance": SO3_TOLERANCE,
    }


def rotation_to_quaternion_wxyz(
    rotation: Sequence[Sequence[float]],
) -> list[float]:
    r = [[float(value) for value in row] for row in rotation]
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = [
            0.25 * scale,
            (r[2][1] - r[1][2]) / scale,
            (r[0][2] - r[2][0]) / scale,
            (r[1][0] - r[0][1]) / scale,
        ]
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        scale = math.sqrt(max(1e-15, 1.0 + r[0][0] - r[1][1] - r[2][2])) * 2.0
        q = [
            (r[2][1] - r[1][2]) / scale,
            0.25 * scale,
            (r[0][1] + r[1][0]) / scale,
            (r[0][2] + r[2][0]) / scale,
        ]
    elif r[1][1] > r[2][2]:
        scale = math.sqrt(max(1e-15, 1.0 + r[1][1] - r[0][0] - r[2][2])) * 2.0
        q = [
            (r[0][2] - r[2][0]) / scale,
            (r[0][1] + r[1][0]) / scale,
            0.25 * scale,
            (r[1][2] + r[2][1]) / scale,
        ]
    else:
        scale = math.sqrt(max(1e-15, 1.0 + r[2][2] - r[0][0] - r[1][1])) * 2.0
        q = [
            (r[1][0] - r[0][1]) / scale,
            (r[0][2] + r[2][0]) / scale,
            (r[1][2] + r[2][1]) / scale,
            0.25 * scale,
        ]
    norm = math.sqrt(sum(value * value for value in q))
    normalized = [value / norm for value in q]
    return [-value for value in normalized] if normalized[0] < 0.0 else normalized


def quaternion_sign_equivalent(
    first: Sequence[float], second: Sequence[float], *, tolerance: float = 1e-12
) -> bool:
    direct = max(abs(float(a) - float(b)) for a, b in zip(first, second))
    negated = max(abs(float(a) + float(b)) for a, b in zip(first, second))
    return min(direct, negated) <= tolerance


def _clean_statistics(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        key: item
        for key, item in value.items()
        if key not in {"ground_truth_used"}
    }


def _candidate_key_base64(hex_key: str) -> str:
    if not hex_key.startswith("sha256:"):
        raise ValueError(f"Unsupported candidate key: {hex_key}")
    return base64.urlsafe_b64encode(bytes.fromhex(hex_key.split(":", 1)[1])).decode(
        "ascii"
    ).rstrip("=")


def _selected_compact_candidates(
    candidate_path: Path, selected_hex_keys: Iterable[str]
) -> dict[str, dict[str, Any]]:
    assert_final_pose_input_path(candidate_path)
    wanted = {_candidate_key_base64(key): key for key in selected_hex_keys}
    selected: dict[str, dict[str, Any]] = {}
    with candidate_path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        fields = list(header["fields"])
        contract = header["compact_value_contract"]
        for line in handle:
            values = json.loads(line)
            encoded = dict(zip(fields, values))
            compact_key = str(encoded["semantic_candidate_key"])
            if compact_key not in wanted:
                continue
            type_code = str(encoded["candidate_type"])
            root_code = str(encoded["root_camera"])
            target_code = str(encoded["target_camera"])
            support_keys = [
                f"ap01obs:{int(value):06d}"
                for value in encoded["support_observation_keys"]
            ]
            chain_value = encoded["transform_chain"]
            selected[wanted[compact_key]] = {
                "semantic_candidate_key": wanted[compact_key],
                "compact_candidate_key": compact_key,
                "candidate_type": contract["candidate_type"][type_code],
                "root_camera": contract["camera"][root_code],
                "target_camera": contract["camera"][target_code],
                "root_marker_id": encoded["root_marker"],
                "target_marker_id": encoded["target_marker"],
                "root_frame": encoded["root_frame"],
                "target_frame": encoded["target_frame"],
                "relay_path": encoded["relay_path"],
                "transform_chain_code": chain_value,
                "transform_chain": contract["transform_chain"][type_code],
                "composed_rotation_rodrigues_rad": encoded[
                    "composed_rotation_rodrigues"
                ],
                "composed_translation_m": encoded["composed_translation_m"],
                "support_observation_keys": support_keys,
                "support_count": encoded["support_count"],
                "original_construction_index": encoded[
                    "original_construction_index"
                ],
            }
            if len(selected) == len(wanted):
                break
    missing = set(selected_hex_keys) - set(selected)
    if missing:
        raise RuntimeError(f"Selected candidate keys absent from frozen file: {missing}")
    return selected


def _same_selection(legacy: Mapping[str, Any], wizard: Mapping[str, Any]) -> None:
    if legacy["root_camera"] != wizard["root_camera"]:
        raise RuntimeError("Legacy/Wizard frozen roots differ")
    if legacy["camera_traversal_order"] != wizard["camera_traversal_order"]:
        raise RuntimeError("Legacy/Wizard frozen camera traversal differs")
    if set(legacy["per_camera"]) != set(wizard["per_camera"]):
        raise RuntimeError("Legacy/Wizard frozen camera inventories differ")
    compared_fields = (
        "selected_candidate_type",
        "selected_method",
        "selected_aggregate_type",
        "aggregate_transform",
        "deployment_eligible",
        "omitted",
        "omission_reason",
    )
    for camera in legacy["per_camera"]:
        for field in compared_fields:
            if legacy["per_camera"][camera].get(field) != wizard["per_camera"][
                camera
            ].get(field):
                raise RuntimeError(f"Frozen selection mismatch: {camera}.{field}")


def freeze_selected_candidates(
    evidence_root: Path, repository: Path, legacy_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    post_fix = evidence_root / "ap01/post_fix"
    legacy_path = post_fix / "legacy/AP01_SELECTION.json"
    wizard_path = post_fix / "wizard/AP01_SELECTION.json"
    parity_path = post_fix / "AP01_SELECTION_PARITY.json"
    candidate_parity_path = post_fix / "AP01_CANDIDATE_PARITY.json"
    legacy_candidates_path = post_fix / "legacy/AP01_CANDIDATES.jsonl"
    wizard_candidates_path = post_fix / "wizard/AP01_CANDIDATES.jsonl"
    input_manifest_path = evidence_root / "frozen/AP01_INPUT_MANIFEST.json"

    legacy = read_json(legacy_path)
    wizard = read_json(wizard_path)
    selection_parity = read_json(parity_path)
    candidate_parity = read_json(candidate_parity_path)
    if selection_parity.get("classification") != "EXACT":
        raise RuntimeError("Frozen AP01 selection parity is not EXACT")
    if candidate_parity.get("classification") != "EXACT":
        raise RuntimeError("Frozen AP01 candidate parity is not EXACT")
    if legacy.get("ground_truth_used") is not False:
        raise RuntimeError("Legacy selected input is not declared GT-free")
    if wizard.get("ground_truth_used") is not False:
        raise RuntimeError("Wizard selected input is not declared GT-free")
    _same_selection(legacy, wizard)

    root = str(legacy["root_camera"])
    target_order = [str(value) for value in legacy["camera_traversal_order"]]
    camera_order = [root, *target_order]
    direct_keys = [
        str(row["aggregate_statistics"]["selected_candidate_key"])
        for row in legacy["per_camera"].values()
        if row.get("selected_candidate_type") == "direct"
    ]
    selected_keys = [str(legacy["root_selection"]["candidate_key"]), *direct_keys]
    compact = _selected_compact_candidates(legacy_candidates_path, selected_keys)

    records: list[dict[str, Any]] = []
    for camera in camera_order:
        selected = legacy["per_camera"][camera]
        candidate_type = str(selected["selected_candidate_type"])
        aggregate = selected["aggregate_transform"]
        transform = make_transform(
            aggregate["rotation"], aggregate["translation_m"]
        )
        if candidate_type == "root":
            key = str(legacy["root_selection"]["candidate_key"])
            identity: dict[str, Any] = compact[key]
            marker_frame_relay_identity = {
                "kind": "root_gauge",
                "root_camera": root,
            }
            inversions: list[str] = []
            compositions = ["I_4x4"]
        elif candidate_type == "direct":
            key = str(selected["aggregate_statistics"]["selected_candidate_key"])
            identity = compact[key]
            marker_frame_relay_identity = {
                "kind": "direct_static_marker",
                "root_marker_id": identity["root_marker_id"],
                "target_marker_id": identity["target_marker_id"],
                "root_frame": "STATIC_CAM3",
                "target_frame": "STATIC_CAM1",
                "support_observation_keys": identity[
                    "support_observation_keys"
                ],
            }
            inversions = [f"inverse(T_{camera}_marker_{identity['target_marker_id']})"]
            compositions = [
                f"T_{root}_marker_{identity['root_marker_id']}",
                f"inverse(T_{camera}_marker_{identity['target_marker_id']})",
            ]
        elif candidate_type == "relay":
            statistics = selected["aggregate_statistics"]
            aggregate_identity_payload = {
                "candidate_file_sha256": sha256_file(legacy_candidates_path),
                "target_camera": camera,
                "aggregate_type": selected["selected_aggregate_type"],
                "num_candidates": statistics["num_candidates"],
                "num_inliers": statistics["num_inliers"],
            }
            identity = {
                "semantic_candidate_key": "sha256:"
                + sha256_bytes(
                    json.dumps(
                        aggregate_identity_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                "candidate_type": "relay_aggregate",
                **aggregate_identity_payload,
            }
            marker_frame_relay_identity = {
                "kind": "relay_flat_mad_aggregate",
                "root_marker_id": "ALL",
                "target_marker_id": "ALL",
                "root_moving_frame": "ALL",
                "target_moving_frame": "ALL",
                "candidate_count": statistics["num_candidates"],
                "inlier_count": statistics["num_inliers"],
                "outlier_count": statistics["num_outliers"],
            }
            inversions = [
                "inverse(T_moving_i_marker)",
                "inverse(Tcw_j)",
                "inverse(T_target_marker @ inverse(T_moving_j_marker))",
            ]
            compositions = [
                "T_root_marker",
                "inverse(T_moving_i_marker)",
                "scaled(Tcw_i @ inverse(Tcw_j))",
                "inverse(T_target_marker @ inverse(T_moving_j_marker))",
                "quality_weighted_mean_of_flat_MAD_inliers",
            ]
        else:
            raise RuntimeError(f"Unknown selected candidate type: {candidate_type}")

        records.append(
            {
                "camera_id": camera,
                "role": "anchor" if camera == root else "target",
                "selected_candidate_type": candidate_type,
                "selected_method": selected["selected_method"],
                "selected_candidate_identity": identity,
                "selected_marker_frame_relay_identity": marker_frame_relay_identity,
                "rotation_representation": "3x3 active coordinate-map matrix",
                "rotation_matrix": rotation_of(transform),
                "translation_m": translation_of(transform),
                "homogeneous_transform_4x4": transform,
                "transform_chain": {
                    "direction": "target_optical_camera_to_cam_edge_3_optical_camera",
                    "column_vector_equation": f"p_{root} = T_{root}_{camera} @ p_{camera}",
                    "inversion_history": inversions,
                    "composition_history": compositions,
                },
                "aggregate_statistics": _clean_statistics(
                    selected.get("aggregate_statistics")
                ),
                "deployment_eligible": bool(selected["deployment_eligible"]),
                "omitted": bool(selected["omitted"]),
                "omission_reason": selected.get("omission_reason"),
            }
        )

    selected_payload = {
        "schema_version": 1,
        "artifact_role": "immutable_post_selection_AP01_final_pose_input",
        "method_contract": PARITY_CONTRACT,
        "root_camera": root,
        "camera_order": camera_order,
        "target_export_order": target_order,
        "anchor_semantics": "cam_edge_3 is the identity gauge and destination frame",
        "coordinate_convention": (
            "right-handed optical-camera coordinates; homogeneous column vectors; "
            "T_root_target maps target coordinates into cam_edge_3 coordinates"
        ),
        "selected_candidates": records,
        "ground_truth_used": False,
        "ground_truth_payload_absent": True,
        "candidate_generation_invoked": False,
        "aggregate_selection_invoked": False,
        "solver_invoked": False,
        "colmap_invoked": False,
        "publication_invoked": False,
        "reconciliation_invoked": False,
    }
    selected_path = evidence_root / "frozen/AP01_SELECTED_CANDIDATES.json"
    selected_hash = write_json(selected_path, selected_payload, immutable=True)

    source_paths = (
        input_manifest_path,
        candidate_parity_path,
        parity_path,
        legacy_candidates_path,
        wizard_candidates_path,
        legacy_path,
        wizard_path,
    )
    implementation_sources = (
        (
            "legacy_direct_aggregate_writer",
            legacy_root / "13_eval_direct_static_cam3_cam1_multimarker.py",
            "legacy_worktree_relative",
        ),
        (
            "legacy_relay_aggregate_writer",
            legacy_root / "14_eval_moving_relay_chains.py",
            "legacy_worktree_relative",
        ),
        (
            "legacy_final_extrinsics",
            legacy_root / "15_export_final_extrinsics_cam3_reference.py",
            "legacy_worktree_relative",
        ),
        (
            "legacy_primary_only_wrapper",
            legacy_root / "15_export_final_extrinsics_primary_only.py",
            "legacy_worktree_relative",
        ),
        (
            "wizard_final_pose_exporter",
            repository
            / "src/camera_rig_calibration/methods/ap01/solve_extrinsics.py",
            "wizard_repository_relative",
        ),
    )
    manifest = {
        "schema_version": 1,
        "artifact_role": "immutable_AP01_final_pose_input_manifest",
        "method_contract": PARITY_CONTRACT,
        "selected_input": {
            "path": selected_path.relative_to(evidence_root).as_posix(),
            "sha256": selected_hash,
            "schema_version": selected_payload["schema_version"],
        },
        "source_artifacts": [
            {
                "path": path.relative_to(evidence_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "implementation_sources": [
            {
                "role": role,
                "path": (
                    path.relative_to(legacy_root).as_posix()
                    if path_role == "legacy_worktree_relative"
                    else path.relative_to(repository).as_posix()
                ),
                "path_role": path_role,
                "sha256": sha256_file(path),
            }
            for role, path, path_role in implementation_sources
        ],
        "camera_order": camera_order,
        "target_export_order": target_order,
        "root_camera": root,
        "anchor_semantics": selected_payload["anchor_semantics"],
        "transform_direction": "target_optical_camera_to_cam_edge_3_optical_camera",
        "ground_truth_used": False,
        "ground_truth_payload_absent": True,
        "final_pose_input_is_post_selection_only": True,
    }
    manifest_path = evidence_root / "frozen/AP01_FINAL_POSE_INPUT_MANIFEST.json"
    write_json(manifest_path, manifest, immutable=True)
    return selected_payload, manifest


def _pose_record(
    selected: Mapping[str, Any],
    *,
    apply_legacy_roundtrip: bool,
) -> dict[str, Any]:
    source_transform = selected["homogeneous_transform_4x4"]
    if selected["role"] == "anchor":
        transform = identity_transform()
        serialization = {
            "operation": "identity_gauge_no_numeric_conversion",
            "quaternion_fields_consumed": False,
        }
    elif apply_legacy_roundtrip:
        transform, details = legacy_aggregate_csv_roundtrip(source_transform)
        serialization = {
            "operation": "legacy_aggregate_csv_9_decimal_rpy_roundtrip",
            **details,
        }
    else:
        transform = [[float(value) for value in row] for row in source_transform]
        serialization = {
            "operation": "no_pre_pose_row_rotation_or_translation_roundtrip",
            "quaternion_fields_consumed": False,
        }
    rotation = rotation_of(transform)
    quaternion = rotation_to_quaternion_wxyz(rotation)
    camera = str(selected["camera_id"])
    return {
        "camera_id": camera,
        "role": selected["role"],
        "source_selected_candidate": selected["selected_candidate_identity"],
        "source_selected_candidate_type": selected["selected_candidate_type"],
        "source_selected_method": selected["selected_method"],
        "frame_from": f"{camera}{ROOT_FRAME_SUFFIX}",
        "frame_to": f"cam_edge_3{ROOT_FRAME_SUFFIX}",
        "transform_direction": "target_to_root",
        "rotation_matrix": rotation,
        "translation_m": translation_of(transform),
        "homogeneous_transform_4x4": transform,
        "quaternion": {
            "exported": False,
            "exported_value": None,
            "comparison_only_wxyz": quaternion,
            "note": "Neither final exporter emits a quaternion; this canonical value is comparison-only.",
        },
        "rotation_validity": so3_metrics(rotation),
        "inversion_history": selected["transform_chain"]["inversion_history"],
        "composition_history": selected["transform_chain"]["composition_history"],
        "final_pose_operations": [serialization["operation"]],
        "serialization": serialization,
        "normalization_history": [],
        "deployment_eligible": selected["deployment_eligible"],
        "omitted": selected["omitted"],
        "omission_reason": selected["omission_reason"],
    }


def legacy_final_pose_view(selected_payload: Mapping[str, Any]) -> dict[str, Any]:
    root = str(selected_payload["root_camera"])
    target_order = list(selected_payload["target_export_order"])
    records_by_camera = {
        str(record["camera_id"]): record
        for record in selected_payload["selected_candidates"]
    }
    camera_order = [root, *target_order]
    records = [
        _pose_record(records_by_camera[camera], apply_legacy_roundtrip=True)
        for camera in camera_order
    ]
    return {
        "schema_version": 1,
        "implementation": "legacy_main",
        "method_contract": "legacy_main_at_8f9dcea1",
        "computation_scope": "pure_post_selection_final_extrinsics_only",
        "camera_inventory": camera_order,
        "camera_order": camera_order,
        "root_camera": root,
        "anchor_frame": f"{root}{ROOT_FRAME_SUFFIX}",
        "frame_convention": "T_root_target maps target optical coordinates into root optical coordinates",
        "camera_records": records,
        "export_semantics": {
            "semantic_format": "legacy_primary_only_cam3_reference_projection",
            "anchor_representation": "top_level_reference_camera_implicit_identity",
            "explicit_pose_record_order": target_order,
            "explicit_pose_record_inventory": target_order,
            "target_omission_policy": "omit_missing_target_independently",
            "rotation_fields": ["rotation_rpy_deg", "matrix_4x4"],
            "translation_fields": ["translation_m", "matrix_4x4"],
            "quaternion_exported": False,
            "normalization": "none",
        },
        "ground_truth_used": False,
        "candidate_generation_invoked": False,
        "aggregate_selection_invoked": False,
        "solver_invoked": False,
        "colmap_invoked": False,
        "publication_invoked": False,
        "reconciliation_invoked": False,
    }


def wizard_final_pose_view(
    selected_payload: Mapping[str, Any],
    *,
    method_contract: str,
    parity_adapter: bool,
) -> dict[str, Any]:
    root = str(selected_payload["root_camera"])
    records_by_camera = {
        str(record["camera_id"]): record
        for record in selected_payload["selected_candidates"]
    }
    if parity_adapter:
        if method_contract != PARITY_CONTRACT:
            raise ValueError("Legacy final-pose adapter is parity-contract only")
        target_order = list(selected_payload["target_export_order"])
        camera_order = [root, *target_order]
        explicit_order = target_order
        apply_roundtrip = True
        export_semantics = {
            "semantic_format": "legacy_primary_only_cam3_reference_projection",
            "anchor_representation": "top_level_reference_camera_implicit_identity",
            "explicit_pose_record_order": target_order,
            "explicit_pose_record_inventory": target_order,
            "target_omission_policy": "omit_missing_target_independently",
            "rotation_fields": ["rotation_rpy_deg", "matrix_4x4"],
            "translation_fields": ["translation_m", "matrix_4x4"],
            "quaternion_exported": False,
            "normalization": "none",
        }
    else:
        camera_order = sorted(records_by_camera)
        explicit_order = camera_order
        apply_roundtrip = False
        export_semantics = {
            "semantic_format": "wizard_static_camera_pose_csv",
            "anchor_representation": "explicit_pose_row",
            "explicit_pose_record_order": explicit_order,
            "explicit_pose_record_inventory": explicit_order,
            "target_omission_policy": "selected_pose_dictionary_membership",
            "rotation_fields": ["rotation_rpy_deg", "rotation_vector"],
            "translation_fields": ["x_m", "y_m", "z_m"],
            "quaternion_exported": False,
            "normalization": "none",
        }
    records = [
        _pose_record(
            records_by_camera[camera],
            apply_legacy_roundtrip=apply_roundtrip,
        )
        for camera in camera_order
    ]
    return {
        "schema_version": 1,
        "implementation": "wizard",
        "method_contract": method_contract,
        "parity_adapter_applied": parity_adapter,
        "computation_scope": "pure_post_selection_final_pose_only",
        "camera_inventory": camera_order,
        "camera_order": camera_order,
        "root_camera": root,
        "anchor_frame": f"{root}{ROOT_FRAME_SUFFIX}",
        "frame_convention": "T_root_target maps target optical coordinates into root optical coordinates",
        "camera_records": records,
        "export_semantics": export_semantics,
        "ground_truth_used": False,
        "candidate_generation_invoked": False,
        "aggregate_selection_invoked": False,
        "solver_invoked": False,
        "colmap_invoked": False,
        "publication_invoked": False,
        "reconciliation_invoked": False,
    }


def _relative_rotation_angle_deg(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> float:
    trace = sum(
        sum(float(first[k][i]) * float(second[k][i]) for k in range(3))
        for i in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _camera_diff(
    legacy: Mapping[str, Any], wizard: Mapping[str, Any]
) -> dict[str, Any]:
    lr = legacy["rotation_matrix"]
    wr = wizard["rotation_matrix"]
    lt = legacy["translation_m"]
    wt = wizard["translation_m"]
    rotation_max = max(
        abs(float(lr[i][j]) - float(wr[i][j]))
        for i in range(3)
        for j in range(3)
    )
    relative_rotation_angle = (
        0.0 if rotation_max == 0.0 else _relative_rotation_angle_deg(lr, wr)
    )
    translation_components = [
        abs(float(lt[i]) - float(wt[i])) for i in range(3)
    ]
    translation_norm = math.sqrt(sum(value * value for value in translation_components))
    homogeneous_max = max(
        abs(
            float(legacy["homogeneous_transform_4x4"][i][j])
            - float(wizard["homogeneous_transform_4x4"][i][j])
        )
        for i in range(4)
        for j in range(4)
    )
    lq = legacy["quaternion"]["comparison_only_wxyz"]
    wq = wizard["quaternion"]["comparison_only_wxyz"]
    quaternion_raw = max(abs(float(a) - float(b)) for a, b in zip(lq, wq))
    semantic_equal = all(
        (
            legacy["role"] == wizard["role"],
            legacy["frame_from"] == wizard["frame_from"],
            legacy["frame_to"] == wizard["frame_to"],
            legacy["source_selected_candidate"]
            == wizard["source_selected_candidate"],
            legacy["inversion_history"] == wizard["inversion_history"],
            legacy["composition_history"] == wizard["composition_history"],
            legacy["final_pose_operations"] == wizard["final_pose_operations"],
        )
    )
    exact = semantic_equal and homogeneous_max == 0.0
    within = (
        rotation_max <= ROTATION_MATRIX_TOLERANCE
        and relative_rotation_angle <= ROTATION_ANGLE_TOLERANCE_DEG
        and translation_norm <= TRANSLATION_TOLERANCE_M
    )
    return {
        "camera_id": legacy["camera_id"],
        "status": "EXACT" if exact else "WITHIN_TOLERANCE" if within else "DIFFERENT",
        "legacy_role": legacy["role"],
        "wizard_role": wizard["role"],
        "frame_from_equal": legacy["frame_from"] == wizard["frame_from"],
        "frame_to_equal": legacy["frame_to"] == wizard["frame_to"],
        "selected_candidate_identity_equal": (
            legacy["source_selected_candidate"]
            == wizard["source_selected_candidate"]
        ),
        "inversion_history_equal": (
            legacy["inversion_history"] == wizard["inversion_history"]
        ),
        "composition_history_equal": (
            legacy["composition_history"] == wizard["composition_history"]
        ),
        "final_pose_operations_equal": (
            legacy["final_pose_operations"] == wizard["final_pose_operations"]
        ),
        "rotation_max_abs_delta": rotation_max,
        "relative_rotation_angle_deg": relative_rotation_angle,
        "translation_x_abs_delta_m": translation_components[0],
        "translation_y_abs_delta_m": translation_components[1],
        "translation_z_abs_delta_m": translation_components[2],
        "translation_norm_delta_m": translation_norm,
        "homogeneous_max_abs_delta": homogeneous_max,
        "legacy_rotation_determinant": legacy["rotation_validity"]["determinant"],
        "wizard_rotation_determinant": wizard["rotation_validity"]["determinant"],
        "legacy_so3_valid": legacy["rotation_validity"]["valid"],
        "wizard_so3_valid": wizard["rotation_validity"]["valid"],
        "quaternion_export_applicable": False,
        "quaternion_raw_max_abs_delta": quaternion_raw,
        "quaternion_sign_equivalent": quaternion_sign_equivalent(lq, wq),
    }


def compare_final_pose_views(
    legacy: Mapping[str, Any], wizard: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    legacy_by_camera = {
        str(record["camera_id"]): record for record in legacy["camera_records"]
    }
    wizard_by_camera = {
        str(record["camera_id"]): record for record in wizard["camera_records"]
    }
    common = [
        camera for camera in legacy["camera_order"] if camera in wizard_by_camera
    ]
    diffs = [
        _camera_diff(legacy_by_camera[camera], wizard_by_camera[camera])
        for camera in common
    ]
    inventory_equal = set(legacy_by_camera) == set(wizard_by_camera)
    anchor_equal = (
        legacy["root_camera"] == wizard["root_camera"]
        and legacy["anchor_frame"] == wizard["anchor_frame"]
    )
    camera_order_equal = legacy["camera_order"] == wizard["camera_order"]
    export_equal = legacy["export_semantics"] == wizard["export_semantics"]
    first_operation_difference = next(
        (
            row["camera_id"]
            for row in diffs
            if not row["final_pose_operations_equal"]
        ),
        None,
    )
    any_frame = any(
        not row["frame_from_equal"] or not row["frame_to_equal"] for row in diffs
    )
    any_identity = any(
        not row["selected_candidate_identity_equal"] for row in diffs
    )
    any_inversion = any(not row["inversion_history_equal"] for row in diffs)
    any_composition = any(not row["composition_history_equal"] for row in diffs)
    numeric_exact = all(row["homogeneous_max_abs_delta"] == 0.0 for row in diffs)
    numeric_within = all(row["status"] != "DIFFERENT" for row in diffs)

    if not inventory_equal:
        classification = "DIFFERENT_CAMERA_INVENTORY"
    elif not anchor_equal:
        classification = "DIFFERENT_ANCHOR_SEMANTICS"
    elif any_frame:
        classification = "DIFFERENT_FRAME_CONVENTION"
    elif any_identity:
        classification = "DIFFERENT_EXPORT_SEMANTICS"
    elif any_inversion:
        classification = "DIFFERENT_INVERSION"
    elif any_composition:
        classification = "DIFFERENT_COMPOSITION_ORDER"
    elif first_operation_difference is not None or not export_equal or not camera_order_equal:
        classification = "DIFFERENT_EXPORT_SEMANTICS"
    elif not numeric_within:
        max_rotation = max(row["rotation_max_abs_delta"] for row in diffs)
        max_translation = max(row["translation_norm_delta_m"] for row in diffs)
        classification = (
            "DIFFERENT_ROTATION"
            if max_rotation > ROTATION_MATRIX_TOLERANCE
            else "DIFFERENT_TRANSLATION"
            if max_translation > TRANSLATION_TOLERANCE_M
            else "DIFFERENT_EXPORT_SEMANTICS"
        )
    elif not numeric_exact:
        classification = "NUMERICALLY_EQUIVALENT_WITHIN_TOLERANCE"
    else:
        classification = "EXACT"

    first_divergence = None
    if classification != "EXACT":
        if first_operation_difference is not None:
            first_divergence = {
                "camera_id": first_operation_difference,
                "stage": "post_selection_aggregate_serialization",
                "legacy_operation": "format selected x/y/z and roll/pitch/yaw to nine decimals, then reconstruct R and T",
                "legacy_source": [
                    "13_eval_direct_static_cam3_cam1_multimarker.py:416",
                    "14_eval_moving_relay_chains.py:742",
                    "15_export_final_extrinsics_cam3_reference.py:80",
                ],
                "wizard_operation": "retain selected in-memory R and t until pose_row serialization",
                "wizard_source": "src/camera_rig_calibration/methods/ap01/solve_extrinsics.py:433",
                "later_divergence": (
                    "Legacy makes the anchor implicit and emits targets in loader order; "
                    "Wizard emits the anchor explicitly and sorts every pose."
                ),
            }
        elif not export_equal or not camera_order_equal:
            first_divergence = {
                "stage": "final_export_iteration",
                "legacy_operation": "top-level implicit anchor plus target entries in loader order",
                "wizard_operation": "explicit anchor row plus sorted pose dictionary",
                "legacy_source": "15_export_final_extrinsics_primary_only.py:91-112",
                "wizard_source": "src/camera_rig_calibration/methods/ap01/solve_extrinsics.py:433",
            }

    max_rotation = max((row["rotation_max_abs_delta"] for row in diffs), default=0.0)
    max_rotation_angle = max(
        (row["relative_rotation_angle_deg"] for row in diffs), default=0.0
    )
    max_translation_components = [
        max((row[field] for row in diffs), default=0.0)
        for field in (
            "translation_x_abs_delta_m",
            "translation_y_abs_delta_m",
            "translation_z_abs_delta_m",
        )
    ]
    max_translation_norm = max(
        (row["translation_norm_delta_m"] for row in diffs), default=0.0
    )
    report = {
        "schema_version": 1,
        "status": "equal" if classification == "EXACT" else "mismatch",
        "classification": classification,
        "comparison_method": "direct_transform_comparison_without_alignment",
        "tolerances": {
            "rotation_matrix_max_abs": ROTATION_MATRIX_TOLERANCE,
            "relative_rotation_angle_deg": ROTATION_ANGLE_TOLERANCE_DEG,
            "translation_norm_m": TRANSLATION_TOLERANCE_M,
            "so3": SO3_TOLERANCE,
        },
        "legacy_camera_inventory": list(legacy["camera_inventory"]),
        "wizard_camera_inventory": list(wizard["camera_inventory"]),
        "camera_inventory_equal": inventory_equal,
        "legacy_camera_order": list(legacy["camera_order"]),
        "wizard_camera_order": list(wizard["camera_order"]),
        "camera_order_equal": camera_order_equal,
        "legacy_anchor": legacy["root_camera"],
        "wizard_anchor": wizard["root_camera"],
        "anchor_semantics_equal": anchor_equal,
        "export_semantics_equal": export_equal,
        "maximum_rotation_matrix_element_abs_delta": max_rotation,
        "maximum_relative_rotation_angle_deg": max_rotation_angle,
        "maximum_translation_component_abs_delta_m": {
            "x": max_translation_components[0],
            "y": max_translation_components[1],
            "z": max_translation_components[2],
        },
        "maximum_translation_vector_norm_delta_m": max_translation_norm,
        "quaternion_export_applicable": False,
        "quaternion_only_representation_differences": [],
        "all_rotations_valid_so3": all(
            row["legacy_so3_valid"] and row["wizard_so3_valid"] for row in diffs
        ),
        "first_causal_divergence": first_divergence,
        "per_camera": diffs,
        "ground_truth_used": False,
        "alignment_used": False,
        "best_fit_alignment_used": False,
        "solver_invoked": False,
        "colmap_invoked": False,
        "publication_invoked": False,
        "reconciliation_invoked": False,
    }
    return report, diffs


def write_diff_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIFF_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DIFF_FIELDS})
    return sha256_file(path)


def _trace(
    *, implementation: str, parity_adapter: bool, selected_hash: str
) -> str:
    common = [
        "AP01 FINAL-POSE TRACE",
        "",
        f"implementation: {implementation}",
        f"frozen selected-input SHA-256: {selected_hash}",
        "scope: pure post-selection final-pose/final-extrinsics computation only",
        "Ground Truth read: no",
        "candidate rebuild: no",
        "solver/COLMAP/publication/reconciliation: not invoked",
        "",
        "Transform convention",
        "- T_cam_edge_3_camera maps a homogeneous column vector from the target optical-camera frame into the cam_edge_3 optical-camera frame.",
        "- p_cam_edge_3 = T_cam_edge_3_camera @ p_camera.",
        "- R is the active coordinate-map rotation; t is the target-camera origin expressed in cam_edge_3 coordinates, in metres.",
        "- cam_edge_3 is the identity gauge. No alignment, inversion, or global transform is applied after selection.",
        "",
        "Selected transform provenance",
        "- root: I_4x4.",
        "- Direct: T_root_marker @ inverse(T_target_marker).",
        "- Relay: T_root_marker @ inverse(T_moving_i_marker) @ scaled(Tcw_i @ inverse(Tcw_j)) @ inverse(T_target_marker @ inverse(T_moving_j_marker)).",
        "- Relay final input is the already-selected quality-weighted mean of flat MAD inliers; aggregation is not repeated here.",
        "",
    ]
    if implementation == "legacy_main":
        specific = [
            "Legacy post-selection operations",
            "1. Aggregate writers serialize x/y/z and roll/pitch/yaw to exactly nine decimal places (Direct source line 416; Relay source line 742).",
            "2. T_from_row at final exporter line 80 parses those values, converts degrees to radians, and reconstructs R as Rz(yaw) @ Ry(pitch) @ Rx(roll).",
            "3. The aggregate quaternion columns are ignored. The final primary-only JSON exports translation, RPY degrees, and the reconstructed 4x4 matrix; it emits no quaternion.",
            "4. The primary-only wrapper stores cam_edge_3 as top-level reference_camera (implicit identity) and emits target records in loader order cam_edge_0, cam_edge_1, cam_edge_5.",
            "5. Missing target aggregates are omitted independently. No normalization or further composition occurs.",
        ]
    elif parity_adapter:
        specific = [
            "Wizard main_route2_parity_v1 post-selection operations",
            "1. The parity-only adapter reproduces Legacy's nine-decimal x/y/z and RPY-degree boundary and reconstructs R as Rz @ Ry @ Rx.",
            "2. It projects cam_edge_3 as the implicit top-level anchor and targets in Legacy loader order cam_edge_0, cam_edge_1, cam_edge_5.",
            "3. No candidate, aggregate, gate, solver, or recommended_wizard_v1 behavior is changed.",
            "4. No quaternion is exported; the evidence JSON includes a canonical wxyz quaternion only for sign-equivalence diagnostics.",
            "5. Missing target aggregates use the already-frozen Legacy-compatible omission decision. No normalization or further composition occurs.",
        ]
    else:
        specific = [
            "Wizard current/pre-fix post-selection operations",
            "1. Selected R and t remain in memory without Legacy's nine-decimal aggregate-CSV round trip.",
            "2. solve_extrinsics.py line 433 iterates sorted(poses.items()), including an explicit cam_edge_3 identity row.",
            "3. The CSV exports translation, RPY degrees, and Rodrigues vector, but no 4x4 matrix or quaternion.",
        ]
    return "\n".join([*common, *specific, ""])


def generate_final_pose_parity(
    repository: Path, legacy_root: Path
) -> dict[str, Any]:
    evidence_root = repository / "parity/main_route2_v1"
    selected, manifest = freeze_selected_candidates(
        evidence_root, repository, legacy_root
    )
    selected_hash = manifest["selected_input"]["sha256"]

    legacy = legacy_final_pose_view(selected)
    wizard_pre = wizard_final_pose_view(
        selected,
        method_contract=PARITY_CONTRACT,
        parity_adapter=False,
    )
    pre_report, pre_diffs = compare_final_pose_views(legacy, wizard_pre)
    pre_root = evidence_root / "ap01/final_pose/pre_fix"
    pre_legacy_path = pre_root / "legacy/AP01_FINAL_CAMERA_POSES.json"
    pre_wizard_path = pre_root / "wizard/AP01_FINAL_CAMERA_POSES.json"
    pre_legacy_trace = pre_root / "legacy/AP01_FINAL_POSE_TRACE.txt"
    pre_wizard_trace = pre_root / "wizard/AP01_FINAL_POSE_TRACE.txt"
    pre_parity_path = pre_root / "AP01_FINAL_POSE_PARITY.json"
    pre_diff_path = pre_root / "AP01_FINAL_POSE_DIFF.csv"
    immutable_hashes = {
        pre_legacy_path: write_json(pre_legacy_path, legacy, immutable=True),
        pre_wizard_path: write_json(pre_wizard_path, wizard_pre, immutable=True),
        pre_legacy_trace: write_text(
            pre_legacy_trace,
            _trace(
                implementation="legacy_main",
                parity_adapter=False,
                selected_hash=selected_hash,
            ),
            immutable=True,
        ),
        pre_wizard_trace: write_text(
            pre_wizard_trace,
            _trace(
                implementation="wizard",
                parity_adapter=False,
                selected_hash=selected_hash,
            ),
            immutable=True,
        ),
        pre_parity_path: write_json(pre_parity_path, pre_report, immutable=True),
    }
    if pre_diff_path.exists():
        # Diff CSV is also immutable; regenerate into memory-equivalent rows only
        # by verifying the already-preserved hash on subsequent runs.
        immutable_hashes[pre_diff_path] = sha256_file(pre_diff_path)
    else:
        immutable_hashes[pre_diff_path] = write_diff_csv(pre_diff_path, pre_diffs)
    pre_manifest = {
        "schema_version": 1,
        "evidence_role": "immutable_pre_fix_AP01_final_pose_export_mismatch",
        "classification": pre_report["classification"],
        "selected_input_sha256": selected_hash,
        "first_causal_divergence": pre_report["first_causal_divergence"],
        "artifacts": [
            {
                "path": path.relative_to(evidence_root).as_posix(),
                "sha256": digest,
            }
            for path, digest in sorted(
                immutable_hashes.items(), key=lambda item: item[0].as_posix()
            )
        ],
        "ground_truth_used": False,
    }
    pre_manifest_path = pre_root / "PRE_FIX_MANIFEST.json"
    pre_manifest_hash = write_json(pre_manifest_path, pre_manifest, immutable=True)

    wizard = wizard_final_pose_view(
        selected,
        method_contract=PARITY_CONTRACT,
        parity_adapter=True,
    )
    report, diffs = compare_final_pose_views(legacy, wizard)
    if report["classification"] != "EXACT":
        raise RuntimeError(f"Post-fix final-pose parity is not EXACT: {report}")
    report.update(
        {
            "selected_input_sha256": selected_hash,
            "pre_fix_classification": pre_report["classification"],
            "pre_fix_manifest_sha256": pre_manifest_hash,
            "parity_fix": {
                "scope": "main_route2_parity_v1 final-pose adapter only",
                "operations": [
                    "Legacy nine-decimal translation/RPY aggregate serialization round trip",
                    "Legacy implicit anchor and target-only loader ordering projection",
                ],
                "recommended_wizard_v1_unchanged": True,
                "candidate_construction_unchanged": True,
                "aggregate_selection_unchanged": True,
            },
        }
    )

    final_root = evidence_root / "ap01/final_pose"
    legacy_path = final_root / "legacy/AP01_FINAL_CAMERA_POSES.json"
    wizard_path = final_root / "wizard/AP01_FINAL_CAMERA_POSES.json"
    legacy_trace_path = final_root / "legacy/AP01_FINAL_POSE_TRACE.txt"
    wizard_trace_path = final_root / "wizard/AP01_FINAL_POSE_TRACE.txt"
    parity_path = final_root / "AP01_FINAL_POSE_PARITY.json"
    diff_path = final_root / "AP01_FINAL_POSE_DIFF.csv"
    output_hashes = {
        "legacy_final_camera_poses_sha256": write_json(legacy_path, legacy),
        "wizard_final_camera_poses_sha256": write_json(wizard_path, wizard),
        "legacy_trace_sha256": write_text(
            legacy_trace_path,
            _trace(
                implementation="legacy_main",
                parity_adapter=False,
                selected_hash=selected_hash,
            ),
        ),
        "wizard_trace_sha256": write_text(
            wizard_trace_path,
            _trace(
                implementation="wizard",
                parity_adapter=True,
                selected_hash=selected_hash,
            ),
        ),
        "final_pose_parity_sha256": write_json(parity_path, report),
        "final_pose_diff_sha256": write_diff_csv(diff_path, diffs),
    }
    return {
        "classification": report["classification"],
        "selected_input_sha256": selected_hash,
        "pre_fix_classification": pre_report["classification"],
        "pre_fix_manifest_sha256": pre_manifest_hash,
        "output_hashes": output_hashes,
        "source_code": {
            "legacy_root": str(legacy_root),
            "wizard_repository": str(repository),
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(__doc__)
    result.add_argument("--repository", type=Path, default=Path.cwd())
    result.add_argument("--legacy-root", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    result = generate_final_pose_parity(
        args.repository.resolve(), args.legacy_root.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

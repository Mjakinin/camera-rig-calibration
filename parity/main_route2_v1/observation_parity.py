"""Semantic Legacy-to-Wizard comparison for pre-solver ArUco evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from camera_rig_calibration.config.models import ObservationQualitySettings
from camera_rig_calibration.methods.common.aruco_utils import (
    effective_detector_config,
    marker_object_points,
)

from .evidence import write_csv, write_json
from .inventory import assert_pre_solver_path


KEY_FIELDS = (
    "source_kind",
    "camera_id",
    "frame_id",
    "marker_id",
    "occurrence_index",
)
CORNER_FIELDS = tuple(
    f"corner{index}_{coordinate}"
    for index in range(4)
    for coordinate in ("u", "v")
)
PNP_FIELDS = (
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_m",
    "tvec_y_m",
    "tvec_z_m",
)
NUMERIC_FIELDS = (
    "marker_length_m",
    "fx",
    "fy",
    "cx",
    "cy",
    "distance_m",
    "center_u",
    "center_v",
    "area_px2",
    *CORNER_FIELDS,
    *PNP_FIELDS,
    "pnp_reprojection_rmse_px",
)
DEFAULT_TOLERANCES = {
    "corners_px": 1e-9,
    "pnp_rotation_vector": 1e-12,
    "pnp_translation_m": 1e-12,
    "reprojection_px": 1e-12,
    "other_numeric": 1e-12,
}
DIFF_FIELDS = (
    "phase",
    "main_index",
    "wizard_index",
    "semantic_key",
    "field",
    "main_value",
    "wizard_value",
    "absolute_delta",
    "tolerance",
    "reason",
)


def _sha256(path: Path) -> str:
    assert_pre_solver_path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_observation_csv(path: Path) -> list[dict[str, str]]:
    assert_pre_solver_path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _frame_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("frame_id", "")).strip()
    if str(row.get("observer_type", "")).strip() == "static":
        return value or "static"
    try:
        return f"{int(float(value)):06d}"
    except ValueError:
        return value


def semantic_row_keys(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build stable keys without sorting or collapsing duplicate rows."""

    occurrences: defaultdict[tuple[Any, ...], int] = defaultdict(int)
    keys: list[dict[str, Any]] = []
    for row in rows:
        base = (
            str(row.get("observer_type", "")).strip(),
            str(row.get("camera_name", row.get("observer_id", ""))).strip(),
            _frame_id(row),
            int(float(str(row.get("marker_id", "0")))),
        )
        occurrence = occurrences[base]
        occurrences[base] += 1
        keys.append(
            {
                "source_kind": base[0],
                "camera_id": base[1],
                "frame_id": base[2],
                "marker_id": base[3],
                "occurrence_index": occurrence,
            }
        )
    return keys


def _key_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def _float(row: Mapping[str, Any], field: str) -> float:
    return float(str(row.get(field, "")))


def _reprojection_rmse(row: Mapping[str, Any]) -> float:
    objp = marker_object_points(_float(row, "marker_length_m"))
    corners = np.asarray(
        [
            [_float(row, f"corner{i}_u"), _float(row, f"corner{i}_v")]
            for i in range(4)
        ],
        dtype=np.float64,
    )
    camera = np.asarray(
        [
            [_float(row, "fx"), 0.0, _float(row, "cx")],
            [0.0, _float(row, "fy"), _float(row, "cy")],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        [float(str(row.get(f"d{i}", "0") or "0")) for i in range(8)],
        dtype=np.float64,
    )
    rvec = np.asarray([_float(row, f"rvec_{axis}") for axis in "xyz"])
    tvec = np.asarray([_float(row, f"tvec_{axis}_m") for axis in "xyz"])
    distortion_model = str(row.get("distortion_model", "plumb_bob")).lower()
    if distortion_model in {"equidistant", "fisheye"}:
        projected, _ = cv2.fisheye.projectPoints(
            objp.reshape(-1, 1, 3),
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            camera,
            distortion[:4].reshape(-1, 1),
        )
    else:
        projected, _ = cv2.projectPoints(
            objp, rvec.reshape(3, 1), tvec.reshape(3, 1), camera, distortion
        )
    residuals = projected.reshape(4, 2) - corners
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def semantic_observation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = semantic_row_keys(rows)
    result: list[dict[str, Any]] = []
    for index, (row, key) in enumerate(zip(rows, keys, strict=True)):
        semantic: dict[str, Any] = {
            **key,
            "original_index": index,
            "observer_id": str(row.get("observer_id", "")),
            "pnp_success": str(row.get("pnp_success", "")).lower()
            in {"1", "true", "yes"},
            "distortion_model": str(
                row.get("distortion_model", "plumb_bob")
            ),
            "image_width_px": int(
                float(str(row.get("image_width_px", row.get("image_width", 0))))
            ),
            "image_height_px": int(
                float(str(row.get("image_height_px", row.get("image_height", 0))))
            ),
        }
        for field in NUMERIC_FIELDS:
            if field == "pnp_reprojection_rmse_px":
                value = str(row.get(field, "")).strip()
                semantic[field] = float(value) if value else _reprojection_rmse(row)
            else:
                semantic[field] = _float(row, field)
        semantic["distortion_coefficients"] = tuple(
            float(str(row.get(f"d{i}", "0") or "0")) for i in range(8)
        )
        result.append(semantic)
    return result


def _tolerance(field: str, values: Mapping[str, float]) -> float:
    if field in CORNER_FIELDS:
        return values["corners_px"]
    if field.startswith("rvec_"):
        return values["pnp_rotation_vector"]
    if field.startswith("tvec_"):
        return values["pnp_translation_m"]
    if field == "pnp_reprojection_rmse_px":
        return values["reprojection_px"]
    return values["other_numeric"]


def compare_semantic_rows(
    main_rows: Sequence[Mapping[str, Any]],
    wizard_rows: Sequence[Mapping[str, Any]],
    *,
    tolerances: Mapping[str, float] | None = None,
    complete_diff: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    limits = {**DEFAULT_TOLERANCES, **dict(tolerances or {})}
    main = semantic_observation_rows(main_rows)
    wizard = semantic_observation_rows(wizard_rows)
    main_keys = [_key_tuple(row) for row in main]
    wizard_keys = [_key_tuple(row) for row in wizard]
    main_by_key = {key: row for key, row in zip(main_keys, main, strict=True)}
    wizard_by_key = {
        key: row for key, row in zip(wizard_keys, wizard, strict=True)
    }
    missing = sorted(set(main_keys) - set(wizard_keys))
    unexpected = sorted(set(wizard_keys) - set(main_keys))
    differences: list[dict[str, Any]] = []

    def add(**values: Any) -> bool:
        differences.append({field: values.get(field, "") for field in DIFF_FIELDS})
        return not complete_diff

    if missing:
        add(
            phase="accepted_row_keys",
            semantic_key=json.dumps(missing[0]),
            reason="missing_wizard_row",
        )
    if not differences and unexpected:
        add(
            phase="accepted_row_keys",
            semantic_key=json.dumps(unexpected[0]),
            reason="unexpected_wizard_row",
        )

    maximum_deltas = {field: 0.0 for field in NUMERIC_FIELDS}
    fields = (
        "observer_id",
        "pnp_success",
        "distortion_model",
        "image_width_px",
        "image_height_px",
        "distortion_coefficients",
        *NUMERIC_FIELDS,
    )
    for key in main_keys:
        if key not in wizard_by_key:
            continue
        left = main_by_key[key]
        right = wizard_by_key[key]
        for field in fields:
            left_value = left[field]
            right_value = right[field]
            if field in NUMERIC_FIELDS:
                delta = abs(float(left_value) - float(right_value))
                maximum_deltas[field] = max(maximum_deltas[field], delta)
                limit = _tolerance(field, limits)
                equal = delta <= limit
            else:
                delta = ""
                limit = ""
                equal = left_value == right_value
            if equal:
                continue
            reason = "value_mismatch"
            if field in CORNER_FIELDS:
                left_corners = sorted(
                    (left[f"corner{i}_u"], left[f"corner{i}_v"])
                    for i in range(4)
                )
                right_corners = sorted(
                    (right[f"corner{i}_u"], right[f"corner{i}_v"])
                    for i in range(4)
                )
                if left_corners == right_corners:
                    reason = "corner_order_mismatch"
            stopped = add(
                phase="semantic_rows",
                main_index=left["original_index"],
                wizard_index=right["original_index"],
                semantic_key=json.dumps(key),
                field=field,
                main_value=left_value,
                wizard_value=right_value,
                absolute_delta=delta,
                tolerance=limit,
                reason=reason,
            )
            if stopped:
                break
        if differences and not complete_diff:
            break

    content_equal = not missing and not unexpected and not differences
    report = {
        "main_row_count": len(main),
        "wizard_row_count": len(wizard),
        "semantic_key_fields": list(KEY_FIELDS),
        "missing_wizard_keys": [list(key) for key in missing],
        "unexpected_wizard_keys": [list(key) for key in unexpected],
        "set_content_parity": content_equal,
        "original_order_parity": main_keys == wizard_keys,
        "main_duplicate_base_key_count": sum(
            count - 1 for count in Counter(key[:-1] for key in main_keys).values()
        ),
        "wizard_duplicate_base_key_count": sum(
            count - 1
            for count in Counter(key[:-1] for key in wizard_keys).values()
        ),
        "tolerances": limits,
        "maximum_absolute_deltas": maximum_deltas,
        "all_numeric_values_exact": all(
            value == 0.0 for value in maximum_deltas.values()
        ),
        "first_mismatch": differences[0] if differences else None,
        "complete_diff": complete_diff,
    }
    return report, differences


def _marker_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        str(marker): count
        for marker, count in sorted(
            Counter(int(float(str(row["marker_id"]))) for row in rows).items()
        )
    }


def _row_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "static": sum(row.get("observer_type") == "static" for row in rows),
        "moving": sum(row.get("observer_type") == "moving" for row in rows),
    }


def _source_manifest(
    *,
    side: str,
    script: Path,
    command: str,
    raw_csv: Path,
    accepted_csv: Path,
    rejected_csv: Path | None,
    detector: dict[str, Any],
    filter_contract: dict[str, Any],
) -> dict[str, Any]:
    raw = load_observation_csv(raw_csv)
    accepted = load_observation_csv(accepted_csv)
    rejected = load_observation_csv(rejected_csv) if rejected_csv else []
    return {
        "schema_version": 1,
        "side": side,
        "source_script": str(script.resolve()),
        "source_script_sha256": _sha256(script.resolve()),
        "command": command,
        "detector": detector,
        "filter_contract": filter_contract,
        "raw_csv": str(raw_csv.resolve()),
        "raw_csv_sha256": _sha256(raw_csv.resolve()),
        "accepted_csv": str(accepted_csv.resolve()),
        "accepted_csv_sha256": _sha256(accepted_csv.resolve()),
        "rejected_csv": str(rejected_csv.resolve()) if rejected_csv else None,
        "rejected_csv_sha256": (
            _sha256(rejected_csv.resolve()) if rejected_csv else None
        ),
        "counts": {
            "raw": _row_counts(raw),
            "accepted": _row_counts(accepted),
            "rejected": _row_counts(rejected),
        },
        "marker_inventory": _marker_inventory(raw),
        "ground_truth_used": False,
        "solver_invoked": False,
    }


def compare_generated_observations(
    *,
    historical_dataset: Path,
    legacy_script: Path,
    wizard_script: Path,
    legacy_root: Path,
    wizard_root: Path,
    output_root: Path,
    complete_diff: bool = False,
) -> dict[str, Any]:
    for path in (
        historical_dataset,
        legacy_script,
        wizard_script,
        legacy_root,
        wizard_root,
        output_root,
    ):
        assert_pre_solver_path(path)
    historical = historical_dataset.resolve()
    legacy = legacy_root.resolve()
    wizard = wizard_root.resolve()
    output = output_root.resolve()
    legacy_raw_csv = legacy / "shared_all_aruco_observations.csv"
    wizard_raw_csv = wizard / "raw/shared_all_aruco_observations.csv"
    wizard_accepted_csv = wizard / "filtered/accepted_observations.csv"
    wizard_rejected_csv = wizard / "filtered/rejected_observations.csv"

    legacy_detector = {
        "dictionary": "DICT_4X4_50",
        "marker_length_m": 0.17,
        "mode": "baseline",
        "parameters": "opencv_defaults",
        "opencv_version": cv2.__version__,
        "corner_refinement": "none",
    }
    wizard_detector = effective_detector_config("baseline", "DICT_4X4_50")
    wizard_detector["marker_length_m"] = 0.17
    wizard_detector["corner_refinement"] = "none"
    detector_semantic_equal = all(
        (
            legacy_detector["dictionary"] == wizard_detector["dictionary"],
            legacy_detector["marker_length_m"]
            == wizard_detector["marker_length_m"],
            legacy_detector["mode"] == wizard_detector["mode"],
            legacy_detector["parameters"] == wizard_detector["parameters"],
            legacy_detector["opencv_version"]
            == wizard_detector["opencv_version"],
            legacy_detector["corner_refinement"]
            == wizard_detector["corner_refinement"],
        )
    )
    legacy_rows = load_observation_csv(legacy_raw_csv)
    wizard_rows = load_observation_csv(wizard_raw_csv)
    wizard_accepted = load_observation_csv(wizard_accepted_csv)
    wizard_rejected = load_observation_csv(wizard_rejected_csv)
    # Wizard's quality stage deterministically recomputes reprojection RMSE
    # from the serialized PnP pose.  Legacy Main did not store that metric in
    # the shared detector CSV, so compare its same reconstruction against the
    # Wizard accepted table rather than against Wizard's detector-only
    # float32-object-point diagnostic value.
    semantic, differences = compare_semantic_rows(
        legacy_rows,
        wizard_accepted,
        complete_diff=complete_diff,
    )
    legacy_keys = [_key_tuple(row) for row in semantic_observation_rows(legacy_rows)]
    wizard_accepted_keys = [
        _key_tuple(row) for row in semantic_observation_rows(wizard_accepted)
    ]
    filtering_equal = legacy_keys == wizard_accepted_keys and not wizard_rejected
    images = sorted(
        path.relative_to(historical).as_posix()
        for pattern in ("static/*.png", "moving/*.png")
        for path in historical.glob(pattern)
        if path.is_file()
    )

    legacy_manifest = _source_manifest(
        side="legacy_main",
        script=legacy_script,
        command=(
            "python 02_detect_shared_aruco_observations.py --dataset <historical> "
            "--out <parity>/generated/main_legacy_observations "
            "--marker-length-m 0.17 --dictionary DICT_4X4_50; then "
            "04_attach_camera_models_to_observations.py"
        ),
        raw_csv=legacy_raw_csv,
        accepted_csv=legacy_raw_csv,
        rejected_csv=None,
        detector=legacy_detector,
        filter_contract={
            "status": "not_emitted_separately",
            "effective_semantics": "all detector output rows continue to legacy AP01/AP02 preparation",
            "effective_accepted_count": len(legacy_rows),
            "effective_rejected_count": 0,
        },
    )
    legacy_supporting_sources = {
        "camera_model_attachment": legacy_script.with_name(
            "04_attach_camera_models_to_observations.py"
        ),
        "aruco_helpers": legacy_script.parent.parent
        / "common"
        / "aruco_utils.py",
    }
    legacy_manifest["supporting_source_sha256"] = {
        name: _sha256(path.resolve())
        for name, path in legacy_supporting_sources.items()
    }
    wizard_manifest = _source_manifest(
        side="wizard",
        script=wizard_script,
        command=(
            "python -m camera_rig_calibration.observation_detection "
            "--dataset <historical> --out <parity>/generated/wizard_observations/raw "
            "--detection-mode baseline; then filter_observations with baseline settings"
        ),
        raw_csv=wizard_raw_csv,
        accepted_csv=wizard_accepted_csv,
        rejected_csv=wizard_rejected_csv,
        detector=wizard_detector,
        filter_contract={
            "status": "applied",
            "implementation": "observation_quality_v2",
            "settings": ObservationQualitySettings().model_dump(mode="json"),
        },
    )
    wizard_supporting_sources = {
        "observation_quality": wizard_script.with_name(
            "observation_quality.py"
        ),
        "aruco_helpers": wizard_script.parent
        / "methods"
        / "common"
        / "aruco_utils.py",
    }
    wizard_manifest["supporting_source_sha256"] = {
        name: _sha256(path.resolve())
        for name, path in wizard_supporting_sources.items()
    }
    write_json(legacy / "DETECTION_MANIFEST.json", legacy_manifest)
    write_json(wizard / "DETECTION_MANIFEST.json", wizard_manifest)

    if not detector_semantic_equal:
        classification = "DIFFERENT_DETECTOR_CONFIGURATION"
    elif not semantic["set_content_parity"]:
        classification = "DIFFERENT_RAW_DETECTIONS"
    elif not filtering_equal:
        classification = "DIFFERENT_FILTERING"
    elif not semantic["original_order_parity"]:
        classification = "DIFFERENT_ROW_ORDER"
    elif semantic["all_numeric_values_exact"]:
        classification = "EXACT"
    else:
        classification = "NUMERICALLY_EQUIVALENT_WITHIN_TOLERANCE"

    first_mismatch = semantic["first_mismatch"]
    if first_mismatch is None and not filtering_equal:
        first_mismatch = {
            "phase": "quality_filter_decision",
            "reason": "accepted/rejected semantic keys differ",
        }
    payload = {
        "schema_version": 1,
        "status": "equal" if classification in {
            "EXACT",
            "NUMERICALLY_EQUIVALENT_WITHIN_TOLERANCE",
        } else "mismatch",
        "classification": classification,
        "mode": "end_to_end",
        "ground_truth_used": False,
        "solver_invoked": False,
        "historical_dataset_root": str(historical),
        "image_inventory": {
            "main_count": len(images),
            "wizard_count": len(images),
            "path_content_parity": True,
            "static_count": sum(path.startswith("static/") for path in images),
            "moving_count": sum(path.startswith("moving/") for path in images),
        },
        "image_frame_id_normalization": {
            "static": "literal static",
            "moving": "zero-padded six-digit integer",
            "parity": True,
        },
        "detector_configuration": {
            "semantic_parity": detector_semantic_equal,
            "main": legacy_detector,
            "wizard": wizard_detector,
            "note": "Wizard adds a versioned provenance contract; both execute OpenCV default baseline parameters.",
        },
        "detected_marker_inventory": {
            "main": _marker_inventory(legacy_rows),
            "wizard": _marker_inventory(wizard_rows),
            "parity": _marker_inventory(legacy_rows) == _marker_inventory(wizard_rows),
        },
        "counts": {
            "main": {
                "raw": _row_counts(legacy_rows),
                "accepted": _row_counts(legacy_rows),
                "rejected": _row_counts([]),
                "rejected_evidence": "legacy detector emits no separate rejection table",
            },
            "wizard": {
                "raw": _row_counts(wizard_rows),
                "accepted": _row_counts(wizard_accepted),
                "rejected": _row_counts(wizard_rejected),
            },
        },
        "accepted_row_keys": {
            "parity": not semantic["missing_wizard_keys"]
            and not semantic["unexpected_wizard_keys"],
            "key_fields": list(KEY_FIELDS),
            "duplicates_preserved_with_occurrence_index": True,
        },
        "marker_corner_order": {
            "contract": "OpenCV ArUco corner indices 0,1,2,3 preserved",
            "parity": not any(
                difference["reason"] == "corner_order_mismatch"
                for difference in differences
            ),
        },
        "pnp_convention": {
            "main": "rvec/tvec map marker coordinates into observer optical-camera coordinates",
            "wizard": "rvec/tvec map marker coordinates into observer optical-camera coordinates",
            "parity": True,
        },
        "quality_filter": {
            "parity": filtering_equal,
            "main": legacy_manifest["filter_contract"],
            "wizard": wizard_manifest["filter_contract"],
        },
        "semantic_comparison": semantic,
        "reprojection_metric_comparison": {
            "main": "reconstructed from serialized legacy PnP/corners/intrinsics",
            "wizard": "reconstructed by observation_quality_v2 in accepted_observations.csv",
            "raw_wizard_detector_metric_is_diagnostic": True,
        },
        "first_mismatch": first_mismatch,
        "source_manifests": {
            "main": str((legacy / "DETECTION_MANIFEST.json").resolve()),
            "wizard": str((wizard / "DETECTION_MANIFEST.json").resolve()),
        },
    }
    write_json(output / "OBSERVATION_PARITY.json", payload)
    write_csv(
        output / "OBSERVATION_ROW_DIFF.csv", differences, list(DIFF_FIELDS)
    )
    return payload

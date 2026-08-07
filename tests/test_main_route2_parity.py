from __future__ import annotations

import json
from pathlib import Path

import pytest

from parity.main_route2_v1.audit import validate_frame_counts
from parity.main_route2_v1.compare import (
    compare_ordered_rows,
    compare_ordered_values,
)
from parity.main_route2_v1.evidence import reserve_unavailable_artifacts
from parity.main_route2_v1.inventory import (
    build_file_inventory,
    inventory_fingerprint,
)
from parity.main_route2_v1.presets import validate_presets
from parity.main_route2_v1.transforms import compare_transforms


def _minimal_dataset(root: Path) -> Path:
    files = {
        "raw_images/static/cam_b.png": b"b",
        "raw_images/static/cam_a.png": b"a",
        "raw_images/moving/frame_0001.png": b"1",
        "raw_images/moving/frame_0000.png": b"0",
        "raw_images/camera_info/cam_a.json": b"{}",
        "metadata/simulation/route_commanded.csv": b"frame,x\n0,0\n",
        "metadata/simulation/world_snapshot.sdf": b"<sdf/>",
        "metadata/simulation/capture_metadata.json": b"{}",
        "dataset.json": b"{}",
        "observations/shared.csv": b"id\n1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_file_inventory_and_hashing_are_deterministic(tmp_path: Path) -> None:
    dataset = _minimal_dataset(tmp_path / "dataset")
    first = build_file_inventory(dataset)
    second = build_file_inventory(dataset)
    assert first == second
    assert [row["path"] for row in first] == sorted(row["path"] for row in first)
    assert inventory_fingerprint(first) == inventory_fingerprint(second)


def test_missing_evidence_is_explicit(tmp_path: Path) -> None:
    reserve_unavailable_artifacts(tmp_path, "legacy rows not generated")
    payload = json.loads((tmp_path / "AP01_CANDIDATE_PARITY.json").read_text())
    assert payload["status"] == "unavailable"
    assert payload["reason"] == "legacy rows not generated"
    assert payload["ground_truth_used"] is False
    assert (tmp_path / "AP01_POSE_PARITY.csv").read_text().splitlines()[1].startswith(
        "unavailable,"
    )


def test_ordered_row_comparison_detects_reordering() -> None:
    left = [{"id": "a"}, {"id": "b"}]
    right = [{"id": "b"}, {"id": "a"}]
    report, differences = compare_ordered_rows(left, right)
    assert report["status"] == "mismatch"
    assert report["first_mismatch"]["row_index"] == 0
    assert differences[0]["field"] == "id"


def test_first_mismatch_is_default_and_complete_diff_is_optional() -> None:
    left = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    right = [{"a": "x", "b": "y"}, {"a": "z", "b": "w"}]
    first, first_differences = compare_ordered_rows(left, right)
    complete, all_differences = compare_ordered_rows(
        left, right, continue_after_mismatch=True
    )
    assert first["stopped_at_first_mismatch"] is True
    assert len(first_differences) == 1
    assert complete["stopped_at_first_mismatch"] is False
    assert len(all_differences) == 4


def test_duplicate_rows_are_preserved_and_reported() -> None:
    rows = [{"id": "same"}, {"id": "same"}]
    report, differences = compare_ordered_rows(rows, rows)
    assert not differences
    assert report["status"] == "equal"
    assert report["main_row_count"] == 2
    assert report["main_duplicate_records"] == [
        {"record": {"id": "same"}, "count": 2}
    ]


def test_float_tolerance_is_opt_in() -> None:
    left = [{"id": "01", "value": "1.0000"}]
    right = [{"id": "1", "value": "1.0009"}]
    report, differences = compare_ordered_rows(
        left,
        right,
        float_fields={"value"},
        float_tolerance=0.001,
        continue_after_mismatch=True,
    )
    assert report["status"] == "mismatch"
    assert [difference["field"] for difference in differences] == ["id"]


def test_transform_comparison_uses_translation_and_rotation_tolerances() -> None:
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    shifted = [row[:] for row in identity]
    shifted[0][3] = 0.001
    equal = compare_transforms(
        identity,
        shifted,
        translation_tolerance_m=0.001,
        rotation_tolerance_deg=0.0,
    )
    mismatch = compare_transforms(
        identity,
        shifted,
        translation_tolerance_m=0.0009,
        rotation_tolerance_deg=0.0,
    )
    assert equal["status"] == "equal"
    assert mismatch["status"] == "mismatch"


def test_ap02_parameter_name_ordering_is_exact() -> None:
    equal = compare_ordered_values(
        [("observer", "cam_a"), ("marker", 7)],
        [("observer", "cam_a"), ("marker", 7)],
    )
    reordered = compare_ordered_values(
        [("observer", "cam_a"), ("marker", 7)],
        [("marker", 7), ("observer", "cam_a")],
    )
    assert equal["status"] == "equal"
    assert reordered["first_mismatch"]["index"] == 0


def test_ap02_residual_row_ordering_is_exact() -> None:
    main = [
        {"observer_id": "frame_0", "marker_id": "7", "corner": "0"},
        {"observer_id": "frame_0", "marker_id": "7", "corner": "1"},
    ]
    wizard = list(reversed(main))
    report, _ = compare_ordered_rows(main, wizard)
    assert report["status"] == "mismatch"
    assert report["first_mismatch"]["field"] == "corner"


def test_pre_solver_inventory_never_reads_ground_truth(tmp_path: Path) -> None:
    dataset = _minimal_dataset(tmp_path / "dataset")
    ground_truth = dataset / "metadata/simulation/ground_truth.json"
    ground_truth.write_text('{"secret": true}\n')
    read_paths: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        read_paths.append(path)
        return path.read_bytes()

    rows = build_file_inventory(dataset, read_bytes=recording_reader)
    assert rows
    assert ground_truth not in read_paths
    assert not any("ground_truth" in row["path"] for row in rows)


def test_raw_and_selected_frame_counts_remain_distinct() -> None:
    counts = {
        "raw_static_images": {"count": 4},
        "raw_moving_images": {"count": 189},
        "method_selected_moving_frames": {"count": 107},
        "graph_initialized_moving_frames": {"count": 107},
        "ba_used_moving_frames": {"count": 107},
        "colmap_registered_moving_frames": {"count": 175},
    }
    validate_frame_counts(counts)
    assert counts["raw_moving_images"]["count"] != counts[
        "method_selected_moving_frames"
    ]["count"]
    invalid = {key: dict(value) for key, value in counts.items()}
    invalid["method_selected_moving_frames"]["count"] = 190
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_frame_counts(invalid)


def test_preset_schema_keeps_fast_mode_non_parity() -> None:
    payload = {
        "schema_version": 1,
        "presets": {
            "main_route2_parity_v1": {
                "parity": True,
                "locked": True,
                "locks": {
                    "dataset_fingerprint": "sha256:fixture",
                    "root_camera": "cam_edge_3",
                    "ap02_reference_marker_id": 14,
                    "evaluation_anchor_marker_id": 14,
                    "observation_semantics": "legacy",
                    "ap01_aggregate_selection": "legacy",
                    "ap02_frame_selection": "legacy",
                    "ap02_graph_initialization": "legacy",
                    "ap02_static_max_nfev": 80,
                    "ap02_combined_max_nfev": 80,
                    "colmap_compute": "cpu",
                    "intrinsics_refinement": {},
                    "implementation_versions": {},
                },
            },
            "recommended_wizard_v1": {"parity": False},
            "fast_50x50": {"parity": False},
        },
    }
    validate_presets(payload)
    payload["presets"]["fast_50x50"]["parity"] = True
    with pytest.raises(ValueError, match="non-parity"):
        validate_presets(payload)


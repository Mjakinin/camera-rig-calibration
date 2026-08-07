from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from camera_rig_calibration.methods.ap01 import core, solve_extrinsics
from parity.main_route2_v1.ap01_candidate_parity import (
    NUMERIC_TOLERANCES,
    _best_static,
    _build_direct,
    _build_relay,
    _candidate_core_row,
    _prepared_rows,
    _root_candidate,
    _wizard_aggregate_and_select,
    _wizard_moving,
    compare_selection,
    freeze_ap01_input,
)


def _frozen_row(
    *,
    camera: str,
    marker: int,
    source_kind: str = "static",
    frame: int | None = None,
    index: int = 0,
    score: float = 0.5,
    translation: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> dict:
    frame_id = "static" if frame is None else f"{frame:06d}"
    key = f"ap01obs:{index:06d}"
    return {
        "schema_version": 1,
        "original_index": index,
        "source_kind": source_kind,
        "camera_id": camera,
        "observer_id": camera,
        "frame_id": frame_id,
        "marker_id": marker,
        "occurrence_index": 0,
        "observation_key": key,
        "image_path": "unused.png",
        "marker_length_m": 0.17,
        "corners_px": [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]],
        "pnp": {
            "success": True,
            "rotation_vector": [0.0, 0.0, 0.0],
            "translation_m": list(translation),
            "reprojection_rmse_px": 0.25,
            "convention": "marker_coordinates_to_observer_optical_camera",
        },
        "geometry": {
            "distance_m": float(np.linalg.norm(translation)),
            "center_u": 640.0,
            "center_v": 360.0,
            "area_px2": 400.0,
        },
        "camera_model": {
            "fx": 900.0,
            "fy": 900.0,
            "cx": 640.0,
            "cy": 360.0,
            "distortion_model": "plumb_bob",
            "distortion_coefficients": [0.0] * 8,
            "image_width_px": 1280,
            "image_height_px": 720,
        },
        "wizard_observation_quality": {
            "selection_score": score,
            "score_area": score,
            "score_reprojection": 1.0,
            "score_border": 1.0,
            "score_distance": 1.0,
        },
        "filter": {"decision": "accepted", "reason": "accepted"},
        "detection_metadata": {},
    }


def _candidate_fixture() -> tuple[dict, dict, dict]:
    frozen = [
        _frozen_row(camera="cam_edge_3", marker=7, score=0.81, index=0),
        _frozen_row(
            camera="cam_edge_1",
            marker=7,
            score=0.49,
            index=1,
            translation=(0.2, 0.0, 1.0),
        ),
    ]
    static_rows, _ = _prepared_rows(frozen, implementation="wizard")
    static = _best_static(static_rows)
    records, _ = _build_direct(
        implementation="wizard",
        targets=("cam_edge_1",),
        static=static,
        starting_index=0,
    )
    return frozen[0], frozen[1], records[0]


def test_direct_candidate_direction_support_order_score_and_multiplicity() -> None:
    _, _, candidate = _candidate_fixture()
    assert candidate["candidate_type"] == "direct"
    assert candidate["support_count"] == 2
    assert candidate["support_observation_keys"] == [
        "ap01obs:000000",
        "ap01obs:000001",
    ]
    assert candidate["transform_chain"]["direction"] == "target->root"
    assert candidate["translation_m"] == pytest.approx([-0.2, 0.0, 0.0])
    assert candidate["aggregate_score"] == pytest.approx(
        np.sqrt(0.81 * 0.49)
    )
    assert candidate["original_construction_index"] == 0
    assert NUMERIC_TOLERANCES["translation_m_absolute"] == 1e-12


def test_relay_candidate_chain_direction_order_support_and_score() -> None:
    frozen = [
        _frozen_row(camera="cam_edge_3", marker=1, score=0.81, index=0),
        _frozen_row(camera="cam_edge_0", marker=2, score=0.16, index=1),
        _frozen_row(
            camera="moving_calib_camera",
            marker=1,
            source_kind="moving",
            frame=1,
            score=0.25,
            index=2,
        ),
        _frozen_row(
            camera="moving_calib_camera",
            marker=2,
            source_kind="moving",
            frame=2,
            score=1.0,
            index=3,
        ),
    ]
    static_rows, moving_rows = _prepared_rows(frozen, implementation="wizard")
    static = _best_static(static_rows)
    moving = {1: [moving_rows[0]], 2: [moving_rows[1]]}
    poses = {1: np.eye(4), 2: np.eye(4)}
    records, next_index = _build_relay(
        implementation="wizard",
        targets=("cam_edge_0",),
        static=static,
        moving=moving,
        poses=poses,
        scale=2.0,
        starting_index=4,
        quality_rank_by_key={
            moving_rows[0]["observation_key"]: 1,
            moving_rows[1]["observation_key"]: 1,
        },
    )
    assert len(records) == 1
    assert next_index == 5
    candidate = records[0]
    assert candidate["support_count"] == 4
    assert candidate["transform_chain"]["direction"] == "target->root"
    assert candidate["transform_chain"]["terms"] == [
        "s0:root_marker->root",
        "inv(s2):moving_i->root_marker",
        "scaled(Tcw_i@inv(Tcw_j)):moving_j->moving_i",
        "inv(s1@inv(s3)):target->moving_j",
    ]
    assert candidate["aggregate_score"] == pytest.approx(
        (0.81 * 0.16 * 0.25 * 1.0) ** 0.25
    )


def test_wizard_moving_selection_is_quality_ranked_with_frame_tie_break() -> None:
    rows = []
    for frame in range(9):
        item = _frozen_row(
            camera="moving_calib_camera",
            marker=7,
            source_kind="moving",
            frame=frame,
            index=frame,
            score=0.5 if frame in {1, 2} else float(frame),
        )
        _, prepared = _prepared_rows([item], implementation="wizard")
        rows.extend(prepared)
    selected, ranking = _wizard_moving(rows, set(range(9)))
    assert [row["_frame"] for row in selected[7]] == list(range(1, 9))
    tied = [row for row in ranking if row["frame_id"] in {1, 2}]
    assert [row["frame_id"] for row in tied] == [1, 2]
    rejected = [row for row in ranking if not row["selected"]]
    assert [row["frame_id"] for row in rejected] == [0]
    assert rejected[0]["rejection_reason"].endswith("_8")


def test_relay_support_is_grouped_by_independent_marker_chain() -> None:
    _, _, direct = _candidate_fixture()
    first = _candidate_core_row(direct)
    first.update({"root_marker": 1, "target_marker": 2})
    second = {**first, "T": first["T"].copy(), "root_marker": 3, "target_marker": 4}
    _, statistics, chains = core.aggregate_relay_marker_chains([first, second])
    assert statistics["raw_candidate_count"] == 2
    assert statistics["independent_marker_chain_count"] == 2
    assert [chain["chain_id"] for chain in chains] == ["1->2", "3->4"]


def test_selection_comparison_classifies_eligibility_before_method_label() -> None:
    per_camera = {
        camera: {
            "deployment_eligible": True,
            "omitted": False,
            "omission_reason": None,
            "selected_candidate_type": "root" if camera == "cam_edge_3" else "relay",
            "selected_method": "gauge_identity" if camera == "cam_edge_3" else "relay",
        }
        for camera in ("cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5")
    }
    legacy = {
        "root_camera": "cam_edge_3",
        "per_camera": per_camera,
        "candidate_eligibility_before_ranking": "available",
        "candidate_sorting": "fixed",
        "tie_breaking": "first",
    }
    wizard = {**legacy, "per_camera": {key: dict(value) for key, value in per_camera.items()}}
    wizard["per_camera"]["cam_edge_0"].update(
        {
            "deployment_eligible": False,
            "omitted": True,
            "omission_reason": "rejected_unstable_consensus",
            "selected_candidate_type": "direct",
        }
    )
    report, differences = compare_selection(legacy, wizard)
    assert differences
    assert report["classification"] == "DIFFERENT_ELIGIBILITY_GATE"
    assert report["first_causal_divergence"]["difference"]["camera_id"] == (
        "cam_edge_0"
    )


def test_wizard_per_camera_aggregate_selection_applies_rejection_gate() -> None:
    _, _, direct = _candidate_fixture()
    selection = _wizard_aggregate_and_select([_root_candidate("wizard"), direct])
    assert selection["per_camera"]["cam_edge_3"]["deployment_eligible"] is True
    cam1 = selection["per_camera"]["cam_edge_1"]
    assert cam1["selected_candidate_type"] == "direct"
    assert cam1["selected_method"] == "direct_multimarker_diagnostic"
    assert cam1["deployment_eligible"] is False
    assert cam1["direct"]["gate_checks"]["minimum_independent_markers"] is False
    assert direct["rejection_reason"] == "direct_path_failed_quality_gate"


def _csv_observation() -> dict[str, str]:
    row = {
        "observer_type": "static",
        "observer_id": "cam_edge_3",
        "camera_name": "cam_edge_3",
        "frame_id": "static",
        "image_path": "static.png",
        "marker_id": "7",
        "marker_length_m": "0.17",
        "fx": "900",
        "fy": "900",
        "cx": "640",
        "cy": "360",
        "pnp_success": "True",
        "rvec_x": "0",
        "rvec_y": "0",
        "rvec_z": "0",
        "tvec_x_m": "0",
        "tvec_y_m": "0",
        "tvec_z_m": "1",
        "distance_m": "1",
        "center_u": "640",
        "center_v": "360",
        "area_px2": "400",
        "distortion_model": "plumb_bob",
        "image_width": "1280",
        "image_height": "720",
        "image_width_px": "1280",
        "image_height_px": "720",
        "pnp_reprojection_rmse_px": "0",
        "selection_score": "0.5",
        "score_area": "0.5",
        "score_reprojection": "1",
        "score_border": "1",
        "score_distance": "1",
        "decision": "accepted",
        "reason": "accepted",
    }
    for index, (u, v) in enumerate(((0, 0), (20, 0), (20, 20), (0, 20))):
        row[f"corner{index}_u"] = str(u)
        row[f"corner{index}_v"] = str(v)
    for index in range(8):
        row[f"d{index}"] = "0"
    return row


def _write_observation_csv(path: Path, row: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_frozen_input_is_deterministic_gt_free_and_does_not_modify_results(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.csv"
    wizard = tmp_path / "wizard.csv"
    row = _csv_observation()
    _write_observation_csv(legacy, row)
    _write_observation_csv(wizard, row)
    images = tmp_path / "images.txt"
    images.write_text("# frozen empty test model\n", encoding="utf-8")
    scale = tmp_path / "scale.txt"
    scale.write_text("1.0\n", encoding="utf-8")
    published = tmp_path / "results" / "SUMMARY.json"
    published.parent.mkdir()
    published.write_text('{"sentinel":true}\n', encoding="utf-8")
    before = hashlib.sha256(published.read_bytes()).hexdigest()

    manifests = []
    payloads = []
    for name in ("first", "second"):
        frozen = tmp_path / name
        manifests.append(
            freeze_ap01_input(
                legacy_accepted_csv=legacy,
                wizard_accepted_csv=wizard,
                colmap_images_txt=images,
                metric_scale_txt=scale,
                frozen_root=frozen,
            )
        )
        payloads.append((frozen / "ap01_accepted_observations.jsonl").read_bytes())
    assert payloads[0] == payloads[1]
    assert manifests[0]["canonical_input"]["sha256"] == manifests[1][
        "canonical_input"
    ]["sha256"]
    assert manifests[0]["ground_truth_fields_present"] is False
    assert hashlib.sha256(published.read_bytes()).hexdigest() == before


def test_candidate_helpers_invoke_neither_solver_nor_colmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden full stage invoked")

    monkeypatch.setattr(core, "run_colmap", forbidden)
    monkeypatch.setattr(solve_extrinsics, "run", forbidden)
    _, _, candidate = _candidate_fixture()
    assert candidate["native_candidate"] is True


def test_ground_truth_named_input_is_rejected_before_read(tmp_path: Path) -> None:
    forbidden = tmp_path / "ground_truth" / "accepted.csv"
    with pytest.raises(PermissionError, match="Ground Truth"):
        freeze_ap01_input(
            legacy_accepted_csv=forbidden,
            wizard_accepted_csv=tmp_path / "wizard.csv",
            colmap_images_txt=tmp_path / "images.txt",
            metric_scale_txt=tmp_path / "scale.txt",
            frozen_root=tmp_path / "frozen",
        )

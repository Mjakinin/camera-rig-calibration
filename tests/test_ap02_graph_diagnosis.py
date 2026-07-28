from __future__ import annotations

import csv
import json
from pathlib import Path

from camera_rig_calibration.ap02_graph import diagnose_ap02_graph
from camera_rig_calibration.methods.ap02 import component_diagnostics
from camera_rig_calibration.pipeline import StageResult
from camera_rig_calibration.queueing import QueueRunner


def _row(
    observer: str,
    marker: int,
    *,
    observer_type: str,
    reason: str = "",
) -> dict[str, str]:
    return {
        "observer_id": observer,
        "camera_name": observer if observer_type == "static" else "",
        "observer_type": observer_type,
        "marker_id": str(marker),
        "reason": reason,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_img4388_graph_shape_is_reproduced_without_claiming_visibility() -> None:
    expected = (
        "camera_back_center",
        "camera_back_left",
        "camera_back_right",
        "camera_center_left",
        "camera_center_right",
        "camera_front_left",
        "camera_front_narrow",
        "camera_front_right",
        "camera_front_wide",
    )
    group_one_markers = (2, 3, 4, 5, 6, 8, 12, 15, 16, 18, 20, 37)
    rows = [
        _row("camera_back_left", 8, observer_type="static"),
        _row("camera_center_left", 2, observer_type="static"),
        _row("camera_center_right", 3, observer_type="static"),
        _row("camera_front_right", 4, observer_type="static"),
        *[
            _row("moving_group_one", marker, observer_type="moving")
            for marker in group_one_markers
        ],
        _row("camera_front_narrow", 7, observer_type="static"),
        _row("camera_front_wide", 9, observer_type="static"),
        _row("moving_group_two", 7, observer_type="moving"),
        _row("moving_group_two", 9, observer_type="moving"),
        _row("camera_back_center", 10, observer_type="static"),
        _row("moving_back_center", 10, observer_type="moving"),
        _row("moving_back_center", 17, observer_type="moving"),
        _row("camera_front_left", 0, observer_type="static"),
        _row("moving_front_left", 0, observer_type="moving"),
        _row("moving_front_left", 19, observer_type="moving"),
        _row("moving_only", 26, observer_type="moving"),
    ]

    diagnosis = diagnose_ap02_graph(
        raw_rows=rows,
        accepted_rows=rows,
        rejected_rows=[],
        static_camera_ids=expected,
        reference_marker_id=8,
    )

    assert len(diagnosis.components) == 5
    assert diagnosis.components[0].static_cameras == (
        "camera_back_left",
        "camera_center_left",
        "camera_center_right",
        "camera_front_right",
    )
    assert diagnosis.components[0].marker_ids == group_one_markers
    assert diagnosis.components[1].static_cameras == (
        "camera_front_narrow",
        "camera_front_wide",
    )
    assert diagnosis.components[1].marker_ids == (7, 9)
    assert {
        component.static_cameras
        for component in diagnosis.components[2:4]
    } == {
        ("camera_back_center",),
        ("camera_front_left",),
    }
    assert diagnosis.components[4].static_cameras == ()
    assert diagnosis.components[4].marker_ids == (26,)
    assert len(diagnosis.calibratable_components) == 2
    assert "required_camera_without_detection" in diagnosis.cause_codes
    assert "no_detected_cross_group_observations" in diagnosis.cause_codes
    assert "camera_back_right" in diagnosis.explanation
    assert "cannot distinguish" in diagnosis.explanation


def test_quality_rejections_are_named_only_when_they_remove_bridges() -> None:
    accepted = [
        _row("cam_1", 1, observer_type="static"),
        _row("cam_2", 1, observer_type="static"),
        _row("moving_a", 1, observer_type="moving"),
        _row("cam_3", 2, observer_type="static"),
        _row("cam_4", 2, observer_type="static"),
        _row("moving_b", 2, observer_type="moving"),
    ]
    rejected = [
        _row(
            "transition_frame",
            1,
            observer_type="moving",
            reason="pnp_reprojection_rmse_above_limit",
        ),
        _row(
            "transition_frame",
            2,
            observer_type="moving",
            reason="pnp_reprojection_rmse_above_limit",
        ),
        _row(
            "unrelated_frame",
            99,
            observer_type="moving",
            reason="marker_area_below_limit",
        ),
    ]
    diagnosis = diagnose_ap02_graph(
        raw_rows=[*accepted, *rejected],
        accepted_rows=accepted,
        rejected_rows=rejected,
        static_camera_ids=("cam_1", "cam_2", "cam_3", "cam_4"),
        reference_marker_id=1,
    )

    assert diagnosis.cause_codes == ("quality_filters_removed_bridges",)
    assert diagnosis.rejected_bridge_reasons == (
        ("pnp_reprojection_rmse_above_limit", 2),
    )
    assert "marker_area_below_limit" not in diagnosis.explanation


def test_disconnected_components_are_calibrated_without_cross_alignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "03_AP02"
    observations = output / "02_aruco_observations" / "components"
    components = [
        {
            "component_id": "component_01",
            "static_cameras": ["cam_1", "cam_2", "cam_3", "cam_4"],
            "marker_ids": [1, 2],
            "moving_frames": ["moving_1"],
            "moving_frame_count": 1,
            "calibratable": True,
            "anchor_marker_id": 1,
        },
        {
            "component_id": "component_02",
            "static_cameras": ["cam_5", "cam_6"],
            "marker_ids": [7, 9],
            "moving_frames": ["moving_2"],
            "moving_frame_count": 1,
            "calibratable": True,
            "anchor_marker_id": 9,
        },
        {
            "component_id": "component_03",
            "static_cameras": ["cam_7"],
            "marker_ids": [10],
            "moving_frames": ["moving_3"],
            "moving_frame_count": 1,
            "calibratable": False,
            "anchor_marker_id": 10,
        },
    ]
    manifest = {
        "schema_version": 5,
        "primary_component_id": "component_01",
        "expected_static_cameras": [
            "cam_1",
            "cam_2",
            "cam_3",
            "cam_4",
            "cam_5",
            "cam_6",
            "cam_7",
        ],
        "components": components,
    }
    manifest_path = output / "02_aruco_observations/component_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _write_csv(
        observations
        / "component_02"
        / "ap02_all_aruco_observations.csv",
        [
            _row("cam_5", 9, observer_type="static"),
            _row("cam_6", 9, observer_type="static"),
            _row("moving_2", 9, observer_type="moving"),
        ],
    )
    _write_csv(
        observations
        / "component_02"
        / "ap02_static_aruco_observations.csv",
        [
            _row("cam_5", 9, observer_type="static"),
            _row("cam_6", 9, observer_type="static"),
        ],
    )
    _write_csv(
        observations
        / "component_02"
        / "ap02_moving_aruco_observations.csv",
        [_row("moving_2", 9, observer_type="moving")],
    )

    calls: list[tuple[str, Path, int]] = []

    def fake_initialization(
        *,
        output_root: Path,
        reference_marker_id: int,
        mode: str,
        log_path: Path | None = None,
    ) -> StageResult:
        calls.append(("initialize", output_root, reference_marker_id))
        assert mode == "with_moving"
        assert log_path is not None
        return StageResult(
            "fake.initialize", "COMPLETED", output_root, {}, 0.0
        )

    def fake_optimization(
        *,
        output_root: Path,
        reference_marker_id: int,
        mode: str,
        maximum_function_evaluations: int,
        robust_loss: str,
        robust_loss_scale_px: float,
        log_path: Path | None = None,
    ) -> StageResult:
        calls.append(("optimize", output_root, reference_marker_id))
        poses = (
            output_root
            / "07_graph_ba/with_moving"
            / "optimized_static_camera_poses_ref_marker.csv"
        )
        _write_csv(
            poses,
            [
                {
                    "entity_id": camera,
                    "x_m": str(index),
                    "y_m": "0",
                    "z_m": "0",
                }
                for index, camera in enumerate(("cam_5", "cam_6"))
            ],
        )
        report = poses.parent / "optimizer_report.json"
        report.write_text(json.dumps({"success": True}), encoding="utf-8")
        return StageResult(
            "fake.optimize", "COMPLETED", output_root, {}, 0.0
        )

    monkeypatch.setattr(
        component_diagnostics, "run_initialization", fake_initialization
    )
    monkeypatch.setattr(
        component_diagnostics, "run_optimization", fake_optimization
    )

    result = component_diagnostics.run(
        output_root=output,
        maximum_function_evaluations=20,
        robust_loss="soft_l1",
        robust_loss_scale_px=3.0,
    )

    assert result.status == "COMPLETED"
    assert [item[0] for item in calls] == ["initialize", "optimize"]
    summary = json.loads(
        (
            output
            / "09_component_diagnostics/AP02_COMPONENT_RESULTS.json"
        ).read_text(encoding="utf-8")
    )
    statuses = {
        item["component_id"]: item["execution_status"]
        for item in summary["components"]
    }
    assert statuses == {
        "component_01": "primary_component",
        "component_02": "available",
        "component_03": "not_calibratable",
    }
    observability = {
        (row["camera_a"], row["camera_b"]): row["status"]
        for row in summary["camera_pair_observability"]
    }
    assert observability[("cam_5", "cam_6")] == "within_component"
    assert observability[("cam_1", "cam_5")] == "not_observable"
    assert (
        output
        / "09_component_diagnostics/component_02/camera_extrinsics.csv"
    ).is_file()


def test_detector_retry_reuses_normalized_input_and_archives_evidence(
    tmp_path: Path,
    prepared_config,
    monkeypatch,
) -> None:
    transaction = tmp_path / "transaction"
    prepared = transaction / "dataset"
    raw_image = prepared / "raw_images/static/cam.png"
    raw_image.parent.mkdir(parents=True, exist_ok=True)
    raw_image.write_bytes(b"normalized-frame")
    observations = prepared / "observations"
    observations.mkdir(parents=True)
    (observations / "shared_all_aruco_observations.csv").write_text(
        "observer_id,marker_id\ncam,1\n", encoding="utf-8"
    )
    (observations / "detection_config.json").write_text(
        json.dumps({"input_id": "input_fixture"}), encoding="utf-8"
    )
    old_debug = observations / "debug_images/baseline/static/cam.png"
    old_debug.parent.mkdir(parents=True)
    old_debug.write_bytes(b"baseline-debug")
    resolved = transaction / "resolved"
    preflight = resolved / "preflight"
    preflight.mkdir(parents=True)
    (preflight / "queue_preflight_summary.json").write_text(
        "{}", encoding="utf-8"
    )
    calls: list[Path] = []

    def fake_detect(
        orchestrator,
        config,
        *,
        dataset_root: Path,
        run_directory: Path,
    ) -> Path:
        calls.append(dataset_root)
        generated = run_directory / "01_OBSERVATIONS"
        generated.mkdir(parents=True)
        (generated / "shared_all_aruco_observations.csv").write_text(
            "observer_id,marker_id\ncam,17\n", encoding="utf-8"
        )
        logs = run_directory / "logs"
        logs.mkdir()
        (logs / "detector.log").write_text(
            "high sensitivity detection", encoding="utf-8"
        )
        (run_directory / "commands.txt").write_text(
            "observation_detection.py --mode high_sensitivity\n",
            encoding="utf-8",
        )
        return generated

    monkeypatch.setattr(
        "camera_rig_calibration.queueing."
        "PipelineOrchestrator.detect_observations_only",
        fake_detect,
    )
    runner = QueueRunner(Path(__file__).resolve().parents[1])
    updated = runner._retry_detector_on_prepared_input(
        transaction_root=transaction,
        resolved_root=resolved,
        prepared_root=prepared,
        preparation_path=transaction / "jobs/queue_preflight/prepared",
        configs=[prepared_config],
        detection_mode="high_sensitivity",
    )

    assert calls == [prepared]
    assert raw_image.read_bytes() == b"normalized-frame"
    assert (
        "cam,17"
        in (
            observations / "shared_all_aruco_observations.csv"
        ).read_text(encoding="utf-8")
    )
    assert updated[0].markers.detection_mode == "high_sensitivity"
    assert updated[0].dataset.id.endswith("__aruco_high_sensitivity")
    attempts = list((resolved / "detector_attempts").iterdir())
    assert len(attempts) == 1
    attempt = attempts[0]
    assert (
        attempt
        / "observations/debug_images/baseline/static/cam.png"
    ).is_file()
    assert (attempt / "preflight/queue_preflight_summary.json").is_file()
    evidence = json.loads((attempt / "ATTEMPT.json").read_text())
    assert evidence["capture_repeated"] is False
    assert evidence["video_extraction_repeated"] is False
    assert evidence["intrinsics_repeated"] is False
    assert (attempt / "retry_execution/logs/detector.log").is_file()


def test_detector_retry_copies_when_windows_blocks_generated_rename(
    tmp_path: Path,
    prepared_config,
    monkeypatch,
) -> None:
    transaction = tmp_path / "transaction"
    prepared = transaction / "dataset"
    observations = prepared / "observations"
    observations.mkdir(parents=True)
    (observations / "shared_all_aruco_observations.csv").write_text(
        "observer_id,marker_id\ncam,1\n", encoding="utf-8"
    )
    (observations / "detection_config.json").write_text(
        json.dumps({"input_id": "input_fixture"}), encoding="utf-8"
    )
    resolved = transaction / "resolved"
    original_rename = Path.rename

    def locked_generated_rename(source: Path, target: Path) -> Path:
        if source.name == "01_OBSERVATIONS":
            raise PermissionError(13, "simulated Windows directory lock")
        return original_rename(source, target)

    def fake_detect(
        orchestrator,
        config,
        *,
        dataset_root: Path,
        run_directory: Path,
    ) -> Path:
        generated = run_directory / "01_OBSERVATIONS"
        generated.mkdir(parents=True)
        (generated / "shared_all_aruco_observations.csv").write_text(
            "observer_id,marker_id\ncam,17\n", encoding="utf-8"
        )
        return generated

    monkeypatch.setattr(
        "camera_rig_calibration.queueing."
        "PipelineOrchestrator.detect_observations_only",
        fake_detect,
    )
    monkeypatch.setattr(Path, "rename", locked_generated_rename)

    updated = QueueRunner(
        Path(__file__).resolve().parents[1]
    )._retry_detector_on_prepared_input(
        transaction_root=transaction,
        resolved_root=resolved,
        prepared_root=prepared,
        preparation_path=transaction / "jobs/queue_preflight/prepared",
        configs=[prepared_config],
        detection_mode="high_sensitivity",
    )

    assert updated[0].markers.detection_mode == "high_sensitivity"
    assert (
        "cam,17"
        in (
            observations / "shared_all_aruco_observations.csv"
        ).read_text(encoding="utf-8")
    )
    attempt = next((resolved / "detector_attempts").iterdir())
    evidence = json.loads((attempt / "ATTEMPT.json").read_text())
    assert (
        evidence["observation_promotion"]
        == "copied_after_locked_rename"
    )
    assert (
        transaction
        / "jobs/queue_preflight/detector_retries"
    ).is_dir()


def test_interrupted_detector_promotion_is_resumed_without_detection(
    tmp_path: Path,
    prepared_config,
) -> None:
    transaction = tmp_path / "transaction"
    generated = (
        transaction
        / "jobs/queue_preflight/detector_retries"
        / "20260728_221241_123456_high_sensitivity"
        / "01_OBSERVATIONS"
    )
    generated.mkdir(parents=True)
    (generated / "shared_all_aruco_observations.csv").write_text(
        "observer_id,marker_id\ncam,17\n", encoding="utf-8"
    )
    (generated / "effective_detection_config.json").write_text(
        json.dumps(
            {
                "mode": "high_sensitivity",
                "dictionary": prepared_config.markers.dictionary,
            }
        ),
        encoding="utf-8",
    )
    resolved = transaction / "resolved"
    attempt = (
        resolved
        / "detector_attempts"
        / "20260728_221241_123456_baseline"
    )
    archived = attempt / "observations"
    archived.mkdir(parents=True)
    (archived / "detection_config.json").write_text(
        json.dumps({"input_id": "input_fixture"}), encoding="utf-8"
    )

    recovered = QueueRunner(
        Path(__file__).resolve().parents[1]
    )._recover_interrupted_detector_retry(
        transaction_root=transaction,
        resolved_root=resolved,
        configs=[prepared_config],
    )

    assert recovered is not None
    updated, previous_mode, next_mode = recovered
    assert previous_mode == "baseline"
    assert next_mode == "high_sensitivity"
    assert updated[0].markers.detection_mode == "high_sensitivity"
    observations = transaction / "dataset/observations"
    assert (
        "cam,17"
        in (
            observations / "shared_all_aruco_observations.csv"
        ).read_text(encoding="utf-8")
    )
    detection = json.loads(
        (observations / "detection_config.json").read_text()
    )
    assert detection["input_id"] == "input_fixture"
    evidence = json.loads((attempt / "ATTEMPT.json").read_text())
    assert evidence["recovered_after_interrupted_promotion"] is True

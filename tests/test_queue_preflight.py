from __future__ import annotations

import csv
import json
from pathlib import Path

from camera_rig_calibration.config.models import (
    DatasetCategory,
    MethodSettings,
    ObservationQualitySettings,
)
from camera_rig_calibration.preflight import PreflightJob, run_queue_preflight


def _row(
    observer_type: str,
    observer_id: str,
    frame: str = "",
    marker_id: int = 7,
) -> dict[str, object]:
    corners = [(40, 60), (60, 60), (60, 40), (40, 40)]
    row: dict[str, object] = {
        "observer_type": observer_type,
        "observer_id": observer_id,
        "camera_name": observer_id if observer_type == "static" else "",
        "frame_id": frame,
        "image_path": f"frame_{frame}.png" if frame else f"{observer_id}.png",
        "marker_id": marker_id,
        "marker_length_m": 0.2,
        "detection_success": True,
        "pnp_success": True,
        "fx": 100,
        "fy": 100,
        "cx": 50,
        "cy": 50,
        "distortion_model": "plumb_bob",
        "rvec_x": 0,
        "rvec_y": 0,
        "rvec_z": 0,
        "tvec_x_m": 0,
        "tvec_y_m": 0,
        "tvec_z_m": 1,
        "area_px2": 400,
    }
    for index, (u, v) in enumerate(corners):
        row[f"corner{index}_u"] = u
        row[f"corner{index}_v"] = v
    for index in range(8):
        row[f"d{index}"] = 0.0
    return row


def test_failed_job_does_not_block_independent_runnable_job(
    prepared_config, tmp_path: Path
) -> None:
    source = tmp_path / "raw.csv"
    rows = [
        _row("static", "front-left"),
        _row("static", "roof.camera"),
        _row("moving", "moving_1", "1"),
        _row("moving", "moving_2", "2"),
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ready = prepared_config.model_copy(
        update={"methods": MethodSettings(enabled=["ap02"])}, deep=True
    )
    rejected = ready.model_copy(
        update={
            "observation_quality": ObservationQualitySettings(
                minimum_marker_area_px2=1000.0
            )
        },
        deep=True,
    )

    result = run_queue_preflight(
        [
            PreflightJob("ready", ready),
            PreflightJob("rejected", rejected),
        ],
        raw_observations_csv=source,
        dataset_root=prepared_config.dataset.prepared_root,
        output_directory=tmp_path / "preflight",
        repository_root=Path(__file__).resolve().parents[1],
    )

    assert result.status == "READY_PARTIAL"
    assert result.ready
    assert [job.status for job in result.jobs] == ["READY", "FAILED_PREFLIGHT"]
    summary = json.loads(
        (tmp_path / "preflight/queue_preflight_summary.json").read_text()
    )
    assert summary["methods_may_start"] is True
    assert summary["runnable_jobs"] == ["ready"]
    assert summary["skipped_jobs"] == ["rejected"]
    for job_id in ("ready", "rejected"):
        root = tmp_path / "preflight/jobs" / job_id
        assert (root / "preflight_summary.json").is_file()
        assert (root / "observation_filter_summary.json").is_file()
        assert (root / "accepted_observations.csv").is_file()
        assert (root / "rejected_observations.csv").is_file()


def test_ap02_partial_static_and_complete_combined_is_ready_without_warning(
    prepared_config, tmp_path: Path
) -> None:
    source = tmp_path / "raw.csv"
    rows = [
        _row("static", "cam_1", marker_id=3),
        _row("static", "cam_2", marker_id=3),
        _row("static", "cam_3", marker_id=3),
        _row("static", "cam_2", marker_id=4),
        _row("static", "cam_3", marker_id=4),
        _row("static", "cam_4", marker_id=5),
        _row("moving", "moving_frame_1", "1", marker_id=4),
        _row("moving", "moving_frame_1", "1", marker_id=5),
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    camera_ids = ["cam_1", "cam_2", "cam_3", "cam_4"]
    raw = prepared_config.dataset.prepared_root / "raw_images"
    existing_intrinsics = (
        raw / "camera_info" / "front-left.json"
    ).read_text(encoding="utf-8")
    for camera_id in camera_ids:
        (raw / "static" / f"{camera_id}.png").write_bytes(b"fixture")
        (raw / "camera_info" / f"{camera_id}.json").write_text(
            existing_intrinsics, encoding="utf-8"
        )
    config = prepared_config.model_copy(
        update={
            "static_cameras": [
                type(prepared_config.static_cameras[0])(id=camera_id)
                for camera_id in camera_ids
            ],
            "methods": MethodSettings(enabled=["ap02"]),
        },
        deep=True,
    )

    result = run_queue_preflight(
        [PreflightJob("ap02", config)],
        raw_observations_csv=source,
        dataset_root=prepared_config.dataset.prepared_root,
        output_directory=tmp_path / "preflight",
        repository_root=Path(__file__).resolve().parents[1],
    )

    job = result.jobs[0]
    assert result.status == "READY"
    assert job.status == "READY"
    assert job.warnings == ()
    assert job.errors == ()
    assert any(
        "AP02 Combined graph: 4/4 cameras" in detail
        for detail in job.details
    )
    assert not any("Static-only coverage" in detail for detail in job.details)
    assert "Status: READY" in job.details
    assert job.selections is not None
    assert job.selections.ap02_reference_marker_id == 3


def test_ap02_incomplete_combined_graph_blocks_preflight(
    prepared_config, tmp_path: Path
) -> None:
    source = tmp_path / "raw.csv"
    rows = [
        _row("static", "cam_1", marker_id=3),
        _row("static", "cam_2", marker_id=3),
        _row("static", "cam_3", marker_id=4),
        _row("static", "cam_4", marker_id=5),
        _row("moving", "moving_frame_1", "1", marker_id=4),
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw = prepared_config.dataset.prepared_root / "raw_images"
    existing_intrinsics = (
        raw / "camera_info" / "front-left.json"
    ).read_text(encoding="utf-8")
    for camera_id in ("cam_1", "cam_2", "cam_3", "cam_4"):
        (raw / "static" / f"{camera_id}.png").write_bytes(b"fixture")
        (raw / "camera_info" / f"{camera_id}.json").write_text(
            existing_intrinsics, encoding="utf-8"
        )
    camera_type = type(prepared_config.static_cameras[0])
    config = prepared_config.model_copy(
        update={
            "static_cameras": [
                camera_type(id=camera_id)
                for camera_id in ("cam_1", "cam_2", "cam_3", "cam_4")
            ],
            "methods": MethodSettings(enabled=["ap02"]),
        },
        deep=True,
    )

    result = run_queue_preflight(
        [PreflightJob("ap02", config)],
        raw_observations_csv=source,
        dataset_root=prepared_config.dataset.prepared_root,
        output_directory=tmp_path / "preflight",
        repository_root=Path(__file__).resolve().parents[1],
    )

    assert result.status == "REVIEW_REQUIRED"
    assert result.review_required
    assert result.jobs[0].status == "FAILED_PREFLIGHT"
    assert result.jobs[0].warnings == ()
    assert any(
        "compatible selection candidates" in error
        or "Combined input graph" in error
        or "no usable component" in error
        for error in result.jobs[0].errors
    )


def test_ap02_three_of_four_combined_is_runnable_diagnostic_partial(
    prepared_config, tmp_path: Path
) -> None:
    source = tmp_path / "raw.csv"
    rows = [
        _row("static", "cam_1", marker_id=3),
        _row("static", "cam_2", marker_id=3),
        _row("static", "cam_3", marker_id=4),
        _row("static", "cam_4", marker_id=9),
        _row("moving", "moving_frame_1", "1", marker_id=3),
        _row("moving", "moving_frame_1", "1", marker_id=4),
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw = prepared_config.dataset.prepared_root / "raw_images"
    existing_intrinsics = (
        raw / "camera_info" / "front-left.json"
    ).read_text(encoding="utf-8")
    camera_type = type(prepared_config.static_cameras[0])
    cameras = ["cam_1", "cam_2", "cam_3", "cam_4"]
    for camera_id in cameras:
        (raw / "static" / f"{camera_id}.png").write_bytes(b"fixture")
        (raw / "camera_info" / f"{camera_id}.json").write_text(
            existing_intrinsics, encoding="utf-8"
        )
    config = prepared_config.model_copy(
        update={
            "static_cameras": [
                camera_type(id=camera_id) for camera_id in cameras
            ],
            "methods": MethodSettings(enabled=["ap02"]),
        },
        deep=True,
    )

    result = run_queue_preflight(
        [PreflightJob("ap02", config)],
        raw_observations_csv=source,
        dataset_root=prepared_config.dataset.prepared_root,
        output_directory=tmp_path / "preflight",
        repository_root=Path(__file__).resolve().parents[1],
    )

    job = result.jobs[0]
    assert result.ready
    assert result.status == "REVIEW_REQUIRED"
    assert result.review_required
    assert result.review_reasons == ("ap02_combined_graph_incomplete",)
    assert job.status == "READY_PARTIAL"
    assert job.errors == ()
    assert job.selections is not None
    assert job.selections.ap02_reference_marker_id == 3
    assert any(
        "AP02 Combined graph: 3/4 cameras" in detail
        for detail in job.details
    )
    assert job.ap02_graph_diagnosis is not None
    assert job.ap02_graph_diagnosis.missing_static_cameras == ("cam_4",)


def test_required_camera_without_raw_observation_is_reported(
    prepared_config, tmp_path: Path
) -> None:
    source = tmp_path / "raw.csv"
    moving_id = prepared_config.moving_camera.id
    rows = [
        _row("static", "front-left"),
        {
            **_row("moving", moving_id, "1"),
            "camera_name": moving_id,
        },
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = prepared_config.model_copy(
        update={"methods": MethodSettings(enabled=["ap01"])},
        deep=True,
    )

    result = run_queue_preflight(
        [PreflightJob("ap01", config)],
        raw_observations_csv=source,
        dataset_root=prepared_config.dataset.prepared_root,
        output_directory=tmp_path / "preflight",
        repository_root=Path(__file__).resolve().parents[1],
    )

    assert result.missing_required_cameras == ("roof.camera",)
    coverage = {
        item.camera_id: item for item in result.camera_coverage
    }
    assert coverage["front-left"].raw_detection_count == 1
    assert coverage["roof.camera"].raw_detection_count == 0
    summary = json.loads(
        (tmp_path / "preflight/queue_preflight_summary.json").read_text()
    )
    assert summary["missing_required_cameras"] == ["roof.camera"]


def test_simulation_ap02_incomplete_graph_uses_the_same_review_gate(
    prepared_config, tmp_path: Path
) -> None:
    source = tmp_path / "simulation_observations.csv"
    rows = [
        _row("static", "sim_cam_1", marker_id=3),
        _row("static", "sim_cam_2", marker_id=4),
        _row("static", "sim_cam_3", marker_id=9),
        _row("moving", "moving_frame_1", "1", marker_id=3),
        _row("moving", "moving_frame_1", "1", marker_id=4),
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw = prepared_config.dataset.prepared_root / "raw_images"
    intrinsics = (
        raw / "camera_info" / "front-left.json"
    ).read_text(encoding="utf-8")
    camera_type = type(prepared_config.static_cameras[0])
    cameras = ["sim_cam_1", "sim_cam_2", "sim_cam_3"]
    for camera_id in cameras:
        (raw / "static" / f"{camera_id}.png").write_bytes(b"fixture")
        (raw / "camera_info" / f"{camera_id}.json").write_text(
            intrinsics, encoding="utf-8"
        )
    config = prepared_config.model_copy(
        update={
            "dataset": prepared_config.dataset.model_copy(
                update={"category": DatasetCategory.SIMULATION}
            ),
            "static_cameras": [
                camera_type(id=camera_id) for camera_id in cameras
            ],
            "methods": MethodSettings(enabled=["ap02"]),
        },
        deep=True,
    )

    result = run_queue_preflight(
        [PreflightJob("simulation_ap02", config)],
        raw_observations_csv=source,
        dataset_root=prepared_config.dataset.prepared_root,
        output_directory=tmp_path / "simulation_preflight",
        repository_root=Path(__file__).resolve().parents[1],
    )

    assert config.dataset.category is DatasetCategory.SIMULATION
    assert result.status == "REVIEW_REQUIRED"
    assert result.review_reasons == ("ap02_combined_graph_incomplete",)
    diagnosis = result.jobs[0].ap02_graph_diagnosis
    assert diagnosis is not None
    assert diagnosis.reached_static_cameras == ("sim_cam_1", "sim_cam_2")
    assert diagnosis.missing_static_cameras == ("sim_cam_3",)

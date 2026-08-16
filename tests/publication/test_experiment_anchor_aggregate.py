from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.anchor_export.aggregate import (
    build_experiment_anchor_aggregate,
    experiment_anchor_aggregate_text,
)


def _write_variant(
    root: Path,
    *,
    method: str,
    label: str,
    x_offset: float,
) -> None:
    target = root / "methods" / method / label / "camera_extrinsics_anchor.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    cameras = []
    for index, camera_id in enumerate(("cam_edge_0", "cam_edge_1")):
        cameras.append(
            {
                "method": method,
                "label": label,
                "anchor_marker_id": 14,
                "parent_frame": "evaluation_anchor_marker_14",
                "camera_id": camera_id,
                "quality_status": "accepted",
                "deployment_eligible": True,
                "x_m": x_offset + index,
                "y_m": 0.0,
                "z_m": 1.0,
                "roll_rad": 0.0,
                "pitch_rad": 0.0,
                "yaw_rad": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            }
        )
    target.write_text(
        json.dumps(
            {
                "method": method,
                "label": label,
                "anchor_marker_id": 14,
                "parent_frame": "evaluation_anchor_marker_14",
                "anchor_export_status": {"available": True},
                "cameras": cameras,
            }
        ),
        encoding="utf-8",
    )


def test_common_anchor_aggregate_contains_public_variants_only(
    tmp_path: Path,
) -> None:
    _write_variant(
        tmp_path,
        method="ap02",
        label="combined_nfev_2__ref_mode_auto",
        x_offset=2.0,
    )
    _write_variant(
        tmp_path,
        method="ap03",
        label="baseline",
        x_offset=2.5,
    )
    _write_variant(
        tmp_path,
        method="ap03_single",
        label="baseline",
        x_offset=3.0,
    )
    _write_variant(
        tmp_path,
        method="ap03_multi",
        label="baseline",
        x_offset=4.0,
    )

    payload = build_experiment_anchor_aggregate(tmp_path)

    variant_keys = {
        (item["method"], item["label"])
        for item in payload["variants"]
    }
    assert variant_keys == {
        ("ap02", "combined_nfev_2__ref_mode_auto"),
        ("ap03_single", "baseline"),
        ("ap03_multi", "baseline"),
    }
    assert ("ap03", "baseline") not in variant_keys
    assert len(payload["rows"]) == 6
    assert payload["rows"] == payload["all_published_variant_rows"]
    assert payload["anchor_marker_id"] == 14

    written = json.loads(
        (tmp_path / "CAMERA_EXTRINSICS_COMMON_ANCHOR.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        (row["method"], row["label"])
        for row in written["rows"]
    } == variant_keys
    assert (tmp_path / "CAMERA_EXTRINSICS_COMMON_ANCHOR.csv").is_file()
    assert (tmp_path / "CAMERA_EXTRINSICS_COMMON_ANCHOR.yaml").is_file()

    text = experiment_anchor_aggregate_text(payload)
    assert "ap02/combined_nfev_2__ref_mode_auto:" in text
    assert "ap03_single/baseline:" in text
    assert "ap03_multi/baseline:" in text
    assert "\nap03/baseline:" not in text


def test_common_anchor_aggregate_keeps_ap03_container_without_both_scale_variants(
    tmp_path: Path,
) -> None:
    _write_variant(
        tmp_path,
        method="ap03",
        label="baseline",
        x_offset=2.5,
    )
    _write_variant(
        tmp_path,
        method="ap03_single",
        label="baseline",
        x_offset=3.0,
    )

    payload = build_experiment_anchor_aggregate(tmp_path)

    assert {
        (item["method"], item["label"])
        for item in payload["variants"]
    } == {
        ("ap03", "baseline"),
        ("ap03_single", "baseline"),
    }


def test_common_anchor_aggregate_replaces_stale_subset(tmp_path: Path) -> None:
    stale = tmp_path / "CAMERA_EXTRINSICS_COMMON_ANCHOR.json"
    stale.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "method": "ap02",
                        "label": "old_only",
                        "camera_id": "cam_edge_0",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_variant(
        tmp_path,
        method="ap03_single",
        label="baseline",
        x_offset=3.0,
    )

    payload = build_experiment_anchor_aggregate(tmp_path)

    assert {
        (row["method"], row["label"])
        for row in payload["rows"]
    } == {("ap03_single", "baseline")}
    assert "old_only" not in stale.read_text(encoding="utf-8")

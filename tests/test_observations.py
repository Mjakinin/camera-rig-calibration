from __future__ import annotations

import csv
import json
from pathlib import Path

from camera_rig_calibration.observations import resolve_references


def _row(observer_type: str, observer_id: str, marker: int, area: float) -> dict[str, object]:
    return {
        "observer_type": observer_type,
        "observer_id": observer_id,
        "marker_id": marker,
        "area_px2": area,
        "pnp_success": True,
    }


def test_automatic_references_use_visible_fields_and_are_logged(
    prepared_config, tmp_path: Path
) -> None:
    root = tmp_path / "observations"
    root.mkdir()
    rows = [
        _row("static", "front-left", 7, 100),
        _row("static", "front-left", 9, 90),
        _row("static", "roof.camera", 7, 80),
        _row("moving", "moving_frame_000001", 7, 70),
        _row("moving", "moving_frame_000002", 7, 65),
        _row("moving", "moving_frame_000003", 9, 60),
    ]
    path = root / "shared_all_aruco_observations.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    resolved = resolve_references(prepared_config, root)
    assert resolved.root_camera == "front-left"
    assert resolved.ap02_reference_marker_id == 7
    assert resolved.ap03_single_scale_marker_id == 7
    assert resolved.evaluation_anchor_marker_id is None
    audit = json.loads((root / "REFERENCE_SELECTIONS.json").read_text())
    assert audit["ap01_root_camera"]["configured"] == "auto"
    assert audit["evaluation_anchor"]["resolution_stage"] == "after_methods"
    assert audit["detected_marker_ids"] == [7, 9]
    serialized = json.dumps(audit).lower()
    assert "bottleneck" not in serialized
    assert '"score"' not in serialized

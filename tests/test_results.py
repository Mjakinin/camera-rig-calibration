from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.results import index_results


def test_legacy_manifests_are_not_indexed(tmp_path: Path) -> None:
    run = tmp_path / "results/real_vehicle/1Hz/legacy/methods/ap02/baseline"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (run.parent.parent.parent / "99_FINAL_RESULTS").mkdir()

    assert index_results(tmp_path / "results") == []

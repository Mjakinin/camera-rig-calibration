from __future__ import annotations

import json
from pathlib import Path

import pytest

from camera_rig_calibration.pipeline import (
    StageContract,
    run_stage,
    validate_stage_dag,
)


def test_stage_writes_atomic_completed_manifest(tmp_path: Path) -> None:
    output = tmp_path / "stage"
    artifact = output / "artifact.txt"

    def action() -> dict[str, Path | int]:
        artifact.write_text("result\n", encoding="utf-8")
        return {"artifact": artifact, "count": 1}

    result = run_stage(
        "fixture.stage",
        output,
        action,
        inputs={"source": tmp_path / "input.csv"},
        parameters={"threshold": 5.0},
    )

    manifest = json.loads((output / "stage_manifest.json").read_text())
    assert result.status == "COMPLETED"
    assert manifest["schema_version"] == 5
    assert manifest["status"] == "COMPLETED"
    assert manifest["outputs"]["artifact"] == str(artifact)
    assert manifest["parameters"] == {"threshold": 5.0}
    assert not (output / "stage_manifest.json.tmp").exists()


def test_failed_primary_stage_records_failure_and_raises(tmp_path: Path) -> None:
    output = tmp_path / "stage"

    with pytest.raises(RuntimeError, match="fixture failure"):
        run_stage(
            "fixture.primary",
            output,
            lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")),
        )

    manifest = json.loads((output / "stage_manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert "fixture failure" in manifest["error"]


def test_failed_diagnostic_stage_is_recorded_without_blocking(
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostic"
    result = run_stage(
        "fixture.diagnostic",
        output,
        lambda: (_ for _ in ()).throw(RuntimeError("diagnostic unavailable")),
        failure_is_diagnostic=True,
    )

    manifest = json.loads((output / "stage_manifest.json").read_text())
    assert result.status == "FAILED_DIAGNOSTIC"
    assert manifest["status"] == "FAILED"
    assert "diagnostic unavailable" in manifest["error"]


def test_stage_dag_accepts_branches_and_rejects_invalid_dependencies() -> None:
    validate_stage_dag(
        [
            StageContract("observations"),
            StageContract("static", ("observations",), diagnostic=True),
            StageContract("combined", ("observations",)),
            StageContract("report", ("static", "combined")),
        ]
    )

    with pytest.raises(ValueError, match="unknown dependencies"):
        validate_stage_dag([StageContract("report", ("missing",))])
    with pytest.raises(ValueError, match="cycle"):
        validate_stage_dag(
            [
                StageContract("first", ("second",)),
                StageContract("second", ("first",)),
            ]
        )

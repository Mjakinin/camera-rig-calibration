from __future__ import annotations

import json
from pathlib import Path

import pytest

from camera_rig_calibration import rerun
from camera_rig_calibration.runtime import PipelineOrchestrator


def _public_target(tmp_path: Path, *, input_id: str = "same-input") -> Path:
    target = tmp_path / "methods/ap01/baseline"
    target.mkdir(parents=True)
    (target / "RESULT.json").write_text(
        json.dumps(
            {
                "status": "available",
                "method_fingerprint": "old-method",
                "input_fingerprint": input_id,
            }
        ),
        encoding="utf-8",
    )
    return target


def test_changed_fingerprint_is_classified_as_nonmatching(tmp_path: Path) -> None:
    target = _public_target(tmp_path)
    orchestrator = object.__new__(PipelineOrchestrator)
    assert not orchestrator._matching_completed_execution(
        target,
        method_sha="new-method",
        input_id="same-input",
    )


def test_explicit_rerun_reaches_execution_and_preserves_old_target_on_failure(
    tmp_path: Path,
) -> None:
    target = _public_target(tmp_path)
    original = (target / "RESULT.json").read_bytes()
    orchestrator = PipelineOrchestrator(
        tmp_path, explicit_method_rerun=True
    )
    executed = False

    orchestrator._validate_conflicting_existing_target(
        target, input_id="same-input"
    )
    try:
        executed = True
        raise RuntimeError("stubbed method failure before publication")
    except RuntimeError as exc:
        assert "stubbed method failure" in str(exc)

    assert executed is True
    assert target.is_dir()
    assert (target / "RESULT.json").read_bytes() == original


def test_normal_execution_rejects_conflicting_target(tmp_path: Path) -> None:
    target = _public_target(tmp_path)
    orchestrator = PipelineOrchestrator(tmp_path)

    with pytest.raises(RuntimeError, match="Variant target exists"):
        orchestrator._validate_conflicting_existing_target(
            target, input_id="same-input"
        )


def test_explicit_rerun_rejects_dataset_mismatch(tmp_path: Path) -> None:
    target = _public_target(tmp_path, input_id="old-input")
    orchestrator = PipelineOrchestrator(
        tmp_path, explicit_method_rerun=True
    )

    with pytest.raises(RuntimeError, match="different immutable dataset"):
        orchestrator._validate_conflicting_existing_target(
            target, input_id="new-input"
        )


def test_explicit_rerun_rejects_prepared_dataset_identity_mismatch(
    tmp_path: Path,
) -> None:
    orchestrator = PipelineOrchestrator(
        tmp_path,
        explicit_method_rerun=True,
        rerun_metadata={"dataset_identity": {"fingerprint": "expected"}},
    )

    with pytest.raises(RuntimeError, match="exact immutable dataset identity"):
        orchestrator._validate_explicit_rerun_dataset_identity(
            {"fingerprint": "actual"}
        )

    orchestrator._validate_explicit_rerun_dataset_identity(
        {"fingerprint": "expected"}
    )


@pytest.mark.parametrize(
    ("runner_result", "expected_reconcile_calls"),
    [
        ({}, 0),
        ({"ap01_baseline": {"status": "failed_published"}}, 0),
        (
            {
                "ap01_baseline": {
                    "status": "completed",
                    "published": True,
                }
            },
            1,
        ),
    ],
)
def test_rerun_context_is_explicit_and_reconcile_waits_for_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner_result: dict[str, dict[str, object]],
    expected_reconcile_calls: int,
) -> None:
    entry = type("Entry", (), {"id": "ap01_baseline"})()
    prepared = type(
        "Prepared",
        (),
        {
            "queue": type("Queue", (), {"entries": [entry]})(),
            "reuse_intermediates_from": None,
            "dataset_identity": {"fingerprint": "dataset"},
            "observation_contract": {"observation_id": "detection"},
        },
    )()
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self, queue):
            return runner_result

    reconciled: list[Path] = []
    monkeypatch.setattr(rerun, "prepare_single_method_rerun", lambda **_: prepared)
    monkeypatch.setattr(rerun, "_latest_failed_attempt", lambda *_: None)
    monkeypatch.setattr(rerun, "QueueRunner", FakeRunner)
    monkeypatch.setattr(
        rerun, "reconcile_experiment", lambda path: reconciled.append(path)
    )

    rerun.run_single_method_rerun(
        repository_root=tmp_path,
        experiment=tmp_path / "experiment",
        method="ap01",
        variant="baseline",
        reuse_prepared_input=True,
        reuse_matching_intermediates=False,
        reconcile_after=True,
        ap01_method_contract="recommended_wizard_v1",
    )

    assert captured["explicit_method_rerun"] is True
    assert captured["rerun_metadata"]["ap01_baseline"][
        "reuse_published_observations"
    ] is True
    assert len(reconciled) == expected_reconcile_calls

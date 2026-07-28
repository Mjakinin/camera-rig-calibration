from __future__ import annotations

from pathlib import Path

import pytest

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.config.models import MethodSettings
from camera_rig_calibration.contracts import RunContext
from camera_rig_calibration.registry import calibration_methods


def test_split_ap03_legacy_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        MethodSettings.model_validate(
            {
                "enabled": ["ap03_single", "ap03_multi"],
                "ap03_single": {"scale_marker_id": 7},
                "ap03_multi": {"marker_ids": [7, 9]},
            }
        )


def test_ap03_component_declares_explicit_stage_commands(
    prepared_config, tmp_path: Path
) -> None:
    register_builtin_components()
    config = prepared_config.model_copy(
        update={"methods": MethodSettings(enabled=["ap03"])}, deep=True
    )
    context = RunContext(
        repository_root=Path(__file__).resolve().parents[1],
        config=config,
        dataset_root=config.dataset.prepared_root,
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run",
        resolved_ap03_single_scale_marker_id=7,
        resolved_ap03_multi_marker_ids=(7, 9),
    )

    commands = calibration_methods.get("ap03").commands(context)

    assert len(commands) == 6
    assert sum(command.stage_id == "ap03_reconstruct" for command in commands) == 1
    assert sum("scale" in command.stage_id for command in commands) == 2
    assert all(
        "camera_rig_calibration.methods.ap03" in " ".join(command.argv)
        for command in commands
    )


def test_ap03_scale_configuration_is_shared() -> None:
    settings = MethodSettings(enabled=["ap03"]).ap03
    assert settings.scale.reprojection_threshold_px == 5.0
    assert settings.scale.ransac_iterations == 1000
    assert settings.scale.minimum_inliers == 4
    assert not hasattr(settings.single, "minimum_area_px2")
    assert not hasattr(settings.multi, "ransac_iterations")

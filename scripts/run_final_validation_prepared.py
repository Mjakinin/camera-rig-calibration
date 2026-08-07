#!/usr/bin/env python3
"""Run the final validation strictly from already prepared Route-2 pixels.

This is a thin compatibility wrapper around ``run_final_validation.py``.
Published Route-2 datasets already contain authoritative CameraInfo JSON files.
When reusing those datasets, the moving-camera intrinsics must therefore remain
implicit during input preparation; PipelineOrchestrator resolves the CameraInfo
from the prepared dataset after finalization.

Keeping an explicit ``moving_camera.intrinsics`` path here would make the input
preparation layer treat the run as an intrinsics-composition override and create
a fresh composition root before the prepared pixels are materialized. For a
simulation-scene dataset the frame-diversity check then sees zero frames. This
wrapper prevents that non-scientific preparation artifact without changing any
image, marker, AP01/AP02/AP03, or evaluation setting.
"""

from __future__ import annotations

from pathlib import Path

import run_final_validation as validation
from camera_rig_calibration.config.models import RigConfig


_base_bind_existing_input = validation._bind_existing_input


def _bind_existing_input_without_intrinsics_override(
    config: RigConfig,
    prepared_root: Path,
) -> RigConfig:
    bound = _base_bind_existing_input(config, prepared_root)
    moving = bound.moving_camera.model_copy(
        update={
            "intrinsics": None,
            "intrinsics_profile": None,
            "intrinsic_calibration_video": None,
            "intrinsic_calibration_images": None,
        },
        deep=True,
    )
    return RigConfig.model_validate(
        bound.model_copy(
            update={"moving_camera": moving},
            deep=True,
        ).model_dump(mode="python")
    )


validation._bind_existing_input = _bind_existing_input_without_intrinsics_override


if __name__ == "__main__":
    raise SystemExit(validation.main())

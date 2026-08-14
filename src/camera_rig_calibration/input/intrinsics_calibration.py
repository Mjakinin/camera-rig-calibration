#!/usr/bin/env python3
"""Compatibility facade for the managed intrinsics-calibration workflow."""
from __future__ import annotations

from camera_rig_calibration.input.intrinsics_detection import (
    _detect_candidates,
    balanced_candidate_indices,
    board_metrics,
    detect_checkerboard_balanced,
    open_video_without_autorotation,
    select_diverse,
)
from camera_rig_calibration.input.intrinsics_reporting import write_contact_sheet
from camera_rig_calibration.input.intrinsics_solver import (
    calibrate,
    calibrate_with_outlier_filter,
    holdout_error,
    model_comparison,
    object_points,
)


def main() -> None:
    """Run the workflow using hooks currently installed on this facade."""
    from camera_rig_calibration.input import intrinsics_workflow

    intrinsics_workflow._detect_candidates = _detect_candidates
    intrinsics_workflow.balanced_candidate_indices = balanced_candidate_indices
    intrinsics_workflow.board_metrics = board_metrics
    intrinsics_workflow.detect_checkerboard_balanced = detect_checkerboard_balanced
    intrinsics_workflow.open_video_without_autorotation = open_video_without_autorotation
    intrinsics_workflow.select_diverse = select_diverse
    intrinsics_workflow.write_contact_sheet = write_contact_sheet
    intrinsics_workflow.calibrate_with_outlier_filter = calibrate_with_outlier_filter
    intrinsics_workflow.model_comparison = model_comparison
    intrinsics_workflow.object_points = object_points
    intrinsics_workflow.main()


if __name__ == "__main__":
    main()

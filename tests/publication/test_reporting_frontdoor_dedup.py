from __future__ import annotations

from camera_rig_calibration.evaluation.reporting_orchestration import (
    _without_common_anchor_6dof_sections,
)


def test_old_common_anchor_6dof_sections_are_removed_before_canonical_append() -> None:
    text = """SIMULATION CALIBRATION RESULTS
==============================

DETAILS
-------
keep me

COMMON-ANCHOR STATIC-CAMERA 6DOF EXPORTS
----------------------------------------
ap01/baseline:
  cam0: old

COMMON-ANCHOR STATIC-CAMERA 6DOF EXPORTS
----------------------------------------
ap03_multi/baseline:
  cam0: stale

RVIZ VISUALIZATION
------------------------------------------------------------------------
Status: OK
"""

    cleaned = _without_common_anchor_6dof_sections(text)

    assert "COMMON-ANCHOR STATIC-CAMERA 6DOF EXPORTS" not in cleaned
    assert "keep me" in cleaned
    assert "RVIZ VISUALIZATION" in cleaned
    assert "Status: OK" in cleaned

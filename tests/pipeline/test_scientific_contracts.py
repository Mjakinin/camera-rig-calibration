from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]


def _geometry_module():
    path = (
        REPOSITORY
        / "src/camera_rig_calibration/methods/common/geometry.py"
    )
    spec = importlib.util.spec_from_file_location("rigcal_characterized_geometry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transform_naming_contract_t_a_b_maps_b_coordinates_to_a() -> None:
    geometry = _geometry_module()
    t_a_b = geometry.make_T(np.eye(3), np.array([1.0, 0.0, 0.0]))
    t_b_c = geometry.make_T(np.eye(3), np.array([0.0, 2.0, 0.0]))
    point_c = np.array([0.0, 0.0, 3.0, 1.0])
    point_a = t_a_b @ t_b_c @ point_c
    np.testing.assert_allclose(point_a, [1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(geometry.invT(t_a_b) @ t_a_b, np.eye(4), atol=1e-12)

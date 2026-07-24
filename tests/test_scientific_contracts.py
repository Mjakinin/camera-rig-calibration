from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _geometry_module():
    path = REPOSITORY / "run/bus_real_data/_shared/common/geometry.py"
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


def test_real_3hz_baseline_characterization_is_preserved() -> None:
    paths = list(
        (
            REPOSITORY
            / "results/real_vehicle/real_05x_4k_3hz/legacy_results"
        ).glob(
            "*/marker_consistency/"
            "REAL_DATA_MARKER_CONSISTENCY_SUMMARY.json"
        )
    )
    if not paths:
        pytest.skip("Compact historical 3 Hz baseline is not available")
    path = paths[0]
    rows = {row["method"]: row for row in json.loads(path.read_text())}
    expected = {
        "AP01": (4, 232, 327.9087),
        "AP02": (4, 138, 1.2620),
        "AP03": (4, 232, 1.7372),
    }
    for method, (cameras, moving, cross_rmse) in expected.items():
        assert rows[method]["status"] == "OK"
        assert rows[method]["available_static_camera_count"] == cameras
        assert rows[method]["registered_moving_frames"] == moving
        assert rows[method]["moving_to_static_reprojection_rmse_px"] == pytest.approx(
            cross_rmse, abs=0.02
        )

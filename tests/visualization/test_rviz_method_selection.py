from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from camera_rig_calibration.ui.result_browser import (
    _available_rviz_methods,
    _parse_rviz_method_selection,
)
from camera_rig_calibration.visualization import _set_rviz_visible_methods


AVAILABLE = ("ap01", "ap02", "ap03_multi", "ap03_single")


def _write_anchor_variant(root: Path, method: str) -> None:
    path = root / "methods" / method / "baseline" / "camera_extrinsics_anchor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": method,
                "label": "baseline",
                "anchor_export_status": {"available": True},
                "cameras": [{"camera_id": "cam0"}],
            }
        ),
        encoding="utf-8",
    )


def test_rviz_discovery_offers_single_and_multi_but_not_shared_ap03_container(
    tmp_path: Path,
) -> None:
    for method in ("ap01", "ap02", "ap03", "ap03_multi", "ap03_single"):
        _write_anchor_variant(tmp_path, method)

    assert _available_rviz_methods(tmp_path) == AVAILABLE


def test_rviz_selection_all_and_ap03_alias() -> None:
    assert _parse_rviz_method_selection("all", AVAILABLE) == set(AVAILABLE)
    assert _parse_rviz_method_selection("ap03", AVAILABLE) == {
        "ap03_single",
        "ap03_multi",
    }
    assert _parse_rviz_method_selection("ap01, ap03_single", AVAILABLE) == {
        "ap01",
        "ap03_single",
    }


def test_rviz_selection_rejects_unknown_method() -> None:
    with pytest.raises(typer.BadParameter):
        _parse_rviz_method_selection("ap01,unknown", AVAILABLE)


def test_rviz_config_can_show_ap03_single_without_enabling_multi_aux_layers(
    tmp_path: Path,
) -> None:
    rviz = tmp_path / "rigcal_result.rviz"
    rviz.write_text(
        """Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/MarkerArray
      Name: ap03_multi/baseline
      Enabled: true
    - Class: rviz_default_plugins/MarkerArray
      Name: ap03_multi/baseline anchor edges
      Enabled: false
    - Class: rviz_default_plugins/MarkerArray
      Name: ap03_single/baseline
      Enabled: false
    - Class: rviz_default_plugins/MarkerArray
      Name: ap01/baseline
      Enabled: false
""",
        encoding="utf-8",
    )

    _set_rviz_visible_methods(rviz, {"ap03_single", "ap01"})
    text = rviz.read_text(encoding="utf-8")

    assert "Name: ap03_single/baseline\n      Enabled: true" in text
    assert "Name: ap01/baseline\n      Enabled: true" in text
    assert "Name: ap03_multi/baseline\n      Enabled: false" in text
    assert "Name: ap03_multi/baseline anchor edges\n      Enabled: false" in text

from __future__ import annotations

from pathlib import Path

from conftest import REPOSITORY_ROOT
from tools.check_source_layout import (
    root_module_violations,
    source_layout_violations,
)


PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "camera_rig_calibration"


def test_active_package_has_no_module_over_999_lines() -> None:
    assert source_layout_violations(PACKAGE_ROOT) == []


def test_active_package_has_no_loose_root_modules() -> None:
    assert root_module_violations(PACKAGE_ROOT) == []


def test_source_layout_accepts_modules_within_budget(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("first = 1\nsecond = 2\n")

    assert source_layout_violations(tmp_path, default_max_lines=2) == []


def test_source_layout_reports_every_oversized_module(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    first.write_text("one\ntwo\nthree\n", encoding="utf-8")
    second = tmp_path / "second.py"
    second.write_text("one\ntwo\n", encoding="utf-8")

    assert source_layout_violations(
        tmp_path,
        default_max_lines=1,
    ) == [
        (Path("first.py"), 3, 1),
        (Path("second.py"), 2, 1),
    ]


def test_root_layout_allows_only_python_entry_files(tmp_path: Path) -> None:
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "__main__.py").write_text("", encoding="utf-8")
    (tmp_path / "stray.py").write_text("value = 1\n", encoding="utf-8")

    assert root_module_violations(tmp_path) == [Path("stray.py")]

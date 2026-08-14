from __future__ import annotations

from pathlib import Path

from tools.check_source_layout import source_layout_violations


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "camera_rig_calibration"
)


def test_active_package_has_no_module_over_999_lines() -> None:
    assert source_layout_violations(PACKAGE_ROOT) == []


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

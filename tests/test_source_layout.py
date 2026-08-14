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

    assert source_layout_violations(
        tmp_path, default_max_lines=2, legacy_budgets={}
    ) == []


def test_source_layout_reports_default_and_legacy_growth(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.py"
    legacy.write_text("one\ntwo\nthree\n", encoding="utf-8")
    new = tmp_path / "new.py"
    new.write_text("one\ntwo\n", encoding="utf-8")

    assert source_layout_violations(
        tmp_path,
        default_max_lines=1,
        legacy_budgets={Path("legacy.py"): 2},
    ) == [
        (Path("legacy.py"), 3, 2),
        (Path("new.py"), 2, 1),
    ]

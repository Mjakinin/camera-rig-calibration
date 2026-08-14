from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MAX_LINES = 2_000

# These are explicit migration ceilings, not preferred module sizes. Lower a
# ceiling whenever the corresponding module is split; do not raise it to make
# the check pass.
LEGACY_MODULE_BUDGETS = {
    Path("wizard.py"): 6_043,
    Path("evaluation/reporting.py"): 3_666,
    Path("queueing.py"): 3_230,
    Path("runtime.py"): 2_479,
}


def source_layout_violations(
    package_root: Path,
    *,
    default_max_lines: int = DEFAULT_MAX_LINES,
    legacy_budgets: dict[Path, int] | None = None,
) -> list[tuple[Path, int, int]]:
    """Return Python modules whose line count exceeds its growth budget."""
    budgets = LEGACY_MODULE_BUDGETS if legacy_budgets is None else legacy_budgets
    violations: list[tuple[Path, int, int]] = []
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root)
        maximum = budgets.get(relative, default_max_lines)
        line_count = len(source.read_text(encoding="utf-8").splitlines())
        if line_count > maximum:
            violations.append((relative, line_count, maximum))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prevent new oversized Python modules and growth of explicitly "
            "budgeted legacy modules."
        )
    )
    parser.add_argument(
        "package_root",
        nargs="?",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "src"
            / "camera_rig_calibration"
        ),
    )
    args = parser.parse_args(argv)
    violations = source_layout_violations(args.package_root.resolve())
    if violations:
        print("Source layout failed: module growth budget exceeded.", file=sys.stderr)
        for path, actual, maximum in violations:
            print(
                f"  - {path}: {actual} lines (budget {maximum})",
                file=sys.stderr,
            )
        print(
            "Move the new responsibility into a focused module; do not raise "
            "a legacy budget.",
            file=sys.stderr,
        )
        return 1
    print(
        "Source layout OK: no module exceeds its maintained growth budget."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MAX_LINES = 999

# Kept as an injectable compatibility seam for the checker unit tests. The
# active package deliberately has no exceptions to the global limit.
LEGACY_MODULE_BUDGETS: dict[Path, int] = {}


def source_layout_violations(
    package_root: Path,
    *,
    default_max_lines: int = DEFAULT_MAX_LINES,
    legacy_budgets: dict[Path, int] | None = None,
) -> list[tuple[Path, int, int]]:
    """Return Python modules whose line count exceeds its size limit."""
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
            "Reject productive Python modules with more than 999 lines."
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
        print("Source layout failed: module size limit exceeded.", file=sys.stderr)
        for path, actual, maximum in violations:
            print(
                f"  - {path}: {actual} lines (budget {maximum})",
                file=sys.stderr,
            )
        print(
            "Move the responsibility into a focused module; the active "
            "package has no legacy exceptions.",
            file=sys.stderr,
        )
        return 1
    print(
        "Source layout OK: every productive module is at most 999 lines."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

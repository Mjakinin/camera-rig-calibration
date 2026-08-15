from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MAX_LINES = 999
ALLOWED_ROOT_MODULES = frozenset({"__init__.py", "__main__.py"})


def source_layout_violations(
    package_root: Path,
    *,
    default_max_lines: int = DEFAULT_MAX_LINES,
) -> list[tuple[Path, int, int]]:
    """Return Python modules whose line count exceeds the package limit."""
    violations: list[tuple[Path, int, int]] = []
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root)
        line_count = len(source.read_text(encoding="utf-8").splitlines())
        if line_count > default_max_lines:
            violations.append((relative, line_count, default_max_lines))
    return violations


def root_module_violations(package_root: Path) -> list[Path]:
    """Reject loose implementation modules at the package root."""
    return sorted(
        source.relative_to(package_root)
        for source in package_root.glob("*.py")
        if source.name not in ALLOWED_ROOT_MODULES
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reject oversized modules and loose Python implementation files "
            "at the camera_rig_calibration package root."
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
    package_root = args.package_root.resolve()
    size_violations = source_layout_violations(package_root)
    root_violations = root_module_violations(package_root)
    if size_violations or root_violations:
        print("Source layout failed.", file=sys.stderr)
        for path, actual, maximum in size_violations:
            print(
                f"  - {path}: {actual} lines (budget {maximum})",
                file=sys.stderr,
            )
        for path in root_violations:
            print(
                f"  - {path}: loose package-root module; move it into a "
                "focused package and expose compatibility centrally",
                file=sys.stderr,
            )
        return 1
    print(
        "Source layout OK: modules are within budget and package-root Python "
        "files are limited to __init__.py/__main__.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

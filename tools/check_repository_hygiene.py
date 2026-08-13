from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


MAX_TRACKED_FILE_BYTES = 40 * 1024 * 1024
_LFS_ATTRIBUTE = re.compile(r"(?:^|\s)(?:filter|diff|merge)=lfs(?:\s|$)")


def _tracked_files(repository_root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        repository_root / raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw
    ]


def _human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def _lfs_violations(
    repository_root: Path, tracked_files: list[Path]
) -> list[str]:
    violations: list[str] = []
    for path in tracked_files:
        relative = path.relative_to(repository_root)
        if relative.name == ".lfsconfig":
            violations.append(f"{relative}: Git LFS configuration is not allowed")
            continue
        if relative.name != ".gitattributes" or not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _LFS_ATTRIBUTE.search(stripped):
                violations.append(
                    f"{relative}:{line_number}: Git LFS attribute is not allowed: "
                    f"{stripped}"
                )
    return violations


def check_repository(
    repository_root: Path,
    *,
    maximum_bytes: int = MAX_TRACKED_FILE_BYTES,
) -> tuple[list[tuple[Path, int]], list[str], list[tuple[Path, int]]]:
    tracked_files = _tracked_files(repository_root)
    sized_files: list[tuple[Path, int]] = []
    for path in tracked_files:
        if not path.is_file():
            continue
        sized_files.append((path.relative_to(repository_root), path.stat().st_size))

    oversized = sorted(
        ((path, size) for path, size in sized_files if size > maximum_bytes),
        key=lambda item: item[1],
        reverse=True,
    )
    largest = sorted(sized_files, key=lambda item: item[1], reverse=True)[:5]
    return oversized, _lfs_violations(repository_root, tracked_files), largest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject Git LFS metadata and oversized tracked repository files."
    )
    parser.add_argument(
        "repository",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of tools/)",
    )
    parser.add_argument(
        "--maximum-mib",
        type=float,
        default=MAX_TRACKED_FILE_BYTES / (1024 * 1024),
        help="maximum tracked file size in MiB",
    )
    args = parser.parse_args(argv)

    repository_root = args.repository.resolve()
    maximum_bytes = int(args.maximum_mib * 1024 * 1024)
    oversized, lfs_violations, largest = check_repository(
        repository_root,
        maximum_bytes=maximum_bytes,
    )

    if lfs_violations:
        print("Repository hygiene failed: Git LFS is not permitted.", file=sys.stderr)
        for violation in lfs_violations:
            print(f"  - {violation}", file=sys.stderr)

    if oversized:
        print(
            "Repository hygiene failed: tracked files exceed "
            f"{_human_size(maximum_bytes)}.",
            file=sys.stderr,
        )
        for path, size in oversized:
            print(f"  - {path}: {_human_size(size)}", file=sys.stderr)

    if lfs_violations or oversized:
        print(
            "Keep generated/raw artifacts outside Git or reduce them before commit; "
            "do not solve this with Git LFS.",
            file=sys.stderr,
        )
        return 1

    print(
        "Repository hygiene OK: Git LFS is not configured and no tracked file "
        f"exceeds {_human_size(maximum_bytes)}."
    )
    if largest:
        print("Largest tracked files:")
        for path, size in largest:
            print(f"  - {path}: {_human_size(size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

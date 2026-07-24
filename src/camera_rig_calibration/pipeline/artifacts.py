from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def require_file(path: Path, *, label: str | None = None) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Missing {label or 'stage input'}: {resolved}")
    return resolved


def require_directory(path: Path, *, label: str | None = None) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"Missing {label or 'stage input'}: {resolved}")
    return resolved


def require_any(paths: Iterable[Path], *, label: str) -> tuple[Path, ...]:
    available = tuple(path.resolve() for path in paths if path.exists())
    if not available:
        raise RuntimeError(f"No {label} artifacts were found")
    return available


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

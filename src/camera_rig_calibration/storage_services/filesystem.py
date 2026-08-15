"""Small, cross-platform filesystem operations used by rigcal storage."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


def rename_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 8,
) -> None:
    """Rename despite short-lived locks on WSL-mounted Windows storage."""
    for attempt in range(attempts):
        try:
            source.rename(target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.1 * (2**attempt), 1.0))


def promote_directory(
    source: Path,
    target: Path,
    *,
    attempts: int = 8,
) -> str:
    """Promote a complete directory, copying if Windows blocks its rename.

    A successful copy deliberately leaves ``source`` in place as execution
    evidence.  The caller may clean that cache later, but publication must not
    fail merely because Windows Explorer, Search, or antivirus briefly holds a
    directory handle.
    """
    if not source.is_dir():
        raise FileNotFoundError(f"Promotion source is missing: {source}")
    if target.exists():
        raise FileExistsError(f"Promotion destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        rename_with_retry(source, target, attempts=attempts)
    except PermissionError:
        if target.exists():
            raise RuntimeError(
                "Promotion destination appeared while recovering from a "
                f"filesystem lock: {target}"
            )
        shutil.copytree(source, target)
        return "copied_after_locked_rename"
    return "renamed"

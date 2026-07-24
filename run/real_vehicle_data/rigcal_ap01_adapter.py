#!/usr/bin/env python3
"""Structured-progress adapter for the unchanged AP01 real-data entry point."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def _argument_value(arguments: list[str], name: str) -> str | None:
    try:
        return arguments[arguments.index(name) + 1]
    except (ValueError, IndexError):
        return None


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    core = Path(__file__).with_name("07_run_ap01_real.py")
    arguments = sys.argv[1:]
    output_text = _argument_value(arguments, "--out")
    if output_text is None:
        raise RuntimeError("AP01 adapter requires --out")
    output = Path(output_text).resolve()
    started = time.monotonic()
    print("RIGCAL_STAGE_START ap01_estimation", flush=True)
    process = subprocess.Popen(
        [sys.executable, str(core), *arguments],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"AP01 scientific entry point exited with {returncode}")
    status_path = output / "METHOD_STATUS.json"
    if not status_path.is_file():
        raise RuntimeError("AP01 completed without METHOD_STATUS.json")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("success", False):
        raise RuntimeError(
            f"AP01 status validation failed: {status.get('status', 'unknown')}"
        )
    print(
        "RIGCAL_STAGE_END ap01_estimation "
        f"elapsed_seconds={time.monotonic() - started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

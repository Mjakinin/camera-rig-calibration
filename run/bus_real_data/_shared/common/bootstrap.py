#!/usr/bin/env python3
from pathlib import Path
import sys


def project_root_from_file(file_path: str) -> Path:
    p = Path(file_path).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "run" / "bus_real_data").is_dir() and (parent / "results").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {file_path}")


def add_bus_real_data_to_sys_path(file_path: str) -> Path:
    root = project_root_from_file(file_path)
    bus_run = root / "run" / "bus_real_data"
    if str(bus_run) not in sys.path:
        sys.path.insert(0, str(bus_run))
    return root

from __future__ import annotations

import importlib.util
import importlib
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_checks(repository_root: Path, *, needs_ros: bool = False) -> list[Check]:
    version = sys.version_info
    checks = [
        Check(
            "Python",
            (3, 10) <= version[:2] < (3, 14),
            platform.python_version(),
        ),
        Check("Repository", (repository_root / ".git").exists(), str(repository_root)),
        Check(
            "Legacy method runners",
            all(
                (repository_root / path).is_file()
                for path in (
                    "run/real_vehicle_data/07_run_ap01_real.py",
                    "run/real_vehicle_data/08_run_ap02_real.py",
                    "run/real_vehicle_data/09_run_ap03_real.py",
                )
            ),
            "AP01/AP02/AP03 wrappers",
        ),
    ]
    for module, required in (
        ("pydantic", True),
        ("yaml", True),
        ("typer", True),
        ("rich", True),
        ("numpy", True),
        ("scipy", True),
        ("cv2", True),
    ):
        available = importlib.util.find_spec(module) is not None
        checks.append(Check(f"Python module: {module}", available, "installed" if available else "missing", required))
    colmap = shutil.which("colmap")
    checks.append(Check("COLMAP", colmap is not None, colmap or "not found in PATH"))
    if needs_ros:
        ros2 = shutil.which("ros2")
        checks.append(Check("ROS 2", ros2 is not None, ros2 or "not found in PATH"))
        checks.append(
            Check(
                "rosbag2_py",
                importlib.util.find_spec("rosbag2_py") is not None,
                "installed" if importlib.util.find_spec("rosbag2_py") else "missing",
            )
        )
        readers: set[str] = set()
        if importlib.util.find_spec("rosbag2_py") is not None:
            try:
                rosbag2_py = importlib.import_module("rosbag2_py")
                readers = set(rosbag2_py.get_registered_readers())
            except (ImportError, AttributeError, RuntimeError):
                readers = set()
        checks.append(
            Check(
                "ROS 2 MCAP storage",
                "mcap" in readers,
                (
                    "mcap reader installed"
                    if "mcap" in readers
                    else "missing: install ros-humble-rosbag2-storage-mcap"
                ),
            )
        )
        ignition = shutil.which("ign") or shutil.which("gz")
        checks.append(
            Check(
                "Gazebo / Ignition",
                ignition is not None,
                ignition or "ign/gz not found in PATH",
                required=False,
            )
        )
        checks.append(
            Check(
                "cv_bridge",
                importlib.util.find_spec("cv_bridge") is not None,
                "installed" if importlib.util.find_spec("cv_bridge") else "missing",
                required=False,
            )
        )
    git_lfs = shutil.which("git-lfs") or shutil.which("git")
    checks.append(Check("Git", git_lfs is not None, git_lfs or "not found"))
    return checks

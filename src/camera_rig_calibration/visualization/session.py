from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOMAIN_START = 101
DOMAIN_END = 199


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        exit_code = wintypes.DWORD()
        try:
            if not kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code)
            ):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reserve_domain(session_root: Path) -> tuple[int, Path]:
    session_root.mkdir(parents=True, exist_ok=True)
    for domain in range(DOMAIN_START, DOMAIN_END + 1):
        lock = session_root / f"domain_{domain}.json"
        if lock.is_file():
            try:
                owner = json.loads(lock.read_text(encoding="utf-8"))
                pid = int(owner.get("pid", -1))
            except (OSError, ValueError, json.JSONDecodeError):
                pid = -1
            if _pid_alive(pid):
                continue
            lock.unlink(missing_ok=True)
        try:
            descriptor = os.open(
                lock,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "reserved_at": _now(),
                    "domain_id": domain,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        return domain, lock
    raise RuntimeError("No isolated ROS_DOMAIN_ID is available in 101..199")


def _check_ros() -> tuple[bool, str]:
    if shutil.which("rviz2") is None:
        return False, "rviz2 is not on PATH; source the ROS 2 environment."
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import rclpy, sensor_msgs.msg, geometry_msgs.msg, "
                "visualization_msgs.msg, std_msgs.msg, "
                "builtin_interfaces.msg, tf2_ros"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        return (
            False,
            "ROS 2 Python packages are unavailable in this shell: "
            + (probe.stderr.strip() or "import failed"),
        )
    return True, ""


def launch_isolated_rviz(
    experiment_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    from .scene import ensure_visualization_artifacts

    manifest = ensure_visualization_artifacts(experiment_root)
    if not manifest.get("available"):
        raise RuntimeError(str(manifest.get("reason") or manifest.get("status")))
    available, reason = _check_ros()
    if not available:
        raise RuntimeError(reason)
    session_root = repository_root.resolve() / "workspace" / "rviz_sessions"
    domain, lock = _reserve_domain(session_root)
    session_id = f"{experiment_root.name}_{int(time.time())}_{domain}"
    log = session_root / f"{session_id}.log"
    session_manifest = session_root / f"{session_id}.json"
    command = [
        sys.executable,
        "-m",
        "camera_rig_calibration.visualization.session",
        "--supervise",
        "--experiment",
        str(experiment_root.resolve()),
        "--domain",
        str(domain),
        "--lock",
        str(lock),
        "--session-manifest",
        str(session_manifest),
    ]
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _write_json(
        lock,
        {
            "pid": process.pid,
            "domain_id": domain,
            "session_id": session_id,
            "experiment": str(experiment_root.resolve()),
            "reserved_at": _now(),
        },
    )
    payload = {
        "status": "running",
        "session_id": session_id,
        "pid": process.pid,
        "ros_domain_id": domain,
        "experiment": str(experiment_root.resolve()),
        "log": str(log.resolve()),
        "manifest": str(session_manifest.resolve()),
        "started_at": _now(),
    }
    _write_json(session_manifest, payload)
    return payload


def _supervise(args: argparse.Namespace) -> int:
    experiment = Path(args.experiment).resolve()
    visualization = experiment / "visualization"
    lock = Path(args.lock).resolve()
    session_manifest = Path(args.session_manifest).resolve()
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(args.domain)
    publisher = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "camera_rig_calibration.visualization.ros_scene",
            "--visualization",
            str(visualization),
        ],
        env=environment,
    )
    rviz = subprocess.Popen(
        [
            "rviz2",
            "-d",
            str(visualization / "rigcal_result.rviz"),
        ],
        env=environment,
    )
    _write_json(
        session_manifest,
        {
            **(
                json.loads(session_manifest.read_text(encoding="utf-8"))
                if session_manifest.is_file()
                else {}
            ),
            "status": "running",
            "supervisor_pid": os.getpid(),
            "publisher_pid": publisher.pid,
            "rviz_pid": rviz.pid,
        },
    )
    return_code = 1
    try:
        return_code = rviz.wait()
    finally:
        if publisher.poll() is None:
            publisher.terminate()
            try:
                publisher.wait(timeout=5)
            except subprocess.TimeoutExpired:
                publisher.kill()
        lock.unlink(missing_ok=True)
        _write_json(
            session_manifest,
            {
                **json.loads(
                    session_manifest.read_text(encoding="utf-8")
                ),
                "status": "closed",
                "rviz_return_code": return_code,
                "closed_at": _now(),
            },
        )
    return return_code


def main_for_generated_scene(script_path: str) -> None:
    experiment = Path(script_path).resolve().parent.parent
    repository = next(
        (
            parent
            for parent in experiment.parents
            if (parent / "pyproject.toml").is_file()
        ),
        Path.cwd(),
    )
    payload = launch_isolated_rviz(experiment, repository)
    print(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--experiment")
    parser.add_argument("--domain", type=int)
    parser.add_argument("--lock")
    parser.add_argument("--session-manifest")
    args = parser.parse_args()
    if args.supervise:
        return _supervise(args)
    parser.error("Use the rigcal result viewer to start an RViz session.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

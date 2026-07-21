#!/usr/bin/env python3

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MIB = 1024 * 1024
WARNING_LIMIT = 50 * MIB
HARD_LIMIT = 100 * MIB

repo = Path(__file__).resolve().parents[1]

obj_relative = (
    "src/calib_lab/bus_real_data/models/"
    "beintelli_bus/meshes/obj/"
    "beintelli_erklarbus.obj"
)

model_prefix = (
    "src/calib_lab/bus_real_data/models/"
    "beintelli_bus/"
)

real_root = "results/real_vehicle_data/"

allowed_real_prefixes = (
    "results/real_vehicle_data/INTRINSIC_RESULTS/iphone_05x_4k/",
    "results/real_vehicle_data/real_05x_4k_1hz/00_shared_input/",
    "results/real_vehicle_data/real_05x_4k_1hz/99_FINAL_RESULTS/",
    "results/real_vehicle_data/real_05x_4k_3hz/99_FINAL_RESULTS/",
    "results/real_vehicle_data/real_05x_4k_5hz/99_FINAL_RESULTS/",
)

allowed_real_exact = {
    "results/real_vehicle_data/real_05x_4k_1hz/EXPERIMENT_CONFIG.txt",
    "results/real_vehicle_data/real_05x_4k_3hz/EXPERIMENT_CONFIG.txt",
    "results/real_vehicle_data/real_05x_4k_5hz/EXPERIMENT_CONFIG.txt",
}

debug_prefix = (
    "results/real_vehicle_data/"
    "real_05x_4k_1hz/00_shared_input/"
    "aruco_observations/debug_images/moving/"
)

forbidden_geometry_suffixes = (
    ".dae",
    ".stl",
    ".fbx",
    ".glb",
    ".gltf",
    ".mesh",
)

expected_lfs_rule = (
    obj_relative
    + " filter=lfs diff=lfs merge=lfs -text"
)

errors: list[str] = []
warnings: list[str] = []

attributes = (repo / ".gitattributes").read_text(
    encoding="utf-8"
)

lfs_rules = [
    line.strip()
    for line in attributes.splitlines()
    if "filter=lfs" in line
]

if lfs_rules != [expected_lfs_rule]:
    errors.append(
        "expected exactly one LFS rule for the bus OBJ"
    )

raw_index = subprocess.check_output(
    ["git", "ls-files", "-s", "-z"],
    cwd=repo,
)

entries: dict[str, tuple[str, int]] = {}

for raw_record in raw_index.split(b"\0"):
    if not raw_record:
        continue

    record = raw_record.decode(
        "utf-8",
        errors="surrogateescape",
    )

    metadata, relative = record.split("\t", 1)
    _mode, object_id, stage = metadata.split()

    if stage != "0":
        errors.append(
            f"non-zero index stage for {relative}: {stage}"
        )
        continue

    size = int(
        subprocess.check_output(
            ["git", "cat-file", "-s", object_id],
            cwd=repo,
            text=True,
        ).strip()
    )

    entries[relative] = (object_id, size)


def blob_content(object_id: str) -> bytes:
    return subprocess.check_output(
        ["git", "cat-file", "-p", object_id],
        cwd=repo,
    )


def real_path_allowed(relative: str) -> bool:
    tail = relative[len(real_root):]

    if "/" not in tail:
        return True

    if relative.startswith(debug_prefix):
        return False

    if relative in allowed_real_exact:
        return True

    return relative.startswith(allowed_real_prefixes)


frame_count = 0
pointer_paths: list[str] = []

for relative, (object_id, object_size) in entries.items():
    if (
        relative.startswith(model_prefix)
        and relative != obj_relative
        and relative.lower().endswith(
            forbidden_geometry_suffixes
        )
    ):
        errors.append(
            f"unused alternative bus geometry tracked: {relative}"
        )

    if (
        relative.startswith(real_root)
        and not real_path_allowed(relative)
    ):
        errors.append(
            f"disallowed real-vehicle payload: {relative}"
        )

    if (
        relative.startswith(
            "results/real_vehicle_data/"
            "real_05x_4k_1hz/00_shared_input/"
            "raw_images/moving/frame_"
        )
        and relative.endswith(".png")
    ):
        frame_count += 1

    if object_size <= 1024:
        content = blob_content(object_id)

        if content.startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            pointer_paths.append(relative)

    if relative == obj_relative:
        content = blob_content(object_id)

        if not content.startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            errors.append(
                "bus OBJ is not stored as a Git LFS pointer"
            )
            continue

        match = re.search(
            rb"(?:^|\n)size ([0-9]+)(?:\n|$)",
            content,
        )

        if not match:
            errors.append(
                "bus OBJ LFS pointer has no declared size"
            )
        elif int(match.group(1)) < HARD_LIMIT:
            errors.append(
                "bus OBJ pointer declares an unexpectedly small file"
            )

        continue

    if object_size >= HARD_LIMIT:
        errors.append(
            f"{object_size / MIB:.2f} MiB regular Git object: "
            f"{relative}"
        )
    elif object_size >= WARNING_LIMIT:
        warnings.append(
            f"{object_size / MIB:.2f} MiB regular Git object: "
            f"{relative}"
        )

if pointer_paths != [obj_relative]:
    errors.append(
        "LFS pointers must consist solely of the bus OBJ; found: "
        + ", ".join(pointer_paths)
    )

if obj_relative not in entries:
    errors.append("bus OBJ is not tracked")

if frame_count != 78:
    errors.append(
        f"expected 78 one-Hz moving frames, found {frame_count}"
    )

model_path = repo / (
    model_prefix + "model.sdf"
)

obj_uri = (
    "model://beintelli_bus/"
    "meshes/obj/beintelli_erklarbus.obj"
)

if not model_path.is_file():
    errors.append("bus model.sdf is missing")
elif obj_uri not in model_path.read_text(encoding="utf-8"):
    errors.append("bus model.sdf does not reference the OBJ")

print(f"Tracked files checked: {len(entries)}")
print(f"One-Hz moving frames: {frame_count}")
print(f"Git LFS pointers: {len(pointer_paths)}")

for warning in warnings:
    print(f"[WARNING] {warning}")

if errors:
    for error in errors:
        print(f"[ERROR] {error}")

    raise SystemExit(1)

print("[OK] Only the bus OBJ uses Git LFS.")
print("[OK] No other regular Git object reaches 100 MiB.")
print("[OK] Real-vehicle payload policy satisfied.")
print("[OK] Bus model references the OBJ.")
print("[OK] No unused alternative bus geometry is tracked.")

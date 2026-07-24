#!/usr/bin/env python3

from __future__ import annotations

import hashlib
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
obj_archive_relative = obj_relative + ".gz"
obj_archive_size = 31_209_135
obj_archive_sha256 = (
    "d435f33474c3fb8b151adf2904f9355a5c0d3aa381975d897f73c2e11a3e4f6f"
)

model_prefix = (
    "src/calib_lab/bus_real_data/models/"
    "beintelli_bus/"
)

legacy_result_roots = (
    "results/bus_real_data/",
    "results/real_vehicle_data/",
)

scientific_result_roots = (
    "results/simulation/",
    "results/real_vehicle/",
)

# Dataset payloads, calibration previews and transient working copies belong in
# datasets/, data_local/ or workspace/temporary_runs/, never in published results.
forbidden_result_directories = {
    ".staging",
    "debug_gallery",
    "debug_images",
    "diagnostics",
    "inputs",
    "observations",
    "raw_images",
    "selected_frames",
    "working_images",
    "workspace",
}

baseline_receipt = (
    "results/simulation/baseline/route2/PUBLISHED.json"
)

forbidden_geometry_suffixes = (
    ".dae",
    ".stl",
    ".fbx",
    ".glb",
    ".gltf",
    ".mesh",
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

if lfs_rules:
    errors.append("Git LFS rules are not allowed; use regular-Git assets")

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


pointer_paths: list[str] = []
published_simulation_experiments = 0

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

    if relative.startswith(legacy_result_roots):
        errors.append(
            f"legacy result tree still tracked after schema-v5 migration: {relative}"
        )

    if relative.startswith(scientific_result_roots):
        result_parts = set(Path(relative).parts)
        forbidden_parts = sorted(
            result_parts & forbidden_result_directories
        )
        if forbidden_parts:
            errors.append(
                "generated input/debug payload tracked in scientific results "
                f"({', '.join(forbidden_parts)}): {relative}"
            )

    if (
        relative.startswith("results/simulation/")
        and relative.endswith("/PUBLISHED.json")
    ):
        published_simulation_experiments += 1

    if object_size <= 1024:
        content = blob_content(object_id)

        if content.startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            pointer_paths.append(relative)

    if relative == obj_relative:
        errors.append(
            "generated bus OBJ must not be tracked; track its .obj.gz archive"
        )

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

if pointer_paths:
    errors.append(
        "Git LFS pointers are not allowed; found: "
        + ", ".join(pointer_paths)
    )

if obj_archive_relative not in entries:
    errors.append("compressed bus OBJ archive is not tracked")
else:
    archive_id, archive_size = entries[obj_archive_relative]
    if archive_size != obj_archive_size:
        errors.append(
            "compressed bus OBJ archive has unexpected size: "
            f"{archive_size} != {obj_archive_size}"
        )
    archive_content = blob_content(archive_id)
    archive_hash = hashlib.sha256(archive_content).hexdigest()
    if archive_hash != obj_archive_sha256:
        errors.append(
            "compressed bus OBJ archive has unexpected SHA-256: "
            f"{archive_hash}"
        )

if baseline_receipt not in entries:
    errors.append(
        "schema-v5 Route-2 baseline publication receipt is missing"
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
print(
    "Published simulation experiments: "
    f"{published_simulation_experiments}"
)
print(f"Git LFS pointers: {len(pointer_paths)}")

for warning in warnings:
    print(f"[WARNING] {warning}")

if errors:
    for error in errors:
        print(f"[ERROR] {error}")

    raise SystemExit(1)

print("[OK] Repository contains no Git LFS pointers or rules.")
print("[OK] Bus OBJ is stored as a verified regular-Git gzip archive.")
print("[OK] No other regular Git object reaches 100 MiB.")
print("[OK] Schema-v5 results contain scientific outputs only.")
print("[OK] Legacy result trees are no longer tracked.")
print("[OK] Bus model references the OBJ.")
print("[OK] No unused alternative bus geometry is tracked.")

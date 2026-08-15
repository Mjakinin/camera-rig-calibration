from __future__ import annotations

import gzip
import hashlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


BUS_MESH_RELATIVE_PATH = Path(
    "src/calib_lab/bus_real_data/models/beintelli_bus/meshes/obj/"
    "beintelli_erklarbus.obj"
)
BUS_MESH_ARCHIVE_RELATIVE_PATH = BUS_MESH_RELATIVE_PATH.with_suffix(".obj.gz")
BUS_MESH_SHA256 = (
    "4ef6d1ece4930395523006996e88895a0f3a5e165dbc3ceae7ce9e5aaab81950"
)
BUS_MESH_SIZE_BYTES = 160_728_894
BUS_MESH_ARCHIVE_SHA256 = (
    "d435f33474c3fb8b151adf2904f9355a5c0d3aa381975d897f73c2e11a3e4f6f"
)


@dataclass(frozen=True)
class MaterializedAsset:
    path: Path
    created: bool
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_gzip_asset(
    archive: Path,
    destination: Path,
    *,
    expected_archive_sha256: str,
    expected_sha256: str,
    expected_size_bytes: int,
    announce: Callable[[str], None] | None = None,
) -> MaterializedAsset:
    """Atomically materialize and verify one regular-Git gzip asset."""
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size_bytes
        and sha256_file(destination) == expected_sha256
    ):
        return MaterializedAsset(
            path=destination,
            created=False,
            sha256=expected_sha256,
            size_bytes=expected_size_bytes,
        )

    if not archive.is_file():
        raise RuntimeError(
            "Required compressed simulation asset is missing: "
            f"{archive}. Restore the tracked .obj.gz file from Git."
        )
    archive_sha256 = sha256_file(archive)
    if archive_sha256 != expected_archive_sha256:
        raise RuntimeError(
            "Compressed simulation asset failed its SHA-256 check: "
            f"{archive} (expected {expected_archive_sha256}, got {archive_sha256})"
        )

    if announce is not None:
        announce(
            "[SETUP] Materializing the BeIntelli bus mesh once "
            f"({archive.stat().st_size / 1_000_000:.1f} MB compressed -> "
            f"{expected_size_bytes / 1_000_000:.1f} MB OBJ)..."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            digest = hashlib.sha256()
            size = 0
            with gzip.open(archive, "rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            output.flush()
            os.fsync(output.fileno())

        actual_sha256 = digest.hexdigest()
        if size != expected_size_bytes or actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Materialized simulation asset failed verification: "
                f"expected {expected_size_bytes} bytes/{expected_sha256}, "
                f"got {size} bytes/{actual_sha256}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if announce is not None:
        announce(f"[OK] Bus mesh ready: {destination}")
    return MaterializedAsset(
        path=destination,
        created=True,
        sha256=expected_sha256,
        size_bytes=expected_size_bytes,
    )


def ensure_bus_mesh(
    repository_root: Path,
    *,
    announce: Callable[[str], None] | None = None,
) -> MaterializedAsset:
    repository = repository_root.resolve()
    return materialize_gzip_asset(
        repository / BUS_MESH_ARCHIVE_RELATIVE_PATH,
        repository / BUS_MESH_RELATIVE_PATH,
        expected_archive_sha256=BUS_MESH_ARCHIVE_SHA256,
        expected_sha256=BUS_MESH_SHA256,
        expected_size_bytes=BUS_MESH_SIZE_BYTES,
        announce=announce,
    )

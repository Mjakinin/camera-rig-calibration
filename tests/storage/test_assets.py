from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from camera_rig_calibration.assets import materialize_gzip_asset


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _archive(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output) as stream:
            stream.write(payload)


def test_materialize_gzip_asset_is_verified_atomic_and_reusable(
    tmp_path: Path,
) -> None:
    payload = (b"v 0 0 0\nf 1 1 1\n" * 1000)
    archive = tmp_path / "mesh.obj.gz"
    destination = tmp_path / "mesh.obj"
    _archive(archive, payload)

    first = materialize_gzip_asset(
        archive,
        destination,
        expected_archive_sha256=_sha256(archive.read_bytes()),
        expected_sha256=_sha256(payload),
        expected_size_bytes=len(payload),
    )
    first_mtime = destination.stat().st_mtime_ns
    second = materialize_gzip_asset(
        archive,
        destination,
        expected_archive_sha256=_sha256(archive.read_bytes()),
        expected_sha256=_sha256(payload),
        expected_size_bytes=len(payload),
    )

    assert first.created is True
    assert second.created is False
    assert destination.read_bytes() == payload
    assert destination.stat().st_mtime_ns == first_mtime


def test_bad_archive_does_not_replace_existing_destination(tmp_path: Path) -> None:
    archive = tmp_path / "mesh.obj.gz"
    destination = tmp_path / "mesh.obj"
    destination.write_bytes(b"existing-invalid-file")
    _archive(archive, b"new mesh")

    with pytest.raises(RuntimeError, match="SHA-256"):
        materialize_gzip_asset(
            archive,
            destination,
            expected_archive_sha256="0" * 64,
            expected_sha256="1" * 64,
            expected_size_bytes=8,
        )

    assert destination.read_bytes() == b"existing-invalid-file"
    assert not list(tmp_path.glob(".mesh.obj.*.tmp"))

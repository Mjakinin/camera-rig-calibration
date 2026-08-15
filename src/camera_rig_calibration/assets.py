"""Compatibility facade for repository asset materialization helpers."""

from .storage_services import assets as _impl
from .storage_services.assets import (
    BUS_MESH_ARCHIVE_RELATIVE_PATH,
    BUS_MESH_ARCHIVE_SHA256,
    BUS_MESH_RELATIVE_PATH,
    BUS_MESH_SHA256,
    BUS_MESH_SIZE_BYTES,
    MaterializedAsset,
    ensure_bus_mesh,
    materialize_gzip_asset,
    sha256_file,
)

__all__ = [
    "BUS_MESH_ARCHIVE_RELATIVE_PATH",
    "BUS_MESH_ARCHIVE_SHA256",
    "BUS_MESH_RELATIVE_PATH",
    "BUS_MESH_SHA256",
    "BUS_MESH_SIZE_BYTES",
    "MaterializedAsset",
    "ensure_bus_mesh",
    "materialize_gzip_asset",
    "sha256_file",
]


def __getattr__(name: str):
    return getattr(_impl, name)

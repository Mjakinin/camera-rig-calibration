"""Compatibility facade for dataset acquisition identity helpers."""

from .dataset import identity as _impl
from .dataset.identity import (
    CONTRACT_VERSION,
    build_dataset_identity,
    identities_match,
    write_dataset_identity,
)

__all__ = [
    "CONTRACT_VERSION",
    "build_dataset_identity",
    "identities_match",
    "write_dataset_identity",
]


def __getattr__(name: str):
    return getattr(_impl, name)

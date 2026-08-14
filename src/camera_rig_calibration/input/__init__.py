"""Public input helpers, loaded lazily to avoid package import cycles."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .preparation import (
        PreparationPlan,
        build_preparation_plan,
        finalize_dataset,
    )
    from .topics import (
        McapTopic,
        RosbagSource,
        list_mcap_topics,
        resolve_rosbag_source,
    )


_PUBLIC_MODULES = {
    "PreparationPlan": ".preparation",
    "build_preparation_plan": ".preparation",
    "finalize_dataset": ".preparation",
    "McapTopic": ".topics",
    "RosbagSource": ".topics",
    "list_mcap_topics": ".topics",
    "resolve_rosbag_source": ".topics",
}


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "McapTopic",
    "RosbagSource",
    "PreparationPlan",
    "build_preparation_plan",
    "finalize_dataset",
    "list_mcap_topics",
    "resolve_rosbag_source",
]

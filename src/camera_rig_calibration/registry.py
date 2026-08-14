from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

from .contracts import (
    CalibrationMethod,
    Evaluator,
    ExperimentProvider,
    InputAdapter,
)


T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    """Ordered component registry with explicit duplicate protection."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, component: T, *, replace: bool = False) -> T:
        component_id = str(getattr(component, "id", "")).strip()
        if not component_id:
            raise ValueError(f"{self.kind} components must define a non-empty id")
        if component_id in self._items and not replace:
            raise ValueError(f"duplicate {self.kind} id: {component_id}")
        if self.kind == "calibration method":
            from pydantic import BaseModel

            from .method_sdk.contracts import method_metadata

            config_model = getattr(component, "config_model", None)
            if not (
                isinstance(config_model, type)
                and issubclass(config_model, BaseModel)
            ):
                raise TypeError(
                    f"calibration method '{component_id}' needs a Pydantic config_model"
                )
            metadata = method_metadata(component)
            if (
                metadata.result_contract_required
                and not callable(getattr(component, "canonical_result", None))
            ):
                raise TypeError(
                    f"calibration method '{component_id}' requires canonical_result()"
                )
        self._items[component_id] = component
        return component

    def get(self, component_id: str) -> T:
        try:
            return self._items[component_id]
        except KeyError as exc:
            available = ", ".join(self._items) or "none"
            raise KeyError(
                f"unknown {self.kind} '{component_id}'; available: {available}"
            ) from exc

    def all(self) -> tuple[T, ...]:
        return tuple(self._items.values())

    def ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __contains__(self, component_id: object) -> bool:
        return component_id in self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


input_adapters = ComponentRegistry[InputAdapter]("input adapter")
calibration_methods = ComponentRegistry[CalibrationMethod]("calibration method")
evaluators = ComponentRegistry[Evaluator]("evaluator")
experiment_providers = ComponentRegistry[ExperimentProvider]("experiment provider")


def reset_registries() -> None:
    """Clear all registries. Intended for isolated extension tests."""

    for registry in (
        input_adapters,
        calibration_methods,
        evaluators,
        experiment_providers,
    ):
        registry.clear()

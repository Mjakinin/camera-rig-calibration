"""Pydantic-driven terminal form support for method extensions."""

from __future__ import annotations

import enum
import types
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefined


@dataclass(frozen=True)
class AutoFormField:
    path: tuple[str, ...]
    label: str
    description: str
    current: Any
    default: Any
    choices: tuple[Any, ...] = ()

    @property
    def key(self) -> str:
        return "extension." + ".".join(self.path)


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        candidates = [item for item in get_args(annotation) if item is not type(None)]
        if len(candidates) == 1:
            return _nested_model(candidates[0])
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _choices(annotation: Any) -> tuple[Any, ...]:
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        values: list[Any] = []
        for item in get_args(annotation):
            if item is type(None):
                continue
            nested = _choices(item)
            if not nested:
                return ()
            values.extend(nested)
        return tuple(values)
    if origin is Literal:
        return tuple(get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return tuple(item.value for item in annotation)
    if annotation is bool:
        return (True, False)
    return ()


def _value(payload: dict[str, Any], path: tuple[str, ...], fallback: Any) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return fallback
        current = current[part]
    return current


def auto_form_fields(
    model_class: type[BaseModel], payload: dict[str, Any]
) -> list[AutoFormField]:
    """Flatten nested Pydantic models while retaining field metadata."""

    current = model_class.model_validate(payload).model_dump(mode="python")
    try:
        defaults = model_class().model_dump(mode="python")
    except Exception:
        defaults = {}
    result: list[AutoFormField] = []

    def walk(
        owner: type[BaseModel],
        prefix: tuple[str, ...],
    ) -> None:
        for name, field in owner.model_fields.items():
            path = (*prefix, name)
            nested = _nested_model(field.annotation)
            if nested is not None:
                walk(nested, path)
                continue
            default = (
                "required"
                if field.default is PydanticUndefined
                and field.default_factory is None
                else _value(
                    defaults,
                    path,
                    field.get_default(call_default_factory=True),
                )
            )
            title = field.title or name.replace("_", " ").strip().title()
            result.append(
                AutoFormField(
                    path=path,
                    label=title,
                    description=(
                        field.description
                        or f"Validated {model_class.__name__} parameter."
                    ),
                    current=_value(current, path, None),
                    default=default,
                    choices=_choices(field.annotation),
                )
            )

    walk(model_class, ())
    return result


def update_auto_form_value(
    model_class: type[BaseModel],
    payload: dict[str, Any],
    path: tuple[str, ...],
    raw_value: Any,
) -> dict[str, Any]:
    """Apply one field update and validate the complete model transactionally."""

    updated = model_class.model_validate(payload).model_dump(mode="python")
    cursor = updated
    for part in path[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    parsed = (
        raw_value
        if not isinstance(raw_value, str)
        else yaml.safe_load(raw_value)
    )
    cursor[path[-1]] = parsed
    return model_class.model_validate(updated).model_dump(mode="python")


def auto_form_field(
    model_class: type[BaseModel],
    payload: dict[str, Any],
    key: str,
) -> AutoFormField:
    return next(
        field
        for field in auto_form_fields(model_class, payload)
        if field.key == key
    )


def prompt_initial_options(
    model_class: type[BaseModel],
    prompt: Any,
) -> dict[str, Any]:
    """Prompt only required fields; validated defaults fill everything else."""

    def build(owner: type[BaseModel]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, field in owner.model_fields.items():
            if field.default is not PydanticUndefined:
                payload[name] = field.default
                continue
            if field.default_factory is not None:
                value = field.default_factory()
                payload[name] = (
                    value.model_dump(mode="python")
                    if isinstance(value, BaseModel)
                    else value
                )
                continue
            nested = _nested_model(field.annotation)
            if nested is not None:
                payload[name] = build(nested)
                continue
            choices = _choices(field.annotation)
            label = field.title or name.replace("_", " ").title()
            suffix = (
                " [" + ", ".join(map(str, choices)) + "]"
                if choices
                else ""
            )
            raw = prompt(f"{label}{suffix}")
            payload[name] = yaml.safe_load(raw) if isinstance(raw, str) else raw
        return payload

    return model_class.model_validate(build(model_class)).model_dump(
        mode="python"
    )


__all__ = [
    "AutoFormField",
    "auto_form_field",
    "auto_form_fields",
    "update_auto_form_value",
    "prompt_initial_options",
]

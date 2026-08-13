from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias


SettingRow: TypeAlias = tuple[str, str, str, object, object, str]

_EXPLICIT_FRAME_LIMIT_LABEL = "explicit per-marker / marker-pair BA frame limits"


def _public_frame_strategy_value(value: object) -> object:
    """Render the stable AP02 compatibility ID as its user-facing semantics."""

    if str(value) == "legacy_smart_v1":
        return _EXPLICIT_FRAME_LIMIT_LABEL
    return value


def render_public_setting_rows(rows: Iterable[SettingRow]) -> list[SettingRow]:
    """Return Wizard setting rows with compatibility-only IDs hidden from users.

    This is deliberately presentation-only: scientific configuration values stay
    untouched so existing configs, fingerprints, and execution behavior remain
    reproducible.
    """

    rendered: list[SettingRow] = []
    for key, group, label, current, baseline, description in rows:
        if key == "ap02_frame_strategy":
            current = _public_frame_strategy_value(current)
            baseline = _public_frame_strategy_value(baseline)
        rendered.append((key, group, label, current, baseline, description))
    return rendered


__all__ = ["SettingRow", "render_public_setting_rows"]

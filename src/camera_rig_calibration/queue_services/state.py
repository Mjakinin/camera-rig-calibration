"""Mutable state shared by queue preflight phases and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueuePreflightState:
    preparation: str = ""
    reports: dict[str, Any] = field(default_factory=dict)
    coverage_override: bool = False
    observation_review: dict[str, Any] = field(default_factory=dict)


__all__ = ["QueuePreflightState"]

from __future__ import annotations

from typing import Any

from .product_policy import _DATASET_CONTEXT


_INSTALLED = False


def install_dataset_context_policy() -> None:
    """Keep product defaults aligned with the input type selected in the wizard.

    Simulation and real-vehicle runs intentionally have different editable
    defaults.  The modular wizard resolves its hooks lazily, so the selected
    input type must be propagated explicitly instead of relying on an ambient
    default ContextVar value.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from .. import wizard

    if getattr(wizard, "_DATASET_CONTEXT_POLICY_INSTALLED", False):
        _INSTALLED = True
        return

    original_choice = wizard._choice

    def choice(label: str, choices: dict[str, str], default: str) -> str:
        selected = original_choice(label, choices, default)
        if label == "Input type":
            if selected == "2":
                _DATASET_CONTEXT.set("simulation")
            elif selected == "1":
                _DATASET_CONTEXT.set("real_vehicle")
        return selected

    original_new_calibration_wizard = wizard.new_calibration_wizard

    def new_calibration_wizard(*args: Any, **kwargs: Any):
        # Every new wizard session starts from the real-vehicle default.  The
        # Input-type choice above switches to simulation before any method jobs
        # are created.  Reset afterwards so a later wizard run in the same
        # process cannot inherit the previous dataset type.
        token = _DATASET_CONTEXT.set("real_vehicle")
        try:
            return original_new_calibration_wizard(*args, **kwargs)
        finally:
            _DATASET_CONTEXT.reset(token)

    choice._rigcal_dataset_context_policy = True  # type: ignore[attr-defined]
    new_calibration_wizard._rigcal_dataset_context_policy = True  # type: ignore[attr-defined]
    wizard._choice = choice
    wizard.new_calibration_wizard = new_calibration_wizard
    wizard._DATASET_CONTEXT_POLICY_INSTALLED = True
    _INSTALLED = True


__all__ = ["install_dataset_context_policy"]

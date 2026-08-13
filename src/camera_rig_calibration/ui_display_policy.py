from __future__ import annotations

from .wizard_presentation import render_public_setting_rows


_INSTALLED = False


def install_ui_display_policy() -> None:
    """Hide compatibility IDs from the user-facing parameter UI only.

    Scientific configuration keeps the stable legacy IDs so old configs and
    method fingerprints remain reproducible. This compatibility installer keeps
    the existing Wizard hook while delegating the presentation transform to a
    pure, independently testable helper.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from . import wizard

    original_setting_rows = wizard._setting_rows
    if not getattr(original_setting_rows, "_rigcal_explicit_ui_values", False):

        def setting_rows(job, groups=None):
            return render_public_setting_rows(original_setting_rows(job, groups))

        setting_rows._rigcal_explicit_ui_values = True  # type: ignore[attr-defined]
        wizard._setting_rows = setting_rows

    _INSTALLED = True


__all__ = ["install_ui_display_policy"]

from __future__ import annotations


_EXPLICIT_FRAME_LIMIT_LABEL = "explicit per-marker / marker-pair BA frame limits"
_INSTALLED = False


def _display_value(value):
    if str(value) == "legacy_smart_v1":
        return _EXPLICIT_FRAME_LIMIT_LABEL
    return value


def install_ui_display_policy() -> None:
    """Hide compatibility IDs from the user-facing parameter UI only.

    Scientific configuration keeps the stable legacy IDs so old configs and
    method fingerprints remain reproducible.  This wrapper only changes what a
    user sees in the Wizard: the effective parameter semantics are shown instead
    of the internal compatibility token ``legacy_smart_v1``.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from . import wizard

    original_setting_rows = wizard._setting_rows
    if not getattr(original_setting_rows, "_rigcal_explicit_ui_values", False):
        def setting_rows(job, groups=None):
            rows = original_setting_rows(job, groups)
            rendered = []
            for key, group, label, current, baseline, description in rows:
                if key == "ap02_frame_strategy":
                    current = _display_value(current)
                    baseline = _display_value(baseline)
                rendered.append(
                    (key, group, label, current, baseline, description)
                )
            return rendered

        setting_rows._rigcal_explicit_ui_values = True  # type: ignore[attr-defined]
        wizard._setting_rows = setting_rows

    _INSTALLED = True


__all__ = ["install_ui_display_policy"]

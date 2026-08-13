from __future__ import annotations

from camera_rig_calibration.wizard_presentation import render_public_setting_rows


def _row(
    key: str,
    current: object,
    baseline: object,
) -> tuple[str, str, str, object, object, str]:
    return (
        key,
        "METHOD-SPECIFIC SETTINGS",
        "Frame-selection strategy",
        current,
        baseline,
        "description",
    )


def test_public_setting_rows_hide_only_ap02_frame_strategy_compatibility_id() -> None:
    source = [
        _row("ap02_frame_strategy", "legacy_smart_v1", "legacy_smart_v1"),
        _row("other_setting", "legacy_smart_v1", "legacy_smart_v1"),
    ]

    rendered = render_public_setting_rows(source)

    expected = "explicit per-marker / marker-pair BA frame limits"
    assert rendered[0][3] == expected
    assert rendered[0][4] == expected
    assert rendered[1][3] == "legacy_smart_v1"
    assert rendered[1][4] == "legacy_smart_v1"

    # The presentation layer must never mutate scientific source values.
    assert source[0][3] == "legacy_smart_v1"
    assert source[0][4] == "legacy_smart_v1"


def test_public_setting_rows_preserve_noncompatibility_values() -> None:
    source = [_row("ap02_frame_strategy", "wizard_graph_preserving_v1", None)]

    assert render_public_setting_rows(source) == source

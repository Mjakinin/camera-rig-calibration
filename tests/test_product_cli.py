from __future__ import annotations

import importlib


def test_import_is_side_effect_free(monkeypatch) -> None:
    from camera_rig_calibration import bootstrap

    calls: list[str] = []
    monkeypatch.setattr(
        bootstrap,
        "install_product_stack",
        lambda: calls.append("install"),
    )

    import camera_rig_calibration.product_cli as product_cli

    importlib.reload(product_cli)

    assert calls == []


def test_main_installs_product_stack_before_cli(monkeypatch) -> None:
    import camera_rig_calibration.cli as cli
    import camera_rig_calibration.product_cli as product_cli

    calls: list[str] = []

    monkeypatch.setattr(
        product_cli.bootstrap,
        "install_product_stack",
        lambda: calls.append("install"),
    )

    def fake_cli_main() -> int:
        calls.append("cli")
        return 17

    monkeypatch.setattr(cli, "main", fake_cli_main)

    assert product_cli.main() == 17
    assert calls == ["install", "cli"]

from __future__ import annotations

import pytest

from camera_rig_calibration.ui import wizard as canonical_wizard


@pytest.fixture(autouse=True)
def _canonical_wizard_module_binding(request, monkeypatch) -> None:
    """Keep modular wizard tests independent of legacy-alias import timing.

    ``test_simulation_wizard`` still imports the historical
    ``camera_rig_calibration.wizard`` path for compatibility coverage.  The
    production flows resolve late-bound policy hooks from ``ui.wizard``.  When
    that legacy import is still a lazy proxy, monkey-patching private ``_...``
    hooks on it is proxy-local and makes focused test runs depend on collection
    order.  Rebind only that test module's helper variable to the canonical
    owner before each test so its monkey-patches exercise the same hook owner as
    the real product flow.
    """

    module = request.module
    if module.__name__.endswith("test_simulation_wizard") and hasattr(
        module, "wizard_module"
    ):
        monkeypatch.setattr(module, "wizard_module", canonical_wizard)

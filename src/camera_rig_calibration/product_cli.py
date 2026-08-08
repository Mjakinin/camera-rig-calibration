from __future__ import annotations

from .marker_preference_policy import install_marker_preference_policy
from .product_policy import install_product_policy
from .reporting_authority_policy import install_reporting_authority_policy
from .submission_bindings import install_submission_bindings
from .submission_policy import install_submission_policy
from .ui_display_policy import install_ui_display_policy


# Install product defaults before the public CLI imports the wizard and queue
# modules. Scientific method implementations remain unchanged; these layers own
# category-specific defaults, data-driven pre-method selections, product wording,
# reporting authority, and derived-publication policy.
install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_marker_preference_policy()
# Bind every already-imported preflight/runtime consumer only after all
# selection wrappers are installed, so Wizard and --config use the same path.
install_submission_bindings()
install_ui_display_policy()

from .cli import main  # noqa: E402


__all__ = ["main"]

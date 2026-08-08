from __future__ import annotations

from .product_policy import install_product_policy
from .ui_display_policy import install_ui_display_policy


# Install product defaults before the public CLI imports the wizard and queue
# modules. Scientific method implementations remain unchanged; this layer owns
# category-specific defaults, product wording, and derived-publication policy.
install_product_policy()
install_ui_display_policy()

from .cli import main  # noqa: E402


__all__ = ["main"]

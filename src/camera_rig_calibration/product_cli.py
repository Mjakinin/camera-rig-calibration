from __future__ import annotations

from .bootstrap import install_product_stack


# Install product-facing compatibility policies before importing the public CLI.
# The numerical AP01/AP02/AP03 implementations are not modified here.
install_product_stack()

from .cli import main  # noqa: E402


__all__ = ["main"]

from __future__ import annotations

from . import bootstrap


def main() -> int:
    """Install product-facing policies, then delegate to the canonical CLI."""
    bootstrap.install_product_stack()

    # Import only after the policy stack is installed.  Keeping this import
    # inside the entry point makes importing product_cli itself side-effect
    # free while preserving the runtime ordering of the ``rigcal`` command.
    from .cli import main as cli_main

    return cli_main()


__all__ = ["main"]

"""Compatibility support kept separate from the product architecture."""

from . import import_aliases as _aliases


def install_import_aliases() -> None:
    """Install aliases in a circular-import-safe order."""

    # Historical root names must exist before importing any relocated package;
    # some established package initializers still reference those names.
    for alias, target in _aliases.LEGACY_MODULE_ALIASES.items():
        _aliases._install_module_alias(_aliases._full(alias), _aliases._full(target))

    # The CLI has one intentionally lazy rerun import in main().
    _aliases._install_module_alias(
        _aliases._full("application.rerun"),
        _aliases._full("runtime_services.rerun"),
    )

    # Package aliases provide __path__ for unchanged relative imports in files
    # that were moved byte-for-byte into their owning subsystem.
    for alias, target in _aliases.SCOPED_PACKAGE_ALIASES.items():
        if alias == target:
            continue
        _aliases._install_package_alias(_aliases._full(alias), _aliases._full(target))

    for alias, target in _aliases.SCOPED_MODULE_ALIASES.items():
        if alias == target:
            continue
        _aliases._install_module_alias(_aliases._full(alias), _aliases._full(target))


__all__ = ["install_import_aliases"]

"""Reproducible orchestration for camera-rig calibration methods."""

from .config.models import RigConfig
from .compat import install_import_aliases as _install_import_aliases

_install_import_aliases()
del _install_import_aliases

__all__ = ["RigConfig"]
__version__ = "0.1.0"

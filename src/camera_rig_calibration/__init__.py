"""Reproducible orchestration for camera-rig calibration methods."""

import sys as _sys

from .config.models import RigConfig
from .compat import install_import_aliases as _install_import_aliases

_install_import_aliases()

# Relocation aliases exist only to preserve historical imports while modules are
# moved into focused packages.  Never let those aliases shadow canonical modules
# that now genuinely live at these paths.
for _canonical_module in (
    "camera_rig_calibration.runtime_services.observations",
    "camera_rig_calibration.runtime_services.progress",
    "camera_rig_calibration.publication_services.dataset",
):
    _sys.modules.pop(_canonical_module, None)
    _parent_name, _, _attribute = _canonical_module.rpartition(".")
    _parent = _sys.modules.get(_parent_name)
    if _parent is not None and _attribute in vars(_parent):
        delattr(_parent, _attribute)

del _attribute
del _canonical_module
del _install_import_aliases
del _parent
del _parent_name
del _sys

__all__ = ["RigConfig"]
__version__ = "0.1.0"

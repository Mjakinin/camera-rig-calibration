"""Compatibility exports for the observation data model.

New code should import from :mod:`camera_rig_calibration.observations` or the
focused :mod:`camera_rig_calibration.observation_services` package.
"""

from .observation_services.core import ResolvedSelections

__all__ = ["ResolvedSelections"]

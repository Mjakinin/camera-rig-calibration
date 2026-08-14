"""Compatibility command for shared marker detection.

The implementation lives with the other observation services.  Keeping this
small launcher preserves existing external invocations without mixing the
detector implementation into the package root.
"""

from .observation_services.detection import main

__all__ = ["main"]


if __name__ == "__main__":
    main()

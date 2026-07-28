# Shared method utilities

This folder contains implementation utilities used by more than one
calibration method:

- `geometry.py`, `projection.py`: pose and projection mathematics
- `camera_io.py`, `observation_io.py`, `io_utils.py`: normalized data I/O
- `colmap.py`, `colmap_io.py`: COLMAP process and model handling
- `aruco_utils.py`: shared ArUco detector construction
- `sdf_utils.py`: simulation transform parsing
- `diagnostics.py`, `constants.py`: common reporting helpers and conventions

Method-specific logic belongs in `ap01/`, `ap02/` or `ap03/`, not here.

# Source layout

There are two source trees here, with different responsibilities:

## `camera_rig_calibration/`

This is the installable `rigcal` Python package and the only calibration
application.

- `methods/ap01/`: marker-direct calibration with a moving-COLMAP relay
- `methods/ap02/`: reference-marker graph and bundle adjustment
- `methods/ap03/`: shared COLMAP reconstruction with ArUco scale
- `methods/common/`: geometry, COLMAP, camera and observation utilities shared
  by the three methods
- `components/`: small input/evaluation/registration adapters used by the
  runtime; method stage plans live in each method's `pipeline.py`
- `input/`, `dataset/`, `pipeline/`, `evaluation/`: common preparation and
  evaluation infrastructure
- `wizard.py`, `queueing.py`, `publication.py`: interactive setup, execution
  and publication

The public launcher is `run/rigcal.py`; it delegates to this package.

## `calib_lab/`

This is not a second calibration implementation. It is the ROS 2/Gazebo asset
package for the built-in bus simulation:

- SDF worlds
- bus and ArUco models
- camera and marker configuration
- Route 1/Route 2 definitions
- four source scripts used to regenerate the world assets

It remains separate because Gazebo/ROS package discovery and Python package
installation have different requirements. `pyproject.toml` deliberately
installs only `camera_rig_calibration`.

Directories named `*.egg-info` and `__pycache__` are generated metadata/cache,
not source code, and are ignored by Git.

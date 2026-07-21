# Common Utilities

`src/calib_lab/common` contains lightweight helper code shared by setup scripts in the simulation/source side of the project.

The folder should stay small. Calibration-method-specific code belongs in `run/bus_real_data/`, and simulation-specific configuration belongs in `src/calib_lab/bus_real_data/`.

## Current purpose

The important utility is transform and pose handling. It supports conversion and composition logic used by world-generation and calibration-support scripts.

## Files

```text
transform_utils.py
```

Transform and pose helper functions. Keep API changes conservative because older scripts may still import from this location.

## Design rule

Put code here only if it is reusable across multiple project areas. Otherwise place it next to the script or approach that uses it.

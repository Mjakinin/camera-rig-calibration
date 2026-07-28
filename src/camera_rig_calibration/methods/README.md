# Calibration methods

Each method owns one folder. Start with its `README.md`, then `pipeline.py` to
see the executed stage order.

```text
methods/
├── ap01/   marker-direct calibration with moving-COLMAP relay
├── ap02/   reference-marker graph and bundle adjustment
├── ap03/   shared COLMAP reconstruction with ArUco scale
└── common/ geometry, camera, observation and COLMAP utilities
```

Every method folder follows the same separation:

- `pipeline.py`: runtime adapter, requirements and stage dependencies
- small stage modules: one reproducible operation each
- algorithm/core modules: numerical implementation
- `report.py`: method-specific result normalization

The generic queue does not implement calibration mathematics. It obtains these
pipelines through the component registry and executes their declared stages.

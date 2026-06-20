# Shared Bus Real-Data Infrastructure

Shared code and tools used by several bus real-data calibration approaches.

```text
_shared/
  baseline/       # shared preprocessing and neutral observation export
  common/         # reusable Python modules
  tools/          # data capture, live simulation, and migration tools
```

The shared layer should stay method-independent. It may generate raw images, camera info, route metadata, and neutral ArUco observations, but it should not contain AP01/AP02/AP03-specific estimation outputs.

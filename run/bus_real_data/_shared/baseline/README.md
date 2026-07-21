# Shared Baseline

Builds the method-independent baseline:

- validates raw static and moving-camera inputs,
- exports camera metadata,
- detects ArUco observations,
- writes neutral static, moving and combined observation tables.

Canonical output:

```text
results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/
```

Method-specific estimates do not belong here.

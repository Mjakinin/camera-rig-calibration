# AP02: Reference-Marker Graph Bundle Adjustment

AP02 estimates a camera/marker graph in the gauge of ArUco marker 14 and
optimizes reprojection error with bundle adjustment.

Entry point:

```bash
bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh --skip-shared-baseline
```

Canonical result root:

```text
results/bus_real_data/02_ref_marker_graph_ba/
```

AP02-specific retained diagnostics include Ref14-relative evaluation,
GT-aligned full-map evaluation, parameter stability, V1/V2 comparison and an
independent validity audit.

FOV40 must remain `INVALID_FULL_COVERAGE`: coverage and validity are distinct.

# AP02 — reference-marker graph and bundle adjustment

AP02 constructs a camera/frame/marker observation graph, initializes poses and
optimizes the combined static + moving graph. The combined result is primary;
static-only and disconnected-component runs are diagnostics.

The canonical `baseline_v1` preserves Legacy Main semantics: marker 14,
Legacy geometric edge weights, maximum-bottleneck initialization on every
quality-valid graph observation, smart 8/4 moving-frame selection at the BA
boundary, deterministic parameter/residual order, pinhole reprojection and
`soft_l1` loss with `f_scale=3`. Newer Wizard robustness behavior remains
available only through explicit advanced strategies.

Normal `baseline_v1` execution always consumes fresh dataset-derived
observations. The optional `historical_reproduction` switch is off by default
and admits the frozen historical AP02 observation intermediate only after
strict dataset, intrinsics, scientific-contract, artifact, ordering/schema and
reference-marker fingerprint validation.

Execution order:

1. `build_graph.py` — construct the accepted observation graph
2. `initialize_stage.py` / `initialize.py` — initialize static-only and
   combined pose graphs
3. `optimize_stage.py` / `optimize.py` / `optimize_core.py` — run robust bundle
   adjustment
4. `component_diagnostics.py` — calibrate independently observable disconnected
   components without joining their gauges
5. `report.py` — publish the combined primary result and diagnostics

Supporting files:

- `pipeline.py` declares stage dependencies and diagnostic/primary roles.
- `common.py` contains AP02-specific CSV, pose and projection helpers.

No transformation is invented between disconnected graph components.

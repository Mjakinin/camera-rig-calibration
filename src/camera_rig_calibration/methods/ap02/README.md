# AP02 — reference-marker graph and bundle adjustment

AP02 constructs a camera/frame/marker observation graph, initializes poses and
optimizes the combined static + moving graph. The combined result is primary;
static-only and disconnected-component runs are diagnostics.

The canonical `baseline_v1` uses marker 14, geometric edge weights,
maximum-frontier initialization on every quality-valid graph observation,
smart 8/4 moving-frame selection at the BA boundary, deterministic
parameter/residual order, pinhole reprojection and `soft_l1` loss with
`f_scale=3`. More configurable graph selection and distortion-aware
reprojection remain available through explicit advanced settings.

Every run derives its observation graph from the selected dataset.

## Method semantics

Each accepted ArUco observation connects an observer to a marker. An observer
is either a static camera or a retained moving-camera frame. Shared marker
observations therefore create paths through which otherwise separated camera
regions can become geometrically related.

A selected reference marker defines the coordinate frame. AP02 initializes the
observable graph and then jointly refines the poses of static cameras, retained
moving-camera frames and all non-reference markers. The reference marker is
fixed and is not included as an optimization variable.

The bundle-adjustment objective is built from the detected 2D marker corners.
For each accepted observation, the known marker corner coordinates are
transformed through the current marker and observer poses and projected into
the image. For one corner, the implemented residual is

$$
\mathbf{r}_n
=
\hat{\mathbf{u}}_n-\mathbf{u}_n,
$$

where $\mathbf{u}_n\in\mathbb{R}^2$ is the observed image coordinate and
$\hat{\mathbf{u}}_n\in\mathbb{R}^2$ is the corresponding projection predicted
by the current marker and observer poses. The optimizer stacks both residual
components from all valid marker corners and minimizes them with SciPy's robust
`soft_l1` least-squares loss. Camera intrinsics and marker geometry are treated
as fixed inputs during this optimization.

The primary `with_moving` result includes the selected moving-camera frames and
is what AP02 publishes as its main calibration result. The `static_only` stage
is retained as a diagnostic.

Graph connectivity is also part of the result interpretation. If accepted
observations form disconnected components, each component has its own
independent coordinate gauge. AP02 can calibrate sufficiently supported
components separately, but it does not invent a transformation between
components that are not connected by observations.

## Execution order

1. `build_graph.py` — construct the accepted observation graph
2. `initialize_stage.py` / `initialize.py` — initialize static-only and
   combined pose graphs
3. `optimize_stage.py` / `optimize.py` / `optimize_core.py` — run robust bundle
   adjustment
4. `component_diagnostics.py` — calibrate independently observable disconnected
   components without joining their gauges
5. `report.py` — publish the combined primary result and diagnostics

## Supporting files

- `pipeline.py` declares stage dependencies and diagnostic/primary roles.
- `common.py` contains AP02-specific CSV, pose and projection helpers.
- `optimize_core.py` defines the pose parameterization, marker-corner
  reprojection residuals and robust least-squares optimization.

No transformation is invented between disconnected graph components.

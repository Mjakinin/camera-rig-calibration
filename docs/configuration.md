# Configuration

All configuration and queue files must use `schema_version: 5`. Unknown fields
and older schemas are rejected, paths are resolved relative to the YAML file,
and every execution stores both requested and resolved configs.

The normal workflow is:

```bash
rigcal
```

For automation:

```bash
rigcal --config workspace/<dataset>/queue/queue.yaml --yes
```

## Minimal AP02 prepared-dataset job

```yaml
schema_version: 5
project:
  workspace_root: workspace
  dataset_cache_root: workspace/preparation_cache
  output_root: results
  experiment_id: vehicle_exterior_day_01
  run_label: baseline
dataset:
  id: vehicle_exterior_day_01
  category: real_vehicle
  source_kind: prepared
  scene_type: exterior
  prepared_root: /data/vehicle_exterior_day_01
static_cameras:
  - id: front_left
  - id: front_right
  - id: rear
moving_camera:
  id: calibration_camera
markers:
  dictionary: DICT_4X4_50
  length_m: 0.17
  accepted_ids: all_detected
  detection_mode: baseline         # baseline | subpixel_refined | high_sensitivity
selection:
  mode: auto
methods:
  enabled: [ap02]
  ap02:
    reference_marker_selection_mode: auto
    reference_marker_id: auto
    reference_marker_maximum_frames: null
    top_per_marker: 8
    top_per_marker_pair: 4
    maximum_total_frames: null
    static_only_ba_max_function_evaluations: 50
    combined_ba_max_function_evaluations: 50
    ba_robust_loss: soft_l1
    ba_robust_loss_scale_px: 3.0
observation_quality:
  maximum_pnp_reprojection_error_px: 25.0
  minimum_marker_area_ratio: 0.000008
  require_positive_depth: true
  maximum_marker_distance_m: disabled
evaluation:
  enabled: true
  anchor_marker_id: auto
  anchor_selection_mode: auto       # auto | review_once | explicit
```

One job snapshot enables exactly one method. Variants belong in a queue.

## Combined AP03

```yaml
methods:
  enabled: [ap03]
  ap03:
    single:
      scale_marker_id: auto
    multi:
      marker_ids: auto
    scale:
      reprojection_threshold_px: 5.0
      ransac_iterations: 1000
      minimum_inliers: 4
      maximum_observations_per_marker: null
colmap:
  executable: auto
  matcher: exhaustive
  compute_mode: cpu_baseline
  maximum_image_size: 2400
  maximum_features: 8192
  mapper_minimum_matches: 8
```

AP03 runs COLMAP once. Single and Multi consume the same reconstruction.

## Queue contract

```yaml
kind: rigcal_queue
schema_version: 5
id: vehicle_exterior_day_01_queue
continue_independent: true
common:
  dataset:
    id: vehicle_exterior_day_01
    category: real_vehicle
    source_kind: prepared
    scene_type: exterior
    prepared_root: /data/vehicle_exterior_day_01
  aruco:
    dictionary: DICT_4X4_50
    length_m: 0.17
    accepted_ids: all_detected
  observation_quality:
    maximum_pnp_reprojection_error_px: 25.0
    minimum_marker_area_ratio: 0.000008
    require_positive_depth: true
    maximum_marker_distance_m: disabled
  evaluation:
    enabled: true
    anchor_marker_id: auto
    anchor_selection_mode: auto
entries:
  - id: vehicle_exterior_day_01__ap01__baseline__01
    config: configs/01_baseline.yaml
  - id: vehicle_exterior_day_01__ap02__baseline__02
    config: configs/02_baseline.yaml
  - id: vehicle_exterior_day_01__ap03__matcher_sequential__03
    config: configs/03_matcher_sequential.yaml
```

New queues contain one dataset. Each config is a complete deep-copy snapshot;
its dataset, markers, observation-quality baseline, and evaluation values must
equal `common`. Dependencies
must refer to preceding entries. Independent entries continue after runtime
failure. Queue-wide preparation is shared, while an invalid preflight row
blocks and archives only that method.

Repeated method selections remain separate entries with unique IDs. Exact
scientific duplicates are skipped at execution; edited variants receive their
own method fingerprint and run.

## Simulation batch contract

```yaml
kind: rigcal_batch
schema_version: 1
id: simulation_batch_01
continue_independent: true
queues:
  - experiment_id: route2
    queue: 01_route2/queue/queue.yaml
  - experiment_id: fov_100deg
    queue: 02_fov_100deg/queue/queue.yaml
```

Every referenced file remains a normal schema-v5 one-experiment queue. Local
canonical inputs are reused. A missing historical simulation input is captured
once for that experiment, then shared by every method row. Batch execution
writes `batch_state.json` and continues after an independent experiment error.

`project.execution_mode: prepare_only` is a dedicated one-row path: it performs
exactly one capture/import/preparation/observation pass, publishes only the
canonical dataset input, and schedules no AP preflight, calibration result or
second preparation.

## Common evaluation and export anchor

`evaluation.anchor_marker_id` is the one marker frame used by common
post-method evaluation, the 6-DoF camera export and RViz. It is independent
from AP02's internal `methods.ap02.reference_marker_id`.

- `anchor_selection_mode: auto` ranks only repeat-supported markers compatible
  with every queued method variant and freezes the deterministic recommendation
  before calibration.
- `anchor_selection_mode: review_once` keeps `anchor_marker_id: auto` in the
  requested config, pauses once after shared detection and lists every raw
  detected marker ID. A problematic marker may be confirmed deliberately; a
  method that cannot reconstruct it reports an unavailable anchor export and
  never substitutes another marker.
- Resolved prompt-free configs store the selected integer together with
  `anchor_selection_mode: explicit`.

Anchor-relative method outputs use `T_parent_child`, with
`p_parent = T_parent_child @ p_child`, an
`evaluation_anchor_marker_<ID>` parent and `<camera>_optical_frame` children.
Translation is expressed in metres; RPY uses radians and
`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`; quaternions use `qx,qy,qz,qw`.

## Observation quality

The queue owns a baseline:

```yaml
observation_quality:
  maximum_pnp_reprojection_error_px: 25.0
  minimum_marker_area_ratio: 0.000008
  require_positive_depth: true
  maximum_marker_distance_m: disabled
```

Each AP method has an `observation_quality` mapping with the same four fields.
`null` means inherit the queue baseline; any explicit value is a method-only
override. Resolved configs and manifests store both the effective values and
whether each came from `global` or `method_override`.

```yaml
methods:
  ap02:
    observation_quality:
      maximum_pnp_reprojection_error_px: 10.0
      minimum_marker_area_ratio: null
      require_positive_depth: null
      maximum_marker_distance_m: null
```

Optional numeric limits use `null` for unlimited and reject `0`. Distance and
PnP maximums also support the explicit string `disabled`. Marker area is a
ratio of marker pixels to total image pixels, so it remains comparable across
resolutions. These settings are separate from AP03 scale RANSAC and common
evaluation reprojection thresholds.

Successful detection, four finite corners, successful PnP, finite
rotation/translation and a positive finite translation norm remain fixed
validity checks. Positive depth is enabled by default and can only be changed
explicitly through `require_positive_depth`.

## Scientific selections

```yaml
selection:
  mode: auto
methods:
  ap01:
    root_camera: auto
    top_moving_per_marker: 8
    scale_top_per_marker: 30
  ap02:
    reference_marker_selection_mode: auto
    reference_marker_id: auto
    reference_marker_maximum_frames: null
    top_per_marker: 8
    top_per_marker_pair: 4
    maximum_total_frames: null
  ap03:
    single:
      scale_marker_id: auto
    multi:
      marker_ids: auto
evaluation:
  enabled: true
  anchor_marker_id: auto
```

`auto` is the default: it deterministically freezes the computed candidates and
continues without a pause. `review_once` always pauses exactly once after the
candidate analysis, displays one combined queue review with a separate
candidate section for every manually configured method variant, and freezes
the accepted values. `explicit` validates values selected from an existing
prepared dataset and continues without a review pause.

The interactive wizard never asks for an unlisted camera or marker ID. Editing
AP01 Root, AP02 Reference, or AP03 Single/Multi first offers `auto` or
`manual`. A prepared dataset with complete observations shows its compatible
numbered candidates immediately, recalculated with that job's current ArUco
and observation-quality settings. A new video, frame set, or simulation
capture defers the same numbered choice until `review_once`, after detection
and filtering. In a non-interactive run this checkpoint is stored as
`waiting_for_selection` and can be resumed without repeating preparation.

`project.execution_mode: prepare_only` intentionally stops after publishing
the complete immutable dataset.

These five scientific roles are independent. AP01 has no Reference ArUco.
The evaluation anchor is selected exactly once in preflight and then frozen
for every method. A method that cannot reconstruct it keeps its calibration
result, while the common evaluation is reported as unavailable; no fallback
anchor is substituted. If no repeat-supported common candidate exists,
preflight stops. The wizard can explicitly set `evaluation.enabled: false`;
this disables only the shared evaluation, not calibration.

## AP02

Supported losses are `soft_l1`, `huber`, and `linear`. Baseline defaults:

```yaml
reference_marker_selection_mode: auto
reference_marker_id: auto
static_only_ba_max_function_evaluations: 50
combined_ba_max_function_evaluations: 50
ba_robust_loss: soft_l1
ba_robust_loss_scale_px: 3.0
```

Reference-marker modes are:

- `baseline`: simulation only; fixes marker 14 and is part of the Route-2
  baseline contract.
- `auto`: freezes the deterministic preflight recommendation.
- `manual`: pauses after preflight, lists every detected marker and requires
  warning confirmation for a weak candidate.
- `explicit`: schema-v5 compatibility mode for an already stored integer ID.

Static-only BA is diagnostic; Combined static/moving BA is primary. No
static-only result is promoted to the main comparison. AP02 selects
quality-ranked reference, per-marker and per-marker-pair frames, deduplicates
them, and preserves the accepted graph. If `maximum_total_frames` is below the
minimum graph-preserving set, preflight fails and reports the required count.
Initialization uses a deterministic maximum-bottleneck path tree rooted at the
resolved reference marker. An unweighted first-hit BFS is written only as an
independent diagnostic and never supplies production initial poses. Static-only
and Combined use exactly their configured function-evaluation budgets; rigcal
does not retry with a larger budget.

`ap02_optimization_summary.json` records SciPy status, `nfev`/`njev`, cost,
RMSE, optimality, runtime, variables and residual counts.
`ap02_optimization_history.csv` numbers actual residual-function evaluations;
it intentionally leaves solver-iteration and per-call `nfev` cells empty when
SciPy does not expose those values reliably.

## COLMAP

AP01 and AP03 snapshots independently configure:

```yaml
colmap:
  executable: auto
  matcher: exhaustive        # exhaustive | sequential
  compute_mode: cpu_baseline # cpu_baseline | gpu | auto
  maximum_image_size: 2400
  maximum_features: 8192
  mapper_minimum_matches: 8
  sequential_overlap: 20
  loop_detection: true
```

Sequential settings apply only to the sequential matcher. Requested config
keeps `executable: auto`; resolved config and the run manifest record the
absolute executable, version, configured compute mode and resolved compute
mode. `gpu` fails preflight without a compatible GPU, while `auto` resolves
explicitly to GPU or CPU before method execution. Within schema v5, deprecated
values migrate as `gpu_mode: false -> cpu_baseline`, `true -> gpu`, and
`auto -> auto`.

The reproducible Route-2 baseline uses AP01 on CPU with exhaustive matching,
4096 features and a 1600-pixel maximum image dimension. AP03 uses the same CPU
and matcher contract with 8192 features, a 2400-pixel maximum image dimension,
eight mapper matches, one physical COLMAP camera ID per real camera and fixed
intrinsics. Configured and effective values are retained separately in
provenance.

## Real input and sampling

Put a recording under `data_local/<dataset-id>/`. Files may include video,
extracted frames, intrinsics JSON/YAML, checkerboard video or checkerboard
image folders, direct images, `.mcap`, or `.db3`. Recommended role names are
`moving_frames/`, `static/`, and `intrinsics_images/` (or `checkerboard/`).
Nested static sequences use `static/<camera-id>/images/`. Recursive discovery
asks only about ambiguity. Role keywords may be embedded in names such as
`static_v2`, `moving_frames_night`, or `iphone_intrinsics_v3`; images and
videos inherit the role from any parent directory. Static videos use a
deterministic middle-frame selection with a recorded first-frame fallback.

`sampling.target_hz` is required for a new moving video and is not requested
for prepared frames:

```yaml
sampling:
  target_hz: 3.0
  start_seconds: 0.0
  end_seconds: null
  maximum_frames: null
```

Moving media and moving-camera intrinsics may be combined independently:

```yaml
moving_camera:
  id: moving_calib_camera
  frames: /absolute/path/to/moving_frames
  intrinsics_profile: iphone_05x_4k@0123456789ab
  intrinsic_scan:
    mode: balanced                  # balanced | full_frame
    target_hz: 3.0
    preview_max_dimension: 1920
```

`intrinsics`, `intrinsics_profile`, `intrinsic_calibration_video`, and
`intrinsic_calibration_images` describe the selected source/profile. Exactly
one existing-intrinsic or calibration source is active. Image folders are
fingerprinted from their sorted filenames and file contents. A resolved run
records the installed intrinsic path, full profile fingerprint, SHA-256,
resolution, and distortion model.
Exact moving-frame and intrinsic resolutions are required; `rigcal` never
rescales K silently.

`balanced` searches the checkerboard adaptively at 3/6/12 Hz and refines
accepted corners in the original resolution. `full_frame` evaluates every
original frame in full resolution and is therefore substantially slower.
For video inputs, `ffprobe` display rotation is applied explicitly before
calibration or frame extraction. Encoded `1920x1080` media with rotation
`-90` consequently produces normalized `1080x1920` frames and requires a
matching portrait intrinsics profile.

The ArUco detection mode is queue-wide. `baseline` preserves OpenCV defaults,
`subpixel_refined` keeps the same candidates with subpixel corner refinement,
and `high_sensitivity` adds a two-pass, mutually confirmed gamma search.
Different modes receive different observation fingerprints and must not be
mixed silently inside one experiment.

Real acquisitions are cached internally under
`workspace/preparation_cache/_acquisitions/<hash>/`. A canonical experiment
combines
that immutable acquisition with one intrinsic fingerprint. Selecting a
different profile therefore reuses video extraction and static inputs while
invalidating PnP observations and dependent methods.

## Simulation

`simulation.enabled: true` requires `dataset.scene_type: simulation`, the
built-in bus SDF and a Route-1/Route-2 JSON. `world_id` must be `bus`; foreign
world IDs are rejected. The large BeIntelli OBJ is not a Git LFS dependency:
the repository tracks `beintelli_erklarbus.obj.gz`, and the first `rigcal`
invocation verifies and atomically materializes the ignored OBJ beside it.
The config also declares the built-in image/CameraInfo topics. The wizard
starts from the committed Route-2 baseline, queues existing bus experiments
by comma list/`all`, and can derive mixed parameter rows from baseline,
historical or already queued experiments.

## Duplicate and result behavior

`duplicate_policy: skip` reuses an exact completed input/method fingerprint.
Labels are generated from effective deviations, including method-quality
override provenance. A same-label/different-fingerprint conflict therefore
signals inconsistent stored evidence instead of asking for a manual
`variant2` name. Public results are immutable and have no `current` or
`run_history` wrapper.

Resolved root/marker/matcher values and non-baseline differences are visible in
variant directory names and complete machine-readable config diffs.

Successful method rows publish independently. A failed method is stored under
`attempts/` with `scientific_validity: incomplete_non_authoritative`; concise
cause codes include
`colmap_sparse_model_failed`, `preflight_failed`, `timeout`,
`configuration_validation_failed` and `optimizer_failed`. Experiment result
status is `available`, `partial`, or `failed` according to authoritative
successes and archived failures.

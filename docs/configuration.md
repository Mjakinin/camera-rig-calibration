# Configuration

All newly written files use `schema_version: 5`. Unknown fields are rejected,
paths are resolved relative to the YAML file, and every execution stores both
requested and resolved configs. Schema v1/v2/v3 files remain readable and are
migrated in memory. Removed v1-v3 selection fields are accepted only while
migrating an older schema; they are rejected in a schema-v5 file.

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
  dataset_cache_root: datasets
  output_root: results
  experiment_id: vehicle_exterior_day_01
  run_label: ap02_baseline
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
selection:
  mode: review_once
methods:
  enabled: [ap02]
  ap02:
    reference_marker_id: auto
    static_only_ba_max_function_evaluations: 100
    combined_ba_max_function_evaluations: 120
    ba_robust_loss: soft_l1
    ba_robust_loss_scale_px: 3.0
observation_quality:
  maximum_pnp_reprojection_error_px: 25.0
  minimum_marker_area_px2: 0.0
  maximum_marker_distance_m: disabled
evaluation:
  anchor_marker_id: auto_common
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
colmap:
  executable: auto
  matcher: exhaustive
  gpu_mode: auto
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
  evaluation:
    anchor_marker_id: auto_common
entries:
  - id: ap01_baseline
    config: configs/01_ap01_baseline.yaml
  - id: ap02_baseline
    config: configs/02_ap02_baseline.yaml
  - id: ap03_exhaustive
    config: configs/03_ap03_exhaustive.yaml
```

New queues contain one dataset. Each config is a complete deep-copy snapshot;
its dataset, markers, and evaluation values must equal `common`. Dependencies
must refer to preceding entries. Independent entries continue after runtime
failure, but queue-wide preflight blocks every method if any job is invalid.

## Observation quality

Every method job owns:

```yaml
observation_quality:
  maximum_pnp_reprojection_error_px: 25.0
  minimum_marker_area_px2: 0.0
  maximum_marker_distance_m: disabled
```

Optional maximums are either `disabled` or greater than zero. Minimum area is
zero or greater. These settings are separate from AP03 scale-estimator
thresholds. Every method consumes all observations that pass these checks.

The following checks are immutable: successful detection, four finite corners,
successful PnP, finite rotation/translation, positive depth, and positive
finite translation norm.

## Scientific selections

```yaml
selection:
  mode: review_once
methods:
  ap01:
    root_camera: auto
  ap02:
    reference_marker_id: auto
  ap03:
    single:
      scale_marker_id: auto
    multi:
      marker_ids: auto
evaluation:
  anchor_marker_id: auto_common
```

`review_once` prepares observations, displays one combined candidate review,
and freezes explicit values before method execution. `auto` is deterministic
and prompt-free. `explicit` rejects unresolved values.

These five scientific roles are independent. AP01 has no Reference ArUco.

## AP02

Supported losses are `soft_l1`, `huber`, and `linear`. Compatibility defaults:

```yaml
static_only_ba_max_function_evaluations: 100
combined_ba_max_function_evaluations: 120
ba_robust_loss: soft_l1
ba_robust_loss_scale_px: 3.0
```

Static-only BA is diagnostic; Combined static/moving BA is primary. No
method-specific smart/stride/Top-N or moving-frame cap is active.

## COLMAP

AP01 and AP03 snapshots independently configure:

```yaml
colmap:
  executable: auto
  matcher: exhaustive        # exhaustive | sequential
  gpu_mode: auto             # auto | true | false
  maximum_image_size: 2400
  maximum_features: 8192
  mapper_minimum_matches: 8
  sequential_overlap: 20
  loop_detection: true
```

Sequential settings apply only to the sequential matcher. Requested config
keeps `executable: auto`; resolved config and the run manifest record the
absolute executable, version, and resolved GPU mode. `gpu_mode: true` fails
preflight without a compatible GPU, while `auto` falls back to CPU.

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
    mode: balanced                  # balanced | exhaustive_compatibility
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

Real acquisitions are cached independently under
`datasets/_acquisitions/<hash>/`. A canonical dataset composition combines
that immutable acquisition with one intrinsic fingerprint. Selecting a
different profile therefore reuses video extraction and static inputs while
invalidating PnP observations and dependent methods.

## Simulation

`simulation.enabled: true` requires `dataset.scene_type: simulation`, an SDF
world and a route JSON. The large BeIntelli OBJ is not a Git LFS dependency:
the repository tracks `beintelli_erklarbus.obj.gz`, and the first `rigcal`
invocation verifies and atomically materializes the ignored OBJ beside it.
The config also declares image/CameraInfo topics. The wizard starts from the
committed Route-2 baseline and edits only selected parameter rows. An exact
existing capture is offered for reuse.

## Duplicate and result behavior

`duplicate_policy: skip` reuses an exact completed input/method fingerprint.
`force` stages a fresh execution and atomically replaces `current` only on
success. Prior compact manifests remain in `run_history`.

Resolved root/marker/matcher values and non-baseline differences are visible in
variant directory names and complete machine-readable config diffs.

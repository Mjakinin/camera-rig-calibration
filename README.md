# `rigcal` — reproducible camera-rig calibration

`rigcal` is the single terminal interface for importing or capturing a camera
rig dataset, running AP01/AP02/AP03, evaluating the results, and preserving the
exact configuration and provenance of every experiment.

The AP01, AP02, and AP03 scientific cores remain the existing repository
implementations. `rigcal` supplies generic inputs, explicit parameters,
per-job observation filtering, orchestration, progress reporting, and
reproducible result paths.

## Install and start

Enter the ROS 2 Humble container and install the package once:

```bash
ros2humble
python3 -m pip install -e ".[scientific]"
hash -r
rigcal
```

The root-user warning from pip is expected inside this disposable development
container. The repository contains a compatibility shim for the older pip
version in the container, so editable installation does not require a pip
upgrade.

The BeIntelli bus mesh does not require Git LFS. Git stores one tracked
31.2 MB `beintelli_erklarbus.obj.gz` archive. On the first `rigcal` start it is
SHA-256-verified and atomically expanded to the ignored 160.7 MB OBJ. Later
starts only verify and reuse the local OBJ.

The normal interface is:

```text
CAMERA RIG CALIBRATION

1. Start a new calibration
2. View results
3. Manage incomplete runs
4. Check installation
5. Cleanup storage
6. Manage intrinsics profiles
0. Exit
```

For a reproducible non-interactive run:

```bash
rigcal --config workspace/<dataset>/queue/queue.yaml --yes
rigcal --config workspace/<dataset>/queue/queue.yaml --dry-run
rigcal --resume <run-id>
```

Run `Check installation` before an actual calibration. A newly built container
may still need COLMAP. AP02 does not use COLMAP; AP01 and AP03 do.

## One guided workflow

Choose `Start a new calibration`. The input inventory separates reusable
simulation experiments from real-vehicle data and offers:

- a prepared dataset whose frames and intrinsics already exist;
- one recursively scanned real-data folder;
- a new Gazebo capture based on the committed Route-2 baseline or an edited
  parameter vector.

A ROS 2 recording (`.mcap` or `.db3`) is detected inside the same real-data
folder. It is not a separate user workflow.

Simulation capture accepts only fresh frames. The recorder waits for pixel data
that was not used by an earlier route pose, requires the configured frame count,
and rejects captures with less than 90% unique images before an AP method can
start. This prevents a slow headless renderer from silently publishing stale
duplicates as a successful experiment.

Additional Gazebo rigs are registered by copying
`config/simulation_worlds/_template.yaml`; the Wizard discovers valid manifests
on its next start. See [docs/simulation_worlds.md](docs/simulation_worlds.md).

After import or capture, the queue-wide pipeline is:

```text
prepare input once
→ detect all raw ArUco observations once
→ apply each job's observation-quality snapshot
→ analyze root/reference/scale candidates
→ resolve automatic scientific selections deterministically
→ freeze the prompt-free queue
→ run all method jobs
→ evaluate and compare
```

Preflight is independent per method job. `READY`, `READY_WITH_WARNINGS`, and
`READY_PARTIAL` jobs run; only the incompatible job is skipped after
`FAILED_PREFLIGHT`. AP02 may publish a clearly labelled diagnostic
`PARTIAL_N_OF_M` result while independent AP01/AP03 jobs continue. Quality
limits are never relaxed automatically. Prepared inputs and raw observations
are reused on resume.

### Camera and sampling detection

A prepared dataset displays a read-only summary:

```text
Static cameras: 4
  cam_edge_0, cam_edge_1, cam_edge_3, cam_edge_5
Moving cameras: 1
  moving_calib_camera
Moving frames: 78
Sampling: 1 Hz (stored metadata)
```

Camera IDs are not prompted when they are unambiguous in a dataset manifest or
in one-to-one image/intrinsic basename bindings. Exactly one moving camera is
supported per dataset. An ambiguity prompt explains the conflicting files.

`sampling.target_hz` is requested only when extracting a new moving-camera
video. Existing frames use stored sampling metadata or `unknown` without a
prompt.

### New real data

The input menu is ordered for the common real-data workflow:

```text
1. Real data from data_local or prepared recordings (recommended)
2. Gazebo simulation
3. Other prepared/manual dataset
```

Put every file belonging to one acquisition below:

```text
data_local/<dataset-id>/
```

The recommended zero-configuration folder names are:

```text
data_local/vehicle_exterior_day_01/
├── moving_frames/
│   ├── frame_000001.png
│   └── frame_000002.png
├── static/
│   ├── front_left.png
│   └── front_right.png
├── intrinsics/front_left.yaml
├── intrinsics/front_right.yaml
├── intrinsics/moving_calib_camera.yaml
├── moving_camera.mp4                 # alternative to moving_frames/
├── checkerboard_intrinsics.mp4       # alternative to moving intrinsics
├── intrinsics_images/                # alternative to checkerboard video
│   ├── view_0001.png
│   └── view_0002.png
└── optional_all_cameras.mcap
```

Use one direct PNG/JPG per static camera and give it the same basename as its
intrinsic file. Multiple static candidates per camera are also supported with
`static/<camera-id>/images/*` plus an intrinsic named `<camera-id>.json|yaml`.
Put extracted moving frames in a directory containing `moving` or `frames`;
`moving_frames/` is the clearest convention. Checkerboard photos belong in
`intrinsics_images/` or `checkerboard/`. Every image in that folder must come
from the same unchanged camera/lens configuration and have the same
resolution. The scanner finds these files recursively, so the subfolders are
role hints rather than hard-coded camera names. When a role is unambiguous it
is accepted automatically; otherwise the wizard asks. Raw files stay
unchanged; normalized frames and intrinsics are published under the experiment
input.

Role words may be combined with arbitrary experiment suffixes or prefixes.
For example, `static_v2/`, `night_static_capture/`, `moving_frames_5hz/`, and
`iphone_intrinsics_v3/` are recognized. The role is inherited by both images
and videos inside the directory, so a generic filename such as `IMG_1001.mov`
does not need to be renamed. For a static-camera video, rigcal deterministically
uses its middle frame (falling back to the first readable frame) and records
the source, policy, frame count, and selected index under
`metadata/static_video_extraction/`.

Moving frames and moving-camera intrinsics are independent selections. A
prepared frame set can use its stored CameraInfo, another intrinsic file, a
catalogued profile, or a newly calculated checkerboard profile. New immutable
profiles are stored below:

```text
results/real_vehicle/_intrinsics/<profile-id>/<profile-hash>/
```

Changing only the profile reuses the acquisition frames and creates a new
content-addressed composition; it never overwrites the prepared input. The
default `balanced` checkerboard scan samples the complete video at 3 Hz,
detects on a maximum 1920-pixel preview, refines corners at the original
resolution, and adds deterministic 6/12 Hz passes only when needed. The old
every-frame detector remains available as `exhaustive_compatibility`.
For a checkerboard image folder, every supplied image is inspected once;
`balanced` still performs preview detection followed by full-resolution corner
refinement. Since manually supplied photos are already distinct views, their
recommended minimum frame gap is `0` instead of the video default `5`.

### New simulation

The immutable recommendation is the committed Route-2 baseline:

- Route 2 and 189 route poses;
- 1280 × 720 moving-camera images;
- 69.1° horizontal FOV;
- original baseline SDF lighting;
- motion-blur kernel 0;
- committed settle, skip, and timeout values.

Route, route-frame count, resolution, horizontal FOV, and motion blur are
strictly moving-camera parameters. Every static camera contributes exactly one
snapshot. Its SDF camera model and intrinsics remain unchanged. Explicitly
provided intrinsics are preserved for either a static or the moving camera and
are never overwritten by Gazebo `CameraInfo`. Lighting is deliberately a world
parameter: it can change rendered pixel appearance for every camera, but it
never changes camera matrices or distortion coefficients.

`Create a new bus-simulation parameter combination` first shows one parameter
table:

```text
# | Parameter | Current | Baseline default | Meaning / valid values
```

Select several rows such as `3,4,6`; only those values are prompted. Resolution,
FOV, route, route-frame count, sampling strategy, lighting, blur, settle time,
frame skip, and timeouts can therefore be combined freely. The lighting detail
uses the RGB and attenuation values from `LIGHTING_VARIANTS.json`; the baseline
row also reports the original SDF values.

After editing, `rigcal` displays the entire vector and searches for an identical
capture. Reuse is recommended when one exists, but a deliberate recapture is
possible and receives a new input ID and input hash. Invalid values are
re-prompted; for example, a blur kernel must be `0` or an odd integer.

## Method queue

The public method list is:

```text
1. AP01 — experimental baseline; marker-direct and moving-COLMAP relay
2. AP02 — primary candidate; static-only and combined bundle adjustment
3. AP03 — primary candidate; one COLMAP reconstruction, single and multi scale
```

Default: `1,2,3`.

Each row stores an independent deep-copy configuration snapshot. Add,
duplicate, edit, or remove rows in the same queue editor. `Remove jobs`
accepts comma-separated numbers such as `1,3`, and `all` after confirmation.
`0` goes back to the previous view.

AP03 is one method job: COLMAP runs once, Single Scale is recorded as a
diagnostic result, and Multi Scale is the primary AP03 result. A missing Single
candidate makes the job `PARTIAL` when Multi remains usable.

The queue editor keeps unrelated settings in separate short menus:

- `Queue-wide ArUco input`
- `Queue-wide common evaluation`
- one method-job menu containing `OBSERVATION QUALITY`,
  `METHOD-SPECIFIC SETTINGS`, and (when applicable) `COLMAP SETTINGS`
- a separate rename action for the run label

AP02 has no COLMAP section. AP01/AP03 jobs independently select
`exhaustive` or `sequential`, `gpu_mode` (`auto`, `true`, `false`), mapping
limits, image size, and feature count. Sequential overlap and loop detection
appear only for a sequential matcher. `auto` resolves the executable, version,
GPU capability, and absolute path during preflight.

### Observation quality

Every AP01/AP02/AP03 job has the same optional thresholds:

```yaml
observation_quality:
  maximum_pnp_reprojection_error_px: 25.0
  minimum_marker_area_px2: 0.0
  maximum_marker_distance_m: disabled
```

These v4 defaults reject only extreme PnP reprojection outliers by default.
Detection success, four finite corners, successful PnP, finite pose, positive
depth, and a positive finite translation norm are always required.

The filter recomputes four-corner PnP reprojection RMSE and writes:

```text
observation_filter_summary.json
accepted_observations.csv
rejected_observations.csv
```

Every row records the job, observer, frame, marker, reason, threshold, and
measured value. Rejected observations cannot re-enter a downstream method.

### Scientific references

The choices have separate owners:

- AP01 Root Camera: AP01 output coordinate origin.
- AP02 Reference Marker: AP02 pose-graph anchor.
- AP03 Single Scale Marker: diagnostic single-marker scale.
- AP03 Multi Marker Set: primary robust AP03 scale.
- Evaluation Anchor Marker: post-method common comparison.

Intrinsics do not choose these values. Candidate ranking uses the complete
filtered static and moving observation graph. The normal `review_once` mode
shows one combined table after detection, confirms all values once, and writes
an explicit prompt-free queue. `auto` is deterministic and stores rankings and
reasons for unattended paper runs.

AP01 has no Reference ArUco and continues to use all suitable markers.

## Progress, resume, and cleanup

Runtime output reports queue/job/stage elapsed seconds and meaningful counts:

```text
[AP03 ap03_exhaustive] Job 2/3, Step 3/8: COLMAP mapping
Stage elapsed: 84.2 s
Job elapsed: 91.6 s
Queue elapsed: 304.8 s
Registered images: 142
Log: ...
```

No ETA is invented. Commands flush immediately. Counts and timings are also
stored in `timings.json` and `run_manifest.json`.

`Ctrl+C` marks an active queue interrupted. Capture, preparation, observations,
successful individual jobs and failures stay exclusively below
`workspace/temporary_runs/<queue-id>/`. `Manage incomplete runs` can resume a
queue, remove failed jobs, or delete one, comma-separated, or all temporary
queues after confirmation. Nothing appears in `View results` until every
remaining job has terminated correctly and the complete queue is published.

## Results

Datasets and scientific outputs are separate, with the same grouping:

```text
datasets/
├── simulation/baseline|fov|resolution|lighting|motion_blur|density|route|capture|mixed/...
└── real_vehicle/video|frames|rosbag|prepared/<sampling>/<experiment>/

results/
├── simulation/<same factor path>/
└── real_vehicle/<same source/sampling path>/
```

The dataset experiment contains reusable inputs:

```text
datasets/<category>/<group>/<experiment>/
├── experiment.yaml
└── inputs/<input-id>/
│   ├── raw_images/
│   ├── metadata/
│   └── observations/<detection-id>/
│       ├── shared_*.csv
│       ├── connectivity_report.json
│       └── debug_gallery/
```

The corresponding result experiment contains only outputs:

```text
results/<category>/<group>/<experiment>/
├── experiment.yaml
├── PUBLISHED.json
├── methods/
│   ├── ap01/root_<camera>__matcher_<matcher>__<diff>_<hash>/
│   ├── ap02/ref_marker_<id>__<diff>_<hash>/
│   └── ap03/single_marker_<id>__multi_<set>__matcher_<matcher>_<hash>/
├── evaluations/
└── comparisons/
```

Each execution snapshot contains normalized input links, observations,
preflight audits, the method stage (`02_AP01`, `03_AP02`, or `04_AP03`),
evaluation, comparison, requested/resolved configs, exact commands, logs,
environment, timings, and the final report.

AP03 uses:

```text
04_AP03/
├── colmap/
├── scale_single/
├── scale_multi/
├── evaluation_single/
├── evaluation_multi/
└── AP03_REPORT.*
```

The Route-2 baseline also has relative links at its default factor values, for
example `fov/69.1deg` and `resolution/1280x720`. There is no `_views` tree.
Discovery follows only canonical `experiment.yaml` plus `PUBLISHED.json` and
therefore never counts these links as another experiment.

Every new observation set also contains a JPEG `debug_gallery/` with one
annotated preview for every moving frame. `View results` reports frames with
and without detections, multi-marker frames, AP02 bridge frames, and the
gallery path.

`Cleanup storage` removes generated frame caches, experiment input images,
debug galleries, inactive working data, and COLMAP image copies while keeping
method results, final figures, configs, logs, observation CSVs, connectivity
reports, and numeric reconstructions. Reclaimable size is hardlink-aware.
`data_local/` is outside the normal cleanup and is deleted only after its own
second confirmation. Cleaned experiments retain `INPUT_REMOVED.json` and stay
visible as `results available; input cleaned; not rerunnable`.

`Manage intrinsics profiles` has exactly three operations: create/recalculate,
rename, and delete. Renaming changes only the display alias. Deletion is
blocked while an active temporary queue uses the profile; completed-run
references are displayed before the final confirmation.

Exact completed input/method fingerprints are skipped by default. A forced
identical rerun replaces `current` only after success and keeps a compact
history. Compatible AP01/AP03 COLMAP artifacts are content-addressed and reused.

## Repository folders

- `data_local/`: new untracked real recordings supplied by the user.
- `workspace/temporary_runs/`: all unpublished capture and queue transactions.
- `workspace/`: saved requested/resolved configs and publication receipts.
- `datasets/`: canonical captured/prepared inputs and observations.
- `results/simulation/`: published scientific outputs grouped by factor.
- `results/real_vehicle/`: canonical real-data experiments.
- `config/simulation_worlds/`: manifest-only Gazebo world registry.
- `tests/`: small software fixtures only; never put recordings here.

Historical successful results remain indexed through migration manifests and
are not rewritten. Missing historical inputs are reported as `input
unavailable / not rerunnable`.

## Development checks

```bash
python3 -m pip install -e ".[scientific,dev]"
python3 -m compileall -q src run tests
pytest -q
```

Normal tests do not execute complete AP methods. See
[architecture](docs/architecture.md), [configuration](docs/configuration.md),
and the [extension guide](docs/extensions.md).

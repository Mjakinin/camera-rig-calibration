# rigcal — Camera Rig Calibration

`rigcal` is a reproducible university tool for calibrating a static camera rig
with a moving calibration camera. It supports real recordings and the built-in
Gazebo bus world, runs AP01/AP02/AP03 as independent method variants, and
compares their primary results on one common anchor.

## Install and start

Python 3.10–3.13 is supported.

```bash
python3 -m pip install -e ".[scientific]"
rigcal
```

For development:

```bash
python3 -m pip install -e ".[scientific,dev]"
python3 -m compileall -q src run tests
pytest -q
```

The menu provides:

```text
1. Start a new calibration
2. View results
3. Manage incomplete runs
4. Check installation
5. Cleanup storage
6. Manage intrinsics profiles
0. Exit
```

The wizard discovers inputs under `data_local`, canonical datasets under
`datasets`, managed intrinsics under `config/intrinsics`, and the built-in bus
simulation assets. It does not prompt for arbitrary filesystem paths.

## Input conventions

Put a real acquisition below `data_local/<experiment>/`. It may contain:

- static images or static-camera videos;
- a moving-camera video or extracted moving frames;
- camera-info JSON/YAML files;
- a checkerboard video or checkerboard image folder;
- a ROS bag (`.mcap` or `.db3`).

Recommended role folders are `static/`, `moving_frames/`, `camera_info/`,
`intrinsics/`, and `checkerboard/`. Role words can be part of a longer name,
such as `static_v2` or `iphone_intrinsics_v3`.

Moving frames and moving-camera intrinsics are independent selections. Managed
profiles are immutable and stored as:

```text
config/intrinsics/<profile-id>/<profile-hash>/
  intrinsics.json
  profile.yaml
  reports and source records
```

The two repository profiles are:

- `iphone_05x_4k@d5444b68272f`
- `iphone_1x_4k_3840x2160@3ed2f8d6f7fe`

## Simulation

The stable release supports only the reviewed bus Gazebo world and its declared
Route-1/Route-2 assets. The experiment queue can:

- add the Route-2 baseline;
- add several existing simulation experiments;
- derive new combinations from the baseline or another queued experiment;
- combine route, density, resolution, FOV, lighting, motion blur and capture
  parameters.

Route, density, resolution, FOV and blur affect the moving camera. Lighting
affects the whole world. `pct` means percent.

One method queue is applied to every selected experiment, forming
`experiments × method variants`. Capture and ArUco observation generation run
once per experiment.

## Method queue

The default choices are:

```text
1. AP01 — Direct / Relay marker-based calibration (baseline_v1)
2. AP02 — Graph initialization + bundle adjustment (baseline_v1)
3. AP03 — SfM / multi-camera calibration (baseline_v1)
```

Repeated selections are preserved. For example `1,2,3,3` produces four
editable queue rows. Public names are generated from deviations to baseline,
for example `baseline`, `combined_nfev_60` or
`matcher_sequential__overlap_13`. An exact duplicate is identified and skipped.

AP02 Combined and AP03 Multi are primary results. AP02 Static-only and AP03
Single remain visible diagnostics and are not counted as separate methods in
the scientific comparison.

Additional registered `CalibrationMethod` components appear automatically.
AP01–AP03 keep their dedicated editors; extension methods use their strict
validated YAML configuration model.

Observation quality has one queue-wide baseline and optional overrides per
method row. The default area threshold is the resolution-neutral marker/image
ratio `0.000008`; PnP RMSE, positive depth and maximum distance are separate
filters. In the editor, `inherit` uses the queue baseline and an explicit
value affects only that method variant. Optional selection limits use `null`
for unlimited and reject `0`.

All three approaches make deterministic selections and write their candidates,
quality metrics and tie-breakers to CSV/JSON diagnostics.

### AP01 — Direct / Relay marker-based calibration

AP01 uses synchronized static-camera marker observations plus the moving-camera
sequence. It estimates a Direct transform where a static camera shares marker
support with the root camera and uses the moving-camera trajectory as a Relay
for the remaining cameras. Useful controls include the root camera, Direct
target, observation quality, and the optional robust-consensus strategy. It
publishes static-camera poses, all available unordered camera pairs, quality
diagnostics and runtime provenance.

### AP02 — Graph initialization + bundle adjustment

AP02 builds a marker/camera observation graph, initializes poses from a
reference marker, and refines the joint problem with bundle adjustment. Useful
controls include the reference marker, frame budgets, graph initialization,
reprojection model, solver budget, robust loss and loss scale. Combined BA is
the primary result; static-only BA remains a diagnostic.

### AP03 — SfM / multi-camera calibration

AP03 jointly registers one image per static camera and the moving-camera frames
in COLMAP, estimates metric scale from the configured marker set, and exports
the registered static-camera poses. Useful controls include matching, compute
mode, mapper support, feature-limit policy, scale inputs, marker area, scale
RANSAC and marker selection. Multi-marker scale is primary; Single is a shared
diagnostic.

The common evaluation anchor is selected and frozen once during preflight. No
method may silently substitute another anchor, and ground truth is never used
to calibrate the rig.

From the Wizard, choose **Start a new calibration**, select prepared or
simulation input, add AP01/AP02/AP03 to the method queue, edit advanced options
if needed, and review the final queue. From the CLI, validate or run the saved
configuration without changing it:

```bash
rigcal --config workspace/<dataset>/queue/queue.yaml --dry-run
rigcal --config workspace/<dataset>/queue/queue.yaml --yes
```

## Pipeline

The terminal shows this central order:

```text
1 Capture or import
  → 2 Normalize, extract frames and resolve intrinsics
  → 3 Validate the dataset
  → 4 Detect ArUco markers and write debug images
  → 5 Check observation quality and select references
  → 6 Run one calibration method and its internal substages
  → 7 Evaluate every primary method on a common anchor
  → 8 Build the cross-method comparison
  → 9 Atomically publish the experiment and summary
```

Video extraction, Gazebo capture and ArUco detection report
`current/expected frames`, percentage and elapsed time. Method, experiment and
batch times are shown separately. Full COLMAP and optimizer output remains in
log files; the terminal shows only useful progress, warnings and summaries.

## Storage layout

Real source kinds (`video`, `frames`, `rosbag`, `prepared`) remain metadata and
do not create path levels:

```text
results/real_vehicle/<rate>Hz/<experiment>/
results/real_vehicle/native_rate/<experiment>/
```

Simulation retains its factor grouping:

```text
results/simulation/<factor>/<value>/
```

An experiment owns exactly one immutable dataset:

```text
dataset.json
raw_images/
  static/
  moving/
  camera_info/
observations/
  shared_static_aruco_observations.csv
  shared_moving_aruco_observations.csv
  shared_all_aruco_observations.csv
  quality and selection reports
  debug_images/
metadata/
methods/
evaluations/
attempts/
```

A byte-identical repeat reuses the dataset. Different content under the same
experiment ID is rejected and requires a new ID. Calibration methods never
modify the canonical dataset.

The result front door is:

```text
RESULTS.txt
RESULTS.json
SUMMARY.json
COMPARISON.csv
COMPARISON.json
methods/<method>/<label>/
  RESULT.txt
  RESULT.json
  camera_extrinsics.csv
  camera_extrinsics_anchor.csv
  camera_extrinsics_anchor.json
  camera_extrinsics_anchor.yaml
  pairwise_camera_extrinsics.csv
  diagnostics/
  logs/
  provenance/
evaluations/
attempts/
visualization/
```

For simulation experiments the same folder additionally contains
`SECONDARY_CAMERA_MAP_RESULTS.txt` and
`SECONDARY_AP02_MARKER_MAP_RESULTS.txt`. `RESULTS.txt` is always the readable
front door; `SUMMARY.json` is the inventory index and `COMPARISON.csv/json`
remain machine-readable.

There are no separate `datasets`, `video/prepared`, `inputs/<hash>`,
`executions`, `current`, `00_INPUT` or `99_FINAL_RESULTS` levels in public
storage.

`camera_extrinsics.csv` records the reference frame and transform convention.
The three `camera_extrinsics_anchor.*` files provide the same primary camera
estimates as 6-DoF poses in the one common evaluation-marker frame. AP02's
internal reference marker remains a separate method setting. The common anchor
is selected automatically from repeat-supported observations or manually once
after preflight from every actually detected marker ID. A warned manual marker
is allowed without a hidden fallback; unsupported method exports are reported
as unavailable.

`View results` can derive missing anchor exports from already published
artifacts without rerunning AP01--AP03. When a scaled AP03 Multi sparse model is
available, it also prepares `visualization/` and can open an isolated RViz 2
session. Each RViz window receives its own `ROS_DOMAIN_ID`, so several
experiments can remain open without mixing TF or marker topics. The displayed
point cloud is explicitly AP03/COLMAP context; all method poses keep separate
namespaces.

`diagnostics` retains all scientifically important intermediate artifacts;
`logs` contains complete process output; `provenance` contains requested and
resolved configs, config diff, commands, environment, manifest and timings.

An identical completed method/input fingerprint is skipped. Different
calibration-affecting settings receive a different automatic label.

## Results and failures

After every queue and batch, rigcal prints one table containing experiment,
method label, status, method/experiment/batch runtime, key metrics, canonical
result path and comparison path. Temporary workspace paths are not reported as
successful results.

`View results` reads only layout-v2 experiment summaries. It opens the
experiment-wide `RESULTS.txt`, shows all variants, runtime, camera coverage,
quality warnings and canonical paths, and can display a selected method
`RESULT.txt`. For simulation it also offers the direct GT and secondary
camera-/marker-map reports.

Result status is:

- `available`: one or more successful methods and no failed attempt;
- `partial`: successful methods plus failed attempts;
- `failed`: failed attempts only.

Failed work is stored under `attempts/` as incomplete/non-authoritative and
never replaces a valid method result. Concise cause codes include COLMAP sparse
model failure, preflight, timeout, configuration and optimizer failures.

## Resume and cleanup

Interrupted, selection-waiting and publication-failed work remains under
`workspace/temporary_runs`. Terminal successful/failed queues close
automatically.

`Cleanup storage` reviews three independent permanent-deletion groups in this
order: published `results` (including their embedded datasets), legacy/prepared
datasets and dataset caches, then temporary workspace runs/queues/batches.
Each group defaults to “no”; selected groups require a final typed `DELETE`
confirmation and are verified after deletion. Cleanup is blocked while another
rigcal process is active. `data_local` and `config/intrinsics` are never
selected or queried by this action.

The public code is package-first:

- `src/camera_rig_calibration/`: all active implementation;
- `run/rigcal.py`: thin source-checkout launcher;
- `run/README.md`: pipeline order;
- `src/calib_lab/`: reviewed Gazebo assets;
- `data_local/`: user recordings;
- `results/`: complete experiments with immutable inputs and scientific outputs;
- `workspace/`: temporary runs and internal reusable caches;
- `tests/`: software fixtures and contracts.

See [architecture](docs/architecture.md),
[configuration](docs/configuration.md), and the
[extension guide](docs/extensions.md).

# Architecture

## One pipeline, one package

All executable capture, preparation, observation, calibration, evaluation and
reporting code lives under `src/camera_rig_calibration`. `run/rigcal.py` is a
thin source-checkout launcher.

The central pipeline order is:

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

Numbers are terminal/documentation order only. Persistent folders are named by
their scientific role.

## UI composition and extension boundary

`wizard.py` remains the stable navigation and product-policy facade. Focused UI
services live below `src/camera_rig_calibration/ui/`:

- `method_settings.py` declares the editable parameters and their help text;
- `method_editor.py` owns interactive editing of one method queue row;
- `wizard_method_jobs.py` and `wizard_method_queue.py` own method creation,
  guided selection and queue editing;
- `wizard_new_flow.py`, `wizard_saved_flow.py` and `wizard_review.py` own the
  new-run, saved-setup and review flows;
- `wizard_media.py`, `wizard_real_input.py`, `wizard_prepared.py` and
  `wizard_simulation.py` own their respective input workflows;
- `run_management.py` owns resume, interrupt and incomplete-run handling;
- `result_browser.py` owns result discovery and visualization dispatch;
- `storage_cleanup.py` and `intrinsics.py` own their respective maintenance
  screens.

The facade injects navigation and policy hooks into those services, so existing
CLI entry points remain stable. New UI screens should be added as focused
modules here and exposed through a thin facade function. New calibration
methods still enter through the registries described in `extensions.md`; the UI
does not duplicate scientific implementations.

## Responsibility-oriented facades

Stable import paths remain intentionally small. Their implementations are
grouped by responsibility:

- `queueing.py` composes models, preflight, execution, evaluation and dataset
  publication from `queue_services/`;
- `runtime.py` composes environment, observation, cache, command and stage
  services from `runtime_services/`;
- `preflight.py` composes job checks, raw-marker inventory, common-anchor
  resolution and queue reporting from `preflight_services/`;
- `evaluation/reporting.py` re-exports core I/O, configuration, quality,
  method, real-data and simulation reporting modules;
- `publication.py` re-exports dataset, method, inventory and transactional
  publication modules;
- `observations.py` re-exports candidate construction, ranking, resolution and
  configuration freezing;
- `input/intrinsics_calibration.py` and `input/preparation.py` retain their
  established entry points while delegating to focused workflow modules;
- AP01 `core.py` and AP02 `initialize.py` remain scientific compatibility
  facades over focused algorithm modules.

Late-bound `WizardBindings`, `QueueBindings`, `RuntimeBindings`,
`ReportingBindings`, `PreflightDependencies` and `AP01CoreBindings` read hooks
from their public facades at execution time. The existing product-policy stack
and tests may therefore patch the established import paths without creating
cycles between implementation modules.

Every productive Python module below `src/camera_rig_calibration/` is limited
to 999 lines by `tools/check_source_layout.py`; there are no active-package
legacy exceptions. New modules should normally stay below 850 lines and split
earlier when they acquire a second independent responsibility.

## Queue and batch model

A strict schema-v5 queue contains one experiment and one or more independent
method rows. A `rigcal_batch` contains an ordered list of those queues. The
batch is the Cartesian product `experiments × method variants`; capture,
normalization and raw ArUco detection happen once per experiment.

Queue preflight validates every method without changing its scientific
parameters. Independent failures do not stop later rows. A successful method
is authoritative and published independently; a failure is retained only as
an incomplete/non-authoritative attempt.

AP02 Static-only and AP03 Single are diagnostics. AP02 Combined and AP03 Multi
are the respective primary results in the common comparison.

## Immutable dataset contract

An experiment ID owns exactly one content fingerprint:

```text
results/real_vehicle/<rate>Hz/<experiment>/
results/real_vehicle/native_rate/<experiment>/
results/simulation/<factor>/<value>/
```

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
  debug_images/
  quality and selection reports
metadata/
methods/
evaluations/
attempts/
```

`video`, `frames`, `rosbag` and `prepared` are source metadata, not directory
levels. A byte-identical dataset is reused. Different content under the same
experiment ID is rejected and requires a new ID. Method publication never
modifies the dataset.

## Result front door

```text
results/<category>/<same rate-or-factor>/<experiment>/
  RESULTS.txt
  RESULTS.json
  SUMMARY.json
  COMPARISON.csv
  COMPARISON.json
  methods/<method>/<label>/
    RESULT.txt
    RESULT.json
    camera_extrinsics.csv
    pairwise_camera_extrinsics.csv
    diagnostics/
    logs/
    provenance/
  evaluations/
  attempts/
```

The method label comes from the queue (`baseline`, `variant2`, and so on). The
same label and fingerprint are skipped. The same label with a different
configuration is a clear conflict; the user must choose another label.

`camera_extrinsics.csv` states the reference frame and transform convention.
Native method outputs and all intermediate scientific artifacts remain under
`diagnostics`; complete subprocess output remains under `logs`; requested and
resolved configs, config diff, command list, environment, manifest and timings
remain under `provenance`.

`RESULTS.txt` is the human-readable experiment front door. `SUMMARY.json` is
the inventory index and `COMPARISON.csv/json` are machine-readable. Simulation
experiments add direct camera-pair GT results plus separate best-fit camera-map
and AP02 marker-map diagnostics.

`View results` indexes only root `SUMMARY.json` files with `layout_version: 2`,
then displays the public `RESULTS.txt` and method reports. It never reconstructs
state from workspace runs or method internals.

## Scientific ownership and reuse

- AP01 owns its Root Camera and moving-COLMAP relay.
- AP02 owns its graph Reference Marker.
- AP03 Single owns a diagnostic scale marker.
- AP03 Multi owns the primary robust marker set.
- Evaluation owns an independent common anchor.

AP01 and AP03 COLMAP caches live under `workspace/cache`; they are not public
results. A cache key includes the immutable input fingerprint and complete
COLMAP settings, but excludes downstream root/scale selections.

The stage implementations preserve transformation direction, ArUco corner
order, PnP and distortion models, COLMAP conventions, residual definitions,
triangulation and robust estimators.

## Runtime records

The terminal shows stage, method, experiment and batch elapsed time plus
meaningful frame or reconstruction progress. Verbose COLMAP and optimizer
output is written unchanged to log files. No ETA is fabricated.

Interrupted, selection-waiting and publication-failed queues remain resumable
under `workspace/temporary_runs`. Terminal successful or failed queues close
automatically.

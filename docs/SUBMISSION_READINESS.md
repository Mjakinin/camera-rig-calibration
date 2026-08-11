# Submission readiness

This checklist defines the final handoff boundary for the university submission.
It is intentionally conservative: scientific estimates are not changed during
submission cleanup.

## Scientific boundary

- AP01, AP02 and AP03 calibration mathematics live behind explicit immutable
  method contracts and fingerprints.
- Ground truth is forbidden during calibration and is used only by simulation
  evaluation.
- Real-data results without an independent physical reference are reported as
  observability, reconstruction, reprojection and metric-consistency evidence,
  not as absolute real-world accuracy.
- AP02 disconnected components remain separate; transforms between components
  are reported as not observable.
- AP03 Multi is the primary AP03 result. AP03 Single is diagnostic.
- A completed AP03 reconstruction is not automatically a valid calibration.
  The existing 10% relative scale-dispersion gate is authoritative for metric
  validity. Results above the gate remain auditable artifacts but are marked
  `calibration_status=rejected_by_quality_gate` and
  `deployment_eligible=false`.
- Partial full-rig coverage is diagnostic and is never silently promoted to a
  deployment-ready full-rig result.

## Frozen simulation evidence

The 25 historical simulation ablations remain frozen at commit
`8f9dcea1e8b3189b3c195db2cafe65d5b0e5756b`. They must be labelled historical
Main evidence. `docs/FROZEN_SIMULATION_ABLATIONS.md` defines the reuse and
parity rules. Do not rerun the full study merely to reproduce those numbers.

## Runtime prerequisites

Python dependencies are installed from the project extras. A standalone
(non-ROS) development environment should use:

```bash
python3 -m pip install -e ".[scientific,standalone,dev]"
```

External executables are intentionally not Python package dependencies:

- COLMAP is required by AP01/AP03 reconstruction.
- FFmpeg/ffprobe is required for video geometry and video import.
- ROS 2 Humble, rosbag2/MCAP and RViz 2 are required only for the corresponding
  ROS input/visualization workflows.
- Gazebo/Ignition is required only for simulation capture.

Run `rigcal`, choose **Check installation**, and resolve every component needed
for the workflow that will be demonstrated.

## Required final verification

Run from the repository root on the submission commit:

```bash
python3 -m pip check
python3 -m compileall -q src run tests
PYTHONPATH="$PWD/src" python3 -m pytest \
  -m "not slow and not requires_colmap and not requires_ros" -q
git diff --check
```

Also run the focused submission/status regression tests:

```bash
PYTHONPATH="$PWD/src" python3 -m pytest -q \
  tests/test_submission_quality_semantics.py \
  tests/test_ap03_camera_model_sensitivity_policy.py \
  tests/test_real_partial_evaluation_policy.py \
  tests/test_ap02_partial_reference_reporting_policy.py \
  tests/test_ap02_native_rviz.py
```

For a machine with COLMAP available, perform a smoke check with an already
prepared small dataset rather than rerunning the frozen simulation ablations.

## Result reconciliation

Reporting-only reconciliation is allowed after submission-quality fixes because
it reads existing native method artifacts and must not rerun AP01/AP02/AP03.
Before and after reconciliation, native calibration hashes should remain
unchanged. The public front door may change only to correct status, quality,
anchor-export or visualization metadata.

## Maintainability boundary

The active package is registry-based and extension points are documented in
`docs/extensions.md`. `run/rigcal.py` is intentionally a thin launcher.

Some mature orchestration modules (`wizard.py`, `publication.py` and
`evaluation/reporting.py`) are larger than ideal because they accumulated the
final product workflow and compatibility/reporting logic. Splitting those files
immediately before submission would create a high regression risk without
changing the scientific result. Treat that refactor as post-submission technical
debt. New scientific algorithms should not be added to those modules; they
belong in dedicated method components/contracts.

Likewise, large files under `parity/` are frozen audit evidence rather than the
runtime application architecture. Do not delete them merely to reduce line
counts; their purpose is reproducibility and historical-method parity.

## Final handoff rule

Do not merge a submission-cleanup branch solely because it is documentation- or
reporting-only. Merge only after the verification commands above pass on the
exact commit to be submitted.

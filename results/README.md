# Results layout

Each experiment directory is both the immutable scientific input record and
the published calibration result. This is intentional: a result cannot be
interpreted or reproduced without knowing exactly which images, intrinsics,
observations, route, and detector settings produced it.

Start with these files; the UI's **View results** screen reads the same front
door:

- `RESULTS.txt` / `RESULTS.json`: human- and machine-readable experiment result;
- `SUMMARY.json`: indexed experiment and method inventory;
- `COMPARISON.csv` / `COMPARISON.json`: common-anchor method comparison;
- `methods/<method>/<label>/RESULT.txt`: one method variant;
- `methods/<method>/<label>/camera_poses_6dof.csv`: canonical static-camera
  6DOF poses.

The remaining folders are structured evidence, not additional competing
results:

| Folder | Contents | Why it exists |
|---|---|---|
| `raw_images/` | Canonical static and moving images plus camera information | Immutable calibration input and reproducibility |
| `observations/` | ArUco tables, accepted/rejected rows, selections, and annotated debug images | Detection and selection audit |
| `metadata/` | Source, preparation, identity, detector configuration, and retry records | Provenance and resumability |
| `methods/` | One directory per method and parameter label | Native output, canonical 6DOF result, diagnostics, logs, provenance |
| `evaluations/` | Common-anchor and real/simulation quality reports | Fair method-independent comparison |
| `attempts/` | Compact records of successful or failed executions | Resume and failure diagnosis without presenting failures as results |
| `visualization/` | Generated pose-only RViz session | Inspection, not calibration input |

Large local trees are expected for a full audit profile. Most space comes from
annotated detection images, detector-attempt snapshots, COLMAP reconstruction
data, and the separately undistorted AP03 image set. AP03's COLMAP input images
are hard-linked to `raw_images/` when the filesystem supports it: they appear
in both logical locations but do not consume a second set of disk blocks. The
undistorted images are genuinely new data and therefore do consume space.

The three local 3 Hz real-vehicle experiments currently occupy roughly
2.1–2.4 GiB each with complete diagnostics. Only deliberately selected compact
reference artifacts are tracked by Git; raw images and large regenerable
diagnostics remain local. Do not manually delete individual files inside a
published experiment, because that would break its completeness contract. Use
the application's **Cleanup storage** screen for temporary workspace/cache
data, or archive/delete an entire experiment as one unit after preserving the
front-door results you need.

# Published results (`results`)

`results/` contains the **published scientific artifact** of completed experiments.

An experiment directory intentionally keeps the immutable calibration input record together with the published method outputs. A calibration result is not reproducible without knowing which images, intrinsics, observations, detector settings, route, and resolved configuration produced it.

## Start here

For most users, the front door is:

```text
RESULTS.txt
```

The main machine-readable files are:

- `RESULTS.json` — experiment result;
- `SUMMARY.json` — experiment/method inventory;
- `COMPARISON.csv` / `COMPARISON.json` — common-anchor comparison.

A method variant has its own readable result below:

```text
methods/<method>/<label>/RESULT.txt
```

and canonical static-camera poses below:

```text
methods/<method>/<label>/camera_poses_6dof.csv
```

The application uses the same files when you choose:

```text
rigcal
  → View results
```

## Experiment layout

A published experiment normally contains:

| Folder | Contents | Purpose |
|---|---|---|
| `raw_images/` | canonical static/moving images and camera information | immutable calibration input |
| `observations/` | ArUco tables, accepted/rejected rows, selections, debug images | detection/selection audit |
| `metadata/` | source, identity, detector configuration, retry records | provenance and resumability |
| `methods/` | one directory per method and parameter label | canonical outputs, diagnostics, logs, provenance |
| `evaluations/` | common-anchor and real/simulation quality reports | method-independent comparison |
| `attempts/` | successful/failed execution records | failure diagnosis without replacing valid results |
| `visualization/` | generated RViz/pose visualization artifacts | inspection only |

Per-method directories may additionally contain:

```text
RESULT.json
canonical_method_result.json
camera_extrinsics.csv
camera_extrinsics_anchor.csv
camera_extrinsics_anchor.json
camera_extrinsics_anchor.yaml
pairwise_camera_extrinsics.csv
diagnostics/
logs/
provenance/
```

Simulation experiments may also publish dedicated Ground-Truth/secondary reports.

## Result status

The public result state distinguishes:

- `available` — one or more successful methods and no failed attempt;
- `partial` — successful methods plus failed attempts;
- `failed` — failed attempts only.

A failed attempt is diagnostic evidence. It does **not** replace an already valid method result.

## Common anchor and comparison

Primary method results are exported onto one frozen common evaluation-marker frame when supported. AP02's internal reference marker is a separate setting.

The common comparison exists to make AP01/AP02/AP03 outputs easier to inspect in a consistent frame. Ground Truth, when available in simulation, is used for evaluation only and is not fed back into calibration.

## RViz

`View results` can derive missing anchor exports from published artifacts without rerunning the calibration methods. When a suitable scaled AP03 Multi sparse model is available, it can also prepare `visualization/` and open an isolated RViz 2 session.

Visualization is an inspection layer, not calibration input.

## Storage and cleanup

A complete audit profile can be large. Typical space consumers are:

- annotated detection images;
- detector-attempt snapshots;
- COLMAP reconstruction data;
- undistorted AP03 images;
- full diagnostics and logs.

Do **not** manually delete individual files from the middle of a published experiment: that breaks the experiment's completeness/reproducibility contract.

Use the application's **Cleanup storage** workflow for managed cleanup, or archive/delete an entire experiment deliberately after preserving the front-door results you need.

Only intentionally selected compact reference artifacts are expected to be tracked by Git; large raw/regenerable data can remain local.

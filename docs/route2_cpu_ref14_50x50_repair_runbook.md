# Route-2 repair runbook: AP02, then AP01

These commands run in WSL. They reuse the published immutable Route-2 input.
They do not start Gazebo/ROS capture and do not run AP03.

## 1. Open the repository and activate the environment

```bash
REPO="/mnt/c/Users/maxim/Desktop/Application of Robotics and Autonomous Systems/camera-rig-calibration"
cd "$REPO"

test "$(git branch --show-current)" = "final/rigcal-v4.3"

if [ -f "${HOME}/.venvs/rigcal/bin/activate" ]; then
  source "${HOME}/.venvs/rigcal/bin/activate"
else
  python3 -m venv "${HOME}/.venvs/rigcal"
  source "${HOME}/.venvs/rigcal/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install -e ".[dev,scientific,standalone]"
python -c "import numpy, scipy; print('NumPy', numpy.__version__, 'SciPy', scipy.__version__)"
```

No existing rigcal-specific WSL environment was present at final validation.
The commands therefore create this portable environment outside the repository:

```text
/home/maxim/.venvs/rigcal
```

The commands intentionally do not run `git pull`: this shared working tree
contains the uncommitted repair. Pull only after the repair has been committed.

For a later clean checkout, the normal branch update is:

```bash
git checkout final/rigcal-v4.3
git pull --ff-only
```

## 2. Run the focused repair tests

```bash
pytest -q -p no:cacheprovider \
  tests/test_repair_contracts.py \
  tests/test_storage_v5.py \
  tests/test_baseline_method_repairs.py \
  tests/test_ap02_frame_selection.py \
  tests/test_method_contracts.py \
  tests/test_anchor_export.py \
  tests/test_simulation_reconcile_v2.py
```

These cover immutable dataset identity, real input-content conflicts, AP02
main-compatible initialization and 50/50 configuration, hierarchical AP01
aggregation, diagnostic-versus-deployment status, publication, the
single-method rerun contract, anchor export and reconcile.

## 3. Rerun only AP02

```bash
EXP="results/simulation/baseline/route2_cpu_ref14_50x50"

rigcal rerun-method \
  --experiment "$EXP" \
  --method ap02 \
  --variant baseline \
  --reuse-prepared-input \
  --reconcile-after
```

This command enforces the saved AP02 baseline contract: reference marker 14,
static BA `max_nfev=50`, combined BA `max_nfev=50`, no retry. It reuses the
published images and compatible observations. It does not run AP01, AP03 or
capture.

## 4. Inspect AP02

```bash
cat "$EXP/methods/ap02/baseline/RESULT.txt"

cat \
  "$EXP/methods/ap02/baseline/diagnostics/method/graph_initialization/with_moving/AP02_MAIN_COMPAT_INITIALIZATION_PARITY.json"

find "$EXP/evaluations" -name camera_pairwise_gt.csv -print
find "$EXP/evaluations" -name anchor_camera_gt.csv -print

python - "$EXP" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
result = json.loads(
    (root / "methods/ap02/baseline/RESULT.json").read_text()
)
print("reference_marker_id:", result.get("reference_marker_id"))
print("quality_status:", result.get("quality_status"))
for name in ("camera_pairwise_gt.csv", "anchor_camera_gt.csv"):
    matches = list((root / "evaluations").rglob(name))
    counts = []
    for path in matches:
        rows = list(csv.DictReader(path.open()))
        counts.append(
            (
                str(path),
                sum(
                    row.get("method") == "ap02"
                    and row.get("label") == "baseline"
                    for row in rows
                ),
            )
        )
    print(name, counts)
PY

sed -n '1,240p' "$EXP/RESULTS.txt"
```

For a complete finite AP02 result, the evaluation must contain six
camera-to-camera pairs and four anchor-relative camera rows. Solver and RMSE
quality are reported honestly and are not forced.

## 5. Rerun only AP01 with matching CPU intermediates

```bash
rigcal rerun-method \
  --experiment "$EXP" \
  --method ap01 \
  --variant baseline \
  --reuse-prepared-input \
  --reuse-matching-intermediates \
  --reconcile-after
```

The command verifies the input and COLMAP fingerprints before reusing AP01's
CPU-COLMAP reconstruction and metric scale. It reruns candidate grouping,
hierarchical Direct/Relay aggregation, status selection, publication,
evaluation, reports and RViz derivation. It does not run AP02, AP03 or capture.

## 6. Inspect AP01

```bash
cat "$EXP/methods/ap01/baseline/RESULT.txt"
sed -n '1,20p' "$EXP/methods/ap01/baseline/camera_extrinsics.csv"
sed -n '1,20p' "$EXP/methods/ap01/baseline/camera_extrinsics_accepted.csv"

python - "$EXP" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
result = json.loads(
    (root / "methods/ap01/baseline/RESULT.json").read_text()
)
for camera, status in sorted(result.get("camera_statuses", {}).items()):
    print(
        camera,
        "estimate_status=" + str(status.get("estimate_status")),
        "quality_status=" + str(status.get("quality_status")),
        "deployment_eligible=" + str(status.get("deployment_eligible")),
        "evaluation_status=" + str(status.get("evaluation_status")),
    )
chains = json.loads(
    (
        root
        / "methods/ap01/baseline/diagnostics/method/"
          "static_extrinsics/AP01_RELAY_CHAIN_DIAGNOSTICS.json"
    ).read_text()
)
for camera, rows in sorted(chains["targets"].items()):
    print(camera, "independent_chains=", len(rows))
for name in ("camera_pairwise_gt.csv", "anchor_camera_gt.csv"):
    matches = list((root / "evaluations").rglob(name))
    counts = []
    for path in matches:
        rows = list(csv.DictReader(path.open()))
        counts.append(
            (
                str(path),
                sum(
                    row.get("method") == "ap01"
                    and row.get("label") == "baseline"
                    for row in rows
                ),
            )
        )
    print(name, counts)
PY
```

Four finite scientific estimates produce six pairwise GT rows and four
anchor-relative rows even if one or more non-root cameras are explicitly
blocked from deployment.

## 7. Reconcile derived reports only

```bash
rigcal reconcile --experiment "$EXP"

test -f "$EXP/RESULTS.txt"
test -f "$EXP/RESULTS.json"
test -f "$EXP/COMPARISON.csv"
test -f "$EXP/COMPARISON.json"
test -f "$EXP/SUMMARY.json"
find "$EXP/methods" -name camera_extrinsics_anchor.json -print
test -f "$EXP/visualization/visualization_manifest.json"
```

`rigcal reconcile` derives reports, GT tables, anchor exports and RViz
artifacts from published results. It never starts AP01, AP02, AP03, COLMAP or
capture.

## 8. Full validation

```bash
pytest -p no:cacheprovider
python -m build
git diff --check
```

The old failed AP02 attempt and the superseded AP01 result remain under
`attempts/`. The current public results are updated atomically only after a
successful method execution and publication.

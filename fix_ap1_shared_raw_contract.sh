#!/usr/bin/env bash
set -eo pipefail

# Patch AP01 cleanly so it can export the canonical shared raw dataset,
# without changing AP01's existing internal paths or breaking the frozen pipeline.
#
# Adds:
#   run/bus_real_data/00_sync_shared_raw_dataset_from_ap1.py
#   run/bus_real_data/99_search_ap1_path_dependencies.sh
#
# Patches:
#   run/bus_real_data/run_approach1_full_pipeline.sh
# by inserting a sync step after AP01 has captured static + moving raw images.
#
# Does NOT:
#   - change AP01 output directories
#   - change AP01 COLMAP input path
#   - delete AP01 outputs
#   - force recapture
#
# Run:
#   cd /workspaces/project
#   bash fix_ap1_shared_raw_contract.sh

PROJECT_ROOT="/workspaces/project"
if [ -d "$PROJECT_ROOT" ]; then
  cd "$PROJECT_ROOT"
fi

echo "=== AP01 shared raw contract patch ==="

mkdir -p run/bus_real_data

cat > run/bus_real_data/00_sync_shared_raw_dataset_from_ap1.py <<'PY'
#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import yaml


DEFAULT_AP1_ROOT = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain")
DEFAULT_RAW_ROOT = Path("results/bus_real_data/00_raw_images/bus_real_data_ref_marker_v1")
DEFAULT_INTRINSICS = Path("src/calib_lab/bus_real_data/config/camera_intrinsics_by_camera.yaml")

STATIC_CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]
MOVING_CAMERA = "moving_calib_camera"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Path, dst: Path, overwrite: bool):
    if not src.exists():
        raise RuntimeError(f"Missing source file: {src}")
    if dst.exists() and not overwrite:
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_dir_contents(src_dir: Path, dst_dir: Path, overwrite: bool):
    if not src_dir.exists():
        raise RuntimeError(f"Missing source directory: {src_dir}")
    ensure_dir(dst_dir)

    if overwrite:
        for p in dst_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    count = 0
    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        if src.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        count += 1

    return count


def intrinsics_to_camera_info_json(intr, width=None, height=None):
    width = int(width if width is not None else intr["width"])
    height = int(height if height is not None else intr["height"])

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    d = [float(x) for x in intr.get("distortion", [0, 0, 0, 0, 0])]

    return {
        "width": width,
        "height": height,
        "distortion_model": "plumb_bob",
        "d": d,
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "source": "AP01 camera_intrinsics_by_camera.yaml",
    }


def write_json(path: Path, data):
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ap1-root", default=str(DEFAULT_AP1_ROOT))
    ap.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--intrinsics", default=str(DEFAULT_INTRINSICS))
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing files in the shared raw dataset.")
    args = ap.parse_args()

    ap1_root = Path(args.ap1_root)
    raw_root = Path(args.raw_root)
    intrinsics_path = Path(args.intrinsics)

    static_src = ap1_root / "01_static_a4_marker_detection" / "raw_images"
    moving_src = ap1_root / "03_moving_camera_sequence" / "images"
    route_src = ap1_root / "03_moving_camera_sequence" / "route_commanded.csv"

    if not static_src.exists():
        raise RuntimeError(
            f"Missing AP01 static raw images: {static_src}\n"
            "Run AP01 step 01 first."
        )

    if not moving_src.exists():
        raise RuntimeError(
            f"Missing AP01 moving raw images: {moving_src}\n"
            "Run AP01 step 04 first."
        )

    if not intrinsics_path.exists():
        raise RuntimeError(f"Missing intrinsics YAML: {intrinsics_path}")

    ensure_dir(raw_root)
    ensure_dir(raw_root / "static")
    ensure_dir(raw_root / "moving")
    ensure_dir(raw_root / "camera_info")
    ensure_dir(raw_root / "ap1_metadata")

    # Copy static snapshots.
    copied_static = []
    for cam in STATIC_CAMERAS:
        src = static_src / f"{cam}.png"
        dst = raw_root / "static" / f"{cam}.png"
        copy_file(src, dst, overwrite=args.overwrite)
        copied_static.append(str(dst))

    # Copy real AP01 moving sequence.
    moving_count = copy_dir_contents(moving_src, raw_root / "moving", overwrite=True)

    # Copy AP01 route metadata so AP02/AP03 can know commanded camera poses if needed,
    # without reading AP01 internals directly later.
    if route_src.exists():
        copy_file(route_src, raw_root / "ap1_metadata" / "route_commanded.csv", overwrite=True)

    # Camera info JSON from AP01 intrinsics YAML.
    intr = yaml.safe_load(intrinsics_path.read_text())

    for cam in STATIC_CAMERAS:
        if cam not in intr:
            raise RuntimeError(f"Missing {cam} in intrinsics YAML")
        write_json(raw_root / "camera_info" / f"{cam}.json", intrinsics_to_camera_info_json(intr[cam]))

    # Moving camera: AP01 COLMAP step uses fixed 1280x720 PINHOLE with fx/fy from HFOV.
    # Use a representative PINHOLE camera_info matching AP01's COLMAP assumptions.
    # This is intentionally stored in the shared raw dataset so AP02/AP03 do not need to
    # import AP01 code.
    moving_info = {
        "width": 1280,
        "height": 720,
        "distortion_model": "plumb_bob",
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "k": [929.46709573, 0.0, 640.0, 0.0, 929.46709573, 360.0, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [929.46709573, 0.0, 640.0, 0.0, 0.0, 929.46709573, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "fx": 929.46709573,
        "fy": 929.46709573,
        "cx": 640.0,
        "cy": 360.0,
        "source": "AP01 COLMAP PINHOLE assumption",
    }
    write_json(raw_root / "camera_info" / f"{MOVING_CAMERA}.json", moving_info)

    manifest = [
        "Dataset: bus_real_data_ref_marker_v1",
        "Purpose: Shared raw-image dataset for AP01/AP02/AP03 fair comparisons.",
        "",
        "Canonical source:",
        f"- static/: copied from {static_src}",
        f"- moving/: copied from {moving_src}",
        f"- camera_info/: generated from {intrinsics_path} and AP01 moving COLMAP assumptions",
        "",
        "Contract:",
        "- This folder contains raw inputs and minimal metadata only.",
        "- Approach-specific detections/results stay in each approach result folder.",
        "- AP01 keeps its original internal paths; this script only exports a shared copy.",
        "- AP02/AP03 should read this folder as raw input.",
        "",
        "Counts:",
        f"- static images: {len(copied_static)}",
        f"- moving images: {moving_count}",
        "",
    ]
    (raw_root / "MANIFEST.txt").write_text("\n".join(manifest))

    print("[OK] synced AP01 raw inputs to shared raw dataset")
    print("[OK] raw root:", raw_root)
    print("[OK] static images:", len(copied_static))
    print("[OK] moving images:", moving_count)
    print("[OK] manifest:", raw_root / "MANIFEST.txt")


if __name__ == "__main__":
    main()
PY

chmod +x run/bus_real_data/00_sync_shared_raw_dataset_from_ap1.py
echo "[OK] wrote run/bus_real_data/00_sync_shared_raw_dataset_from_ap1.py"

cat > run/bus_real_data/99_search_ap1_path_dependencies.sh <<'SH'
#!/usr/bin/env bash
set -eo pipefail

cd /workspaces/project

echo "=== AP01/AP02/AP03 path dependency search ==="
echo

echo "[1] Hardcoded AP01 moving sequence references:"
grep -R "03_moving_camera_sequence" -n run src results/bus_real_data/*.md results/bus_real_data/*/README.md 2>/dev/null || true
echo

echo "[2] Hardcoded AP01 static detection references:"
grep -R "01_static_a4_marker_detection" -n run src results/bus_real_data/*.md results/bus_real_data/*/README.md 2>/dev/null || true
echo

echo "[3] Hardcoded AP01 result root references:"
grep -R "01_marker_direct_relay_multimarker_multichain" -n run src 2>/dev/null || true
echo

echo "[4] Shared raw image dataset references:"
grep -R "00_raw_images\|bus_real_data_ref_marker_v1" -n run src results/bus_real_data/*.md 2>/dev/null || true
echo

echo "[5] Stale AP02-internal raw dataset references:"
grep -R "02_ref_marker_graph_ba/01_dataset\|AP02_ROOT / \"01_dataset\"\|AP02_ROOT / '01_dataset'" -n run src 2>/dev/null || true
echo

echo "[6] Old stale run3/minimal refs:"
grep -R "05_moving_camera_sequence_run3\|06_colmap_moving_sequence_run3\|minimal_world\|beintelli_bus_model" -n run src results 2>/dev/null || true
echo

echo "[7] AP02 must not import AP01 internals directly:"
if [ -d run/bus_real_data/approach2_ref_marker_graph_ba ]; then
  grep -R "01_marker_direct_relay\|AP1_ROOT\|AP01_ROOT\|moving_detections.csv\|01_static_a4_marker_detection" -n run/bus_real_data/approach2_ref_marker_graph_ba || true
else
  echo "[INFO] AP02 folder not present yet."
fi
echo

echo "=== Done ==="
SH

chmod +x run/bus_real_data/99_search_ap1_path_dependencies.sh
echo "[OK] wrote run/bus_real_data/99_search_ap1_path_dependencies.sh"

RUNNER="run/bus_real_data/run_approach1_full_pipeline.sh"
if [ -f "$RUNNER" ]; then
  echo "=== Patch AP01 full runner to export shared raw dataset ==="
  python3 - <<'PY'
from pathlib import Path

runner = Path("run/bus_real_data/run_approach1_full_pipeline.sh")
s = runner.read_text()

sync_line = "run_step 04b_sync_shared_raw_dataset_from_ap1 run/bus_real_data/00_sync_shared_raw_dataset_from_ap1.py"

if sync_line in s:
    print("[OK] AP01 runner already contains sync step.")
else:
    anchor = "run_step 05_detect_moving_a4_markers run/bus_real_data/05_detect_moving_a4_markers.py"
    if anchor not in s:
        raise SystemExit("[ERROR] Could not find AP01 step 05 anchor in run_approach1_full_pipeline.sh")

    s = s.replace(
        anchor,
        sync_line + "\n" + anchor
    )

    runner.write_text(s)
    print("[OK] inserted AP01 shared raw sync step before AP01 moving detection.")
PY
else
  echo "[WARN] AP01 full runner not found: $RUNNER"
fi

echo
echo "=== Run dependency search ==="
bash run/bus_real_data/99_search_ap1_path_dependencies.sh

echo
echo "=== Optional immediate sync if AP01 raw outputs already exist ==="
if [ -d "results/bus_real_data/01_marker_direct_relay_multimarker_multichain/01_static_a4_marker_detection/raw_images" ] && \
   [ -d "results/bus_real_data/01_marker_direct_relay_multimarker_multichain/03_moving_camera_sequence/images" ]; then
  python3 run/bus_real_data/00_sync_shared_raw_dataset_from_ap1.py --overwrite
else
  echo "[INFO] AP01 raw outputs not complete yet. Sync will run during full AP01 pipeline after step 04."
fi

echo
echo "[OK] AP01 shared raw contract patch complete."
echo
echo "Next recommended checks:"
echo "  git diff -- run/bus_real_data/run_approach1_full_pipeline.sh run/bus_real_data/00_sync_shared_raw_dataset_from_ap1.py"
echo "  bash run/bus_real_data/99_search_ap1_path_dependencies.sh"

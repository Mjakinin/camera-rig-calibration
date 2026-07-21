#!/usr/bin/env python3
from pathlib import Path
import shutil
import json

ROOT = Path("results/bus_real_data/ablation/moving_cam/fov")
PREP = ROOT / "00_prepared_datasets"
OBS = ROOT / "01_shared_observations"

VARIANTS = [
    "fov_40deg",
    "fov_69deg_baseline",
    "fov_100deg",
    "fov_140deg_extreme",
]

def copytree_clean(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

for v in VARIANTS:
    dst = ROOT / v
    raw = PREP / v / "raw_images"
    obs = OBS / v

    if not raw.exists():
        raise SystemExit(f"[ERROR] missing raw: {raw}")
    if not obs.exists():
        raise SystemExit(f"[ERROR] missing obs: {obs}")

    dst.mkdir(parents=True, exist_ok=True)
    copytree_clean(raw, dst / "raw_images")
    copytree_clean(obs, dst / "aruco_observations")

    (dst / "VARIANT_METADATA.json").write_text(json.dumps({
        "variant": v,
        "ablation_scope": "moving_cam",
        "ablation_study": "fov",
        "raw_images": str(dst / "raw_images"),
        "aruco_observations": str(dst / "aruco_observations"),
    }, indent=2))

    print("[OK]", v)

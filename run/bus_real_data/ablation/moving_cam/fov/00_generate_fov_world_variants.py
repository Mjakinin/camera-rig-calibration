#!/usr/bin/env python3
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import math
import copy

SRC_WORLD = Path("src/calib_lab/bus_real_data/worlds/ablation/moving_cam/res/bus_real_data_moving_camera_res_1280x720_baseline.sdf")
FALLBACK_WORLD = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")

OUT_WORLD_DIR = Path("src/calib_lab/bus_real_data/worlds/ablation/moving_cam/fov")
OUT_INFO_DIR = Path("results/bus_real_data/ablation/moving_cam/fov/00_world_variants")

BASE_INFO = Path("results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images/camera_info/moving_calib_camera.json")

WIDTH = 1280
HEIGHT = 720

VARIANTS = [
    ("fov_40deg", 40.0),
    ("fov_69deg_baseline", 69.1),
    ("fov_100deg", 100.0),
    ("fov_140deg_extreme", 140.0),
]

def source_world():
    if SRC_WORLD.exists():
        return SRC_WORLD
    if FALLBACK_WORLD.exists():
        return FALLBACK_WORLD
    raise FileNotFoundError(f"Missing source world:\n  {SRC_WORLD}\n  {FALLBACK_WORLD}")

def set_text(parent, tag, text):
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    node.text = str(text)
    return node

def sensor_is_moving(sensor):
    txt = sensor.attrib.get("name", "").lower()
    for e in sensor.iter():
        if e.text:
            txt += " " + e.text.lower()
    return "moving" in txt or "moving_calib_camera" in txt

def patch_world_hfov(src, dst, hfov_rad):
    tree = ET.parse(src)
    root = tree.getroot()

    changed = 0
    for sensor in root.iter("sensor"):
        if not sensor_is_moving(sensor):
            continue

        cam = sensor.find("camera")
        if cam is None:
            continue

        set_text(cam, "horizontal_fov", f"{hfov_rad:.12f}")

        image = cam.find("image")
        if image is None:
            image = ET.SubElement(cam, "image")

        set_text(image, "width", WIDTH)
        set_text(image, "height", HEIGHT)
        changed += 1

    if changed == 0:
        cameras = list(root.iter("camera"))
        if len(cameras) == 1:
            cam = cameras[0]
            set_text(cam, "horizontal_fov", f"{hfov_rad:.12f}")

            image = cam.find("image")
            if image is None:
                image = ET.SubElement(cam, "image")

            set_text(image, "width", WIDTH)
            set_text(image, "height", HEIGHT)
            changed = 1

    if changed == 0:
        raise RuntimeError("Could not find moving camera sensor in SDF.")

    ET.indent(tree, space="  ")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return changed

def patch_camera_info(base_info, hfov_deg):
    info = copy.deepcopy(base_info)

    hfov_rad = math.radians(hfov_deg)
    fx = WIDTH / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx = WIDTH / 2.0
    cy = HEIGHT / 2.0

    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    info["width"] = WIDTH
    info["height"] = HEIGHT
    info["fx"] = fx
    info["fy"] = fy
    info["cx"] = cx
    info["cy"] = cy
    info["horizontal_fov_deg"] = hfov_deg
    info["horizontal_fov_rad"] = hfov_rad
    info["ablation_scope"] = "moving_cam"
    info["ablation_study"] = "fov"

    if isinstance(info.get("K"), list) and len(info["K"]) == 9:
        info["K"] = K
    if isinstance(info.get("P"), list) and len(info["P"]) == 12:
        info["P"] = P

    if isinstance(info.get("camera_matrix"), dict):
        info["camera_matrix"]["data"] = K
    if isinstance(info.get("projection_matrix"), dict):
        info["projection_matrix"]["data"] = P

    return info, fx

def main():
    src = source_world()
    print(f"[INFO] source world: {src}")

    if not BASE_INFO.exists():
        raise FileNotFoundError(BASE_INFO)

    base_info = json.loads(BASE_INFO.read_text())

    OUT_WORLD_DIR.mkdir(parents=True, exist_ok=True)
    OUT_INFO_DIR.mkdir(parents=True, exist_ok=True)

    for variant, hfov_deg in VARIANTS:
        hfov_rad = math.radians(hfov_deg)

        world_out = OUT_WORLD_DIR / f"bus_real_data_moving_camera_{variant}.sdf"
        changed = patch_world_hfov(src, world_out, hfov_rad)

        info, fx = patch_camera_info(base_info, hfov_deg)
        info_dir = OUT_INFO_DIR / variant
        info_dir.mkdir(parents=True, exist_ok=True)
        info_out = info_dir / "moving_calib_camera.json"
        info_out.write_text(json.dumps(info, indent=2))

        print(f"[OK] {variant}: hfov={hfov_deg:.2f} deg fx={fx:.3f} changed_cameras={changed}")
        print(f"     world: {world_out}")
        print(f"     info:  {info_out}")

    print("\n[DONE] FOV world variants generated.")

if __name__ == "__main__":
    main()

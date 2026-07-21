#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

import cv2

SRC = Path("results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1")
OUT_ROOT = Path("results/bus_real_data/ablation/moving_cam/res")

# Extreme but interpretable post-hoc moving-camera image resolution variants.
# Static camera images stay unchanged.
VARIANTS = {
    "moving_res_160x90_extreme_pixel": (160, 90),
    "moving_res_320x180_low": (320, 180),
    "moving_res_1280x720_baseline": (1280, 720),
    "moving_res_2560x1440_upscaled": (2560, 1440),
}

MOVING_CAMERA_INFO_NAMES = {
    "moving_calib_camera.json",
    "moving_camera.json",
}

def scale_camera_info(obj, new_w, new_h):
    old_w = int(obj.get("width", obj.get("image_width", 1280)))
    old_h = int(obj.get("height", obj.get("image_height", 720)))
    sx = new_w / old_w
    sy = new_h / old_h

    out = dict(obj)
    out["width"] = new_w
    out["height"] = new_h
    out["image_width"] = new_w
    out["image_height"] = new_h

    for k in ["fx", "cx"]:
        if k in out:
            out[k] = float(out[k]) * sx
    for k in ["fy", "cy"]:
        if k in out:
            out[k] = float(out[k]) * sy

    for key in ["K", "k"]:
        if key in out and len(out[key]) >= 9:
            K = list(map(float, out[key]))
            K[0] *= sx
            K[2] *= sx
            K[4] *= sy
            K[5] *= sy
            out[key] = K

    if "P" in out and len(out["P"]) >= 12:
        P = list(map(float, out["P"]))
        P[0] *= sx
        P[2] *= sx
        P[5] *= sy
        P[6] *= sy
        out["P"] = P

    out["ablation_resolution_source_width"] = old_w
    out["ablation_resolution_source_height"] = old_h
    out["ablation_resolution_width"] = new_w
    out["ablation_resolution_height"] = new_h
    out["ablation_note"] = "Only moving camera image resolution is changed. Static cameras remain at baseline resolution."
    return out

def resize_dir(src_dir, dst_dir, new_w, new_h):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {p}")
        if img.shape[1] == new_w and img.shape[0] == new_h:
            out = img
        else:
            interp = cv2.INTER_AREA if new_w < img.shape[1] else cv2.INTER_LINEAR
            out = cv2.resize(img, (new_w, new_h), interpolation=interp)
        cv2.imwrite(str(dst_dir / p.name), out)

def copytree_clean(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def main():
    src_raw = SRC / "raw_images"
    if not src_raw.exists():
        raise SystemExit(f"Missing canonical shared baseline source: {src_raw}")

    for name, (w, h) in VARIANTS.items():
        root = OUT_ROOT / name
        raw = root / "raw_images"
        if root.exists():
            shutil.rmtree(root)

        raw.mkdir(parents=True)

        # Static cameras unchanged.
        copytree_clean(src_raw / "static", raw / "static")

        # Moving camera varied.
        resize_dir(src_raw / "moving", raw / "moving", w, h)

        # Camera infos: static unchanged, moving scaled.
        ci_src = src_raw / "camera_info"
        ci_dst = raw / "camera_info"
        ci_dst.mkdir(parents=True, exist_ok=True)

        for jp in sorted(ci_src.glob("*.json")):
            obj = json.loads(jp.read_text())
            if jp.name in MOVING_CAMERA_INFO_NAMES or "moving" in jp.stem:
                obj2 = scale_camera_info(obj, w, h)
            else:
                obj2 = obj
                obj2["ablation_note"] = "Static camera info unchanged for moving-camera resolution ablation."
            (ci_dst / jp.name).write_text(json.dumps(obj2, indent=2) + "\n")

        meta_src = SRC / "metadata"
        if meta_src.exists():
            shutil.copytree(meta_src, root / "metadata", dirs_exist_ok=True)

        manifest = {
            "variant": name,
            "parameter": "moving_camera_image_resolution_posthoc_resize",
            "moving_width": w,
            "moving_height": h,
            "static_images": "unchanged from canonical shared baseline",
            "moving_images": "resized from canonical shared baseline moving capture",
            "source": str(SRC),
            "raw_images": str(raw),
            "interpretation": (
                "Post-hoc moving-camera image resize ablation. Low-resolution variants are meaningful "
                "for degradation sensitivity. Upscaled high-resolution variants do not add real image detail "
                "and must not be interpreted as true high-resolution Gazebo rendering."
            ),
        }
        (root / "VARIANT_METADATA.json").write_text(json.dumps(manifest, indent=2) + "\n")

        print(f"[OK] {name}: moving={w}x{h}, static=unchanged")

if __name__ == "__main__":
    main()

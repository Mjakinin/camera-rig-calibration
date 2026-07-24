from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _frame_key(path: Path) -> str:
    tail = path.stem.rsplit("_", 1)[-1]
    try:
        return str(int(tail))
    except ValueError:
        return path.stem


def build_moving_debug_gallery(
    *,
    dataset_root: Path,
    observations_root: Path,
    maximum_dimension: int = 1280,
    jpeg_quality: int = 85,
) -> dict[str, Any]:
    """Create a compact annotated preview for every moving-camera frame."""
    source_root = dataset_root / "raw_images" / "moving"
    if not source_root.is_dir():
        source_root = dataset_root / "moving"
    frames = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise RuntimeError(f"No moving frames found for debug gallery: {source_root}")
    existing_manifest = (
        observations_root / "debug_gallery" / "gallery_manifest.json"
    )
    if existing_manifest.is_file():
        try:
            existing = json.loads(
                existing_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            existing = {}
        existing_frames = existing.get("frames", [])
        if (
            existing.get("total_moving_frames") == len(frames)
            and len(existing_frames) == len(frames)
            and all(
                Path(str(item.get("preview", ""))).is_file()
                for item in existing_frames
            )
        ):
            return existing

    rows_by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    moving_csv = observations_root / "shared_moving_aruco_observations.csv"
    if moving_csv.is_file():
        with moving_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw = str(row.get("frame_id", "")).strip()
                try:
                    key = str(int(float(raw)))
                except ValueError:
                    key = raw
                rows_by_frame[key].append(row)

    static_marker_cameras: dict[int, set[str]] = defaultdict(set)
    static_csv = observations_root / "shared_static_aruco_observations.csv"
    if static_csv.is_file():
        with static_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    marker = int(float(row["marker_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
                camera = str(
                    row.get("observer_id") or row.get("camera_name") or ""
                ).strip()
                if camera:
                    static_marker_cameras[marker].add(camera)

    gallery = observations_root / "debug_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    debug_root = observations_root / "debug_images" / "moving"
    items: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        frame_key = _frame_key(frame)
        rows = rows_by_frame.get(frame_key, [])
        marker_ids = sorted(
            {
                int(float(row["marker_id"]))
                for row in rows
                if row.get("marker_id") not in {None, ""}
            }
        )
        cameras = sorted(
            {
                camera
                for marker in marker_ids
                for camera in static_marker_cameras.get(marker, set())
            }
        )
        bridge = len(marker_ids) >= 2 and len(cameras) >= 2
        debug = debug_root / f"{frame.stem}_detections.png"
        image = cv2.imread(str(debug if debug.is_file() else frame))
        if image is None:
            raise RuntimeError(f"Could not read moving gallery frame: {frame}")
        height, width = image.shape[:2]
        scale = min(1.0, maximum_dimension / max(width, height))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        label = (
            f"frame {frame_key} | detections {len(rows)} | "
            f"markers {','.join(map(str, marker_ids)) or 'none'}"
        )
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        preview = gallery / f"{index:06d}_{frame.stem}.jpg"
        if not cv2.imwrite(
            str(preview), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        ):
            raise RuntimeError(f"Could not write gallery preview: {preview}")
        items.append(
            {
                "frame_id": frame_key,
                "source": str(frame.resolve()),
                "preview": str(preview.resolve()),
                "detection_count": len(rows),
                "marker_ids": marker_ids,
                "static_camera_groups": cameras,
                "ap02_bridge_frame": bridge,
            }
        )

    summary = {
        "schema_version": 1,
        "total_moving_frames": len(items),
        "frames_with_detections": sum(
            item["detection_count"] > 0 for item in items
        ),
        "frames_without_markers": sum(
            item["detection_count"] == 0 for item in items
        ),
        "frames_with_multiple_markers": sum(
            len(item["marker_ids"]) > 1 for item in items
        ),
        "ap02_bridge_frames": sum(
            item["ap02_bridge_frame"] for item in items
        ),
        "gallery_path": str(gallery.resolve()),
        "preview_contract": {
            "format": "jpeg",
            "maximum_dimension_px": maximum_dimension,
            "jpeg_quality": jpeg_quality,
        },
        "frames": items,
    }
    _write_json(gallery / "gallery_manifest.json", summary)
    _write_json(
        observations_root / "connectivity_report.json",
        {
            key: value
            for key, value in summary.items()
            if key not in {"frames", "preview_contract"}
        },
    )
    return summary

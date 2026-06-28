#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import cv2


def get_aruco_dict(dict_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco fehlt. Installiere ggf. OpenCV mit ArUco-Unterstützung."
        )

    if not hasattr(cv2.aruco, dict_name):
        valid = [x for x in dir(cv2.aruco) if x.startswith("DICT_")]
        raise RuntimeError(f"Unknown ArUco dict: {dict_name}. Valid examples: {valid[:20]}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def detect_markers(image, aruco_dict):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # New OpenCV API
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        # Old OpenCV API
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None:
        return [], corners, ids

    ids_list = [int(x[0]) for x in ids]
    return ids_list, corners, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Pairs like name:path, e.g. original:results/.../images blur_15:results/.../blur_15/images",
    )
    parser.add_argument(
        "--dict",
        default="DICT_4X4_50",
        help="ArUco dictionary, e.g. DICT_4X4_50, DICT_5X5_100, DICT_6X6_250",
    )
    parser.add_argument(
        "--output",
        default="results/bus_real_data/ablation/aruco_detection_summary.csv",
    )
    parser.add_argument(
        "--annotated-root",
        default="results/bus_real_data/ablation/aruco_annotated",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save images with detected marker borders and IDs",
    )
    args = parser.parse_args()

    aruco_dict = get_aruco_dict(args.dict)

    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    annotated_root = Path(args.annotated_root)
    annotated_root.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {}

    for item in args.datasets:
        if ":" not in item:
            raise SystemExit(f"Dataset must be name:path, got: {item}")

        name, folder = item.split(":", 1)
        folder = Path(folder)

        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue

        image_paths = sorted(
            list(folder.glob("*.png"))
            + list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
        )

        if not image_paths:
            print(f"[WARN] No images in: {folder}")
            continue

        annotated_dir = annotated_root / name
        if args.save_annotated:
            annotated_dir.mkdir(parents=True, exist_ok=True)

        all_ids = set()
        image_counts = []
        detected_images = 0

        print(f"\n=== Dataset: {name} ===")
        print(f"Images: {len(image_paths)}")

        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[WARN] Could not read: {path}")
                continue

            ids_list, corners, ids = detect_markers(image, aruco_dict)
            ids_sorted = sorted(set(ids_list))

            if ids_sorted:
                detected_images += 1

            all_ids.update(ids_sorted)
            image_counts.append(len(ids_sorted))

            rows.append({
                "dataset": name,
                "image": path.name,
                "marker_count": len(ids_sorted),
                "ids": " ".join(map(str, ids_sorted)),
            })

            print(f"{path.name}: count={len(ids_sorted):02d}, ids={ids_sorted}")

            if args.save_annotated:
                vis = image.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.imwrite(str(annotated_dir / path.name), vis)

        avg_count = sum(image_counts) / max(len(image_counts), 1)

        summary[name] = {
            "num_images": len(image_paths),
            "detected_images": detected_images,
            "avg_marker_count": avg_count,
            "all_ids": sorted(all_ids),
        }

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "image", "marker_count", "ids"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n================ SUMMARY ================")
    for name, s in summary.items():
        print(f"\n{name}")
        print(f"  images:          {s['num_images']}")
        print(f"  detected_images: {s['detected_images']}")
        print(f"  avg_count:       {s['avg_marker_count']:.2f}")
        print(f"  all_ids:         {s['all_ids']}")

    print("\nCSV written to:")
    print(output_csv)

    if args.save_annotated:
        print("\nAnnotated images written to:")
        print(annotated_root)


if __name__ == "__main__":
    main()

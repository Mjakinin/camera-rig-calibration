#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--output", required=True, help="Output image folder")
    parser.add_argument("--kernel", type=int, default=15, help="Gaussian kernel size, must be odd")
    args = parser.parse_args()

    if args.kernel % 2 == 0:
        raise SystemExit("Gaussian kernel size must be odd, e.g. 7, 15, 25")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg"))
    )

    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping unreadable image: {path}")
            continue

        blurred = cv2.GaussianBlur(img, (args.kernel, args.kernel), 0)

        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), blurred)

    print(f"Processed {len(image_paths)} images")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

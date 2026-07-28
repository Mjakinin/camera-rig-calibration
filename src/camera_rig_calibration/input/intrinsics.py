from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from ..intrinsics_profiles import (
    profile_directory,
    profile_fingerprint,
    profile_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate moving-camera intrinsics and install canonical CameraInfo."
    )
    parser.add_argument("--script", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video")
    source_group.add_argument("--images")
    parser.add_argument("--work-directory", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--repository")
    parser.add_argument("--profile-id")
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--max-views", type=int, default=80)
    parser.add_argument("--minimum-frame-gap", type=int, default=5)
    parser.add_argument("--minimum-detections", type=int, default=20)
    parser.add_argument(
        "--scan-mode",
        choices=("balanced", "full_frame"),
        default="balanced",
    )
    parser.add_argument("--scan-target-hz", type=float, default=3.0)
    parser.add_argument("--preview-max-dimension", type=int, default=1920)
    args = parser.parse_args()

    media_source = Path(args.video or args.images).resolve()
    source_option = "--video" if args.video else "--images"
    destination = Path(args.destination).resolve()
    settings = {
        "checkerboard_columns": args.cols,
        "checkerboard_rows": args.rows,
        "maximum_views": args.max_views,
        "minimum_frame_gap": args.minimum_frame_gap,
        "minimum_detections": args.minimum_detections,
        "scan_mode": args.scan_mode,
        "scan_target_hz": args.scan_target_hz,
        "preview_max_dimension": args.preview_max_dimension,
    }
    fingerprint = profile_fingerprint(
        media_source,
        columns=args.cols,
        rows=args.rows,
        maximum_views=args.max_views,
        minimum_frame_gap=args.minimum_frame_gap,
        minimum_detections=args.minimum_detections,
        scan_mode=args.scan_mode,
        scan_target_hz=args.scan_target_hz,
        preview_max_dimension=args.preview_max_dimension,
    )
    profile_target: Path | None = None
    if args.repository and args.profile_id:
        profile_target = profile_directory(
            Path(args.repository), args.profile_id, fingerprint
        )
        work = profile_target
    else:
        work = Path(args.work_directory).resolve()

    intrinsic_source = work / (
        "intrinsics.json" if profile_target is not None else "moving_calib_camera.json"
    )
    if (
        profile_target is not None
        and (work / "profile.yaml").is_file()
        and intrinsic_source.is_file()
    ):
        print(
            f"[OK] reusing intrinsic profile {args.profile_id}@{fingerprint[:12]}",
            flush=True,
        )
    else:
        staging: Path | None = None
        if profile_target is not None:
            profile_target.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{profile_target.name}.staging-",
                    dir=profile_target.parent,
                )
            )
            engine_work = staging
        else:
            engine_work = work
        command = [
            sys.executable,
            str(Path(args.script).resolve()),
            source_option,
            str(media_source),
            "--out",
            str(engine_work),
            "--cols",
            str(args.cols),
            "--rows",
            str(args.rows),
            "--max-views",
            str(args.max_views),
            "--minimum-frame-gap",
            str(args.minimum_frame_gap),
            "--minimum-detections",
            str(args.minimum_detections),
            "--scan-mode",
            args.scan_mode,
            "--scan-target-hz",
            str(args.scan_target_hz),
            "--preview-max-dimension",
            str(args.preview_max_dimension),
        ]
        started = time.monotonic()
        try:
            subprocess.run(command, check=True)
            generated = engine_work / "moving_calib_camera.json"
            if not generated.is_file():
                raise RuntimeError(
                    f"Intrinsic calibration did not produce {generated}"
                )
            if profile_target is not None:
                shutil.copy2(generated, engine_work / "intrinsics.json")
                diagnostics = engine_work / "diagnostics"
                diagnostics.mkdir(exist_ok=True)
                public_names = {
                    "intrinsics.json",
                    "INTRINSICS_REPORT.txt",
                    "selected_frames",
                    "diagnostics",
                }
                for path in list(engine_work.iterdir()):
                    if path.name not in public_names:
                        path.replace(diagnostics / path.name)
                manifest = profile_manifest(
                    profile_id=args.profile_id,
                    fingerprint=fingerprint,
                    video=media_source,
                    intrinsics=engine_work / "intrinsics.json",
                    settings=settings,
                    elapsed_seconds=time.monotonic() - started,
                )
                (engine_work / "profile.yaml").write_text(
                    yaml.safe_dump(
                        manifest, sort_keys=False, allow_unicode=True
                    ),
                    encoding="utf-8",
                )
                (engine_work / "timings.json").write_text(
                    json.dumps(
                        {
                            "total_seconds": time.monotonic() - started,
                            "scan_mode": args.scan_mode,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if profile_target.exists():
                    shutil.rmtree(engine_work)
                else:
                    engine_work.replace(profile_target)
                work = profile_target
                intrinsic_source = work / "intrinsics.json"
            else:
                intrinsic_source = generated
        except BaseException:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)
            raise
    payload = json.loads(intrinsic_source.read_text(encoding="utf-8"))
    payload["camera_name"] = args.camera_id
    if profile_target is not None:
        payload["rigcal_intrinsics_profile"] = (
            f"{args.profile_id}@{fingerprint}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if profile_target is None:
        shutil.copy2(destination, work / f"{args.camera_id}.json")
    print(f"[OK] installed intrinsics: {destination}")


if __name__ == "__main__":
    main()

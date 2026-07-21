#!/usr/bin/env python3
"""Rename completed 0.5x sampling-ablation result folders to clean names.

The ablation runner originally encoded the AP02 max_nfev budget and a version
suffix in each directory name. Those settings remain recorded inside
EXPERIMENT_CONFIG.txt, so they do not need to be part of the folder name.

Example:
  real_05x_4k_1hz_ap02nfev120_v1 -> real_05x_4k_1hz

All readable text metadata below the renamed directory is updated so absolute
paths and dataset_name fields continue to point to the new location.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TEXT_SUFFIXES = {
    ".txt",
    ".csv",
    ".json",
    ".log",
    ".md",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/real_vehicle_data",
        help="Root containing the real-vehicle result folders.",
    )
    parser.add_argument(
        "--rate",
        action="append",
        type=float,
        dest="rates",
        help="Sampling rate to normalize. Repeat as needed. Default: 1, 3, 5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without renaming or rewriting files.",
    )
    return parser.parse_args()


def rate_label(rate: float) -> str:
    if rate <= 0:
        raise ValueError(f"Sampling rate must be positive: {rate}")
    if rate.is_integer():
        return str(int(rate))
    return f"{rate:g}".replace(".", "p")


def find_source(root: Path, label: str) -> Path | None:
    clean = root / f"real_05x_4k_{label}hz"
    if clean.is_dir():
        return clean

    matches = sorted(root.glob(f"real_05x_4k_{label}hz_ap02nfev*_v1"))
    if not matches:
        matches = sorted(root.glob(f"real_05x_4k_{label}hz_ap02nfev*"))
    if not matches:
        matches = sorted(root.glob(f"real_05x_4k_{label}hz_v1"))

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple source folders found for {label} Hz:\n"
            + "\n".join(f"  {path}" for path in matches)
        )
    return matches[0] if matches else None


def rewrite_text_paths(folder: Path, old_name: str, new_name: str, dry_run: bool) -> int:
    changed = 0
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text.replace(old_name, new_name)
        if updated == text:
            continue
        changed += 1
        print(f"  rewrite: {path}")
        if not dry_run:
            path.write_text(updated, encoding="utf-8")
    return changed


def normalize_one(root: Path, rate: float, dry_run: bool) -> None:
    label = rate_label(rate)
    target = root / f"real_05x_4k_{label}hz"
    source = find_source(root, label)

    if source is None:
        print(f"[SKIP] {label} Hz: no result folder found")
        return

    if source == target:
        print(f"[OK] {label} Hz already clean: {target}")
        return

    if target.exists():
        raise RuntimeError(f"Target already exists; refusing to overwrite: {target}")

    old_name = source.name
    new_name = target.name
    print(f"[RENAME] {source} -> {target}")

    if dry_run:
        rewrite_text_paths(source, old_name, new_name, dry_run=True)
        return

    source.rename(target)
    changed = rewrite_text_paths(target, old_name, new_name, dry_run=False)
    print(f"[OK] {label} Hz renamed; updated {changed} readable metadata files")


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).resolve()
    if not root.is_dir():
        raise RuntimeError(f"Results root not found: {root}")

    rates = args.rates or [1.0, 3.0, 5.0]
    for rate in rates:
        normalize_one(root, float(rate), args.dry_run)


if __name__ == "__main__":
    main()

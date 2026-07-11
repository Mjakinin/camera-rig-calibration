#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


SIM_OUTPUTS = [
    Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain"),
    Path("results/bus_real_data/02_ref_marker_graph_ba"),
    Path("results/bus_real_data/03_targetless_colmap_aruco_scale"),
    Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"),
]

REAL_ROOT = Path("results/real_vehicle_data/real_05x_4k_3hz_v1")
REAL_SHARED = REAL_ROOT / "00_shared_input"
REAL_RAW = REAL_SHARED / "raw_images"
REAL_OBS = REAL_SHARED / "aruco_observations"

OLD_DATASET_ROOT = (
    "data_local/real_vehicle_2026_07_10/datasets/"
    "real_05x_4k_3hz_v1"
)
NEW_SHARED_ROOT = (
    "results/real_vehicle_data/real_05x_4k_3hz_v1/"
    "00_shared_input"
)
WRONG_SHARED_RAW_ROOT = NEW_SHARED_ROOT + "/raw_images"

TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg",
}

VALIDATOR_CONTENT = '#!/usr/bin/env python3\nfrom __future__ import annotations\n\nfrom pathlib import Path\n\n\ndef repository_root() -> Path:\n    current = Path(__file__).resolve()\n    for parent in current.parents:\n        if (parent / ".git").exists():\n            return parent\n    raise RuntimeError("Repository root not found")\n\n\ndef main() -> None:\n    root = repository_root()\n    shared = (\n        root\n        / "results/real_vehicle_data/real_05x_4k_3hz_v1/"\n        "00_shared_input"\n    )\n    raw = shared / "raw_images"\n    observations = shared / "aruco_observations"\n\n    required = [\n        raw / "moving",\n        raw / "static",\n        raw / "camera_info",\n        observations / "shared_all_aruco_observations.csv",\n        observations / "shared_moving_aruco_observations.csv",\n        observations / "shared_static_aruco_observations.csv",\n    ]\n    missing = [str(path) for path in required if not path.exists()]\n    if missing:\n        raise SystemExit(\n            "[ERROR] incomplete canonical real shared input:\\n- "\n            + "\\n- ".join(missing)\n        )\n\n    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}\n    image_count = sum(\n        1\n        for path in raw.rglob("*")\n        if path.is_file() and path.suffix.lower() in image_suffixes\n    )\n    if image_count <= 0:\n        raise SystemExit("[ERROR] canonical real shared input has no images")\n\n    print("[OK] canonical real shared input")\n    print("     root:", shared)\n    print("     raw images:", raw)\n    print("     ArUco observations:", observations)\n    print("     image count:", image_count)\n\n\nif __name__ == "__main__":\n    main()\n'
ROOT_README = '# Arbitrary Camera Rig Calibration\n\nFinal fixed methods:\n\n- AP01 baseline: direct multimarker with moving-COLMAP relay where required\n- AP02 baseline: distortion-aware reference-marker graph bundle adjustment\n- AP03: grouped calibrated targetless COLMAP with ArUco marker-size scale\n\nCanonical shared inputs:\n\n```text\nresults/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/\n├── raw_images/\n├── aruco_observations/\n└── metadata/\n\nresults/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input/\n├── raw_images/\n├── aruco_observations/\n├── calibration/\n└── metadata/\n```\n\nAP01, AP02 and AP03 read the same immutable input dataset. Their outputs are\nstored separately and are overwritten by the corresponding rerun pipeline.\n\nSimulation development baseline: Route 2 with the extended ArUco layout.\n'


def run(args, *, cwd, check=True, capture=False):
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def repo_root():
    result = run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        capture=True,
    )
    return Path(result.stdout.strip()).resolve()


def require(path, label):
    if not path.exists():
        raise SystemExit(f"[ERROR] missing {label}: {path}")


def restore_from_head(repo, rel, apply):
    tracked = run(
        ["git", "cat-file", "-e", f"HEAD:{rel.as_posix()}"],
        cwd=repo,
        check=False,
        capture=True,
    ).returncode == 0
    if not tracked:
        raise SystemExit(
            f"[ERROR] cannot restore from HEAD because it is not tracked: {rel}"
        )

    print("RESTORE_FROM_HEAD", rel)
    if apply:
        run(
            [
                "git", "restore",
                "--source=HEAD",
                "--staged",
                "--worktree",
                "--",
                str(rel),
            ],
            cwd=repo,
        )


def flatten_real_shared_input(repo, apply):
    shared = repo / REAL_SHARED
    outer_raw = repo / REAL_RAW
    inner_raw = outer_raw / "raw_images"

    require(shared, "real shared input")
    require(repo / REAL_OBS, "real shared ArUco observations")

    if inner_raw.is_dir():
        print(f"FLATTEN {inner_raw} -> {outer_raw}")
        if apply:
            for child in sorted(inner_raw.iterdir()):
                target = outer_raw / child.name
                if target.exists() or target.is_symlink():
                    raise SystemExit(
                        f"[ERROR] flattening conflict: {child} -> {target}"
                    )
                shutil.move(str(child), str(target))
            inner_raw.rmdir()

    for name in ["calibration", "metadata"]:
        wrong = outer_raw / name
        correct = shared / name
        if wrong.exists():
            if correct.exists():
                raise SystemExit(
                    f"[ERROR] both wrong and canonical locations exist: "
                    f"{wrong} and {correct}"
                )
            print(f"MOVE {wrong} -> {correct}")
            if apply:
                shutil.move(str(wrong), str(correct))

    if apply:
        for required in ["moving", "static", "camera_info"]:
            require(outer_raw / required, f"real raw_images/{required}")


def patch_text_paths(repo, apply):
    roots = [repo / "run/real_vehicle_data", repo / "README.md"]
    candidates = []

    for root in roots:
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
            )

    for path in sorted(set(candidates)):
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue

        updated = text.replace(WRONG_SHARED_RAW_ROOT, NEW_SHARED_ROOT)
        updated = updated.replace(OLD_DATASET_ROOT, NEW_SHARED_ROOT)

        if updated != text:
            print("PATCH", path.relative_to(repo))
            if apply:
                path.write_text(updated)


def rewrite_validator(repo, apply):
    path = repo / "run/real_vehicle_data/00_validate_and_prepare_shared_input.py"
    print("REWRITE", path.relative_to(repo))
    if apply:
        path.write_text(VALIDATOR_CONTENT)
        path.chmod(0o755)


def relocate_method_lock(repo, apply):
    source = repo / "docs/experiments/final_method_selection"
    destination = repo / "results/bus_real_data/00_method_lock"
    wanted = ["FINAL_METHODS_LOCK.json", "FINAL_METHODS_LOCK.txt"]

    print("RELOCATE method lock ->", destination.relative_to(repo))
    if not apply:
        return

    destination.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        src = source / name
        dst = destination / name
        if src.is_file():
            shutil.copy2(src, dst)
        elif not dst.is_file():
            raise SystemExit(
                f"[ERROR] final method lock missing in both locations: {name}"
            )

    docs = repo / "docs"
    if docs.exists():
        shutil.rmtree(docs)


def relocate_or_remove_data_local(repo, apply):
    local = repo / "data_local"
    if not local.exists() and not local.is_symlink():
        return

    legacy = repo / OLD_DATASET_ROOT
    if legacy.is_symlink():
        print("REMOVE compatibility symlink", legacy.relative_to(repo))
        if apply:
            legacy.unlink()

    remaining_files = []
    if local.exists():
        remaining_files = [
            p for p in local.rglob("*")
            if p.is_file() and not p.is_symlink()
        ]

    if remaining_files:
        external = repo.parent / f"{repo.name}_local_data"
        if external.exists():
            raise SystemExit(
                "[ERROR] data_local still contains source/local files and "
                f"safe destination already exists: {external}"
            )
        print(
            "MOVE_LOCAL_ONLY_DATA",
            local,
            "->",
            external,
            f"({len(remaining_files)} files)",
        )
        if apply:
            shutil.move(str(local), str(external))
    else:
        print("REMOVE_EMPTY_LOCAL_ROOT", local.relative_to(repo))
        if apply and local.exists():
            shutil.rmtree(local)


def remove_temporary_material(repo, apply):
    cleanup = repo / "tools/project_cleanup"
    if cleanup.exists():
        print("DELETE_TEMPORARY", cleanup.relative_to(repo))
        if apply:
            shutil.rmtree(cleanup)

    tools = repo / "tools"
    if tools.is_dir() and not any(tools.iterdir()):
        print("REMOVE_EMPTY", tools.relative_to(repo))
        if apply:
            tools.rmdir()

    src_real = repo / "src/calib_lab/real_vehicle_data"
    if src_real.is_dir():
        has_files = any(p.is_file() for p in src_real.rglob("*"))
        if not has_files:
            print("REMOVE_EMPTY", src_real.relative_to(repo))
            if apply:
                shutil.rmtree(src_real)
        else:
            print("[KEEP] src/calib_lab/real_vehicle_data contains files")

    github = repo / ".github"
    if github.is_dir():
        files = sorted(p for p in github.rglob("*") if p.is_file())
        if not files:
            print("REMOVE_EMPTY", github.relative_to(repo))
            if apply:
                shutil.rmtree(github)
        else:
            print("[KEEP] .github contains repository configuration/workflows:")
            for path in files:
                print("       ", path.relative_to(repo))


def clean_gitattributes(repo, apply):
    path = repo / ".gitattributes"
    if not path.is_file():
        return

    text = path.read_text()
    marker = "\n# Canonical raw image datasets\n"
    if marker in text:
        before, _ = text.split(marker, 1)
        print("REMOVE redundant .gitattributes raw-image block")
        if apply:
            path.write_text(before.rstrip() + "\n")


def write_root_readme(repo, apply):
    print("REWRITE README.md")
    if apply:
        (repo / "README.md").write_text(ROOT_README)


def scan_wrong_references(repo):
    findings = []
    roots = [repo / "run/real_vehicle_data", repo / "README.md"]
    needles = [WRONG_SHARED_RAW_ROOT + "/raw_images", OLD_DATASET_ROOT]
    candidates = []

    for root in roots:
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
            )

    for path in candidates:
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for needle in needles:
            if needle in text:
                findings.append(
                    f"{path.relative_to(repo)} contains {needle}"
                )
    return findings


def verify(repo):
    for rel in SIM_OUTPUTS:
        require(repo / rel, f"restored simulation output {rel}")

    report = repo / "results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"
    report_files = [p for p in report.rglob("*") if p.is_file()]
    if len(report_files) <= 1:
        raise SystemExit(
            "[ERROR] simulation final report still looks empty after restore"
        )

    required = [
        repo / REAL_RAW / "moving",
        repo / REAL_RAW / "static",
        repo / REAL_RAW / "camera_info",
        repo / REAL_OBS / "shared_all_aruco_observations.csv",
        repo / "results/bus_real_data/00_method_lock/FINAL_METHODS_LOCK.json",
        repo / "results/bus_real_data/00_method_lock/FINAL_METHODS_LOCK.txt",
    ]
    for path in required:
        require(path, "required final structure item")

    if (repo / REAL_RAW / "raw_images").exists():
        raise SystemExit("[ERROR] nested raw_images/raw_images still exists")
    if (repo / "docs").exists():
        raise SystemExit("[ERROR] docs still exists")
    if (repo / "tools/project_cleanup").exists():
        raise SystemExit("[ERROR] temporary cleanup tooling still exists")

    wrong = scan_wrong_references(repo)
    if wrong:
        raise SystemExit(
            "[ERROR] stale real-input references remain:\n- "
            + "\n- ".join(wrong)
        )

    run(
        ["python3", "run/real_vehicle_data/00_validate_and_prepare_shared_input.py"],
        cwd=repo,
    )

    print()
    print("VERIFY: PASSED")
    for rel in SIM_OUTPUTS:
        count = sum(1 for p in (repo / rel).rglob("*") if p.is_file())
        print(f"- {rel}: {count} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()

    if args.stage and not args.apply:
        raise SystemExit("[ERROR] --stage requires --apply")

    repo = repo_root()

    print("PROJECT STRUCTURE CORRECTION")
    print("=" * 78)
    print("mode:", "APPLY" if args.apply else "PLAN ONLY")
    print()

    for rel in SIM_OUTPUTS:
        restore_from_head(repo, rel, args.apply)

    flatten_real_shared_input(repo, args.apply)
    patch_text_paths(repo, args.apply)
    rewrite_validator(repo, args.apply)
    relocate_method_lock(repo, args.apply)
    relocate_or_remove_data_local(repo, args.apply)
    remove_temporary_material(repo, args.apply)
    clean_gitattributes(repo, args.apply)
    write_root_readme(repo, args.apply)

    if not args.apply:
        print()
        print("PLAN ONLY: nothing changed.")
        return

    if args.stage:
        run(["git", "add", "-A"], cwd=repo)

    verify(repo)

    print()
    print("Root directories:")
    for path in sorted(p for p in repo.iterdir() if p.is_dir()):
        print("-", path.name)
    print()
    run(["git", "status", "--short"], cwd=repo, check=False)


if __name__ == "__main__":
    main()

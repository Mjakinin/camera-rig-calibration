from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .inventory import discover_simulation_experiments


ARCHIVE_BRANCH = "archive/pre-rigcal-results-v2-20260723"
ARCHIVE_TAG = "pre-rigcal-results-v2-20260723"


@dataclass(frozen=True)
class LegacyMigrationRecord:
    category: str
    experiment_id: str
    source: str
    has_input: bool
    has_results: bool
    rerunnable: bool
    input_id: str
    target: str
    factor: str
    value: str
    parameters: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_inventory(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    if not root.is_dir():
        return digest.hexdigest(), count, size
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        file_sha = _sha256(path)
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(file_sha.encode("ascii"))
        digest.update(str(stat.st_size).encode("ascii"))
        count += 1
        size += stat.st_size
    return digest.hexdigest(), count, size


def _input_identity(root: Path) -> str:
    digest = hashlib.sha256()
    found = False
    for name in ("raw_images", "metadata"):
        tree_sha, count, size = _tree_inventory(root / name)
        if count:
            found = True
        digest.update(name.encode("utf-8"))
        digest.update(tree_sha.encode("ascii"))
        digest.update(str(count).encode("ascii"))
        digest.update(str(size).encode("ascii"))
    return (
        f"input_{digest.hexdigest()[:12]}"
        if found
        else f"unavailable_{hashlib.sha256(str(root).encode()).hexdigest()[:12]}"
    )


def _recorded_path(repository: Path, value: str) -> Path:
    path = Path(value)
    container_root = Path("/workspaces/project")
    try:
        relative = path.relative_to(container_root)
    except ValueError:
        return path
    return repository / relative


def discover_legacy_migration(
    repository_root: Path,
) -> list[LegacyMigrationRecord]:
    repository = repository_root.resolve()
    records: list[LegacyMigrationRecord] = []
    for experiment in discover_simulation_experiments(
        repository, include_v2=False
    ):
        source = experiment.dataset_root
        if source is None or not source.is_dir():
            continue
        input_id = _input_identity(source)
        target = repository / "results" / "simulation" / experiment.variant
        records.append(
            LegacyMigrationRecord(
                category="simulation",
                experiment_id=experiment.variant,
                source=str(source),
                has_input=(source / "raw_images").is_dir(),
                has_results=experiment.has_results,
                rerunnable=(source / "raw_images").is_dir(),
                input_id=input_id,
                target=str(target),
                factor=experiment.factor,
                value=experiment.value,
                parameters=dict(experiment.parameters),
            )
        )

    real_root = repository / "results" / "real_vehicle_data"
    if real_root.is_dir():
        for source in sorted(
            item
            for item in real_root.iterdir()
            if item.is_dir() and item.name != "INTRINSIC_RESULTS"
        ):
            prepared = (
                source / "00_shared_input"
                if (source / "00_shared_input").is_dir()
                else source
            )
            has_input = (prepared / "raw_images").is_dir()
            input_id = _input_identity(prepared if has_input else source)
            has_results = (source / "99_FINAL_RESULTS").is_dir()
            records.append(
                LegacyMigrationRecord(
                    category="real_vehicle",
                    experiment_id=source.name,
                    source=str(source),
                    has_input=has_input,
                    has_results=has_results,
                    rerunnable=has_input,
                    input_id=input_id,
                    target=str(
                        repository
                        / "results"
                        / "real_vehicle"
                        / source.name
                    ),
                    factor="historical real-data experiment",
                    value=source.name,
                    parameters={},
                )
            )
    return records


def _materialize(source: Path, destination: Path) -> dict[str, int]:
    counts = {"hardlinked": 0, "copied": 0, "existing": 0}
    if not source.is_dir():
        return counts
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256(target) != _sha256(item):
                raise RuntimeError(f"Migration target conflict: {target}")
            counts["existing"] += 1
            continue
        try:
            os.link(item, target)
            counts["hardlinked"] += 1
        except OSError:
            shutil.copy2(item, target)
            counts["copied"] += 1
    return counts


def _materialize_file(source: Path, destination: Path) -> str:
    if not source.is_file():
        return "missing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(
                f"Migration target conflict: {destination}"
            )
        return "existing"
    try:
        os.link(source, destination)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def _archive_refs_exist(repository: Path) -> bool:
    checks = [
        ["git", "show-ref", "--verify", f"refs/heads/{ARCHIVE_BRANCH}"],
        ["git", "show-ref", "--verify", f"refs/tags/{ARCHIVE_TAG}"],
    ]
    return all(
        subprocess.run(
            command,
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
        for command in checks
    )


def execute_legacy_migration(
    repository_root: Path,
    *,
    records: list[LegacyMigrationRecord] | None = None,
) -> Path:
    repository = repository_root.resolve()
    if not _archive_refs_exist(repository):
        raise RuntimeError(
            f"Create archive branch '{ARCHIVE_BRANCH}' and tag '{ARCHIVE_TAG}' "
            "before migrating historical results"
        )
    selected = records or discover_legacy_migration(repository)
    verification: list[dict[str, Any]] = []
    for record in selected:
        source = Path(record.source)
        target = Path(record.target)
        target.mkdir(parents=True, exist_ok=True)
        prepared = (
            source / "00_shared_input"
            if (source / "00_shared_input").is_dir()
            else source
        )
        input_target = target / "inputs" / record.input_id
        counts = {"hardlinked": 0, "copied": 0, "existing": 0}
        if record.has_input:
            for name in ("raw_images", "metadata"):
                materialized = _materialize(
                    prepared / name, input_target / name
                )
                for key, value in materialized.items():
                    counts[key] += value
            for metadata_name in (
                "VARIANT_METADATA.json",
                "EXPERIMENT_CONFIG.txt",
            ):
                status = _materialize_file(
                    source / metadata_name,
                    input_target / "metadata" / metadata_name,
                )
                if status in counts:
                    counts[status] += 1
        else:
            input_target.mkdir(parents=True, exist_ok=True)
            (input_target / "INPUT_UNAVAILABLE.json").write_text(
                json.dumps(
                    {
                        "status": "input unavailable",
                        "rerunnable": False,
                        "legacy_source": str(source),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        observations_source = (
            prepared / "aruco_observations"
            if (prepared / "aruco_observations").is_dir()
            else source / "aruco_observations"
        )
        if observations_source.is_dir():
            _materialize(
                observations_source,
                target / "observations" / record.input_id / "legacy",
            )
        results_source = (
            source / "FINAL_RESULTS"
            if (source / "FINAL_RESULTS").is_dir()
            else source / "99_FINAL_RESULTS"
        )
        if results_source.is_dir():
            _materialize(
                results_source,
                target / "legacy_results" / record.input_id,
            )

        source_hash, source_count, source_size = (
            _tree_inventory(prepared / "raw_images")
            if record.has_input
            else ("", 0, 0)
        )
        target_hash, target_count, target_size = (
            _tree_inventory(input_target / "raw_images")
            if record.has_input
            else ("", 0, 0)
        )
        verified = (
            not record.has_input
            or (
                source_hash == target_hash
                and source_count == target_count
                and source_size == target_size
            )
        )
        manifest = {
            "schema_version": 2,
            "migration": "legacy_read_only",
            **asdict(record),
            "storage": counts,
            "input_verification": {
                "verified": verified,
                "sha256": source_hash or None,
                "file_count": source_count,
                "size_bytes": source_size,
            },
            "status": (
                "input unavailable / not rerunnable"
                if not record.has_input
                else "available"
                if record.has_results
                else "input only"
            ),
        }
        (target / "legacy_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        verification.append(manifest)
        if not verified:
            raise RuntimeError(
                f"Legacy migration verification failed: {record.experiment_id}"
            )

    migrated_v1_runs: list[dict[str, Any]] = []
    intrinsic_source = (
        repository / "results" / "real_vehicle_data" / "INTRINSIC_RESULTS"
    )
    intrinsic_target = (
        repository / "results" / "real_vehicle" / "_intrinsics"
    )
    intrinsic_storage = (
        _materialize(intrinsic_source, intrinsic_target)
        if intrinsic_source.is_dir()
        else {}
    )
    method_directories = {
        "ap01": "02_AP01",
        "ap02": "03_AP02",
        "ap03_single": "04_AP03_SINGLE",
        "ap03_multi": "05_AP03_MULTI",
    }
    for manifest_path in (repository / "results").glob(
        "*/runs/*/run_manifest.json"
    ):
        run_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if run_manifest.get("status") != "completed":
            continue
        run = manifest_path.parent
        config_path = run / "resolved_config.yaml"
        if not config_path.is_file():
            continue
        config_payload = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        ) or {}
        if (
            config_payload.get("dataset", {}).get("scene_type")
            != "simulation"
        ):
            continue
        pointer_path = run / "00_INPUT" / "dataset_pointer.json"
        if not pointer_path.is_file():
            continue
        dataset_root = _recorded_path(
            repository,
            json.loads(pointer_path.read_text(encoding="utf-8"))[
                "dataset_root"
            ],
        )
        experiment_id = dataset_root.name
        input_id = _input_identity(dataset_root)
        experiment_root = (
            repository / "results" / "simulation" / experiment_id
        )
        config_sha = str(
            run_manifest.get("resolved_config_sha256")
            or run_manifest.get("config_sha256")
            or hashlib.sha256(config_path.read_bytes()).hexdigest()
        )
        for method_id in run_manifest.get("enabled_methods", []):
            directory_name = method_directories.get(str(method_id))
            if directory_name is None or not (run / directory_name).is_dir():
                continue
            variant = f"legacy_v1_{config_sha[:8]}"
            current = (
                experiment_root
                / "methods"
                / str(method_id)
                / variant
                / "executions"
                / input_id
                / "current"
            )
            _materialize(
                run / directory_name, current / directory_name
            )
            for name in (
                "requested_config.yaml",
                "resolved_config.yaml",
                "timings.json",
                "environment.json",
                "commands.txt",
            ):
                source_file = run / name
                if source_file.is_file():
                    current.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, current / name)
            migrated_manifest = {
                "schema_version": 2,
                "status": "completed",
                "migration": "successful_rigcal_v1_run",
                "legacy_source_run": str(run),
                "experiment_id": experiment_id,
                "result_category": "simulation",
                "input_id": input_id,
                "method_id": method_id,
                "variant": variant,
                "method_fingerprint": config_sha,
                "enabled_methods": [method_id],
                "run_id": run_manifest.get("run_id", run.name),
            }
            (current / "run_manifest.json").write_text(
                json.dumps(migrated_manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            migrated_v1_runs.append(migrated_manifest)

    failed_runs = []
    for manifest_path in (repository / "results").glob(
        "*/runs/*/run_manifest.json"
    ):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            failed_runs.append(
                {
                    "path": str(manifest_path.parent),
                    "status": payload.get("status"),
                    "action": "not migrated; delete only after explicit confirmation",
                }
            )
    report = repository / "workspace" / "migrations" / "results_v2.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "archive_branch": ARCHIVE_BRANCH,
                "archive_tag": ARCHIVE_TAG,
                "records": verification,
                "migrated_successful_v1_runs": migrated_v1_runs,
                "migrated_intrinsics": {
                    "source": str(intrinsic_source),
                    "target": str(intrinsic_target),
                    "storage": intrinsic_storage,
                },
                "incomplete_runs": failed_runs,
                "legacy_sources_removed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def refresh_v2_legacy_manifests(repository_root: Path) -> list[Path]:
    """Normalize migrated display status without reading archived source trees."""
    repository = repository_root.resolve()
    changed: list[Path] = []
    for path in [
        *repository.glob(
            "results/simulation/*/legacy_manifest.json"
        ),
        *repository.glob(
            "results/real_vehicle/*/legacy_manifest.json"
        ),
    ]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = (
            "input unavailable / not rerunnable"
            if not payload.get("has_input", False)
            else "available"
            if payload.get("has_results", False)
            else "input only"
        )
        if payload.get("status") == status:
            continue
        payload["status"] = status
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        changed.append(path)
    report = repository / "workspace/migrations/results_v2.json"
    if report.is_file():
        report_payload = json.loads(
            report.read_text(encoding="utf-8")
        )
        for row in report_payload.get("records", []):
            row["status"] = (
                "input unavailable / not rerunnable"
                if not row.get("has_input", False)
                else "available"
                if row.get("has_results", False)
                else "input only"
            )
        report.write_text(
            json.dumps(report_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return changed


def remove_verified_legacy_sources(repository_root: Path) -> list[Path]:
    """Archive only verified active legacy roots; incomplete runs are excluded.

    Moving the trees out of ``results/`` removes obsolete active paths while
    keeping a local recovery copy in addition to the Git branch and tag.
    """
    repository = repository_root.resolve()
    report = repository / "workspace" / "migrations" / "results_v2.json"
    if not report.is_file():
        raise RuntimeError("Run the verified results-v2 migration first")
    payload = json.loads(report.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records or not all(
        row.get("input_verification", {}).get("verified") for row in records
    ):
        raise RuntimeError(
            "Legacy sources cannot be removed because migration verification "
            "is incomplete"
        )
    if not _archive_refs_exist(repository):
        raise RuntimeError("Archive branch/tag is missing")

    results = (repository / "results").resolve()
    targets = [
        (results / "bus_real_data").resolve(),
        (results / "real_vehicle_data").resolve(),
    ]
    for candidate in sorted(results.iterdir()):
        if not candidate.is_dir() or candidate in targets:
            continue
        manifests = list(candidate.glob("runs/*/run_manifest.json"))
        if not manifests:
            dataset_manifest = candidate / "dataset_manifest.json"
            if dataset_manifest.is_file():
                try:
                    dataset_payload = json.loads(
                        dataset_manifest.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError:
                    dataset_payload = {}
                verified_sources = {
                    str(row.get("source")) for row in records
                }
                if (
                    int(dataset_payload.get("schema_version", 0)) == 1
                    and str(dataset_payload.get("prepared_root"))
                    in verified_sources
                ):
                    targets.append(candidate.resolve())
            continue
        statuses = {
            str(json.loads(path.read_text(encoding="utf-8")).get("status"))
            for path in manifests
        }
        if statuses == {"completed"}:
            targets.append(candidate.resolve())

    archive_root = (
        repository / "workspace" / "migrations" / "legacy_sources_v1"
    )
    archive_root.mkdir(parents=True, exist_ok=True)
    removed: list[Path] = []
    archived: list[dict[str, str]] = list(
        payload.get("archived_verified_legacy_roots", [])
    )
    for target in targets:
        target.relative_to(results)
        if target.name in {
            "simulation",
            "real_vehicle",
            "reuse_route2_20260722_221934",
            "simulation_bus_baseline_20260722_224454",
        }:
            continue
        if target.is_dir():
            destination = archive_root / target.name
            if destination.exists():
                raise RuntimeError(
                    f"Legacy recovery destination already exists: {destination}"
                )
            target.rename(destination)
            removed.append(target)
            archived.append(
                {
                    "active_source": str(target),
                    "recovery_copy": str(destination),
                }
            )
    payload["legacy_sources_removed"] = True
    payload["removed_verified_legacy_roots"] = list(
        dict.fromkeys(
            [
                *payload.get("removed_verified_legacy_roots", []),
                *(str(path) for path in removed),
            ]
        )
    )
    payload["archived_verified_legacy_roots"] = archived
    report.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory or materialize legacy results into rigcal v2."
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--remove-verified-legacy", action="store_true")
    parser.add_argument("--refresh-v2-manifests", action="store_true")
    args = parser.parse_args()
    repository = Path(args.repository).resolve()
    records = discover_legacy_migration(repository)
    if args.refresh_v2_manifests:
        for path in refresh_v2_legacy_manifests(repository):
            print(path)
        return
    if args.remove_verified_legacy:
        for path in remove_verified_legacy_sources(repository):
            print(path)
        return
    if not args.execute:
        print(json.dumps([asdict(record) for record in records], indent=2))
        return
    print(execute_legacy_migration(repository, records=records))


if __name__ == "__main__":
    main()

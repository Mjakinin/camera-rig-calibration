"""Command line interface for pre-solver Route-2 parity evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from .audit import generate_audit
from .compare import compare_ordered_rows
from .evidence import write_csv, write_json
from .finalize import finalize_observation_phase
from .inventory import assert_pre_solver_path
from .materialization import verify_historical_materialization
from .observation_parity import compare_generated_observations


MODES = ("end_to_end", "frozen_observations")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    assert_pre_solver_path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    assert_pre_solver_path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(arguments: argparse.Namespace) -> int:
    generate_audit(
        arguments.repository_root,
        arguments.dataset_root,
        arguments.output,
        tests_summary=arguments.tests_summary,
    )
    return 0


def _compare_observations(arguments: argparse.Namespace) -> int:
    main_rows = _csv_rows(arguments.main)
    wizard_rows = _csv_rows(arguments.wizard)
    report, differences = compare_ordered_rows(
        main_rows,
        wizard_rows,
        float_fields=arguments.float_field,
        float_tolerance=arguments.float_tolerance,
        continue_after_mismatch=arguments.complete_diff,
    )
    report.update(
        {
            "mode": arguments.mode,
            "main_path": str(arguments.main.resolve()),
            "wizard_path": str(arguments.wizard.resolve()),
            "main_sha256": _sha(arguments.main),
            "wizard_sha256": _sha(arguments.wizard),
            "ground_truth_used": False,
            "solver_invoked": False,
        }
    )
    if arguments.mode == "frozen_observations":
        report["same_frozen_input_bytes"] = (
            report["main_sha256"] == report["wizard_sha256"]
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output / "OBSERVATION_PARITY.json", report)
    fields = [
        "row_index",
        "field",
        "main_value",
        "wizard_value",
        "reason",
        "absolute_delta",
    ]
    write_csv(arguments.output / "OBSERVATION_ROW_DIFF.csv", differences, fields)
    return 0 if report["status"] == "equal" else 1


def _verify_materialization(arguments: argparse.Namespace) -> int:
    result = verify_historical_materialization(
        worktree=arguments.worktree,
        inventory_csv=arguments.inventory,
        output=arguments.output / "HISTORICAL_INPUT_MATERIALIZATION.json",
    )
    return 0 if result["status"] == "verified" else 1


def _compare_generated(arguments: argparse.Namespace) -> int:
    result = compare_generated_observations(
        historical_dataset=arguments.historical_dataset,
        legacy_script=arguments.legacy_script,
        wizard_script=arguments.wizard_script,
        legacy_root=arguments.legacy_root,
        wizard_root=arguments.wizard_root,
        output_root=arguments.output,
        complete_diff=arguments.complete_diff,
    )
    return 0 if result["status"] == "equal" else 1


def _finalize_observations(arguments: argparse.Namespace) -> int:
    finalize_observation_phase(
        repository=arguments.repository_root,
        output=arguments.output,
        tests_summary=arguments.tests_summary,
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="python -m parity.main_route2_v1",
        description="Main Route-2 pre-solver parity harness",
    )
    commands = result.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="generate immutable input/config evidence")
    audit.add_argument("--repository-root", type=Path, default=Path.cwd())
    audit.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("results/simulation/baseline/route2_cpu_ref14_50x50"),
    )
    audit.add_argument("--output", type=Path, default=Path("parity/main_route2_v1"))
    audit.add_argument("--tests-summary", default="pending focused test execution")
    audit.set_defaults(handler=_audit)

    compare = commands.add_parser(
        "compare-observations",
        help="compare ordered detector/frozen-observation records without solvers",
    )
    compare.add_argument("--mode", choices=MODES, required=True)
    compare.add_argument("--main", type=Path, required=True)
    compare.add_argument("--wizard", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=Path("parity/main_route2_v1"))
    compare.add_argument("--float-field", action="append", default=[])
    compare.add_argument("--float-tolerance", type=float, default=0.0)
    compare.add_argument("--complete-diff", action="store_true")
    compare.set_defaults(handler=_compare_observations)

    materialize = commands.add_parser(
        "verify-materialization",
        help="verify the detached Main raw-image worktree against the audit inventory",
    )
    materialize.add_argument("--worktree", type=Path, required=True)
    materialize.add_argument(
        "--inventory",
        type=Path,
        default=Path("parity/main_route2_v1/INPUT_FILE_HASHES.csv"),
    )
    materialize.add_argument(
        "--output", type=Path, default=Path("parity/main_route2_v1")
    )
    materialize.set_defaults(handler=_verify_materialization)

    generated = commands.add_parser(
        "compare-generated",
        help="compare generated Legacy and Wizard pre-solver ArUco evidence",
    )
    generated.add_argument("--historical-dataset", type=Path, required=True)
    generated.add_argument("--legacy-script", type=Path, required=True)
    generated.add_argument("--wizard-script", type=Path, required=True)
    generated.add_argument("--legacy-root", type=Path, required=True)
    generated.add_argument("--wizard-root", type=Path, required=True)
    generated.add_argument(
        "--output", type=Path, default=Path("parity/main_route2_v1")
    )
    generated.add_argument("--complete-diff", action="store_true")
    generated.set_defaults(handler=_compare_generated)

    finalize = commands.add_parser(
        "finalize-observation-phase",
        help="update the lock and final report from verified observation evidence",
    )
    finalize.add_argument("--repository-root", type=Path, default=Path.cwd())
    finalize.add_argument(
        "--output", type=Path, default=Path("parity/main_route2_v1")
    )
    finalize.add_argument("--tests-summary", required=True)
    finalize.set_defaults(handler=_finalize_observations)
    return result


def main() -> int:
    arguments = parser().parse_args()
    return int(arguments.handler(arguments))

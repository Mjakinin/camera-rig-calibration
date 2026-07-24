from __future__ import annotations

import argparse
import json
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage


def _read_optional(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run(*, output_root: Path) -> StageResult:
    stage_root = output_root / "report"

    def action() -> dict[str, Path | str]:
        single = _read_optional(
            output_root
            / "scale_single"
            / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
        )
        multi = _read_optional(
            output_root
            / "scale_multi"
            / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
        )
        multi_success = multi.get("scale_m_per_colmap_unit") is not None
        if not multi_success:
            raise RuntimeError("AP03 multi-marker primary scale is unavailable")
        single_success = single.get("scale_m_per_colmap_unit") is not None
        status = "OK" if single_success else "PARTIAL"
        report = {
            "schema_version": 5,
            "method": "AP03",
            "status": status,
            "primary_result": "multi",
            "colmap_runs": 1,
            "single": single,
            "multi": multi,
            "shared_scale_configuration": True,
        }
        report_json = output_root / "AP03_REPORT.json"
        report_json.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        report_text = output_root / "AP03_REPORT.txt"
        report_text.write_text(
            "\n".join(
                [
                    "AP03 COMBINED RESULT",
                    "=" * 72,
                    f"Status: {status}",
                    "Primary result: multi-marker scale",
                    "Diagnostic result: single-marker scale",
                    "COLMAP reconstructions: 1",
                    "Single and multi use one shared RANSAC configuration.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        for name in ("evaluation_single", "evaluation_multi"):
            (output_root / name).mkdir(parents=True, exist_ok=True)
        method_status = output_root / "METHOD_STATUS.json"
        method_status.write_text(
            json.dumps(
                {
                    "method": "AP03",
                    "status": status,
                    "success": True,
                    "primary_result": "multi",
                    "single_diagnostic_success": single_success,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "method_status": method_status,
            "json_report": report_json,
            "text_report": report_text,
            "status": status,
        }

    return run_stage(
        "ap03.report",
        stage_root,
        action,
        inputs={
            "single_scale": output_root / "scale_single",
            "multi_scale": output_root / "scale_multi",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(output_root=args.out.resolve())


if __name__ == "__main__":
    main()

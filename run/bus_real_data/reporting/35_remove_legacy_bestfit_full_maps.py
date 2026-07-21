#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SECONDARY = ROOT / "results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/details/secondary"
OLD_HEADER = "AP02 OPTIONAL GT-ALIGNED FULL MAP"
INLINE_BEGIN = "=== AP02 REF14-ANCHORED AVAILABLE MAP BEGIN:"


def is_variant_divider(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 40 and set(stripped) == {"#"}


def clean(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    output: list[str] = []
    skipping = False
    removed = 0

    for line in lines:
        if not skipping and line.strip() == OLD_HEADER:
            skipping = True
            removed += 1
            while output and not output[-1].strip():
                output.pop()
            continue

        if skipping:
            if line.startswith(INLINE_BEGIN) or is_variant_divider(line):
                skipping = False
                if output and output[-1].strip():
                    output.append("")
                output.append(line)
            continue

        output.append(line)

    while output and not output[-1].strip():
        output.pop()
    return "\n".join(output) + "\n", removed


def main() -> None:
    if not SECONDARY.is_dir():
        raise SystemExit(f"[ERROR] Missing secondary report directory: {SECONDARY}")

    total = 0
    for report in sorted(SECONDARY.glob("*_MAP_TO_GT.txt")):
        original = report.read_text(encoding="utf-8", errors="replace")
        updated, removed = clean(original)
        report.write_text(updated, encoding="utf-8")
        total += removed
        print(f"[OK] {report.name}: removed {removed} legacy best-fit full-map block(s)")

    print(f"[OK] Removed {total} legacy best-fit full-map block(s) total")


if __name__ == "__main__":
    main()

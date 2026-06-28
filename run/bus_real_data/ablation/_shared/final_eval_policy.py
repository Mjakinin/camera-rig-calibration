#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
from statistics import mean

CAM_MAP = {
    "cam0": "cam_edge_0",
    "cam1": "cam_edge_1",
    "cam3": "cam_edge_3",
    "cam5": "cam_edge_5",
}

CAM_ORDER = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

METHOD_POLICY = {
    "AP01": {
        "canonical": "AP01",
        "source_policy": "relative cam3-local camera-map; evaluation-only SE(3) alignment to GT camera map; no scale; no absolute-world-pose claim",
    },
    "AP02": {
        "canonical": "AP02",
        "source_policy": "official/full-map GT-aligned SE(3) evaluation; no scale; static cameras + markers 0..13 aligned, marker14 held out",
    },
    "AP03-SINGLE-REF14": {
        "canonical": "AP03-SINGLE-REF14",
        "source_policy": "targetless COLMAP reconstruction followed by single Ref14 metric registration; GT camera poses evaluation-only",
    },
    "AP03-MULTI-ARUCO": {
        "canonical": "AP03-MULTI-ARUCO",
        "source_policy": "targetless COLMAP reconstruction followed by Multi-ArUco metric registration; GT camera poses evaluation-only",
    },
}

def sha16(path: Path) -> str:
    if not path.exists():
        return "-"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def fmt(x) -> str:
    if x is None or str(x).strip() in {"", "-"}:
        return "-"
    return f"{float(x):.3f}".rstrip("0").rstrip(".")

def mean_float(values) -> str:
    vals = [float(v) for v in values if str(v).strip() not in {"", "-"}]
    if not vals:
        return "-"
    return fmt(mean(vals))

def parse_clean_final_table(path: Path) -> list[dict]:
    """
    Parse clean comparison TXT created by write_clean_ablation_comparison.py.
    Expected columns:
    resolution | method | status | mean_t_cm | mean_r_deg | cam0_t | cam0_r | ...
    """
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    header = None

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip("\n")
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 14:
            continue

        if parts[0] == "resolution" and parts[1] == "method":
            header = parts
            continue

        if header and set(parts[0]) <= {"-", "+"}:
            continue

        if header and parts[0] and parts[1]:
            # Skip policy bullets accidentally containing pipes.
            if parts[0].startswith("- ") or parts[0].lower() == "scope:":
                continue
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            if row.get("resolution") and row.get("method"):
                rows.append(row)

    return rows

def row_to_camera_rows(row: dict, method: str) -> list[dict]:
    out = []
    for short, cam in CAM_MAP.items():
        out.append({
            "method": method,
            "status": row["status"],
            "entity_type": "camera",
            "entity_id": cam,
            "translation_error_cm": row[f"{short}_t"],
            "rotation_error_deg": row[f"{short}_r"],
            "source_policy": METHOD_POLICY[method]["source_policy"],
            "note": row.get("note", ""),
        })
    return out

def validate_four_camera_csv(path: Path) -> tuple[bool, str]:
    rows = read_csv(path)
    found = set()
    for r in rows:
        cam = r.get("entity_id") or r.get("camera") or r.get("cam") or ""
        if cam in CAM_ORDER:
            found.add(cam)
    ok = sorted(found) == sorted(CAM_ORDER)
    return ok, ",".join(sorted(found)) if found else "-"

def classify_ap03_single(mean_t_cm: float, max_t_cm: float, mean_r_deg: float, max_r_deg: float) -> str:
    if mean_t_cm > 100 or max_t_cm > 300 or mean_r_deg > 20 or max_r_deg > 45:
        return "REJECT_UNSTABLE"
    if mean_t_cm > 25 or max_t_cm > 75 or mean_r_deg > 2.0 or max_r_deg > 5.0:
        return "UNSTABLE"
    return "OK"

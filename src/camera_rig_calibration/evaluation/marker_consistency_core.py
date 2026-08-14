#!/usr/bin/env python3
"""Real-data marker-length and reprojection evaluation for AP01/AP02/AP03.

Marker corners are triangulated only from moving-camera observations. A metric
anchor fixes the scale for each method; all other reconstructed markers validate
the estimated scale and the moving-to-static camera relationship. Calibration
methods are never re-run or re-optimized by this evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ("cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5")
METHOD_DIRS = {"AP01": "02_ap01_real", "AP02": "03_ap02_real", "AP03": "04_ap03_real"}


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument("--observations-root", default="")
    p.add_argument("--anchor-marker-id", type=int)
    p.add_argument("--marker-length-m", type=float, default=0.17)
    p.add_argument("--reprojection-threshold-px", type=float, default=5.0)
    p.add_argument("--min-inliers", type=int, default=4)
    p.add_argument("--ransac-iters", type=int, default=800)
    p.add_argument("--min-triangulation-angle-deg", type=float, default=0.5)
    p.add_argument("--max-moving-observations-per-marker", type=int, default=80)
    p.add_argument(
        "--cameras",
        default=",".join(CAMERAS),
        help="Comma-separated static camera IDs.",
    )
    p.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=DIRECTORY",
        help="Method label and directory relative to --results-root; repeatable.",
    )
    p.add_argument(
        "--method-anchor",
        action="append",
        default=[],
        metavar="NAME=MARKER_ID",
        help=(
            "Optional metric anchor for one --method entry. When omitted, "
            "--anchor-marker-id is used."
        ),
    )
    p.add_argument(
        "--output-root",
        default="",
        help=("Output directory (default: " "workspace/standalone_evaluation)."),
    )
    return p.parse_args()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def num(value, default=float("nan")):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def vals(values):
    return [num(v) for v in values if math.isfinite(num(v))]


def med(values):
    values = vals(values)
    return float(np.median(values)) if values else None


def rmse(values):
    values = np.asarray(vals(values), dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else None


def T(R=np.eye(3), t=np.zeros(3)):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return out


def inv(T0):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T0[:3, :3].T
    out[:3, 3] = -T0[:3, :3].T @ T0[:3, 3]
    return out


def qR(qw, qx, qy, qz):
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-15)
    qw, qx, qy, qz = q
    return np.asarray(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def Rq(R):
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-15:
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    axis = rvec.reshape(3) / angle
    return np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)]


def rotation_error(A, B):
    x = float(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))
    return math.degrees(math.acos(x))


def mean_transform(transforms, weights):
    translations = np.asarray([x[:3, 3] for x in transforms])
    weights = np.asarray([max(float(x), 1e-12) for x in weights])
    weights /= weights.sum()
    t0 = np.median(translations, axis=0)
    quats = [Rq(x[:3, :3]) for x in transforms]
    ref = quats[0]
    A = np.zeros((4, 4))
    for q, w in zip(quats, weights):
        if np.dot(q, ref) < 0:
            q = -q
        A += w * np.outer(q, q)
    _, V = np.linalg.eigh(A)
    R0 = qR(*V[:, -1])
    td = np.linalg.norm(translations - t0, axis=1)
    rd = np.asarray([rotation_error(R0, x[:3, :3]) for x in transforms])
    tm, rm = np.median(td), np.median(rd)
    tmad = 1.4826 * np.median(np.abs(td - tm))
    rmad = 1.4826 * np.median(np.abs(rd - rm))
    keep = [
        i
        for i, (a, b) in enumerate(zip(td, rd))
        if a <= max(0.25, tm + 3 * tmad) and b <= max(8, rm + 3 * rmad)
    ]
    if not keep:
        keep = [int(np.argmax(weights))]
    w = weights[keep]
    w /= w.sum()
    t = np.sum(translations[keep] * w[:, None], axis=0)
    ref = quats[keep[0]]
    A = np.zeros((4, 4))
    for i, wi in zip(keep, w):
        q = quats[i]
        if np.dot(q, ref) < 0:
            q = -q
        A += wi * np.outer(q, q)
    _, V = np.linalg.eigh(A)
    q = V[:, -1]
    if q[0] < 0:
        q = -q
    return T(qR(*q), t)


def pose_rows(path, allowed=None):
    result = {}
    for row in read_csv(path):
        entity = str(row.get("entity_id", row.get("observer_id", "")))
        if allowed is not None and entity not in allowed:
            continue
        r = np.asarray(
            [num(row.get("rvec_x")), num(row.get("rvec_y")), num(row.get("rvec_z"))]
        )
        t = np.asarray([num(row.get("x_m")), num(row.get("y_m")), num(row.get("z_m"))])
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
            continue
        R, _ = cv2.Rodrigues(r.reshape(3, 1))
        result[entity] = T(R, t)
    return result


def status(method_root):
    path = method_root / "METHOD_STATUS.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def static_poses(method_root):
    s = status(method_root)
    value = s.get("pose_file")
    candidates = [
        method_root
        / "graph_ba/with_moving/optimized_static_camera_poses_ref_marker.csv",
        method_root
        / "07_graph_ba/with_moving/optimized_static_camera_poses_ref_marker.csv",
    ]
    if value:
        p = Path(str(value))
        candidates += [
            p,
            (
                method_root / Path(*p.parts[p.parts.index(method_root.name) + 1 :])
                if method_root.name in p.parts
                else p
            ),
        ]
        candidates += list(method_root.rglob(p.name))
    candidates += list(method_root.rglob("*STATIC*CAMERA*POSES*.csv"))
    candidates += list(
        method_root.rglob("optimized_static_camera_poses_ref_marker.csv")
    )
    for path in candidates:
        if path.is_file():
            poses = pose_rows(path, set(CAMERAS))
            if poses:
                return poses, path
    raise RuntimeError("No static camera pose file")


def frame_id(value):
    m = re.findall(r"(\d+)", str(value))
    return int(m[-1]) if m else None


def colmap_images(path):
    result = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 10 or not parts[9].lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        name = Path(parts[9]).name
        if "moving" not in name and not name.startswith("frame_"):
            continue
        f = frame_id(Path(name).stem)
        if f is None:
            continue
        result[f] = T(
            qR(*(float(x) for x in parts[1:5])), [float(x) for x in parts[5:8]]
        )
    return result


def scaled_colmap_poses(images_txt, scale):
    out = {}
    for f, Tcw in colmap_images(images_txt).items():
        Tcw = Tcw.copy()
        Tcw[:3, 3] *= scale
        out[f] = inv(Tcw)
    return out


def success(row):
    return str(row.get("pnp_success", "")).lower() in {"true", "1", "yes"}


def area(row):
    return max(num(row.get("area_px2"), 0), 0)


def quality(row):
    return math.sqrt(max(area(row), 1)) / max(num(row.get("distance_m"), 99), 0.1)


def pnp_pose(row):
    r = np.asarray(
        [num(row.get("rvec_x")), num(row.get("rvec_y")), num(row.get("rvec_z"))]
    )
    t = np.asarray(
        [num(row.get("tvec_x_m")), num(row.get("tvec_y_m")), num(row.get("tvec_z_m"))]
    )
    R, _ = cv2.Rodrigues(r.reshape(3, 1))
    return T(R, t)


def best_static(rows):
    out = {}
    for row in rows:
        if not success(row) or row.get("observer_type") != "static":
            continue
        cam = str(row.get("camera_name", row.get("observer_id", "")))
        if cam not in CAMERAS:
            continue
        marker = int(float(row["marker_id"]))
        key = (cam, marker)
        if key not in out or area(row) > area(out[key]):
            out[key] = row
    return out


def moving_rows(rows):
    return [r for r in rows if success(r) and r.get("observer_type") == "moving"]


def load_ap01(root, static_pose, static_obs, moving_obs, anchor):
    diag_path = root / "static_extrinsics/AP01_DIAGNOSTICS.json"
    if not diag_path.is_file():
        diag_path = root / "03_static_extrinsics/AP01_DIAGNOSTICS.json"
    if not diag_path.is_file():
        diag_path = next(iter(root.rglob("AP01_DIAGNOSTICS.json")))
    diag = json.loads(diag_path.read_text())
    scale = num(diag["metric_scale"]["scale_m_per_colmap_unit"])
    root_cam = diag.get("root_camera", "cam_edge_3")
    images = root / "moving_colmap/sparse_txt_best/images.txt"
    if not images.is_file():
        images = root / "01_moving_colmap/sparse_txt_best/images.txt"
    if not images.is_file():
        images = next(iter(root.rglob("sparse_txt_best/images.txt")))
    col = scaled_colmap_poses(images, scale)
    root_anchor = static_obs.get((root_cam, anchor))
    if root_anchor is None:
        raise RuntimeError(f"Anchor {anchor} not visible in {root_cam}")
    Trm = pnp_pose(root_anchor)
    candidates = []
    weights = []
    for row in moving_obs:
        if int(float(row["marker_id"])) != anchor:
            continue
        f = frame_id(row.get("frame_id", row.get("observer_id")))
        if f not in col:
            continue
        Trootmoving = Trm @ inv(pnp_pose(row))
        candidates.append(Trootmoving @ inv(col[f]))
        weights.append(quality(row))
    if not candidates:
        raise RuntimeError("No AP01 anchor alignment observation")
    Trootcol = mean_transform(candidates, weights)
    return {f: Trootcol @ pose for f, pose in col.items()}, {
        "source": str(images),
        "native_scale": scale,
        "anchor_alignment_observations": len(candidates),
    }


def load_ap02(root):
    path = root / "graph_ba/with_moving/optimized_moving_frame_poses_ref_marker.csv"
    if not path.is_file():
        path = (
            root / "07_graph_ba/with_moving/optimized_moving_frame_poses_ref_marker.csv"
        )
    if not path.is_file():
        path = next(
            iter(root.rglob("with_moving/optimized_moving_frame_poses_ref_marker.csv"))
        )
    named = pose_rows(path)
    out = {}
    for name, pose in named.items():
        f = frame_id(name)
        if f is not None:
            out[f] = pose
    if not out:
        raise RuntimeError("No AP02 moving poses")
    return out, {"source": str(path)}


def load_ap03(root):
    path = root / "scale_multi/AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
    if not path.is_file():
        path = root / "07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
    if not path.is_file():
        candidates = sorted(root.rglob("AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"))
        path = next(
            (value for value in candidates if "scale_multi" in value.parts),
            candidates[0],
        )
    meta = json.loads(path.read_text())
    scale = num(meta.get("scale_m_per_colmap_unit"))
    images = Path(str(meta.get("model_dir", ""))) / "images.txt"
    if not images.is_file():
        images = (
            root
            / "colmap/reconstruction/sparse_txt"
            / str(meta.get("best_model", "0"))
            / "images.txt"
        )
    if not images.is_file():
        images = (
            root
            / "02_colmap_sparse/sparse_txt"
            / str(meta.get("best_model", "0"))
            / "images.txt"
        )
    if not images.is_file():
        candidates = sorted(
            root.rglob(f"sparse_txt/{meta.get('best_model','0')}/images.txt")
        )
        if candidates:
            images = candidates[0]
    out = scaled_colmap_poses(images, scale)
    if not out:
        raise RuntimeError("No AP03 moving poses")
    return out, {"source": str(images), "native_scale": scale}


def load_poses(method, root, static_obs, moving_obs, anchor):
    sposes, spfile = static_poses(root)
    if method.startswith("AP01"):
        mposes, meta = load_ap01(root, sposes, static_obs, moving_obs, anchor)
    elif method.startswith("AP02"):
        mposes, meta = load_ap02(root)
    else:
        mposes, meta = load_ap03(root)
    return sposes, mposes, spfile, meta


def camera(row):
    K = np.asarray(
        [
            [num(row.get("fx")), 0, num(row.get("cx"))],
            [0, num(row.get("fy")), num(row.get("cy"))],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    d = [num(row.get(f"d{i}"), 0) for i in range(8)]
    model = str(row.get("distortion_model", "plumb_bob")).lower()
    return K, np.asarray(
        d[:4] if model in {"equidistant", "fisheye"} else d, dtype=np.float64
    )


def obs(row, Twc, corner):
    K, D = camera(row)
    point = np.asarray(
        [num(row.get(f"corner{corner}_u")), num(row.get(f"corner{corner}_v"))]
    )
    return {"point": point, "K": K, "D": D, "Twc": Twc}


def und(o):
    return cv2.undistortPoints(
        o["point"].reshape(1, 1, 2).astype(np.float64), o["K"], o["D"]
    ).reshape(2)


def ray(o):
    x, y = und(o)
    r = np.asarray([x, y, 1.0])
    r /= np.linalg.norm(r)
    r = o["Twc"][:3, :3] @ r
    return r / np.linalg.norm(r)


def angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(ray(a), ray(b)), -1, 1))))


def triangulate(observations):
    A = []
    for o in observations:
        x, y = und(o)
        P = inv(o["Twc"])[:3, :]
        A += [x * P[2] - P[0], y * P[2] - P[1]]
    _, _, V = np.linalg.svd(np.asarray(A))
    X = V[-1]
    if abs(X[3]) < 1e-12:
        raise RuntimeError("point at infinity")
    return X[:3] / X[3]


def project(X, o):
    Tcw = inv(o["Twc"])
    Xc = Tcw[:3, :3] @ X + Tcw[:3, 3]
    if Xc[2] <= 1e-9:
        return None
    r, _ = cv2.Rodrigues(Tcw[:3, :3])
    p, _ = cv2.projectPoints(X.reshape(1, 3), r, Tcw[:3, 3], o["K"], o["D"])
    return p.reshape(2)


def errors(X, observations):
    out = []
    for o in observations:
        p = project(X, o)
        out.append(float("inf") if p is None else float(np.linalg.norm(p - o["point"])))
    return out


def robust_triangulate(observations, args, seed):
    if len(observations) < args.min_inliers:
        return None, [], [], 0
    pairs = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            a = angle(observations[i], observations[j])
            if a >= args.min_triangulation_angle_deg:
                pairs.append((a, i, j))
    if not pairs:
        return None, [], [], 0
    pairs.sort(reverse=True)
    pairs = pairs[: min(300, len(pairs))]
    rng = random.Random(seed)
    best = None
    for _ in range(args.ransac_iters):
        a, i, j = rng.choice(pairs)
        try:
            X = triangulate([observations[i], observations[j]])
        except Exception:
            continue
        e = errors(X, observations)
        inside = [
            k
            for k, x in enumerate(e)
            if math.isfinite(x) and x <= args.reprojection_threshold_px
        ]
        score = (len(inside), -med([e[k] for k in inside]) if inside else -1e12, a)
        if best is None or score > best[0]:
            best = (score, inside)
    if best is None or len(best[1]) < args.min_inliers:
        return None, [], [], pairs[0][0]
    inside = best[1]
    X = triangulate([observations[k] for k in inside])
    e = errors(X, observations)
    inside = [
        k
        for k, x in enumerate(e)
        if math.isfinite(x) and x <= args.reprojection_threshold_px
    ]
    if len(inside) < args.min_inliers:
        return None, [], [], pairs[0][0]
    X = triangulate([observations[k] for k in inside])
    return X, inside, errors(X, observations), pairs[0][0]


def selected_marker_rows(rows, mposes, marker, maximum):
    out = []
    for row in rows:
        if int(float(row["marker_id"])) != marker:
            continue
        f = frame_id(row.get("frame_id", row.get("observer_id")))
        if f in mposes:
            out.append(row)
    if len(out) <= maximum:
        return out
    out = sorted(out, key=quality, reverse=True)[: maximum * 3]
    out.sort(key=lambda r: frame_id(r.get("frame_id")) or 0)
    return [
        out[i] for i in sorted(set(np.linspace(0, len(out) - 1, maximum, dtype=int)))
    ]


def marker_lengths(c):
    return [
        np.linalg.norm(c[1] - c[0]),
        np.linalg.norm(c[2] - c[1]),
        np.linalg.norm(c[3] - c[2]),
        np.linalg.norm(c[0] - c[3]),
        np.linalg.norm(c[2] - c[0]) / math.sqrt(2),
        np.linalg.norm(c[3] - c[1]) / math.sqrt(2),
    ]


def evaluate(method, sposes, mposes, spfile, meta, srows, mrows, anchor, length, args):
    pose_frames = sorted(mposes)
    marker_frames = sorted(
        {
            frame_id(r.get("frame_id", r.get("observer_id")))
            for r in mrows
            if frame_id(r.get("frame_id", r.get("observer_id"))) is not None
        }
    )
    candidate_frames = sorted(set(pose_frames).intersection(marker_frames))
    markers = sorted({int(float(r["marker_id"])) for r in mrows})
    reconstructed = {}
    reproj_rows = []
    for marker in markers:
        rows = selected_marker_rows(
            mrows, mposes, marker, args.max_moving_observations_per_marker
        )
        if len(rows) < args.min_inliers:
            continue
        corners = []
        fit = []
        frames = set()
        angles = []
        ok = True
        for ci in range(4):
            oo = []
            ff = []
            for row in rows:
                f = frame_id(row.get("frame_id", row.get("observer_id")))
                if f not in mposes:
                    continue
                oo.append(obs(row, mposes[f], ci))
                ff.append(f)
            X, inside, e, a = robust_triangulate(oo, args, 7919 + 17 * marker + ci)
            if X is None:
                ok = False
                break
            corners.append(X)
            angles.append(a)
            for k in inside:
                fit.append(e[k])
                frames.add(ff[k])
        if not ok:
            continue
        static_errors = []
        static_cams = []
        for cam in CAMERAS:
            row = srows.get((cam, marker))
            if row is None or cam not in sposes:
                continue
            camera_errors = []
            for ci, X in enumerate(corners):
                o = obs(row, sposes[cam], ci)
                p = project(X, o)
                e = float("inf") if p is None else float(np.linalg.norm(p - o["point"]))
                if math.isfinite(e):
                    static_errors.append(e)
                    camera_errors.append(e)
                reproj_rows.append(
                    {
                        "method": method,
                        "marker_id": marker,
                        "corner_index": ci,
                        "static_camera": cam,
                        "cross_camera_reprojection_error_px": e,
                    }
                )
            if camera_errors:
                static_cams.append(cam)
        ll = marker_lengths(corners)
        reconstructed[marker] = {
            "raw": float(np.median(ll)),
            "fit": fit,
            "cross": static_errors,
            "cams": static_cams,
            "frames": frames,
            "angle": max(angles),
        }
    inlier_frames = sorted(
        set().union(*(data["frames"] for data in reconstructed.values()))
        if reconstructed
        else set()
    )
    frame_support = {
        "pose_frame_ids": pose_frames,
        "pose_frame_count": len(pose_frames),
        "marker_frame_ids": marker_frames,
        "marker_frame_count": len(marker_frames),
        "candidate_frame_ids": candidate_frames,
        "candidate_frame_count": len(candidate_frames),
        "inlier_frame_ids": inlier_frames,
        "inlier_frame_count": len(inlier_frames),
        "rejected_frame_ids": {
            "pose_without_marker": sorted(set(pose_frames) - set(marker_frames)),
            "marker_without_pose": sorted(set(marker_frames) - set(pose_frames)),
            "candidate_not_used_as_inlier": sorted(
                set(candidate_frames) - set(inlier_frames)
            ),
        },
    }
    if anchor not in reconstructed:
        return (
            {
                "method": method,
                "status": "NOT_AVAILABLE_ANCHOR_NOT_RECONSTRUCTED",
                "available_static_cameras": sorted(sposes),
                "available_static_camera_count": len(sposes),
                "registered_moving_frames": len(mposes),
                "static_pose_file": str(spfile),
                "trajectory": meta,
                "anchor_marker_id": anchor,
                "reconstructed_markers_total": len(reconstructed),
                "evaluated_non_anchor_markers": 0,
                **frame_support,
            },
            [],
            reproj_rows,
        )
    scale = length / reconstructed[anchor]["raw"]
    marker_rows = []
    size_cm = []
    size_pct = []
    cross = []
    fit = []
    validated = 0
    for marker, data in sorted(reconstructed.items()):
        estimated = data["raw"] * scale
        is_anchor = marker == anchor
        err_cm = 100 * abs(estimated - length)
        err_pct = 100 * abs(estimated - length) / length
        if not is_anchor:
            size_cm.append(err_cm)
            size_pct.append(err_pct)
            cross += data["cross"]
            fit += data["fit"]
            validated += int(bool(data["cross"]))
        marker_rows.append(
            {
                "method": method,
                "marker_id": marker,
                "is_scale_anchor": is_anchor,
                "moving_inlier_frame_count": len(data["frames"]),
                "moving_inlier_frames": ";".join(map(str, sorted(data["frames"]))),
                "max_triangulation_angle_deg": data["angle"],
                "static_validation_cameras": ";".join(data["cams"]),
                "static_validation_camera_count": len(data["cams"]),
                "raw_reconstructed_size_units": data["raw"],
                "anchor_scale_m_per_unit": scale,
                "estimated_marker_size_cm": 100 * estimated,
                "expected_marker_size_cm": 100 * length,
                "absolute_size_error_cm": 0 if is_anchor else err_cm,
                "relative_size_error_percent": 0 if is_anchor else err_pct,
                "moving_fit_reprojection_rmse_px": rmse(data["fit"]),
                "moving_to_static_reprojection_rmse_px": rmse(data["cross"]),
                "moving_to_static_reprojection_median_px": med(data["cross"]),
                "moving_to_static_reprojection_observations": len(data["cross"]),
            }
        )
    expected = len(CAMERAS)
    summary = {
        "method": method,
        "status": (
            "OK" if len(sposes) == expected else f"PARTIAL_{len(sposes)}_OF_{expected}"
        ),
        "available_static_cameras": sorted(sposes),
        "available_static_camera_count": len(sposes),
        "registered_moving_frames": len(mposes),
        "static_pose_file": str(spfile),
        "trajectory": meta,
        "anchor_marker_id": anchor,
        "anchor_expected_size_cm": 100 * length,
        "anchor_raw_reconstructed_size_units": reconstructed[anchor]["raw"],
        "anchor_scale_m_per_unit": scale,
        "reconstructed_markers_total": len(reconstructed),
        "evaluated_non_anchor_markers": len(size_cm),
        "markers_with_moving_to_static_validation": validated,
        "marker_length_rmse_cm": rmse(size_cm),
        "marker_length_rmse_percent": rmse(size_pct),
        "median_absolute_size_error_cm": med(size_cm),
        "median_relative_size_error_percent": med(size_pct),
        "moving_fit_reprojection_rmse_px": rmse(fit),
        "moving_fit_reprojection_median_px": med(fit),
        "moving_to_static_reprojection_rmse_px": rmse(cross),
        "moving_to_static_reprojection_median_px": med(cross),
        "moving_to_static_reprojection_observations": len(cross),
        **frame_support,
    }
    return summary, marker_rows, reproj_rows

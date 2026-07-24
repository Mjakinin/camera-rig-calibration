#!/usr/bin/env python3
import math
import random

import cv2
import numpy as np

from run.bus_real_data._shared.common.geometry import clamp, R_to_rvec, median


def undistort_to_normalized(u, v, K, D):
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, np.asarray(K, dtype=np.float64), np.asarray(D, dtype=np.float64))
    return float(und[0, 0, 0]), float(und[0, 0, 1])


def project_colmap_point(X, image_payload, camera_payload):
    X = np.asarray(X, dtype=np.float64).reshape(3)
    T = image_payload["T_cam_col"]
    Xc = T[:3, :3] @ X + T[:3, 3]

    if Xc[2] <= 1e-9:
        return None, False

    rvec = R_to_rvec(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)

    pts, _ = cv2.projectPoints(
        X.reshape(1, 3),
        rvec.reshape(3, 1),
        tvec,
        camera_payload["K"],
        camera_payload["D"],
    )
    u, v = pts.reshape(2)
    return np.array([float(u), float(v)], dtype=np.float64), True


def reproj_errors_px(X, observations, images, cameras):
    errs = []
    for o in observations:
        im = images[o["image_name"]]
        cam = cameras[im["camera_id"]]
        p, ok = project_colmap_point(X, im, cam)
        if not ok:
            errs.append(float("inf"))
            continue
        q = np.array([float(o["u"]), float(o["v"])], dtype=np.float64)
        errs.append(float(np.linalg.norm(p - q)))
    return errs


def triangulate_dlt(observations, images, cameras):
    A = []

    for o in observations:
        im = images[o["image_name"]]
        cam = cameras[im["camera_id"]]

        x, y = undistort_to_normalized(o["u"], o["v"], cam["K"], cam["D"])

        P = np.zeros((3, 4), dtype=np.float64)
        P[:3, :3] = im["T_cam_col"][:3, :3]
        P[:3, 3] = im["T_cam_col"][:3, 3]

        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])

    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1, :]

    if abs(Xh[3]) < 1e-12:
        raise RuntimeError("Triangulation produced invalid homogeneous point")

    return (Xh[:3] / Xh[3]).astype(np.float64)


def ray_world_from_observation(o, images, cameras):
    im = images[o["image_name"]]
    cam = cameras[im["camera_id"]]

    x, y = undistort_to_normalized(o["u"], o["v"], cam["K"], cam["D"])
    ray_cam = np.array([x, y, 1.0], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)

    R_cw = im["T_cam_col"][:3, :3]
    ray_w = R_cw.T @ ray_cam
    ray_w /= np.linalg.norm(ray_w)
    return ray_w


def pair_baseline_angle_deg(o1, o2, images, cameras):
    r1 = ray_world_from_observation(o1, images, cameras)
    r2 = ray_world_from_observation(o2, images, cameras)
    arg = clamp(float(np.dot(r1, r2)))
    return math.degrees(math.acos(arg))


def robust_triangulate_point(
    observations,
    images,
    cameras,
    ransac_iters=1000,
    reproj_thresh_px=5.0,
    min_inliers=4,
    random_seed=7,
):
    if len(observations) < 2:
        raise RuntimeError("Need at least two observations for triangulation")

    rng = random.Random(random_seed)

    pairs = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            angle = pair_baseline_angle_deg(observations[i], observations[j], images, cameras)
            pairs.append((angle, i, j))

    if not pairs:
        raise RuntimeError("No observation pairs for triangulation")

    pairs = sorted(pairs, reverse=True)
    candidate_pairs = pairs[:min(len(pairs), 200)]

    best = None

    for _ in range(int(ransac_iters)):
        _, i, j = rng.choice(candidate_pairs)
        sample = [observations[i], observations[j]]

        try:
            X = triangulate_dlt(sample, images, cameras)
        except Exception:
            continue

        errs = reproj_errors_px(X, observations, images, cameras)
        inlier_idx = [
            k for k, e in enumerate(errs)
            if math.isfinite(e) and e <= reproj_thresh_px
        ]

        score = (
            len(inlier_idx),
            -median([errs[k] for k in inlier_idx]) if inlier_idx else -1e9,
        )

        if best is None or score > best["score"]:
            best = {
                "X": X,
                "inlier_idx": inlier_idx,
                "errs": errs,
                "score": score,
            }

    if best is None or len(best["inlier_idx"]) < min_inliers:
        X = triangulate_dlt(observations, images, cameras)
        errs = reproj_errors_px(X, observations, images, cameras)
        inlier_idx = [i for i, e in enumerate(errs) if math.isfinite(e)]
    else:
        inliers = [observations[i] for i in best["inlier_idx"]]
        X = triangulate_dlt(inliers, images, cameras)
        errs = reproj_errors_px(X, observations, images, cameras)
        inlier_idx = [
            i for i, e in enumerate(errs)
            if math.isfinite(e) and e <= reproj_thresh_px
        ]

    return {
        "X": X,
        "inlier_idx": inlier_idx,
        "all_errors": errs,
        "inlier_count": len(inlier_idx),
        "obs_count": len(observations),
    }

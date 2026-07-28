#!/usr/bin/env python3
import math
import numpy as np
import cv2


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def qvec_to_R(qvec):
    qw, qx, qy, qz = [float(v) for v in qvec]
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def R_to_rpy_deg(R):
    R = np.asarray(R, dtype=np.float64)
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def R_to_rvec(R):
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.reshape(3)


def rvec_to_R(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def invert_transform(transform):
    """Canonical descriptive alias retained for package callers."""
    return invT(transform)


def trans_error_cm(T_est, T_gt):
    dT = invT(T_gt) @ T_est
    return 100.0 * float(np.linalg.norm(dT[:3, 3]))


def rot_error_deg(T_est, T_gt):
    dT = invT(T_gt) @ T_est
    arg = clamp((float(np.trace(dT[:3, :3])) - 1.0) / 2.0)
    return float(math.degrees(math.acos(arg)))


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted([float(x) for x in xs])
    if not xs:
        return float("nan")
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def umeyama_similarity(X, Y):
    """Estimate Y ~= scale * R * X + t."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape != Y.shape or X.shape[0] < 3 or X.shape[1] != 3:
        raise RuntimeError(f"Need Nx3 arrays with N>=3, got {X.shape}, {Y.shape}")

    n = X.shape[0]
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    Xc = X - mu_x
    Yc = Y - mu_y

    var_x = np.mean(np.sum(Xc * Xc, axis=1))
    Sigma = (Yc.T @ Xc) / n

    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / var_x)
    t = mu_y - scale * R @ mu_x
    return scale, R, t


def apply_sim3_point(p_col, scale, R_ref_col, t_ref_col):
    p_col = np.asarray(p_col, dtype=np.float64).reshape(3)
    return scale * (R_ref_col @ p_col) + t_ref_col


def apply_sim3_pose(T_col_cam, scale, R_ref_col, t_ref_col):
    T_col_cam = np.asarray(T_col_cam, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_ref_col @ T_col_cam[:3, :3]
    T[:3, 3] = scale * (R_ref_col @ T_col_cam[:3, 3]) + t_ref_col
    return T


def metric_pose_columns(prefix, T):
    rpy = R_to_rpy_deg(T[:3, :3])
    rvec = R_to_rvec(T[:3, :3])
    return {
        f"{prefix}_x_m": T[0, 3],
        f"{prefix}_y_m": T[1, 3],
        f"{prefix}_z_m": T[2, 3],
        f"{prefix}_roll_deg": rpy[0],
        f"{prefix}_pitch_deg": rpy[1],
        f"{prefix}_yaw_deg": rpy[2],
        f"{prefix}_rvec_x": rvec[0],
        f"{prefix}_rvec_y": rvec[1],
        f"{prefix}_rvec_z": rvec[2],
    }

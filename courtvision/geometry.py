"""Small projective-geometry helpers."""
from __future__ import annotations
import numpy as np


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography (…,3,3) to points (N,2) -> (…,N,2)."""
    pts = np.asarray(pts, dtype=np.float64)
    ph = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], -1)
    q = ph @ np.swapaxes(H, -1, -2)
    return q[..., :2] / q[..., 2:3]


def h_from_4pts(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Batched exact homography from 4 correspondences. src,dst: (B,4,2) -> (B,3,3)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    B = src.shape[0]
    A = np.zeros((B, 8, 8))
    b = np.zeros((B, 8))
    x, y, u, v = src[..., 0], src[..., 1], dst[..., 0], dst[..., 1]
    A[:, 0::2, 0], A[:, 0::2, 1], A[:, 0::2, 2] = x, y, 1
    A[:, 0::2, 6], A[:, 0::2, 7] = -u * x, -u * y
    A[:, 1::2, 3], A[:, 1::2, 4], A[:, 1::2, 5] = x, y, 1
    A[:, 1::2, 6], A[:, 1::2, 7] = -v * x, -v * y
    b[:, 0::2], b[:, 1::2] = u, v
    A = A + 1e-12 * np.eye(8)
    try:
        h = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        h = np.stack([np.linalg.lstsq(A[i], b[i], rcond=None)[0] for i in range(B)])
    H = np.concatenate([h, np.ones((B, 1))], 1).reshape(B, 3, 3)
    return H


def line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    l = np.cross(np.append(p, 1.0), np.append(q, 1.0))
    return l / (np.hypot(l[0], l[1]) + 1e-12)


def intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return np.array([np.inf, np.inf])
    return p[:2] / p[2]


def corners_to_h(template_corners: np.ndarray, img_corners: np.ndarray) -> np.ndarray:
    """(…,4,2) image corners -> (…,3,3) homographies court->image."""
    shp = img_corners.shape[:-2]
    dst = img_corners.reshape(-1, 4, 2)
    src = np.broadcast_to(template_corners, dst.shape)
    return h_from_4pts(src, dst).reshape(*shp, 3, 3)

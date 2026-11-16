"""Metric camera recovered from the court homography (square pixels, principal point at centre)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    f: float
    resid: float            # self-consistency residual of the pinhole constraints (0 = perfect)

    @property
    def C(self):
        return -self.R.T @ self.t

    @property
    def P(self):
        return self.K @ np.concatenate([self.R, self.t[:, None]], 1)

    def project(self, xyz):
        xyz = np.atleast_2d(np.asarray(xyz, np.float64))
        q = np.concatenate([xyz, np.ones((len(xyz), 1))], 1) @ self.P.T
        return q[:, :2] / q[:, 2:3]

    def height_from_pixel(self, xy, uv):
        """Height z (m) of a point above ground position xy that best explains pixel uv.

        Returns (z, px_per_m): px_per_m is the image displacement per metre of height, i.e. the
        observability of vertical motion from this viewpoint.
        """
        xy = np.atleast_2d(xy)
        uv = np.atleast_2d(uv)
        p0 = self.project(np.c_[xy, np.zeros(len(xy))])
        p1 = self.project(np.c_[xy, np.ones(len(xy))])
        d = p1 - p0
        n2 = (d ** 2).sum(1) + 1e-12
        z = ((uv - p0) * d).sum(1) / n2
        lat = ((uv - p0) - z[:, None] * d)
        return z, np.sqrt(n2), np.linalg.norm(lat, axis=1)

    def up_direction(self, xy):
        """Unit image direction of +z at ground point(s) xy."""
        xy = np.atleast_2d(xy)
        d = self.project(np.c_[xy, np.ones(len(xy))]) - self.project(np.c_[xy, np.zeros(len(xy))])
        return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)


def _resid(H, f, c):
    Kinv = np.array([[1 / f, 0, -c[0] / f], [0, 1 / f, -c[1] / f], [0, 0, 1.0]])
    a = Kinv @ H
    n1, n2 = np.linalg.norm(a[:, 0]), np.linalg.norm(a[:, 1])
    e1 = (a[:, 0] @ a[:, 1]) / (n1 * n2)
    e2 = (n1 - n2) / (n1 + n2)
    return e1 * e1 + 4 * e2 * e2


def calibrate_focal(H, size, f_range=(0.35, 6.0)):
    """Focal length (px) from one plane homography. Returns (f, residual, observability).

    For near fronto-parallel views f is unobservable; a weak prior keeps it near ~0.9*width
    and `observability` (ratio worst/best residual over the search range) is close to 1.
    """
    w, h = size
    c = (w / 2.0, h / 2.0)
    fs = w * np.geomspace(f_range[0], f_range[1], 80)
    prior = 2e-5 * np.log(fs / (0.9 * w)) ** 2
    r = np.array([_resid(H, f, c) for f in fs])
    i = int(np.argmin(r + prior))
    lo, hi = fs[max(i - 1, 0)], fs[min(i + 1, len(fs) - 1)]
    for _ in range(25):                                   # golden-section-ish refinement
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if _resid(H, m1, c) + 2e-5 * np.log(m1 / (0.9 * w)) ** 2 < _resid(H, m2, c) + 2e-5 * np.log(m2 / (0.9 * w)) ** 2:
            hi = m2
        else:
            lo = m1
    f = 0.5 * (lo + hi)
    return float(f), float(_resid(H, f, c)), float((r.max() + 1e-6) / (r.min() + 1e-6))


def pose_from_h(H, f, size) -> Camera:
    w, h = size
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    a = np.linalg.inv(K) @ H
    s = 2.0 / (np.linalg.norm(a[:, 0]) + np.linalg.norm(a[:, 1]))
    if a[2, 2] * s < 0:
        s = -s
    a = a * s
    r1, r2, t = a[:, 0], a[:, 1], a[:, 2]
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], 1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if (-R.T @ t)[2] < 0:            # camera must be above the floor
        R[:, 2] *= -1
    return Camera(K, R, t, float(f), _resid(H, f, (w / 2.0, h / 2.0)))

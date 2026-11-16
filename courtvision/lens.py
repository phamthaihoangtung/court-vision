"""Lens module: one-parameter radial distortion from straight lines (plumb-line + RANSAC).

Division model (Fitzgibbon 2001), radius normalised by the half-diagonal:
    x_u = c + (x_d - c) / (1 + lam * r_d^2)
Barrel distortion (wide-angle / action cams) corresponds to lam < 0.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class LensModel:
    lam: float
    size: tuple          # (w, h)
    rms_before: float = 0.0
    rms_after: float = 0.0
    n_chains: int = 0
    n_inliers: int = 0

    @property
    def center(self):
        return np.array([self.size[0] / 2.0, self.size[1] / 2.0])

    @property
    def norm(self):
        return 0.5 * float(np.hypot(*self.size))

    @property
    def significant(self) -> bool:
        """Only undistort when straightening clearly helps; a rectilinear lens (phone, drone)
        gives a small spurious lambda that would warp the frame for no gain."""
        return (abs(self.lam) > 0.02 and self.rms_after < 0.6 * self.rms_before
                and self.rms_before - self.rms_after > 0.3 * max(self.size[0] / 1280.0, 1.0))

    def undistort_points(self, pts):
        return undistort_points(pts, self.lam, self.center, self.norm)

    def distort_points(self, pts):
        return distort_points(pts, self.lam, self.center, self.norm)

    def maps(self):
        """Remap tables: undistorted output pixel -> distorted source pixel."""
        w, h = self.size
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        d = self.distort_points(np.stack([gx.ravel(), gy.ravel()], 1))
        return (d[:, 0].reshape(h, w).astype(np.float32), d[:, 1].reshape(h, w).astype(np.float32))

    def undistort_image(self, img, maps=None):
        mx, my = maps if maps is not None else self.maps()
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def undistort_points(pts, lam, center, norm):
    p = (np.asarray(pts, np.float64) - center) / norm
    r2 = (p ** 2).sum(-1, keepdims=True)
    return center + norm * p / (1.0 + lam * r2)


def distort_points(pts, lam, center, norm):
    p = (np.asarray(pts, np.float64) - center) / norm
    ru = np.sqrt((p ** 2).sum(-1, keepdims=True))
    if abs(lam) < 1e-12:
        return center + norm * p
    disc = np.clip(1.0 - 4.0 * lam * ru ** 2, 0.0, None)
    rd = np.where(ru > 1e-9, (1.0 - np.sqrt(disc)) / (2.0 * lam * np.maximum(ru, 1e-9)), ru)
    scale = np.where(ru > 1e-9, rd / np.maximum(ru, 1e-9), 1.0)
    return center + norm * p * scale


def _line_rms(pts):
    c = pts.mean(0)
    s = np.linalg.svd(pts - c, compute_uv=False)
    return s[-1] / np.sqrt(len(pts))


def extract_chains(img, min_len_frac=0.12, max_chains=60):
    """Link LSD segments into smooth chains that follow (possibly curved) straight-world edges."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(cv2.GaussianBlur(gray, (3, 3), 0))[0]
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(np.float64)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    keep = length > 0.012 * w
    segs, length = segs[keep], length[keep]
    order = np.argsort(-length)[:500]
    segs, length = segs[order], length[order]
    n = len(segs)
    ends = segs.reshape(n, 2, 2)
    dirs = ends[:, 1] - ends[:, 0]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    used = np.zeros(n, bool)
    link_d, link_cos = 0.008 * w, np.cos(np.deg2rad(9))
    chains = []
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        chain = [ends[i, 0], ends[i, 1]]
        for side in (1, 0):                       # grow at the tail, then at the head
            tip = chain[-1] if side else chain[0]
            d = dirs[i] if side else -dirs[i]
            while True:
                cand = np.where(~used)[0]
                if len(cand) == 0:
                    break
                d0 = np.linalg.norm(ends[cand, 0] - tip, axis=1)
                d1 = np.linalg.norm(ends[cand, 1] - tip, axis=1)
                which = d1 < d0
                dmin = np.minimum(d0, d1)
                cdir = np.where(which[:, None], -dirs[cand], dirs[cand])
                # must continue smoothly AND be collinear-ish (rejects the parallel twin edge)
                ok = (dmin < link_d) & ((cdir @ d) > link_cos)
                if not ok.any():
                    break
                j = cand[ok][np.argmin(dmin[ok])]
                far = ends[j, 0] if np.linalg.norm(ends[j, 1] - tip) < np.linalg.norm(ends[j, 0] - tip) else ends[j, 1]
                near = ends[j, 1] if far is ends[j, 0] else ends[j, 0]
                used[j] = True
                d = (far - near) / (np.linalg.norm(far - near) + 1e-9)
                if side:
                    chain += [near, far]
                else:
                    chain = [far, near] + chain
                tip = far
        pts = np.asarray(chain)
        clen = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        if clen > min_len_frac * w and len(pts) >= 4:
            # densify so long segments weigh by length
            dense = [pts[0]]
            for a, b in zip(pts[:-1], pts[1:]):
                k = max(1, int(np.linalg.norm(b - a) / (0.01 * w)))
                dense += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
            chains.append((clen, np.asarray(dense)))
    chains.sort(key=lambda c: -c[0])
    return [c[1] for c in chains[:max_chains]]


def estimate_plumbline(img, iters=40, inlier_px=None, lam_range=(-0.6, 0.3), seed=0) -> LensModel:
    """RANSAC plumb-line: hypothesise lam from one chain, score by straightened chains."""
    h, w = img.shape[:2]
    center, norm = np.array([w / 2.0, h / 2.0]), 0.5 * float(np.hypot(w, h))
    chains = extract_chains(img)
    model = LensModel(0.0, (w, h), n_chains=len(chains))
    if len(chains) < 3:
        return model
    inlier_px = inlier_px or max(0.6, 0.0006 * w)
    lens = np.array([len(c) for c in chains], float)

    def rms(lam, idx):
        e = [(_line_rms(undistort_points(chains[i], lam, center, norm)) ** 2) * lens[i] for i in idx]
        return np.sqrt(sum(e) / lens[list(idx)].sum())

    def fit(idx):
        r = minimize_scalar(lambda l: rms(l, idx), bounds=lam_range, method="bounded",
                            options={"xatol": 1e-4})
        return float(r.x)

    rng = np.random.default_rng(seed)
    all_idx = list(range(len(chains)))
    best = (-1.0, 0.0, [])
    cands = [0.0] + [fit([i]) for i in rng.choice(len(chains), size=min(iters, len(chains)),
                                                  replace=False, p=lens / lens.sum())]
    for lam in cands:
        res = np.array([_line_rms(undistort_points(c, lam, center, norm)) for c in chains])
        inl = [i for i in all_idx if res[i] < inlier_px]
        score = lens[inl].sum() if inl else 0.0
        if score > best[0]:
            best = (score, lam, inl)
    _, lam, inl = best
    if len(inl) >= 3:
        lam = fit(inl)
        res = np.array([_line_rms(undistort_points(c, lam, center, norm)) for c in chains])
        inl = [i for i in all_idx if res[i] < inlier_px] or inl
        lam = fit(inl)
    model.lam = lam
    model.n_inliers = len(inl)
    model.rms_before = float(rms(0.0, inl)) if inl else 0.0
    model.rms_after = float(rms(lam, inl)) if inl else 0.0
    return model

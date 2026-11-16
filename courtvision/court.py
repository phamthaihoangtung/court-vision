"""Court registration: line detection -> RANSAC-style hypothesise-and-verify over line
correspondences -> robust point-on-line refinement -> bootstrap homography uncertainty.

Convention: H maps court metres (x, y, 1) to image pixels of the full-resolution frame.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import cv2
import numpy as np
from .geometry import apply_h, h_from_4pts, intersect
from .templates import CourtTemplate
from .camera import calibrate_focal


@dataclass
class Registration:
    H: np.ndarray                 # court -> image (full-res px)
    H_samples: np.ndarray         # (K,3,3) bootstrap replicas
    coverage: float               # fraction of template line samples lying on detected line pixels
    rms_px: float                 # point-to-line residual of supporting pixels (work-res px)
    ok: bool
    source: str                   # "detect" | "track" | "hold"


# ----------------------------------------------------------------------------- line evidence
def line_response(img, color="white"):
    """Thin-bright-structure response. 'white' uses min(R,G,B), which suppresses coloured floors
    and coloured lines of other sports; 'any' uses luminance (use when your lines are coloured)."""
    if color == "white":
        base = img.min(axis=2)
    else:
        base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(7, int(round(img.shape[1] / 1280 * 11))) | 1
    th = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return th


def line_mask(img, color="white"):
    th = line_response(img, color)
    # hysteresis: weak line pixels survive only when connected to a confident one, so faint
    # lines (shadow, worn paint) are kept without admitting low-contrast texture
    # thresholds follow the sensor-noise floor (median top-hat response of the textureless floor)
    t_lo = max(20.0, 3.0 * float(np.median(th)))
    t_hi = 2.0 * t_lo
    lo = (th > t_lo).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lo, connectivity=8)
    strong = np.zeros(n, bool)
    strong[np.unique(lab[th > t_hi])] = True
    strong[0] = False
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = strong & (stats[:, cv2.CC_STAT_AREA] >= 12) & (extent >= 0.008 * img.shape[1])
    mask = keep[lab].astype(np.uint8)
    return mask, th


def evidence_map(mask, th, ksize):
    """Line evidence in [0,1]: response relative to the strongest lines in the frame, so that
    faint/coloured lines of other sports count far less than the painted volleyball lines."""
    resp = th[mask > 0]
    if resp.size == 0:
        return np.zeros(mask.shape, np.float32)
    ref = 0.7 * np.percentile(resp, 95)
    ev = np.clip(th.astype(np.float32) / max(ref, 1.0), 0, 1) * mask
    return cv2.dilate(ev, np.ones((ksize, ksize), np.uint8))


def detect_segments(th, mask, min_len):
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        segs = lsd.detect(cv2.GaussianBlur(th, (3, 3), 0))[0]
        segs = np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4).astype(np.float64)
    except cv2.error:                                               # LSD missing in some builds
        segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, 40, minLineLength=min_len, maxLineGap=8)
        segs = np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4).astype(np.float64)
    if len(segs) == 0:
        return segs
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    mid = ((segs[:, :2] + segs[:, 2:]) / 2).astype(int)
    h, w = mask.shape
    dm = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    on = dm[np.clip(mid[:, 1], 0, h - 1), np.clip(mid[:, 0], 0, w - 1)] > 0     # colour filter
    keep = (length >= min_len) & on
    segs, length = segs[keep], length[keep]
    return segs[np.argsort(-length)[:400]]


def merge_segments(segs, ang_tol_deg=2.5, dist_tol=7.0, max_gap=None):
    """Greedy merge of collinear segments into lines: dict(l=(a,b,c), p, d, support, t0, t1)."""
    lines = []
    cos_tol = np.cos(np.deg2rad(ang_tol_deg))
    for s in segs:
        p, q = s[:2], s[2:]
        d = (q - p) / (np.linalg.norm(q - p) + 1e-9)
        L = float(np.linalg.norm(q - p))
        hit = None
        for ln in lines:
            if abs(d @ ln["d"]) < cos_tol:
                continue
            nrm = np.array([-ln["d"][1], ln["d"][0]])
            if abs((p - ln["p"]) @ nrm) > dist_tol or abs((q - ln["p"]) @ nrm) > dist_tol:
                continue
            ta, tb = sorted([(p - ln["p"]) @ ln["d"], (q - ln["p"]) @ ln["d"]])
            if max_gap is not None and (ta - ln["t1"] > max_gap or ln["t0"] - tb > max_gap):
                continue
            hit = ln
            break
        if hit is None:
            lines.append({"p": p.copy(), "d": d, "support": L, "t0": 0.0, "t1": L, "pts": [p, q], "w": [L, L]})
        else:
            hit["pts"] += [p, q]
            hit["w"] += [L, L]
            hit["support"] += L
            hit["t0"] = min(hit["t0"], ta)
            hit["t1"] = max(hit["t1"], tb)
    out = []
    for ln in lines:
        pts, w = np.asarray(ln["pts"]), np.asarray(ln["w"])
        c = (pts * w[:, None]).sum(0) / w.sum()
        _, _, vt = np.linalg.svd((pts - c) * np.sqrt(w)[:, None], full_matrices=False)
        d = vt[0]
        t = (pts - c) @ d
        l = np.array([-d[1], d[0], d[1] * c[0] - d[0] * c[1]])
        # both edges of a painted line are merged, so support ~ 2x length; cap by extent
        out.append({"l": l, "p": c, "d": d, "support": min(ln["support"], 2 * (t.max() - t.min())),
                    "a": c + t.min() * d, "b": c + t.max() * d})
    out.sort(key=lambda x: -x["support"])
    return out


# ----------------------------------------------------------------------------- registrar
class CourtRegistrar:
    def __init__(self, template: CourtTemplate, work_width=1280, n_boot=16, line_color="white",
                 max_lines=18, min_coverage=0.45, seed=0):
        self.tpl = template
        self.work_width = work_width
        self.n_boot = n_boot
        self.line_color = line_color
        self.max_lines = max_lines
        self.min_coverage = min_coverage
        self.rng = np.random.default_rng(seed)
        self.samples, self.sample_line = template.sample(0.5)
        self.segs = template.segments
        # bootstrap blocks: ~1.5 m chunks of template line share one resampling weight, and the
        # weights are fixed over time so replicas are temporally coherent (systematic, not jitter)
        self.chunk_len = 1.5
        seg_len = np.hypot(self.segs[:, 2] - self.segs[:, 0], self.segs[:, 3] - self.segs[:, 1])
        self.chunk_off = np.concatenate([[0], np.cumsum(np.ceil(seg_len / self.chunk_len).astype(int))])
        self.boot_w = np.random.default_rng(1234).exponential(1.0, (n_boot, self.chunk_off[-1]))
        self.prev: Registration | None = None
        self.lost = 0
        self.debug = {}

    # -- public -----------------------------------------------------------------------------
    def update(self, frame) -> Registration:
        """Stateful per-frame registration: track from the previous frame, re-detect if lost."""
        reg = None
        if self.prev is not None and self.prev.ok:
            reg = self.track(frame, self.prev)
        if reg is None or not reg.ok:
            det = self.detect(frame)
            if det is not None and det.ok:
                if self.prev is not None:
                    det = self._align_to(det, self.prev)
                reg = det
        if reg is None or not reg.ok:
            self.lost += 1
            if self.prev is None:
                return Registration(np.eye(3), np.tile(np.eye(3), (self.n_boot, 1, 1)), 0.0, 0.0, False, "none")
            p = self.prev
            return Registration(p.H, p.H_samples, 0.0, p.rms_px, False, "hold")
        self.prev = reg
        return reg

    def detect(self, frame) -> Registration | None:
        small, s = self._resize(frame)
        mask, th = line_mask(small, self.line_color)
        h, w = mask.shape
        segs = detect_segments(th, mask, min_len=0.03 * w)
        lines = [l for l in merge_segments(segs, max_gap=0.35 * w) if l["support"] > 0.07 * w][: self.max_lines]
        self.debug = {"mask": mask, "lines": lines, "scale": s}
        if len(lines) < 4:
            return None
        cands = self._hypotheses(lines, mask, th)
        pts = self._mask_points(mask)
        best = None
        tight = evidence_map(mask, th, 3)
        for score0, H0 in cands:
            r = self._refine(H0, pts, (10.0, 5.0, 3.0), (w, h))
            if r is None:
                continue
            H, rms, boots = r
            cov, minline = self._coverage(H, tight)
            _, resid, _ = calibrate_focal(H, (w, h))
            # pinhole-consistency penalty separates fits that explain the lines equally well
            score = cov - 0.5 * np.sqrt(resid) - max(0.0, np.sqrt(resid) - 0.05) - (0.5 if minline < 0.15 else 0.0)
            if best is None or score > best[0]:
                best = (score, H, rms, boots, cov)
        if best is None:
            return None
        score, H, rms, boots, cov = best
        H, boots = self._canonical(H, boots, (w, h))
        S = np.diag([1 / s, 1 / s, 1.0])
        return Registration(S @ H, S @ boots, float(cov), float(rms), bool(score >= self.min_coverage), "detect")

    def track(self, frame, prev: Registration) -> Registration | None:
        small, s = self._resize(frame)
        mask, th = line_mask(small, self.line_color)
        h, w = mask.shape
        S = np.diag([s, s, 1.0])
        r = self._refine(S @ prev.H, self._mask_points(mask), (6.0, 3.0), (w, h))
        if r is None:
            return None
        H, rms, boots = r
        cov, minline = self._coverage(H, evidence_map(mask, th, 3))
        Si = np.diag([1 / s, 1 / s, 1.0])
        ok = cov >= self.min_coverage and minline >= 0.1
        return Registration(Si @ H, Si @ boots, float(cov), float(rms), bool(ok), "track")

    # -- internals --------------------------------------------------------------------------
    def _resize(self, frame):
        s = self.work_width / frame.shape[1]
        if abs(s - 1) < 1e-3:
            return frame, 1.0
        return cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s

    def _mask_points(self, mask, max_pts=9000):
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs, ys], 1).astype(np.float64)
        if len(pts) > max_pts:
            pts = pts[self.rng.choice(len(pts), max_pts, replace=False)]
        return pts

    def _coverage(self, H, dmask):
        h, w = dmask.shape
        p = apply_h(H, self.samples)
        pi = np.round(p).astype(int)
        inside = (pi[:, 0] >= 0) & (pi[:, 0] < w) & (pi[:, 1] >= 0) & (pi[:, 1] < h)
        hit = np.zeros(len(p))
        hit[inside] = dmask[pi[inside, 1], pi[inside, 0]]
        cov = hit.sum() / max(inside.sum(), 0.6 * len(p))
        per = [hit[self.sample_line == j].mean() for j in range(len(self.segs))
               if inside[self.sample_line == j].mean() > 0.5]
        # second-worst line: one line may be fully hidden (e.g. the far attack line behind the net)
        per = sorted(per)
        return float(cov), float(per[min(1, len(per) - 1)] if per else 0.0)

    def _hypotheses(self, lines, mask, th, top=8):
        """Sample 2+2 image lines and a template assignment; verify by line-pixel consensus.
        The space is small enough to enumerate exhaustively (deterministic RANSAC); it is
        randomly subsampled above `budget` hypotheses."""
        h, w = mask.shape
        n = len(lines)
        L = [ln["l"] for ln in lines]
        X = np.full((n, n, 2), np.inf)
        for i, j in combinations(range(n), 2):
            X[i, j] = X[j, i] = intersect(L[i], L[j])

        def in_img(p, m):
            return np.isfinite(p).all() and -m * w <= p[0] <= (1 + m) * w and -m * h <= p[1] <= (1 + m) * h

        def beyond(i, p):
            """How far (px) point p lies beyond the detected extent of line i."""
            ln = lines[i]
            t = (p - ln["a"]) @ ln["d"]
            return max(0.0, -t, t - np.linalg.norm(ln["b"] - ln["a"]))

        # two lines can be opposite sides of the court if they do not meet near either of their
        # detected extents (their intersection is a vanishing point, possibly inside the frame)
        fam = [(i, j) for i, j in combinations(range(n), 2)
               if not np.isfinite(X[i, j]).all() or min(beyond(i, X[i, j]), beyond(j, X[i, j])) > 0.1 * w]
        tx = list(combinations(self.tpl.xs, 2))
        ty = list(combinations(self.tpl.ys, 2))
        Wd, Ld = self.tpl.width, self.tpl.length
        src, dst = [], []
        for (a, b) in fam:
            for (c, d) in fam:
                if len({a, b, c, d}) < 4:
                    continue
                quad = np.array([X[a, c], X[a, d], X[b, d], X[b, c]])
                if not all(in_img(p, 0.6) for p in quad):
                    continue
                e = np.roll(quad, -1, 0) - quad
                cr = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
                if not (np.all(cr > 0) or np.all(cr < 0)):
                    continue
                area = 0.5 * abs(np.sum(quad[:, 0] * np.roll(quad[:, 1], -1) - quad[:, 1] * np.roll(quad[:, 0], -1)))
                if area < 0.01 * w * h:
                    continue
                for (y0, y1) in ty:
                    for (x0, x1) in tx:
                        src.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                        dst.append(quad)
        if not src:
            return []
        src, dst = np.asarray(src, float), np.asarray(dst, float)
        budget = 120000
        if len(src) > budget:
            sel = self.rng.choice(len(src), budget, replace=False)
            src, dst = src[sel], dst[sel]
        dmask = evidence_map(mask, th, 7)
        sub = slice(None, None, 2)
        P, PL = self.samples[sub], self.sample_line[sub]
        onehot = (PL[:, None] == np.arange(len(self.segs))[None]).astype(np.float64)
        cnt = onehot.sum(0)
        scores = np.zeros(len(src))
        for i0 in range(0, len(src), 20000):
            Hs = h_from_4pts(src[i0:i0 + 20000], dst[i0:i0 + 20000])
            ph = np.concatenate([P, np.ones((len(P), 1))], 1)
            q = ph @ np.swapaxes(Hs, 1, 2)
            z = q[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.round(q[..., 0] / z)
                v = np.round(q[..., 1] / z)
            samez = (np.sign(z) == np.sign(z[:, :1])).all(1)
            inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            ui = np.where(inside, u, 0).astype(int)
            vi = np.where(inside, v, 0).astype(int)
            hit = dmask[vi, ui] * inside
            cov = hit.sum(1) / np.maximum(inside.sum(1), 0.6 * len(P))
            per = (hit @ onehot) / cnt
            per_in = (inside @ onehot) / cnt
            per = np.where(per_in > 0.5, per, 1.0)
            scores[i0:i0 + 20000] = cov * (np.sort(per, 1)[:, min(1, per.shape[1] - 1)] >= 0.2) * samez
        order = np.argsort(-scores)
        out, seen = [], []
        for i in order[:200]:
            if scores[i] < 0.35 or len(out) >= top:
                break
            H = h_from_4pts(src[i:i + 1], dst[i:i + 1])[0]
            # signature invariant to the template's mirror symmetries but not to a 90-degree
            # relabelling (same outline, long and short sides swapped), which must stay a candidate
            Lc, Wc = self.tpl.length, self.tpl.width
            sig = np.concatenate([np.sort(apply_h(H, self.tpl.corners).ravel()),
                                  np.sort(apply_h(H, np.array([[Lc / 4, Wc / 2], [3 * Lc / 4, Wc / 2]])).ravel())])
            if any(np.abs(sig - s).max() < 6 for s in seen):
                continue
            seen.append(sig)
            out.append((scores[i], H))
        return out

    def _refine(self, H, pts, radii, size):
        """Robust DLT on point-on-line constraints l^T H^-1 q = 0 (IRLS, Huber).
        Returns (H, rms_px, bootstrap replicas) or None."""
        w, h = size
        if len(pts) < 50:
            return None
        Tn = np.array([[1 / w, 0, -0.5], [0, 1 / w, -0.5 * h / w], [0, 0, 1.0]])      # image norm
        Lc = self.tpl.length
        Tc = np.array([[1 / Lc, 0, -0.5], [0, 1 / Lc, -0.5 * self.tpl.width / Lc], [0, 0, 1.0]])
        qn = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ Tn.T
        segs = self.segs
        # court lines in normalised court coords, unit normal
        e0 = np.c_[segs[:, :2], np.ones(len(segs))] @ Tc.T
        e1 = np.c_[segs[:, 2:], np.ones(len(segs))] @ Tc.T
        ln = np.cross(e0, e1)
        ln /= np.linalg.norm(ln[:, :2], axis=1, keepdims=True)
        A = cm = None
        for r in radii:
            p0, p1 = apply_h(H, segs[:, :2]), apply_h(H, segs[:, 2:])
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                return None
            d = p1 - p0
            n2 = (d ** 2).sum(1) + 1e-9
            rel = pts[:, None, :] - p0[None]
            t = (rel * d[None]).sum(2) / n2
            dist = np.abs(rel[..., 0] * d[None, :, 1] - rel[..., 1] * d[None, :, 0]) / np.sqrt(n2)
            cand = (dist < r) & (t > -0.02) & (t < 1.02)
            nc = cand.sum(1)
            use = nc == 1                                     # drop junction-ambiguous pixels
            j = np.argmax(cand, 1)[use]
            if use.sum() < 40 or len(np.unique(j)) < 4:
                return None
            q = qn[use]
            A = (ln[j][:, :, None] * q[:, None, :]).reshape(-1, 9)       # kron(l, q) rows
            G = Tc @ np.linalg.inv(H) @ np.linalg.inv(Tn)
            g = G.reshape(9) / np.linalg.norm(G)
            wz = np.abs(q @ g.reshape(3, 3)[2]) + 1e-9
            for _ in range(3):
                res = (A @ g) / wz * Lc                         # metres on the court plane
                delta = max(0.01, 1.5 * np.median(np.abs(res)))
                wt = np.where(np.abs(res) <= delta, 1.0, delta / (np.abs(res) + 1e-12)) / wz ** 2
                M = (A * wt[:, None]).T @ A
                g = np.linalg.eigh(M)[1][:, 0]
                wz = np.abs(q @ g.reshape(3, 3)[2]) + 1e-9
            G = np.linalg.inv(Tc) @ g.reshape(3, 3) @ Tn
            if abs(np.linalg.det(G)) < 1e-14:
                return None
            H = np.linalg.inv(G)
            H = H / H[2, 2]
            # bookkeeping for bootstrap
            tt = np.clip(t[use, j], 0, 1 - 1e-9)
            seg_len = np.hypot(segs[j, 2] - segs[j, 0], segs[j, 3] - segs[j, 1])
            cm = (self.chunk_off[j] + np.floor(tt * seg_len / self.chunk_len).astype(int), wt, dist[use, j])
        chunk, wt, dist = cm
        rms = float(np.sqrt(np.mean(np.minimum(dist, radii[-1]) ** 2)))
        boots = np.tile(H, (self.n_boot, 1, 1))
        Aw = A * np.sqrt(wt)[:, None]
        nchunk = self.chunk_off[-1]
        Mc = np.zeros((nchunk, 9, 9))
        np.add.at(Mc, chunk, Aw[:, :, None] * Aw[:, None, :])
        Tci = np.linalg.inv(Tc)
        for k in range(self.n_boot):
            M = np.tensordot(self.boot_w[k], Mc, axes=1)
            gk = np.linalg.eigh(M)[1][:, 0]
            Gk = Tci @ gk.reshape(3, 3) @ Tn
            if abs(np.linalg.det(Gk)) < 1e-14:
                continue
            Hk = np.linalg.inv(Gk)
            boots[k] = Hk / Hk[2, 2]
        return H, rms, boots

    def _canonical(self, H, boots, size):
        """Resolve template symmetries: camera above the floor (proper chirality) and +x
        pointing down/right in the image, so the minimap matches the view."""
        Lc, Wc = self.tpl.length, self.tpl.width
        c = np.array([Lc / 2, Wc / 2])
        o, px, py = apply_h(H, np.array([c, c + [1, 0], c + [0, 1]]))
        flip_y = np.array([[1, 0, 0], [0, -1, Wc], [0, 0, 1.0]])
        rot = np.array([[-1, 0, Lc], [0, -1, Wc], [0, 0, 1.0]])
        if (px - o) @ np.array([1.0, 2.0]) < 0:
            H, boots = H @ rot, boots @ rot
            o, px, py = apply_h(H, np.array([c, c + [1, 0], c + [0, 1]]))
        dx, dy = px - o, py - o
        if dx[0] * dy[1] - dx[1] * dy[0] > 0:
            H, boots = H @ flip_y, boots @ flip_y
        return H / H[2, 2], boots / boots[:, 2:3, 2:3]

    def _align_to(self, det: Registration, prev: Registration) -> Registration:
        """After re-detection keep the same symmetric labelling as before."""
        Lc, Wc = self.tpl.length, self.tpl.width
        rot = np.array([[-1, 0, Lc], [0, -1, Wc], [0, 0, 1.0]])
        c0 = apply_h(prev.H, self.tpl.corners)
        if np.abs(apply_h(det.H @ rot, self.tpl.corners) - c0).sum() < np.abs(apply_h(det.H, self.tpl.corners) - c0).sum():
            det.H = det.H @ rot
            det.H_samples = det.H_samples @ rot
        return det


def smooth_registrations(regs, template: CourtTemplate, fps, tau=0.4):
    """Temporal stabilisation: Gaussian-smooth the image positions of the 4 court corners
    (for H and every bootstrap replica) and rebuild homographies. Held frames are bridged."""
    from scipy.ndimage import gaussian_filter1d
    from .geometry import corners_to_h
    T = len(regs)
    K = regs[0].H_samples.shape[0]
    allH = np.stack([np.concatenate([r.H[None], r.H_samples]) for r in regs])       # (T,K+1,3,3)
    cor = apply_h(allH.reshape(-1, 3, 3), template.corners).reshape(T, K + 1, 4, 2)
    ok = np.array([r.ok for r in regs])
    if ok.sum() >= 2 and not ok.all():
        idx = np.arange(T)
        flat = cor.reshape(T, -1)
        for c in range(flat.shape[1]):
            flat[~ok, c] = np.interp(idx[~ok], idx[ok], flat[ok, c])
        cor = flat.reshape(T, K + 1, 4, 2)
    sigma = max(tau * fps, 1e-3)
    if T > 3:
        cor = gaussian_filter1d(cor, sigma, axis=0, mode="nearest")
    Hs = corners_to_h(template.corners, cor)
    return Hs[:, 0], Hs[:, 1:]

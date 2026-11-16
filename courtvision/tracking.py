"""Track post-processing in court metres: stitching, team assignment, Kalman/RTS smoothing."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .geometry import apply_h


@dataclass
class Track:
    tid: int
    frames: np.ndarray            # indices of processed frames (increasing)
    feet: np.ndarray              # (N,2) foot pixel
    boxes: np.ndarray             # (N,4) xyxy
    colors: np.ndarray | None = None      # (N,3) torso Lab
    team: int = -1
    label: str = ""
    meta: dict = field(default_factory=dict)


def kalman_rts(z, dt, sigma_a=4.0, sigma_m=0.15):
    """Constant-velocity Kalman filter + RTS smoother. z: (T, D) with NaN rows = missing
    (all D columns share the missingness pattern, so one 2x2 covariance serves every column).
    Returns pos (T,D), vel (T,D)."""
    z = np.asarray(z, np.float64)
    T, D = z.shape
    F = np.array([[1, dt], [0, 1.0]])
    Q = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    R = sigma_m ** 2
    valid = np.isfinite(z).all(1)
    first = int(np.argmax(valid))
    x = np.zeros((2, D)); x[0] = z[first]
    P = np.diag([R, 25.0])
    xs_f = np.zeros((T, 2, D)); Ps_f = np.zeros((T, 2, 2))
    xs_p = np.zeros((T, 2, D)); Ps_p = np.zeros((T, 2, 2))
    for t in range(T):
        if t > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xs_p[t], Ps_p[t] = x, P
        if valid[t]:
            k = P[:, 0] / (P[0, 0] + R)
            x = x + k[:, None] * (z[t] - x[0])[None]
            P = P - np.outer(k, P[0])
        xs_f[t], Ps_f[t] = x, P
    xs = xs_f.copy()
    for t in range(T - 2, -1, -1):
        C = Ps_f[t] @ F.T @ np.linalg.inv(Ps_p[t + 1])
        xs[t] = xs_f[t] + C @ (xs[t + 1] - xs_p[t + 1])
    return xs[:, 0], xs[:, 1]


def dense(track: Track, n_frames=None):
    """Dense per-frame arrays between first and last observation (NaN where unobserved)."""
    f0, f1 = int(track.frames[0]), int(track.frames[-1])
    uv = np.full((f1 - f0 + 1, 2), np.nan)
    uv[track.frames - f0] = track.feet
    return f0, f1, uv


def court_positions(track: Track, Hinv):
    return np.stack([apply_h(Hinv[f], track.feet[i:i + 1])[0] for i, f in enumerate(track.frames)])


def _link(a, xa, b, xb, fps, max_gap, max_overlap, same_person_m):
    """Cost of continuing track a with track b, or None if b cannot be a's continuation.

    Gap: b starts after a ends; positions must agree within a running-speed budget.
    Overlap: ByteTrack often starts a new id while still emitting the old one for a few frames
    (up to ~1.6 s seen on real clips). b may start before a ends if b outlives a, the overlap is
    short, and where both are observed they sit on the same spot (same player measured
    <= 0.82 m apart, different players >= 1.23 m on the demo clips). a's overlapping tail is
    dropped by the caller. Returns (cost, n_keep) with n_keep = observations of a to keep.
    """
    t0b = b.frames[0]
    if t0b <= a.frames[0] or b.frames[-1] <= a.frames[-1]:
        return None
    if t0b > a.frames[-1]:                                    # plain gap
        gap = (t0b - a.frames[-1]) / fps
        if gap > max_gap:
            return None
        d = np.linalg.norm(np.median(xb[:3], 0) - np.median(xa[-3:], 0))
        if d > 0.8 + 3.0 * gap:
            return None
        return d + 0.5 * gap, len(a.frames)
    overlap = (a.frames[-1] - t0b + 1) / fps
    if overlap > max_overlap:
        return None
    shared, ia, ib = np.intersect1d(a.frames, b.frames, return_indices=True)
    if len(shared) and np.median(np.linalg.norm(xa[ia] - xb[ib], axis=1)) > same_person_m:
        return None
    n_keep = int(np.searchsorted(a.frames, t0b))             # a's observations before b starts
    if n_keep < 3:
        return None
    d = np.linalg.norm(np.median(xb[:3], 0) - np.median(xa[n_keep - 3:n_keep], 0))
    if d > same_person_m:
        return None
    return d + 0.5 * overlap, n_keep


def _crowded(b, xb, items, i, j, radius, n=5):
    """True if any other track stands within `radius` of b's first observations: an id handover
    inside a cluster (players side by side at the net) is ambiguous, so it is not stitched."""
    for k, (o, xo) in enumerate(items):
        if k in (i, j):
            continue
        _, io, ih = np.intersect1d(o.frames, b.frames[:n], return_indices=True)
        if len(io) and np.min(np.linalg.norm(xo[io] - xb[:n][ih], axis=1)) < radius:
            return True
    return False


def filter_and_stitch(tracks, Hinv, fps, tpl, margin=3.0, min_dur=0.6, max_gap=3.0, max_players=12,
                      max_overlap=2.0, same_person_m=1.0):
    """Keep tracks on/near the court, stitch fragments broken by occlusion or id handover, keep the longest."""
    items = []
    for tr in tracks:
        if len(tr.frames) < max(3, min_dur * fps):
            continue
        xy = court_positions(tr, Hinv)
        med = np.median(xy, 0)
        if not (-margin <= med[0] <= tpl.length + margin and -margin <= med[1] <= tpl.width + margin):
            continue
        items.append([tr, xy])
    items.sort(key=lambda it: it[0].frames[0])
    merged = True
    while merged:
        merged = False
        best = None
        for i, (a, xa) in enumerate(items):
            for j, (b, xb) in enumerate(items):
                if i == j:
                    continue
                link = _link(a, xa, b, xb, fps, max_gap, max_overlap, same_person_m)
                if link is None:
                    continue
                cost, n_keep = link
                if n_keep < len(a.frames) and _crowded(b, xb, items, i, j, same_person_m):
                    continue
                # do not stitch across the net
                if tpl.has_net and (np.median(xa[:, 0]) - tpl.net_x) * (np.median(xb[:, 0]) - tpl.net_x) < 0 \
                        and min(abs(np.median(xa[:, 0]) - tpl.net_x), abs(np.median(xb[:, 0]) - tpl.net_x)) > 0.7:
                    continue
                if best is None or cost < best[0]:
                    best = (cost, i, j, n_keep)
        if best is not None:
            _, i, j, k = best
            a, xa = items[i]; b, xb = items[j]
            col = None if a.colors is None or b.colors is None else np.concatenate([a.colors[:k], b.colors])
            m = Track(a.tid, np.concatenate([a.frames[:k], b.frames]), np.concatenate([a.feet[:k], b.feet]),
                      np.concatenate([a.boxes[:k], b.boxes]), col, meta={"stitched": a.meta.get("stitched", 1) + 1})
            items[i] = [m, np.concatenate([xa[:k], xb])]
            del items[j]
            merged = True
    items.sort(key=lambda it: -len(it[0].frames))
    return [it[0] for it in items[:max_players]]


def _kmeans2(X, iters=20):
    c = np.stack([X[np.argmin(X[:, 0])], X[np.argmax(X[:, 0])]])
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(2), 1)
        for k in (0, 1):
            if (lab == k).any():
                c[k] = X[lab == k].mean(0)
    return lab, c


def assign_teams(tracks, Hinv, tpl):
    """Team = side of the net, refined by jersey-colour clustering for players hugging the net."""
    if not tracks:
        return
    xs = np.array([np.median(court_positions(t, Hinv)[:, 0]) for t in tracks]) - tpl.net_x
    side = (xs > 0).astype(int)
    team = side.copy()
    info = {"method": "side"}
    if all(t.colors is not None and len(t.colors) for t in tracks) and len(tracks) >= 4:
        feats = np.stack([np.median(t.colors, 0) for t in tracks]).astype(float)
        lab, cen = _kmeans2(feats)
        sep = np.linalg.norm(cen[0] - cen[1])
        agree = (lab == side).mean()
        if agree < 0.5:
            lab, agree = 1 - lab, 1 - agree
        info = {"method": "side+colour", "colour_separation": float(sep), "colour_side_agreement": float(agree)}
        if sep > 25 and agree >= 0.75:          # distinct kits that line up with the sides
            near = np.abs(xs) < 1.0
            team[near] = lab[near]
    for t, tm in zip(tracks, team):
        t.team = int(tm)
    for tm in (0, 1):
        grp = sorted([t for t in tracks if t.team == tm], key=lambda t: np.median(court_positions(t, Hinv)[:, 1]))
        for i, t in enumerate(grp):
            t.label = f"{'AB'[tm]}{i + 1}"
    return info

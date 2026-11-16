"""Airborne-phase detection and jump height from single-view geometry.

A jump shows up as a short transient of the foot pixel along the image of the vertical
direction while the ground position (interpolated from take-off to landing) barely moves.
Height is measured two independent ways: (1) geometry - displacement along the projected
vertical through the interpolated ground point; (2) ballistics - h = g*T^2/8 from flight time.
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .geometry import apply_h

G = 9.81


def rolling_nanmedian(x, win):
    win = int(win) | 1
    pad = win // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), constant_values=np.nan)
    v = sliding_window_view(xp, win, axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmedian(v, axis=-1)


def heights(P, xy, uv):
    """Per-frame height of pixel uv above ground point xy. P: (T,3,4). Returns z, px_per_m, lateral_px."""
    g0 = np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))]
    g1 = g0.copy(); g1[:, 2] = 1.0
    q0 = np.einsum("tij,tj->ti", P, g0); q1 = np.einsum("tij,tj->ti", P, g1)
    p0 = q0[:, :2] / q0[:, 2:3]; p1 = q1[:, :2] / q1[:, 2:3]
    d = p1 - p0
    n2 = (d ** 2).sum(1) + 1e-12
    z = ((uv - p0) * d).sum(1) / n2
    lat = np.linalg.norm((uv - p0) - z[:, None] * d, axis=1)
    return z, np.sqrt(n2), lat


def detect_jumps(uv, Hinv, P, fps, min_height=0.15, min_dur=0.3, max_dur=0.9,
                 foot_sigma_px=2.0, max_resid_frac=0.25, ballistic_tol=(0.6, 1.7), min_equiv_accel=7.5,
                 box_h=None, min_box_ratio=0.9, posture_frac=0.03, top_uv=None, min_top_ratio=0.6):
    """Matched-filter jump detector.

    For every candidate flight window [a, b] the ground track is the straight line between the
    positions just before take-off and just after landing (horizontal velocity is constant in
    flight), the foot pixel is converted to height above that line, and a ballistic template
    z = h * 4*tau*(1-tau) is fitted. A window is accepted when the fit is tight and the fitted
    height agrees with free fall for that flight time (h ~ g*T^2/8) - horizontal accelerations
    of a running player cannot mimic that curvature unless the camera looks almost straight down.

    Observability gate: seen from a camera at height Cz and horizontal distance rho, a height z
    shifts the apparent ground point by k*z with k = rho/Cz. A runner would need a horizontal
    acceleration of 8*h*k/T^2 (~ k*g) to fake the bump; windows are accepted only when that
    exceeds `min_equiv_accel` (m/s^2, above what athletes sustain). Under steep, near-overhead
    cameras (k << 1) jumps are therefore reported as unobservable rather than guessed.

    Bbox cue: when legs are hidden (net, another player) the box bottom rises but the box gets
    shorter; a real jump lifts the whole body. Windows where the box height (box_h, px) drops
    below `min_box_ratio` of its value just before take-off or just after landing are rejected
    (occlusion, or a crouch / forward bend that moves the box bottom). Body-posture changes also
    jitter the foot point in proportion to the person's image size, so the noise model uses
    sqrt(foot_sigma_px^2 + (posture_frac * box_h)^2) pixels.

    Whole-body cue: in a real jump the head rises with the feet. With top_uv (box-top pixels),
    the rise of the box top along the image vertical must be at least `min_top_ratio` of the rise
    of the foot point; a crouch (top drops) or legs hidden behind the net (top stays) fail it.

    uv: (T,2) foot pixels (NaN = missing); Hinv, P: per-frame (T,3,3)/(T,3,4).
    Returns (airborne mask, ground_xy with airborne frames interpolated, z series, jumps)."""
    T = len(uv)
    ph = np.c_[uv, np.ones(T)]
    q = np.einsum("tij,tj->ti", Hinv, ph)
    xy = q[:, :2] / q[:, 2:3]
    valid = np.isfinite(xy).all(1)
    airborne = np.zeros(T, bool)
    ground = xy.copy()
    z_disp = np.full(T, np.nan)
    if valid.sum() < 8:
        return airborne, ground, z_disp, []
    base = rolling_nanmedian(xy, max(5, round(1.5 * fps)))
    okb = valid & np.isfinite(base).all(1)
    z_disp[okb] = heights(P[okb], base[okb], uv[okb])[0]
    z_disp -= np.nanmedian(z_disp)
    pre = rolling_nanmedian(xy, 3)                      # robust take-off / landing anchors
    Cc = -np.einsum("tij,tj->ti", np.linalg.inv(P[:, :, :3]), P[:, :, 3])      # camera centres (T,3)
    if box_h is not None:
        k = max(3, int(round(0.2 * fps)))
        bp = np.pad(np.asarray(box_h, float), (k, k), constant_values=np.nan)
        win = sliding_window_view(bp, k + 1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            box_before = np.nanmedian(win[:T], 1)        # frames t-k..t
            box_after = np.nanmedian(win[k:k + T], 1)    # frames t..t+k
    if top_uv is not None:
        uv_s, top_s = rolling_nanmedian(uv, 3), rolling_nanmedian(top_uv, 3)
    cands = []
    for n in range(max(3, int(np.floor(min_dur * fps))), int(np.ceil(max_dur * fps)) + 1):
        a = np.arange(1, T - n - 1)                     # a = last ground frame, a+n = first ground frame
        if len(a) == 0:
            break
        tau = np.arange(1, n) / n
        m = 4 * tau * (1 - tau)
        idx = a[:, None] + np.arange(1, n)[None]        # airborne frames
        g0, g1 = pre[a], pre[a + n]
        gl = g0[:, None] * (1 - tau)[None, :, None] + g1[:, None] * tau[None, :, None]
        v = valid[idx] & np.isfinite(gl).all(2)
        flat = v.ravel()
        zz = np.zeros(idx.size); pm = np.ones(idx.size)
        if not flat.any():
            continue
        zz[flat], pm[flat], _ = heights(P[idx.ravel()[flat]], gl.reshape(-1, 2)[flat], uv[idx.ravel()[flat]])
        zz = zz.reshape(idx.shape); pm = pm.reshape(idx.shape)
        mm = m[None] * v
        den = (mm ** 2).sum(1) + 1e-9
        h = (zz * mm).sum(1) / den
        nv = v.sum(1)
        res = np.sqrt((((zz - h[:, None] * m[None]) * v) ** 2).sum(1) / np.maximum(nv - 1, 1))
        Tf = n / fps
        hb = G * Tf ** 2 / 8
        pxm = (pm * v).sum(1) / np.maximum(nv, 1)
        if top_uv is not None:
            gi = gl.reshape(-1, 2)
            Pi = P[idx.ravel()]
            q0 = np.einsum("tij,tj->ti", Pi, np.c_[gi, np.zeros(len(gi)), np.ones(len(gi))])
            q1 = np.einsum("tij,tj->ti", Pi, np.c_[gi, np.ones(len(gi)), np.ones(len(gi))])
            up = q1[:, :2] / q1[:, 2:3] - q0[:, :2] / q0[:, 2:3]
            up = (up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-12)).reshape(*idx.shape, 2)
            def rise(sig):
                lin = sig[a][:, None] * (1 - tau)[None, :, None] + sig[a + n][:, None] * tau[None, :, None]
                d = np.nan_to_num(((sig[idx] - lin) * up).sum(2))
                vv = v & np.isfinite(sig[idx]).all(2) & np.isfinite(lin).all(2)
                return (d * m[None] * vv).sum(1) / ((m[None] * vv) ** 2).sum(1).clip(1e-9), vv.sum(1)
            rb, _ = rise(uv_s)
            rt, nt = rise(top_s)
            top_ok = (nt < 3) | (rt >= min_top_ratio * rb)
        else:
            top_ok = np.ones(len(a), bool)
        if box_h is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = np.nanmax(np.stack([box_before[a], box_after[a + n]]), 0)
                inair = np.nanmedian(np.where(v, box_h[idx], np.nan), 1)
            box_ok = ~(inair < min_box_ratio * ref)             # NaN -> no evidence -> keep
            noise_px = np.sqrt(foot_sigma_px ** 2 + (posture_frac * np.nan_to_num(ref, nan=0.0)) ** 2)
        else:
            box_ok = np.ones(len(a), bool)
            noise_px = np.full(len(a), float(foot_sigma_px))
        noise = noise_px / np.maximum(pxm, 1e-6)
        mid = 0.5 * (g0 + g1)
        kobs = np.linalg.norm(mid - Cc[a, :2], axis=1) / np.maximum(Cc[a, 2], 1e-3)
        a_eq = 8 * h * kobs / Tf ** 2
        good = (nv >= max(3, 0.6 * (n - 1))) & (h >= min_height) & (res <= max_resid_frac * h + 0.25 * noise) \
            & (h >= ballistic_tol[0] * hb) & (h <= ballistic_tol[1] * hb) & (h > 4.0 * noise / np.sqrt(np.maximum(nv, 1) / 2)) & (a_eq >= min_equiv_accel) & box_ok & top_ok
        for i in np.nonzero(good)[0]:
            cands.append((h[i] / (res[i] + 0.25 * noise[i] + 0.01) * (1 - abs(np.log(h[i] / hb)) / 2), int(a[i]), int(a[i] + n),
                          float(h[i]), float(res[i]), float(hb), float(pxm[i]), int(nv[i]), float(kobs[i]), float(noise_px[i])))
    cands.sort(key=lambda c: -c[0])
    jumps = []
    for sc, a, b, h, res, hb, pxm, nv, kob, npx in cands:
        if airborne[max(a - 1, 0):b + 2].any():
            continue
        airborne[a + 1:b] = True
        w = (np.arange(a, b + 1) - a) / (b - a)
        ground[a:b + 1] = pre[a][None] * (1 - w[:, None]) + pre[b][None] * w[:, None]
        sigma = np.sqrt((res / np.sqrt(max(nv / 3.0, 1.0))) ** 2 + (npx / max(pxm, 1e-6)) ** 2)
        jumps.append({"f0": a, "f1": b, "f_peak": (a + b) // 2, "height_m": h, "sigma_m": float(sigma),
                      "height_ballistic_m": hb, "flight_s": (b - a) / fps, "px_per_m": pxm, "observability_k": kob,
                      "xy": ground[(a + b) // 2].tolist()})
    jumps.sort(key=lambda j: j["f0"])
    return airborne, ground, z_disp, jumps

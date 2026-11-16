"""Per-player statistics with Monte-Carlo propagation of homography uncertainty."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .camera import calibrate_focal, pose_from_h
from .jumps import detect_jumps, heights
from .tracking import Track, dense, kalman_rts


@dataclass
class AnalysisConfig:
    sigma_a: float = 4.0          # m/s^2, process noise of the constant-velocity model
    sigma_m: float = 0.15         # m, measurement noise of projected foot points
    speed_deadband: float = 0.10  # m/s, below this a player is "standing" (jitter not counted)
    min_jump_m: float = 0.18
    jump_snr: float = 2.0         # keep a jump only if height >= jump_snr * sigma
    foot_bias_px: float = 3.0     # systematic foot-localisation error sampled per track in MC
    focal_rel_sigma: float = 0.08
    smoothing_bias_frac: float = 0.015   # path-length bias of the smoother seen in validation, added to bands
    heat_cell: float = 0.5


@dataclass
class PlayerResult:
    label: str
    team: int
    tid: int
    f0: int
    f1: int
    xy: np.ndarray                # (T,2) smoothed ground position, metres
    speed: np.ndarray             # (T,) m/s
    airborne: np.ndarray          # (T,) bool
    z: np.ndarray                 # (T,) raw height signal, metres
    observed: np.ndarray          # (T,) bool
    jumps: list
    stats: dict = field(default_factory=dict)
    heat: np.ndarray | None = None
    boxes: dict = field(default_factory=dict)


def cameras_from_h(H, size, f=None):
    """Per-frame projection matrices (T,3,4) with a single clip-level focal length."""
    if f is None:
        est = [calibrate_focal(H[i], size) for i in np.linspace(0, len(H) - 1, min(len(H), 9)).astype(int)]
        f = float(np.median([e[0] for e in est]))
        obs = float(np.median([e[2] for e in est]))
    else:
        obs = np.nan
    P = np.stack([pose_from_h(h, f, size).P for h in H])
    return P, f, obs


def _path_stats(xy, vel, dt, tpl, deadband):
    sp = np.linalg.norm(vel, axis=-1)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    moving = (0.5 * (sp[1:] + sp[:-1])) > deadband
    dist = (step * moving).sum(0)
    dur = (len(xy) - 1) * dt
    front = (np.abs(xy[..., 0] - tpl.net_x) <= tpl.front_zone).mean(0)
    return {"distance_m": dist, "top_speed_ms": sp.max(0), "avg_speed_ms": dist / max(dur, 1e-6),
            "front_row_frac": front}


def analyze(tracks, H, Hs, fps, tpl, size, cfg: AnalysisConfig | None = None, seed=0):
    """tracks: list[Track] (already filtered / team-labelled). H: (T,3,3), Hs: (T,K,3,3)."""
    cfg = cfg or AnalysisConfig()
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    K = Hs.shape[1]
    Hinv = np.linalg.inv(H)
    Hsinv = np.linalg.inv(Hs)
    P, f, f_obs = cameras_from_h(H, size)
    results = []
    for tr in tracks:
        f0, f1, uv = dense(tr)
        sl = slice(f0, f1 + 1)
        bh = np.full(len(uv), np.nan)
        bh[tr.frames - f0] = tr.boxes[:, 3] - tr.boxes[:, 1]
        air, ground, z, jumps = detect_jumps(uv, Hinv[sl], P[sl], fps, min_height=cfg.min_jump_m, box_h=bh)
        meas = ground.copy()
        obs = np.isfinite(uv).all(1)
        pos, vel = kalman_rts(meas, dt, cfg.sigma_a, cfg.sigma_m)
        base = _path_stats(pos, vel, dt, tpl, cfg.speed_deadband)
        # ---- Monte Carlo: bootstrap homographies (temporally coherent) + per-track foot bias
        bias = rng.normal(0, cfg.foot_bias_px, (K, 2))
        uvk = uv[:, None, :] + bias[None]                                  # (T,K,2)
        ph = np.concatenate([uvk, np.ones((*uvk.shape[:2], 1))], -1)
        q = np.einsum("tkij,tkj->tki", Hsinv[sl], ph)
        xyk = q[..., :2] / q[..., 2:3]
        # airborne frames: interpolate between take-off and landing in every replica
        for j in jumps:
            a, b = j["f0"], j["f1"]
            w = np.linspace(0, 1, b - a + 1)[:, None, None]
            ga = np.nanmedian(xyk[max(a - 3, 0):a + 1], 0); gb = np.nanmedian(xyk[b:b + 4], 0)
            xyk[a:b + 1] = ga[None] * (1 - w) + gb[None] * w
        # tracker jitter: resample measurement noise at the level observed around the smoothed path
        dev = meas - pos
        sig_meas = 1.4826 * np.nanmedian(np.abs(dev - np.nanmedian(dev, 0)), 0) if np.isfinite(dev).any() else np.zeros(2)
        xyk = xyk + rng.normal(0, 1, xyk.shape) * np.nan_to_num(sig_meas)[None, None]
        pk, vk = kalman_rts(xyk.reshape(len(uv), -1), dt, cfg.sigma_a, cfg.sigma_m)
        pk, vk = pk.reshape(len(uv), K, 2), vk.reshape(len(uv), K, 2)
        mc = _path_stats(pk, vk, dt, tpl, cfg.speed_deadband)
        stats = {"duration_s": (f1 - f0) * dt, "observed_frac": float(obs.mean())}
        for key in base:
            # replicas carry the measurement noise twice (observed + resampled), which inflates
            # path length; use their spread around their own median, centred on the estimate
            lo, mid, hi = np.percentile(mc[key], [5, 50, 95])
            v = float(base[key])
            lo, hi = v + (lo - mid), v + (hi - mid)
            if key in ("distance_m", "avg_speed_ms"):
                lo, hi = lo - cfg.smoothing_bias_frac * v, hi + cfg.smoothing_bias_frac * v
            stats[key] = v
            stats[key + "_lo"], stats[key + "_hi"] = float(max(min(lo, v), 0.0)), float(max(hi, v))
        # ---- jump heights under each replica (+ focal uncertainty)
        for j in jumps:
            a, b = j["f0"], j["f1"]
            idx = np.arange(a, b + 1)
            v = np.isfinite(uv[idx]).all(1)
            hk = []
            for k in range(K):
                fk = f * float(np.exp(rng.normal(0, cfg.focal_rel_sigma)))
                Pk = np.stack([pose_from_h(Hs[f0 + i, k], fk, size).P for i in idx[v]])
                w = ((idx[v] - a) / max(b - a, 1))[:, None]
                ga = np.nanmedian(xyk[max(a - 3, 0):a + 1, k], 0); gb = np.nanmedian(xyk[b:b + 4, k], 0)
                zz, _, _ = heights(Pk, ga[None] * (1 - w) + gb[None] * w, uv[idx[v]] + bias[k])
                tau = ((idx[v] - a) / max(b - a, 1)); m = 4 * tau * (1 - tau)
                hk.append(float((zz * m).sum() / ((m ** 2).sum() + 1e-9)))
            j["sigma_m"] = float(np.hypot(j["sigma_m"], np.std(hk)))
            j["f0"] += f0; j["f1"] += f0; j["f_peak"] += f0
            j["t_s"] = j["f_peak"] * dt
        keep = [j for j in jumps if j["height_m"] >= cfg.jump_snr * j["sigma_m"]]
        for j in jumps:
            if j not in keep:
                air[j["f0"] - f0:j["f1"] - f0 + 1] = False
        jumps = keep
        hs = np.array([j["height_m"] for j in jumps]); ss = np.array([j["sigma_m"] for j in jumps])
        stats["n_jumps"] = len(jumps)
        stats["max_jump_cm"] = float(hs.max() * 100) if len(hs) else 0.0
        stats["max_jump_sigma_cm"] = float(ss[np.argmax(hs)] * 100) if len(hs) else 0.0
        stats["avg_jump_cm"] = float(hs.mean() * 100) if len(hs) else 0.0
        stats["avg_jump_sigma_cm"] = float(np.sqrt((ss ** 2).sum()) / len(ss) * 100) if len(hs) else 0.0
        stats["back_row_frac"] = 1 - stats["front_row_frac"]
        nx, ny = int(np.ceil(tpl.length / cfg.heat_cell)), int(np.ceil(tpl.width / cfg.heat_cell))
        heat, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[nx, ny], range=[[0, tpl.length], [0, tpl.width]])
        res = PlayerResult(tr.label or str(tr.tid), tr.team, tr.tid, f0, f1, pos, np.linalg.norm(vel, axis=1),
                           air, z, obs, jumps, stats, heat * dt)
        res.boxes = {int(fr): bx for fr, bx in zip(tr.frames, tr.boxes)}
        results.append(res)
    return results, {"focal_px": f, "focal_observability": f_obs}

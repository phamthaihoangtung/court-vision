"""Validation harness on synthetic scenes with known ground truth."""
from __future__ import annotations
import json
import numpy as np
from .court import CourtRegistrar
from .geometry import apply_h
from .lens import estimate_plumbline
from .stats import AnalysisConfig, analyze
from .synth import random_camera, render_scene, simulate_detections, simulate_players
from .templates import get_template

G = 9.81


def court_error(H_est, cam, tpl, n=9):
    """Reprojection error of a grid of court points: (rms px, rms metres on the ground)."""
    gx, gy = np.meshgrid(np.linspace(0, tpl.length, n), np.linspace(0, tpl.width, n))
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    best = (np.inf, np.inf)
    L, W = tpl.length, tpl.width
    syms = [np.eye(3), np.array([[-1, 0, L], [0, -1, W], [0, 0, 1.0]]), np.array([[1, 0, 0], [0, -1, W], [0, 0, 1.0]]),
            np.array([[-1, 0, L], [0, 1, 0], [0, 0, 1.0]])]
    for S in syms:                                          # template symmetries are equivalent
        p_est = apply_h(H_est @ S, g); p_gt = apply_h(cam.H, g)
        e_px = np.sqrt(((p_est - p_gt) ** 2).sum(1).mean())
        back = apply_h(np.linalg.inv(H_est @ S), p_gt)
        e_m = np.sqrt(((back - g) ** 2).sum(1).mean())
        if e_px < best[0]:
            best = (e_px, e_m, S)
    return best


def run_trial(seed, court="indoor", lam=0.0, clutter=True, clutter_white=False, noise_px=1.5, elev=None,
              duration=20.0, fps=25.0, undistort=True, dropout=0.08, img_noise=6.0, occluders=True, acc_max=4.5):
    rng = np.random.default_rng(seed)
    tpl = get_template(court)
    cam = random_camera(tpl, rng, lam=lam, elev_deg=elev)
    gt = simulate_players(tpl, rng, duration=duration, fps=fps, acc_max=acc_max)
    pl = np.c_[gt["xy"][:, 0], gt["z"][:, 0]] if occluders else None
    img = render_scene(cam, tpl, rng, clutter=clutter, clutter_white=clutter_white, players_xyz=pl, noise=img_noise)
    out = {"seed": seed, "lam_true": lam, "elev_deg": float(np.degrees(np.arcsin(cam.C[2] / np.linalg.norm(cam.C - cam.target))))}
    lam_est = 0.0
    if undistort:
        lens = estimate_plumbline(img)
        if lens.significant:
            lam_est = lens.lam
            img = lens.undistort_image(img)
        out["lam_est"] = lam_est
    reg = CourtRegistrar(tpl).detect(img)
    if reg is None or not reg.ok:
        out["ok"] = False
        return out
    # ground truth homography lives in the *true* undistorted pixel space
    e_px, e_m, S = court_error(reg.H, cam, tpl)
    out.update(ok=bool(e_m < 0.5), reproj_px=float(e_px), reproj_m=float(e_m), coverage=reg.coverage)
    if not out["ok"]:
        return out
    out["boot_spread_m"] = float(np.sqrt(np.mean([((apply_h(np.linalg.inv(Hk), apply_h(reg.H, tpl.corners)) - tpl.corners) ** 2).sum(1).mean() for Hk in reg.H_samples])))
    # players: detections are generated in the pixel space the pipeline works in
    tracks = simulate_detections(cam, gt, rng, noise_px=noise_px, dropout=dropout, undistorted=True)
    T = gt["xy"].shape[1]
    Hseq = np.tile(reg.H @ S, (T, 1, 1)); Hs = np.tile((reg.H_samples @ S)[None], (T, 1, 1, 1))
    res, info = analyze(tracks, Hseq, Hs, fps, tpl, cam.size)
    d_err, v_err, pos_err, in_band = [], [], [], []
    tp = fn = fp = 0; h_err = []; h_z = []; hb_err = []
    for r, p in zip(res, range(len(res))):
        gxy = gt["xy"][p]
        d_true = np.linalg.norm(np.diff(gxy, axis=0), axis=1).sum()
        v_true = np.linalg.norm(np.diff(gxy, axis=0), axis=1) * fps
        k = int(0.4 * fps) | 1
        v_true_s = np.convolve(v_true, np.ones(k) / k, "same")
        d_err.append((r.stats["distance_m"] - d_true) / d_true * 100)
        v_err.append(r.stats["top_speed_ms"] - v_true_s.max())
        pos_err.append(np.sqrt(((r.xy - gxy[r.f0:r.f1 + 1]) ** 2).sum(1).mean()))
        in_band.append(r.stats["distance_m_lo"] <= d_true <= r.stats["distance_m_hi"])
        used = set()
        for j in gt["jumps"][p]:
            m = [i for i, e in enumerate(r.jumps) if abs(e["f_peak"] - j["f_peak"]) <= 0.4 * fps and i not in used]
            if m:
                e = r.jumps[m[0]]; used.add(m[0]); tp += 1
                h_err.append((e["height_m"] - j["height_m"]) * 100)
                hb_err.append((e["height_ballistic_m"] - j["height_m"]) * 100)
                h_z.append((e["height_m"] - j["height_m"]) / max(e["sigma_m"], 1e-3))
            else:
                fn += 1
        fp += len(r.jumps) - len(used)
    out.update(dist_err_pct=float(np.mean(np.abs(d_err))), dist_bias_pct=float(np.mean(d_err)),
               top_speed_err=float(np.mean(np.abs(v_err))), dist_in_band=float(np.mean(in_band)), pos_rmse_m=float(np.mean(pos_err)),
               jump_tp=tp, jump_fn=fn, jump_fp=fp, jump_err_cm=[float(x) for x in h_err],
               jump_ball_err_cm=[float(x) for x in hb_err], jump_z=[float(x) for x in h_z],
               f_true=cam.f, f_est=info["focal_px"])
    return out


def summarize(trials):
    ok = [t for t in trials if t.get("ok")]
    s = {"n": len(trials), "court_success_pct": 100.0 * len(ok) / max(len(trials), 1)}
    if not ok:
        return s
    def med(k): return float(np.median([t[k] for t in ok if k in t]))
    je = np.array(sum([t["jump_err_cm"] for t in ok], [])); jb = np.array(sum([t["jump_ball_err_cm"] for t in ok], []))
    jz = np.array(sum([t["jump_z"] for t in ok], []))
    tp = sum(t["jump_tp"] for t in ok); fn = sum(t["jump_fn"] for t in ok); fp = sum(t["jump_fp"] for t in ok)
    s.update(reproj_px_med=med("reproj_px"), reproj_cm_med=100 * med("reproj_m"),
             reproj_cm_p90=float(100 * np.percentile([t["reproj_m"] for t in ok], 90)),
             pos_rmse_cm=100 * med("pos_rmse_m"), dist_err_pct=med("dist_err_pct"), dist_bias_pct=med("dist_bias_pct"),
             top_speed_err_ms=med("top_speed_err"), dist_in_band_pct=100 * float(np.mean([t["dist_in_band"] for t in ok])),
             jump_recall_pct=100.0 * tp / max(tp + fn, 1), jump_precision_pct=100.0 * tp / max(tp + fp, 1),
             jump_mae_cm=float(np.mean(np.abs(je))) if len(je) else float("nan"),
             jump_bias_cm=float(np.mean(je)) if len(je) else float("nan"),
             jump_ballistic_mae_cm=float(np.mean(np.abs(jb))) if len(jb) else float("nan"),
             jump_sigma_calibration=float(np.std(jz)) if len(jz) else float("nan"),
             focal_err_pct=float(np.median([abs(t["f_est"] / t["f_true"] - 1) * 100 for t in ok])))
    if any("lam_est" in t for t in ok):
        s["lam_abs_err"] = float(np.median([abs(t.get("lam_est", 0) - t["lam_true"]) for t in ok]))
    return s

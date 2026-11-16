"""End-to-end pipeline: video -> registration + tracking (pass 1) -> analysis -> render (pass 2)."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from .court import CourtRegistrar, smooth_registrations
from .lens import estimate_plumbline
from .render import Minimap, compose, draw_frame, stat_card, stats_panel, team_card
from .stats import AnalysisConfig, analyze, cameras_from_h
from .templates import get_template
from .tracking import Track, assign_teams, filter_and_stitch


@dataclass
class Config:
    court: str = "indoor"
    device: str = "auto"
    model_size: str = "n"
    pose: bool = False
    imgsz: int = 1280
    frame_stride: int = 1
    max_frames: int = 0            # 0 = all
    undistort: str = "auto"        # auto | off | <lambda>
    line_color: str = "white"
    max_players: int = 12
    render: bool = True
    gif: bool = True
    cards: bool = True
    reuse_pass1: bool = False      # reuse cached registration + detections from a previous run
    out_width: int = 1920
    out_height: int = 1080


class Timer:
    def __init__(self):
        self.t = {}
        self.n = {}

    def add(self, k, dt, n=1):
        self.t[k] = self.t.get(k, 0.0) + dt
        self.n[k] = self.n.get(k, 0) + n

    def table(self):
        return {k: {"total_s": round(v, 3), "calls": self.n[k], "ms_per_frame": round(1000 * v / max(self.n[k], 1), 2),
                    "fps": round(self.n[k] / v, 2) if v > 0 else None} for k, v in self.t.items()}


class FFmpegWriter:
    """H.264 (libx264, yuv420p) through an ffmpeg pipe; falls back to OpenCV mp4v."""

    def __init__(self, path, size, fps, crf=20):
        self.path, self.size = path, size
        self.proc = self.cv = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                   "-r", f"{fps}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, img):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.cv.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def make_gif(mp4, gif, start=0.0, dur=6.0, width=720, fps=12, max_mb=10.0):
    """Palette-optimised GIF; frame rate / width are reduced until it fits max_mb."""
    if not shutil.which("ffmpeg"):
        return None
    for w, f, colors in [(width, fps, 128), (width, 10, 96), (width, 8, 64), (640, 8, 64), (560, 8, 48)]:
        vf = (f"fps={f},scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
              f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", mp4, "-vf", vf, gif], check=True)
        if os.path.getsize(gif) <= max_mb * 1e6:
            break
    return {"path": gif, "mb": os.path.getsize(gif) / 1e6, "width": w, "fps": f, "start_s": start, "dur_s": dur}


def _frames(path, stride, max_frames):
    cap = cv2.VideoCapture(path)
    i = n = 0
    while True:
        t0 = time.perf_counter()
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield fr, time.perf_counter() - t0
            n += 1
            if max_frames and n >= max_frames:
                break
        i += 1
    cap.release()


def run(video, out_dir, cfg: Config | None = None, progress=None, detector=None):
    cfg = cfg or Config()
    os.makedirs(out_dir, exist_ok=True)
    tpl = get_template(cfg.court)
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {video}")
    fps = src_fps / cfg.frame_stride
    tm = Timer()
    name = os.path.splitext(os.path.basename(video))[0]

    # ---- lens (once per clip: the camera is fixed)
    t0 = time.perf_counter()
    lens, maps = None, None
    lens_info = {"mode": cfg.undistort, "lambda": 0.0, "applied": False}
    if cfg.undistort == "auto":
        lm = estimate_plumbline(first)
        lens_info.update(lambda_est=lm.lam, rms_before_px=lm.rms_before, rms_after_px=lm.rms_after, chains=lm.n_chains, inliers=lm.n_inliers)
        if lm.significant:
            lens = lm
    elif cfg.undistort != "off":
        from .lens import LensModel
        lens = LensModel(float(cfg.undistort), (W, H))
    if lens is not None:
        maps = lens.maps()
        lens_info.update({"lambda": lens.lam, "applied": True})
    tm.add("lens_estimate", time.perf_counter() - t0)

    # ---- pass 1 (cached to <out>/pass1.pkl so analysis/render settings can be changed cheaply)
    import pickle
    cache = os.path.join(out_dir, "pass1.pkl")
    key = (os.path.abspath(video), cfg.court, cfg.model_size, cfg.pose, cfg.imgsz, cfg.frame_stride, cfg.max_frames,
           lens_info["applied"], cfg.line_color)
    regs, dets = [], []
    if cfg.reuse_pass1 and os.path.exists(cache):
        c = pickle.load(open(cache, "rb"))
        if c["key"] == key:
            regs, dets = c["regs"], c["dets"]
    if not regs:
        if detector is None:
            from .players import PlayerDetector
            detector = PlayerDetector(cfg.model_size, cfg.device, cfg.pose, cfg.imgsz)
        registrar = CourtRegistrar(tpl, line_color=cfg.line_color)
        total = min(cfg.max_frames or 10 ** 9, (n_src + cfg.frame_stride - 1) // cfg.frame_stride)
        for fi, (frame, tdec) in enumerate(_frames(video, cfg.frame_stride, cfg.max_frames)):
            tm.add("decode", tdec)
            if maps is not None:
                t0 = time.perf_counter(); frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR); tm.add("undistort", time.perf_counter() - t0)
            t0 = time.perf_counter(); regs.append(registrar.update(frame)); tm.add("court_registration", time.perf_counter() - t0)
            t0 = time.perf_counter(); dets.append(detector(frame)); tm.add("detect_track", time.perf_counter() - t0)
            if progress and fi % 10 == 0:
                progress(0.6 * fi / max(total, 1), f"pass 1: frame {fi}/{total}")
        pickle.dump({"key": key, "regs": regs, "dets": dets, "device": getattr(detector, "device", "n/a"),
                     "weights": getattr(detector, "weights", "n/a")}, open(cache, "wb"))
        dev_used, weights_used = getattr(detector, "device", "n/a"), getattr(detector, "weights", "n/a")
    else:
        dev_used, weights_used = c["device"], c["weights"]
    T = len(regs)
    if not any(r.ok for r in regs):
        raise RuntimeError("court not found in any frame - check --court / --line-color")

    # ---- analysis
    t0 = time.perf_counter()
    Hseq, Hs = smooth_registrations(regs, tpl, fps)
    Hinv = np.linalg.inv(Hseq)
    by = {}
    for fi, dl in enumerate(dets):
        for d in dl:
            by.setdefault(d["tid"], []).append((fi, d))
    raw = [Track(tid, np.array([f for f, _ in v]), np.array([d["foot"] for _, d in v]), np.array([d["box"] for _, d in v]),
                 np.array([d["color"] for _, d in v])) for tid, v in by.items()]
    tracks = filter_and_stitch(raw, Hinv, fps, tpl, max_players=cfg.max_players)
    team_info = assign_teams(tracks, Hinv, tpl) or {}
    results, cam_info = analyze(tracks, Hseq, Hs, fps, tpl, (W, H))
    results.sort(key=lambda r: r.label)
    tm.add("analysis", time.perf_counter() - t0, T)

    # ---- exports
    cov = np.array([r.coverage for r in regs]); okf = np.array([r.ok for r in regs])
    cor = np.stack([np.stack([cv2.perspectiveTransform(tpl.corners[None].astype(np.float64), r.H)[0] for r in regs])])[0]
    jitter = float(np.sqrt(((np.diff(cor, axis=0)) ** 2).sum(-1)).mean()) if T > 1 else 0.0
    from .geometry import apply_h
    cs = apply_h(Hseq, tpl.corners)
    jitter_s = float(np.sqrt((np.diff(cs, axis=0) ** 2).sum(-1)).mean()) if T > 1 else 0.0
    summary = {
        "clip": name, "source": {"width": W, "height": H, "fps": src_fps, "frames": n_src},
        "processed": {"frames": T, "fps": fps, "stride": cfg.frame_stride, "device": dev_used, "weights": weights_used},
        "court": {"template": tpl.name, "frames_ok_pct": float(100 * okf.mean()), "coverage_mean": float(cov[okf].mean()),
                  "redetections": int(sum(r.source == "detect" for r in regs)), "rms_px_mean": float(np.mean([r.rms_px for r in regs if r.ok])),
                  "corner_motion_px_per_frame_raw": jitter, "corner_motion_px_per_frame_smoothed": jitter_s,
                  "bootstrap_corner_sigma_px": float(apply_h(Hs.reshape(-1, 3, 3), tpl.corners).reshape(T, -1, 4, 2).std(1).mean())},
        "camera": cam_info, "lens": lens_info, "teams": team_info,
        "tracks": {"raw_ids": len(raw), "players": len(results), "stitched_fragments": int(sum(t.meta.get("stitched", 1) - 1 for t in tracks))},
        "players": [{"label": r.label, "team": "AB"[r.team % 2], **r.stats, "jumps": r.jumps} for r in results],
    }
    import pandas as pd
    pd.DataFrame([{"player": r.label, "team": "AB"[r.team % 2], **r.stats} for r in results]).to_csv(os.path.join(out_dir, "player_stats.csv"), index=False)
    rows = []
    for r in results:
        for i in range(len(r.xy)):
            rows.append((r.f0 + i, (r.f0 + i) / fps, r.label, r.xy[i, 0], r.xy[i, 1], r.speed[i], int(r.airborne[i]), int(r.observed[i])))
    pd.DataFrame(rows, columns=["frame", "t_s", "player", "x_m", "y_m", "speed_ms", "airborne", "observed"]).round(3).to_csv(os.path.join(out_dir, "positions.csv"), index=False)
    pd.DataFrame([{"player": r.label, **{k: v for k, v in j.items() if k != "xy"}} for r in results for j in r.jumps]).to_csv(os.path.join(out_dir, "jumps.csv"), index=False)

    if cfg.cards:
        t0 = time.perf_counter()
        os.makedirs(os.path.join(out_dir, "cards"), exist_ok=True)
        for r in results:
            stat_card(r, tpl, os.path.join(out_dir, "cards", f"player_{r.label}.png"), name)
        team_card(results, tpl, os.path.join(out_dir, "cards", "summary.png"), name)
        tm.add("stat_cards", time.perf_counter() - t0, max(len(results), 1))

    # ---- pass 2: render
    if cfg.render:
        mp4 = os.path.join(out_dir, f"{name}_annotated.mp4")
        mini = Minimap(tpl, width=480, height=cfg.out_height - 80)
        wr = FFmpegWriter(mp4, (cfg.out_width, cfg.out_height), fps)
        vw = cfg.out_width - 480
        vh = int(round(vw * H / W)); ph = cfg.out_height - vh
        # busiest moment (most players + a jump) for the still / GIF
        act = np.zeros(T)
        for r in results:
            act[r.f0:r.f1 + 1] += 1 + 0.2 * r.speed
            for j in r.jumps:
                act[max(j["f0"] - int(fps), 0):j["f1"] + int(fps)] += 3
        k = max(int(6 * fps), 1)
        win = np.convolve(act, np.ones(k), "valid") if T > k else np.array([act.sum()])
        g0 = int(np.argmax(win)); best_f = min(g0 + int(np.argmax(act[g0:g0 + k])), T - 1)
        for fi, (frame, _) in enumerate(_frames(video, cfg.frame_stride, cfg.max_frames)):
            if fi >= T:
                break
            t0 = time.perf_counter()
            if maps is not None:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            draw_frame(frame, Hseq[fi], tpl, results, fi, fps)
            canvas = compose(frame, mini.draw(results, fi, fps), stats_panel(results, fi, fps, vw, ph),
                             "court-vision", f"{tpl.name} {tpl.length:.0f}x{tpl.width:.0f} m  |  t={fi / fps:5.1f}s  |  court fit {regs[fi].coverage * 100:.0f}%",
                             (cfg.out_width, cfg.out_height))
            tm.add("render", time.perf_counter() - t0)
            t0 = time.perf_counter(); wr.write(canvas); tm.add("encode", time.perf_counter() - t0)
            if fi == best_f:
                cv2.imwrite(os.path.join(out_dir, "best_frame.png"), canvas)
            if progress and fi % 10 == 0:
                progress(0.6 + 0.4 * fi / T, f"pass 2: frame {fi}/{T}")
        wr.close()
        summary["video"] = mp4
        if cfg.gif:
            summary["gif"] = make_gif(mp4, os.path.join(out_dir, f"{name}_demo.gif"), start=g0 / fps, dur=min(6.5, T / fps))
    summary["timing"] = tm.table()
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    return summary, results

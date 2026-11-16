"""Rendering: annotated frame + live top-down minimap + stats panel; stat cards (PNG)."""
from __future__ import annotations
import cv2
import numpy as np
from .geometry import apply_h

TEAM_BGR = [(60, 170, 255), (255, 170, 60)]          # team A orange, team B blue (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 22, 20)


def _text(img, s, org, scale=0.5, col=(235, 235, 235), th=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, col, th, cv2.LINE_AA)


class Minimap:
    def __init__(self, tpl, width=480, height=820, margin_m=2.0):
        self.tpl = tpl
        self.s = min(width / (tpl.width + 2 * margin_m), height / (tpl.length + 2 * margin_m))
        self.w, self.h = width, height
        self.ox = (width - tpl.width * self.s) / 2
        self.oy = (height - tpl.length * self.s) / 2
        base = np.full((height, width, 3), BG, np.uint8)
        a, b = self.to_px(np.array([[0, 0], [tpl.length, tpl.width]]))
        cv2.rectangle(base, tuple(a.astype(int)), tuple(b.astype(int)), (58, 84, 150), -1)
        for x0, y0, x1, y1 in tpl.segments:
            p = self.to_px(np.array([[x0, y0], [x1, y1]])).astype(int)
            cv2.line(base, tuple(p[0]), tuple(p[1]), (240, 240, 240), 2, cv2.LINE_AA)
        if tpl.has_net:
            p = self.to_px(np.array([[tpl.net_x, -0.8], [tpl.net_x, tpl.width + 0.8]])).astype(int)
            cv2.line(base, tuple(p[0]), tuple(p[1]), (255, 255, 255), 4, cv2.LINE_AA)
        self.base = base

    def to_px(self, xy):
        """Court x runs down the minimap, y to the right (matches the registration convention)."""
        xy = np.atleast_2d(xy)
        return np.stack([self.ox + xy[:, 1] * self.s, self.oy + xy[:, 0] * self.s], 1)

    def draw(self, results, fi, fps, trail_s=2.0):
        img = self.base.copy()
        n = int(trail_s * fps)
        for r in results:
            if not (r.f0 <= fi <= r.f1):
                continue
            i = fi - r.f0
            col = TEAM_BGR[r.team % 2]
            tr = self.to_px(r.xy[max(i - n, 0):i + 1]).astype(np.int32)
            if len(tr) > 1:
                cv2.polylines(img, [tr], False, tuple(int(c * 0.7) for c in col), 2, cv2.LINE_AA)
            p = tuple(tr[-1])
            if r.airborne[i]:
                cv2.circle(img, p, 17, (80, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, p, 11, col, -1, cv2.LINE_AA)
            cv2.circle(img, p, 11, (20, 20, 20), 1, cv2.LINE_AA)
            _text(img, r.label, (p[0] - 9, p[1] + 4), 0.38, (15, 15, 15), 1)
        return img


def draw_frame(frame, H, tpl, results, fi, fps, scale=1.0):
    """Court overlay + player boxes/labels/jump tags on the source frame (in place)."""
    lw = max(2, int(round(frame.shape[1] / 960)))
    ov = frame.copy()
    for x0, y0, x1, y1 in tpl.segments:
        p = apply_h(H, np.array([[x0, y0], [x1, y1]]))
        if np.isfinite(p).all() and np.abs(p).max() < 1e5:
            cv2.line(ov, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (255, 230, 0), lw, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    fs = frame.shape[1] / 1920 * 0.62
    for r in results:
        if not (r.f0 <= fi <= r.f1):
            continue
        i = fi - r.f0
        col = TEAM_BGR[r.team % 2]
        g = apply_h(H, r.xy[i:i + 1])[0]
        cv2.ellipse(frame, (int(g[0]), int(g[1])), (int(16 * lw / 2), int(7 * lw / 2)), 0, 0, 360, col, lw, cv2.LINE_AA)
        b = r.boxes.get(fi)
        if b is not None:
            x0, y0, x1, y1 = [int(v) for v in b]
            cv2.rectangle(frame, (x0, y0), (x1, y1), col, lw, cv2.LINE_AA)
            tag = f"{r.label} {r.speed[i] * 3.6:.0f} km/h"
            (tw, th_), _ = cv2.getTextSize(tag, FONT, fs, 2)
            cv2.rectangle(frame, (x0, y0 - th_ - 8), (x0 + tw + 6, y0), col, -1)
            _text(frame, tag, (x0 + 3, y0 - 5), fs, (15, 15, 15), 2)
            for j in r.jumps:
                if j["f0"] <= fi <= j["f1"] + int(0.8 * fps):
                    tag = f"JUMP {j['height_m'] * 100:.0f}+-{j['sigma_m'] * 100:.0f} cm"
                    _text(frame, tag, (x0 - 10, y0 - th_ - 16), fs * 1.1, (0, 0, 0), 5)
                    _text(frame, tag, (x0 - 10, y0 - th_ - 16), fs * 1.1, (80, 255, 255), 2)
    return frame


def stats_panel(results, fi, fps, w, h):
    """Live per-player table: distance so far, current speed, jumps so far."""
    img = np.full((h, w, 3), BG, np.uint8)
    dt = 1.0 / fps
    colw = w // 2
    for tm in (0, 1):
        x = tm * colw + 24
        _text(img, f"TEAM {'AB'[tm]}", (x, 30), 0.7, TEAM_BGR[tm], 2)
        _text(img, "player   dist m   km/h   top   jumps  best cm", (x, 58), 0.48, (150, 150, 150), 1)
        rows = [r for r in results if r.team == tm]
        for k, r in enumerate(rows[: max(1, (h - 70) // 30)]):
            i = int(np.clip(fi - r.f0, 0, len(r.xy) - 1))
            active = r.f0 <= fi <= r.f1
            step = np.linalg.norm(np.diff(r.xy[:i + 1], axis=0), axis=1)
            mov = r.speed[1:i + 1] > 0.10
            dist = float((step * mov).sum()) if i > 0 else 0.0
            js = [j for j in r.jumps if j["f_peak"] <= fi]
            best = max([j["height_m"] for j in js], default=0) * 100
            top = r.speed[:i + 1].max() * 3.6
            col = (235, 235, 235) if active else (110, 110, 110)
            y = 88 + 30 * k
            cv2.circle(img, (x + 8, y - 6), 7, TEAM_BGR[tm], -1, cv2.LINE_AA)
            _text(img, f"  {r.label:<4} {dist:7.1f}  {r.speed[i] * 3.6 if active else 0:5.1f}  {top:5.1f}   {len(js):3d}   {best:5.0f}", (x + 10, y), 0.55, col, 1)
    return img


def compose(frame, mini, panel, title, sub, out_size=(1920, 1080)):
    W, Hh = out_size
    canvas = np.full((Hh, W, 3), BG, np.uint8)
    vw, vh = W - mini.shape[1], int(round((W - mini.shape[1]) * frame.shape[0] / frame.shape[1]))
    canvas[:vh, :vw] = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA)
    ph = Hh - vh
    canvas[vh:, :vw] = cv2.resize(panel, (vw, ph)) if panel.shape[:2] != (ph, vw) else panel
    _text(canvas, title, (vw + 20, 38), 0.8, (255, 255, 255), 2)
    _text(canvas, sub, (vw + 20, 64), 0.45, (160, 160, 160), 1)
    mh = min(mini.shape[0], Hh - 80)
    canvas[80:80 + mh, vw:] = mini[:mh]
    return canvas


def stat_card(r, tpl, path, clip_name=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    s = r.stats
    col = np.array(TEAM_BGR[r.team % 2][::-1]) / 255
    fig = plt.figure(figsize=(8, 4.5), dpi=150, facecolor="#141618")
    ax = fig.add_axes([0.03, 0.08, 0.36, 0.84]); ax.set_facecolor("#964f3a")
    heat = gaussian_filter(r.heat, 1.0)
    ax.imshow(np.ma.masked_less(heat, heat.max() * 0.03), extent=[0, tpl.width, tpl.length, 0], cmap="magma", alpha=0.9, aspect="equal")
    for x0, y0, x1, y1 in tpl.segments:
        ax.plot([y0, y1], [x0, x1], color="white", lw=1.5)
    ax.plot([-0.5, tpl.width + 0.5], [tpl.net_x] * 2, color="white", lw=3)
    for j in r.jumps:
        ax.plot(j["xy"][1], j["xy"][0], "^", color="#50ffff", ms=7, mec="k")
    ax.set_xlim(-1, tpl.width + 1); ax.set_ylim(tpl.length + 1, -1); ax.axis("off")
    fig.text(0.43, 0.86, f"PLAYER {r.label}", color=col, fontsize=22, fontweight="bold")
    fig.text(0.43, 0.80, f"Team {'AB'[r.team % 2]}  |  {s['duration_s']:.1f} s tracked  |  {clip_name}", color="#aaaaaa", fontsize=8)
    rows = [("Distance", f"{s['distance_m']:.1f} m", f"[{s['distance_m_lo']:.1f}, {s['distance_m_hi']:.1f}]"),
            ("Top speed", f"{s['top_speed_ms'] * 3.6:.1f} km/h", f"[{s['top_speed_ms_lo'] * 3.6:.1f}, {s['top_speed_ms_hi'] * 3.6:.1f}]"),
            ("Avg speed", f"{s['avg_speed_ms'] * 3.6:.1f} km/h", f"[{s['avg_speed_ms_lo'] * 3.6:.1f}, {s['avg_speed_ms_hi'] * 3.6:.1f}]"),
            ("Jumps", f"{s['n_jumps']}", ""),
            ("Max jump", f"{s['max_jump_cm']:.0f} cm" if s["n_jumps"] else "-", f"+-{s['max_jump_sigma_cm']:.0f} cm" if s["n_jumps"] else ""),
            ("Avg jump", f"{s['avg_jump_cm']:.0f} cm" if s["n_jumps"] else "-", f"+-{s['avg_jump_sigma_cm']:.0f} cm" if s["n_jumps"] else ""),
            ("Front row", f"{s['front_row_frac'] * 100:.0f} %", f"[{s['front_row_frac_lo'] * 100:.0f}, {s['front_row_frac_hi'] * 100:.0f}]"),
            ("Back row", f"{s['back_row_frac'] * 100:.0f} %", "")]
    for k, (a, b, c) in enumerate(rows):
        y = 0.71 - 0.058 * k
        fig.text(0.43, y, a, color="#bbbbbb", fontsize=11)
        fig.text(0.60, y, b, color="white", fontsize=12, fontweight="bold")
        fig.text(0.78, y, c, color="#888888", fontsize=9)
    fig.text(0.78, 0.765, "90% band (MC)", color="#888888", fontsize=7)
    ax2 = fig.add_axes([0.45, 0.08, 0.5, 0.14]); ax2.set_facecolor("#141618")
    t = np.arange(len(r.speed)) / max(s["duration_s"] / max(len(r.speed) - 1, 1), 1e-9) ** -1
    ax2.fill_between(t, r.speed * 3.6, color=col, alpha=0.6, lw=0)
    for j in r.jumps:
        ax2.axvline((j["f_peak"] - r.f0) * (t[1] if len(t) > 1 else 0), color="#50ffff", lw=1)
    ax2.set_ylabel("km/h", color="#aaaaaa", fontsize=7); ax2.tick_params(colors="#aaaaaa", labelsize=6)
    for sp in ax2.spines.values():
        sp.set_color("#444444")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def team_card(results, tpl, path, clip_name="", meta=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9.6, 5.4), dpi=150, facecolor="#141618")
    ax = fig.add_axes([0.03, 0.06, 0.3, 0.86]); ax.set_facecolor("#964f3a")
    for x0, y0, x1, y1 in tpl.segments:
        ax.plot([y0, y1], [x0, x1], color="white", lw=1.5)
    ax.plot([-0.5, tpl.width + 0.5], [tpl.net_x] * 2, color="white", lw=3)
    for r in results:
        c = np.array(TEAM_BGR[r.team % 2][::-1]) / 255
        ax.plot(r.xy[:, 1], r.xy[:, 0], color=c, lw=0.8, alpha=0.8)
        ax.text(r.xy[-1, 1], r.xy[-1, 0], r.label, color="white", fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc=c, ec="k", lw=0.5))
    ax.set_xlim(-2, tpl.width + 2); ax.set_ylim(tpl.length + 2, -2); ax.axis("off")
    fig.text(0.36, 0.9, "MATCH SUMMARY", color="white", fontsize=18, fontweight="bold")
    fig.text(0.36, 0.855, f"{clip_name}  |  positions in metres from a single fixed camera, no training", color="#aaaaaa", fontsize=8)
    hdr = ["", "dist m", "90% band", "top km/h", "avg km/h", "jumps", "max cm", "front %"]
    xs = [0.36, 0.42, 0.49, 0.60, 0.68, 0.76, 0.82, 0.90]
    for x, hname in zip(xs, hdr):
        fig.text(x, 0.79, hname, color="#888888", fontsize=8)
    for k, r in enumerate(sorted(results, key=lambda r: r.label)):
        s = r.stats; y = 0.74 - 0.052 * k
        c = np.array(TEAM_BGR[r.team % 2][::-1]) / 255
        vals = [r.label, f"{s['distance_m']:.1f}", f"{s['distance_m_lo']:.1f}-{s['distance_m_hi']:.1f}", f"{s['top_speed_ms'] * 3.6:.1f}",
                f"{s['avg_speed_ms'] * 3.6:.1f}", f"{s['n_jumps']}", f"{s['max_jump_cm']:.0f}+-{s['max_jump_sigma_cm']:.0f}" if s["n_jumps"] else "-",
                f"{s['front_row_frac'] * 100:.0f}"]
        for x, v in zip(xs, vals):
            fig.text(x, y, v, color=c if x == xs[0] else "white", fontsize=9, fontweight="bold" if x == xs[0] else "normal")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)

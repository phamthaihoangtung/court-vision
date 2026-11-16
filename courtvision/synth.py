"""Synthetic scenes with known ground truth: camera (intrinsics, pose, radial distortion),
court + clutter lines from other sports, player trajectories with jumps, occlusions."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .lens import LensModel
from .templates import CourtTemplate
from .tracking import Track

G = 9.81


@dataclass
class SynthCamera:
    size: tuple
    f: float
    C: np.ndarray
    target: np.ndarray
    lam: float = 0.0

    def __post_init__(self):
        w, h = self.size
        self.K = np.array([[self.f, 0, w / 2], [0, self.f, h / 2], [0, 0, 1.0]])
        fwd = self.target - self.C
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])
        self.t = -self.R @ self.C
        self.lens = LensModel(self.lam, self.size)

    def project(self, xyz, distort=True):
        xyz = np.atleast_2d(np.asarray(xyz, float))
        q = (xyz @ self.R.T + self.t) @ self.K.T
        p = q[:, :2] / q[:, 2:3]
        return self.lens.distort_points(p) if (distort and self.lam != 0) else p

    @property
    def H(self):
        """Ground-plane homography court metres -> undistorted pixels."""
        H = self.K @ np.stack([self.R[:, 0], self.R[:, 1], self.t], 1)
        return H / H[2, 2]


def random_camera(tpl: CourtTemplate, rng, size=(1280, 720), lam=0.0, elev_deg=None, behind=None):
    """Camera on a tripod/balcony/ceiling mount looking at the court centre."""
    L, W = tpl.length, tpl.width
    elev = np.deg2rad(elev_deg if elev_deg is not None else rng.uniform(25, 60))
    az = rng.uniform(0, 2 * np.pi) if behind is None else (np.deg2rad(rng.normal(0, 12)) + (np.pi if behind else np.pi / 2))
    dist = rng.uniform(1.35, 1.7) * L
    C = np.array([L / 2 + dist * np.cos(elev) * np.cos(az), W / 2 + dist * np.cos(elev) * np.sin(az), dist * np.sin(elev)])
    target = np.array([L / 2 + rng.normal(0, 0.8), W / 2 + rng.normal(0, 0.5), 0.0])
    # focal so that the court fills ~70% of the frame
    cam = SynthCamera(size, size[0], C, target, 0.0)
    p = cam.project(np.c_[tpl.corners, np.zeros(4)])
    ext = max((p[:, 0].max() - p[:, 0].min()) / size[0], (p[:, 1].max() - p[:, 1].min()) / size[1])
    return SynthCamera(size, size[0] * rng.uniform(0.62, 0.78) / ext, C, target, lam)


def _polyline(img, cam, pts3, color, thick):
    for a, b in zip(pts3[:-1], pts3[1:]):
        n = max(2, int(np.linalg.norm(b - a) / 0.25) + 1)
        seg = a[None] + (b - a)[None] * np.linspace(0, 1, n)[:, None]
        p = cam.project(seg)
        cv2.polylines(img, [np.round(p * 8).astype(np.int32)], False, color, thick, cv2.LINE_AA, shift=3)


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0.0]])


def _arc(cx, cy, r, a0, a1, n=40):
    a = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a), np.zeros(n)]


def clutter_lines(tpl: CourtTemplate):
    """Basketball court around the volleyball court + two crosswise badminton courts."""
    cx, cy = tpl.length / 2, tpl.width / 2
    bb = []                                              # basketball 28 x 15
    x0, x1, y0, y1 = cx - 14, cx + 14, cy - 7.5, cy + 7.5
    bb += [_rect(x0, y0, x1, y1), np.array([[cx, y0, 0], [cx, y1, 0.0]]), _arc(cx, cy, 1.8, 0, 2 * np.pi)]
    for sx, e in ((1, x0), (-1, x1)):
        bb += [_rect(e, cy - 2.45, e + sx * 5.8, cy + 2.45), _arc(e + sx * 5.8, cy, 1.8, -np.pi / 2, np.pi / 2) * [1, 1, 0] if sx > 0
               else _arc(e + sx * 5.8, cy, 1.8, np.pi / 2, 3 * np.pi / 2),
               _arc(e + sx * 1.575, cy, 6.75, -1.35, 1.35) if sx > 0 else _arc(e + sx * 1.575, cy, 6.75, np.pi - 1.35, np.pi + 1.35)]
    bd = []                                              # badminton 13.4 x 6.1, rotated 90 deg
    for ox in (cx - 4.6, cx + 4.6):
        a0, a1, b0, b1 = ox - 3.05, ox + 3.05, cy - 6.7, cy + 6.7
        bd += [_rect(a0, b0, a1, b1), _rect(a0 + 0.46, b0, a1 - 0.46, b1), _rect(a0, b0 + 0.76, a1, b1 - 0.76),
               np.array([[a0, cy - 1.98, 0], [a1, cy - 1.98, 0.0]]), np.array([[a0, cy + 1.98, 0], [a1, cy + 1.98, 0.0]]),
               np.array([[ox, b0, 0], [ox, cy - 1.98, 0.0]]), np.array([[ox, cy + 1.98, 0], [ox, b1, 0.0]])]
    return bb, bd


def render_scene(cam: SynthCamera, tpl: CourtTemplate, rng, clutter=True, clutter_white=False,
                 players_xyz=None, noise=6.0, floor=None):
    """Render a multipurpose-hall frame. Volleyball lines are white; other sports use their own
    colours unless clutter_white=True (worst case: every line white)."""
    w, h = cam.size
    floor = np.array(floor if floor is not None else rng.choice([[70, 120, 170], [120, 110, 60], [60, 90, 150]]), float)
    img = np.empty((h, w, 3), np.uint8); img[:] = floor
    # boards / low-frequency shading
    shade = cv2.resize(rng.normal(0, 1, (6, 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img + 8 * shade[..., None], 0, 255).astype(np.uint8)
    px_per_m = np.linalg.norm(cam.project([[tpl.length / 2 + 0.5, tpl.width / 2, 0]]) - cam.project([[tpl.length / 2 - 0.5, tpl.width / 2, 0]]))
    thick = max(2, int(round(0.05 * px_per_m)) + 1)
    if clutter:
        bb, bd = clutter_lines(tpl)
        for poly in bb:
            _polyline(img, cam, poly, (255, 255, 255) if clutter_white else (40, 40, 40), thick)
        for poly in bd:
            _polyline(img, cam, poly, (255, 255, 255) if clutter_white else (60, 220, 240), max(2, thick - 1))
    for x0, y0, x1, y1 in tpl.segments:
        _polyline(img, cam, np.array([[x0, y0, 0], [x1, y1, 0.0]]), (255, 255, 255), thick)
    if players_xyz is not None:
        for i, (x, y, z) in enumerate(players_xyz):
            foot = cam.project([[x, y, z]])[0]; head = cam.project([[x, y, z + 1.8]])[0]
            hgt = np.linalg.norm(head - foot); wd = 0.3 * hgt
            mid = (foot + head) / 2
            col = (235, 235, 235) if i % 2 == 0 else (30, 30, 30)
            ang = np.degrees(np.arctan2(head[1] - foot[1], head[0] - foot[0])) + 90
            cv2.ellipse(img, ((float(mid[0]), float(mid[1])), (float(max(wd, 1)), float(max(hgt, 1))), float(ang)), col, -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def simulate_players(tpl: CourtTemplate, rng, n_per_team=6, duration=30.0, fps=25.0, jump_rate=0.12, acc_max=4.5):
    """Random-waypoint motion inside each half + ballistic jumps. Returns dict of ground truth.
    acc_max (m/s^2) bounds the horizontal acceleration of a running player."""
    T = int(duration * fps); dt = 1 / fps
    n = 2 * n_per_team
    xy = np.zeros((n, T, 2)); z = np.zeros((n, T)); jumps = [[] for _ in range(n)]
    for p in range(n):
        side = p % 2
        lo = np.array([0.3 if side == 0 else tpl.net_x + 0.5, 0.3])
        hi = np.array([tpl.net_x - 0.5 if side == 0 else tpl.length - 0.3, tpl.width - 0.3])
        pos = rng.uniform(lo, hi); vel = np.zeros(2); goal = rng.uniform(lo, hi); vmax = rng.uniform(1.5, 5.0)
        t_air = 0
        for t in range(T):
            if t_air > 0:                                   # ballistic: constant horizontal velocity
                t_air -= 1
            else:
                if np.linalg.norm(goal - pos) < 0.3 or rng.random() < 0.01:
                    goal = rng.uniform(lo, hi); vmax = rng.uniform(0.8, 5.0)
                    if rng.random() < 0.35:
                        goal = pos.copy()                   # stand still for a while
                d = goal - pos
                want = d / (np.linalg.norm(d) + 1e-9) * min(vmax, 3.0 * np.linalg.norm(d))
                acc = (want - vel) / 0.25
                acc = acc * min(1.0, acc_max / (np.linalg.norm(acc) + 1e-9))
                vel = vel + acc * dt
                if rng.random() < jump_rate * dt and t + fps < T:
                    hj = rng.uniform(0.2, 0.75); Tf = np.sqrt(8 * hj / G)
                    nfl = int(round(Tf * fps))
                    tau = (np.arange(1, nfl) * dt)
                    z[p, t + 1:t + nfl] = hj - 0.5 * G * (tau - Tf / 2) ** 2 + 0 * tau
                    z[p, t + 1:t + nfl] = np.clip(4 * hj * (tau / Tf) * (1 - tau / Tf), 0, None)
                    vel = vel * 0.3
                    jumps[p].append({"f_peak": t + nfl // 2, "height_m": float(z[p, t + 1:t + nfl].max()), "f0": t, "f1": t + nfl})
                    t_air = nfl
            pos = np.clip(pos + vel * dt, lo, hi)
            xy[p, t] = pos
    return {"xy": xy, "z": z, "jumps": jumps, "fps": fps}


def simulate_detections(cam: SynthCamera, gt, rng, noise_px=1.5, dropout=0.08, undistorted=True):
    """What a detector+tracker would output: noisy foot pixels with occlusion gaps."""
    xy, z = gt["xy"], gt["z"]
    n, T, _ = xy.shape
    tracks = []
    for p in range(n):
        xyz = np.c_[xy[p], z[p]]
        foot = cam.project(xyz, distort=not undistorted)
        head = cam.project(xyz + [0, 0, 1.8], distort=not undistorted)
        foot_n = foot + rng.normal(0, noise_px, foot.shape)
        keep = np.ones(T, bool)
        t = 0
        while t < T:                                        # occlusion bursts of 0.2-1.2 s
            if rng.random() < dropout / (0.6 * gt["fps"]):
                L = int(rng.uniform(0.2, 1.2) * gt["fps"]); keep[t:t + L] = False; t += L
            t += 1
        keep[0] = keep[-1] = True
        hgt = np.linalg.norm(head - foot, axis=1)
        boxes = np.c_[foot_n[:, 0] - 0.2 * hgt, foot_n[:, 1] - hgt, foot_n[:, 0] + 0.2 * hgt, foot_n[:, 1]]
        fr = np.nonzero(keep)[0]
        tracks.append(Track(p, fr, foot_n[fr], boxes[fr], None, team=p % 2, label=f"{'AB'[p % 2]}{p // 2 + 1}"))
    return tracks

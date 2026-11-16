"""Online analysis with a fixed delay: frames go in one at a time, results come out `lag_s` later.

Nothing here looks further ahead than the lag, so the same logic can run on a live camera feed.

    warm-up (first `warmup_s`)   buffer frames, then run the offline logic on that window once:
                                  focal length, 12 player slots (stitched tracks), teams, jersey colours.
    every frame                   causal Kalman per slot predicts the court position; detections are
                                  assigned to the slots being followed by Hungarian matching on court
                                  distance, jersey colour and ByteTrack id ownership. Ids nobody follows
                                  become tracklets; after `resolve_s` all live tracklets and all lost slots
                                  are matched jointly and the ripe ones are written back into their slot
                                  (the output delay leaves room to revise those frames).
    every `hop_s`                 frames that are now `lag_s` old are finalised: the court homography is
                                  smoothed, jumps whose landing has been seen are committed, and positions
                                  are smoothed by RTS over a window that reaches `lag_s` into the future.
                                  With lag_s >= ~1.6 s (Gaussian support of the homography smoother, and a
                                  0.9 s jump plus landing context) this matches the offline result.

Per-player stats are accumulated from finalised frames only, so at any time they describe the match up
to `now - lag_s`. Distance/speed carry no Monte Carlo bands online; jump heights keep their sigma.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from .camera import calibrate_focal, pose_from_h
from .court import smooth_registrations
from .geometry import apply_h
from .jumps import detect_jumps
from .stats import AnalysisConfig, PlayerResult, band, jump_sigma, replica_positions
from .tracking import Track, assign_teams, court_positions, filter_and_stitch, kalman_rts


@dataclass
class OnlineConfig:
    lag_s: float = 2.0            # output delay
    hop_s: float = 0.2            # finalise frames in batches of this length
    burn_s: float = 2.0           # history in front of each smoothing window
    warmup_s: float = 3.0         # frames used to set up slots, teams, focal
    max_players: int = 12
    margin_m: float = 3.0         # detections this far outside the court lines are ignored
    gate_m: float = 1.0           # association gate for a slot seen in the previous frame
    lost_speed: float = 3.0       # gate grows by this (m/s) while a slot is lost ...
    max_gate_m: float = 4.0       # ... up to this
    colour_w: float = 0.02        # cost per Lab unit of jersey-colour difference
    kit_sep: float = 25.0         # team kits this far apart (Lab) are used to refuse wrong-team matches
    kit_frames: int = 5           # consecutive wrong-kit frames before a slot gives up its ByteTrack id
    own_s: float = 1.0            # a ByteTrack id stays owned by its slot for this long
    short_s: float = 0.2          # a tracklet that misses frames for longer than this is dropped
    resolve_s: float = 0.5        # a tracklet is given to a lost slot once it has lasted this long (< lag_s)
    coast_s: float = 0.5          # a lost slot is extrapolated with its last velocity for at most this long
    new_slot_s: float = 0.6       # an unclaimed tracklet must persist this long to open a free slot
    net_margin_m: float = 0.7     # a slot does not cross to the other side of the net beyond this
    max_interp_s: float = 3.0     # longest gap bridged between two observations of a slot
    max_overlap_s: float = 2.0    # a new id may start this long before the slot's old id stops (handover)
    same_person_m: float = 1.0    # ... if both sit within this distance where they overlap


@dataclass
class Slot:
    sid: int
    team: int = -1
    label: str = ""
    colour: np.ndarray | None = None
    tid: int | None = None        # ByteTrack id seen last
    tid_frame: int = -1
    x: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))   # [pos, vel] x [x, y]
    P: np.ndarray = field(default_factory=lambda: np.diag([0.05, 4.0]))
    last_seen: int = -1
    wrong_kit: int = 0            # consecutive frames its own id showed the other team's kit
    x_ls: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))  # state when last seen
    obs: dict = field(default_factory=dict)   # frame -> (foot uv, box, source)
    jumps: list = field(default_factory=list)
    # accumulated from finalised frames
    n_tracked: int = 0
    distance: float = 0.0
    top_speed: float = 0.0
    front: int = 0
    prev: tuple | None = None     # (frame, pos, speed) of the previous finalised tracked frame
    heat: np.ndarray | None = None
    # Monte Carlo replicas (bootstrap homographies + a foot bias drawn once per player)
    bias: np.ndarray | None = None
    mc_dist: np.ndarray | None = None
    mc_top: np.ndarray | None = None
    mc_front: np.ndarray | None = None
    mc_prev: tuple | None = None  # (frame, pos (K,2), speed (K,))


class OnlineAnalyzer:
    def __init__(self, tpl, fps, size, cfg: OnlineConfig | None = None, acfg: AnalysisConfig | None = None, seed=0):
        self.tpl, self.fps, self.size = tpl, float(fps), size
        self.cfg, self.acfg = cfg or OnlineConfig(), acfg or AnalysisConfig()
        self.dt = 1.0 / self.fps
        self.rng = np.random.default_rng(seed)
        self.L = int(round(self.cfg.lag_s * self.fps))
        self.hop = max(1, int(round(self.cfg.hop_s * self.fps)))
        self.burn = int(round(self.cfg.burn_s * self.fps))
        self.n_warm = int(round(self.cfg.warmup_s * self.fps))
        self.regs, self.dets = [], []
        self.H_live = []              # causal homography per frame (court -> image)
        self.slots: list[Slot] = []
        self.kits = None              # median jersey colour of team 0 / team 1, if they differ
        self.team_side = {}           # team -> +1 / -1: its side of the net
        self.f = None
        self.ready = False
        self.done = 0                 # frames [0, done) are finalised
        self.pending: dict = {}       # tracklets: ByteTrack ids nobody follows -> their observations
        self.frames_out: list = []    # per finalised frame: {sid: (xy, speed, airborne, observed)}
        self.n_commits = 0            # id changes resolved through tracklets
        nx, ny = int(np.ceil(tpl.length / self.acfg.heat_cell)), int(np.ceil(tpl.width / self.acfg.heat_cell))
        self.heat_shape = (nx, ny)

    # ------------------------------------------------------------------ input
    def push(self, reg, dets):
        """Add one frame (court registration + detections from the online detector/tracker).
        Returns the frames finalised by this call (list of dicts sid -> (xy, speed, airborne, observed))."""
        t = len(self.regs)
        self.regs.append(reg)
        self.dets.append(dets)
        prev = self.H_live[-1] if self.H_live else None
        self.H_live.append(reg.H if (reg.ok or prev is None) else prev)
        if not self.ready:
            if t + 1 >= self.n_warm:
                self._warmup()
            return []
        self._associate(t)
        return self._finalise(t + 1 - self.L)

    def flush(self):
        """End of stream: finalise everything that is left."""
        if not self.ready and self.regs:
            self._warmup()
        return self._finalise(len(self.regs), final=True)

    # ------------------------------------------------------------------ warm-up
    def _warmup(self):
        T = len(self.regs)
        Hseq, _ = smooth_registrations(self.regs, self.tpl, self.fps)
        Hinv = np.linalg.inv(Hseq)
        est = [calibrate_focal(Hseq[i], self.size) for i in np.linspace(0, T - 1, min(T, 9)).astype(int)]
        self.f = float(np.median([e[0] for e in est]))
        by = {}
        for fi, dl in enumerate(self.dets):
            for d in dl:
                by.setdefault(d["tid"], []).append((fi, d))
        raw = [Track(tid, np.array([f for f, _ in v]), np.array([d["foot"] for _, d in v]),
                     np.array([d["box"] for _, d in v]), np.array([d["color"] for _, d in v])) for tid, v in by.items()]
        tracks = filter_and_stitch(raw, Hinv, self.fps, self.tpl, min_dur=min(0.6, 0.5 * self.cfg.warmup_s),
                                   max_players=self.cfg.max_players)
        assign_teams(tracks, Hinv, self.tpl)
        for tr in tracks:
            s = Slot(len(self.slots), tr.team, tr.label, np.median(tr.colors, 0))
            for fr, foot, box in zip(tr.frames, tr.feet, tr.boxes):
                s.obs[int(fr)] = (foot, box, "det")
            last = int(tr.frames[-1])
            s.tid = next((d["tid"] for d in self.dets[last] if np.allclose(d["foot"], tr.feet[-1])), None)
            s.tid_frame = s.last_seen = last
            xy = court_positions(tr, Hinv)
            pos, vel = kalman_rts(_dense(tr.frames, xy), self.dt, self.acfg.sigma_a, self.acfg.sigma_m)
            s.x = np.stack([pos[-1], vel[-1]])
            s.P = np.diag([self.acfg.sigma_m ** 2, 1.0])
            s.x_ls = s.x.copy()
            for _ in range(T - 1 - last):
                self._predict(s)
            self.slots.append(s)
        kits = [np.median([s.colour for s in self.slots if s.team == k], 0) for k in (0, 1)
                if any(s.team == k for s in self.slots)]
        self.kits = kits if len(kits) == 2 and np.linalg.norm(kits[0] - kits[1]) > self.cfg.kit_sep else None
        self.team_side = {}
        for k in (0, 1):
            xs = [np.median(court_positions(tr, Hinv)[:, 0]) for tr in tracks if tr.team == k]
            if xs:
                self.team_side[k] = float(np.sign(np.median(xs) - self.tpl.net_x)) or None
        # ids still alive at the end of the warm-up that no slot took become tracklets with their history
        claimed = {(f, tuple(np.round(o[0], 3))) for s in self.slots for f, o in s.obs.items()}
        for tid, v in by.items():
            if T - 1 - v[-1][0] > self.cfg.short_s * self.fps:
                continue
            v = [(f, d) for f, d in v if (f, tuple(np.round(d["foot"], 3))) not in claimed]
            if not v:
                continue
            xy = np.stack([apply_h(Hinv[f], np.asarray(d["foot"])[None])[0] for f, d in v])
            if not ((xy[:, 0] > -self.cfg.margin_m).all() and (xy[:, 0] < self.tpl.length + self.cfg.margin_m).all()):
                continue
            self.pending[tid] = {"f": [f for f, _ in v], "foot": [np.asarray(d["foot"]) for _, d in v],
                                 "box": [np.asarray(d["box"]) for _, d in v], "col": [np.asarray(d["color"]) for _, d in v],
                                 "xy": list(xy)}
        self.ready = True

    # ------------------------------------------------------------------ per frame
    def _predict(self, s):
        F = np.array([[1, self.dt], [0, 1.0]])
        q = self.acfg.sigma_a ** 2 * np.array([[self.dt ** 4 / 4, self.dt ** 3 / 2], [self.dt ** 3 / 2, self.dt ** 2]])
        s.x = F @ s.x
        s.P = F @ s.P @ F.T + q

    def _update(self, s, z):
        R = self.acfg.sigma_m ** 2
        k = s.P[:, 0] / (s.P[0, 0] + R)
        s.x = s.x + k[:, None] * (z - s.x[0])[None]
        s.P = s.P - np.outer(k, s.P[0])

    def _associate(self, t):
        """Frame by frame a slot only keeps its own ByteTrack id. Detections with any other id nobody
        owns become tracklets, and id changes are decided in `_resolve` with a short look-ahead."""
        c = self.cfg
        for s in self.slots:
            self._predict(s)
        dl = self.dets[t]
        xy = apply_h(np.linalg.inv(self.H_live[t]), np.array([d["foot"] for d in dl])) if dl else np.zeros((0, 2))
        on = ((xy[:, 0] > -c.margin_m) & (xy[:, 0] < self.tpl.length + c.margin_m)
              & (xy[:, 1] > -c.margin_m) & (xy[:, 1] < self.tpl.width + c.margin_m))
        owner = {s.tid: s.sid for s in self.slots if s.tid is not None and t - s.tid_frame <= c.own_s * self.fps}
        short = c.short_s * self.fps
        BIG = 1e6
        release = set()
        cost = np.full((len(self.slots), len(dl)), BIG)
        for i, s in enumerate(self.slots):
            lost_f = t - s.last_seen
            gate = min(c.gate_m + c.lost_speed * max(lost_f - 1, 0) * self.dt, c.max_gate_m) + 3 * np.sqrt(s.P[0, 0])
            side = self._side(s)
            for j, d in enumerate(dl):
                if not on[j]:
                    continue
                own = owner.get(d["tid"])
                if own is not None and own != s.sid:
                    continue                                     # that id belongs to another slot
                if not (d["tid"] == s.tid and own == s.sid):
                    continue                                     # id changes are decided through tracklets
                dist = np.linalg.norm(xy[j] - s.x[0])
                if dist > 2 * gate:
                    continue
                if side and (xy[j, 0] - self.tpl.net_x) * side < -c.net_margin_m:
                    continue                                     # never across the net
                if self._wrong_kit(d["color"], s):
                    s.wrong_kit += 1
                    if s.wrong_kit >= c.kit_frames:
                        release.add(s.sid)                       # ByteTrack moved this id to another player
                    continue
                s.wrong_kit = 0
                cost[i, j] = dist
        for s in self.slots:
            if s.sid in release:
                s.tid, s.wrong_kit = None, 0
                owner = {k: v for k, v in owner.items() if v != s.sid}
        used = set()
        if (cost < BIG).any():
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < BIG:
                    self._observe(self.slots[i], t, dl[j], xy[j])
                    used.add(dl[j]["tid"])
        # detections nobody follows -> tracklets
        for j, d in enumerate(dl):
            if on[j] and d["tid"] not in used and d["tid"] not in owner:
                p = self.pending.setdefault(d["tid"], {"f": [], "foot": [], "box": [], "col": [], "xy": []})
                for key, v in (("f", t), ("foot", d["foot"]), ("box", d["box"]), ("col", d["color"]), ("xy", xy[j])):
                    p[key].append(np.asarray(v) if key != "f" else v)
        for k in used:
            self.pending.pop(k, None)
        self.pending = {k: p for k, p in self.pending.items() if t - p["f"][-1] <= short}
        self._resolve(t)

    def _observe(self, s, t, d, xy):
        self._update(s, xy)
        s.obs[t] = (np.asarray(d["foot"]), np.asarray(d["box"]), "det")
        s.last_seen = t
        s.x_ls = s.x.copy()
        s.tid, s.tid_frame = d["tid"], t
        col = np.asarray(d["color"], float)
        s.colour = col if s.colour is None else 0.98 * s.colour + 0.02 * col

    def _resolve(self, t):
        """Give tracklets that have lasted `resolve_s` to lost slots (or to new slots).

        All live tracklets, also the young ones, compete in one assignment, so a tracklet is not given
        to a slot whose own player reappears a moment later. Only ripe tracklets are committed; their
        observations are written back into the slot (they are younger than the output delay)."""
        c = self.cfg
        if not self.pending:
            return
        R = int(round(c.resolve_s * self.fps))
        keys = [k for k, p in self.pending.items() if p["f"][-1] == t]
        ripe = {k for k in keys if t - self.pending[k]["f"][0] + 1 >= R}
        if not ripe:
            return
        lost = [s for s in self.slots if s.last_seen < t]
        BIG = 1e6
        cost = np.full((len(keys), len(lost)), BIG)
        for a, k in enumerate(keys):
            p = self.pending[k]
            f0 = p["f"][0]
            z0 = np.median(np.array(p["xy"][:3]), 0)
            col = np.median(np.array(p["col"]), 0)
            for b, s in enumerate(lost):
                if f0 <= s.last_seen:
                    # id handover: the new id started while the slot's old id was still reported
                    if (s.last_seen - f0 + 1) * self.dt > c.max_overlap_s:
                        continue
                    shared = [i for i, f in enumerate(p["f"]) if f in s.obs]
                    if shared and np.median([np.linalg.norm(p["xy"][i] - self._slot_xy(s, p["f"][i]))
                                             for i in shared]) > c.same_person_m:
                        continue
                    before = [f for f in s.obs if f < f0]
                    if not before:
                        continue
                    gate = c.same_person_m
                    dist = np.linalg.norm(z0 - self._slot_xy(s, max(before)))
                else:
                    gap = (f0 - s.last_seen) * self.dt
                    if gap > c.max_interp_s:
                        continue
                    pred = s.x_ls[0] + s.x_ls[1] * min(gap, c.coast_s)
                    gate = c.gate_m + c.lost_speed * gap            # same running-speed budget as offline stitching
                    dist = np.linalg.norm(z0 - pred)
                if dist > gate:
                    continue
                side = self._side(s)
                if side and (z0[0] - self.tpl.net_x) * side < -c.net_margin_m:
                    continue
                if self._wrong_kit(col, s):
                    continue
                dc = np.linalg.norm(col - s.colour) if s.colour is not None else 0.0
                cost[a, b] = dist / gate + c.colour_w * dc
        taken = set()
        if len(lost) and (cost < BIG).any():
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] < BIG and keys[a] in ripe:
                    self._commit(lost[b], keys[a], t)
                    taken.add(keys[a])
        for k in ripe - taken:
            p = self.pending[k]
            if len(self.slots) < c.max_players and t - p["f"][0] + 1 >= c.new_slot_s * self.fps:
                z = np.median(np.array(p["xy"]), 0)
                col = np.median(np.array(p["col"]), 0)
                if self.kits is not None:                        # clear kits: the jersey decides the team
                    team = int(np.linalg.norm(col - self.kits[1]) < np.linalg.norm(col - self.kits[0]))
                else:
                    team = int(z[0] > self.tpl.net_x) if self.tpl.has_net else 0
                s = Slot(len(self.slots), team, f"{'AB'[team]}{1 + sum(o.team == team for o in self.slots)}", col)
                self.slots.append(s)
                self._commit(s, k, t)

    def _side(self, s):
        """Side of the net a player belongs to: its team's side (a player at the net can stand a few
        centimetres over the centre line, so the current position is not used)."""
        if not self.tpl.has_net:
            return 0
        return self.team_side.get(s.team) or np.sign(s.x_ls[0, 0] - self.tpl.net_x)

    def _slot_xy(self, s, f):
        return apply_h(np.linalg.inv(self.H_live[f]), np.asarray(s.obs[f][0])[None])[0]

    def _commit(self, s, k, t):
        p = self.pending.pop(k)
        self.n_commits += 1
        for f in [f for f in s.obs if f >= max(p["f"][0], self.done)]:
            del s.obs[f]                                         # handover: the new id replaces the old tail
        R = self.acfg.sigma_m ** 2
        s.x, s.P = np.stack([p["xy"][0], np.zeros(2)]), np.diag([R, 4.0])
        prev = p["f"][0]
        for f, foot, box, xy in zip(p["f"], p["foot"], p["box"], p["xy"]):
            for _ in range(f - prev):
                self._predict(s)
            self._update(s, xy)
            if f >= self.done:
                s.obs[f] = (foot, box, "det")
            prev = f
        s.last_seen, s.x_ls = p["f"][-1], s.x.copy()
        s.tid, s.tid_frame = k, t

    def _wrong_kit(self, colour, s):
        """Jersey nearer the other team's kit than to this player's own colour (only when the two
        kits are clearly different; comparing with the player's own colour tolerates odd kits)."""
        if self.kits is None or s.team not in (0, 1) or s.colour is None:
            return False
        c = np.asarray(colour, float)
        return np.linalg.norm(c - self.kits[1 - s.team]) < np.linalg.norm(c - s.colour)

    # ------------------------------------------------------------------ finalisation
    def _finalise(self, upto, final=False):
        """Finalise frames [done, upto) in hops; each sees data up to the newest frame."""
        out = []
        T = len(self.regs)
        while self.ready and (self.done + self.hop <= upto or (final and self.done < upto)):
            e0, e1 = self.done, min(self.done + self.hop, upto)
            w0, w1 = max(0, e0 - self.burn), T
            Hw, Hsw = smooth_registrations(self.regs[w0:w1], self.tpl, self.fps)
            Hinv = np.linalg.inv(Hw)
            Pw = np.stack([pose_from_h(h, self.f, self.size).P for h in Hw])
            batch = [dict() for _ in range(e1 - e0)]
            for s in self.slots:
                self._finalise_slot(s, w0, w1, e0, e1, Hw, Hsw, Hinv, Pw, batch, final)
            out += batch
            self.done = e1
        self.frames_out += out
        return out

    def _finalise_slot(self, s, w0, w1, e0, e1, Hw, Hsw, Hinv, Pw, batch, final):
        fr = np.array(sorted(f for f in s.obs if w0 <= f < w1))
        if not len(fr):
            return
        n = w1 - w0
        uv = np.full((n, 2), np.nan); bh = np.full(n, np.nan); top = np.full((n, 2), np.nan)
        for f in fr:
            foot, box, _ = s.obs[f]
            uv[f - w0] = foot; bh[f - w0] = box[3] - box[1]; top[f - w0] = ((box[0] + box[2]) / 2, box[1])
        # frames that can be emitted: between two observations no more than max_interp_s apart
        a, b = fr[0] - w0, fr[-1] - w0
        air, ground, z, jumps = detect_jumps(uv[a:b + 1], Hinv[a:b + 1], Pw[a:b + 1], self.fps,
                                             min_height=self.acfg.min_jump_m, box_h=bh[a:b + 1], top_uv=top[a:b + 1])
        meas = np.full((n, 2), np.nan); meas[a:b + 1] = ground
        # commit jumps whose landing context is in, and that are not committed yet
        horizon = (w1 - w0) if final else (w1 - w0 - 4)
        for j in jumps:
            ja, jb = j["f0"] + a, j["f1"] + a
            if jb >= horizon or any(ja + w0 <= c["f1"] and jb + w0 >= c["f0"] for c in s.jumps):
                continue
            if ja + w0 < e0:
                continue                      # belongs to frames already finalised without it
            jj = dict(j, f0=ja, f1=jb, f_peak=j["f_peak"] + a)
            sig = self._jump_sigma(jj, uv, meas, Hsw, s)
            if jj["height_m"] >= self.acfg.jump_snr * sig:
                jj.update(sigma_m=sig, f0=ja + w0, f1=jb + w0, f_peak=jj["f_peak"] + w0, t_s=(jj["f_peak"] + w0) * self.dt)
                s.jumps.append(jj)
        # airborne frames of committed jumps: ground position interpolated between take-off and landing
        xy_obs = np.full((n, 2), np.nan)
        ok = np.isfinite(uv).all(1)
        xy_obs[ok] = np.einsum("tij,tj->ti", Hinv[ok], np.c_[uv[ok], np.ones(ok.sum())])[:, :2] / \
            np.einsum("tij,tj->ti", Hinv[ok], np.c_[uv[ok], np.ones(ok.sum())])[:, 2:3]
        airborne = np.zeros(n, bool)
        for c in s.jumps:
            ca, cb = c["f0"] - w0, c["f1"] - w0
            if cb < 0 or ca >= n:
                continue
            pa, pb = xy_obs[max(ca - 3, 0):max(ca + 1, 0)], xy_obs[max(cb, 0):cb + 4]
            pa, pb = pa[np.isfinite(pa).all(1)], pb[np.isfinite(pb).all(1)]
            if not len(pa) or not len(pb):
                continue                      # take-off or landing outside this window
            ga, gb = np.median(pa, 0), np.median(pb, 0)
            w = np.linspace(0, 1, cb - ca + 1)[:, None]
            seg = ga[None] * (1 - w) + gb[None] * w
            lo, hi = max(ca, 0), min(cb, n - 1)
            xy_obs[lo:hi + 1] = seg[lo - ca:hi - ca + 1]
            airborne[lo:hi + 1] = True
        pos, vel = kalman_rts(xy_obs, self.dt, self.acfg.sigma_a, self.acfg.sigma_m)
        sp = np.linalg.norm(vel, axis=1)
        # the same path under every bootstrap homography (+ per-player foot bias), for the 90% bands
        K = Hsw.shape[1]
        if s.bias is None:
            s.bias = self.rng.normal(0, self.acfg.foot_bias_px, (K, 2))
        jw = [{"f0": c["f0"] - w0, "f1": c["f1"] - w0} for c in s.jumps if c["f0"] - w0 >= 3 and c["f1"] - w0 + 4 <= n]
        xyk = replica_positions(uv, np.linalg.inv(Hsw), s.bias, jw, xy_obs, pos, self.rng)
        pk, vk = kalman_rts(xyk.reshape(n, -1), self.dt, self.acfg.sigma_a, self.acfg.sigma_m)
        pk, spk = pk.reshape(n, K, 2), np.linalg.norm(vk.reshape(n, K, 2), axis=-1)
        obs_idx = np.where(ok)[0]
        maxgap = self.cfg.max_interp_s * self.fps
        for e in range(e0, e1):
            i = e - w0
            kp = np.searchsorted(obs_idx, i, side="right")       # observations at or before i: [0, kp)
            kn = np.searchsorted(obs_idx, i, side="left")        # observations at or after i: [kn, ...)
            if kp == 0 or kn == len(obs_idx) or obs_idx[kn] - obs_idx[kp - 1] > maxgap:
                continue                      # before first / after last sighting, or gap too long
            self._accumulate(s, e, pos[i], sp[i], pk[i], spk[i])
            batch[e - e0][s.sid] = (pos[i], float(sp[i]), bool(airborne[i]), bool(ok[i]))

    def _jump_sigma(self, j, uv, meas, Hsw, s):
        K = Hsw.shape[1]
        bias = self.rng.normal(0, self.acfg.foot_bias_px, (K, 2))
        a, b = j["f0"], j["f1"]
        sl = slice(max(a - 10, 0), min(b + 11, len(uv)))
        uvs, ms = uv[sl], meas[sl]
        pos, _ = kalman_rts(np.where(np.isfinite(ms).all(1)[:, None], ms, np.nan), self.dt, self.acfg.sigma_a, self.acfg.sigma_m)
        jl = dict(j, f0=a - sl.start, f1=b - sl.start)
        xyk = replica_positions(uvs, np.linalg.inv(Hsw[sl]), bias, [jl], ms, pos, self.rng)
        return jump_sigma(jl, uvs, xyk, Hsw[sl], self.f, self.size, bias, self.rng, self.acfg)

    def _accumulate(self, s, e, pos, sp, pk, spk):
        if s.mc_dist is None:
            s.mc_dist, s.mc_top, s.mc_front = np.zeros(len(pk)), np.zeros(len(pk)), np.zeros(len(pk))
        s.mc_top = np.maximum(s.mc_top, spk)
        s.mc_front += np.abs(pk[:, 0] - self.tpl.net_x) <= self.tpl.front_zone
        if s.mc_prev is not None and s.mc_prev[0] == e - 1:
            s.mc_dist += np.linalg.norm(pk - s.mc_prev[1], axis=1) * (0.5 * (s.mc_prev[2] + spk) > self.acfg.speed_deadband)
        s.mc_prev = (e, pk, spk)
        s.n_tracked += 1
        s.top_speed = max(s.top_speed, sp)
        s.front += int(abs(pos[0] - self.tpl.net_x) <= self.tpl.front_zone)
        if s.heat is None:
            s.heat = np.zeros(self.heat_shape)
        ix = int(np.clip(pos[0] / self.acfg.heat_cell, 0, self.heat_shape[0] - 1))
        iy = int(np.clip(pos[1] / self.acfg.heat_cell, 0, self.heat_shape[1] - 1))
        if 0 <= pos[0] <= self.tpl.length and 0 <= pos[1] <= self.tpl.width:
            s.heat[ix, iy] += self.dt
        if s.prev is not None and s.prev[0] == e - 1:
            if 0.5 * (s.prev[2] + sp) > self.acfg.speed_deadband:
                s.distance += float(np.linalg.norm(pos - s.prev[1]))
        s.prev = (e, pos, sp)

    # ------------------------------------------------------------------ output
    def live_positions(self):
        """Undelayed (filtered, not smoothed) court position of every slot seen in the last second."""
        t = len(self.regs) - 1
        return {s.sid: s.x[0].copy() for s in self.slots if t - s.last_seen <= self.fps}

    def stats(self):
        """Per-player stats over the finalised frames [0, done), same keys as the offline analysis."""
        return [dict(self._stats(s), label=s.label, team=s.team, sid=s.sid) for s in self.slots]

    def _stats(self, s):
        dur = s.n_tracked * self.dt
        js = [j for j in s.jumps if j["f_peak"] < self.done]
        hs = np.array([j["height_m"] for j in js]); ss = np.array([j["sigma_m"] for j in js])
        n = max(s.n_tracked, 1)
        st = {"duration_s": dur, "tracked_s": dur}
        if s.mc_dist is None:
            s.mc_dist = s.mc_top = s.mc_front = np.zeros(1)
        for key, v, mc in (("distance_m", s.distance, s.mc_dist), ("top_speed_ms", s.top_speed, s.mc_top),
                           ("avg_speed_ms", s.distance / max(dur, 1e-6), s.mc_dist / max(dur, 1e-6)),
                           ("front_row_frac", s.front / n, s.mc_front / n)):
            st[key] = float(v)
            st[key + "_lo"], st[key + "_hi"] = band(float(v), mc, key, self.acfg)
        st.update(n_jumps=len(js), max_jump_cm=float(hs.max() * 100) if len(hs) else 0.0,
                  max_jump_sigma_cm=float(ss[np.argmax(hs)] * 100) if len(hs) else 0.0,
                  avg_jump_cm=float(hs.mean() * 100) if len(hs) else 0.0,
                  avg_jump_sigma_cm=float(np.sqrt((ss ** 2).sum()) / len(ss) * 100) if len(hs) else 0.0,
                  back_row_frac=1 - st["front_row_frac"])
        return st

    def results(self):
        """Finalised output in the offline `PlayerResult` form (for rendering and exports). Frames inside
        a player's span that could not be bridged (gap > max_interp_s) are NaN."""
        T = len(self.frames_out)
        out = []
        for s in self.slots:
            fr = [f for f in range(T) if s.sid in self.frames_out[f]]
            if not fr:
                continue
            f0, f1 = fr[0], fr[-1]
            m = f1 - f0 + 1
            xy = np.full((m, 2), np.nan); sp = np.zeros(m); air = np.zeros(m, bool); obs = np.zeros(m, bool)
            for f in fr:
                p, v, a, o = self.frames_out[f][s.sid]
                xy[f - f0], sp[f - f0], air[f - f0], obs[f - f0] = p, v, a, o
            st = self._stats(s)
            st["observed_frac"] = float(obs.mean())
            jumps = [dict(j) for j in s.jumps if j["f_peak"] < self.done]
            r = PlayerResult(s.label, s.team, s.sid, f0, f1, xy, sp, air, np.full(m, np.nan), obs, jumps, st,
                             s.heat if s.heat is not None else np.zeros(self.heat_shape))
            r.boxes = {int(f): o[1] for f, o in s.obs.items() if f0 <= f <= f1}
            out.append(r)
        return out


def _dense(frames, xy):
    out = np.full((frames[-1] - frames[0] + 1, 2), np.nan)
    out[frames - frames[0]] = xy
    return out

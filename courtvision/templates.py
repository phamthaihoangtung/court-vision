"""Court templates in metres. x runs along the court length (net at x = L/2), y across."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CourtTemplate:
    name: str
    length: float
    width: float
    xs: list            # full-width lines of constant x (used for hypothesis generation)
    ys: list            # full-length lines of constant y
    extra: list = field(default_factory=list)   # additional partial segments (x0,y0,x1,y1)
    front_zone: float = 3.0                     # depth of the "front row" zone from the net
    has_net: bool = True

    @property
    def net_x(self) -> float:
        return self.length / 2.0

    @property
    def segments(self) -> np.ndarray:
        segs = [(x, 0.0, x, self.width) for x in self.xs]
        segs += [(0.0, y, self.length, y) for y in self.ys]
        segs += list(self.extra)
        return np.asarray(segs, dtype=np.float64)

    @property
    def corners(self) -> np.ndarray:
        L, W = self.length, self.width
        return np.array([[0, 0], [L, 0], [L, W], [0, W]], dtype=np.float64)

    def sample(self, step: float = 0.5):
        """Points sampled along every template line -> (pts[N,2], line_index[N])."""
        pts, idx = [], []
        for j, (x0, y0, x1, y1) in enumerate(self.segments):
            n = max(2, int(round(np.hypot(x1 - x0, y1 - y0) / step)) + 1)
            t = np.linspace(0, 1, n)
            pts.append(np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], 1))
            idx.append(np.full(n, j))
        return np.concatenate(pts), np.concatenate(idx)


def _circle(cx, cy, r, n=16):
    a = np.linspace(0, 2 * np.pi, n + 1)
    p = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    return [(p[i, 0], p[i, 1], p[i + 1, 0], p[i + 1, 1]) for i in range(n)]


TEMPLATES = {
    # FIVB indoor: 18 x 9 m, centre line, attack lines 3 m from the centre line
    "indoor": CourtTemplate("indoor", 18.0, 9.0, xs=[0.0, 6.0, 9.0, 12.0, 18.0], ys=[0.0, 9.0]),
    # FIVB beach: 16 x 8 m, boundary lines only
    "beach": CourtTemplate("beach", 16.0, 8.0, xs=[0.0, 16.0], ys=[0.0, 8.0], front_zone=2.5),
    # Padel: 20 x 10 m, service lines 6.95 m from the net, centre service line between them
    "padel": CourtTemplate("padel", 20.0, 10.0, xs=[0.0, 3.05, 16.95, 20.0], ys=[0.0, 10.0],
                           extra=[(3.05, 5.0, 16.95, 5.0)], front_zone=3.0),
    # Small-sided football (typical pitch sizes; penalty areas vary by venue and are omitted)
    "football5": CourtTemplate("football5", 40.0, 20.0, xs=[0.0, 20.0, 40.0], ys=[0.0, 20.0],
                               extra=_circle(20.0, 10.0, 3.0), front_zone=10.0, has_net=False),
    "football7": CourtTemplate("football7", 60.0, 40.0, xs=[0.0, 30.0, 60.0], ys=[0.0, 40.0],
                               extra=_circle(30.0, 20.0, 6.0), front_zone=15.0, has_net=False),
}


def get_template(name: str) -> CourtTemplate:
    if name not in TEMPLATES:
        raise KeyError(f"unknown court '{name}', choose from {sorted(TEMPLATES)}")
    return TEMPLATES[name]

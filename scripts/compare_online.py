"""Online (frame by frame, fixed delay) vs offline (whole clip) analysis on the cached clip runs.

Both analyses read the same pass-1 cache (court registration + YOLO/ByteTrack detections) in
results/clip_*/pass1.pkl. Online players are matched to offline players by position; the table shows
coverage, distance and how often both name the same person (< 0.5 m apart). Writes
results/online_vs_offline.json.

    python scripts/compare_online.py [--lag 2.0]
"""
import argparse, glob, json, os, pickle, sys, time, warnings
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from courtvision.court import smooth_registrations
from courtvision.online import OnlineAnalyzer, OnlineConfig
from courtvision.stats import analyze
from courtvision.templates import get_template
from courtvision.tracking import Track, assign_teams, filter_and_stitch

warnings.filterwarnings("ignore", category=RuntimeWarning)


def offline(regs, dets, tpl, fps, size):
    H, Hs = smooth_registrations(regs, tpl, fps)
    Hinv = np.linalg.inv(H)
    by = {}
    for fi, dl in enumerate(dets):
        for d in dl:
            by.setdefault(d["tid"], []).append((fi, d))
    raw = [Track(t, np.array([f for f, _ in v]), np.array([d["foot"] for _, d in v]), np.array([d["box"] for _, d in v]),
                 np.array([d["color"] for _, d in v])) for t, v in by.items()]
    tracks = filter_and_stitch(raw, Hinv, fps, tpl)
    assign_teams(tracks, Hinv, tpl)
    return analyze(tracks, H, Hs, fps, tpl, size)[0]


def positions(results, T):
    P = np.full((T, len(results), 2), np.nan)
    for k, r in enumerate(results):
        P[r.f0:r.f1 + 1, k] = r.xy
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=float, default=2.0)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    tpl = get_template("indoor")
    report = {}
    for d in sorted(glob.glob(f"{a.results}/clip_*")):
        if not os.path.exists(f"{d}/pass1.pkl"):
            continue
        s = json.load(open(f"{d}/summary.json"))
        size, fps = (s["source"]["width"], s["source"]["height"]), s["processed"]["fps"]
        c = pickle.load(open(f"{d}/pass1.pkl", "rb"))
        regs, dets = c["regs"], c["dets"]
        T = len(regs)
        off = offline(regs, dets, tpl, fps, size)
        on = OnlineAnalyzer(tpl, fps, size, OnlineConfig(lag_s=a.lag))
        t0 = time.perf_counter()
        for r, dl in zip(regs, dets):
            on.push(r, dl)
        on.flush()
        ms = (time.perf_counter() - t0) / T * 1000
        onr = on.results()
        A, B = positions(onr, T), positions(off, T)
        cost = np.full((len(onr), len(off)), 99.0)
        for i in range(len(onr)):
            for k in range(len(off)):
                dd = np.linalg.norm(A[:, i] - B[:, k], axis=1)
                if np.isfinite(dd).sum() > 10:
                    cost[i, k] = np.nanmedian(dd)
        rows = []
        for i, k in zip(*linear_sum_assignment(cost)):
            dd = np.linalg.norm(A[:, i] - B[:, k], axis=1)
            m = np.isfinite(dd)
            rows.append({"online": onr[i].label, "offline": off[k].label,
                         "coverage_online": float(np.isfinite(A[:, i, 0]).mean()), "coverage_offline": float(np.isfinite(B[:, k, 0]).mean()),
                         "distance_online": onr[i].stats["distance_m"], "distance_offline": off[k].stats["distance_m"],
                         "same_person": float((dd[m] < 0.5).mean()) if m.any() else 0.0})
        tag = os.path.basename(d).replace("clip_", "")
        report[tag] = {"frames": T, "lag_s": a.lag, "ms_per_frame": ms, "players": rows,
                       "coverage_online": float(np.mean([r["coverage_online"] for r in rows])),
                       "coverage_offline": float(np.mean([r["coverage_offline"] for r in rows])),
                       "same_person": float(np.mean([r["same_person"] for r in rows]))}
        q = report[tag]
        print(f"{tag:9s} coverage offline {q['coverage_offline'] * 100:5.1f}% -> online {q['coverage_online'] * 100:5.1f}%   "
              f"same person {q['same_person'] * 100:5.1f}%   {ms:.0f} ms/frame (analysis only)")
        for r in rows:
            if r["same_person"] < 0.995 or abs(r["coverage_online"] - r["coverage_offline"]) > 0.02:
                print(f"    {r['offline']:>3} -> {r['online']:<3} coverage {r['coverage_offline'] * 100:3.0f}% -> {r['coverage_online'] * 100:3.0f}%  "
                      f"distance {r['distance_offline']:5.1f} -> {r['distance_online']:5.1f} m  same person {r['same_person'] * 100:5.1f}%")
    json.dump(report, open(f"{a.results}/online_vs_offline.json", "w"), indent=1)


if __name__ == "__main__":
    main()

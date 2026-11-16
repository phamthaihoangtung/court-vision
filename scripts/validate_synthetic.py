"""Synthetic validation + sensitivity analysis. Usage: python scripts/validate_synthetic.py [--trials 20]"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from courtvision.validate import run_trial, summarize

SWEEPS = {
    "baseline (clutter, coloured other-sport lines)": {},
    "no clutter": {"clutter": False},
    "clutter, ALL lines white (worst case)": {"clutter_white": True},
    "foot noise 0.5 px": {"noise_px": 0.5},
    "foot noise 3 px": {"noise_px": 3.0},
    "foot noise 5 px": {"noise_px": 5.0},
    "low camera (elev 20 deg)": {"elev": 20},
    "high camera (elev 65 deg)": {"elev": 65},
    "occlusion gaps x3": {"dropout": 0.25},
    "aggressive movement (accel up to 8 m/s^2)": {"acc_max": 8.0},
    "mid camera (elev 45 deg)": {"elev": 45},
    "barrel distortion lam=-0.10, corrected": {"lam": -0.10},
    "barrel distortion lam=-0.25, corrected": {"lam": -0.25},
    "barrel distortion lam=-0.10, NOT corrected": {"lam": -0.10, "undistort": False},
    "heavy image noise (sigma 15)": {"img_noise": 15.0},
    "beach court (4 lines only)": {"court": "beach", "clutter": False},
}


def _job(a):
    try:
        return run_trial(a[0], **a[1])
    except Exception as e:                                   # a crash is a failed trial, not a skipped one
        return {"seed": a[0], "ok": False, "error": repr(e)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="results/validation.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for name, kw in SWEEPS.items():
            if a.only and a.only not in name:
                continue
            trials = list(ex.map(_job, [(s, kw) for s in range(a.trials)]))
            res[name] = summarize(trials)
            res[name]["errors"] = [t["error"] for t in trials if "error" in t][:3]
            s = res[name]
            print(f"{name:48s} court ok {s['court_success_pct']:5.1f}%  reproj {s.get('reproj_cm_med', float('nan')):5.1f} cm"
                  f"  dist err {s.get('dist_err_pct', float('nan')):4.1f}%  vtop err {s.get('top_speed_err_ms', float('nan')):.2f} m/s"
                  f"  jump MAE {s.get('jump_mae_cm', float('nan')):4.1f} cm  recall {s.get('jump_recall_pct', float('nan')):3.0f}%"
                  f"  prec {s.get('jump_precision_pct', float('nan')):3.0f}%", flush=True)
            if s["errors"]:
                print("   errors:", s["errors"][0][:300])
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")

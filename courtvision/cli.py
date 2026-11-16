"""Command line: python -m courtvision run clip.mp4 --out results/clip"""
from __future__ import annotations
import argparse
import json


def add_runtime_args(ap):
    ap.add_argument("--court", default="indoor", help="indoor | beach | padel | football5 | football7")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    ap.add_argument("--model-size", default="n", choices=["n", "s", "m"])
    ap.add_argument("--pose", action="store_true", help="use the YOLO pose model (ankle keypoints) for foot points / jumps")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--undistort", default="auto", help="auto | off | <lambda of the division model>")
    ap.add_argument("--line-color", default="white", choices=["white", "any"])
    ap.add_argument("--max-players", type=int, default=12)


def config_from_args(a):
    from .pipeline import Config
    return Config(court=a.court, device=a.device, model_size=a.model_size, pose=a.pose, imgsz=a.imgsz,
                  frame_stride=a.frame_stride, max_frames=a.max_frames, undistort=a.undistort,
                  line_color=a.line_color, max_players=a.max_players)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="courtvision")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse a clip")
    r.add_argument("video")
    r.add_argument("--out", default="results/run")
    r.add_argument("--no-render", action="store_true")
    r.add_argument("--no-gif", action="store_true")
    r.add_argument("--reuse-pass1", action="store_true", help="reuse cached detections/registration in --out")
    add_runtime_args(r)
    a = ap.parse_args(argv)
    from .pipeline import run
    cfg = config_from_args(a)
    cfg.render, cfg.gif, cfg.reuse_pass1 = not a.no_render, not a.no_gif, a.reuse_pass1
    summary, _ = run(a.video, a.out, cfg, progress=lambda p, m: print(f"\r[{p * 100:5.1f}%] {m}   ", end="", flush=True))
    print()
    print(json.dumps({k: summary[k] for k in ("processed", "court", "camera", "lens", "tracks", "timing")}, indent=1, default=float))
    for p in summary["players"]:
        print(f"{p['label']:>3} dist {p['distance_m']:6.1f} m [{p['distance_m_lo']:.1f},{p['distance_m_hi']:.1f}]  top {p['top_speed_ms'] * 3.6:5.1f} km/h  "
              f"jumps {p['n_jumps']}  max {p['max_jump_cm']:.0f}+-{p['max_jump_sigma_cm']:.0f} cm  front {p['front_row_frac'] * 100:.0f}%")


if __name__ == "__main__":
    main()

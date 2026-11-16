"""Per-stage FPS benchmark. Rerun on your GPU:
    python scripts/benchmark.py clip.mp4 --device cuda --model-size s --max-frames 200
"""
import argparse, json, os, platform, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from courtvision.cli import add_runtime_args, config_from_args
from courtvision.pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--json", default=None, help="write the table to this file")
    add_runtime_args(ap)
    ap.set_defaults(max_frames=150, frame_stride=2)
    a = ap.parse_args()
    cfg = config_from_args(a)
    cfg.gif = False
    with tempfile.TemporaryDirectory() as d:
        summary, _ = run(a.video, d, cfg)
    import torch
    dev = summary["processed"]["device"]
    hw = torch.cuda.get_device_name(0) if dev.startswith("cuda") else f"{platform.processor() or platform.machine()} x{os.cpu_count()}"
    print(f"\ndevice: {dev} ({hw})  weights: {summary['processed']['weights']}  imgsz: {a.imgsz}  frames: {summary['processed']['frames']}"
          f"  source: {summary['source']['width']}x{summary['source']['height']}")
    print(f"{'stage':22s} {'ms/frame':>10s} {'FPS':>8s}")
    tot = 0.0
    for k, v in summary["timing"].items():
        if k in ("lens_estimate", "stat_cards"):
            print(f"{k:22s} {v['total_s'] * 1000:10.1f} {'(once)':>8s}")
            continue
        tot += v["ms_per_frame"]
        print(f"{k:22s} {v['ms_per_frame']:10.2f} {v['fps']:8.1f}")
    print(f"{'TOTAL (per frame)':22s} {tot:10.2f} {1000 / tot:8.1f}")
    if a.json:
        json.dump({"device": dev, "hardware": hw, "timing": summary["timing"], "total_fps": 1000 / tot}, open(a.json, "w"), indent=1)

"""Gradio demo: upload a clip -> annotated video, stat cards, CSV/JSON.   python app.py"""
import glob, os, tempfile
import gradio as gr
from courtvision.pipeline import Config, run


def analyse(video, court, model_size, device, pose, stride, max_frames, undistort, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload a clip first")
    out = tempfile.mkdtemp(prefix="courtvision_")
    cfg = Config(court=court, model_size=model_size, device=device, pose=pose, frame_stride=int(stride),
                 max_frames=int(max_frames), undistort="auto" if undistort else "off", gif=False)
    try:
        summary, _ = run(video, out, cfg, progress=lambda p, m: progress(p, desc=m))
    except RuntimeError as e:
        raise gr.Error(str(e))
    files = [os.path.join(out, f) for f in ("player_stats.csv", "positions.csv", "jumps.csv", "summary.json")]
    import pandas as pd
    return (summary["video"], sorted(glob.glob(os.path.join(out, "cards", "*.png")), key=lambda p: ("summary" not in p, p)),
            pd.read_csv(files[0]).round(2), files)


with gr.Blocks(title="court-vision") as demo:
    gr.Markdown("# court-vision\nFixed-camera volleyball footage -> player positions in metres, speed, distance, jumps. No training.")
    with gr.Row():
        with gr.Column(scale=1):
            vid = gr.Video(label="Clip (fixed camera, court lines visible)")
            court = gr.Dropdown(["indoor", "beach", "padel", "football5", "football7"], value="indoor", label="Court template")
            size = gr.Radio(["n", "s", "m"], value="n", label="YOLO size")
            device = gr.Radio(["auto", "cpu", "cuda"], value="auto", label="Device")
            pose = gr.Checkbox(False, label="Pose model for foot points / jumps")
            und = gr.Checkbox(True, label="Auto lens undistortion")
            stride = gr.Slider(1, 6, value=2, step=1, label="Frame stride")
            maxf = gr.Number(300, label="Max frames (0 = all)", precision=0)
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=2):
            out_vid = gr.Video(label="Annotated video + minimap")
            gallery = gr.Gallery(label="Stat cards", columns=3)
            table = gr.Dataframe(label="Per-player stats")
            files = gr.Files(label="CSV / JSON")
    btn.click(analyse, [vid, court, size, device, pose, stride, maxf, und], [out_vid, gallery, table, files])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("HOST", "127.0.0.1"))

"""Player detection + ByteTrack using pretrained Ultralytics YOLO weights (no training)."""
from __future__ import annotations
import cv2
import numpy as np


def resolve_device(device="auto"):
    if device in (None, "auto"):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


class PlayerDetector:
    """size: n|s|m. pose=True loads the pose variant (boxes + ankle keypoints from one model)."""

    def __init__(self, size="n", device="auto", pose=False, imgsz=1280, conf=0.1, family="yolo11"):
        from ultralytics import YOLO
        self.device = resolve_device(device)
        self.pose = pose
        self.imgsz = imgsz
        self.conf = conf
        self.weights = f"{family}{size}{'-pose' if pose else ''}.pt"     # fetched from GitHub releases
        self.model = YOLO(self.weights)

    def __call__(self, frame):
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], imgsz=self.imgsz,
                             conf=self.conf, iou=0.6, device=self.device, verbose=False)[0]
        out = []
        if r.boxes is None or r.boxes.id is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        kp = r.keypoints.data.cpu().numpy() if (self.pose and r.keypoints is not None) else None
        for i, (b, tid, c) in enumerate(zip(boxes, ids, confs)):
            foot = np.array([(b[0] + b[2]) / 2, b[3]])
            src = "bbox"
            if kp is not None:
                ank = kp[i, 15:17]
                good = ank[:, 2] > 0.5
                if good.all():
                    # lower ankle = supporting foot; +4% of box height reaches the sole
                    foot = np.array([ank[:, 0].mean(), ank[:, 1].max() + 0.04 * (b[3] - b[1])])
                    src = "pose"
            out.append({"tid": int(tid), "box": b, "conf": float(c), "foot": foot, "foot_src": src,
                        "color": torso_color(frame, b)})
        return out


def torso_color(frame, box):
    """Median Lab colour of the jersey region (upper-middle of the box)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    xa, xb = int(x0 + 0.3 * w), int(x0 + 0.7 * w) + 1
    ya, yb = int(y0 + 0.2 * h), int(y0 + 0.5 * h) + 1
    H, W = frame.shape[:2]
    patch = frame[max(ya, 0):min(yb, H), max(xa, 0):min(xb, W)]
    if patch.size == 0:
        return np.zeros(3)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, 0).astype(float)

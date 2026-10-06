"""Development helper: speed and success of per-frame court registration (half-size SIFT)."""
import time

import cv2
import numpy as np

from src.bball.config import load_config
from src.bball.video.homography import alignment_score, refine_homography
from src.bball.video.register import build_ref, register

cfg = load_config()["video"]
cap = cv2.VideoCapture(cfg["path"])


def grab(n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    return cap.read()[1]


refs = [build_ref(r["name"], r["frame"], grab(r["frame"]), r["picks"]) for r in cfg["references"]]
frames = list(range(100, 12000, 400))
t_reg = t_ref = 0.0
rows = []
for n in frames:
    gray = cv2.cvtColor(grab(n), cv2.COLOR_BGR2GRAY)
    t = time.time()
    H, name, inl = register(gray, refs, min_inliers=15)
    t_reg += time.time() - t
    if H is None:
        rows.append((n, name, inl, np.nan))
        continue
    t = time.time()
    Hr, s = refine_homography(gray, H, stages=((3.0, 12.0), (1.5, 6.0)), reg=2e-3)
    t_ref += time.time() - t
    rows.append((n, name, inl, s))
print(f"avg register {t_reg / len(frames):.3f} s/frame, avg refine {t_ref / len(frames):.3f} s/frame")
for n, name, inl, s in rows:
    print(f"frame {n:5d} ref {name} inliers {inl:4d} score {s:.2f}")

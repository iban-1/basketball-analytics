"""Development helper: show overlays for borderline registrations to choose thresholds."""
import cv2
import numpy as np

from src.bball.config import load_config
from src.bball.video.homography import draw_court, refine_homography
from src.bball.video.register import build_ref, register

cfg = load_config()["video"]
cap = cv2.VideoCapture(cfg["path"])


def grab(n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    return cap.read()[1]


refs = [build_ref(r["name"], r["frame"], grab(r["frame"]), r["picks"]) for r in cfg["references"]]
tiles = []
for n in (1700, 2900, 6900, 9300, 10900, 5700, 6100, 11300):
    fr = grab(n)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    H, name, inl = register(gray, refs, min_inliers=15)
    Hr, s = refine_homography(gray, H, stages=((3.0, 12.0), (1.5, 6.0)), reg=2e-3)
    t = cv2.resize(draw_court(fr, Hr, (0, 255, 0)), (640, 360))
    cv2.putText(t, f"{n} in={inl} s={s:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    tiles.append(t)
cv2.imwrite("results/_margin_sheet.jpg", np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))

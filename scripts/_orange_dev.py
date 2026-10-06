"""Development helper: HSV statistics of the box centre for detections judged by eye."""
import cv2
import numpy as np
import pandas as pd

from src.bball.analytics.ball import OVERLAY_Y, candidate_table, drop_static

balls = pd.read_csv("results/video/balls.csv")
frames = pd.read_csv("results/video/frames.csv")
cand = candidate_table(balls, frames)
good = pd.concat([drop_static(g) for _, g in cand[cand.cy < OVERLAY_Y].groupby("shot")])
hi = good[good.conf >= 0.2].sort_values("conf", ascending=False)
known_bad = {762, 919, 770, 155, 503}            # shoes, judged by eye in the contact sheet
cap = cv2.VideoCapture("data/video/2022Game4Q3.mp4")
rows = []
for r in hi.itertuples():
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
    fr = cap.read()[1]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    x, y = int(r.cx), int(r.cy)
    hw = max(3, int(r.w * 0.25))
    patch = hsv[max(0, y - hw):y + hw + 1, max(0, x - hw):x + hw + 1].reshape(-1, 3)
    frac = np.mean((patch[:, 0] >= 3) & (patch[:, 0] <= 20) & (patch[:, 1] >= 90) & (patch[:, 2] >= 50) & (patch[:, 2] <= 235))
    rows.append((int(r.frame), round(r.conf, 2), int(np.median(patch[:, 0])), int(np.median(patch[:, 1])),
                 int(np.median(patch[:, 2])), round(float(frac), 2), "SHOE" if int(r.frame) in known_bad else ""))
print(pd.DataFrame(rows, columns=["frame", "conf", "H", "S", "V", "orange_frac", "judged"]).to_string(index=False))

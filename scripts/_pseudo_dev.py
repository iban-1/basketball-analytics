"""Development experiment: how many high-confidence ball detections exist, and are they real?"""
import cv2
import numpy as np
import pandas as pd

from src.bball.analytics.ball import OVERLAY_Y, candidate_table, drop_static

balls = pd.read_csv("results/video/balls.csv")
frames = pd.read_csv("results/video/frames.csv")
people = pd.read_csv("results/video/people.csv")
cand = candidate_table(balls, frames)
n_frames = int(frames.mapped.sum())
print("mapped frames so far:", n_frames, "| candidates:", len(cand))
good = []
for shot, g in cand[cand.cy < OVERLAY_Y].groupby("shot"):
    good.append(drop_static(g))
good = pd.concat(good)
for thr in (0.15, 0.25, 0.35, 0.5):
    hi = good[good.conf >= thr]
    print(f"conf >= {thr}: {len(hi)} detections in {hi.groupby(['shot', 'frame']).ngroups} frames "
          f"({hi.groupby(['shot', 'frame']).ngroups / n_frames:.1%} of mapped frames)")
pool = good[good.conf >= 0.2]
hi = pool.sample(min(30, len(pool)), random_state=5)
cap = cv2.VideoCapture("data/video/2022Game4Q3.mp4")
tiles = []
for r in hi.itertuples():
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
    fr = cap.read()[1]
    x, y = int(r.cx), int(r.cy)
    cv2.rectangle(fr, (x - 14, y - 14), (x + 14, y + 14), (0, 255, 255), 1)
    crop = cv2.resize(fr[max(0, y - 45):y + 45, max(0, x - 80):x + 80], (320, 180))
    cv2.putText(crop, f"f{int(r.frame)} c={r.conf:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    tiles.append(crop)
while len(tiles) % 5:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite("results/_pseudo_check.jpg", np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))

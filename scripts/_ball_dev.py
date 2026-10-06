"""Development helper: ball selection on balls_v2.csv + a visual check sheet of the SELECTED ball."""
import cv2
import numpy as np
import pandas as pd

from src.bball.analytics.ball import ball_track

res = "results/video/"
balls = pd.read_csv(res + "balls_v2.csv")
frames = pd.read_csv(res + "frames.csv")
people = pd.read_csv(res + "people.csv")
done = set(frames["shot"].unique())
balls = balls[balls["shot"].isin(done)]
tr = ball_track(balls, frames, people)
n_mapped = int(frames["mapped"].sum())
det = tr[~tr.interpolated]
print(f"mapped frames {n_mapped} | frames with a selected ball {len(tr)} ({len(tr) / n_mapped:.0%}) | "
      f"of which detected {len(det)} ({len(det) / n_mapped:.0%}), interpolated {len(tr) - len(det)}")
print("selected conf: median", round(det.conf.median(), 2), "| share >= 0.2:", round((det.conf >= 0.2).mean(), 2))
cap = cv2.VideoCapture("data/video/2022Game4Q3.mp4")
tiles, rows = [], []
for i, r in enumerate(det.sample(36, random_state=21).sort_values("frame").itertuples(), 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
    fr = cap.read()[1]
    x, y = int(r.cx), int(r.cy)
    cv2.circle(fr, (x, y), 24, (0, 255, 255), 2)
    t = cv2.resize(fr[max(0, y - 100):y + 100, max(0, x - 180):x + 180], (360, 200))
    cv2.putText(t, f"#{i} f{int(r.frame)} c={r.conf:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    tiles.append(t)
    rows.append((i, int(r.frame), round(r.conf, 3)))
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite("results/_track_check.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))

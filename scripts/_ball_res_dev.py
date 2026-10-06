"""Development experiment: does a larger model / higher resolution find the basketball better?

For 24 random frames from finished shots, run each setting, take the most confident 'sports ball'
box that is not in the TV overlay bar, and save crops (one sheet per setting) for visual judging.
"""
import sys
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

frames = pd.read_csv("results/video/frames.csv")
sample = frames[frames.mapped].sample(24, random_state=11).sort_values("frame")
cap = cv2.VideoCapture("data/video/2022Game4Q3.mp4")
imgs = []
for f in sample.frame:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    imgs.append((int(f), cap.read()[1]))

settings = [("yolo11s.pt", 1280), ("yolo11m.pt", 1280), ("yolo11s.pt", 1920), ("yolo11m.pt", 1920)]
for name, sz in settings:
    model = YOLO(name)
    tiles, confs, t0 = [], [], time.time()
    for f, fr in imgs:
        r = model.predict(fr, imgsz=sz, conf=0.02, classes=[32], verbose=False)[0]
        b = r.boxes
        xy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
        cf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
        ok = (xy[:, 3] < 640) if len(xy) else np.zeros(0, bool)
        crop = np.zeros((180, 320, 3), np.uint8)
        if ok.any():
            i = np.where(ok)[0][np.argmax(cf[ok])]
            cx, cy = int((xy[i, 0] + xy[i, 2]) / 2), int((xy[i, 1] + xy[i, 3]) / 2)
            cv2.circle(fr := fr.copy(), (cx, cy), 20, (0, 255, 255), 2)
            crop = cv2.resize(fr[max(0, cy - 90):cy + 90, max(0, cx - 160):cx + 160], (320, 180))
            confs.append(float(cf[i]))
            cv2.putText(crop, f"f{f} c={cf[i]:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            confs.append(0.0)
        tiles.append(crop)
    print(f"{name} @ {sz}: {(time.time() - t0) / len(imgs):.2f} s/frame | top-1 conf median {np.median(confs):.2f} | "
          f">=0.25 in {np.mean(np.array(confs) >= 0.25):.0%} of frames", flush=True)
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)])
    cv2.imwrite(f"results/_ball_{name[:-3]}_{sz}.jpg", cv2.resize(sheet, None, fx=0.75, fy=0.75))

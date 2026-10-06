"""Development helper: fit a homography from rough landmark picks and draw the court on a frame."""
import json
import sys

import cv2
import numpy as np

from src.bball.video.court import court_lines, landmark

frame_no, picks_file, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
with open(picks_file, encoding="utf-8") as f:
    picks = json.load(f)                            # {"landmark name": [x, y], ...}
cap = cv2.VideoCapture("data/video/2022Game4Q3.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ok, frame = cap.read()
pix = np.array(list(picks.values()), np.float64)
court = np.array([landmark(k) for k in picks], np.float64)
H_img_from_court, _ = cv2.findHomography(court, pix, 0)
for line in court_lines():
    p = cv2.perspectiveTransform(line.reshape(-1, 1, 2), H_img_from_court).reshape(-1, 2)
    if np.all(np.abs(p) < 1e5):
        cv2.polylines(frame, [p.astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
for (x, y) in pix:
    cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), 2)
cv2.imwrite(out, frame)
print("fit residual px:", np.round(np.linalg.norm(cv2.perspectiveTransform(court.reshape(-1, 1, 2), H_img_from_court).reshape(-1, 2) - pix, axis=1), 1))

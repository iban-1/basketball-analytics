"""Development helper: refine a rough calibration and compare before/after."""
import json
import sys
import time

import cv2
import numpy as np

from src.bball.video.court import landmark
from src.bball.video.homography import alignment_score, draw_court, refine_homography

frame_no, picks_file, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
picks = json.load(open(picks_file, encoding="utf-8"))
cap = cv2.VideoCapture("data/video/2022Game4Q3.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ok, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
court = np.array([landmark(k) for k in picks], np.float64)
pix = np.array(list(picks.values()), np.float64)
H0, _ = cv2.findHomography(court, pix, 0)
s0 = alignment_score(gray, H0)
t = time.time()
H1, s1 = refine_homography(gray, H0)
print(f"alignment before {s0:.2f} -> after {s1:.2f} ({time.time() - t:.1f} s)")
both = np.vstack([draw_court(frame, H0, (0, 0, 255))[250:660], draw_court(frame, H1, (0, 255, 0))[250:660]])
cv2.imwrite(out, both)
np.save(out.replace(".jpg", "_H.npy"), H1)

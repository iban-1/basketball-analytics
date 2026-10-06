"""Development helper: contact sheet of detected interception/pass events for judging by eye.

Shows, for each event, the frame where the NEW holder starts, with the old holder's team colour and the
new holder's. Output: results/_events_sheet_{type}.jpg and a printed key.
"""
import sys

import cv2
import numpy as np
import pandas as pd

kind = sys.argv[1] if len(sys.argv) > 1 else "interception"
res = "results/video/"
ev = pd.read_csv(res + "events.csv")
st = pd.read_csv(res + "states.csv")
people = pd.read_csv(res + "people_labelled.csv")
ball = pd.read_csv(res + "ball_track.csv").set_index("frame")
ev = ev[ev["type"] == kind].sort_values("frame")
cap = cv2.VideoCapture("data/video/2022Game4Q3.mp4")
col = {"light": (255, 200, 60), "dark": (40, 140, 255)}
tiles = []
for i, e in enumerate(ev.itertuples(), 1):
    s_new = st[(st.shot == e.shot) & (st.pid == e.to_pid)].sort_values("start")
    s_new = s_new[s_new.start == e.frame].iloc[0]
    f_mid = int((s_new.start + 0) )
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_mid)
    fr = cap.read()[1]
    for pid, c in ((e.from_pid, (0, 0, 255)), (e.to_pid, (0, 255, 0))):   # old = red box, new = green box
        r = people[(people.pid == pid) & (people.frame == f_mid)]
        for q in r.itertuples():
            cv2.rectangle(fr, (int(q.x1), int(q.y1)), (int(q.x2), int(q.y2)), c, 2)
    if f_mid in ball.index:
        b = ball.loc[f_mid]
        cv2.circle(fr, (int(b.cx), int(b.cy)), 14, (0, 255, 255), 2)
    cx = int(np.clip(ball.loc[f_mid].cx if f_mid in ball.index else 640, 260, 1020))
    tile = cv2.resize(fr[120:640, cx - 260:cx + 260], (520, 360))
    cv2.putText(tile, f"#{i} f{e.frame} {e.team} gap={e.gap_s}s d={e.dist_m}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    tiles.append(tile)
    print(i, "frame", e.frame, "team", e.team, "from", e.from_pid, "to", e.to_pid)
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f"results/_events_sheet_{kind}.jpg", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))

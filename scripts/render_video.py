"""Stage 3: draw the annotated video (object detection boxes, per-player speed and distance,
ball, top-left radar, team passes / interceptions / possession) from the analysis tables.

Run:  python -m scripts.render_video [--max-seconds N]
Writes results/video/annotated.mp4 (mp4v) and, if ffmpeg is installed, annotated_web.mp4 (H.264, for
the dashboard). The videos contain the broadcast footage: LOCAL USE ONLY, git-ignored.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from src.bball.config import load_config, resolve
from src.bball.video import render as R
from src.bball.video.web import to_h264

FPS = 30.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=0)
    args = ap.parse_args()
    cfg = load_config()
    v = cfg["video"]
    res = resolve(cfg["paths"]["results"]) / "video"
    people = pd.read_csv(res / "people_labelled.csv")
    pframes = pd.read_csv(res / "player_frames.csv")
    ball = pd.read_csv(res / "ball_track.csv")
    events = pd.read_csv(res / "events.csv")
    poss = pd.read_csv(res / "possession_frames.csv")
    frames = pd.read_csv(res / "frames.csv")
    shots = pd.read_csv(res / "shots.csv")
    names = {k: v["teams"][k] for k in ("light", "dark")}

    cap = cv2.VideoCapture(str(resolve(v["path"])))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_seconds:
        n = min(n, int(args.max_seconds * FPS))
    w, h = int(cap.get(3)), int(cap.get(4))
    counts = R.cumulative_counts(events, poss, n)
    analysed = np.zeros(n, bool)
    for r in shots[shots["usable"]].itertuples():
        analysed[r.start:min(r.end, n)] = True
    mapped = set(frames.loc[frames["mapped"], "frame"])
    by_frame = {f: g for f, g in people.groupby("frame")}
    motion = pframes.set_index(["pid", "frame"])[["Xs", "Ys", "speed", "dist"]].to_dict("index")
    ball_by = {int(r.frame): r for r in ball.itertuples()}
    poss_team = dict(zip(poss["frame"], poss["team"]))
    ev_by = events.sort_values("frame").groupby("frame").first().to_dict("index")
    banner_until, banner_txt, banner_col = -1, "", (0, 255, 255)

    out_path = res / "annotated.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if analysed[f]:
            dots = []
            for r in by_frame.get(f, pd.DataFrame()).itertuples():
                m = motion.get((r.pid, f))
                spd = m["speed"] * 3.6 if m and m["speed"] == m["speed"] else None
                dist = m["dist"] if m else None
                R.draw_player(frame, np.array([r.x1, r.y1, r.x2, r.y2]), r.team, spd, dist)
                if f in mapped:
                    x, y = (m["Xs"], m["Ys"]) if m and m["Xs"] == m["Xs"] else (r.X, r.Y)
                    dots.append((x, y, r.team))
            b = ball_by.get(f)
            if b is not None:
                R.draw_ball(frame, b.cx, b.cy, bool(b.interpolated))
            R.draw_radar(frame, dots, None, f in mapped)
        else:
            cv2.putText(frame, "not analysed (close-up / other camera)", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        e = ev_by.get(f)
        if e:
            banner_txt = f"{e['type'].upper()}  {names[e['team']]}"
            banner_col = R.TEAM_BGR[e["team"]]
            banner_until = f + int(1.2 * FPS)
        if f <= banner_until:
            R.banner(frame, banner_txt, banner_col)
        cur = {k: {m_: counts[k][m_][f] for m_ in ("passes", "interceptions", "possession")} for k in names}
        R.draw_stats(frame, names, cur, poss_team.get(f))
        vw.write(frame)
        if f % 600 == 0:
            print(f"frame {f}/{n}", flush=True)
    vw.release()
    print("wrote", out_path)
    if to_h264(out_path, res / "annotated_web.mp4"):
        print("wrote", res / "annotated_web.mp4")
    else:
        print("ffmpeg not found: no browser-playable copy was made")


if __name__ == "__main__":
    main()

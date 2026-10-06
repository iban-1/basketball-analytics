"""Stage 1d: run the fine-tuned ball detector over every frame of the usable shots.

Run:  python -m scripts.run_ball_detection [--weights path] [--max-shots N]
Writes results/video/balls_v2.csv (same columns as balls.csv). Checkpointed per shot.
"""
from __future__ import annotations

import argparse
import time

import cv2
import pandas as pd
from ultralytics import YOLO

from src.bball.config import load_config, resolve
from src.bball.video.detect_track import BALL_COLUMNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()
    cfg = load_config()
    res = resolve(cfg["paths"]["results"])
    weights = args.weights or str(res / "ball_model" / "run" / "weights" / "best.pt")
    out = res / "video" / "balls_v2.csv"
    shots = pd.read_csv(res / "video" / "shots.csv")
    done_shots = set(pd.read_csv(out)["shot"].unique()) if out.exists() else set()
    todo = shots[shots["usable"] & ~shots["shot"].isin(done_shots)]
    if args.max_shots:
        todo = todo.head(args.max_shots)
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(resolve(cfg["video"]["path"])))
    t0 = time.time()
    for r in todo.itertuples():
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.start)
        rows = []
        for f in range(r.start, r.end):
            ok, frame = cap.read()
            if not ok:
                break
            res_ = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            for b, c in zip(res_.boxes.xyxy.cpu().numpy(), res_.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = b
                rows.append((r.shot, f, float(c), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        pd.DataFrame(rows, columns=BALL_COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"[{time.time() - t0:6.0f}s] shot {r.shot} ({r.frames} frames): {len(rows)} candidates", flush=True)
    print("done")


if __name__ == "__main__":
    main()

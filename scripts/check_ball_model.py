"""Visual check of the fine-tuned ball detector on RANDOM analysed frames (not training crops).

Run:  python -m scripts.check_ball_model [--n 36] [--seed 7] [--weights path]
For each random mapped frame it runs the model, keeps the most confident box, and writes a numbered
contact sheet of crops to results/ball_model/check_sheet.jpg. A person then judges each crop (real
ball / not the ball / ball not visible); the verdicts are saved in results/ball_model/visual_check.json
by scripts.record_ball_check. This is a hand check, not ground truth at scale.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.bball.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.1)
    args = ap.parse_args()
    cfg = load_config()
    res = resolve(cfg["paths"]["results"])
    weights = args.weights or str(res / "ball_model" / "run" / "weights" / "best.pt")
    frames = pd.read_csv(res / "video" / "frames.csv")
    pool = frames[frames["mapped"]]
    sample = pool.sample(args.n, random_state=args.seed).sort_values("frame")
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(resolve(cfg["video"]["path"])))
    tiles, rows = [], []
    for i, r in enumerate(sample.itertuples(), 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
        fr = cap.read()[1]
        out = model.predict(fr, imgsz=1280, conf=args.conf, verbose=False)[0]
        b = out.boxes
        tile = np.zeros((200, 360, 3), np.uint8)
        top = None
        if len(b):
            k = int(b.conf.argmax())
            x1, y1, x2, y2 = b.xyxy[k].cpu().numpy()
            cx, cy, conf = (x1 + x2) / 2, (y1 + y2) / 2, float(b.conf[k])
            top = (cx, cy, conf)
            cv2.circle(fr, (int(cx), int(cy)), 24, (0, 255, 255), 2)
            tile = cv2.resize(fr[max(0, int(cy) - 100):int(cy) + 100, max(0, int(cx) - 180):int(cx) + 180], (360, 200))
        cv2.putText(tile, f"#{i} f{int(r.frame)}" + (f" c={top[2]:.2f}" if top else " no detection"),
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        tiles.append(tile)
        rows.append({"id": i, "frame": int(r.frame), "conf": None if top is None else round(top[2], 3)})
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    out_dir = res / "ball_model"
    cv2.imwrite(str(out_dir / "check_sheet.jpg"), sheet)
    pd.DataFrame(rows).to_csv(out_dir / "check_sample.csv", index=False)
    print("detections:", sum(r["conf"] is not None for r in rows), "of", len(rows), "| sheet saved")


if __name__ == "__main__":
    main()

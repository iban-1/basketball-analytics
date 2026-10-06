"""Stage 1b: build a self-training dataset for a ball-only detector from the clip's own detections.

Run (after scripts.run_detection has finished):  python -m scripts.make_ball_dataset
Writes data/ball_dataset/ (images + YOLO labels + data.yaml). That folder contains frames of the
user's video: it is git-ignored and must never be committed.
"""
from __future__ import annotations

import shutil

import cv2
import numpy as np
import pandas as pd

from src.bball.analytics.ball import OVERLAY_Y, candidate_table, drop_static
from src.bball.analytics.ball_labels import crop_with_label, is_ball_coloured
from src.bball.config import load_config, resolve, set_seed

POS_CONF = 0.2
MAX_NEAR_PLAYER = 4.0          # metres (pseudo-court) from some tracked person
EVERY_NTH = 3                  # at most one positive per 3 frames inside a shot (neighbours look alike)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    res = resolve(cfg["paths"]["results"]) / "video"
    balls, frames, people = (pd.read_csv(res / f"{n}.csv") for n in ("balls", "frames", "people"))
    cand = candidate_table(balls, frames)
    cand = pd.concat([drop_static(g) for _, g in cand[cand.cy < OVERLAY_Y].groupby("shot")])
    pos_by_frame = people[people.mapped].groupby("frame")[["X", "Y"]].apply(lambda g: g.to_numpy())

    cap = cv2.VideoCapture(str(resolve(cfg["video"]["path"])))
    out = resolve("data/ball_dataset")
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    pos_rows, neg_rows = [], []
    last_pos: dict[int, int] = {}
    for r in cand.sort_values(["shot", "frame"]).itertuples():
        if r.conf < 0.05:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
        ok, fr = cap.read()
        if not ok:
            continue
        coloured = is_ball_coloured(fr, r.cx, r.cy, r.w)
        arr = pos_by_frame.get(r.frame)
        near = arr is not None and len(arr) and np.hypot(arr[:, 0] - r.Xp, arr[:, 1] - r.Yp).min() <= MAX_NEAR_PLAYER
        if r.conf >= POS_CONF and coloured and near:
            if r.frame - last_pos.get(r.shot, -99) >= EVERY_NTH:
                last_pos[r.shot] = r.frame
                pos_rows.append((r, fr))
        elif not coloured and r.conf >= 0.08:
            neg_rows.append((r, fr))
    rng.shuffle(neg_rows)
    neg_rows = neg_rows[: int(1.0 * len(pos_rows))]
    print(f"positives (confident + orange + near a player): {len(pos_rows)} | hard negatives: {len(neg_rows)}")

    n = 0
    for rows, positive in ((pos_rows, True), (neg_rows, False)):
        for r, fr in rows:
            split = "val" if r.shot % 5 == 0 else "train"
            crop, label = crop_with_label(fr, r.cx, r.cy, r.w, r.h, rng=rng, with_ball=positive)
            name = f"s{r.shot}_f{r.frame}_{'p' if positive else 'n'}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), crop)
            (out / "labels" / split / f"{name}.txt").write_text(label + ("\n" if label else ""))
            n += 1
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    for split in ("train", "val"):
        imgs = list((out / "images" / split).glob("*.jpg"))
        pos = sum(1 for p in imgs if p.stem.endswith("_p"))
        print(f"{split}: {len(imgs)} images ({pos} with a ball)")
    print("dataset in", out)


if __name__ == "__main__":
    main()

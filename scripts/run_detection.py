"""Stage 1 (slow): detect, map to the court and track people in every usable shot of the clip.

Run:  python -m scripts.run_detection [--max-shots N] [--restart]
Writes to results/video/: shots.csv, people.csv, balls.csv, frames.csv. Results are appended after
each shot, so an interrupted run resumes where it stopped (use --restart to start over).
Expect roughly 0.4-0.5 s per frame on a laptop CPU.
"""
from __future__ import annotations

import argparse
import time

import cv2
import pandas as pd
from ultralytics import YOLO

from src.bball.config import load_config, resolve, set_seed
from src.bball.video.detect_track import BALL_COLUMNS, DET_COLUMNS, FRAME_COLUMNS, process_shot
from src.bball.video.references import auto_refs, hand_refs, read_frame
from src.bball.video.register import register
from src.bball.video.segments import detect_cuts, segments_from_cuts, video_info


def append(df: pd.DataFrame, path, columns) -> None:
    df.to_csv(path, mode="a", header=not path.exists(), index=False, columns=columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shots", type=int, default=0, help="process only the first N usable shots")
    ap.add_argument("--restart", action="store_true", help="ignore earlier results and start over")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config()
    v = cfg["video"]
    set_seed(cfg["seed"])
    out = resolve(cfg["paths"]["results"]) / "video"
    out.mkdir(parents=True, exist_ok=True)
    if args.restart:
        for f in out.glob("*.csv"):
            f.unlink()

    path = resolve(v["path"])
    info = video_info(path)
    print("Clip:", info)
    cap = cv2.VideoCapture(str(path))
    refs = hand_refs(cap, v) + auto_refs(cap, resolve(cfg["paths"]["results"]) / "auto_references.json")
    print("Reference views:", [r.name for r in refs])

    shots_path = out / "shots.csv"
    if shots_path.exists():
        shots = pd.read_csv(shots_path)
    else:
        cuts = detect_cuts(path, v["cut_threshold"])
        segs = segments_from_cuts(cuts, info["frames"], v["min_shot_frames"])
        rows = []
        for i, (a, b) in enumerate(segs):
            hits = 0
            for frac in (0.25, 0.5, 0.75):
                gray = cv2.cvtColor(read_frame(cap, a + int((b - a) * frac)), cv2.COLOR_BGR2GRAY)
                hits += register(gray, refs, min_inliers=v["min_inliers"])[0] is not None
            rows.append({"shot": i, "start": a, "end": b, "frames": b - a, "seconds": round((b - a) / info["fps"], 1),
                         "court_view_samples": hits, "usable": hits >= 1})
        shots = pd.DataFrame(rows)
        shots.to_csv(shots_path, index=False)
    usable = shots[shots["usable"]]
    print(f"{len(cuts) if 'cuts' in dir() else '?'} cuts -> {len(shots)} shots >= {v['min_shot_frames']} frames; "
          f"{len(usable)} show the court ({usable['frames'].sum()} frames = "
          f"{usable['frames'].sum() / info['fps']:.0f} s of {info['frames'] / info['fps']:.0f} s)")

    done_path = out / "done_shots.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    model = YOLO(v["model"])
    todo = [r for r in usable.itertuples() if str(r.shot) not in done]
    if args.max_shots:
        todo = todo[:args.max_shots]
    total = sum(r.frames for r in todo)
    processed = 0
    for r in todo:
        people, balls, frames = process_shot(model, cap, r.shot, r.start, r.end, refs, v["imgsz"],
                                             v["detect_conf"], v["tracker"], v["min_inliers"],
                                             v["min_alignment"])
        append(people, out / "people.csv", DET_COLUMNS)
        append(balls, out / "balls.csv", BALL_COLUMNS)
        append(frames, out / "frames.csv", FRAME_COLUMNS)
        with open(done_path, "a") as fh:
            fh.write(f"{r.shot}\n")
        processed += r.frames
        eta = (time.time() - t0) / max(processed, 1) * (total - processed)
        print(f"[{time.time() - t0:6.0f}s] shot {r.shot:3d} frames {r.start}-{r.end} "
              f"mapped {frames['mapped'].mean():.0%} | {processed}/{total} frames, ETA {eta / 60:.0f} min", flush=True)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

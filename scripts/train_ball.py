"""Stage 1c: fine-tune a ball-only detector on the self-training dataset (CPU).

Run:  python -m scripts.train_ball [--epochs 25] [--model yolo11s.pt]
Weights go to results/ball_model/run/weights/best.pt (git-ignored: *.pt).
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

from src.bball.config import load_config, resolve, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0, help="limit CPU threads (0 = PyTorch default)")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    model = YOLO(args.model)                      # starts from COCO weights, then adapts to this ball
    model.train(data=str(resolve("data/ball_dataset") / "data.yaml"), imgsz=640, epochs=args.epochs,
                batch=args.batch, device="cpu", workers=0, seed=cfg["seed"], deterministic=True,
                plots=False, project=str(resolve(cfg["paths"]["results"]) / "ball_model"), name="run",
                exist_ok=True, patience=10, close_mosaic=3, verbose=True)
    print("best weights:", resolve(cfg["paths"]["results"]) / "ball_model" / "run" / "weights" / "best.pt")


if __name__ == "__main__":
    main()

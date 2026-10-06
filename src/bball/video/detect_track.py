"""Per-frame detection, court mapping and tracking for one camera shot (basketball).

Order of work for every frame (this order keeps the tracker clean):
  1. YOLO (COCO-pretrained) finds `person` and `sports ball` boxes. The ball is only found at
     low confidence, so its candidates are all kept and resolved later, over time.
  2. The frame is registered to the court (see register.py / homography.py). If that fails the
     frame is "unmapped".
  3. A person counts as a PLAYER candidate only if the feet are on the court: inside the court
     rectangle (+margin) when mapped, otherwise on floor-coloured pixels. This removes the crowd,
     bench, photographers and most staff.
  4. Only those people go to the tracker (BoT-SORT, with camera-motion compensation), which gives
     them IDs that survive short occlusions and pans.
  5. Each box's median shirt colour is recorded, with floor-coloured pixels removed, for teams.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from src.bball.video.court import L, W
from src.bball.video.homography import apply_h, feet_points, refine_homography
from src.bball.video.register import RefView, register

PERSON, BALL = 0, 32
MARGIN_X, MARGIN_Y = 1.5, 1.5       # metres allowed outside the court lines (players, refs)

DET_COLUMNS = ["shot", "frame", "track_id", "conf", "x1", "y1", "x2", "y2", "L", "a", "b",
               "mapped", "X", "Y"]
BALL_COLUMNS = ["shot", "frame", "conf", "cx", "cy", "w", "h"]
FRAME_COLUMNS = ["shot", "frame", "mapped", "inliers", "score", "n_people", "n_court"] + \
    [f"h{i}{j}" for i in range(3) for j in range(3)]


def floor_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """True on floor-coloured pixels: varnished wood (orange-tan) or the green paint/border."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    wood = (h >= 5) & (h <= 28) & (s >= 40) & (v >= 90)
    green = (h >= 35) & (h <= 85) & (s >= 50) & (v >= 40)
    return wood | green


def feet_on_floor(floor: np.ndarray, box: np.ndarray, min_fraction: float = 0.35) -> bool:
    """Fallback when the court is not mapped: is the strip just under the feet floor-coloured?"""
    h, w = floor.shape
    x1, y1, x2, y2 = box.astype(int)
    x1, x2 = max(0, x1), min(w, x2)
    top, bottom = (y2, min(h, y2 + 12)) if y2 + 4 < h else (max(0, y2 - 12), h)
    strip = floor[top:bottom, x1:x2]
    return bool(strip.size and strip.mean() >= min_fraction)


def shirt_colour(frame: np.ndarray, floor: np.ndarray, box: np.ndarray,
                 min_pixels: int = 40) -> tuple[float, float, float]:
    """Median Lab colour of the torso (central 50% width, 15-55% height), floor pixels removed."""
    x1, y1, x2, y2 = box.astype(int)
    w, h = x2 - x1, y2 - y1
    tx1, tx2 = x1 + int(0.25 * w), x2 - int(0.25 * w)
    ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.55 * h)
    crop = frame[max(0, ty1):ty2, max(0, tx1):tx2]
    fm = floor[max(0, ty1):ty2, max(0, tx1):tx2]
    if crop.size == 0:
        return (np.nan, np.nan, np.nan)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~fm.reshape(-1)]
    if len(lab) < min_pixels:
        return (np.nan, np.nan, np.nan)
    med = np.median(lab, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def inside_court(xy: np.ndarray) -> np.ndarray:
    """Boolean mask: court coordinates (N, 2) within the court rectangle plus a small margin."""
    return ((xy[:, 0] >= -MARGIN_X) & (xy[:, 0] <= L + MARGIN_X)
            & (xy[:, 1] >= -MARGIN_Y) & (xy[:, 1] <= W + MARGIN_Y))


def make_tracker(name: str = "botsort.yaml"):
    """A fresh BoT-SORT / ByteTrack tracker (one per shot: IDs cannot survive a camera cut)."""
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(name)))
    return TRACKER_MAP[name.split(".")[0]](cfg)


def map_frame(gray: np.ndarray, refs: list[RefView], boxes: np.ndarray | None, min_inliers: int,
              min_alignment: float) -> tuple[np.ndarray | None, int, float]:
    """Register + refine one frame. Returns (H_img_from_court or None, inliers, alignment)."""
    H, _, inl = register(gray, refs, boxes, min_inliers=min_inliers)
    if H is None:
        return None, inl, float("nan")
    H, score = refine_homography(gray, H, stages=((3.0, 12.0), (1.5, 6.0)), reg=2e-3)
    if not np.isfinite(score) or score < min_alignment:
        return None, inl, score
    return H, inl, score


def process_shot(model, cap: cv2.VideoCapture, shot: int, start: int, end: int, refs: list[RefView],
                 imgsz: int, conf: float, tracker_name: str, min_inliers: int, min_alignment: float,
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detect, map and track every frame in [start, end). Returns (people, ball candidates, frames)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    tracker = make_tracker(tracker_name)
    people, balls, frames = [], [], []
    for f in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=imgsz, conf=conf, classes=[PERSON, BALL], verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else np.zeros(0, int)
        xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p_idx = np.where((cls == PERSON) & (scores >= 0.25))[0]
        H, inl, score = map_frame(gray, refs, xyxy[p_idx] if len(p_idx) else None, min_inliers, min_alignment)
        mapped = H is not None
        floor = floor_mask(frame)

        cand = np.where(cls == PERSON)[0]
        if mapped:
            court_xy = apply_h(np.linalg.inv(H), feet_points(xyxy[cand])) if len(cand) else np.zeros((0, 2))
            keep = inside_court(court_xy) if len(cand) else np.zeros(0, bool)
        else:
            court_xy = np.full((len(cand), 2), np.nan)
            keep = np.array([feet_on_floor(floor, xyxy[i]) for i in cand], bool) if len(cand) else np.zeros(0, bool)
        kept = cand[keep]
        sel = res.boxes[kept].cpu().numpy() if len(kept) else None
        tracks = tracker.update(sel, frame) if sel is not None else tracker.update(res.boxes[:0].cpu().numpy(), frame)
        pos_in_cand = {int(k): j for j, k in enumerate(cand)}
        for t in tracks:
            src = int(kept[int(t[7])])
            j = pos_in_cand[src]
            box = xyxy[src]
            labv = shirt_colour(frame, floor, box)
            people.append((shot, f, int(t[4]), float(scores[src]), *box, *labv, mapped,
                           float(court_xy[j, 0]), float(court_xy[j, 1])))
        for i in np.where(cls == BALL)[0][:6]:
            x1, y1, x2, y2 = xyxy[i]
            balls.append((shot, f, float(scores[i]), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        frames.append((shot, f, mapped, inl, score, int((cls == PERSON).sum()), int(keep.sum()),
                       *(H.flatten().tolist() if mapped else [np.nan] * 9)))
    return (pd.DataFrame(people, columns=DET_COLUMNS), pd.DataFrame(balls, columns=BALL_COLUMNS),
            pd.DataFrame(frames, columns=FRAME_COLUMNS))

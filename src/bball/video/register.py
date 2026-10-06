"""Register a frame to hand-calibrated reference views of the court (feature matching).

The sideline TV camera pans and zooms about a fixed point, so images from it are related by
homographies. SIFT features on the static scene (floor logos, boards, stands) are matched between
a new frame and each reference; RANSAC keeps the consistent matches. Frames from other cameras
(close-ups, behind-the-basket, replays) share few features and simply fail to register.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.bball.video.court import landmark
from src.bball.video.homography import SCOREBOARD_Y, fit_homography


@dataclass
class RefView:
    name: str
    frame_no: int
    H_img_from_court: np.ndarray
    keypoints: np.ndarray        # (N, 2) full-size pixel coordinates
    descriptors: np.ndarray
    fit_residual_px: float


def feature_mask(shape: tuple[int, int], boxes: np.ndarray | None = None) -> np.ndarray:
    """Where features may be taken: not the TV graphics bar, and not on players."""
    m = np.full(shape[:2], 255, np.uint8)
    m[SCOREBOARD_Y:] = 0
    if boxes is not None:
        for x1, y1, x2, y2 in np.asarray(boxes).astype(int):
            m[max(0, y1 - 6):y2 + 6, max(0, x1 - 6):x2 + 6] = 0
    return m


SCALE = 0.5            # features are found on a half-size image (4x faster); points are rescaled


def _sift():
    return cv2.SIFT_create(nfeatures=2000)


def _features(gray: np.ndarray, boxes: np.ndarray | None = None):
    """SIFT keypoints/descriptors on the half-size image; keypoint coordinates in FULL-size pixels."""
    small = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(feature_mask(gray.shape, boxes), small.shape[::-1], interpolation=cv2.INTER_NEAREST)
    kp, desc = _sift().detectAndCompute(small, mask)
    pts = np.array([k.pt for k in kp], np.float32) / SCALE if kp else np.zeros((0, 2), np.float32)
    return pts, desc


def build_ref(name: str, frame_no: int, frame_bgr: np.ndarray, picks: dict[str, list[float]],
              refine: bool = False) -> RefView:
    """Fit court->pixel homography from the hand-picked landmarks and cache SIFT features.

    `refine` then slides the model lines onto the painted lines (see homography.refine_homography).
    `fit_residual_px` is the mean distance between the picks and where the fitted H puts them.
    """
    from src.bball.video.homography import refine_homography
    court = np.array([landmark(k) for k in picks], np.float64)
    pix = np.array(list(picks.values()), np.float64)
    H, _ = cv2.findHomography(court, pix, 0)
    resid = float(np.linalg.norm(cv2.perspectiveTransform(court.reshape(-1, 1, 2), H).reshape(-1, 2) - pix,
                                 axis=1).mean())
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if refine:
        H, _ = refine_homography(gray, H)
    pts, desc = _features(gray)
    return RefView(name, frame_no, H, pts, desc, resid)


def register(gray: np.ndarray, refs: list[RefView], boxes: np.ndarray | None = None,
             ratio: float = 0.75, min_inliers: int = 25) -> tuple[np.ndarray | None, str | None, int]:
    """Best court->pixel homography for `gray` by matching it to every reference.

    Returns (H_img_from_court or None, reference name, inlier count). `None` if no reference
    reaches `min_inliers` consistent matches.
    """
    pts, desc = _features(gray, boxes)
    if desc is None or len(pts) < min_inliers:
        return None, None, 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    best = (None, None, 0)
    for ref in refs:
        pairs = matcher.knnMatch(ref.descriptors, desc, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
        if len(good) < min_inliers:
            continue
        src = np.float32([ref.keypoints[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst = np.float32([pts[m.trainIdx] for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            continue
        n = int(inl.sum())
        if n > best[2]:
            best = (H @ ref.H_img_from_court, ref.name, n)
    if best[2] < min_inliers:
        return None, None, best[2]
    return best

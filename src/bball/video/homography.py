"""Image <-> court mapping for broadcast shots.

Approach (same ideas as the football project, adapted to a panning, zooming TV camera):
1. A homography maps COURT metres to PIXELS for a few hand-calibrated reference frames.
2. For any other frame, an initial guess is carried over (feature matching to a reference, or
   chained camera motion from the previous frame).
3. The guess is REFINED by sliding the projected court markings until they sit on the painted
   white lines (maximising a line-response score). This removes drift and calibration error.
4. The final alignment score says how well the markings sit on the painted lines; frames below
   a threshold are not mapped at all.
Players are mapped with the feet point (bottom-centre of the box) assumed on the floor.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize

from src.bball.video.court import L, W, sample_points

# Four anchor points spread over the court. Refinement moves their IMAGE positions (8 numbers);
# the homography is the exact one through the moved anchors, so the search stays smooth.
ANCHORS = np.array([[L * 0.2, W * 0.2], [L * 0.8, W * 0.2], [L * 0.8, W * 0.8], [L * 0.2, W * 0.8]])
SAMPLES = sample_points(0.12)
SCOREBOARD_Y = 640           # the TV graphic bar starts here; lines below are not floor


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def fit_homography(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares homography src -> dst; also returns per-point residuals (in dst units)."""
    H, _ = cv2.findHomography(np.asarray(src, np.float64), np.asarray(dst, np.float64), 0)
    return H, np.linalg.norm(apply_h(H, src) - np.asarray(dst), axis=1)


def line_response(gray: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Bright thin structures (painted lines) via a top-hat filter, optionally blurred.

    Blurring widens the basin of the score so the optimiser can pull a misplaced overlay
    toward a line that is several pixels away (coarse stage), then a sharp stage settles it.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    resp = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k).astype(np.float32)
    resp[SCOREBOARD_Y:] = 0
    if sigma > 0:
        resp = cv2.GaussianBlur(resp, (0, 0), sigma)
    return resp


def _score(resp: np.ndarray, H_img_from_court: np.ndarray, min_points: int = 40) -> float:
    pts = apply_h(H_img_from_court, SAMPLES)
    h, w = resp.shape
    xi, yi = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < SCOREBOARD_Y) & np.isfinite(pts).all(axis=1)
    if ok.sum() < min_points:
        return 0.0
    return float(resp[yi[ok], xi[ok]].mean())


def alignment_score(gray: np.ndarray, H_img_from_court: np.ndarray, shift_px: int = 14) -> float:
    """Ratio: line response under the projected markings / under the same markings shifted.

    About 1 means no better than chance; clearly above 1 means the overlay sits on real lines.
    NaN if too few marking points fall inside the visible floor.
    """
    resp = line_response(gray, 1.0)
    pts = apply_h(H_img_from_court, SAMPLES)
    h, w = resp.shape

    def mean_at(p: np.ndarray) -> tuple[float, int]:
        xi, yi = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < SCOREBOARD_Y)
        return (float(resp[yi[ok], xi[ok]].mean()) if ok.any() else 0.0), int(ok.sum())

    on, n = mean_at(pts)
    off, _ = mean_at(pts + np.array([0, shift_px]))
    return float("nan") if n < 40 else on / max(off, 1e-6)


def _h_from_offsets(H0: np.ndarray, delta: np.ndarray) -> np.ndarray | None:
    base = apply_h(H0, ANCHORS)
    moved = base + delta.reshape(4, 2)
    H, _ = cv2.findHomography(ANCHORS, moved, 0)
    return H


def refine_homography(gray: np.ndarray, H0: np.ndarray, stages=((8.0, 40.0), (3.0, 12.0), (1.0, 5.0)),
                      reg: float = 2e-4) -> tuple[np.ndarray, float]:
    """Slide the court model onto the painted lines. Returns (refined H, alignment score).

    `stages` = (blur sigma in px, max shift in px per anchor) from coarse to fine. A small
    penalty on the total shift keeps the solution near the starting guess.
    """
    H = H0.copy()
    for sigma, _ in stages:
        resp = line_response(gray, sigma)
        scale = max(resp.max(), 1.0)

        def cost(d: np.ndarray) -> float:
            Ht = _h_from_offsets(H, d)
            if Ht is None:
                return 1e3
            return -_score(resp, Ht) / scale + reg * float(d @ d) / 100.0

        res = minimize(cost, np.zeros(8), method="Powell",
                       options={"xtol": 0.3, "ftol": 1e-4, "maxfev": 900})
        Ht = _h_from_offsets(H, res.x)
        if Ht is not None and cost(res.x) < cost(np.zeros(8)):
            H = Ht
    return H, alignment_score(gray, H)


def draw_court(frame: np.ndarray, H_img_from_court: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """Project the court markings into the image (for visual checks)."""
    from src.bball.video.court import court_lines
    out = frame.copy()
    for line in court_lines():
        p = apply_h(H_img_from_court, line)
        if np.all(np.abs(p) < 1e5):
            cv2.polylines(out, [p.astype(np.int32)], False, color, 2, cv2.LINE_AA)
    return out


# ---------- camera motion between consecutive frames ----------

def mask_for_features(shape: tuple[int, int], boxes: np.ndarray | None,
                      static_regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    """255 where background features may be taken; players and TV graphics are blanked."""
    m = np.full(shape[:2], 255, np.uint8)
    for x1, y1, x2, y2 in static_regions:
        m[y1:y2, x1:x2] = 0
    if boxes is not None:
        for x1, y1, x2, y2 in boxes.astype(int):
            pad = 8
            m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
    return m


def frame_to_frame_h(prev_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray,
                     ) -> tuple[np.ndarray, int]:
    """Homography mapping pixels of `prev` to `gray` from optical flow of static points."""
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=700, qualityLevel=0.01, minDistance=8, mask=mask)
    if p0 is None or len(p0) < 12:
        return np.eye(3), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 12:
        return np.eye(3), 0
    H, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
    if H is None:
        return np.eye(3), 0
    return H, int(inl.sum())


def feet_points(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre of each (x1, y1, x2, y2) box: where the player touches the floor."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]

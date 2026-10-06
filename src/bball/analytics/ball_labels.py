"""Turn the generic detector's confident 'sports ball' hits into training labels for a ball-only model.

The COCO model finds this clip's basketball with high confidence in only a few percent of frames,
but those hits are mostly real. Two cheap tests clean them:
  * COLOUR: the ball is orange-brown. Median HSV hue of the box centre must be orange (3-17 of 179),
    saturated enough and neither dark nor very bright. This rejects purple/pink/yellow/white shoes,
    which were the main false positives when checked by eye.
  * NOT STATIC: dashes painted on the floor and TV graphics repeat in the same place (see ball.py).
These pseudo-labels let a small detector be fine-tuned on this very broadcast. Because they come from
the same model, they are NOT independent ground truth; the fine-tuned model is checked separately
by eye on random frames.
"""
from __future__ import annotations

import cv2
import numpy as np

HUE = (3, 17)
MIN_SAT = 70
VAL = (90, 200)


def is_ball_coloured(frame_bgr: np.ndarray, cx: float, cy: float, w: float) -> bool:
    """True if the centre of the box (a square patch of ~0.5 box widths) looks like an orange ball."""
    hw = max(3, int(w * 0.25))
    x, y = int(cx), int(cy)
    patch = frame_bgr[max(0, y - hw):y + hw + 1, max(0, x - hw):x + hw + 1]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = (float(np.median(hsv[:, i])) for i in range(3))
    return HUE[0] <= h <= HUE[1] and s >= MIN_SAT and VAL[0] <= v <= VAL[1]


def crop_with_label(frame: np.ndarray, cx: float, cy: float, bw: float, bh: float, size: int = 640,
                    rng: np.random.Generator | None = None, with_ball: bool = True,
                    ) -> tuple[np.ndarray, str]:
    """A `size` x `size` crop containing the ball at a random position, plus its YOLO label line.

    The label is empty for a negative crop (`with_ball=False`: centred on a false candidate so the
    model learns what NOT to fire on).
    """
    rng = rng or np.random.default_rng(0)
    h, w = frame.shape[:2]
    margin = 60
    ox = int(np.clip(cx - rng.uniform(margin, size - margin), 0, w - size))
    oy = int(np.clip(cy - rng.uniform(margin, size - margin), 0, h - size))
    crop = frame[oy:oy + size, ox:ox + size]
    if not with_ball:
        return crop, ""
    bx, by = (cx - ox) / size, (cy - oy) / size
    bw_, bh_ = max(bw, 14) / size, max(bh, 14) / size
    return crop, f"0 {bx:.6f} {by:.6f} {bw_:.6f} {bh_:.6f}"

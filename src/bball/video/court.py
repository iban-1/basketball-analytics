"""NBA court model (metres) and its markings.

Court frame (plan view, a proper non-mirrored map of what the main sideline camera sees):
  X runs along the court from the LEFT baseline (X = 0) to the RIGHT baseline (X = 28.65),
  Y runs from the NEAR sideline (camera side, Y = 0) to the FAR sideline (Y = 15.24).
"Left" and "right" are as seen by that camera. Dimensions are NBA (94 x 50 ft).
"""
from __future__ import annotations

import numpy as np

FT = 0.3048
L, W = 94 * FT, 50 * FT                     # 28.65 x 15.24 m
CY = W / 2
HOOP_X = 5.25 * FT                          # hoop centre from the baseline
LANE_HALF = 8 * FT                          # lane is 16 ft wide
LANE_DEPTH = 19 * FT                        # baseline -> free-throw line
FT_RADIUS = 6 * FT
ARC_R = 23.75 * FT                          # three-point arc radius
CORNER_Y = 3 * FT                           # corner three line is 3 ft from the sideline
RESTRICTED_R = 4 * FT
CENTRE_R = 6 * FT


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 60) -> np.ndarray:
    t = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def _basket_lines(mirror: bool) -> list[np.ndarray]:
    """Lane, free-throw circle, three-point line for one basket (left unless mirrored)."""
    lane = np.array([[0, CY - LANE_HALF], [LANE_DEPTH, CY - LANE_HALF],
                     [LANE_DEPTH, CY + LANE_HALF], [0, CY + LANE_HALF]])
    ft_circle = _arc(LANE_DEPTH, CY, FT_RADIUS, -np.pi / 2, np.pi / 2)        # outer half
    # three-point line: corner straights + arc around the hoop
    x_join = HOOP_X + np.sqrt(ARC_R ** 2 - (CY - CORNER_Y) ** 2)
    a = np.arcsin((CY - CORNER_Y) / ARC_R)
    three = np.vstack([[0, CORNER_Y], [x_join, CORNER_Y],
                       _arc(HOOP_X, CY, ARC_R, -a, a), [x_join, W - CORNER_Y], [0, W - CORNER_Y]])
    restricted = _arc(HOOP_X, CY, RESTRICTED_R, -np.pi / 2, np.pi / 2, 30)
    lines = [lane, ft_circle, three, restricted]
    if mirror:
        lines = [np.c_[L - ln[:, 0], ln[:, 1]] for ln in lines]
    return lines


def court_lines() -> list[np.ndarray]:
    """All markings as polylines in court metres."""
    outer = np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]])
    half = np.array([[L / 2, 0], [L / 2, W]])
    centre = _arc(L / 2, CY, CENTRE_R, 0, 2 * np.pi, 80)
    return [outer, half, centre] + _basket_lines(False) + _basket_lines(True)


def landmark(name: str) -> tuple[float, float]:
    """Named court points used for hand calibration (court metres)."""
    pts = {
        "L baseline near lane": (0.0, CY - LANE_HALF),
        "L baseline far lane": (0.0, CY + LANE_HALF),
        "L ft near corner": (LANE_DEPTH, CY - LANE_HALF),
        "L ft far corner": (LANE_DEPTH, CY + LANE_HALF),
        "L arc apex": (HOOP_X + ARC_R, CY),
        "L ft circle top": (LANE_DEPTH + FT_RADIUS, CY),
        "L corner-3 near end": (HOOP_X + np.sqrt(ARC_R ** 2 - (CY - CORNER_Y) ** 2), CORNER_Y),
        "L corner-3 far end": (HOOP_X + np.sqrt(ARC_R ** 2 - (CY - CORNER_Y) ** 2), W - CORNER_Y),
        "R baseline near lane": (L, CY - LANE_HALF),
        "R baseline far lane": (L, CY + LANE_HALF),
        "R ft near corner": (L - LANE_DEPTH, CY - LANE_HALF),
        "R ft far corner": (L - LANE_DEPTH, CY + LANE_HALF),
        "R arc apex": (L - HOOP_X - ARC_R, CY),
        "R ft circle top": (L - LANE_DEPTH - FT_RADIUS, CY),
        "centre": (L / 2, CY),
        "centre line near end": (L / 2, 0.0),
        "centre line far end": (L / 2, W),
        "centre circle near": (L / 2, CY - CENTRE_R),
        "centre circle far": (L / 2, CY + CENTRE_R),
        "centre circle left": (L / 2 - CENTRE_R, CY),
        "centre circle right": (L / 2 + CENTRE_R, CY),
    }
    return pts[name]


def sample_points(step_m: float = 0.1) -> np.ndarray:
    """Dense points (court metres) along every marking, for alignment scoring."""
    out = []
    for line in court_lines():
        for a, b in zip(line[:-1], line[1:]):
            n = max(2, int(np.linalg.norm(b - a) / step_m))
            out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out)

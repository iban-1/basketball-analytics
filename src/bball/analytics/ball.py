"""Pick the real basketball out of many noisy detections.

YOLO (COCO) only finds the basketball at LOW confidence (often 0.05-0.2) and also reports
look-alike round objects, so a single frame cannot say which candidate is the ball. Over time
the true ball moves smoothly while false candidates jump around. We therefore choose, for each
shot, the set of candidates that maximises (confidence reward) minus (penalties for unlikely
motion) with dynamic programming.

Motion is judged in "pseudo-court" metres: the candidate's pixel is mapped to the floor plane
with the frame's court homography. A ball in the air is not on the floor, so these coordinates
are not true positions, but they ARE stable while the camera pans, which is what the
consistency check needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.bball.video.court import L, W
from src.bball.video.homography import apply_h

FPS = 30.0
VMAX = 30.0               # m/s in pseudo-court coordinates (a fast pass is ~20 m/s)
SLACK = 0.8               # metres of tolerance for mapping noise
MAX_GAP = 10              # frames: the ball may be missed for up to ~0.33 s inside one path
BREAK = 0.6               # cost of starting a new path segment (isolated weak candidates lose)
SIZE_MIN, SIZE_MAX = 5, 45          # ball box side in pixels
ASPECT = (0.55, 1.8)


def candidate_table(balls: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Size-filtered candidates with pseudo-court coordinates, on mapped frames only."""
    b = balls[(balls["w"].between(SIZE_MIN, SIZE_MAX)) & (balls["h"].between(SIZE_MIN, SIZE_MAX))].copy()
    ar = b["w"] / b["h"]
    b = b[ar.between(*ASPECT)]
    hcols = [f"h{i}{j}" for i in range(3) for j in range(3)]
    fh = frames[frames["mapped"]].set_index(["shot", "frame"])[hcols]
    b = b.join(fh, on=["shot", "frame"], how="inner")
    xy = np.full((len(b), 2), np.nan)
    for k, (_, row) in enumerate(b.iterrows()):
        H = row[hcols].to_numpy(float).reshape(3, 3)
        xy[k] = apply_h(np.linalg.inv(H), np.array([[row["cx"], row["cy"]]]))[0]
    b["Xp"], b["Yp"] = xy[:, 0], xy[:, 1]
    b = b[b["Xp"].between(-4, L + 4) & b["Yp"].between(-4, W + 12)]
    return b.reset_index(drop=True).drop(columns=hcols)


OVERLAY_Y = 640            # TV graphics bar (scoreboard, trophy icon) starts here
STATIC_WINDOW = 15         # +/- frames in which a candidate must NOT keep reappearing in place
STATIC_REPEATS = 6         # a real ball never sits still this long among the candidates
PLAYER_NEAR = 2.2          # metres: candidates this close to a player get a bonus
NEAR_BONUS = 0.25


def drop_static(c: pd.DataFrame) -> pd.DataFrame:
    """Remove candidates that reappear in the same place again and again.

    Court markings (white dashes in the paint), logos and TV graphics are detected as a "ball"
    over and over at the same spot. Two kinds of "same spot": the same IMAGE pixel (overlays that
    do not move with the camera) and the same COURT position (markings on the floor). A real
    ball moves, so it does not repeat in either.
    """
    c = c.sort_values("frame").reset_index(drop=True)
    frame = c["frame"].to_numpy()
    img = c[["cx", "cy"]].to_numpy()
    crt = c[["Xp", "Yp"]].to_numpy()
    keep = np.ones(len(c), bool)
    lo = np.searchsorted(frame, frame - STATIC_WINDOW, "left")
    hi = np.searchsorted(frame, frame + STATIC_WINDOW, "right")
    for i in range(len(c)):
        sl = slice(lo[i], hi[i])
        same_img = (np.hypot(*(img[sl] - img[i]).T) <= 3.0).sum() - 1
        same_crt = (np.hypot(*(crt[sl] - crt[i]).T) <= 0.3).sum() - 1
        if same_img >= STATIC_REPEATS or same_crt >= STATIC_REPEATS:
            keep[i] = False
    return c[keep].reset_index(drop=True)


def add_player_bonus(c: pd.DataFrame, people: pd.DataFrame | None) -> np.ndarray:
    """Reward for candidates near a tracked person (the ball is almost always near someone)."""
    bonus = np.zeros(len(c))
    if people is None or len(people) == 0:
        return bonus
    pos = people[people["mapped"]].groupby("frame")[["X", "Y"]].apply(lambda g: g.to_numpy())
    for i, (f, xp, yp) in enumerate(zip(c["frame"], c["Xp"], c["Yp"])):
        arr = pos.get(f)
        if arr is not None and len(arr) and np.hypot(arr[:, 0] - xp, arr[:, 1] - yp).min() <= PLAYER_NEAR:
            bonus[i] = NEAR_BONUS
    return bonus


def select_ball(cand: pd.DataFrame, people: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dynamic programme over the candidates of ONE shot; returns the chosen rows."""
    c = cand[cand["cy"] < OVERLAY_Y]
    c = drop_static(c) if len(c) else c
    n = len(c)
    if n == 0:
        return c
    frame = c["frame"].to_numpy()
    xy = c[["Xp", "Yp"]].to_numpy()
    reward = np.sqrt(c["conf"].to_numpy()) + add_player_bonus(c, people)
    F = np.full(n, -np.inf)
    prev = np.full(n, -1)
    by_frame: dict[int, list[int]] = {}
    for i, f in enumerate(frame):
        by_frame.setdefault(int(f), []).append(i)
    best_val, best_idx = -np.inf, -1        # best F among nodes with a SMALLER frame number
    pending: list[int] = []
    cur = None
    for i in range(n):
        if cur is None or frame[i] != cur:
            for j in pending:                # nodes of the previous frame become eligible
                if F[j] > best_val:
                    best_val, best_idx = F[j], j
            pending, cur = [], frame[i]
        start = reward[i] - BREAK
        F[i], prev[i] = start, -1
        if best_idx >= 0 and best_val - BREAK + reward[i] > F[i]:
            F[i], prev[i] = best_val - BREAK + reward[i], best_idx
        for df_ in range(1, MAX_GAP + 1):
            for j in by_frame.get(int(frame[i]) - df_, ()):
                d = float(np.hypot(*(xy[i] - xy[j])))
                lim = VMAX * df_ / FPS + SLACK
                if d > lim:
                    continue
                pen = 0.03 * (df_ - 1) + 0.15 * d / lim
                if F[j] - pen + reward[i] > F[i]:
                    F[i], prev[i] = F[j] - pen + reward[i], j
        pending.append(i)
    end = int(np.argmax(F))
    chosen = []
    while end >= 0:
        chosen.append(end)
        end = prev[end]
    return c.loc[chosen[::-1]].reset_index(drop=True)


def interpolate(track: pd.DataFrame, max_gap: int = MAX_GAP) -> pd.DataFrame:
    """Fill short gaps between chosen detections (linear in pixels). Flags filled rows."""
    rows = []
    t = track.sort_values("frame")
    prev = None
    for r in t.itertuples():
        if prev is not None and 1 < r.frame - prev.frame <= max_gap:
            for f in range(prev.frame + 1, r.frame):
                a = (f - prev.frame) / (r.frame - prev.frame)
                rows.append({"shot": r.shot, "frame": f, "cx": prev.cx + a * (r.cx - prev.cx),
                             "cy": prev.cy + a * (r.cy - prev.cy), "conf": np.nan, "interpolated": True})
        rows.append({"shot": r.shot, "frame": r.frame, "cx": r.cx, "cy": r.cy, "conf": r.conf,
                     "interpolated": False})
        prev = r
    return pd.DataFrame(rows)


def ball_track(balls: pd.DataFrame, frames: pd.DataFrame, people: pd.DataFrame | None = None,
               ) -> pd.DataFrame:
    """Selected, gap-filled ball position per frame for every shot."""
    cand = candidate_table(balls, frames)
    out = []
    for shot, grp in cand.groupby("shot"):
        chosen = select_ball(grp, people[people["shot"] == shot] if people is not None else None)
        if len(chosen):
            out.append(interpolate(chosen))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["shot", "frame", "cx", "cy", "conf", "interpolated"])

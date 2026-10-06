"""Drawing for the annotated video: player boxes with speed/distance, ball, the top-left radar
(top-down court), and the team statistics panel.

Colours (BGR): the light kit is drawn in light blue, the dark kit in orange, referees in grey, so that
both teams stay visible against white and dark jerseys alike.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from src.bball.video.court import L, W, court_lines

TEAM_BGR = {"light": (255, 200, 60), "dark": (40, 140, 255), "other": (170, 170, 170)}
RADAR_W = 320
RADAR_H = int(RADAR_W * W / L)
PAD = 6


def _radar_base() -> np.ndarray:
    """Top-down court picture (floor colour, white lines) at RADAR_W x RADAR_H plus padding."""
    img = np.full((RADAR_H + 2 * PAD, RADAR_W + 2 * PAD, 3), (70, 130, 90), np.uint8)
    for line in court_lines():
        pts = np.array([[PAD + x / L * RADAR_W, PAD + (W - y) / W * RADAR_H] for x, y in line], np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
    return img


_BASE = _radar_base()


def court_to_radar(x: float, y: float) -> tuple[int, int]:
    return int(PAD + x / L * RADAR_W), int(PAD + (W - y) / W * RADAR_H)


def draw_radar(frame: np.ndarray, dots: list[tuple[float, float, str]], ball: tuple[float, float] | None,
               mapped: bool) -> None:
    """Paste the radar into the TOP-LEFT corner of `frame` (in place)."""
    radar = _BASE.copy()
    if mapped:
        for x, y, team in dots:
            if -1 <= x <= L + 1 and -1 <= y <= W + 1:
                p = court_to_radar(x, y)
                cv2.circle(radar, p, 5, TEAM_BGR.get(team, (170, 170, 170)), -1, cv2.LINE_AA)
                cv2.circle(radar, p, 5, (0, 0, 0), 1, cv2.LINE_AA)
        if ball is not None and -1 <= ball[0] <= L + 1 and -1 <= ball[1] <= W + 1:
            cv2.circle(radar, court_to_radar(*ball), 3, (0, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.putText(radar, "court not mapped", (60, RADAR_H // 2 + PAD), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    h, w = radar.shape[:2]
    roi = frame[8:8 + h, 8:8 + w]
    frame[8:8 + h, 8:8 + w] = cv2.addWeighted(roi, 0.25, radar, 0.75, 0)
    cv2.rectangle(frame, (8, 8), (8 + w, 8 + h), (255, 255, 255), 1)


def draw_stats(frame: np.ndarray, names: dict[str, str], counts: dict[str, dict[str, float]],
               holder_team: str | None) -> None:
    """Team statistics panel (top-right): passes, interceptions, possession %."""
    x0, y0, w, h = frame.shape[1] - 250, 8, 242, 104
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    cols = {"light": x0 + 120, "dark": x0 + 185}
    for kit in ("light", "dark"):
        cv2.putText(frame, names[kit], (cols[kit], y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    TEAM_BGR[kit], 2, cv2.LINE_AA)
    rows = [("Passes", "passes", "{:.0f}"), ("Intercept.", "interceptions", "{:.0f}"),
            ("Possession", "possession", "{:.0f}%")]
    for i, (label, key, fmt) in enumerate(rows):
        y = y0 + 46 + i * 24
        cv2.putText(frame, label, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for kit in ("light", "dark"):
            v = counts[kit][key]
            txt = "-" if v != v else fmt.format(v)
            cv2.putText(frame, txt, (cols[kit], y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if holder_team in cols:
        cv2.circle(frame, (cols[holder_team] - 12, y0 + 16), 5, (0, 255, 255), -1)


def draw_player(frame: np.ndarray, box: np.ndarray, team: str, speed_kmh: float | None,
                dist_m: float | None) -> None:
    color = TEAM_BGR.get(team, (170, 170, 170))
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if team in ("light", "dark") and speed_kmh is not None and dist_m is not None:
        label = f"{speed_kmh:.1f} km/h  {dist_m:.0f} m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cx = (x1 + x2) // 2
        cv2.rectangle(frame, (cx - tw // 2 - 2, y2 + 2), (cx + tw // 2 + 2, y2 + th + 8), (20, 20, 20), -1)
        cv2.putText(frame, label, (cx - tw // 2, y2 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_ball(frame: np.ndarray, cx: float, cy: float, interpolated: bool) -> None:
    p = (int(cx), int(cy))
    cv2.circle(frame, p, 11, (0, 255, 255), 2 if not interpolated else 1, cv2.LINE_AA)


def banner(frame: np.ndarray, text: str, color=(0, 255, 255)) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x = (frame.shape[1] - tw) // 2
    y = frame.shape[0] - 110
    cv2.rectangle(frame, (x - 10, y - th - 8), (x + tw + 10, y + 8), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def cumulative_counts(events: pd.DataFrame, poss: pd.DataFrame, n_frames: int) -> dict[str, dict[str, np.ndarray]]:
    """Running totals per frame, per kit: passes, interceptions, possession % so far."""
    out = {}
    for kit in ("light", "dark"):
        d = {}
        for typ, key in (("pass", "passes"), ("interception", "interceptions")):
            ind = np.zeros(n_frames)
            fr = events.loc[(events["type"] == typ) & (events["team"] == kit), "frame"].to_numpy(int)
            np.add.at(ind, fr[fr < n_frames], 1)
            d[key] = np.cumsum(ind)
        ind = np.zeros(n_frames)
        fr = poss.loc[poss["team"] == kit, "frame"].to_numpy(int)
        np.add.at(ind, fr[fr < n_frames], 1)
        d["_poss"] = np.cumsum(ind)
        out[kit] = d
    total = out["light"]["_poss"] + out["dark"]["_poss"]
    for kit in out:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[kit]["possession"] = np.where(total > 0, 100 * out[kit]["_poss"] / total, np.nan)
    return out

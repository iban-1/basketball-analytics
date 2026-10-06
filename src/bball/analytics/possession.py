"""Who has the ball, how long, and what that implies: possession %, passes, interceptions.

Definitions (all approximations from video, stated here so nothing is hidden):
* HOLDER of a frame: the player (stitched track) whose box the ball centre falls into, with a
  little margin; if several, the one whose box centre is closest to the ball.
* HOLDING STATE: the same player is the holder for at least 5 frames (tiny flickers of up to
  3 frames are bridged). Shorter contacts are ignored.
* POSSESSION: from the moment a team's player starts a holding state until the other team starts
  one, that team has the ball (so the flight of a pass still counts for the passer's team).
  Frames before the first holding state of a shot count for nobody.
* PASS: a holding state of player A is followed, within 1.5 s, by a holding state of a DIFFERENT
  player B of the SAME team who is at least 1.5 m from where A last held the ball.
* INTERCEPTION: a holding state of one team is followed, within 1.0 s, by a holding state of the
  OTHER team, where the new holder is more than 4.5 m from both baskets. The distance rule
  deliberately excludes rebounds and the inbound pass after a made basket, which also change
  possession but are not interceptions. This still counts steals out of a dribble and
  intercepted passes; shots that are rebounded far from the basket would be wrongly included.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.bball.video.court import HOOP_X, L, W

FPS = 30.0
MIN_HOLD = 5                # frames for a holding state
CONTEST_MARGIN = 0.3        # opposing player must be this much "less holder-like" or the frame is contested
FLIPFLOP_FRAMES = 45        # an interception answered by the other team within 1.5 s is a flicker, not a steal
MAX_FLICKER = 3             # frames of other/no holder tolerated inside one state
PASS_MAX_GAP = 75           # frames (2.5 s): the ball is often not detected mid-flight
PASS_MIN_DIST = 1.5         # metres between the two holders
STEAL_MAX_GAP = 30          # frames (1.0 s)
BASKET_CLEARANCE = 4.5      # metres: closer than this to a basket => rebound/inbound, not a steal
BASKETS = np.array([[HOOP_X, W / 2], [L - HOOP_X, W / 2]])
TEAMS = ("light", "dark")


def frame_holders(people: pd.DataFrame, ball: pd.DataFrame) -> pd.DataFrame:
    """The holder (pid, team) of every frame that has a ball position."""
    p = people[people["team"].isin(TEAMS) & (people["pid"] != "")]
    m = p.merge(ball[["shot", "frame", "cx", "cy"]], on=["shot", "frame"])
    if m.empty:
        return pd.DataFrame(columns=["shot", "frame", "pid", "team"])
    w = m["x2"] - m["x1"]
    h = m["y2"] - m["y1"]
    inside = ((m["cx"] >= m["x1"] - 0.15 * w) & (m["cx"] <= m["x2"] + 0.15 * w)
              & (m["cy"] >= m["y1"] - 0.10 * h) & (m["cy"] <= m["y1"] + 0.95 * h))
    m = m[inside].copy()
    w, h = m["x2"] - m["x1"], m["y2"] - m["y1"]
    m["metric"] = np.hypot((m["cx"] - (m["x1"] + m["x2"]) / 2) / (0.65 * w),
                           (m["cy"] - (m["y1"] + 0.55 * h)) / (0.5 * h))
    m = m.sort_values("metric")
    best = m.drop_duplicates(["shot", "frame"])
    # Contested frame: the ball also sits convincingly inside a player of the OTHER team (overlapping
    # boxes). Who holds it cannot be told, so the frame gets no holder rather than a coin-flip.
    second = m.groupby(["shot", "frame"]).apply(
        lambda g: g[g["team"] != g["team"].iloc[0]]["metric"].min() if (g["team"] != g["team"].iloc[0]).any() else np.inf,
        include_groups=False).rename("other_metric")
    best = best.join(second, on=["shot", "frame"])
    best = best[best["other_metric"] - best["metric"] >= CONTEST_MARGIN]
    return best[["shot", "frame", "pid", "team"]].sort_values(["shot", "frame"]).reset_index(drop=True)


def holder_states(holders: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Runs of the same holder (>= MIN_HOLD frames, flickers <= MAX_FLICKER bridged).

    `positions` supplies the court position (X, Y) of each pid per frame, for distances.
    """
    pos = positions.dropna(subset=["X", "Y"]).drop_duplicates(["pid", "frame"]).set_index(["pid", "frame"])[["X", "Y"]]
    rows = []
    for shot, g in holders.groupby("shot"):
        run = None
        for r in g.itertuples():
            if run and r.pid == run["pid"] and r.frame - run["end"] <= MAX_FLICKER + 1:
                run["end"] = r.frame
                run["n"] += 1
            else:
                if run:
                    rows.append(run)
                run = {"shot": shot, "pid": r.pid, "team": r.team, "start": r.frame, "end": r.frame, "n": 1}
        if run:
            rows.append(run)
    st = pd.DataFrame(rows)
    if st.empty:
        return pd.DataFrame(columns=["shot", "pid", "team", "start", "end", "n", "X0", "Y0", "X1", "Y1"])
    st = st[st["n"] >= MIN_HOLD].reset_index(drop=True)

    def xy(pid: str, frame: int):
        try:
            v = pos.loc[(pid, frame)]
            return float(v["X"]), float(v["Y"])
        except KeyError:
            return (np.nan, np.nan)

    pts0 = [xy(r.pid, r.start) for r in st.itertuples()]
    pts1 = [xy(r.pid, r.end) for r in st.itertuples()]
    st[["X0", "Y0"]] = pts0
    st[["X1", "Y1"]] = pts1
    return st


def detect_events(states: pd.DataFrame) -> pd.DataFrame:
    """Passes and interceptions between consecutive holding states of each shot."""
    rows = []
    for shot, g in states.groupby("shot"):
        g = g.sort_values("start").reset_index(drop=True)
        for a, b in zip(g.itertuples(), g.iloc[1:].itertuples()):
            gap = b.start - a.end
            if gap < 0 or a.pid == b.pid:
                continue
            d = float(np.hypot(b.X0 - a.X1, b.Y0 - a.Y1)) if np.isfinite([a.X1, a.Y1, b.X0, b.Y0]).all() else np.nan
            if a.team == b.team and gap <= PASS_MAX_GAP and np.isfinite(d) and d >= PASS_MIN_DIST:
                rows.append({"shot": shot, "frame": b.start, "type": "pass", "team": a.team,
                             "from_pid": a.pid, "to_pid": b.pid, "dist_m": round(d, 1), "gap_s": round(gap / FPS, 2)})
            elif a.team != b.team and gap <= STEAL_MAX_GAP and np.isfinite([b.X0, b.Y0]).all():
                if np.hypot(*(BASKETS - [b.X0, b.Y0]).T).min() > BASKET_CLEARANCE:
                    rows.append({"shot": shot, "frame": b.start, "type": "interception", "team": b.team,
                                 "from_pid": a.pid, "to_pid": b.pid, "dist_m": round(d, 1) if np.isfinite(d) else np.nan,
                                 "gap_s": round(gap / FPS, 2)})
    ev = pd.DataFrame(rows, columns=["shot", "frame", "type", "team", "from_pid", "to_pid", "dist_m", "gap_s"])
    # Drop turnover flip-flops: an interception followed (<= 1.5 s) by a change back is ID/box noise.
    drop = set()
    ics = ev[ev["type"] == "interception"].sort_values(["shot", "frame"])
    for (_, a), (j, b) in zip(ics.iloc[:-1].iterrows(), ics.iloc[1:].iterrows()):
        if a["shot"] == b["shot"] and b["frame"] - a["frame"] <= FLIPFLOP_FRAMES and a["team"] != b["team"]:
            drop.update([a.name, b.name])
    return ev.drop(index=list(drop)).reset_index(drop=True)


def possession_frames(states: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Team in possession for every analysed frame: columns shot, frame, team (or None)."""
    out = []
    for r in shots.itertuples():
        frames = np.arange(r.start, r.end)
        team = np.full(len(frames), None, dtype=object)
        g = states[states["shot"] == r.shot].sort_values("start")
        for s in g.itertuples():
            team[frames >= s.start] = s.team
        out.append(pd.DataFrame({"shot": r.shot, "frame": frames, "team": team}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["shot", "frame", "team"])


def possession_share(poss: pd.DataFrame) -> dict[str, float]:
    """Fraction of frames with a known possessor that belong to each team."""
    known = poss[poss["team"].isin(TEAMS)]
    if known.empty:
        return {t: float("nan") for t in TEAMS}
    return {t: float((known["team"] == t).mean()) for t in TEAMS}

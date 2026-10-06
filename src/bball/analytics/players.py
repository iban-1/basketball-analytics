"""Teams, player filtering, track stitching, and per-player distance/speed.

Everything here works in COURT METRES (stable while the camera pans), using the court
coordinates stored with each detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans

from src.bball.video.court import L, W

FPS = 30.0
STRICT_MARGIN = 0.5            # metres outside the lines still counted as "on court"
MIN_TRACK_FRAMES = 15          # ignore flickers shorter than 0.5 s
MIN_INSIDE_SHARE = 0.6         # a real player/referee is inside the lines most of the time
MAX_SPEED = 11.0               # m/s; faster than any basketball player => mapping glitch
STILL_SPEED = 0.35             # m/s; below this a person counts as standing still (jitter dead-band)
SMOOTH_WINDOW = 15             # frames (0.5 s) for Savitzky-Golay smoothing of positions
STITCH_MAX_GAP = 30            # frames (1 s) between the end of one track and the start of the next
STITCH_SPEED = 7.0             # m/s a player may plausibly cover while the tracker lost him
STITCH_SLACK = 0.7             # metres of extra tolerance for mapping noise


def inside_strict(df: pd.DataFrame) -> pd.Series:
    return (df["X"].between(-STRICT_MARGIN, L + STRICT_MARGIN)
            & df["Y"].between(-STRICT_MARGIN, W + STRICT_MARGIN))


# ---------- teams ----------

def track_summary(people: pd.DataFrame) -> pd.DataFrame:
    """One row per (shot, track): frames, inside-the-lines share, median shirt colour."""
    m = people[people["mapped"]].copy()
    m["inside"] = inside_strict(m)
    g = m.groupby(["shot", "track_id"])
    s = g.agg(frames=("frame", "size"), first=("frame", "min"), last=("frame", "max"),
              inside_share=("inside", "mean"), L=("L", "median"), a=("a", "median"), b=("b", "median"),
              n_colour=("L", "count"))
    return s


def assign_teams(summary: pd.DataFrame, seed: int = 42) -> tuple[pd.Series, dict]:
    """Label each track 'light', 'dark' or 'other' by shirt colour (K-means, 3 clusters, Lab space).

    The brightest cluster is the light kit, the darkest is the dark kit, and the middle one is
    'other' (referees, who wear grey). Tracks with too little colour data are 'unknown'.
    """
    ok = summary[(summary["n_colour"] >= 10) & (summary["inside_share"] >= MIN_INSIDE_SHARE)
                 & (summary["frames"] >= MIN_TRACK_FRAMES)]
    X = ok[["L", "a", "b"]].to_numpy()
    km = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(X)
    order = np.argsort(km.cluster_centers_[:, 0])            # dark, other, light
    names = {int(order[0]): "dark", int(order[1]): "other", int(order[2]): "light"}
    label = pd.Series("unknown", index=summary.index)
    label.loc[ok.index] = [names[int(c)] for c in km.labels_]
    centres = {names[i]: km.cluster_centers_[i].round(1).tolist() for i in range(3)}
    return label, centres


# ---------- stitching ----------

def stitch_tracks(people: pd.DataFrame, summary: pd.DataFrame, team: pd.Series) -> pd.Series:
    """Join broken tracks of the same player into chains ("pid"), per shot.

    Track B continues track A if (same team) AND B starts within 1 s after A ends AND the court
    distance between A's last position and B's first position is no more than a player could
    run in that time (plus slack). Greedy, in order of start frame; B joins the closest
    feasible chain. Returns a Series mapping (shot, track_id) -> pid like 's3-p7'.
    """
    pid = pd.Series("", index=summary.index, dtype=object)
    pos = people[people["mapped"]].sort_values("frame")
    first = pos.groupby(["shot", "track_id"]).head(3).groupby(["shot", "track_id"])[["X", "Y"]].mean()
    last = pos.groupby(["shot", "track_id"]).tail(3).groupby(["shot", "track_id"])[["X", "Y"]].mean()
    keep = summary[(summary["frames"] >= MIN_TRACK_FRAMES) & (summary["inside_share"] >= MIN_INSIDE_SHARE)
                   & team.reindex(summary.index).isin(["light", "dark", "other"])]
    for shot, grp in keep.groupby(level="shot"):
        chains: list[dict] = []
        for (_, tid), row in grp.sort_values("first").iterrows():
            t = team[(shot, tid)]
            best, best_d = None, np.inf
            for ch in chains:
                gap = row["first"] - ch["last_frame"]
                if ch["team"] != t or not (0 < gap <= STITCH_MAX_GAP):
                    continue
                d = float(np.hypot(*(first.loc[(shot, tid)].to_numpy() - ch["last_xy"])))
                if d <= STITCH_SPEED * gap / FPS + STITCH_SLACK and d < best_d:
                    best, best_d = ch, d
            if best is None:
                best = {"id": f"s{shot}-p{len(chains)}", "team": t, "last_frame": 0, "last_xy": None}
                chains.append(best)
            best["last_frame"] = int(row["last"])
            best["last_xy"] = last.loc[(shot, tid)].to_numpy()
            pid[(shot, tid)] = best["id"]
    return pid


# ---------- motion ----------

def smooth_player(df: pd.DataFrame, fps: float = FPS) -> pd.DataFrame:
    """Smoothed court position, speed (m/s) and cumulative distance for ONE player chain.

    Frames are merged across the chain's tracks (one row per frame), small gaps (<= 5 frames)
    are bridged linearly, and positions are smoothed with a Savitzky-Golay filter before
    differentiating: frame-to-frame court positions jitter by tens of centimetres (court
    mapping noise), which would turn into absurd speeds if differenced raw. Longer gaps break
    the path, so no distance is counted across a gap.
    """
    d = df.dropna(subset=["X", "Y"]).sort_values("frame").drop_duplicates("frame")[["frame", "X", "Y"]]
    if len(d) < 2:
        return pd.DataFrame(columns=["frame", "Xs", "Ys", "speed", "ok", "dist"])
    frames = np.arange(d["frame"].min(), d["frame"].max() + 1)
    X = np.interp(frames, d["frame"], d["X"])
    Y = np.interp(frames, d["frame"], d["Y"])
    have = np.zeros(len(frames), bool)
    have[np.searchsorted(frames, d["frame"].to_numpy())] = True
    # distance (in frames) to the nearest real observation, to cut paths at long gaps
    idx = np.arange(len(frames))
    last_seen = np.maximum.accumulate(np.where(have, idx, -10 ** 6))
    next_seen = np.minimum.accumulate(np.where(have, idx, 10 ** 6)[::-1])[::-1]
    gap = np.minimum(idx - last_seen, next_seen - idx)
    ok = gap <= 5
    out = pd.DataFrame({"frame": frames, "Xs": np.nan, "Ys": np.nan, "speed": np.nan, "ok": ok})
    edges = np.diff(np.r_[0, ok.astype(int), 0])
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        n = e - s
        if n < SMOOTH_WINDOW:
            continue
        xs = savgol_filter(X[s:e], SMOOTH_WINDOW, 2)
        ys = savgol_filter(Y[s:e], SMOOTH_WINDOW, 2)
        vx = savgol_filter(X[s:e], SMOOTH_WINDOW, 2, deriv=1, delta=1 / fps)
        vy = savgol_filter(Y[s:e], SMOOTH_WINDOW, 2, deriv=1, delta=1 / fps)
        h = SMOOTH_WINDOW // 2
        sp = np.hypot(vx, vy)
        # The filter's last half-window has one-sided data, so those speeds are unreliable. Rather than
        # lose that distance (a big share of a short track), hold the nearest reliable speed there.
        sp[:h], sp[-h:] = sp[h], sp[-h - 1]
        out.loc[s:e - 1, ["Xs", "Ys"]] = np.c_[xs, ys]
        out.loc[s:e - 1, "speed"] = sp
    out.loc[out["speed"] > MAX_SPEED, "speed"] = np.nan      # impossible => discard, never count
    # Dead-band: mapping jitter makes a person who stands still "move" at ~0.2 m/s. Speeds below the
    # threshold count as standing still, so jitter cannot add up to fake distance.
    out.loc[out["speed"] < STILL_SPEED, "speed"] = 0.0
    out["dist"] = (out["speed"].fillna(0) / fps).cumsum()
    return out


def player_table(people: pd.DataFrame, fps: float = FPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-chain stats, per-frame smoothed motion). `people` must have pid and team columns."""
    rows, frames = [], []
    for pid, g in people.groupby("pid"):
        sm = smooth_player(g, fps)
        v = sm["speed"].dropna() if len(sm) else pd.Series(dtype=float)
        if len(v) < fps:                                   # need at least ~1 s of valid motion
            continue
        sm["pid"] = pid
        frames.append(sm)
        rows.append({"pid": pid, "shot": int(g["shot"].iloc[0]), "team": g["team"].iloc[0],
                     "first_frame": int(g["frame"].min()), "last_frame": int(g["frame"].max()),
                     "seconds_tracked": round(len(v) / fps, 1),
                     "distance_m": round(float(v.sum() / fps), 1),
                     "avg_speed_ms": round(float(v.mean()), 2),
                     "avg_speed_kmh": round(float(v.mean() * 3.6), 1),
                     "top_speed_ms": round(float(v.max()), 2),
                     "top_speed_kmh": round(float(v.max() * 3.6), 1),
                     "p90_speed_ms": round(float(v.quantile(0.9)), 2),
                     "inside_lines_share": round(float(((g["X"].between(0, L)) & (g["Y"].between(0, W))).mean()), 2),
                     "median_X": round(float(g["X"].median()), 1), "median_Y": round(float(g["Y"].median()), 1)})
    return pd.DataFrame(rows), (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())

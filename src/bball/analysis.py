"""Team and player analysis computed from one game's NBA.com tables (no video involved).

All functions are pure (tables in, tables out) so they can be tested offline. Everything is derived
from the official box-score tables and the play-by-play log of that game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = [("At the rim (0-4 ft)", 0, 4), ("Paint / short (5-9 ft)", 5, 9),
         ("Mid-range (10 ft to the arc)", 10, 99)]   # 3-pointers are split off by shot value


def _clock_seconds(clock: str) -> float:
    """'PT11M49.00S' -> 709.0 seconds left in the period."""
    s = str(clock)
    try:
        m, rest = s[2:].split("M")
        return float(m) * 60 + float(rest.rstrip("S"))
    except Exception:
        return float("nan")


def elapsed_minutes(period: int, clock: str) -> float:
    """Game minutes elapsed (regulation periods 12 min, overtimes 5 min)."""
    length = 12.0 if period <= 4 else 5.0
    before = 12.0 * min(period - 1, 4) + 5.0 * max(period - 5, 0)
    left = _clock_seconds(clock) / 60
    return before + (length - left)


def four_factors(team: pd.DataFrame, adv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dean Oliver's four factors per team: shooting (eFG%), turnovers, rebounding, free throws.

    `team` is the team-level traditional table (one row per team). OREB% and the ratings come from
    the advanced table when it is given.
    """
    rows = []
    for r in team.itertuples():
        fga, fta, tov = r.fieldGoalsAttempted, r.freeThrowsAttempted, r.turnovers
        row = {"TEAM": r.teamTricode,
               "eFG%": 100 * (r.fieldGoalsMade + 0.5 * r.threePointersMade) / fga if fga else np.nan,
               "TOV% (turnovers per 100 plays)": 100 * tov / (fga + 0.44 * fta + tov) if fga + fta + tov else np.nan,
               "FT rate (FTA per FGA)": fta / fga if fga else np.nan}
        if adv is not None and "offensiveReboundPercentage" in adv:
            a = adv[adv["teamTricode"] == r.teamTricode]
            if len(a):
                row["OREB%"] = 100 * float(a.iloc[0]["offensiveReboundPercentage"])
                row["ORTG"] = float(a.iloc[0]["offensiveRating"])
                row["DRTG"] = float(a.iloc[0]["defensiveRating"])
                row["NETRTG"] = float(a.iloc[0]["netRating"])
                row["PACE"] = float(a.iloc[0]["pace"])
        rows.append(row)
    out = pd.DataFrame(rows).round(1)
    out["FT rate (FTA per FGA)"] = pd.DataFrame(rows)["FT rate (FTA per FGA)"].round(2)
    return out


def scoring_sources(misc: pd.DataFrame, trad_split: pd.DataFrame | None, ft_points: dict[str, int] | None = None) -> pd.DataFrame:
    """Points by source per team: paint, fast break, second chance, off turnovers, and bench points."""
    rows = []
    for r in misc.itertuples():
        row = {"TEAM": r.teamTricode, "Points in the paint": r.pointsPaint, "Fast-break points": r.pointsFastBreak,
               "Second-chance points": r.pointsSecondChance, "Points off turnovers": r.pointsOffTurnovers}
        if trad_split is not None:
            b = trad_split[(trad_split["teamTricode"] == r.teamTricode) & (trad_split["startersBench"] == "Bench")]
            if len(b):
                row["Bench points"] = int(b.iloc[0]["points"])
        rows.append(row)
    return pd.DataFrame(rows)


def score_timeline(pbp: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Home-minus-away margin after every scoring play, with game minutes elapsed."""
    p = pbp.dropna(subset=["scoreHome", "scoreAway"]).copy()
    p = p[(p["actionType"].astype(str) != "period") | (p["period"] == 1)]
    p["home"] = pd.to_numeric(p["scoreHome"], errors="coerce")
    p["away"] = pd.to_numeric(p["scoreAway"], errors="coerce")
    p = p.dropna(subset=["home", "away"])
    p["minute"] = [elapsed_minutes(int(a), b) for a, b in zip(p["period"], p["clock"])]
    p = p[(p["home"].diff().fillna(p["home"]) != 0) | (p["away"].diff().fillna(p["away"]) != 0) | (p.index == p.index[0])]
    out = pd.DataFrame({"minute": p["minute"].to_numpy(), home: p["home"].to_numpy(), away: p["away"].to_numpy()})
    out["margin"] = out[home] - out[away]
    return pd.concat([pd.DataFrame({"minute": [0.0], home: [0.0], away: [0.0], "margin": [0.0]}), out], ignore_index=True)


def scoring_runs(timeline: pd.DataFrame, home: str, away: str, top: int = 5) -> pd.DataFrame:
    """Biggest unanswered scoring runs (points one team scored while the other scored none)."""
    runs, who, pts, start = [], None, 0, 0.0
    prev_h, prev_a = 0.0, 0.0
    for r in timeline.iloc[1:].itertuples():
        h, a = getattr(r, home), getattr(r, away)
        dh, da = h - prev_h, a - prev_a
        prev_h, prev_a = h, a
        side = home if dh > 0 else away if da > 0 else None
        if side is None:
            continue
        gained = dh if side == home else da
        if side == who:
            pts += gained
        else:
            if who is not None:
                runs.append((who, pts, start, r.minute))
            who, pts, start = side, gained, r.minute
    if who is not None:
        runs.append((who, pts, start, timeline["minute"].iloc[-1]))
    df = pd.DataFrame(runs, columns=["TEAM", "Points in the run", "From (min)", "To (min)"])
    return df.sort_values("Points in the run", ascending=False).head(top).round(1).reset_index(drop=True)


def shots(pbp: pd.DataFrame) -> pd.DataFrame:
    """Every field-goal attempt: player, team, result, location (tenths of feet from the hoop), zone."""
    s = pbp[pbp["isFieldGoal"] == 1].copy()
    s["made"] = s["shotResult"].astype(str).str.lower().eq("made")
    s["is3"] = s["shotValue"] == 3
    zone = np.full(len(s), "Mid-range (10 ft to the arc)", dtype=object)
    d = s["shotDistance"].to_numpy()
    for name, lo, hi in ZONES:
        zone[(d >= lo) & (d <= hi)] = name
    zone[s["is3"].to_numpy()] = "Three-pointers"
    s["zone"] = zone
    return s[["period", "clock", "teamTricode", "playerNameI", "xLegacy", "yLegacy", "shotDistance",
              "shotValue", "made", "is3", "zone", "description"]].reset_index(drop=True)


def zone_table(shot_rows: pd.DataFrame) -> pd.DataFrame:
    """Makes, attempts and FG% per shot zone."""
    order = [z[0] for z in ZONES] + ["Three-pointers"]
    g = shot_rows.groupby("zone")["made"].agg(["sum", "count"]).reindex(order).fillna(0).astype(int)
    g.columns = ["Made", "Attempted"]
    g["FG%"] = (100 * g["Made"] / g["Attempted"].replace(0, np.nan)).round(1)
    g.index.name = "Zone"
    return g.reset_index()


def share_of_team(players: pd.DataFrame, team: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """Each player's share (%) of his team's total in the given columns: {table column: label}."""
    t = team.iloc[0]
    rows = []
    for r in players.itertuples():
        row = {"PLAYER": r.nameI}
        for raw, label in cols.items():
            tot = float(t[raw])
            row[label] = round(100 * float(getattr(r, raw)) / tot, 1) if tot else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

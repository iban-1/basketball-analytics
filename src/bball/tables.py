"""Turn raw NBA.com tables into display-ready stat tabs.

Tabs follow the stat groups requested for the dashboard: box score, shooting, advanced,
miscellaneous, hustle and player tracking. Each builder returns {team_tricode: DataFrame}
with one row per player who played plus a final TEAM row. If NBA.com has no table for a
game (older or incomplete games), the builder returns an empty dict and the page says so.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MILE_KM = 1.609344

# (raw column, label, kind). kinds: int, f1 (1 decimal), pct (fraction -> %), min (m:ss -> minutes)
BOX = [("minutes", "MIN", "min"), ("points", "PTS", "int"), ("reboundsTotal", "REB", "int"),
       ("assists", "AST", "int"), ("steals", "STL", "int"), ("blocks", "BLK", "int"),
       ("turnovers", "TOV", "int"), ("foulsPersonal", "PF", "int"),
       ("reboundsOffensive", "OREB", "int"), ("reboundsDefensive", "DREB", "int"),
       ("plusMinusPoints", "+/-", "int")]
SHOOT = [("fieldGoalsMade", "FGM", "int"), ("fieldGoalsAttempted", "FGA", "int"),
         ("fieldGoalsPercentage", "FG%", "pct"), ("threePointersMade", "3PM", "int"),
         ("threePointersAttempted", "3PA", "int"), ("threePointersPercentage", "3P%", "pct"),
         ("freeThrowsMade", "FTM", "int"), ("freeThrowsAttempted", "FTA", "int"),
         ("freeThrowsPercentage", "FT%", "pct")]
ADV = [("offensiveRating", "ORTG", "f1"), ("defensiveRating", "DRTG", "f1"),
       ("netRating", "NETRTG", "f1"), ("usagePercentage", "USG%", "pct"),
       ("trueShootingPercentage", "TS%", "pct"), ("effectiveFieldGoalPercentage", "eFG%", "pct"),
       ("assistPercentage", "AST%", "pct"), ("reboundPercentage", "REB%", "pct"),
       ("offensiveReboundPercentage", "OREB%", "pct"), ("defensiveReboundPercentage", "DREB%", "pct"),
       ("turnoverRatio", "TOV ratio", "f1"), ("pace", "PACE", "f1"), ("possessions", "POSS", "int"),
       ("PIE", "PIE%", "pct")]
MISC = [("pointsPaint", "PITP", "int"), ("pointsFastBreak", "FB-PTS", "int"),
        ("pointsSecondChance", "SCP", "int"), ("pointsOffTurnovers", "PTS OFF TOV", "int"),
        ("blocksAgainst", "BLKA", "int"), ("foulsDrawn", "PFD", "int")]
HUSTLE = [("deflections", "Deflections", "int"), ("chargesDrawn", "Charges drawn", "int"),
          ("contestedShots", "Contested shots", "int"), ("contestedShots2pt", "Contested 2PT", "int"),
          ("contestedShots3pt", "Contested 3PT", "int"), ("screenAssists", "Screen assists", "int"),
          ("screenAssistPoints", "Screen ast pts", "int"),
          ("looseBallsRecoveredOffensive", "Loose balls (off)", "int"),
          ("looseBallsRecoveredDefensive", "Loose balls (def)", "int"),
          ("looseBallsRecoveredTotal", "Loose balls", "int"), ("offensiveBoxOuts", "Box outs (off)", "int"),
          ("defensiveBoxOuts", "Box outs (def)", "int"), ("boxOuts", "Box outs", "int")]
TRACK = [("distance", "Distance (mi)", "f1"), ("speed", "Avg speed (mph)", "f1"),
         ("touches", "Touches", "int"), ("passes", "Passes", "int"),
         ("secondaryAssists", "Secondary ast", "int"), ("freeThrowAssists", "FT ast", "int"),
         ("reboundChancesOffensive", "Reb chances (off)", "int"),
         ("reboundChancesDefensive", "Reb chances (def)", "int"),
         ("reboundChancesTotal", "Reb chances", "int"),
         ("contestedFieldGoalsMade", "Contested FGM", "int"),
         ("contestedFieldGoalsAttempted", "Contested FGA", "int"),
         ("contestedFieldGoalPercentage", "Contested FG%", "pct"),
         ("uncontestedFieldGoalsMade", "Uncontested FGM", "int"),
         ("uncontestedFieldGoalsAttempted", "Uncontested FGA", "int"),
         ("uncontestedFieldGoalsPercentage", "Uncontested FG%", "pct"),
         ("defendedAtRimFieldGoalsMade", "Rim FGM against", "int"),
         ("defendedAtRimFieldGoalsAttempted", "Rim FGA against", "int"),
         ("defendedAtRimFieldGoalPercentage", "Rim FG% against", "pct")]

NOT_AVAILABLE = [
    ("PER (Player Efficiency Rating)", "Not published by NBA.com per game. PIE% (a similar all-in-one rating) is shown instead."),
    ("Win Shares (WS, OWS, DWS)", "Not published by NBA.com; it is a Basketball-Reference formula that needs season totals."),
    ("Drives and drive pass-out %", "Not in NBA.com's per-game box score tables."),
    ("Time of possession, average seconds per touch", "Not in the per-game tables (touches and passes are)."),
    ("Potential assists", "Only secondary assists and free-throw assists are provided."),
    ("Catch-and-shoot vs pull-up, defender distance", "Only contested / uncontested shot splits are provided."),
    ("Contested rebound %", "Only rebound chances are provided."),
]


def minutes_to_float(value) -> float:
    """'42:40' or 'PT42M40.00S' -> 42.67 minutes; blank/NaN -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    m = re.fullmatch(r"PT(\d+)M([\d.]+)S", s)
    if m:
        return int(m.group(1)) + float(m.group(2)) / 60
    m = re.fullmatch(r"(\d+):(\d+)", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60
    try:
        return float(s)
    except ValueError:
        return np.nan


def _convert(series: pd.Series, kind: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce") if kind != "min" else series.map(minutes_to_float)
    if kind == "pct":
        return (s * 100).round(1)
    if kind in ("f1", "min"):
        return s.round(1)
    return s.round(0).astype("Int64")


def _frame(df: pd.DataFrame, spec: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Select + rename + convert the columns of `spec` that exist in `df`."""
    out = pd.DataFrame(index=df.index)
    for raw, label, kind in spec:
        if raw in df.columns:
            out[label] = _convert(df[raw], kind)
    return out


def _team_tables(players: pd.DataFrame | None, teams: pd.DataFrame | None,
                 spec: list[tuple[str, str, str]], extra: pd.DataFrame | None = None,
                 ) -> dict[str, pd.DataFrame]:
    """Per team: sorted player rows + TEAM row."""
    if players is None or len(players) == 0:
        return {}
    if not any(raw == "minutes" for raw, _, _ in spec):   # every tab shows and sorts by MIN
        spec = [("minutes", "MIN", "min")] + list(spec)
    out: dict[str, pd.DataFrame] = {}
    for tri, grp in players.groupby("teamTricode", sort=False):
        body = _frame(grp, spec)
        body.insert(0, "POS", grp["position"].fillna("").to_numpy() if "position" in grp else "")
        body.insert(0, "PLAYER", grp["nameI"].to_numpy())
        if extra is not None:
            body = body.merge(extra, left_on="PLAYER", right_index=True, how="left")
        played = _frame(grp, [("minutes", "MIN", "min")])["MIN"].fillna(0) > 0
        body = body[played.to_numpy()]
        if "MIN" in body:
            body = body.sort_values("MIN", ascending=False)
        if teams is not None and len(teams):
            t = teams[teams["teamTricode"] == tri]
            if len(t):
                total = _frame(t.iloc[[0]], spec)
                total.insert(0, "POS", "")
                total.insert(0, "PLAYER", "TEAM")
                body = pd.concat([body, total], ignore_index=True)
        out[tri] = body.reset_index(drop=True)
    return out


def _get(tables: dict[str, list[pd.DataFrame]], name: str, i: int) -> pd.DataFrame | None:
    frames = tables.get(name) or []
    return frames[i] if len(frames) > i and len(frames[i]) else None


def technical_flagrant(pbp: pd.DataFrame | None) -> pd.DataFrame:
    """Technical and flagrant fouls per player from the play-by-play, indexed by player label."""
    cols = ["Technical fouls", "Flagrant fouls"]
    if pbp is None or "actionType" not in pbp:
        return pd.DataFrame(columns=cols)
    f = pbp[pbp["actionType"].astype(str).str.lower() == "foul"].copy()
    sub = f["subType"].astype(str).str.lower()
    f["tech"] = sub.str.contains("technical")
    f["flag"] = sub.str.contains("flagrant")
    g = f.groupby("playerNameI")[["tech", "flag"]].sum()
    g.columns = cols
    return g.astype(int)


def build_tabs(tables: dict[str, list[pd.DataFrame]]) -> dict[str, dict[str, pd.DataFrame]]:
    """All stat tabs for one game: {tab name: {team: table}}."""
    trad_p, trad_t = _get(tables, "traditional", 0), _get(tables, "traditional", 2)
    adv_p, adv_t = _get(tables, "advanced", 0), _get(tables, "advanced", 1)
    misc_p, misc_t = _get(tables, "misc", 0), _get(tables, "misc", 1)
    hus_p, hus_t = _get(tables, "hustle", 0), _get(tables, "hustle", 1)
    trk_p, trk_t = _get(tables, "player_track", 0), _get(tables, "player_track", 1)
    tf = technical_flagrant(_get(tables, "playbyplay", 0))

    misc_spec = MISC
    tabs = {
        "Box score": _team_tables(trad_p, trad_t, BOX),
        "Shooting": _team_tables(trad_p, trad_t, SHOOT),
        "Advanced": _team_tables(adv_p, adv_t, ADV),
        "Miscellaneous": _team_tables(misc_p, misc_t, misc_spec, extra=tf if len(tf) else None),
        "Hustle": _team_tables(hus_p, hus_t, HUSTLE),
        "Player tracking": _team_tables(trk_p, trk_t, TRACK),
    }
    # Add kilometre / km/h columns next to the miles / mph that NBA.com publishes.
    for tri, df in tabs["Player tracking"].items():
        if "Distance (mi)" in df:
            df.insert(df.columns.get_loc("Distance (mi)") + 1, "Distance (km)",
                      (df["Distance (mi)"] * MILE_KM).round(1))
        if "Avg speed (mph)" in df:
            df.insert(df.columns.get_loc("Avg speed (mph)") + 1, "Avg speed (km/h)",
                      (df["Avg speed (mph)"] * MILE_KM).round(1))
    return tabs


def team_summary(tables: dict[str, list[pd.DataFrame]]) -> pd.DataFrame | None:
    """Team-level game summary (bench points, biggest lead, lead changes, ...)."""
    summ = _get(tables, "summary", 7)
    if summ is None:
        return None
    keep = {"teamTricode": "TEAM", "points": "PTS", "pointsInThePaint": "PITP",
            "pointsFastBreak": "FB-PTS", "pointsSecondChance": "SCP",
            "pointsFromTurnovers": "PTS OFF TOV", "benchPoints": "Bench pts",
            "biggestLead": "Biggest lead", "leadChanges": "Lead changes",
            "timesTied": "Times tied", "biggestScoringRun": "Biggest run"}
    cols = [c for c in keep if c in summ.columns]
    return summ[cols].rename(columns=keep)


def game_info(tables: dict[str, list[pd.DataFrame]]) -> dict:
    """Scoreboard facts for the page header (teams, score by period, arena, attendance)."""
    info: dict = {}
    teams = _get(tables, "summary", 4)
    if teams is not None:
        info["teams"] = teams
    arena, att = _get(tables, "summary", 2), _get(tables, "summary", 1)
    if arena is not None:
        info["arena"] = f"{arena.iloc[0]['arenaName']}, {arena.iloc[0]['arenaCity']}"
    if att is not None:
        info["attendance"] = int(att.iloc[0]["attendance"]) if str(att.iloc[0]["attendance"]).isdigit() else None
    return info


def did_not_play(tables: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """Players on the roster who did not play, with the reason NBA.com gives."""
    t = _get(tables, "traditional", 0)
    if t is None:
        return pd.DataFrame(columns=["PLAYER", "TEAM", "REASON"])
    mins = t["minutes"].map(minutes_to_float).fillna(0)
    dnp = t[mins <= 0]
    return pd.DataFrame({"PLAYER": dnp["nameI"], "TEAM": dnp["teamTricode"],
                         "REASON": dnp["comment"].fillna("")}).reset_index(drop=True)

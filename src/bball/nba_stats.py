"""Official game stats from NBA.com via the free `nba_api` package.

* The code of nba_api is MIT-licensed; the DATA belongs to NBA.com and is covered by NBA.com's
  Terms of Use. Responses are therefore cached only on the user's machine (data/cache/,
  git-ignored) and are never committed.
* The endpoints are unofficial: they can change, rate-limit, or block cloud hosts. Every
  request is retried a few times and paused between calls.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.bball.config import load_config, resolve

# name -> (nba_api class name, ...). Imported lazily so tests that only use cached or
# synthetic tables do not need the network package at import time.
ENDPOINTS = {
    "traditional": "BoxScoreTraditionalV3",
    "advanced": "BoxScoreAdvancedV3",
    "misc": "BoxScoreMiscV3",
    "hustle": "BoxScoreHustleV2",
    "player_track": "BoxScorePlayerTrackV3",
    "summary": "BoxScoreSummaryV3",
    "playbyplay": "PlayByPlayV3",
}


def _cache_dir(game_id: str) -> Path:
    d = resolve(load_config()["paths"]["cache"]) / game_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_cached(game_id: str, name: str) -> list[pd.DataFrame] | None:
    files = sorted(_cache_dir(game_id).glob(f"{name}_*.csv"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not files:
        return None
    return [pd.read_csv(f, dtype={"gameId": str}, keep_default_na=False, na_values=[""])
            for f in files]


def _fetch(name: str, game_id: str, retries: int = 3) -> list[pd.DataFrame]:
    import nba_api.stats.endpoints as E
    cfg = load_config()["nba"]
    cls = getattr(E, ENDPOINTS[name])
    last: Exception | None = None
    for attempt in range(retries):
        try:
            frames = cls(game_id=game_id, timeout=cfg["request_timeout_s"]).get_data_frames()
            time.sleep(cfg["pause_between_requests_s"])
            return frames
        except Exception as exc:  # network errors, rate limits, malformed replies
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"NBA.com request '{name}' for game {game_id} failed: {last}")


def get_endpoint(game_id: str, name: str) -> list[pd.DataFrame]:
    """Tables of one endpoint for one game: from the local cache, else fetched then cached."""
    cached = _read_cached(game_id, name)
    if cached is not None:
        return cached
    frames = _fetch(name, game_id)
    for i, df in enumerate(frames):
        df.to_csv(_cache_dir(game_id) / f"{name}_{i}.csv", index=False)
    return frames


def load_game_tables(game_id: str) -> dict[str, list[pd.DataFrame]]:
    """Every endpoint we use, for one game."""
    return {name: get_endpoint(game_id, name) for name in ENDPOINTS}


def list_games(season: str, season_type: str) -> pd.DataFrame:
    """One row per game of a season (cached): game_id, date, matchup, scores."""
    path = resolve(load_config()["paths"]["cache"]) / f"games_{season}_{season_type.replace(' ', '_')}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"game_id": str})
    from nba_api.stats.endpoints import leaguegamefinder
    cfg = load_config()["nba"]
    raw = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, season_type_nullable=season_type, league_id_nullable="00",
        timeout=cfg["request_timeout_s"]).get_data_frames()[0]
    games = games_from_finder(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(path, index=False)
    return games


def games_from_finder(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the finder's two-rows-per-game table into one row per game (away @ home)."""
    rows = []
    for gid, g in raw.groupby("GAME_ID"):
        away = g[g["MATCHUP"].str.contains("@")]
        home = g[g["MATCHUP"].str.contains("vs.")]
        if len(away) != 1 or len(home) != 1:
            continue
        a, h = away.iloc[0], home.iloc[0]
        rows.append({"game_id": gid, "date": a["GAME_DATE"], "away": a["TEAM_ABBREVIATION"],
                     "home": h["TEAM_ABBREVIATION"], "away_pts": int(a["PTS"]),
                     "home_pts": int(h["PTS"])})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["label"] = (out["date"] + "  " + out["away"] + " " + out["away_pts"].astype(str) + " @ "
                    + out["home"] + " " + out["home_pts"].astype(str))
    return out

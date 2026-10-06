"""Official game stats from NBA.com via the free `nba_api` package.

* The code of nba_api is MIT-licensed; the DATA belongs to NBA.com and is covered by NBA.com's
  Terms of Use. Live responses are therefore cached only on the user's machine (data/cache/,
  git-ignored). The only data in the repository is the small demo sample in data/sample/ (the Finals
  games of 2016-2022), used when NBA.com cannot be reached.
* The endpoints are unofficial: they can change, rate-limit, or block cloud hosts. Every
  request is retried a few times and paused between calls.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.bball.config import ROOT, load_config, resolve

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


SAMPLE_DIR = ROOT / "data" / "sample"     # every Finals game 2016-2022, built by scripts/build_sample.py


def _read_dir(directory: Path, name: str) -> list[pd.DataFrame] | None:
    files = sorted(directory.glob(f"{name}_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not files:
        return None
    return [pd.read_csv(f, dtype={"gameId": str}, keep_default_na=False, na_values=[""])
            for f in files]


def _read_cached(game_id: str, name: str) -> list[pd.DataFrame] | None:
    return _read_dir(_cache_dir(game_id), name)


def sample_games(season: str | None = None) -> pd.DataFrame:
    """The bundled sample's game list (same columns as list_games); empty if no sample is present."""
    path = SAMPLE_DIR / "games.csv"
    if not path.exists():
        return pd.DataFrame()
    g = pd.read_csv(path, dtype={"game_id": str})
    return g if season is None else g[g["season"] == season].reset_index(drop=True)


def load_sample_tables(game_id: str) -> dict[str, list[pd.DataFrame]]:
    """Tables of one bundled sample game, read from data/sample (no network)."""
    out = {}
    for name in ENDPOINTS:
        tables = _read_dir(SAMPLE_DIR / game_id, name)
        if tables is None:
            raise FileNotFoundError(f"game {game_id} is not in the bundled sample ({name} missing)")
        out[name] = tables
    return out


def season_list_cached(season: str, season_type: str = "Playoffs") -> bool:
    return _season_path(season, season_type).exists()


def nba_reachable(timeout: float = 8.0) -> bool:
    """Does NBA.com answer at all? One small request; False on any failure or silence."""
    try:
        from nba_api.stats.endpoints import boxscoresummaryv3
        boxscoresummaryv3.BoxScoreSummaryV3(game_id=load_config()["nba"]["default_game_id"], timeout=timeout).get_data_frames()
        return True
    except Exception:
        return False


def _fetch(name: str, game_id: str, retries: int = 3, timeout: float | None = None) -> list[pd.DataFrame]:
    import nba_api.stats.endpoints as E
    cfg = load_config()["nba"]
    cls = getattr(E, ENDPOINTS[name])
    last: Exception | None = None
    for attempt in range(retries):
        try:
            frames = cls(game_id=game_id, timeout=timeout or cfg["request_timeout_s"]).get_data_frames()
            time.sleep(cfg["pause_between_requests_s"])
            return frames
        except Exception as exc:  # network errors, rate limits, malformed replies
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"NBA.com request '{name}' for game {game_id} failed: {last}")


def get_endpoint(game_id: str, name: str, retries: int = 3, timeout: float | None = None) -> list[pd.DataFrame]:
    """Tables of one endpoint for one game: from the local cache, else fetched then cached."""
    cached = _read_cached(game_id, name)
    if cached is not None:
        return cached
    frames = _fetch(name, game_id, retries=retries, timeout=timeout)
    for i, df in enumerate(frames):
        df.to_csv(_cache_dir(game_id) / f"{name}_{i}.csv", index=False)
    return frames


def load_game_tables(game_id: str) -> dict[str, list[pd.DataFrame]]:
    """Every endpoint we use, for one game.

    The first request is a quick probe (one try, short timeout): when NBA.com does not answer at all,
    as it often does for cloud servers, fail within seconds instead of retrying seven endpoints.
    """
    cfg = load_config()["nba"]
    tables = {"traditional": get_endpoint(game_id, "traditional", retries=1, timeout=cfg["probe_timeout_s"])}
    for name in ENDPOINTS:
        if name not in tables:
            tables[name] = get_endpoint(game_id, name)
    return {name: tables[name] for name in ENDPOINTS}


def _season_path(season: str, season_type: str) -> Path:
    return resolve(load_config()["paths"]["cache"]) / f"games_{season}_{season_type.replace(' ', '_')}.csv"


def list_games(season: str, season_type: str) -> pd.DataFrame:
    """One row per game of a season (cached): game_id, date, matchup, scores."""
    path = _season_path(season, season_type)
    if path.exists():
        return pd.read_csv(path, dtype={"game_id": str})
    from nba_api.stats.endpoints import leaguegamefinder
    cfg = load_config()["nba"]
    raw = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, season_type_nullable=season_type, league_id_nullable="00",
        timeout=cfg["probe_timeout_s"]).get_data_frames()[0]
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

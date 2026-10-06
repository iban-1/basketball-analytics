"""Shared helpers for the dashboard pages: paths, cached loaders, attribution."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.bball.config import load_config, resolve  # noqa: E402
from src.bball.nba_stats import list_games, load_game_tables  # noqa: E402
from src.bball.tables import build_tabs, did_not_play, game_info, team_summary  # noqa: E402

RESULTS = resolve(load_config()["paths"]["results"])
__all__ = ["load_config"]   # re-exported for pages
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2022)]   # playoffs of 2016 ... 2022


def season_label(season: str) -> str:
    """'2021-22' -> '2022 playoffs (2021-22 season)'."""
    return f"{int(season[:4]) + 1} playoffs ({season} season)"


def setup_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} · Basketball analytics", layout="wide")
    st.title(title)


def select_game():
    """Sidebar season + game pickers shared by every page; the choice survives page changes.

    Returns (game row, bundle) or stops the page with a message.
    """
    default_game = load_config()["nba"]["default_game_id"]
    ss = st.session_state
    ss.setdefault("sel_season", SEASONS[-1])

    def pick_season():
        ss["sel_season"] = ss["w_season"]
        ss.pop("sel_game", None)

    def pick_game():
        ss["sel_game"] = ss["w_game"]

    season = st.sidebar.selectbox("Season", SEASONS, index=SEASONS.index(ss["sel_season"]),
                                  format_func=season_label, key="w_season", on_change=pick_season)
    try:
        games = games_for(season)
    except Exception as exc:
        st.error(f"Could not load the game list from NBA.com: {exc}")
        st.stop()
    labels = games["label"].tolist()
    ids = games["game_id"].tolist()
    if ss.get("sel_game") in labels:
        idx = labels.index(ss["sel_game"])
    else:
        idx = ids.index(default_game) if default_game in ids else len(ids) - 1
    label = st.sidebar.selectbox("Game", labels, index=idx, key=f"w_game_{season}",
                                 on_change=lambda: ss.__setitem__("sel_game", ss[f"w_game_{season}"]))
    ss["sel_game"] = label
    game = games[games["label"] == label].iloc[0]
    st.sidebar.caption(f"Game ID {game['game_id']}. {len(games)} playoff games in this season. "
                       "The first time you open a game it is downloaded from NBA.com; after that it "
                       "loads instantly.")
    try:
        bundle = game_bundle(game["game_id"])
    except Exception as exc:
        st.error(f"Could not fetch this game from NBA.com: {exc}\n\nNBA.com sometimes blocks or "
                 "rate-limits requests. Wait a minute and reload, or pick another game.")
        st.stop()
    return game, bundle


def nba_attribution() -> None:
    st.divider()
    st.caption("Game statistics: **NBA.com** (stats.nba.com), fetched with the free `nba_api` "
               "package. The data belongs to NBA.com and is covered by NBA.com's Terms of Use; "
               "it is downloaded to your computer when you open a game and is not part of the "
               "project's repository. This project is not affiliated with or endorsed by the NBA.")


@st.cache_data(show_spinner="Loading the list of games…")
def games_for(season: str) -> pd.DataFrame:
    return list_games(season, "Playoffs")


@st.cache_data(show_spinner="Fetching this game from NBA.com (first time takes ~10 s)…",
               max_entries=20)
def game_bundle(game_id: str) -> dict:
    tables = load_game_tables(game_id)
    return {"tabs": build_tabs(tables), "summary": team_summary(tables),
            "info": game_info(tables), "dnp": did_not_play(tables), "tables": tables}

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

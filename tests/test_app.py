"""Smoke tests: dashboard pages run without an exception."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from src.bball.config import load_config, resolve  # noqa: E402
from src.bball.nba_stats import _cache_dir  # noqa: E402

CACHED_GAME = (_cache_dir(load_config()["nba"]["default_game_id"]) / "traditional_0.csv").exists()
HAVE_VIDEO = (resolve(load_config()["paths"]["results"]) / "video" / "summary.json").exists()


@pytest.mark.skipif(not CACHED_GAME, reason="default game not cached on this machine")
def test_game_stats_page_runs_and_shows_tables():
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.dataframe) >= 6                       # line score + box score for both teams + ...


@pytest.mark.skipif(not CACHED_GAME, reason="default game not cached on this machine")
def test_changing_season_changes_the_game_list():
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=180).run()
    first = at.selectbox[1].options
    try:
        at.selectbox[0].set_value("2015-16").run()
    except Exception:
        pytest.skip("no network for another season")
    assert at.selectbox[1].options != first


@pytest.mark.skipif(not CACHED_GAME, reason="default game not cached on this machine")
@pytest.mark.parametrize("page", ["1_Team_analysis", "2_Player_analysis", "3_Tracking"])
def test_analysis_pages_run(page):
    at = AppTest.from_file(str(APP / f"pages/{page}.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.dataframe) >= 1


@pytest.mark.skipif(not HAVE_VIDEO, reason="video analysis not run")
def test_video_page_runs():
    at = AppTest.from_file(str(APP / "pages/4_Video_analysis.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]

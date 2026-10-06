"""The bundled demo sample (every Finals game 2016-2022) must work with NO network and NO local cache."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.bball import nba_stats  # noqa: E402
from src.bball.config import load_config  # noqa: E402
from src.bball.tables import build_tabs  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
HAVE_SAMPLE = (nba_stats.SAMPLE_DIR / "games.csv").exists()
pytestmark = pytest.mark.skipif(not HAVE_SAMPLE, reason="run scripts/build_sample.py first")


def test_sample_lists_every_finals_game_of_every_season():
    g = nba_stats.sample_games()
    assert g["season"].nunique() == 7
    assert g.groupby("season").size().min() >= 4                      # a Finals has 4 to 7 games
    assert all(gid[7] == "4" for gid in g["game_id"])                 # round 4 = Finals
    assert load_config()["nba"]["default_game_id"] in set(g["game_id"])


def test_every_sample_game_loads_and_builds_all_tabs_offline(monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("the sample must not touch the network")
    monkeypatch.setattr(nba_stats, "_fetch", no_network)
    for gid in nba_stats.sample_games()["game_id"]:
        tabs = build_tabs(nba_stats.load_sample_tables(gid))
        assert len(tabs) == 6 and all(tabs[name] for name in tabs)    # six stat tabs, none empty


def test_site_falls_back_to_the_sample_when_nba_is_unreachable():
    import common
    common.nba_live.clear()
    common.games_for.clear()
    common.game_bundle.clear()
    orig = (common.nba_reachable, common.season_list_cached)
    common.nba_reachable = lambda *a, **k: False
    common.season_list_cached = lambda *a, **k: False
    try:
        at = AppTest.from_file(str(APP / "Home.py"), default_timeout=120).run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Demo sample" in i.value for i in at.info)
        assert len(at.dataframe) >= 6                                  # line score + box score tables render
        assert len(at.selectbox[1].options) == len(nba_stats.sample_games("2021-22"))
    finally:
        common.nba_reachable, common.season_list_cached = orig
        common.nba_live.clear()
        common.games_for.clear()
        common.game_bundle.clear()

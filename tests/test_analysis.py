"""Team/player analysis functions: tiny hand-checked tables, plus a check against a cached real game."""
import numpy as np
import pandas as pd
import pytest

from src.bball import analysis
from src.bball.config import load_config
from src.bball.nba_stats import _cache_dir, load_game_tables


def test_four_factors_hand_computed():
    team = pd.DataFrame([dict(teamTricode="AAA", fieldGoalsMade=40, fieldGoalsAttempted=80, threePointersMade=10,
                              freeThrowsAttempted=20, turnovers=10)])
    ff = analysis.four_factors(team)
    assert ff.loc[0, "eFG%"] == pytest.approx(100 * (40 + 5) / 80, abs=0.05)                   # 56.25 -> 56.2/56.3
    assert ff.loc[0, "TOV% (turnovers per 100 plays)"] == pytest.approx(100 * 10 / (80 + 8.8 + 10), abs=0.06)
    assert ff.loc[0, "FT rate (FTA per FGA)"] == pytest.approx(0.25)


def test_clock_and_elapsed_minutes():
    assert analysis.elapsed_minutes(1, "PT12M00.00S") == pytest.approx(0)
    assert analysis.elapsed_minutes(2, "PT06M00.00S") == pytest.approx(18)
    assert analysis.elapsed_minutes(5, "PT05M00.00S") == pytest.approx(48)          # start of overtime
    assert analysis.elapsed_minutes(5, "PT00M00.00S") == pytest.approx(53)


def _pbp(rows):
    return pd.DataFrame(rows, columns=["period", "clock", "scoreHome", "scoreAway", "actionType"])


def test_timeline_and_runs():
    pbp = _pbp([(1, "PT11M00.00S", 2, 0, "Made Shot"), (1, "PT10M00.00S", 5, 0, "Made Shot"),
                (1, "PT09M00.00S", 5, 2, "Made Shot"), (1, "PT08M00.00S", 5, 2, "Missed Shot"),
                (1, "PT07M00.00S", 5, 4, "Made Shot"), (1, "PT06M00.00S", 5, 7, "Made Shot")])
    tl = analysis.score_timeline(pbp, "HOM", "AWY")
    assert list(tl["margin"]) == [0, 2, 5, 3, 1, -2]
    runs = analysis.scoring_runs(tl, "HOM", "AWY")
    assert list(runs["TEAM"]) == ["AWY", "HOM"] and list(runs["Points in the run"]) == [7, 5]


def test_zone_table_counts():
    pbp = pd.DataFrame({"isFieldGoal": [1, 1, 1, 1, 0], "shotResult": ["Made", "Missed", "Made", "Missed", None],
                        "shotValue": [2, 2, 3, 2, 0], "shotDistance": [2, 7, 25, 15, 0],
                        "period": 1, "clock": "PT10M00.00S", "teamTricode": "AAA", "playerNameI": "X. Y",
                        "xLegacy": 0, "yLegacy": 0, "description": ""})
    z = analysis.zone_table(analysis.shots(pbp)).set_index("Zone")
    assert z.loc["At the rim (0-4 ft)", "Made"] == 1 and z.loc["Paint / short (5-9 ft)", "Attempted"] == 1
    assert z.loc["Three-pointers", "FG%"] == 100.0 and z["Attempted"].sum() == 4


CACHED = (_cache_dir(load_config()["nba"]["default_game_id"]) / "playbyplay_0.csv").exists()


@pytest.mark.skipif(not CACHED, reason="default game not cached on this machine")
def test_real_game_ties_to_official_final_score_and_shots():
    t = load_game_tables(load_config()["nba"]["default_game_id"])
    pbp = t["playbyplay"][0]
    tl = analysis.score_timeline(pbp, "BOS", "GSW")
    assert (tl["BOS"].iloc[-1], tl["GSW"].iloc[-1]) == (97, 107)                    # official final score
    team = t["traditional"][2].set_index("teamTricode")
    sh = analysis.shots(pbp)
    for tri in ("BOS", "GSW"):
        mine = sh[sh["teamTricode"] == tri]
        assert len(mine) == team.loc[tri, "fieldGoalsAttempted"]                    # every attempt found
        assert mine["made"].sum() == team.loc[tri, "fieldGoalsMade"]
        assert mine["is3"].sum() == team.loc[tri, "threePointersAttempted"]
    assert not np.isnan(tl["minute"]).any()

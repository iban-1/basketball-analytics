"""Stats-layer tests on tiny made-up tables (offline), plus a check on a cached real game."""
import numpy as np
import pandas as pd
import pytest

from src.bball.nba_stats import _cache_dir, games_from_finder
from src.bball.tables import (MILE_KM, build_tabs, did_not_play, minutes_to_float,
                              technical_flagrant)


def test_minutes_formats():
    assert minutes_to_float("42:30") == pytest.approx(42.5)
    assert minutes_to_float("PT05M30.00S") == pytest.approx(5.5)
    assert np.isnan(minutes_to_float("")) and np.isnan(minutes_to_float(None))
    assert np.isnan(minutes_to_float(float("nan")))


def _player(tri, name, minutes, pts, fgp=0.5, comment=""):
    return dict(teamTricode=tri, nameI=name, position="G", comment=comment, minutes=minutes,
                points=pts, reboundsTotal=3, assists=2, steals=1, blocks=0, turnovers=1,
                foulsPersonal=2, reboundsOffensive=1, reboundsDefensive=2, plusMinusPoints=4.0,
                fieldGoalsMade=4, fieldGoalsAttempted=8, fieldGoalsPercentage=fgp,
                threePointersMade=1, threePointersAttempted=3, threePointersPercentage=0.333,
                freeThrowsMade=2, freeThrowsAttempted=2, freeThrowsPercentage=1.0)


def _tables():
    players = pd.DataFrame([_player("AAA", "B. Bench", "10:00", 4),
                            _player("AAA", "S. Star", "35:30", 30, fgp=0.456),
                            _player("AAA", "N. Nobody", "", 0, comment="DNP - Coach's Decision"),
                            _player("BBB", "O. Other", "20:15", 12)])
    team = pd.DataFrame([_player("AAA", "", "240:00", 34), _player("BBB", "", "240:00", 12)])
    team = team.drop(columns=["nameI", "position", "comment"])
    track = pd.DataFrame([dict(teamTricode="AAA", nameI="S. Star", position="G", comment="",
                               minutes="35:30", distance=2.0, speed=4.0, touches=50, passes=30)])
    track_t = pd.DataFrame([dict(teamTricode="AAA", distance=10.0, speed=4.1, touches=300,
                                 passes=200)])
    pbp = pd.DataFrame({"actionType": ["Foul", "Foul", "Foul", "Made Shot"],
                        "subType": ["Technical", "Flagrant Type 1", "Personal", ""],
                        "playerNameI": ["S. Star", "S. Star", "S. Star", "S. Star"]})
    return {"traditional": [players, pd.DataFrame(), team], "player_track": [track, track_t],
            "playbyplay": [pbp]}


def test_box_score_tab_orders_players_drops_dnp_and_adds_team_row():
    tabs = build_tabs(_tables())
    aaa = tabs["Box score"]["AAA"]
    assert list(aaa["PLAYER"]) == ["S. Star", "B. Bench", "TEAM"]       # by minutes; DNP removed
    assert aaa.loc[0, "PTS"] == 30 and aaa.loc[2, "PTS"] == 34
    assert aaa.loc[0, "MIN"] == pytest.approx(35.5)


def test_shooting_percentages_are_scaled_to_percent():
    tabs = build_tabs(_tables())
    assert tabs["Shooting"]["AAA"].loc[0, "FG%"] == pytest.approx(45.6)
    assert tabs["Shooting"]["AAA"].loc[0, "FT%"] == pytest.approx(100.0)


def test_missing_tables_give_empty_tabs_not_errors():
    tabs = build_tabs(_tables())
    assert tabs["Advanced"] == {} and tabs["Hustle"] == {}


def test_tracking_gets_km_columns():
    star = build_tabs(_tables())["Player tracking"]["AAA"].iloc[0]
    assert star["Distance (km)"] == pytest.approx(round(2.0 * MILE_KM, 1))
    assert star["Avg speed (km/h)"] == pytest.approx(round(4.0 * MILE_KM, 1))


def test_technical_and_flagrant_counts():
    tf = technical_flagrant(_tables()["playbyplay"][0])
    assert tf.loc["S. Star", "Technical fouls"] == 1 and tf.loc["S. Star", "Flagrant fouls"] == 1
    assert technical_flagrant(None).empty


def test_did_not_play_lists_reason():
    dnp = did_not_play(_tables())
    assert dnp.to_dict("records") == [{"PLAYER": "N. Nobody", "TEAM": "AAA",
                                       "REASON": "DNP - Coach's Decision"}]


def test_games_from_finder_one_row_per_game():
    raw = pd.DataFrame({"GAME_ID": ["1", "1", "2", "2"], "GAME_DATE": ["2022-06-02"] * 2 + ["2022-06-05"] * 2,
                        "MATCHUP": ["BOS @ GSW", "GSW vs. BOS", "BOS @ GSW", "GSW vs. BOS"],
                        "TEAM_ABBREVIATION": ["BOS", "GSW", "BOS", "GSW"], "PTS": [120, 108, 88, 107]})
    g = games_from_finder(raw)
    assert list(g["game_id"]) == ["1", "2"]
    assert g.loc[0, "label"] == "2022-06-02  BOS 120 @ GSW 108"


@pytest.mark.skipif(not (_cache_dir("0042100404") / "traditional_0.csv").exists(),
                    reason="game 0042100404 not cached on this machine")
def test_cached_real_game_matches_the_known_result():
    """2022 Finals Game 4 (GSW 107, BOS 97): team points must equal the sum of players."""
    from src.bball.nba_stats import load_game_tables
    tabs = build_tabs(load_game_tables("0042100404"))
    for tri, expected in (("GSW", 107), ("BOS", 97)):
        box = tabs["Box score"][tri]
        players, team = box[box["PLAYER"] != "TEAM"], box[box["PLAYER"] == "TEAM"].iloc[0]
        assert team["PTS"] == expected == players["PTS"].sum()

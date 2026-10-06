"""Teams, stitching, and motion maths on small synthetic data with known answers."""
import numpy as np
import pandas as pd
import pytest

from src.bball.analytics.players import (assign_teams, player_table, smooth_player, stitch_tracks,
                                         track_summary)

FPS = 30.0


def _track(shot, tid, f0, f1, x0, y0, vx, vy, lab=(180, 130, 130), noise=0.0, seed=0):
    rng = np.random.default_rng(seed + tid)
    frames = np.arange(f0, f1 + 1)
    t = (frames - f0) / FPS
    return pd.DataFrame({"shot": shot, "frame": frames, "track_id": tid, "mapped": True,
                         "X": x0 + vx * t + rng.normal(0, noise, len(t)),
                         "Y": y0 + vy * t + rng.normal(0, noise, len(t)),
                         "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
                         "L": lab[0], "a": lab[1], "b": lab[2]})


def test_constant_speed_is_recovered_despite_mapping_noise():
    df = _track(0, 1, 0, 299, 5.0, 7.0, 3.0, 0.0, noise=0.08).assign(pid="p", team="light")
    sm = smooth_player(df)
    v = sm["speed"].dropna()
    assert v.mean() == pytest.approx(3.0, abs=0.15)
    assert sm["dist"].iloc[-1] == pytest.approx(3.0 * 299 / FPS, rel=0.05)


def test_noise_alone_does_not_create_distance_when_standing_still():
    df = _track(0, 1, 0, 299, 10.0, 5.0, 0.0, 0.0, noise=0.08).assign(pid="p", team="light")
    sm = smooth_player(df)
    assert sm["dist"].iloc[-1] < 1.5          # raw frame-to-frame differencing would give > 20 m


def test_gap_longer_than_5_frames_breaks_the_path():
    a = _track(0, 1, 0, 99, 0.0, 5.0, 2.0, 0.0)
    b = _track(0, 1, 130, 229, 0.0, 5.0, 2.0, 0.0)           # 30-frame hole; position jumps back to 0
    sm = smooth_player(pd.concat([a, b]).assign(pid="p", team="light"))
    hole = sm[(sm.frame > 110) & (sm.frame < 120)]
    assert hole["speed"].isna().all()                          # nothing counted across the hole
    assert sm["dist"].iloc[-1] == pytest.approx(2.0 * (99 + 99) / FPS, rel=0.1)   # edges are kept, not trimmed


def test_impossible_speed_is_discarded():
    df = _track(0, 1, 0, 120, 2.0, 5.0, 30.0, 0.0).assign(pid="p", team="light")   # 30 m/s
    sm = smooth_player(df)
    assert sm["speed"].dropna().empty or sm["speed"].max() <= 11.0


def test_stitching_joins_a_broken_track_of_the_same_team_only():
    people = pd.concat([
        _track(0, 1, 0, 59, 5.0, 7.0, 2.0, 0.0),                       # ends at X = 6.97
        _track(0, 2, 70, 150, 7.4, 7.0, 2.0, 0.0),                     # starts 10 frames later, 0.4 m on: same man
        _track(0, 3, 70, 150, 7.4, 7.0, 2.0, 0.0, lab=(40, 138, 137)),  # same place, other team
        _track(0, 4, 70, 150, 20.0, 7.0, 2.0, 0.0),                    # far away: someone else
    ])
    summ = track_summary(people)
    team = pd.Series({(0, 1): "light", (0, 2): "light", (0, 3): "dark", (0, 4): "light"})
    team.index = pd.MultiIndex.from_tuples(team.index, names=["shot", "track_id"])
    pid = stitch_tracks(people, summ, team)
    assert pid[(0, 1)] == pid[(0, 2)]
    assert pid[(0, 3)] != pid[(0, 1)] and pid[(0, 4)] != pid[(0, 1)]


def test_team_assignment_separates_light_dark_and_referee_grey():
    rows = []
    rng = np.random.default_rng(0)
    for tid in range(30):
        lab = (185, 133, 130) if tid < 10 else (35, 137, 137) if tid < 25 else (90, 135, 132)
        rows.append(_track(0, tid, 0, 40, 10.0, 7.0, 0.5, 0.0,
                           lab=tuple(np.array(lab) + rng.normal(0, 2, 3))))
    summ = track_summary(pd.concat(rows))
    team, _ = assign_teams(summ)
    assert (team.loc[(0, 0):(0, 9)] == "light").all()
    assert (team.loc[(0, 10):(0, 24)] == "dark").all()
    assert (team.loc[(0, 25):(0, 29)] == "other").all()


def test_player_table_reports_distance_and_speed():
    df = _track(0, 1, 0, 299, 3.0, 7.0, 4.0, 0.0).assign(pid="s0-p0", team="dark")
    stats, frames = player_table(df)
    r = stats.iloc[0]
    assert r["pid"] == "s0-p0" and r["team"] == "dark"
    assert r["avg_speed_ms"] == pytest.approx(4.0, abs=0.1) and r["avg_speed_kmh"] == pytest.approx(14.4, abs=0.4)
    assert r["distance_m"] == pytest.approx(4.0 * 299 / FPS * (len(frames.dropna()) / 300), rel=0.15)

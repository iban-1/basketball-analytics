"""Possession / pass / interception logic on small synthetic scenes with known answers."""
import numpy as np
import pandas as pd
import pytest

from src.bball.analytics.possession import (detect_events, frame_holders, holder_states,
                                            possession_frames, possession_share)


def _person(shot, frame, pid, team, x, y, X, Y, w=40, h=100):
    return dict(shot=shot, frame=frame, pid=pid, team=team, x1=x - w / 2, x2=x + w / 2,
                y1=y - h, y2=y, X=X, Y=Y)


def _scene(holders):
    """holders: list of (pid, team, frame_start, frame_end, X, Y). Ball sits in the holder's box."""
    people, ball = [], []
    for pid, team, f0, f1, X, Y in holders:
        for f in range(f0, f1 + 1):
            people.append(_person(0, f, pid, team, 300 + 10 * X, 600, X, Y))
            ball.append(dict(shot=0, frame=f, cx=300 + 10 * X, cy=560))
    return pd.DataFrame(people), pd.DataFrame(ball)


def _run(holders, extra_people=None):
    people, ball = _scene(holders)
    if extra_people is not None:
        people = pd.concat([people, extra_people])
    h = frame_holders(people, ball)
    st = holder_states(h, people)
    return st, detect_events(st)


def test_pass_between_teammates():
    st, ev = _run([("a", "light", 0, 20, 8.0, 7.0), ("b", "light", 40, 60, 12.0, 7.0)])
    assert len(st) == 2
    assert ev["type"].tolist() == ["pass"] and ev.loc[0, "team"] == "light"
    assert ev.loc[0, "dist_m"] == pytest.approx(4.0)


def test_same_player_id_flicker_is_not_a_pass():
    st, ev = _run([("a", "light", 0, 20, 8.0, 7.0), ("a", "light", 22, 40, 8.2, 7.0)])
    assert len(st) == 1 and ev.empty


def test_two_players_standing_together_is_not_a_pass():
    _, ev = _run([("a", "light", 0, 20, 8.0, 7.0), ("b", "light", 30, 50, 8.5, 7.0)])
    assert ev.empty                                   # only 0.5 m apart: more likely an ID swap


def test_interception_by_other_team_away_from_baskets():
    st, ev = _run([("a", "light", 0, 20, 10.0, 7.0), ("c", "dark", 35, 55, 13.0, 7.0)])
    assert ev["type"].tolist() == ["interception"] and ev.loc[0, "team"] == "dark"


def test_possession_change_next_to_basket_is_not_an_interception():
    _, ev = _run([("a", "light", 0, 20, 6.0, 7.0), ("c", "dark", 35, 55, 2.0, 7.0)])   # rebound under the hoop
    assert ev.empty


def test_slow_change_of_possession_is_not_an_interception():
    _, ev = _run([("a", "light", 0, 20, 10.0, 7.0), ("c", "dark", 100, 120, 14.0, 7.0)])
    assert ev.empty


def test_short_contact_is_ignored():
    st, _ = _run([("a", "light", 0, 20, 10.0, 7.0), ("c", "dark", 22, 24, 13.0, 7.0)])
    assert st["pid"].tolist() == ["a"]


def test_possession_share_counts_frames_after_first_holder():
    people, ball = _scene([("a", "light", 10, 29, 10.0, 7.0), ("c", "dark", 50, 89, 14.0, 7.0)])
    st = holder_states(frame_holders(people, ball), people)
    shots = pd.DataFrame({"shot": [0], "start": [0], "end": [100]})
    poss = possession_frames(st, shots)
    assert poss.loc[poss.frame < 10, "team"].isna().all()           # nobody before the first holder
    share = possession_share(poss)
    assert share["light"] == pytest.approx(40 / 90) and share["dark"] == pytest.approx(50 / 90)
    assert share["light"] + share["dark"] == pytest.approx(1.0)


def test_referees_never_hold_the_ball():
    people, ball = _scene([("r", "other", 0, 20, 10.0, 7.0)])
    assert frame_holders(people, ball).empty

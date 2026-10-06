"""Ball selection: a smoothly moving candidate must beat static look-alikes and random clutter."""
import numpy as np
import pandas as pd

from src.bball.analytics.ball import drop_static, interpolate, select_ball


def _cand(frame, xp, yp, conf, cx=None, cy=None, shot=0):
    return dict(shot=shot, frame=frame, conf=conf, cx=cx if cx is not None else 100 + 10 * xp,
                cy=cy if cy is not None else 300 + 10 * yp, w=14, h=14, Xp=xp, Yp=yp)


def test_moving_ball_beats_stronger_static_clutter():
    rows = []
    for f in range(0, 60):
        rows.append(_cand(f, 5.0 + 0.3 * f, 7.0, 0.12, cx=200 + 9 * f, cy=350))       # true ball, weak
        rows.append(_cand(f, 3.0, 4.0, 0.30, cx=500, cy=420))                          # painted dash: strong but static
    sel = select_ball(pd.DataFrame(rows).sample(frac=1, random_state=0))
    assert len(sel) >= 50
    assert (sel["conf"] == 0.12).mean() > 0.95            # chose the mover, not the static 0.30 hits


def test_static_candidates_are_removed_but_a_bouncing_ball_is_kept():
    static = [_cand(f, 3.0, 4.0, 0.2, cx=500, cy=420) for f in range(40)]
    moving = [_cand(f, 3.0 + 0.4 * f, 4.0, 0.2, cx=100 + 12 * f, cy=420) for f in range(40)]
    kept = drop_static(pd.DataFrame(static + moving))
    assert (kept["cx"] == 500).sum() == 0
    assert len(kept) >= 38      # the mover crosses the static spot once; losing that frame is expected


def test_tv_overlay_region_is_excluded():
    rows = [_cand(f, 10.0 + 0.2 * f, 7.0, 0.5, cx=300 + 5 * f, cy=670) for f in range(30)]   # trophy icon in the score bar
    assert select_ball(pd.DataFrame(rows)).empty


def test_interpolation_fills_short_gaps_only():
    t = pd.DataFrame([dict(shot=0, frame=0, cx=0.0, cy=0.0, conf=0.5),
                      dict(shot=0, frame=5, cx=10.0, cy=0.0, conf=0.5),
                      dict(shot=0, frame=30, cx=60.0, cy=0.0, conf=0.5)])
    out = interpolate(t)
    assert out["frame"].tolist() == [0, 1, 2, 3, 4, 5, 30]
    assert out.loc[out.frame == 2, "cx"].iloc[0] == 4.0 and out.loc[out.frame == 2, "interpolated"].iloc[0]

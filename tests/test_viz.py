"""Drawing helpers run on tiny inputs (no data needed)."""
import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from src.bball import viz  # noqa: E402


def test_tracking_frame_draws_players_ball_ring_and_banner():
    players = pd.DataFrame({"Xs": [5.0, 12.0, 20.0], "Ys": [4.0, 8.0, 10.0], "team": ["light", "dark", "other"],
                            "has_ball": [True, False, False], "speed_kmh": [3.0, 10.0, 0.0]})
    fig = viz.tracking_frame(players, {"light": "AAA", "dark": "BBB"}, "Clip time 1.0 s", "PASS  AAA")
    ax = fig.axes[0]
    assert len(ax.collections) == 3 + 2            # one scatter per team + ring + ball for the holder
    assert any("PASS" in t.get_text() for t in ax.texts)
    assert ax.get_xlim()[1] > 28 and ax.get_ylim()[1] > 15      # whole 28.65 x 15.24 m court in view


def test_shot_map_counts_in_legend():
    shots = pd.DataFrame({"xLegacy": [0, 50, -80], "yLegacy": [10, 60, 120], "made": [True, False, True]})
    fig = viz.shot_map(shots, "t")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Missed (1)", "Made (2)"]

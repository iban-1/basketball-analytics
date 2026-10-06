"""Drawing helpers: an NBA half court in NBA.com shot coordinates (tenths of a foot from the hoop)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Rectangle

LINE = "#cfd3da"


def draw_half_court(ax, colour: str = LINE) -> None:
    """Half court, hoop at (0, 0), baseline y = -52.5, +y towards half court, units 0.1 ft."""
    kw = dict(color=colour, lw=1.4, fill=False)
    ax.add_patch(Circle((0, 0), 7.5, **kw))                                   # hoop
    ax.plot([-30, 30], [-40, -40], color=colour, lw=1.6)                     # backboard
    ax.add_patch(Rectangle((-80, -52.5), 160, 190, **kw))                    # paint
    ax.add_patch(Rectangle((-60, -52.5), 120, 190, **kw))                    # lane lines
    ax.add_patch(Arc((0, 137.5), 120, 120, theta1=0, theta2=180, **kw))      # free-throw circle (top)
    ax.add_patch(Arc((0, 137.5), 120, 120, theta1=180, theta2=360, ls="--", **kw))
    ax.add_patch(Arc((0, 0), 80, 80, theta1=0, theta2=180, **kw))            # restricted area
    ax.plot([-220, -220], [-52.5, 89.5], color=colour, lw=1.4)               # corner threes
    ax.plot([220, 220], [-52.5, 89.5], color=colour, lw=1.4)
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, **kw))         # three-point arc
    ax.add_patch(Rectangle((-250, -52.5), 500, 470, **kw))                   # court edge to half court
    ax.add_patch(Arc((0, 417.5), 120, 120, theta1=180, theta2=360, **kw))   # centre circle
    ax.set_xlim(-260, 260)
    ax.set_ylim(-60, 425)
    ax.set_aspect("equal")
    ax.axis("off")


def shot_map(shots: pd.DataFrame, title: str = "", figsize=(5.2, 5.0)):
    """Made shots (filled green) and misses (red crosses) on a half court."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    draw_half_court(ax)
    miss, make = shots[~shots["made"]], shots[shots["made"]]
    ax.scatter(miss["xLegacy"], miss["yLegacy"], marker="x", s=34, c="#e5484d", lw=1.4, label=f"Missed ({len(miss)})", zorder=3)
    ax.scatter(make["xLegacy"], make["yLegacy"], marker="o", s=44, c="#30a46c", edgecolors="white", lw=0.6,
               label=f"Made ({len(make)})", zorder=4)
    if title:
        ax.set_title(title, color="#9aa0aa", fontsize=11)
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor="#9aa0aa", bbox_to_anchor=(0.5, -0.04))
    return fig


def margin_chart(timeline: pd.DataFrame, home: str, away: str):
    """Score margin over the game; above zero the home team leads."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    x, y = timeline["minute"].to_numpy(), timeline["margin"].to_numpy()
    ax.step(x, y, where="post", color="#8a8f98", lw=1.2)
    ax.fill_between(x, y, 0, where=y >= 0, step="post", color="#3e63dd", alpha=0.55, label=f"{home} ahead")
    ax.fill_between(x, y, 0, where=y <= 0, step="post", color="#f76b15", alpha=0.55, label=f"{away} ahead")
    for q in range(1, 5):
        ax.axvline(12 * q, color="#555a63", lw=0.7, ls=":")
    ax.axhline(0, color="#9aa0aa", lw=0.8)
    ax.set_xlabel("Game minutes", color="#9aa0aa")
    ax.set_ylabel("Margin (points)", color="#9aa0aa")
    ax.tick_params(colors="#9aa0aa")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, labelcolor="#9aa0aa", loc="upper left", ncol=2)
    return fig

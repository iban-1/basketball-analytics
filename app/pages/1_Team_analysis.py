"""Team analysis: how each team won or lost the selected game."""
import pandas as pd
import streamlit as st

import common
from src.bball import analysis, viz

common.setup_page("Team analysis")
game, bundle = common.select_game()
tables = bundle["tables"]
home, away = game["home"], game["away"]
st.header(f"{game['away']} {game['away_pts']} @ {game['home']} {game['home_pts']}")
st.caption(f"{game['date']} · same game picker as the other pages")

trad = tables["traditional"]
team_trad = trad[2] if len(trad) > 2 else None
adv_team = tables["advanced"][1] if len(tables.get("advanced", [])) > 1 else None
misc_team = tables["misc"][1] if len(tables.get("misc", [])) > 1 else None
pbp = tables["playbyplay"][0] if tables.get("playbyplay") else None

# ---------- four factors ----------
st.subheader("The four factors")
st.caption("Dean Oliver's four things that decide basketball games: shoot well (eFG%), do not turn the ball "
           "over (TOV%), grab your own misses (OREB%) and get to the free-throw line (FT rate). "
           "Green marks the team that was better in that factor.")
if team_trad is not None:
    ff = analysis.four_factors(team_trad, adv_team)
    better_high = {"eFG%": True, "TOV% (turnovers per 100 plays)": False, "OREB%": True,
                   "FT rate (FTA per FGA)": True, "ORTG": True, "DRTG": False, "NETRTG": True, "PACE": True}

    def paint(col):
        if col.name not in better_high or col.isna().any():
            return [""] * len(col)
        best = col.max() if better_high[col.name] else col.min()
        return ["background-color: rgba(48,164,108,0.35)" if v == best and col.nunique() > 1 else "" for v in col]

    st.dataframe(ff.style.apply(paint).format(precision=1, subset=[c for c in ff.columns if c not in ("TEAM", "FT rate (FTA per FGA)")])
                 .format(precision=2, subset=["FT rate (FTA per FGA)"]), hide_index=True)

# ---------- scoring sources ----------
st.subheader("Where the points came from")
if misc_team is not None:
    src = analysis.scoring_sources(misc_team, trad[1] if len(trad) > 1 else None)
    cols = [c for c in src.columns if c != "TEAM"]
    long = src.melt(id_vars="TEAM", value_vars=cols, var_name="Source", value_name="Points")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.bar_chart(long, x="Source", y="Points", color="TEAM", stack=False, horizontal=True)
    with c2:
        st.dataframe(src, hide_index=True)
    st.caption("These groups overlap (a fast-break basket in the paint is in both), so they do not add up to the score.")

# ---------- game flow ----------
if pbp is not None and "scoreHome" in pbp:
    st.subheader("Game flow")
    tl = analysis.score_timeline(pbp, home, away)
    st.pyplot(viz.margin_chart(tl, home, away), clear_figure=True)
    lead_home, lead_away = tl["margin"].max(), -tl["margin"].min()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Biggest {home} lead", f"{max(lead_home, 0):.0f}")
    c2.metric(f"Biggest {away} lead", f"{max(lead_away, 0):.0f}")
    c3.metric("Lead changes", int(((tl["margin"].shift().fillna(0) * tl["margin"]) < 0).sum()))
    st.caption("Biggest scoring runs (points scored without an answer from the other team)")
    st.dataframe(analysis.scoring_runs(tl, home, away), hide_index=True)

    # ---------- shot profile ----------
    st.subheader("Shot profile")
    sh = analysis.shots(pbp)
    tabs = st.tabs([home, away])
    for tab, tri in zip(tabs, (home, away)):
        with tab:
            mine = sh[sh["teamTricode"] == tri]
            left, right = st.columns([2, 3])
            with left:
                st.pyplot(viz.shot_map(mine, f"{tri} shots ({len(mine)})"), clear_figure=True)
            with right:
                st.dataframe(analysis.zone_table(mine), hide_index=True)
                st.caption("Zones by shot distance from the play-by-play log; three-pointers are counted separately. "
                           "Shot locations are NBA.com's, rounded to the nearest foot.")
else:
    st.info("NBA.com has no play-by-play for this game, so game flow and shot charts are not shown.")

common.nba_attribution()

"""Player analysis: one player's game in detail, plus a side-by-side comparison."""
import pandas as pd
import streamlit as st

import common
from src.bball import analysis, viz

common.setup_page("Player analysis")
game, bundle = common.select_game()
tables = bundle["tables"]
home, away = game["home"], game["away"]
st.header(f"{game['away']} {game['away_pts']} @ {game['home']} {game['home_pts']}")

def clean(df):
    """Some games carry a ghost row for a player (same id, no minutes): keep only rows with minutes."""
    if df is None:
        return None
    df = df[df["minutes"].notna() & (df["minutes"].astype(str).str.strip() != "")]
    return df.drop_duplicates("personId")


trad_p = tables["traditional"][0]
adv_p = clean(tables["advanced"][0]) if tables.get("advanced") else None
trk_p = clean(tables["player_track"][0]) if tables.get("player_track") else None
hus_p = clean(tables["hustle"][0]) if tables.get("hustle") else None
team_t = tables["traditional"][2]
pbp = tables["playbyplay"][0] if tables.get("playbyplay") else None


def minutes(v) -> float:
    from src.bball.tables import minutes_to_float
    return minutes_to_float(v)


played = trad_p[trad_p["minutes"].map(minutes).fillna(0) > 0].copy()
played["mins"] = played["minutes"].map(minutes)
played = played.sort_values(["teamTricode", "mins"], ascending=[True, False])

team_choice = st.radio("Team", [home, away], horizontal=True)
roster = played[played["teamTricode"] == team_choice]
name = st.selectbox("Player", roster["nameI"].tolist(), format_func=lambda n: f"{n}  ({roster.set_index('nameI').loc[n, 'mins']:.0f} min)")
p = roster[roster["nameI"] == name].iloc[0]
pid = p["personId"]

# ---------- headline numbers ----------
st.subheader(f"{p['firstName']} {p['familyName']}  ·  {team_choice}")
a = adv_p[adv_p["personId"] == pid].iloc[0] if adv_p is not None and (adv_p["personId"] == pid).any() else None
c = st.columns(6)
c[0].metric("Minutes", f"{p['mins']:.0f}")
c[1].metric("Points", int(p["points"]))
c[2].metric("Rebounds", int(p["reboundsTotal"]))
c[3].metric("Assists", int(p["assists"]))
c[4].metric("Steals / blocks", f"{int(p['steals'])} / {int(p['blocks'])}")
c[5].metric("+/-", f"{p['plusMinusPoints']:+.0f}")
d = st.columns(6)
d[0].metric("FG", f"{int(p['fieldGoalsMade'])}/{int(p['fieldGoalsAttempted'])}")
d[1].metric("3P", f"{int(p['threePointersMade'])}/{int(p['threePointersAttempted'])}")
d[2].metric("FT", f"{int(p['freeThrowsMade'])}/{int(p['freeThrowsAttempted'])}")
d[3].metric("Turnovers", int(p["turnovers"]))
if a is not None:
    d[4].metric("True shooting %", f"{100 * a['trueShootingPercentage']:.1f}")
    d[5].metric("Usage %", f"{100 * a['usagePercentage']:.1f}")

# ---------- share of team ----------
st.subheader("Share of his team's production")
tt = team_t[team_t["teamTricode"] == team_choice]
share = analysis.share_of_team(roster[roster["nameI"] == name], tt, {
    "points": "Points", "reboundsTotal": "Rebounds", "assists": "Assists", "steals": "Steals",
    "blocks": "Blocks", "turnovers": "Turnovers", "fieldGoalsAttempted": "Shot attempts"}).drop(columns="PLAYER")
st.bar_chart(share.T.rename(columns={0: "% of team total"}), horizontal=True)
st.caption("A player who played 36 of 48 minutes would take about 75% of a team total if production were even per minute.")

# ---------- shots ----------
if pbp is not None and "xLegacy" in pbp:
    mine = analysis.shots(pbp)
    mine = mine[mine["playerNameI"] == name]
    st.subheader("Shots")
    if len(mine):
        left, right = st.columns([2, 3])
        with left:
            st.pyplot(viz.shot_map(mine, f"{name}: {int(mine['made'].sum())}/{len(mine)} from the field"), clear_figure=True)
        with right:
            st.dataframe(analysis.zone_table(mine), hide_index=True)
            log = mine.assign(Made=mine["made"].map({True: "Made", False: "Missed"}))[
                ["period", "clock", "Made", "shotDistance", "description"]].rename(
                columns={"period": "Q", "shotDistance": "Dist (ft)", "description": "Play"})
            log["clock"] = log["clock"].str.replace("PT", "").str.replace("M", ":").str.replace(r"\.\d+S", "", regex=True)
            st.dataframe(log.rename(columns={"clock": "Clock"}), hide_index=True, height=260)
    else:
        st.info("This player took no field-goal attempts.")

# ---------- hustle and tracking ----------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Hustle")
    if hus_p is not None and (hus_p["personId"] == pid).any():
        h = hus_p[hus_p["personId"] == pid].iloc[0]
        keep = {"deflections": "Deflections", "chargesDrawn": "Charges drawn", "contestedShots": "Contested shots",
                "screenAssists": "Screen assists", "screenAssistPoints": "Screen assist points",
                "looseBallsRecoveredTotal": "Loose balls recovered", "boxOuts": "Box outs"}
        st.dataframe(pd.DataFrame({"Stat": list(keep.values()), "Value": [h[k] for k in keep]}), hide_index=True)
    else:
        st.caption("No hustle row for this player.")
with c2:
    st.subheader("Player tracking (NBA.com cameras)")
    if trk_p is not None and (trk_p["personId"] == pid).any():
        t = trk_p[trk_p["personId"] == pid].iloc[0]
        rows = [("Distance (km)", round(t["distance"] * 1.609344, 2)), ("Average speed (km/h)", round(t["speed"] * 1.609344, 1)),
                ("Touches", t["touches"]), ("Passes", t["passes"]), ("Rebound chances", t["reboundChancesTotal"]),
                ("Contested FG", f"{int(t['contestedFieldGoalsMade'])}/{int(t['contestedFieldGoalsAttempted'])}"),
                ("Uncontested FG", f"{int(t['uncontestedFieldGoalsMade'])}/{int(t['uncontestedFieldGoalsAttempted'])}")]
        st.dataframe(pd.DataFrame(rows, columns=["Stat", "Value"]).astype(str), hide_index=True)
    else:
        st.caption("No tracking row for this player.")

# ---------- comparison ----------
st.subheader("Compare players")
everyone = played["nameI"].tolist()
picks = st.multiselect("Players to compare (any team)", everyone,
                       default=played.sort_values("points", ascending=False)["nameI"].head(4).tolist())
if picks:
    cmp = played[played["nameI"].isin(picks)].copy()
    out = pd.DataFrame({"PLAYER": cmp["nameI"], "TEAM": cmp["teamTricode"], "MIN": cmp["mins"].round(0),
                        "PTS": cmp["points"], "REB": cmp["reboundsTotal"], "AST": cmp["assists"],
                        "STL": cmp["steals"], "BLK": cmp["blocks"], "TOV": cmp["turnovers"],
                        "FG%": (100 * cmp["fieldGoalsPercentage"]).round(1), "+/-": cmp["plusMinusPoints"]})
    if adv_p is not None:
        extra = adv_p.set_index("personId")[["trueShootingPercentage", "usagePercentage", "PIE"]]
        e = extra.reindex(cmp["personId"]).reset_index(drop=True)
        out["TS%"] = (100 * e["trueShootingPercentage"]).round(1).to_numpy()
        out["USG%"] = (100 * e["usagePercentage"]).round(1).to_numpy()
        out["PIE%"] = (100 * e["PIE"]).round(1).to_numpy()
    st.dataframe(out.sort_values("PTS", ascending=False), hide_index=True)

common.nba_attribution()

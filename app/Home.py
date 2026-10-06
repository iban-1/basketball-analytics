"""Game stats: pick any NBA playoff game from 2016 to 2022 and see its full stats."""
import pandas as pd
import streamlit as st

import common
from src.bball.tables import NOT_AVAILABLE

common.setup_page("Game stats")

game, bundle = common.select_game()

# ---------- header ----------
st.header(f"{game['away']} {game['away_pts']} @ {game['home']} {game['home_pts']}")
meta = [game["date"]]
if "arena" in bundle["info"]:
    meta.append(bundle["info"]["arena"])
if bundle["info"].get("attendance"):
    meta.append(f"attendance {bundle['info']['attendance']:,}")
st.caption(" · ".join(meta))

teams = bundle["info"].get("teams")
if teams is not None:
    cols = ["teamTricode"] + [c for c in teams.columns if c.startswith("period") and c.endswith("Score")] + ["score"]
    line = teams[[c for c in cols if c in teams.columns]].rename(columns={
        "teamTricode": "TEAM", "score": "FINAL",
        **{f"period{i}Score": (f"Q{i}" if i <= 4 else f"OT{i - 4}") for i in range(1, 12)}})
    st.dataframe(line.dropna(axis=1, how="all"), hide_index=True)

with st.expander("What do these abbreviations mean?"):
    st.markdown("""
**Box score.** PTS points · REB rebounds (OREB offensive, DREB defensive) · AST assists ·
STL steals · BLK blocks · TOV turnovers · PF personal fouls · +/- team point difference while
the player was on court · MIN minutes. **Shooting.** FGM/FGA field goals made/attempted · 3PM/3PA
three-pointers · FTM/FTA free throws · FG%, 3P%, FT% accuracy · eFG% counts a three as worth 1.5
twos · TS% true shooting (also counts free throws).
**Advanced.** ORTG / DRTG points scored / allowed per 100 possessions while on court · NETRTG the
difference · USG% share of team plays used · AST%, REB% shares of teammates' baskets / available
rebounds · TOV ratio turnovers per 100 plays · PACE possessions per 48 min · PIE% share of all
game events (NBA.com's all-in-one rating).
**Miscellaneous.** PITP points in the paint · FB-PTS fast-break points · SCP second-chance points ·
PTS OFF TOV points off opponent turnovers · BLKA shots of his that were blocked · PFD fouls drawn.
**Hustle.** Deflections, charges drawn, contested shots, screen assists (and the points they led
to), loose balls recovered and box outs. **Player tracking** (optical cameras). Distance run, average
speed, touches (times he held the ball), passes, rebound chances, and shots contested /
uncontested / against the rim. NBA.com publishes distance in miles and speed in mph; km columns
are converted by this app.
""")

# ---------- stat tabs ----------
tabs = bundle["tabs"]
names = list(tabs)
for tab, name in zip(st.tabs(names), names):
    with tab:
        by_team = tabs[name]
        if not by_team:
            st.info(f"NBA.com has no '{name}' table for this game.")
            continue
        for tri, df in by_team.items():
            st.subheader(tri)
            st.dataframe(df, hide_index=True)
        if name == "Miscellaneous" and bundle["summary"] is not None:
            st.subheader("Team summary")
            st.caption("Bench points, biggest lead and scoring run come from NBA.com's game summary.")
            st.dataframe(bundle["summary"], hide_index=True)
        if name == "Box score" and len(bundle["dnp"]):
            st.caption("Did not play")
            st.dataframe(bundle["dnp"], hide_index=True)

with st.expander("Stats NBA.com does not provide for a single game (so they are not shown)"):
    st.table(pd.DataFrame(NOT_AVAILABLE, columns=["Stat", "Why it is missing"]))

common.nba_attribution()

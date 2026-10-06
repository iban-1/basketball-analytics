"""Tracking: NBA.com's optical player-tracking numbers for any game, plus the video clip's own tracking."""
import numpy as np
import pandas as pd
import streamlit as st

import common
from src.bball.tables import MILE_KM, minutes_to_float

common.setup_page("Tracking")
game, bundle = common.select_game()
tables = bundle["tables"]
home, away = game["home"], game["away"]
st.header(f"{game['away']} {game['away_pts']} @ {game['home']} {game['home_pts']}")

trk = tables["player_track"][0] if tables.get("player_track") else None
if trk is None or not len(trk):
    st.info("NBA.com has no player-tracking table for this game.")
else:
    t = trk.copy()
    t["mins"] = t["minutes"].map(minutes_to_float)
    t = t[t["mins"].fillna(0) > 0]
    t["km"] = t["distance"] * MILE_KM
    t["kmh"] = t["speed"] * MILE_KM
    t["PLAYER"] = t["nameI"]

    st.subheader("Distance and speed")
    st.caption("From NBA.com's optical tracking cameras (25 positions per second), converted from miles to km. "
               "Distance covers the whole game, so players with more minutes run further; the right-hand chart "
               "divides by minutes played.")
    t["Team"] = t["teamTricode"]
    t["Distance (km)"] = t["km"]
    t["Distance per minute (km)"] = t["km"] / t["mins"]
    t["Minutes played"] = t["mins"]
    t["Average speed (km/h)"] = t["kmh"]
    height = 60 + 26 * len(t)                  # tall enough that every player gets a label
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Distance run in the game (km)")
        st.bar_chart(t.sort_values("Distance (km)", ascending=False), x="PLAYER", y="Distance (km)", color="Team",
                     horizontal=True, height=height, sort="-Distance (km)")
    with c2:
        st.caption("Distance per minute played (km), players with 5+ minutes")
        s = t[t["mins"] >= 5].sort_values("Distance per minute (km)", ascending=False)
        st.bar_chart(s, x="PLAYER", y="Distance per minute (km)", color="Team", horizontal=True,
                     height=60 + 26 * len(s), sort="-Distance per minute (km)")

    st.subheader("Average speed")
    st.scatter_chart(t, x="Minutes played", y="Average speed (km/h)", color="Team", size=60)
    st.caption("Average speed includes standing and walking, so it is far below sprint speed.")

    st.subheader("Touches and passing")
    tp = t.assign(Touches=t["touches"], Passes=t["passes"])
    st.scatter_chart(tp, x="Touches", y="Passes", color="Team", size=60)
    st.caption("A touch is a time the player held the ball; a pass is a pass he threw. Time of possession and "
               "potential assists are not published per game by NBA.com.")

    st.subheader("Rebounding and shot-contest tracking")
    r = pd.DataFrame({"PLAYER": t["PLAYER"], "TEAM": t["teamTricode"], "MIN": t["mins"].round(0),
                      "OREB chances": t["reboundChancesOffensive"], "DREB chances": t["reboundChancesDefensive"],
                      "Contested FG": t["contestedFieldGoalsMade"].astype(int).astype(str) + "/" + t["contestedFieldGoalsAttempted"].astype(int).astype(str),
                      "Uncontested FG": t["uncontestedFieldGoalsMade"].astype(int).astype(str) + "/" + t["uncontestedFieldGoalsAttempted"].astype(int).astype(str),
                      "Defended at rim": t["defendedAtRimFieldGoalsMade"].astype(int).astype(str) + "/" + t["defendedAtRimFieldGoalsAttempted"].astype(int).astype(str)})
    st.dataframe(r.sort_values(["TEAM", "MIN"], ascending=[True, False]), hide_index=True)

    st.subheader("Team totals")
    tt = tables["player_track"][1]
    out = pd.DataFrame({"TEAM": tt["teamTricode"], "Distance (km)": (tt["distance"] * MILE_KM).round(1),
                        "Avg speed (km/h)": (tt["speed"] * MILE_KM).round(2), "Touches": tt["touches"], "Passes": tt["passes"],
                        "Rebound chances": tt["reboundChancesTotal"]})
    st.dataframe(out, hide_index=True)

# ---------- our own tracking from video ----------
st.subheader("Tracking from the video clip (this project's own computer vision)")
V = common.RESULTS / "video"
clip_game = common.load_config()["video"]["official_game_id"]
if game["game_id"] == clip_game and (V / "players.csv").exists():
    pl = pd.read_csv(V / "players.csv")
    teams = common.load_config()["video"]["teams"]
    pl = pl[pl["team"].isin(teams)].copy()
    pl["TEAM"] = pl["team"].map(teams)
    st.caption("Measured from the broadcast clip of Q3 only, from the camera shots that show the whole court. "
               "Each row is one player within one camera shot (identity is lost at camera cuts), so these "
               "are partial distances, not whole-game totals, and they are estimates without ground truth.")
    show = pl.sort_values("distance_m", ascending=False).head(25)
    st.dataframe(pd.DataFrame({"TEAM": show["TEAM"], "Seconds tracked": show["seconds_tracked"].round(1),
                               "Distance (m)": show["distance_m"].round(1),
                               "Top speed (km/h)": (show["top_speed_ms"] * 3.6).round(1)}), hide_index=True)
    st.caption("The annotated video, radar and pass/interception counts are on the Video analysis page.")

    # ---------- frame viewer: top-down court, slider and 10-second playback ----------
    viewer_path = V / "viewer_frames.csv"
    if viewer_path.exists():
        import time

        import matplotlib.pyplot as plt

        from src.bball import viz
        st.subheader("Frame viewer")
        vf = pd.read_csv(viewer_path)
        ev = pd.read_csv(V / "events.csv") if (V / "events.csv").exists() else pd.DataFrame(columns=["frame", "type", "team"])
        FPS = 30
        frames = np.sort(vf["frame"].unique())
        by_frame = {f: g for f, g in vf.groupby("frame")}
        ev_frames = ev.sort_values("frame")[["frame", "type", "team"]].to_numpy()
        start_s = st.slider("Time in the clip (seconds)", 0.0, float(frames[-1] / FPS), 12.0, step=0.5)
        near = frames[np.searchsorted(frames, int(start_s * FPS)):]
        holder = st.empty()

        def draw(f: int) -> None:
            banner = None
            for ef, typ, team in ev_frames:
                if 0 <= f - ef <= 36:
                    banner = f"{str(typ).upper()}  {teams.get(team, team)}"
            fig = viz.tracking_frame(by_frame[f], teams, f"Clip time {f / FPS:5.1f} s", banner)
            holder.pyplot(fig)
            plt.close(fig)

        if len(near):
            draw(int(near[0]))
            if st.button("▶ Play next 10 seconds"):
                end = int(near[0]) + 10 * FPS
                for f in near[near <= end][::6]:                 # 5 pictures per second of clip time
                    draw(int(f))
                    time.sleep(0.05)
        st.caption("Blue = light kit, orange = dark kit, grey = referees and others. The yellow ring marks the player "
                   "the analysis says holds the ball, with the ball drawn beside him (the ball is not placed from the "
                   "video's pixels because a ball in the air cannot be located on the floor). Only camera shots that "
                   "show the whole court are tracked, so playback skips the gaps; players are not tracked across cuts.")
    else:
        st.info("Run `python -m scripts.run_analysis` to create the frame viewer data.")
else:
    st.caption("Video tracking exists only for the clip supplied by the user "
               f"(game {clip_game}). Select the 2022 Finals Game 4 to see it here.")

common.nba_attribution()

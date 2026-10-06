"""Video analysis: detection, tracking, passes, interceptions, possession and player motion for a clip."""
import json

import pandas as pd
import streamlit as st

import common

common.setup_page("Video analysis")
V = common.RESULTS / "video"
cfg = common.load_config()["video"]
FPS = 30.0

with st.expander("What am I looking at? (read this first)", expanded=True):
    st.markdown("""
This page analyses a **broadcast video clip** (the 3rd quarter of 2022 NBA Finals Game 4, 6 min 43 s of
television footage) with computer vision. It is separate from the *Game stats* page, which shows
NBA.com's official numbers for a whole game.

**What the video shows:** boxes around the players (blue = light kit, orange = dark kit, grey =
referees), each player's **current speed and the distance run so far**, a yellow ring on the **ball**, a
**radar in the top-left corner** (the court seen from above, with every tracked player), and in the
top-right **a running count for each team** of **passes, interceptions and ball possession %**.

**How it works:** a YOLO neural network finds people and the ball. A tracker gives each person an ID. The
camera view is matched to the painted court lines so pixels can be turned into real court metres, which
makes distance, speed and the radar possible. Teams are told apart by shirt colour.

**Important limits (details in the quality panel below):** only shots where the court can be mapped are
analysed (close-ups and other camera angles are skipped); the ball is hard to see and is found by a model
trained on this clip itself; "players" are tracked **within one camera shot** (the clip has many cuts), so
a player's distance restarts at every cut; and **pass/interception/possession numbers are estimates, not
official stats**.
""")

DEMO_URL = "https://github.com/iban-1/basketball-analytics/releases/download/v1.0.0/basketball-analytics-annotated-demo.mp4"
video = V / "annotated_web.mp4"
if video.exists():
    st.video(str(video))
    st.caption("The footage belongs to its rights holder. The numbers drawn on it are estimates from computer "
               "vision (see the quality panel below).")
else:
    st.video(DEMO_URL)
    st.caption("Demo video from the project's GitHub release (the footage belongs to its rights holder). To "
               "produce your own, put a clip in `data/video/` and run the pipeline from the README: "
               "`run_detection`, `run_analysis`, then `render_video`.")

summary_path = V / "summary.json"
if not summary_path.exists():
    st.stop()
S = json.loads(summary_path.read_text())

st.subheader("Team statistics (this clip)")
ts = pd.DataFrame(S["team_stats"]).T.reset_index().rename(columns={
    "index": "team", "passes": "passes", "interceptions": "interceptions (incl. steals)",
    "possession_pct": "ball possession %", "distance_m_tracked_players": "distance covered by tracked players (m)",
    "players_tracked_seconds": "player-seconds tracked"})
st.dataframe(ts.drop(columns=["kit"]), hide_index=True)
st.caption("Possession % is the share of analysed time in which each team was the last to hold the ball. Passes = "
           "ball moves between two different teammates within 1.5 s and at least 1.5 m. Interceptions = the ball "
           "passes to the other team within 1 s, more than 4.5 m from either basket (so rebounds and inbounds are "
           "excluded; deflections and steals out of a dribble are included).")

ctx = S.get("official_period_context")
if ctx:
    st.info(f"**Official context, Q3 (NBA.com play-by-play):** steals {ctx['steals_by_team']}, turnovers "
            f"{ctx['turnovers_by_team']}. Our video clip covers about {S['analysed_seconds']:.0f} s of court-view "
            f"footage, so the counts are not expected to match exactly; interceptions here also include deflections "
            f"and loose-ball changes that the official scorer does not log as steals.")

st.subheader("Players: distance and speed")
pl_path = V / "players.csv"
if pl_path.exists():
    pl = pd.read_csv(pl_path)
    team = st.radio("Team", ["both"] + sorted(pl["team_name"].dropna().unique()), horizontal=True)
    if team != "both":
        pl = pl[pl["team_name"] == team]
    show = pl[["pid", "team_name", "shot", "seconds_tracked", "distance_m", "avg_speed_kmh", "top_speed_kmh"]]
    show = show.rename(columns={"pid": "track chain", "team_name": "team", "shot": "camera shot",
                                "seconds_tracked": "seconds tracked", "distance_m": "distance (m)",
                                "avg_speed_kmh": "avg speed (km/h)", "top_speed_kmh": "top speed (km/h)"})
    st.dataframe(show.sort_values("seconds tracked", ascending=False), hide_index=True)
    st.caption("Each row is one player followed through ONE camera shot (a cut starts a new row; the clip cannot "
               "tell which named player it is). Speeds under 1.3 km/h count as standing still. Positions come from "
               "mapping the video to the court and carry an error of a few tenths of a metre.")

st.subheader("Pass and interception events")
ev_path = V / "events.csv"
if ev_path.exists():
    ev = pd.read_csv(ev_path)
    ev["clip time"] = (ev["frame"] / FPS).map(lambda s: f"{int(s // 60)}:{s % 60:04.1f}")
    names = cfg["teams"]
    ev["team"] = ev["team"].map(names)
    st.dataframe(ev[["clip time", "type", "team", "dist_m", "gap_s"]].rename(columns={
        "dist_m": "distance (m)", "gap_s": "ball in flight (s)"}), hide_index=True)

st.subheader("Data quality: how much of the clip was actually analysed")
q = {
    "Clip length": f"{S['clip_seconds']:.0f} s",
    "Shots showing the court (analysed)": f"{S['usable_shots']} shots, {S['analysed_seconds']:.0f} s "
                                          f"({S['analysed_seconds'] / S['clip_seconds']:.0%})",
    "Frames mapped to the court": f"{S['mapped_share_of_analysed_frames']:.0%} of analysed frames",
    "Ball found (detected or briefly interpolated)": f"{S['ball_share_of_mapped_frames']:.0%} of mapped frames",
    "Frames with a known ball holder": f"{S['holder_frames']}",
    "Ball model": S["ball_source"],
}
st.table(pd.Series(q, name="value"))
common.nba_attribution()

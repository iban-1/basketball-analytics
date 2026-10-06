"""Stage 2 (fast): teams, players, ball, possession, passes, interceptions, distance and speed.

Run:  python -m scripts.run_analysis
Reads results/video/{people,frames,balls(_v2),shots}.csv from the detection stage and writes
players.csv, player_frames.csv, ball_track.csv, states.csv, events.csv, possession_frames.csv,
people_labelled.csv and summary.json into results/video/.
"""
from __future__ import annotations

import json

import pandas as pd

from src.bball.analytics.ball import ball_track
from src.bball.analytics.players import (assign_teams, player_table, stitch_tracks, track_summary)
from src.bball.analytics.possession import (detect_events, frame_holders, holder_states,
                                            possession_frames, possession_share)
from src.bball.config import load_config, resolve, set_seed
from src.bball.video.court import L, W

FPS = 30.0
MIN_TOP_SPEED = 3.0        # m/s (11 km/h): near the lines a person must reach this to count (jitter stays below)
MIN_INSIDE = 0.7           # share of frames inside the court lines
MIN_SECONDS = 1.0          # a chain must be tracked for at least this long
EDGE = 0.4                 # metres: a typical spot this close to a boundary line is never "on court"
INTERIOR_GAP = 1.5         # metres from every line: such a person counts even when standing


def official_q3_steals(game_id: str, period: int) -> dict | None:
    """Official steals / turnovers per team in one period, from NBA.com play-by-play (context only)."""
    try:
        from src.bball.nba_stats import get_endpoint
        pbp = get_endpoint(game_id, "playbyplay")[0]
    except Exception as exc:                               # offline etc.: the comparison is optional
        print("official stats unavailable:", exc)
        return None
    in_period = pbp[pbp["period"] == period]
    turnovers = in_period[in_period["actionType"].astype(str) == "Turnover"]
    # A steal is its own row ("Tatum STEAL (1 STL)") whose team is the team that made the steal.
    steals = in_period[in_period["description"].astype(str).str.contains(r"\bSTEAL\b", case=True, regex=True)]
    return {"period": period,
            "turnovers_by_team": {k: int(v) for k, v in turnovers["teamTricode"].value_counts().items()},
            "steals_by_team": {k: int(v) for k, v in steals["teamTricode"].value_counts().items()}}


def main() -> None:
    cfg = load_config()
    v = cfg["video"]
    set_seed(cfg["seed"])
    res = resolve(cfg["paths"]["results"]) / "video"
    people = pd.read_csv(res / "people.csv")
    frames = pd.read_csv(res / "frames.csv")
    shots = pd.read_csv(res / "shots.csv")
    ball_file = res / "balls_v2.csv" if (res / "balls_v2.csv").exists() else res / "balls.csv"
    balls = pd.read_csv(ball_file)
    print(f"ball detections from {ball_file.name}: {len(balls)} candidates")

    # ---- teams and player chains ----
    summ = track_summary(people)
    team, centres = assign_teams(summ, cfg["seed"])
    pid = stitch_tracks(people, summ, team)
    key = list(zip(people["shot"], people["track_id"]))
    people["pid"] = [pid.get(k, "") for k in key]
    people["team"] = [team.get(k, "unknown") for k in key]
    kept = people[people["pid"] != ""].copy()
    print("tracks by team:", team.value_counts().to_dict(), "| chains:", kept["pid"].nunique())

    players, pframes = player_table(kept, FPS)
    # Real players and referees MOVE. Seated spectators, photographers and bench staff within the
    # court margin barely do, so a chain must reach a minimum speed to count at all.
    # Distance of a chain's typical position to the nearest boundary line. A person standing in the
    # INTERIOR of the court is a player or referee even if he is not running; a person standing
    # near the lines (bench, staff, photographers, seated fans) only counts if he really moves.
    edge_gap = pd.concat([players["median_X"], L - players["median_X"], players["median_Y"],
                          W - players["median_Y"]], axis=1).min(axis=1)
    interior = edge_gap >= INTERIOR_GAP
    moves = players["top_speed_ms"] >= MIN_TOP_SPEED
    moving = players[(interior | moves) & (edge_gap >= EDGE) & (players["inside_lines_share"] >= MIN_INSIDE)
                     & (players["seconds_tracked"] >= MIN_SECONDS)]
    print(f"chains: {len(players)} -> {len(moving)} after the on-court test")
    players = moving
    pframes = pframes[pframes["pid"].isin(players["pid"])]
    kept = kept[kept["pid"].isin(players["pid"])]
    refs = players[players["team"] == "other"]
    players = players[players["team"].isin(["light", "dark"])].copy()
    players["team_name"] = players["team"].map(v["teams"])

    # ---- ball, holder, possession, events ----
    ball = ball_track(balls, frames, kept)
    holders = frame_holders(kept, ball)
    states = holder_states(holders, kept)
    events = detect_events(states)
    analysed = shots[shots["usable"]]
    poss = possession_frames(states, analysed)
    share = possession_share(poss)

    mapped_frames = int(frames["mapped"].sum())
    labelled = kept[kept["team"].isin(["light", "dark"])]
    per_frame = labelled.groupby("frame").size().reindex(frames.loc[frames["mapped"], "frame"], fill_value=0)
    quality = {
        "analysed_frames": int(len(frames)), "mapped_frames": mapped_frames,
        "alignment_score_median": round(float(frames.loc[frames["mapped"], "score"].median()), 2),
        "feature_matches_median": int(frames.loc[frames["mapped"], "inliers"].median()),
        "players_labelled_per_mapped_frame_median": float(per_frame.median()),
        "mapped_frames_with_8_or_more_players_labelled": round(float((per_frame >= 8).mean()), 3),
        "raw_tracks": int(people.groupby(["shot", "track_id"]).ngroups),
        "player_and_referee_chains_kept": int(kept["pid"].nunique()),
    }
    team_stats = {}
    for t in ("light", "dark"):
        team_stats[v["teams"][t]] = {
            "kit": t,
            "passes": int(((events["type"] == "pass") & (events["team"] == t)).sum()),
            "interceptions": int(((events["type"] == "interception") & (events["team"] == t)).sum()),
            "possession_pct": round(100 * share[t], 1) if share[t] == share[t] else None,
            "players_tracked_seconds": round(float(players.loc[players["team"] == t, "seconds_tracked"].sum()), 1),
            "distance_m_tracked_players": round(float(players.loc[players["team"] == t, "distance_m"].sum()), 0),
        }
    summary = {
        "clip_seconds": round(float(shots["frames"].sum() / FPS), 1),
        "usable_shots": int(analysed.shape[0]), "analysed_seconds": round(float(analysed["frames"].sum() / FPS), 1),
        "mapped_share_of_analysed_frames": round(mapped_frames / max(len(frames), 1), 3),
        "ball_frames_selected": int(len(ball)), "ball_frames_detected": int((~ball["interpolated"]).sum()),
        "ball_share_of_mapped_frames": round(len(ball) / max(mapped_frames, 1), 3),
        "holder_frames": int(len(holders)), "holding_states": int(len(states)),
        "frames_with_known_possession": int(poss["team"].isin(["light", "dark"]).sum()),
        "team_stats": team_stats, "team_colour_centres_Lab": centres,
        "referee_like_chains": int(len(refs)), "quality": quality,
        "official_period_context": official_q3_steals(v["official_game_id"], v["period"]),
        "ball_source": ball_file.name,
    }
    players.to_csv(res / "players.csv", index=False)
    pframes.to_csv(res / "player_frames.csv", index=False)
    ball.to_csv(res / "ball_track.csv", index=False)
    states.to_csv(res / "states.csv", index=False)
    events.to_csv(res / "events.csv", index=False)
    poss.to_csv(res / "possession_frames.csv", index=False)
    kept.to_csv(res / "people_labelled.csv", index=False)
    (res / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

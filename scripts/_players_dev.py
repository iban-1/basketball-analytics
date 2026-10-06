"""Development helper: run the players module on whatever shots are finished so far."""
import pandas as pd

from src.bball.analytics.players import (assign_teams, player_table, stitch_tracks, track_summary)

people = pd.read_csv("results/video/people.csv")
summ = track_summary(people)
team, centres = assign_teams(summ)
print("track teams:", team.value_counts().to_dict(), "| colour centres (L,a,b):", centres)
pid = stitch_tracks(people, summ, team)
print("tracks kept for stitching:", int((pid != "").sum()), "-> chains:", pid[pid != ""].nunique())
key = list(zip(people["shot"], people["track_id"]))
people["pid"] = [pid.get(k, "") for k in key]
people["team"] = [team.get(k, "unknown") for k in key]
pl = people[people["pid"] != ""]
stats, frames = player_table(pl)
print(stats.sort_values("seconds_tracked", ascending=False).head(14).to_string(index=False))
print("chains per shot (players+refs):", stats.groupby("shot").size().to_dict())
print("distance_m by team (tracked chains):", stats.groupby("team")["distance_m"].sum().round(0).to_dict())

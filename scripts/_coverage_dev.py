"""Development helper: where do on-court people get lost between raw tracks and labelled players?"""
import pandas as pd

from src.bball.analytics.players import inside_strict, track_summary

res = "results/video/"
people = pd.read_csv(res + "people.csv")
lab = pd.read_csv(res + "people_labelled.csv")
frames = pd.read_csv(res + "frames.csv")
mapped = frames[frames.mapped]["frame"]
m = people[people.mapped].copy()
m["inside"] = inside_strict(m)
raw_in = m[m.inside].groupby("frame").size().reindex(mapped, fill_value=0)
print("raw on-court detections per mapped frame: median", raw_in.median(), "| >=8:", round((raw_in >= 8).mean(), 2))
summ = track_summary(people)
print("tracks:", len(summ), "| by length (frames):", summ["frames"].describe(percentiles=[.25, .5, .75]).round(0).to_dict())
print("tracks < 15 frames:", int((summ.frames < 15).sum()), f"({(summ.frames < 15).mean():.0%}) | no colour (n_colour<10):", int((summ.n_colour < 10).sum()))
long_in = summ[(summ.frames >= 15) & (summ.inside_share >= 0.6)]
print("tracks passing the length+inside test:", len(long_in))
# frame coverage of people vs labelled
lab_in = lab[lab.team.isin(["light", "dark"])].groupby("frame").size().reindex(mapped, fill_value=0)
print("labelled players per mapped frame: median", lab_in.median(), "| >=8:", round((lab_in >= 8).mean(), 2))
short = m[m.inside].merge(summ[["frames"]], on=["shot", "track_id"])
print("share of on-court detections that sit in tracks shorter than 15 frames:", round((short.frames < 15).mean(), 2))

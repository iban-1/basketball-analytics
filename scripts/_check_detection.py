"""Development helper: sanity-check the output of the detection stage."""
import pandas as pd

p = pd.read_csv("results/video/people.csv")
b = pd.read_csv("results/video/balls.csv")
f = pd.read_csv("results/video/frames.csv")
print("frames", len(f), "| mapped share", round(f.mapped.mean(), 3), "| inliers median", int(f.inliers.median()),
      "| alignment score median", round(f.score.median(), 2))
pf = p.groupby(["shot", "frame"]).size()
print("tracked people per frame: median", int(pf.median()), "p10", int(pf.quantile(.1)), "p90", int(pf.quantile(.9)))
for s, g in p.groupby("shot"):
    lens = g.groupby("track_id").size()
    per_frame = g.groupby("frame").size().median()
    print(f"shot {s}: {g.frame.nunique()} frames, {g.track_id.nunique()} track ids, median track length "
          f"{int(lens.median())} frames, ids per person-on-screen {g.track_id.nunique() / per_frame:.1f}")
print("ball candidates per frame:", round(len(b) / len(f), 2), "| frames with >=1 candidate:",
      round(b.frame.nunique() / len(f), 3), "| frames with candidate conf>=0.15:",
      round(b[b.conf >= 0.15].frame.nunique() / len(f), 3))
m = p[p.mapped]
print("court coords of kept people: X", round(m.X.min(), 1), "to", round(m.X.max(), 1), "| Y", round(m.Y.min(), 1), "to", round(m.Y.max(), 1))

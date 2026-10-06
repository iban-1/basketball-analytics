"""Development helper: look at track-level shirt colours to see how teams/referees separate."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

p = pd.read_csv("results/video/people.csv")
p = p[p.mapped & p.L.notna()]
inside = (p.X.between(0, 28.65)) & (p.Y.between(0, 15.24))
per = p[inside].groupby(["shot", "track_id"]).agg(L=("L", "median"), a=("a", "median"), b=("b", "median"),
                                                  n=("L", "size"), x=("X", "median"), y=("Y", "median"))
per = per[per.n >= 15]
print(len(per), "tracks with >=15 coloured frames inside the court")
print(per[["L", "a", "b"]].describe().round(1).to_string())
X = per[["L", "a", "b"]].to_numpy()
for k in (2, 3, 4):
    km = KMeans(k, n_init=10, random_state=42).fit(X)
    print(f"k={k} centres (L,a,b):", np.round(km.cluster_centers_, 0).tolist(), "sizes", np.bincount(km.labels_).tolist())
print(per.sort_values("L").round(0).iloc[[0, 5, 10, 20, 30, -30, -20, -10, -5, -1]].to_string())

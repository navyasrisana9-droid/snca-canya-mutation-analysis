import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AA = "ACDEFGHIKLMNPQRSTVWY"
known = ["A30P", "E46K", "H50Q", "G51D", "A53T"]

w = pd.read_csv("results/step8_scored_windows.csv")
wt = (w[["window_start", "wt_score"]].drop_duplicates("window_start")
      .sort_values("window_start"))
df = pd.read_csv("results/step11_scores_with_ranks.csv")
pos = pd.read_csv("results/step19_position_summary.csv")
L = df["position"].max()

grid = np.full((len(AA), L), np.nan)
for r in df.itertuples():
    grid[AA.index(r.mut_aa), r.position - 1] = r.delta
lim = np.nanmax(np.abs(grid))
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("lightgray")
k = df[df["mutation"].isin(known)].set_index("mutation").loc[known]

fig = plt.figure(figsize=(14, 12), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.6, 1], width_ratios=[3, 1.2])

a = fig.add_subplot(gs[0, :])
a.plot(wt["window_start"] + 9.5, wt["wt_score"], color="#4a6fa5")
a.axvspan(61, 95, color="gray", alpha=0.2, label="NAC (~61-95)")
a.set_title("A. Wild-type CANYA score along the sequence")
a.set_xlabel("Position (window center)")
a.set_ylabel("CANYA score")
a.legend()

b = fig.add_subplot(gs[1, :])
im = b.imshow(grid, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
b.set_yticks(range(len(AA)))
b.set_yticklabels(list(AA))
b.set_xticks(range(9, L, 10))
b.set_xticklabels(range(10, L + 1, 10))
b.set_title("B. Predicted change in score for every single mutation")
b.set_xlabel("Position")
b.set_ylabel("Substituted amino acid")
fig.colorbar(im, ax=b, label="Change in CANYA score", pad=0.01)

c = fig.add_subplot(gs[2, 0])
c.bar(pos["position"], pos["mean_abs"], color="#4a6fa5")
c.axvspan(61, 95, color="gray", alpha=0.2)
c.set_title("C. Mutational sensitivity by position")
c.set_xlabel("Position")
c.set_ylabel("Mean |change in score|")

d = fig.add_subplot(gs[2, 1])
d.barh(known, k["delta_rank_pct"], color="#8a8a8a")
d.axvline(50, color="black", linestyle="--", linewidth=1)
d.set_xlim(0, 100)
d.invert_yaxis()
d.set_title("D. Known mutations: rank percentile")
d.set_xlabel("Percentile (100 = most increasing)")

fig.savefig("results/step21_figure.png", dpi=300)
print("Saved results/step21_figure.png")
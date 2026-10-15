import numpy as np
import pandas as pd

known = ["A30P", "E46K", "H50Q", "G51D", "A53T"]

windows = pd.read_csv("results/step8_scored_windows.csv")
windows["window_delta"] = windows["mut_score"] - windows["wt_score"]
per_mut = windows.groupby("mutation")["window_delta"].agg(
    delta_mean="mean", delta_peak="max").reset_index()

df = pd.read_csv("results/step11_scores_with_ranks.csv").merge(per_mut, on="mutation")
methods = ["delta", "delta_mean", "delta_peak"]

print("Rank correlation between methods (1.0 = identical ordering):")
print(df[methods].corr(method="spearman").round(2).to_string())

for m in methods:
    df[m + "_pct"] = df[m].rank(pct=True) * 100
k = df[df["mutation"].isin(known)]
print("\nRank percentile of each known mutation (100 = most increasing):")
print(k[["mutation"] + [m + "_pct" for m in methods]].round(1).to_string(index=False))

print("\nChance a random set of 5 mutations scores this high or higher:")
rng = np.random.default_rng(0)
for m in methods:
    vals = df[m].to_numpy()
    observed = k[m].mean()
    random_means = np.array([
        rng.choice(vals, size=len(known), replace=False).mean()
        for _ in range(10000)
    ])
    print(f"  {m}: {(random_means >= observed).mean():.3f}")

df.to_csv("results/step14_robustness.csv", index=False)
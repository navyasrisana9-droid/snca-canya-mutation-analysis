import numpy as np
import pandas as pd

df = pd.read_csv("results/step9_mutation_scores.csv")
known = ["A30P", "E46K", "H50Q", "G51D", "A53T"]

# 100 = ranks among the most aggregation-increasing mutations, 0 = most decreasing
df["delta_rank_pct"] = df["delta"].rank(pct=True) * 100
k = df[df["mutation"].isin(known)]
print(k[["mutation", "delta", "delta_rank_pct"]].round(3).to_string(index=False))

# Permutation test: how often does a random set of 5 mutations score this high?
observed = k["delta"].mean()
all_deltas = df["delta"].to_numpy()
rng = np.random.default_rng(0)
random_means = np.array([
    rng.choice(all_deltas, size=len(known), replace=False).mean()
    for _ in range(10000)
])
p = (random_means >= observed).mean()

print("\nMean delta of the 5 known mutations:", round(observed, 4))
print("Mean delta of all", len(df), "mutations:", round(all_deltas.mean(), 4))
print("Chance a random set of 5 scores this high or higher:", round(p, 4))

df.to_csv("results/step11_scores_with_ranks.csv", index=False)
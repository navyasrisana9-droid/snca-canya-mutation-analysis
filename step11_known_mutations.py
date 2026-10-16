import os
import numpy as np
import pandas as pd

df = pd.read_csv("results/step9_mutation_scores.csv")

# 100 = ranks among the most aggregation-increasing mutations, 0 = most decreasing
df["delta_rank_pct"] = df["delta"].rank(pct=True) * 100

# Generic output: needed by later steps for any protein
df.to_csv("results/step11_scores_with_ranks.csv", index=False)

# Known disease mutations: alpha-synuclein only (empty list for other proteins)
is_snca = os.environ.get("PROTEIN_NAME", "snca") == "snca"
known = ["A30P", "E46K", "H50Q", "G51D", "A53T"] if is_snca else []

if known:
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
else:
    print("No known mutations configured for this protein; saved ranks only.")
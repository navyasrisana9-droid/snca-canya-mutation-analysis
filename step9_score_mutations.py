import pandas as pd

windows = pd.read_csv("results/step8_scored_windows.csv")

per_mut = windows.groupby("mutation").agg(
    wt_score=("wt_score", "max"),
    mut_score=("mut_score", "max"),
).reset_index()
per_mut["delta"] = per_mut["mut_score"] - per_mut["wt_score"]

meta = pd.read_csv("results/mutations.csv")[["mutation", "wt_aa", "position", "mut_aa"]]
result = meta.merge(per_mut, on="mutation")
result.to_csv("results/step9_mutation_scores.csv", index=False)

cols = ["mutation", "wt_score", "mut_score", "delta"]
print("Mutations scored:", len(result))
print("\nTop 5 predicted to INCREASE aggregation:")
print(result.nlargest(5, "delta")[cols].round(3).to_string(index=False))
print("\nTop 5 predicted to DECREASE aggregation:")
print(result.nsmallest(5, "delta")[cols].round(3).to_string(index=False))

known = ["A30P", "E46K", "H50Q", "G51D", "A53T"]
print("\nKnown Parkinson's mutations:")
print(result[result["mutation"].isin(known)][cols].round(3).to_string(index=False))
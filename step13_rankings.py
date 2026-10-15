import pandas as pd

df = pd.read_csv("results/step11_scores_with_ranks.csv")
cols = ["mutation", "wt_score", "mut_score", "delta"]

print("Delta summary across all", len(df), "mutations:")
print(df["delta"].describe().round(4).to_string())

top_up = df.nlargest(10, "delta")[cols]
top_down = df.nsmallest(10, "delta")[cols]

print("\nTop 10 predicted to INCREASE aggregation:")
print(top_up.round(3).to_string(index=False))
print("\nTop 10 predicted to DECREASE aggregation:")
print(top_down.round(3).to_string(index=False))

top_up.to_csv("results/step13_top10_increase.csv", index=False)
top_down.to_csv("results/step13_top10_decrease.csv", index=False)
print("\nSaved the two top-10 tables to results/")
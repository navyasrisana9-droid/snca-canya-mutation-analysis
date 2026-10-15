import pandas as pd

# id -> peptide (what we sent to CANYA)
ids = pd.read_csv("results/step6_canya_input.tsv", sep="\t", header=None,
                  names=["id", "peptide"])

# id -> score (what CANYA sent back); tolerate a header row if there is one
out = pd.read_csv("results/step6_canya_output.tsv", sep="\t", header=None,
                  usecols=[0, 1], names=["id", "score"])
out["score"] = pd.to_numeric(out["score"], errors="coerce")
out = out.dropna(subset=["score"])

scores = ids.merge(out, on="id")
peptide_score = dict(zip(scores["peptide"], scores["score"]))

windows = pd.read_csv("results/step6_mutation_windows.csv")
windows["wt_score"] = windows["wt_peptide"].map(peptide_score)
windows["mut_score"] = windows["mut_peptide"].map(peptide_score)

missing = windows[["wt_score", "mut_score"]].isna().any(axis=1).sum()
print("Peptides scored:", len(peptide_score))
print("Window rows:", len(windows))
print("Rows with a missing score:", missing)
print("Score range:", round(scores["score"].min(), 4), "to", round(scores["score"].max(), 4))
print(windows[windows["mutation"] == "A53T"][["window_start", "wt_score", "mut_score"]].head(3))

windows.to_csv("results/step8_scored_windows.csv", index=False)


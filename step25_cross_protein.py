import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

PROTEINS = {
    "alpha-synuclein": "results/step9_mutation_scores.csv",
    "amyloid-beta 42": "runs/abeta/results/step9_mutation_scores.csv",
    "tau repeat domain": "runs/tau/results/step9_mutation_scores.csv",
}

data, rows = {}, []
for name, path in PROTEINS.items():
    if not Path(path).exists():
        print("Skipping", name, "(missing", path + ")")
        continue
    df = pd.read_csv(path)
    df["abs_delta"] = df["delta"].abs()
    df["d_hydro"] = df["mut_aa"].map(KD) - df["wt_aa"].map(KD)
    data[name] = df
    rho = spearmanr(df["delta"], df["d_hydro"])[0]
    rows.append({
        "protein": name,
        "n_mutations": len(df),
        "mean_abs_delta": df["abs_delta"].mean(),
        "fraction_increasing": (df["delta"] > 0).mean(),
        "spearman_delta_vs_hydrophobicity_change": rho,
        "mean_delta_to_proline": df.loc[df["mut_aa"] == "P", "delta"].mean(),
        "mean_delta_to_charged_DEKR": df.loc[df["mut_aa"].isin(list("DEKR")), "delta"].mean(),
    })

summary = pd.DataFrame(rows)
summary.to_csv("results/cross_protein_summary.csv", index=False)
print(summary.round(3).to_string(index=False))

fig, axes = plt.subplots(len(data), 1, figsize=(11, 3 * len(data)))
axes = np.atleast_1d(axes)
for ax, (name, df) in zip(axes, data.items()):
    pos = df.groupby("position")["abs_delta"].mean()
    ax.bar(pos.index, pos.values, color="#4a6fa5")
    ax.set_title(name)
    ax.set_ylabel("Mean |delta|")
axes[-1].set_xlabel("Residue position (within the analyzed sequence)")
fig.tight_layout()
fig.savefig("results/cross_protein_profiles.png", dpi=300)
print("Saved results/cross_protein_summary.csv and cross_protein_profiles.png")
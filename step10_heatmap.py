import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

df = pd.read_csv("results/step9_mutation_scores.csv")
length = df["position"].max()

# Rows = new amino acid, columns = position, cells = change in score
grid = np.full((len(AMINO_ACIDS), length), np.nan)
for _, row in df.iterrows():
    grid[AMINO_ACIDS.index(row["mut_aa"]), row["position"] - 1] = row["delta"]

limit = np.nanmax(np.abs(grid))  # symmetric color scale so white = no change

cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("lightgray")  # gray cells = the original residue (no mutation)

fig, ax = plt.subplots(figsize=(16, 5))
im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
ax.set_yticks(range(len(AMINO_ACIDS)))
ax.set_yticklabels(list(AMINO_ACIDS))
ax.set_xticks(range(9, length, 10))
ax.set_xticklabels(range(10, length + 1, 10))
ax.set_xlabel("Position in alpha-synuclein")
ax.set_ylabel("Amino acid substituted in")
ax.set_title("Predicted change in aggregation score for every single mutation")
fig.colorbar(im, ax=ax, label="Change in CANYA score (mutant - wild type)")
fig.tight_layout()
fig.savefig("results/step10_heatmap.png", dpi=200)

print("Saved results/step10_heatmap.png")
print("Filled cells:", int(np.sum(~np.isnan(grid))), "(expected 2660)")

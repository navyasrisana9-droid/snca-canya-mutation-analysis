"""Step 3: turn peptide scores into per-mutation delta scores, rankings,
a mutation heatmap, a position-sensitivity plot, and a check of known
familial Parkinson's mutations.

Per-mutation score = the maximum window score among the windows that cover the
mutated residue (aggregation is driven by the most nucleation-prone stretch).
Delta = mutant value - wild-type value over the SAME windows.
Use --agg mean to try the average instead; reporting both is good robustness.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

AA = "ACDEFGHIKLMNPQRSTVWY"
# Familial SNCA point mutations commonly cited in the literature.
# Verify each against primary sources before publishing.
KNOWN = ["A30P", "E46K", "H50Q", "G51D", "A53T", "A53E"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--agg", choices=["max", "mean"], default="max")
    ap.add_argument("--label", default="", help="text added to plot titles, e.g. DEMO SCORES")
    a = ap.parse_args()
    r = Path(a.results)

    variants = pd.read_csv(r / "variants.csv")
    vw = pd.read_csv(r / "variant_windows.csv")
    sc = pd.read_csv(r / "peptide_scores.csv").set_index("peptide")["score"]

    vw["score_mut"] = vw["mut_peptide"].map(sc)
    vw["score_wt"] = vw["wt_peptide"].map(sc)
    if vw[["score_mut", "score_wt"]].isna().any().any():
        raise SystemExit("Some peptides have no score; peptide_scores.csv is incomplete.")

    g = vw.groupby("mutation")[["score_mut", "score_wt"]].agg(a.agg)
    df = variants[["mutation", "position", "wt", "mut"]].merge(g, on="mutation")
    df["delta"] = df["score_mut"] - df["score_wt"]
    df.to_csv(r / "variant_scores.csv", index=False)

    # Rankings
    cols = ["mutation", "score_wt", "score_mut", "delta"]
    df.nlargest(10, "delta")[cols].to_csv(r / "top10_increase_aggregation.csv", index=False)
    df.nsmallest(10, "delta")[cols].to_csv(r / "top10_decrease_aggregation.csv", index=False)

    # Heatmap: rows = substituted residue, columns = position
    L = df["position"].max()
    mat = np.full((len(AA), L), np.nan)
    for _, row in df.iterrows():
        mat[AA.index(row["mut"]), row["position"] - 1] = row["delta"]
    lim = np.nanmax(np.abs(mat))
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_yticks(range(len(AA)))
    ax.set_yticklabels(list(AA))
    ax.set_xlabel("Position in alpha-synuclein")
    ax.set_ylabel("Substituted amino acid")
    ax.set_title(f"Change in predicted aggregation score (mutant - WT) {a.label}".strip())
    fig.colorbar(im, ax=ax, label="Delta score")
    fig.tight_layout()
    fig.savefig(r / "heatmap.png", dpi=200)
    plt.close(fig)

    # Position sensitivity
    pos = df.groupby("position")["delta"].agg(mean="mean", abs_mean=lambda x: x.abs().mean())
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(pos.index, pos["abs_mean"], color="#4a6fa5")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean |delta score|")
    ax.set_title(f"Mutational sensitivity by position {a.label}".strip())
    fig.tight_layout()
    fig.savefig(r / "position_sensitivity.png", dpi=200)
    plt.close(fig)

    # Validation against known familial mutations
    df["rank_pct"] = df["delta"].rank(pct=True) * 100
    known = df[df["mutation"].isin(KNOWN)][["mutation", "delta", "rank_pct"]]
    known.to_csv(r / "known_mutations.csv", index=False)
    rest = df[~df["mutation"].isin(KNOWN)]["delta"]
    if len(known) >= 3:
        p = mannwhitneyu(known["delta"], rest, alternative="greater").pvalue
        print(f"Known mutations vs all others (one-sided Mann-Whitney): p = {p:.3g}")
    print(known.to_string(index=False))
    print(f"\n{len(df)} variants analysed with agg={a.agg}. Outputs in {r}/")


if __name__ == "__main__":
    main()

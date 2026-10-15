import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

rng = np.random.default_rng(0)
df = pd.read_csv("results/step15_features.csv")
df["abs_delta"] = df["delta"].abs()
df["label"] = df["wt_aa"] + df["position"].astype(str) + df["mutation"].str[-1] if False else df["wt_aa"] + df["position"].astype(str) + df["mut_aa"]
df["percentile"] = df["delta"].rank(pct=True) * 100
pos = pd.read_csv("results/step19_position_summary.csv")

def boot_ci(x, n=2000):
    x = x.to_numpy()
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])

# ---- Table 1: region sensitivity with bootstrap 95% CI ----
regions = {"N-terminal (1-60)": (1, 60), "NAC pre-core (61-67)": (61, 67),
           "GAVV (68-71)": (68, 71), "NAC core (72-82)": (72, 82),
           "NAC end (83-95)": (83, 95), "C-terminal (96-140)": (96, 140)}
rows = []
for name, (lo, hi) in regions.items():
    s = df[df["position"].between(lo, hi)]["abs_delta"]
    ci = boot_ci(s)
    rows.append({"region": name, "n": len(s), "mean_abs_delta": s.mean(),
                 "ci_low": ci[0], "ci_high": ci[1]})
t1 = pd.DataFrame(rows)
t1.to_csv("results/table1_region_sensitivity.csv", index=False)
print(t1.round(4).to_string(index=False))

# ---- Statistical tests ----
core = df["position"].between(61, 82)
u = mannwhitneyu(df.loc[core, "abs_delta"], df.loc[~core, "abs_delta"], alternative="greater")
print(f"\nTest 1: |delta| residues 61-82 vs rest, Mann-Whitney p = {u.pvalue:.2e}")
nac = df["position"].between(61, 95)
pro = df["mut_is_proline"] == 1
u2 = mannwhitneyu(df.loc[nac & pro, "delta"], df.loc[nac & ~pro, "delta"], alternative="less")
print(f"Test 2: proline vs other delta inside NAC, Mann-Whitney p = {u2.pvalue:.2e}")
print("(Residues within a position are not independent, so treat p-values as descriptive.)")

# ---- Table 2: familial mutations ----
fam = ["A30P", "E46K", "H50Q", "G51D", "A53T", "A53E"]
t2 = df[df["label"].isin(fam)][["label", "position", "delta", "percentile"]]
t2 = t2.set_index("label").loc[fam].reset_index()
t2.to_csv("results/table2_familial_mutations.csv", index=False)

# ---- Table 3: model performance (values copied from Steps 16 and 18) ----
t3 = pd.DataFrame([
    ["all features", -0.068, 0.482, 0.378, 0.550],
    ["no position", -0.210, 0.454, 0.347, 0.531],
    ["substitution only", -0.039, 0.489, 0.537, 0.506]],
    columns=["feature set", "R2", "Spearman (all)", "Spearman (NAC)", "Spearman (outside NAC)"])
t3.to_csv("results/table3_model_performance.csv", index=False)

# ---- Figure 1: position sensitivity with familial sites ----
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(pos["position"], pos["mean_abs"], color="#4a6fa5")
ax.axvspan(61, 95, color="gray", alpha=0.2, label="NAC (61-95)")
ax.axvspan(71, 82, color="orange", alpha=0.25, label="Core 71-82 (deletion blocks aggregation)")
for p, n in [(30, "A30P"), (46, "E46K"), (50, "H50Q"), (51, "G51D"), (53, "A53T/E")]:
    ax.axvline(p, color="red", ls=":", lw=1)
    ax.text(p, ax.get_ylim()[1] * 0.95, n, rotation=90, va="top", ha="right", fontsize=7, color="red")
ax.set_xlabel("Position in alpha-synuclein")
ax.set_ylabel("Mean |change in CANYA score|")
ax.set_title("Mutation sensitivity concentrates in the NAC region")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("results/fig1_position_sensitivity.png", dpi=300)

# ---- Figure 2: region comparison ----
fig, ax = plt.subplots(figsize=(7, 4))
err = [t1["mean_abs_delta"] - t1["ci_low"], t1["ci_high"] - t1["mean_abs_delta"]]
ax.bar(range(len(t1)), t1["mean_abs_delta"], yerr=err, capsize=3, color="#4a6fa5")
ax.set_xticks(range(len(t1)))
ax.set_xticklabels(t1["region"], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean |change in CANYA score| (95% bootstrap CI)")
ax.set_title("Sensitivity by region")
fig.tight_layout()
fig.savefig("results/fig2_region_sensitivity.png", dpi=300)

# ---- Figure 3: familial mutations vs all mutations ----
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(t2["label"], t2["percentile"], color="#c0504d")
ax.axvline(50, color="black", ls="--", lw=1)
ax.set_xlim(0, 100)
ax.set_xlabel("Percentile of delta among all 2,660 mutations")
ax.set_title("Familial PD mutations are not predicted\nto increase nucleation")
fig.tight_layout()
fig.savefig("results/fig3_familial_mutations.png", dpi=300)
print("\nSaved fig1-3 and table1-3 in results/")
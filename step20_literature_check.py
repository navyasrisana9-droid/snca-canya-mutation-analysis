import pandas as pd

df = pd.read_csv("results/step15_features.csv")
print("Columns:", list(df.columns))
print("Position range:", df["position"].min(), "-", df["position"].max())
df["abs_delta"] = df["delta"].abs()

# --- Check 1: published hotspot regions ---
regions = {
    "N-terminal (1-60)": (1, 60),
    "NAC pre-core (61-67)": (61, 67),
    "GAVV motif (68-71)": (68, 71),
    "NAC core (71-82)": (71, 82),
    "NAC end (83-95)": (83, 95),
    "C-terminal (96-140)": (96, 140),
}
rows = []
for name, (lo, hi) in regions.items():
    sub = df[df["position"].between(lo, hi)]
    rows.append({"region": name, "n": len(sub),
                 "mean_abs_delta": sub["abs_delta"].mean(),
                 "mean_delta": sub["delta"].mean()})
print("\nCheck 1: sensitivity by published region")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# --- Check 2: proline substitutions ---
nac = df["position"].between(61, 95)
print("\nCheck 2: mean delta for proline vs other substitutions")
for label, mask in [("inside NAC", nac), ("outside NAC", ~nac)]:
    p = df[mask & (df["mut_is_proline"] == 1)]["delta"].mean()
    o = df[mask & (df["mut_is_proline"] == 0)]["delta"].mean()
    print(f"  {label}: proline {p:.3f} vs other {o:.3f}")

# --- Check 3: familial Parkinson's mutations ---
wt_col = next((c for c in ["wt", "wt_aa", "ref", "ref_aa"] if c in df.columns), None)
mut_col = next((c for c in ["mut", "mut_aa", "alt", "alt_aa"] if c in df.columns), None)
if wt_col and mut_col:
    df["label"] = df[wt_col] + df["position"].astype(str) + df[mut_col]
    df["percentile"] = df["delta"].rank(pct=True) * 100
    fam = ["A30P", "E46K", "H50Q", "G51D", "A53T", "A53E"]
    out = df[df["label"].isin(fam)][["label", "delta", "percentile"]]
    print("\nCheck 3: familial mutations (percentile among all mutations)")
    print(out.round(3).to_string(index=False))
    print("Missing from file:", [m for m in fam if m not in set(df["label"])])
else:
    print("\nCheck 3 skipped: no wild-type/mutant amino acid columns found.")
    print("Tell me the column names above and I'll adjust.")
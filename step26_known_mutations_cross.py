import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(0)

# (label, scores file, offset from published numbering to this file's numbering, mutations)
SETS = [
    ("amyloid-beta 42", "runs/abeta/results/step9_mutation_scores.csv", 0,
     ["A21G", "E22G", "E22K", "E22Q", "D23N"]),
    ("tau repeat domain", "runs/tau/results/step9_mutation_scores.csv", 243,
     ["N279K", "P301L", "P301S", "S305N", "V337M"]),
]

rows = []
for label, path, offset, muts in SETS:
    if not Path(path).exists():
        print("Missing", path)
        continue
    df = pd.read_csv(path)
    df["percentile"] = df["delta"].rank(pct=True) * 100
    found = []
    for m in muts:
        wt, pos, mut = m[0], int(m[1:-1]) - offset, m[-1]
        hit = df[(df["position"] == pos) & (df["mut_aa"] == mut)]
        if hit.empty:
            print(f"{label}: {m} not found (position {pos}); skipped")
            continue
        row = hit.iloc[0]
        if row["wt_aa"] != wt:
            print(f"{label}: {m} wild-type residue mismatch ({row['wt_aa']}); skipped")
            continue
        found.append(row["delta"])
        rows.append({"protein": label, "mutation": m, "delta": row["delta"],
                     "percentile": row["percentile"]})
    if found:
        obs = np.mean(found)
        all_d = df["delta"].to_numpy()
        rand = np.array([rng.choice(all_d, size=len(found), replace=False).mean()
                         for _ in range(10000)])
        print(f"\n{label}: mean delta of {len(found)} known mutations = {obs:.4f} "
              f"(all mutations: {all_d.mean():.4f})")
        print(f"  chance a random set scores this high or higher: {(rand >= obs).mean():.4f}")
        print(f"  chance a random set scores this low or lower:   {(rand <= obs).mean():.4f}")

out = pd.DataFrame(rows)
out.to_csv("results/known_mutations_cross_protein.csv", index=False)
print("\n", out.round(3).to_string(index=False))
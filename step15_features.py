import os
from pathlib import Path
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydrophobicity (higher = more hydrophobic)
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

# Residue volume in cubic angstroms (Zamyatnin)
VOLUME = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
          "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
          "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
          "Y": 193.6, "V": 140.0}

# Charge at neutral pH (histidine treated as neutral for simplicity)
CHARGE = {aa: 0 for aa in AA}
CHARGE.update({"D": -1, "E": -1, "K": 1, "R": 1})

df = pd.read_csv("results/step11_scores_with_ranks.csv")[
    ["mutation", "wt_aa", "position", "mut_aa", "delta"]]

props = {"hydro": KD, "volume": VOLUME, "charge": CHARGE}
for name, table in props.items():
    df[f"wt_{name}"] = df["wt_aa"].map(table)
    df[f"mut_{name}"] = df["mut_aa"].map(table)
    df[f"d_{name}"] = df[f"mut_{name}"] - df[f"wt_{name}"]

# Optional annotated region, e.g. PROTEIN_REGION="61-95". Defaults to the NAC for alpha-synuclein.
region = os.environ.get("PROTEIN_REGION") or ("61-95" if os.environ.get("PROTEIN_NAME", "snca") == "snca" else "")
if region:
    lo, hi = map(int, region.split("-"))
    df["in_NAC"] = df["position"].between(lo, hi).astype(int)
else:
    df["in_NAC"] = 0  # no annotated region for this proteindf["mut_is_proline"] = (df["mut_aa"] == "P").astype(int)

df.to_csv("results/step15_features.csv", index=False)
print("Table shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print(df[df["mutation"] == "V71P"].T)
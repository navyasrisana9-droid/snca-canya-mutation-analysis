from pathlib import Path

SEQ_FILE = Path("data/snca_P37840.fasta")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path):
    lines = Path(path).read_text().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


wt = read_fasta(SEQ_FILE)

mutations = []  # list of (name, mutant_sequence)
for i, original in enumerate(wt):  # i counts from 0
    for new in AMINO_ACIDS:
        if new == original:
            continue  # skip "no change"
        name = f"{original}{i + 1}{new}"  # position counted from 1 in the name
        mutant_seq = wt[:i] + new + wt[i + 1:]
        mutations.append((name, mutant_seq))

print("Total mutations:", len(mutations))
print("First 3:", [m[0] for m in mutations[:3]])

# Sanity check on one known mutation
lookup = dict(mutations)
a53t = lookup["A53T"]
print("A53T differs from wild type at exactly one position:",
      sum(a != b for a, b in zip(wt, a53t)) == 1)
print("Residue 53 in A53T is now:", a53t[52])
print("Same length as wild type:", len(a53t) == len(wt))

import pandas as pd

rows = []
for name, mutant_seq in mutations:
    rows.append({
        "mutation": name,
        "wt_aa": name[0],
        "position": int(name[1:-1]),
        "mut_aa": name[-1],
        "mutant_sequence": mutant_seq,
    })

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/mutations.csv", index=False)

print(df.shape)
print(df.head(3).drop(columns="mutant_sequence"))
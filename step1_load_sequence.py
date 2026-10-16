import os
from pathlib import Path

SEQ_FILE = Path(os.environ.get("PROTEIN_FASTA", "data/snca_P37840.fasta"))


def read_fasta(path):
    lines = Path(path).read_text().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


wt = read_fasta(SEQ_FILE)
print("Length:", len(wt))
print("First 20:", wt[:20])

# Known-mutation check applies to alpha-synuclein only.
# Positions are counted from 1, but Python counts from 0, so we use pos - 1.
if os.environ.get("PROTEIN_NAME", "snca") == "snca":
    for name in ["A30P", "E46K", "H50Q", "G51D", "A53T"]:
        original, pos = name[0], int(name[1:-1])
        print(name, "original residue matches:", wt[pos - 1] == original)
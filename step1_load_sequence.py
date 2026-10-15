from pathlib import Path

SEQ_FILE = Path("data/snca_P37840.fasta")


def read_fasta(path):
    lines = Path(path).read_text().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


wt = read_fasta(SEQ_FILE)
print("Length:", len(wt))
print("First 20:", wt[:20])

# Positions are counted from 1, but Python counts from 0, so we use pos - 1.
for name in ["A30P", "E46K", "H50Q", "G51D", "A53T"]:
    original, pos = name[0], int(name[1:-1])
    print(name, "original residue matches:", wt[pos - 1] == original)
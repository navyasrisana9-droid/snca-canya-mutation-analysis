import csv
from pathlib import Path

SEQ_FILE = Path(__import__("os").environ.get("PROTEIN_FASTA", "data/snca_P37840.fasta"))
WINDOW = 20
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path):
    lines = Path(path).read_text().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


wt = read_fasta(SEQ_FILE)

rows = []  # (mutation, window_start, wt_peptide, mut_peptide)
for i, original in enumerate(wt):
    pos = i + 1
    for new in AMINO_ACIDS:
        if new == original:
            continue
        mutation = f"{original}{pos}{new}"
        mutant = wt[:i] + new + wt[i + 1:]
        first_start = max(1, pos - WINDOW + 1)
        last_start = min(pos, len(wt) - WINDOW + 1)
        for start in range(first_start, last_start + 1):
            rows.append((mutation, start,
                         wt[start - 1:start - 1 + WINDOW],
                         mutant[start - 1:start - 1 + WINDOW]))

Path("results").mkdir(exist_ok=True)
with open("results/step6_mutation_windows.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mutation", "window_start", "wt_peptide", "mut_peptide"])
    writer.writerows(rows)

# Every different peptide that CANYA must score (wild-type and mutant windows)
unique_peptides = sorted({p for _, _, w, m in rows for p in (w, m)})
with open("results/step6_canya_input.tsv", "w") as f:
    for n, pep in enumerate(unique_peptides):
        f.write(f"p{n}\t{pep}\n")  # id <TAB> peptide, no header

print("Mutation-window rows:", len(rows))
print("Unique peptides to score:", len(unique_peptides))
print("A53T has", sum(1 for r in rows if r[0] == "A53T"), "windows")
print("First peptide:", unique_peptides[0])
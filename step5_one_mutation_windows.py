from pathlib import Path

SEQ_FILE = Path(__import__("os").environ.get("PROTEIN_FASTA", "data/snca_P37840.fasta"))
WINDOW = 20


def read_fasta(path):
    lines = Path(path).read_text().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


wt = read_fasta(SEQ_FILE)

mutation = "A53T"
original, pos, new = mutation[0], int(mutation[1:-1]), mutation[-1]
mutant = wt[:pos - 1] + new + wt[pos:]

# A window starting at `start` covers residues start .. start+19 (counted from 1)
first_start = max(1, pos - WINDOW + 1)
last_start = min(pos, len(wt) - WINDOW + 1)

pairs = []  # (start, wild-type peptide, mutant peptide)
for start in range(first_start, last_start + 1):
    wt_pep = wt[start - 1:start - 1 + WINDOW]
    mut_pep = mutant[start - 1:start - 1 + WINDOW]
    pairs.append((start, wt_pep, mut_pep))

print("Windows covering", mutation, ":", len(pairs))
print("Start positions:", first_start, "to", last_start)
print("First:", pairs[0])
print("Last:", pairs[-1])
print("Each mutant window differs from its wild-type window at exactly one spot:",
      all(sum(a != b for a, b in zip(w, m)) == 1 for _, w, m in pairs))
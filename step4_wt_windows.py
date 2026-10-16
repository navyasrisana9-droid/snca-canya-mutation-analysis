from pathlib import Path

SEQ_FILE = Path(__import__("os").environ.get("PROTEIN_FASTA", "data/snca_P37840.fasta"))
WINDOW = 20


def read_fasta(path):
    lines = Path(path).read_text().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


wt = read_fasta(SEQ_FILE)

windows = []  # list of (start_position_counted_from_1, peptide)
for start in range(len(wt) - WINDOW + 1):
    windows.append((start + 1, wt[start:start + WINDOW]))

print("Number of windows:", len(windows))
print("First window:", windows[0])
print("Last window:", windows[-1])
print("All windows are length 20:", all(len(p) == WINDOW for _, p in windows))
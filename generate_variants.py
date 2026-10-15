"""Step 1: generate every single amino-acid substitution of a protein and the
20-residue windows that CANYA needs to score them.

CANYA was trained on 20-aa peptides, so a whole 140-aa protein is scored by
sliding a 20-aa window along it. For a mutation at position p, only the windows
that contain p can change, so we score just those (<= 20 per mutation).

Outputs (in --out-dir):
  variants.csv         one row per mutation (e.g. A53T) with the full mutant sequence
  variant_windows.csv  mutation -> every window covering it (mutant + matching WT peptide)
  peptides.csv         the unique peptides to score (WT + mutant windows)
"""
import argparse
import csv
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path):
    return "".join(l.strip() for l in Path(path).read_text().splitlines() if not l.startswith(">"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="data/snca_P37840.fasta")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()

    seq = read_fasta(a.fasta)
    L, w = len(seq), a.window
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    variants, vwins, peptides = [], [], set()
    for i, wt in enumerate(seq):  # i is 0-based; reported position is i+1
        for mut in AA:
            if mut == wt:
                continue
            name = f"{wt}{i + 1}{mut}"
            mseq = seq[:i] + mut + seq[i + 1:]
            variants.append((name, i + 1, wt, mut, mseq))
            for s in range(max(0, i - w + 1), min(i, L - w) + 1):
                wt_pep, mut_pep = seq[s:s + w], mseq[s:s + w]
                vwins.append((name, s + 1, wt_pep, mut_pep))
                peptides.update((wt_pep, mut_pep))

    with open(out / "variants.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["mutation", "position", "wt", "mut", "mutant_sequence"])
        cw.writerows(variants)
    with open(out / "variant_windows.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["mutation", "window_start", "wt_peptide", "mut_peptide"])
        cw.writerows(vwins)
    with open(out / "peptides.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["peptide"])
        cw.writerows([[p] for p in sorted(peptides)])

    print(f"sequence length {L}; {len(variants)} variants; "
          f"{len(vwins)} variant-window rows; {len(peptides)} unique peptides to score")


if __name__ == "__main__":
    main()

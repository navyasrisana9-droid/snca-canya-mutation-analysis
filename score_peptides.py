"""Step 2: score every peptide in peptides.csv -> peptide_scores.csv (peptide,score).

Two ways to use this:

  A) Real run: fill in `canya_scores()` below with however you run CANYA
     (its Python package/command line from the authors' repository, or a loop
     over a web form), then:  python score_peptides.py --scorer canya
     OR skip this script entirely and produce peptide_scores.csv yourself with
     columns `peptide,score` for every row of peptides.csv.

  B) Pipeline test: `--scorer demo` uses a crude hydrophobicity heuristic.
     It is NOT CANYA and its numbers must never appear in a poster or paper.
"""
import argparse
import csv
import math

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def demo_scores(peptides):
    return [1 / (1 + math.exp(-(sum(KD[c] for c in p) / len(p)))) for p in peptides]


def canya_scores(peptides):
    raise NotImplementedError(
        "Plug CANYA in here: return a list of floats, one per peptide, in order. "
        "Check the CANYA repository/paper (Thompson et al., Sci. Adv. 2025) for the "
        "exact install and call, and confirm whether it expects exactly 20-aa input.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/peptides.csv")
    ap.add_argument("--output", default="results/peptide_scores.csv")
    ap.add_argument("--scorer", choices=["demo", "canya"], required=True)
    a = ap.parse_args()

    with open(a.input) as f:
        peps = [r["peptide"] for r in csv.DictReader(f)]
    scores = demo_scores(peps) if a.scorer == "demo" else canya_scores(peps)
    with open(a.output, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["peptide", "score"])
        cw.writerows(zip(peps, scores))
    print(f"scored {len(peps)} peptides with '{a.scorer}' -> {a.output}")


if __name__ == "__main__":
    main()

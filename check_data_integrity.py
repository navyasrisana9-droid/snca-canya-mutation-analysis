#!/usr/bin/env python3
"""
Data-integrity checks for an alpha-synuclein (SNCA) mutation-scan dataset.

Usage:
  python check_data_integrity.py --fasta snca.fasta --csv scores.csv \
      [--mut-col mutation] [--score-col score] [--seq-col sequence]

CSV needs at least a mutation column (e.g. A53T, 1-based) and a score column.
Optional: a column holding the full mutant sequence submitted to Canya.
"""
import argparse, re, sys
from collections import Counter
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_LEN = 140                      # human SNCA (UniProt P37840)
LANDMARKS = {30: "A", 46: "E", 50: "H", 51: "G", 53: "A"}  # A30P E46K H50Q G51D A53T
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

fails, warns = [], []
def ok(msg):   print(f"  [PASS] {msg}")
def fail(msg): print(f"  [FAIL] {msg}"); fails.append(msg)
def warn(msg): print(f"  [WARN] {msg}"); warns.append(msg)

def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq).upper()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--mut-col", default="mutation")
    p.add_argument("--score-col", default="score")
    p.add_argument("--seq-col", default=None)
    a = p.parse_args()

    # ---- 1. Wild-type sequence
    print("\n1. Wild-type sequence")
    wt = read_fasta(a.fasta)
    ok(f"length = {len(wt)}") if len(wt) == EXPECTED_LEN else fail(f"length is {len(wt)}, expected {EXPECTED_LEN}")
    bad = sorted(set(wt) - AA)
    ok("only the 20 standard amino acids") if not bad else fail(f"non-standard characters: {bad}")
    for pos, aa in LANDMARKS.items():
        got = wt[pos-1] if pos <= len(wt) else "?"
        ok(f"position {pos} = {aa}") if got == aa else fail(f"position {pos} is {got}, expected {aa} (sequence or indexing issue)")

    # ---- 2. Load table
    print("\n2. Table structure")
    df = pd.read_csv(a.csv)
    for c in [a.mut_col, a.score_col] + ([a.seq_col] if a.seq_col else []):
        if c not in df.columns:
            fail(f"column '{c}' not found. Columns: {list(df.columns)}"); sys.exit(report())
    ok(f"{len(df)} rows loaded")
    df[a.mut_col] = df[a.mut_col].astype(str).str.strip().str.upper()

    # ---- 3. Mutation labels
    print("\n3. Mutation labels vs. wild-type sequence")
    parsed, unparsable = [], []
    for m in df[a.mut_col]:
        r = MUT_RE.match(m)
        parsed.append(r.groups() if r else None)
        if not r: unparsable.append(m)
    ok("all labels parse as <WT><pos><MUT>") if not unparsable else fail(f"{len(unparsable)} unparsable labels, e.g. {unparsable[:5]}")

    mismatch, out_of_range, same_aa, bad_mut = [], [], [], []
    for m, g in zip(df[a.mut_col], parsed):
        if not g: continue
        w, pos, mu = g[0], int(g[1]), g[2]
        if not 1 <= pos <= len(wt): out_of_range.append(m); continue
        if wt[pos-1] != w: mismatch.append(f"{m} (sequence has {wt[pos-1]})")
        if w == mu: same_aa.append(m)
        if mu not in AA: bad_mut.append(m)
    ok("every WT residue matches the sequence at its position") if not mismatch else fail(f"{len(mismatch)} WT-residue mismatches (possible 0/1-based indexing error), e.g. {mismatch[:5]}")
    ok("all positions within 1-140") if not out_of_range else fail(f"out-of-range positions: {out_of_range[:5]}")
    ok("no synonymous 'mutations'") if not same_aa else fail(f"WT == mutant: {same_aa[:5]}")
    ok("all mutant residues are standard amino acids") if not bad_mut else fail(f"invalid mutant residues: {bad_mut[:5]}")

    # Off-by-one hint
    if mismatch:
        shifted = sum(1 for g in parsed if g and 1 <= int(g[1])+1 <= len(wt) and wt[int(g[1])] == g[0])
        if shifted > 0.8 * len([g for g in parsed if g]):
            warn("most labels match if positions are shifted by +1: likely a 0-based vs 1-based indexing bug")

    # ---- 4. Duplicates and completeness
    print("\n4. Duplicates and completeness")
    dups = df[a.mut_col][df[a.mut_col].duplicated()].unique().tolist()
    ok("no duplicate mutations") if not dups else fail(f"{len(dups)} duplicated mutations, e.g. {dups[:5]}")
    expected = {f"{wt[i]}{i+1}{m}" for i in range(len(wt)) for m in AA if m != wt[i]}
    have = set(df[a.mut_col])
    missing = sorted(expected - have, key=lambda x: (int(x[1:-1]), x))
    extra = sorted(have - expected)
    print(f"  [INFO] full landscape = {len(expected)} variants; you have {len(have & expected)} of them")
    if missing: warn(f"{len(missing)} landscape variants missing (fine if you did a subset), e.g. {missing[:8]}")
    else: ok("complete 140 x 19 landscape")
    if extra: warn(f"{len(extra)} labels not in the landscape (e.g. {extra[:5]})")
    for m in ["A30P", "E46K", "H50Q", "G51D", "A53T"]:
        ok(f"{m} present") if m in have else fail(f"disease mutation {m} missing")

    # ---- 5. Scores
    print("\n5. Scores")
    s = pd.to_numeric(df[a.score_col], errors="coerce")
    nbad = int(s.isna().sum())
    ok("all scores numeric and non-missing") if nbad == 0 else fail(f"{nbad} missing/non-numeric scores (failed Canya runs?)")
    sv = s.dropna()
    if len(sv):
        print(f"  [INFO] min={sv.min():.3f}  median={sv.median():.3f}  max={sv.max():.3f}  sd={sv.std():.3f}")
        top, n = Counter(sv.round(6)).most_common(1)[0]
        if n / len(sv) > 0.10: warn(f"{n/len(sv):.0%} of scores equal {top}: same input submitted repeatedly, or score saturation?")
        else: ok("no suspicious pile-up of identical scores")
        if sv.std() == 0: fail("zero variance in scores")
        if (sv < 0).any() or (sv > 1).any(): warn("scores fall outside 0-1; fine only if Canya's scale isn't 0-1")

    # ---- 6. Submitted sequences (optional)
    if a.seq_col:
        print("\n6. Submitted mutant sequences")
        bad_seq = []
        for m, g, seq in zip(df[a.mut_col], parsed, df[a.seq_col].astype(str).str.upper()):
            if not g: continue
            diffs = [i for i, (x, y) in enumerate(zip(wt, seq)) if x != y]
            if len(seq) != len(wt) or diffs != [int(g[1])-1] or seq[int(g[1])-1] != g[2]:
                bad_seq.append(m)
        ok("every sequence differs from WT at exactly the labeled position") if not bad_seq else fail(f"{len(bad_seq)} sequences don't match their label, e.g. {bad_seq[:5]}")
    else:
        print("\n6. Submitted mutant sequences\n  [SKIP] no --seq-col given")
        print("  Tip: rebuild each mutant from the label and compare with what you pasted into Canya.")

    sys.exit(report())

def report():
    print("\n" + "=" * 50)
    print(f"{len(fails)} failure(s), {len(warns)} warning(s)")
    return 1 if fails else 0

if __name__ == "__main__":
    main()
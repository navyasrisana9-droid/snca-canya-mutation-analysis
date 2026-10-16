"""
canya_wrapper.py - one function, score_sequences(list[str]) -> list[float].

cross_disease_analysis.py imports this and nothing else, so whatever you used to get
Canya scores in Components 1-4 just needs to be plugged in below.

Pick ONE backend by setting CANYA_BACKEND (default: "csv"):

  csv    You score sequences on the Canya web tool (or any external route) and paste results back.
         Workflow: run the pipeline -> it writes canya_to_score.csv and stops -> score that file ->
         save results as canya_scores.csv (columns: sequence,score) -> rerun the pipeline.
  local  You call Canya from Python (authors' code/package). Fill in _load_local_model() and _score_local().

Whichever you use, it MUST be the same Canya version/settings you used for alpha-synuclein.
"""
import os
import sys
from pathlib import Path

import pandas as pd

BACKEND = os.environ.get("CANYA_BACKEND", "csv")
TO_SCORE = Path("canya_to_score.csv")
SCORES = Path("canya_scores.csv")

# ----------------------------------------------------------------------------- csv backend
def _score_csv(seqs):
    known = {}
    if SCORES.exists():
        df = pd.read_csv(SCORES)
        known = dict(zip(df["sequence"], df["score"]))
    missing = sorted({s for s in seqs if s not in known})
    if missing:
        pd.DataFrame({"sequence": missing}).to_csv(TO_SCORE, index=False)
        sys.exit(
            f"\n{len(missing)} sequences still need Canya scores.\n"
            f"1) Score the sequences in {TO_SCORE} with Canya\n"
            f"2) Save results as {SCORES} with columns: sequence,score\n"
            f"3) Re-run the pipeline.\n"
        )
    return [known[s] for s in seqs]

# ----------------------------------------------------------------------------- local backend
_model = None

def _load_local_model():
    """TODO: load Canya here (e.g. import the authors' package and load weights). Return the model object."""
    raise NotImplementedError("Fill in _load_local_model() using the Canya code you used before.")

def _score_local(seqs):
    """TODO: return one float per sequence. Batch the calls; there are ~30,000 unique windows overall."""
    global _model
    if _model is None:
        _model = _load_local_model()
    raise NotImplementedError("Fill in _score_local() to call the loaded model on `seqs`.")

# ----------------------------------------------------------------------------- public API
def score_sequences(seqs):
    seqs = list(seqs)
    out = _score_csv(seqs) if BACKEND == "csv" else _score_local(seqs)
    assert len(out) == len(seqs), "scorer must return one score per sequence"
    return [float(x) for x in out]

if __name__ == "__main__":  # quick self-test: python canya_wrapper.py
    test = ["MDVFMKGLSKAKEGVVAAAE", "KTKEGVLYVGSKTKEGVVHG"]
    print(score_sequences(test))
# Alpha-synuclein aggregation: mutation sensitivity with CANYA

Predicts how all 2,660 single-amino-acid substitutions in alpha-synuclein (SNCA)
change amyloid nucleation propensity using the CANYA model, then tests which
simple features explain the effects and compares results to published
Parkinson's disease literature.

## Key findings
- Predicted sensitivity concentrates in residues 61-82 (NAC), ~4-5x higher than
  the N- and C-termini.
- Proline substitutions in the NAC lower predicted nucleation.
- Hydrophobicity, size and charge rank mutation effects (Spearman ~0.5) but do
  not predict magnitudes in held-out regions (R2 <= 0).
- Familial PD mutations are not flagged by CANYA (limitation).

## Run
    pip install -r requirements.txt
    python run_all.py

Figures and tables are written to `results/`.

## Structure
- `stepN_*.py`: pipeline scripts, run in order by `run_all.py`
- `data/`: input sequence and intermediate files
- `results/`: figures and tables
- `METHODS_AND_LIMITATIONS.md`: methods and limitations

## Citation / credits
CANYA: Thompson et al., 2024, "Interpretably deep learning amyloid nucleation
by massive experimental quantification of random sequences" (bioRxiv).
# Alpha-synuclein aggregation: mutation sensitivity with CANYA

Uses the CANYA model to predict how all 2,660 single-amino-acid substitutions in
alpha-synuclein (SNCA, UniProt P37840) change amyloid nucleation propensity, then
tests which simple amino-acid features explain the predictions and compares them
with published Parkinson's disease literature. All scores are model predictions,
not experimental measurements.

## Key findings
- Predicted sensitivity concentrates in residues ~61-82 (mean |delta| 0.028-0.031
  vs 0.006-0.008 in the N- and C-terminal regions, ~3.5-5x). The rest of the NAC
  (83-95) is not elevated (0.009).
- In the NAC, proline substitutions lower the predicted score on average (-0.016
  vs +0.013 for other substitutions). The largest increases are hydrophobic
  substitutions (e.g. T64I, T72I, A76F).
- Amino-acid properties, mainly hydrophobicity of the introduced residue, rank
  mutation effects moderately (Spearman 0.45-0.49) but do not predict magnitudes
  in held-out regions (R2 ~ 0 or negative).
- Familial PD mutations (A30P, E46K, H50Q, G51D, A53E, A53T) are not flagged by
  CANYA: percentiles 5-51. See `METHODS_AND_LIMITATIONS.md`.

## Run
CANYA needs numpy 1.19.5, so it runs in its own environment.

1. **Score peptides with CANYA** (separate Python 3.9 environment; see the CANYA
   repository for installation):
       canya --input results/step6_canya_input.tsv --output results/step6_canya_output.tsv
   The output file is included in this repository, so you can skip this step.
2. **Run the analysis:**
       pip install -r requirements.txt
       python run_all.py

Figures and tables are written to `results/`.

## Structure
- `stepN_*.py`: pipeline scripts run in order by `run_all.py` (CANYA is step 7,
  run separately)
- `data/`: input sequence
- `results/`: figures and tables
- `METHODS_AND_LIMITATIONS.md`: methods and limitations

## References
- Thompson M, et al. Massive experimental quantification allows interpretable deep
  learning of protein aggregation. Science Advances 11(18):eadt5111 (2025).
- Sequence: UniProt P37840.
- Rodriguez JA, et al. Structure of the toxic core of alpha-synuclein from invisible
  crystals. Nature (2015).
- Sahay S, et al. Familial Parkinson disease-associated mutations alter the
  site-specific microenvironment and dynamics of alpha-synuclein. J Biol Chem
  290:7804 (2015).

  ## Cross-protein comparison
The same pipeline was run on amyloid-beta 42 and the tau repeat domain (residues
244-372) and compared with alpha-synuclein.

    python run_all.py --fasta data/abeta42.fasta --name abeta
    python run_all.py --fasta data/tau_RD.fasta --name tau
    python step25_cross_protein.py

Across all three proteins, mutations that increase hydrophobicity tend to raise
the predicted nucleation score (Spearman 0.38-0.49), while proline and charged
(D/E/K/R) substitutions tend to lower it. The share of mutations that raise the
score differs between proteins (67%, 44%, 56%).
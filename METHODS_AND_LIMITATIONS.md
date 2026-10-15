# Methods summary

**Sequence and variants.** The human alpha-synuclein (SNCA) sequence (140 aa) was
loaded and all 2,660 single amino acid substitutions (140 positions x 19) were
generated.

**Scoring.** Each variant was scored with CANYA, a convolution-attention neural
network trained on nucleation data for >100,000 random 20-residue peptides
(Thompson et al., 2024). Mutation effect (delta) = [CONFIRM: mutant score minus
wild-type score, aggregated across overlapping 20-aa windows by mean / max].

**Feature analysis.** Features: wild-type and mutant hydrophobicity, volume and
charge, their differences, a proline indicator, position, and NAC membership
(residues 61-95). A random forest regressor (200 trees, min_samples_leaf=5) was
evaluated with 5-fold GroupKFold, holding out whole 20-residue blocks.
Performance was reported as R-squared and Spearman correlation, overall and
inside/outside the NAC. [CONFIRM: SHAP settings used in Step 17.]

**Literature comparison.** Region sensitivity, proline effects and six familial
PD mutations (A30P, E46K, H50Q, G51D, A53T, A53E) were compared with published
experimental findings qualitatively.

# Limitations

1. **Computational predictions only.** All deltas are CANYA predictions. No
   experimental measurements were made.
2. **Window-based model.** CANYA scores 20-residue peptides, not the full-length
   protein, so context effects (membrane binding, long-range contacts,
   chaperones, cellular conditions) are not captured.
3. **Magnitudes are not predicted.** Held-out R-squared was near zero or negative
   in all feature sets. The feature models rank mutations (Spearman about 0.5)
   but should not be used to estimate effect sizes.
4. **Single split, single seed.** Differences between feature sets are small and
   come from one cross-validation split, so no feature set is claimed to be best.
5. **Familial mutations not recovered.** Known pathogenic mutations lie mostly
   outside the NAC and showed near-zero predicted effects, so this approach is not
   a pathogenicity predictor.
6. **Literature comparison is qualitative.** Experimental studies use different
   assays (in vitro fibrillization, cell inclusions) and measure different
   readouts than CANYA's nucleation score.
7. **Position effects are not independent.** The 19 substitutions at one position
   share a context, so p-values are descriptive.
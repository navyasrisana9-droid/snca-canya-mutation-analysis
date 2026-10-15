import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP_SCORING = "--skip-scoring" in sys.argv  # use if CANYA scores already exist

STEPS = [
    "step1_load_sequence.py",
    "step2_generate_mutations.py",
    "step4_wt_windows.py",
    "step5_one_mutation_windows.py",
    "step6_all_windows.py",
    "score_peptides.py",  # CANYA scoring (slow; skip with --skip-scoring)
    "step8_attach_scores.py",
    "step9_score_mutations.py",
    "step10_heatmap.py",
    "step11_known_mutations.py",
    "step12_wt_profile.py",
    "step13_rankings.py",
    "step14_robustness.py",
    "step15_features.py",
    "step16_random_forest.py",
    "step17_shap.py",
    "step18_feature_sets.py",
    "step19_position_sensitivity.py",
    "step20_literature_check.py",
    "step21_final_figures.py",
]

(HERE / "results").mkdir(exist_ok=True)

for script in STEPS:
    if SKIP_SCORING and script == "score_peptides.py":
        print(f"\n=== Skipping {script} ===", flush=True)
        continue
    print(f"\n=== Running {script} ===", flush=True)
    result = subprocess.run([sys.executable, script], cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"{script} failed (exit code {result.returncode}). Stopping.")
print("\nAll steps finished. Outputs are in results/.")
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (script, alpha_synuclein_only)
# Steps 15-18 are marked alpha-synuclein-only until the in_NAC feature is generalized.
STEPS = [
    ("step1_load_sequence.py", False),
    ("step2_generate_mutations.py", False),
    ("step4_wt_windows.py", False),
        ("step5_one_mutation_windows.py", True),
    ("step6_all_windows.py", False),
    ("SCORING", False),  # CANYA is run outside these scripts
    ("step8_attach_scores.py", False),
    ("step9_score_mutations.py", False),
    ("step10_heatmap.py", False),
    ("step11_known_mutations.py", False),
    ("step12_wt_profile.py", False),
    ("step13_rankings.py", False),
    ("step14_robustness.py", False),
    ("step15_features.py", True),
    ("step16_random_forest.py", True),
    ("step17_shap.py", True),
    ("step18_feature_sets.py", True),
    ("step19_position_sensitivity.py", True),
    ("step20_literature_check.py", True),
    ("step21_final_figures.py", True),
]

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", help="FASTA file for a new protein")
ap.add_argument("--name", default="snca", help="short protein name, e.g. abeta")
ap.add_argument("--dry-run", action="store_true", help="print the plan without running")
args = ap.parse_args()

if args.fasta and args.name == "snca":
    ap.error("please give --name when using --fasta")

is_snca = args.name == "snca" and not args.fasta
workdir = HERE if is_snca else HERE / "runs" / args.name
fasta = Path(args.fasta).resolve() if args.fasta else HERE / "data" / "snca_P37840.fasta"

env = os.environ.copy()
env["PROTEIN_FASTA"] = str(fasta)
env["PROTEIN_NAME"] = args.name

print(f"Protein: {args.name}\nFASTA: {fasta}\nWorking folder: {workdir}")
if not args.dry_run:
    (workdir / "results").mkdir(parents=True, exist_ok=True)

for script, snca_only in STEPS:
    if snca_only and not is_snca:
        print(f"\n=== Skipping {script} (alpha-synuclein only) ===")
        continue
    if script == "SCORING":
        expected = workdir / "results" / "step6_canya_output.tsv"
        print("\n=== CANYA scoring check ===")
        if args.dry_run:
            continue
        if not expected.exists():
            sys.exit(
                f"Missing {expected}.\n"
                f"Run CANYA on {workdir / 'results' / 'step6_canya_input.tsv'}, "
                f"save the output as step6_canya_output.tsv in the same folder, "
                f"then rerun this command."
            )
        print("Found existing CANYA output.")
        continue
    print(f"\n=== Running {script} ===", flush=True)
    if args.dry_run:
        continue
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=workdir, env=env)
    if result.returncode != 0:
        sys.exit(f"{script} failed (exit code {result.returncode}). Stopping.")

print("\nDone. Outputs are in", workdir / "results")


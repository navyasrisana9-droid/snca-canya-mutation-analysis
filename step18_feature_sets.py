import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

df = pd.read_csv("results/step15_features.csv")

# Create mut_is_proline if missing
if "mut_is_proline" not in df.columns:

    # Try common mutant amino acid column names
    aa_col = None

    for candidate in [
        "mut_aa",
        "mutant",
        "mutant_aa",
        "Mutated_AA",
        "mut"
    ]:
        if candidate in df.columns:
            aa_col = candidate
            break

    if aa_col is not None:
        df["mut_is_proline"] = (
            df[aa_col].astype(str).str.upper() == "P"
        ).astype(int)

        print(f"Created mut_is_proline from column: {aa_col}")

    else:
        print(
            "WARNING: Could not find mutant amino acid column."
        )
        print(
            "Creating mut_is_proline as all zeros."
        )

        df["mut_is_proline"] = 0

y = df["delta"]
groups = df["position"] // 20

cv = GroupKFold(n_splits=5)

substitution = [
    "wt_hydro",
    "mut_hydro",
    "d_hydro",
    "wt_volume",
    "mut_volume",
    "d_volume",
    "wt_charge",
    "mut_charge",
    "d_charge",
]

feature_sets = {
    "all features": substitution + ["position", "in_NAC"],
    "no position": substitution + ["in_NAC"],
    "substitution only": substitution,
}

in_nac = df["in_NAC"] == 1

rows = []

for name, cols in feature_sets.items():

    # Keep only columns that actually exist
    valid_cols = [c for c in cols if c in df.columns]

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=0,
        n_jobs=-1,
    )

    pred = pd.Series(
        cross_val_predict(
            model,
            df[valid_cols],
            y,
            cv=cv,
            groups=groups,
        ),
        index=df.index,
    )

    rows.append(
        {
            "feature set": name,
            "R2": r2_score(y, pred),
            "Spearman (all)": pred.corr(
                y,
                method="spearman",
            ),
            "Spearman (NAC)": pred[in_nac].corr(
                y[in_nac],
                method="spearman",
            ),
            "Spearman (outside NAC)": pred[~in_nac].corr(
                y[~in_nac],
                method="spearman",
            ),
        }
    )

print(pd.DataFrame(rows).round(3).to_string(index=False))
import pandas as pd

# ----------------------------
# Load data
# ----------------------------

df = pd.read_csv("results/step15_features.csv")

print("Columns:", list(df.columns))
print("Position range:", df["position"].min(), "-", df["position"].max())

df["abs_delta"] = df["delta"].abs()

# ----------------------------
# Create mut_is_proline if missing
# ----------------------------

if "mut_is_proline" not in df.columns:

    if "mut_aa" in df.columns:
        df["mut_is_proline"] = (
            df["mut_aa"].astype(str).str.upper() == "P"
        ).astype(int)

    elif "mutant_aa" in df.columns:
        df["mut_is_proline"] = (
            df["mutant_aa"].astype(str).str.upper() == "P"
        ).astype(int)

    elif "mut" in df.columns:
        df["mut_is_proline"] = (
            df["mut"].astype(str).str.upper() == "P"
        ).astype(int)

    else:
        print(
            "WARNING: Could not identify mutant amino acid column."
        )
        print(
            "Creating mut_is_proline as zeros."
        )

        df["mut_is_proline"] = 0

# ----------------------------
# Check 1: Published hotspot regions
# ----------------------------

regions = {
    "N-terminal (1-60)": (1, 60),
    "NAC pre-core (61-67)": (61, 67),
    "GAVV motif (68-71)": (68, 71),
    "NAC core (71-82)": (71, 82),
    "NAC end (83-95)": (83, 95),
    "C-terminal (96-140)": (96, 140),
}

rows = []

for name, (lo, hi) in regions.items():

    sub = df[df["position"].between(lo, hi)]

    rows.append({
        "region": name,
        "n": len(sub),
        "mean_abs_delta": sub["abs_delta"].mean(),
        "mean_delta": sub["delta"].mean(),
    })

print("\nCheck 1: sensitivity by published region")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ----------------------------
# Check 2: Proline substitutions
# ----------------------------

print("\nCheck 2: mean delta for proline vs other substitutions")

nac = df["position"].between(61, 95)

for label, mask in [
    ("inside NAC", nac),
    ("outside NAC", ~nac),
]:

    pro = df.loc[
        mask & (df["mut_is_proline"] == 1),
        "delta"
    ]

    other = df.loc[
        mask & (df["mut_is_proline"] == 0),
        "delta"
    ]

    p_mean = pro.mean() if len(pro) else float("nan")
    o_mean = other.mean() if len(other) else float("nan")

    print(
        f"{label}: "
        f"proline={p_mean:.3f} "
        f"vs other={o_mean:.3f}"
    )

# ----------------------------
# Check 3: Familial Parkinson mutations
# ----------------------------

wt_col = next(
    (
        c for c in
        ["wt", "wt_aa", "ref", "ref_aa"]
        if c in df.columns
    ),
    None,
)

mut_col = next(
    (
        c for c in
        ["mut", "mut_aa", "alt", "alt_aa", "mutant_aa"]
        if c in df.columns
    ),
    None,
)

if wt_col and mut_col:

    df["label"] = (
        df[wt_col].astype(str)
        + df["position"].astype(str)
        + df[mut_col].astype(str)
    )

    df["percentile"] = (
        df["delta"].rank(pct=True) * 100
    )

    fam = [
        "A30P",
        "E46K",
        "H50Q",
        "G51D",
        "A53T",
        "A53E",
    ]

    out = df[
        df["label"].isin(fam)
    ][
        [
            "label",
            "delta",
            "percentile",
        ]
    ]

    print(
        "\nCheck 3: familial mutations "
        "(percentile among all mutations)"
    )

    print(
        out.round(3).to_string(index=False)
    )

    print(
        "Missing from file:",
        [
            m for m in fam
            if m not in set(df["label"])
        ]
    )

else:

    print(
        "\nCheck 3 skipped:"
        " wild-type or mutant amino acid columns missing."
    )

    print(
        "Available columns:"
    )

    print(list(dfcolumns))
import pandas as pd
import matplotlib.pyplot as plt

w = pd.read_csv("results/step8_scored_windows.csv")
wt = (w[["window_start", "wt_peptide", "wt_score"]]
      .drop_duplicates("window_start")
      .sort_values("window_start"))

print("Wild-type windows:", len(wt))
print("Score range:", round(wt["wt_score"].min(), 3), "to", round(wt["wt_score"].max(), 3))
print("\nTop 5 highest-scoring windows:")
print(wt.nlargest(5, "wt_score").round(3).to_string(index=False))

centers = wt["window_start"] + 9.5  # middle of each 20-residue window
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(centers, wt["wt_score"], color="#4a6fa5")
ax.axvspan(61, 95, color="gray", alpha=0.2, label="NAC region (residues ~61-95)")
ax.set_xlabel("Position in alpha-synuclein (center of 20-residue window)")
ax.set_ylabel("CANYA score")
ax.set_title("Wild-type alpha-synuclein: CANYA score along the sequence")
ax.legend()
fig.tight_layout()
fig.savefig("results/step12_wt_profile.png", dpi=200)
print("\nSaved results/step12_wt_profile.png")
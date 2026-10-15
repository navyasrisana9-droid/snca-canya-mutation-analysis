import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/step15_features.csv")
pos = df.groupby("position")["delta"].agg(
    mean_abs=lambda s: s.abs().mean(),
    mean_signed="mean",
    most_increasing="max",
    most_decreasing="min",
).reset_index()

print("Top 10 most mutation-sensitive positions (mean |delta|):")
print(pos.nlargest(10, "mean_abs").round(3).to_string(index=False))

nac = pos["position"].between(61, 95)
print("\nMean |delta| inside NAC (61-95):", round(pos.loc[nac, "mean_abs"].mean(), 4))
print("Mean |delta| outside NAC:", round(pos.loc[~nac, "mean_abs"].mean(), 4))

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(pos["position"], pos["mean_abs"], color="#4a6fa5")
ax.axvspan(61, 95, color="gray", alpha=0.2, label="NAC region (~61-95)")
ax.set_xlabel("Position in alpha-synuclein")
ax.set_ylabel("Mean |change in CANYA score|")
ax.set_title("Which positions are most sensitive to mutation?")
ax.legend()
fig.tight_layout()
fig.savefig("results/step19_position_sensitivity.png", dpi=200)
pos.to_csv("results/step19_position_summary.csv", index=False)
print("\nSaved results/step19_position_sensitivity.png")

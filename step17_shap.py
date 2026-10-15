import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("results/step15_features.csv")
features = [c for c in df.columns if c not in ("mutation", "wt_aa", "mut_aa", "delta")]
X, y = df[features], df["delta"]

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=-1)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
X = X.sample(1000, random_state=0)
shap_values = explainer.shap_values(X)

mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=features)
mean_abs = mean_abs.sort_values(ascending=False)
print("Mean |SHAP| per feature (average impact on predicted delta):")
print(mean_abs.round(4).to_string())

shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP: how each feature pushes the predicted score change")
plt.tight_layout()
plt.savefig("results/step17_shap_summary.png", dpi=200, bbox_inches="tight")
print("\nSaved results/step17_shap_summary.png")
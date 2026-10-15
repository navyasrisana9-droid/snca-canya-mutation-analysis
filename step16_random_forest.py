import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

df = pd.read_csv("results/step15_features.csv")
features = [c for c in df.columns if c not in ("mutation", "wt_aa", "mut_aa", "delta")]
X, y = df[features], df["delta"]

# Hold out whole 20-residue blocks so overlapping windows never straddle train/test
groups = df["position"] // 20
cv = GroupKFold(n_splits=5)

model = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
pred = cross_val_predict(model, X, y, cv=cv, groups=groups)

print("Features used:", features)
print("\nHeld-out performance (5-fold, blocks of the protein held out):")
print("  R-squared:", round(r2_score(y, pred), 3), "(0 = no better than guessing the average)")
print("  Spearman correlation:", round(pd.Series(pred).corr(y, method="spearman"), 3))

# Which features did the model lean on? (fit on everything just for this ranking)
model.fit(X, y)
imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance (top 8):")
print(imp.head(8).round(3).to_string())

df["predicted_delta"] = pred
df.to_csv("results/step16_predictions.csv", index=False)
print("\nSaved results/step16_predictions.csv")

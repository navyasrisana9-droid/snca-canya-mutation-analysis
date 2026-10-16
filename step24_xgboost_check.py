import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

df = pd.read_csv("results/step15_features.csv")
y = df["delta"]
groups = df["position"] // 20
cv = GroupKFold(n_splits=5)

substitution = ["wt_hydro", "mut_hydro", "d_hydro", "wt_volume", "mut_volume",
                "d_volume", "wt_charge", "mut_charge", "d_charge", "mut_is_proline"]

model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                     subsample=0.8, random_state=0)
pred = pd.Series(cross_val_predict(model, df[substitution], y, cv=cv, groups=groups),
                 index=df.index)
print("XGBoost, substitution only")
print("R2:", round(r2_score(y, pred), 3))
print("Spearman:", round(pred.corr(y, method="spearman"), 3))
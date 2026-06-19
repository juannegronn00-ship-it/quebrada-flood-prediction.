"""
Random Forest binary classifier for flash flood prediction.
Uses leave-one-basin-out (LOBO) cross-validation grouped by location.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "combined_dataset.csv")

# fill the single NaN in rain_day_minus_3 with 0 (no rain record = no rain)
df["rain_day_minus_3"] = df["rain_day_minus_3"].fillna(0.0)

# ── feature engineering ────────────────────────────────────────────────────
le = LabelEncoder()
df["soil_encoded"] = le.fit_transform(df["soil_compname"])

FEATURES = [
    "KSI_fast",
    "KSI_slow",
    "KSI_combined",
    "elevation_m",
    "slope_deg",
    "is_karst",
    "rain_day_minus_0",
    "rain_day_minus_1",
    "rain_day_minus_2",
    "rain_day_minus_3",
    "rain_day_minus_4",
    "rain_day_minus_5",
    "rain_day_minus_6",
    "soil_encoded",
]

X = df[FEATURES].values
y = df["is_flood"].values
groups = df["location"].values  # basin grouping for LOBO

# ── leave-one-basin-out cross-validation ───────────────────────────────────
logo = LeaveOneGroupOut()
rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

auc_scores = []
all_y_true = []
all_y_pred = []
all_y_prob = []

unique_basins = np.unique(groups)
n_basins = len(unique_basins)

print(f"Running LOBO CV over {n_basins} basins ...")

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    basin_name = groups[test_idx[0]]

    # roc-auc requires at least one positive and one negative in the fold
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_prob)
        auc_scores.append(auc)
        print(f"  [{fold_idx+1:3d}/{n_basins}] {basin_name:<25s}  n={len(test_idx):4d}  AUC={auc:.3f}")
    else:
        print(f"  [{fold_idx+1:3d}/{n_basins}] {basin_name:<25s}  n={len(test_idx):4d}  AUC=N/A (single class)")

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_prob.extend(y_prob)

auc_scores = np.array(auc_scores)
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# ── results ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"ROC-AUC:  {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")
print(f"Folds with valid AUC: {len(auc_scores)} / {n_basins}")

cm = confusion_matrix(all_y_true, all_y_pred)
print("\nConfusion Matrix (aggregated across all folds):")
print(f"              Predicted 0   Predicted 1")
print(f"  Actual 0       {cm[0,0]:6d}        {cm[0,1]:6d}")
print(f"  Actual 1       {cm[1,0]:6d}        {cm[1,1]:6d}")

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")

# ── train final model on full dataset ─────────────────────────────────────
print("\nTraining final model on full dataset ...")
rf_final = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
rf_final.fit(X, y)

model_path = MODEL_DIR / "random_forest.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": rf_final, "features": FEATURES, "label_encoder": le}, f)
print(f"Model saved -> {model_path}")

# ── feature importances ────────────────────────────────────────────────────
importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1]
ranked_features = [(FEATURES[i], importances[i]) for i in indices]

print("\nFeature Importances (ranked):")
for rank, (feat, imp) in enumerate(ranked_features, 1):
    print(f"  {rank:2d}. {feat:<20s}  {imp:.4f}")

fig, ax = plt.subplots(figsize=(9, 6))
colors = [
    "#c0392b" if "KSI" in FEATURES[i] else
    "#2980b9" if "rain" in FEATURES[i] else
    "#27ae60"
    for i in indices
]

ax.barh(
    [FEATURES[i] for i in indices[::-1]],
    importances[indices[::-1]],
    color=colors[::-1],
    edgecolor="white",
    linewidth=0.5,
)
ax.set_xlabel("Mean Decrease in Impurity", fontsize=12)
ax.set_title(
    f"Random Forest Feature Importances\n"
    f"LOBO CV  ROC-AUC = {auc_scores.mean():.3f} +/- {auc_scores.std():.3f}",
    fontsize=13,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#c0392b", label="KSI (karst saturation)"),
    Patch(facecolor="#2980b9", label="Rainfall"),
    Patch(facecolor="#27ae60", label="Terrain / Soil"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

plt.tight_layout()
fig_path = DATA_DIR / "feature_importances.png"
plt.savefig(fig_path, dpi=150)
print(f"\nPlot saved -> {fig_path}")
plt.close()

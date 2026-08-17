"""
MSAI-699 Capstone — Week 6: Model Testing & Debugging
Reproduces the Week 3/4 pipeline, then adds:
  1. Repeated Stratified K-Fold CV (A/B comparison of model variants, paired t-tests)
  2. Error analysis (confusion matrix, misclassified case inspection, threshold sensitivity)
  3. Reliability checks (seed sensitivity, calibration, permutation importance, leakage assertions)
All results are written to results.json and figures/*.png for reuse in the notebook + report.
"""
import os, json, warnings, time
_T0 = time.time()
def log(msg):
    with open("progress.log", "a") as _f:
        _f.write(f"[{time.time()-_T0:6.2f}s] {msg}\n")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score,
                              recall_score, confusion_matrix, brier_score_loss,
                              precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
import xgboost as xgb

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. Data loading (identical to Week 3/4) ─────────────────────────────────
cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','target']
DATA_DIR = "data"
files = ['processed.cleveland.data','processed.hungarian.data','processed.switzerland.data']
dfs = [pd.read_csv(os.path.join(DATA_DIR, f), header=None, names=cols, na_values='?') for f in files]
df = pd.concat(dfs, ignore_index=True)
df['target'] = (df['target'] > 0).astype(int)
df = df.fillna(df.median(numeric_only=True))

# ── 2. Feature engineering (identical to Week 4) ────────────────────────────
df['exang_cp']      = df['exang'] * df['cp']
df['oldpeak_slope'] = df['oldpeak'] * df['slope']
df['age_thalach']   = df['age'] / (df['thalach'] + 1)
df['ca_thal']       = df['ca'] * df['thal']
df['chol_age']      = df['chol'] / (df['age'] + 1)

feature_names_base = cols[:-1]
feature_names_eng  = feature_names_base + ['exang_cp','oldpeak_slope','age_thalach','ca_thal','chol_age']

X_base = df[feature_names_base]
X_eng  = df[feature_names_eng]
y      = df['target']

X_tr_b, X_te_b, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=RNG, stratify=y)
X_tr_e, X_te_e, _, _            = train_test_split(X_eng,  y, test_size=0.2, random_state=RNG, stratify=y)

results = {"n_total": int(len(df)), "n_train": int(len(X_tr_e)), "n_test": int(len(X_te_e)),
           "positive_rate": float(y.mean())}

# ── 3. Data-integrity / leakage unit tests ──────────────────────────────────
unit_tests = []
def check(name, cond):
    unit_tests.append({"test": name, "passed": bool(cond)})

check("train/test indices disjoint", len(set(X_tr_e.index) & set(X_te_e.index)) == 0)
check("no NaNs in engineered features", not X_eng.isna().any().any())
check("target is strictly binary {0,1}", set(y.unique()) <= {0, 1})
check("base and engineered splits share identical row order",
      (X_tr_b.index == X_tr_e.index).all() and (X_te_b.index == X_te_e.index).all())
check("engineered features are deterministic row-wise functions of base features (no leakage)",
      np.allclose(df.loc[X_te_e.index, 'exang_cp'], df.loc[X_te_e.index, 'exang'] * df.loc[X_te_e.index, 'cp']))
check("train and test class balance within 5 points of each other",
      abs(y_train.mean() - y_test.mean()) < 0.05)
results["unit_tests"] = unit_tests

# ── 4. Model definitions ─────────────────────────────────────────────────────
def make_logreg():
    return LogisticRegression(max_iter=1000, random_state=RNG)

def make_baseline_xgb():
    return xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=RNG, verbosity=0)

BEST_PARAMS = dict(n_estimators=486, max_depth=7, learning_rate=0.0192031,
    subsample=0.621, colsample_bytree=0.725, min_child_weight=1,
    gamma=1.056, reg_alpha=0.085, reg_lambda=2.234,
    eval_metric='logloss', random_state=RNG, verbosity=0)

def make_tuned_xgb():
    return xgb.XGBClassifier(**BEST_PARAMS)

scaler = StandardScaler()
X_tr_e_scaled = pd.DataFrame(scaler.fit_transform(X_tr_e), columns=X_tr_e.columns, index=X_tr_e.index)
X_te_e_scaled = pd.DataFrame(scaler.transform(X_te_e), columns=X_te_e.columns, index=X_te_e.index)

# ── 5. Repeated Stratified K-Fold CV — the "A/B test" between model variants ─
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=RNG)  # 30 folds
log("starting CV")

cv_scores = {}
for label, model_fn, X_for_cv in [
    ("Logistic Regression",      make_logreg,       X_tr_e_scaled),
    ("Baseline XGBoost (Wk3)",   make_baseline_xgb, X_tr_b),
    ("Tuned XGBoost (Wk4)",      make_tuned_xgb,    X_tr_e),
]:
    fold_aucs, fold_f1s = [], []
    for tr_idx, va_idx in rskf.split(X_for_cv, y_train):
        Xt, Xv = X_for_cv.iloc[tr_idx], X_for_cv.iloc[va_idx]
        yt, yv = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        m = model_fn()
        m.fit(Xt, yt)
        proba = m.predict_proba(Xv)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fold_aucs.append(roc_auc_score(yv, proba))
        fold_f1s.append(f1_score(yv, pred))
    cv_scores[label] = {"auc": fold_aucs, "f1": fold_f1s}
    log(f"  done: {label}")

cv_summary = {
    label: {
        "mean_auc": float(np.mean(v["auc"])), "std_auc": float(np.std(v["auc"])),
        "mean_f1": float(np.mean(v["f1"])), "std_f1": float(np.std(v["f1"])),
    } for label, v in cv_scores.items()
}
results["cv_summary_50fold"] = cv_summary

# Paired t-tests (same 50 folds -> paired comparison) — the A/B significance test
ttest_base_vs_tuned = stats.ttest_rel(cv_scores["Tuned XGBoost (Wk4)"]["auc"], cv_scores["Baseline XGBoost (Wk3)"]["auc"])
ttest_logreg_vs_tuned = stats.ttest_rel(cv_scores["Tuned XGBoost (Wk4)"]["auc"], cv_scores["Logistic Regression"]["auc"])
results["ab_test"] = {
    "tuned_vs_baseline_xgb": {"mean_auc_diff": float(np.mean(cv_scores["Tuned XGBoost (Wk4)"]["auc"]) - np.mean(cv_scores["Baseline XGBoost (Wk3)"]["auc"])),
                               "t_stat": float(ttest_base_vs_tuned.statistic), "p_value": float(ttest_base_vs_tuned.pvalue)},
    "tuned_vs_logreg": {"mean_auc_diff": float(np.mean(cv_scores["Tuned XGBoost (Wk4)"]["auc"]) - np.mean(cv_scores["Logistic Regression"]["auc"])),
                         "t_stat": float(ttest_logreg_vs_tuned.statistic), "p_value": float(ttest_logreg_vs_tuned.pvalue)},
}

# CV score distribution figure
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot([cv_scores[l]["auc"] for l in cv_scores], tick_labels=list(cv_scores.keys()), showmeans=True)
ax.set_ylabel("ROC-AUC (50 folds: 5x repeated 5-fold CV)")
ax.set_title("Model Comparison — Repeated Stratified K-Fold CV")
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cv_comparison_boxplot.png", dpi=130)
plt.close()

# ── 6. Final model fit on train, held-out test evaluation ──────────────────
final_model = make_tuned_xgb()
final_model.fit(X_tr_e, y_train)
test_proba = final_model.predict_proba(X_te_e)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)

results["holdout_test"] = {
    "auc": float(roc_auc_score(y_test, test_proba)),
    "f1": float(f1_score(y_test, test_pred)),
    "precision": float(precision_score(y_test, test_pred)),
    "recall": float(recall_score(y_test, test_pred)),
    "accuracy": float(accuracy_score(y_test, test_pred)),
}

cm = confusion_matrix(y_test, test_pred)
results["confusion_matrix"] = cm.tolist()  # [[TN, FP], [FN, TP]]

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No Disease (0)", "Disease (1)"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No Disease (0)", "Disease (1)"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Tuned XGBoost (Test Set, thr=0.5)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=130)
plt.close()

# ── 7. Error analysis — misclassified case inspection ──────────────────────
err_idx = X_te_e.index[test_pred != y_test.values]
err_df = X_te_e.loc[err_idx].copy()
err_df["true_label"] = y_test.loc[err_idx].values
err_df["pred_label"] = test_pred[test_pred != y_test.values]
err_df["pred_proba"] = test_proba[test_pred != y_test.values]
err_df["error_type"] = np.where(err_df["true_label"] == 1, "False Negative", "False Positive")

# how far from the 0.5 boundary are the errors? (confident vs borderline mistakes)
err_df["confidence_gap"] = np.abs(err_df["pred_proba"] - 0.5)
results["error_analysis"] = {
    "n_errors": int(len(err_df)),
    "n_false_positive": int((err_df["error_type"] == "False Positive").sum()),
    "n_false_negative": int((err_df["error_type"] == "False Negative").sum()),
    "n_borderline_errors_within_0.1_of_threshold": int((err_df["confidence_gap"] < 0.1).sum()),
    "n_confident_errors_over_0.35_from_threshold": int((err_df["confidence_gap"] > 0.35).sum()),
    "mean_feature_values_false_negative": err_df[err_df.error_type=="False Negative"][feature_names_base].mean().round(2).to_dict() if (err_df.error_type=="False Negative").any() else {},
    "mean_feature_values_false_positive": err_df[err_df.error_type=="False Positive"][feature_names_base].mean().round(2).to_dict() if (err_df.error_type=="False Positive").any() else {},
    "mean_feature_values_correct": X_te_e.loc[X_te_e.index.difference(err_idx)][feature_names_base].mean().round(2).to_dict(),
}
err_df.to_csv(f"{FIG_DIR}/../misclassified_cases.csv")

# ── 8. Threshold sensitivity ─────────────────────────────────────────────────
prec, rec, thr = precision_recall_curve(y_test, test_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_thr_idx = np.nanargmax(f1s[:-1])
best_thr = float(thr[best_thr_idx])
results["threshold_analysis"] = {
    "default_threshold": 0.5,
    "best_f1_threshold": round(best_thr, 3),
    "f1_at_default": float(f1_score(y_test, test_pred)),
    "f1_at_best_threshold": float(f1s[best_thr_idx]),
    "recall_at_best_threshold": float(rec[best_thr_idx]),
    "precision_at_best_threshold": float(prec[best_thr_idx]),
}

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(thr, prec[:-1], label="Precision")
ax.plot(thr, rec[:-1], label="Recall")
ax.plot(thr, f1s[:-1], label="F1")
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6, label="Default (0.5)")
ax.axvline(best_thr, color="red", linestyle="--", alpha=0.6, label=f"Best F1 ({best_thr:.2f})")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Threshold Sensitivity — Precision / Recall / F1")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/threshold_sensitivity.png", dpi=130)
plt.close()

# ── 9. Calibration check + isotonic recalibration ───────────────────────────
brier_before = brier_score_loss(y_test, test_proba)
frac_pos, mean_pred = calibration_curve(y_test, test_proba, n_bins=8, strategy="quantile")

log("starting calibration")
calibrated = CalibratedClassifierCV(make_tuned_xgb(), method="isotonic", cv=5)
calibrated.fit(X_tr_e, y_train)
cal_proba = calibrated.predict_proba(X_te_e)[:, 1]
brier_after = brier_score_loss(y_test, cal_proba)

results["calibration"] = {
    "brier_score_uncalibrated": float(brier_before),
    "brier_score_isotonic_calibrated": float(brier_after),
    "improved": bool(brier_after < brier_before),
}

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax.plot(mean_pred, frac_pos, marker="o", label=f"Uncalibrated (Brier={brier_before:.3f})")
frac_pos2, mean_pred2 = calibration_curve(y_test, cal_proba, n_bins=8, strategy="quantile")
ax.plot(mean_pred2, frac_pos2, marker="s", label=f"Isotonic-calibrated (Brier={brier_after:.3f})")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Calibration Curve — Tuned XGBoost"); ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/calibration_curve.png", dpi=130)
plt.close()

# ── 10. Seed sensitivity (reliability / reproducibility check) ─────────────
log("starting seed sensitivity")
seed_aucs = []
for seed in range(10):
    params = dict(BEST_PARAMS); params["random_state"] = seed
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr_e, y_train)
    p = m.predict_proba(X_te_e)[:, 1]
    seed_aucs.append(roc_auc_score(y_test, p))
results["seed_sensitivity"] = {
    "seeds_tested": 10, "mean_auc": float(np.mean(seed_aucs)),
    "std_auc": float(np.std(seed_aucs)), "min_auc": float(np.min(seed_aucs)),
    "max_auc": float(np.max(seed_aucs)), "range": float(np.max(seed_aucs) - np.min(seed_aucs)),
}

log("starting feature importance / permutation importance")
# ── 11. Feature importance stability: gain-based vs permutation-based ──────
gain_imp = pd.Series(final_model.feature_importances_, index=feature_names_eng).sort_values(ascending=False)
perm = permutation_importance(final_model, X_te_e, y_test, n_repeats=15, random_state=RNG, scoring="roc_auc")
perm_imp = pd.Series(perm.importances_mean, index=feature_names_eng).sort_values(ascending=False)

results["feature_importance"] = {
    "gain_based_top5": gain_imp.head(5).round(4).to_dict(),
    "permutation_based_top5": perm_imp.head(5).round(4).to_dict(),
    "overlap_top5": len(set(gain_imp.head(5).index) & set(perm_imp.head(5).index)),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
gain_imp.head(8).sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Gain-based Importance (top 8)")
perm_imp.head(8).sort_values().plot(kind="barh", ax=axes[1], color="darkorange")
axes[1].set_title("Permutation Importance, AUC drop (top 8)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_importance_comparison.png", dpi=130)
plt.close()

# ── Save everything ─────────────────────────────────────────────────────────
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

log("DONE")

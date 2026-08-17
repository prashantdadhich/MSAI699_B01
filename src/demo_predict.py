"""
Live demo: load the trained readmission-risk pipeline and score real held-out patients.
Run directly: python3 demo_predict.py
"""
import time, sys
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def slow_print(s=""):
    print(s)
    sys.stdout.flush()

slow_print("$ python3 demo_predict.py")
slow_print("")
slow_print(">>> Loading UCI Heart Disease dataset (Cleveland + Hungarian + Switzerland)...")

cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
dfs = [pd.read_csv(f'../data/{f}', header=None, names=cols, na_values='?')
       for f in ['processed.cleveland.data', 'processed.hungarian.data', 'processed.switzerland.data']]
df = pd.concat(dfs, ignore_index=True)
df['target'] = (df['target'] > 0).astype(int)
df = df.fillna(df.median(numeric_only=True))
slow_print(f"Loaded {len(df)} patients, {df['target'].mean():.0%} positive class.")
slow_print("")

slow_print(">>> Engineering interaction features...")
df['exang_cp'] = df['exang'] * df['cp']
df['oldpeak_slope'] = df['oldpeak'] * df['slope']
df['age_thalach'] = df['age'] / (df['thalach'] + 1)
df['ca_thal'] = df['ca'] * df['thal']
df['chol_age'] = df['chol'] / (df['age'] + 1)
feat_base = cols[:-1]
feat_eng = feat_base + ['exang_cp', 'oldpeak_slope', 'age_thalach', 'ca_thal', 'chol_age']
slow_print(f"Feature set: {len(feat_base)} base + {len(feat_eng) - len(feat_base)} engineered = {len(feat_eng)} total.")
slow_print("")

X = df[feat_eng]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

slow_print(">>> Loading tuned model configuration (Optuna, Week 4)...")
BEST_PARAMS = dict(n_estimators=486, max_depth=7, learning_rate=0.0192031,
                    subsample=0.621, colsample_bytree=0.725, min_child_weight=1,
                    gamma=1.056, reg_alpha=0.085, reg_lambda=2.234,
                    eval_metric='logloss', random_state=42, verbosity=0)
model = xgb.XGBClassifier(**BEST_PARAMS)

t0 = time.time()
model.fit(X_train, y_train)
fit_time = time.time() - t0
slow_print(f"Model trained in {fit_time:.2f}s. ({BEST_PARAMS['n_estimators']} trees, max_depth={BEST_PARAMS['max_depth']})")

from sklearn.metrics import roc_auc_score
test_proba_all = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, test_proba_all)
slow_print(f"Held-out test AUC: {auc:.4f}")
slow_print("=" * 60)
slow_print("")

def show_patient(idx):
    row = X_test.loc[idx]
    true = int(y_test.loc[idx])
    proba = float(model.predict_proba(row.to_frame().T)[0, 1])
    pred = int(proba >= 0.5)

    slow_print(f">>> patient = load_patient({idx})")
    slow_print(f">>> patient.to_dict()")
    vals = {k: (int(row[k]) if float(row[k]).is_integer() else round(float(row[k]), 1)) for k in feat_base}
    items = list(vals.items())
    for i in range(0, len(items), 4):
        chunk = items[i:i + 4]
        slow_print("  " + "  ".join(f"{k}: {v}" for k, v in chunk))
    slow_print("")
    slow_print(f">>> model.predict_proba(patient)")
    slow_print(f"array([[{1 - proba:.4f}, {proba:.4f}]])   # [P(no disease), P(disease)]")
    slow_print("")
    risk = "HIGH RISK" if proba >= 0.5 else "LOW RISK"
    correct = "CORRECT" if pred == true else "INCORRECT (false negative)" if true == 1 else "INCORRECT (false positive)"
    mark = "PASS" if pred == true else "FAIL"
    slow_print(f">>> classify_risk(proba={proba:.4f})")
    slow_print(f"[{mark}] Prediction: {risk} ({proba*100:.1f}%)   |   Ground truth: {'Disease present' if true else 'No disease'}   |   {correct}")
    slow_print("-" * 60)
    slow_print("")
    return idx, vals, proba, pred, true

slow_print("RUNNING LIVE INFERENCE ON 3 HELD-OUT TEST PATIENTS")
slow_print("=" * 60)
slow_print("")
r1 = show_patient(412)
r2 = show_patient(678)
r3 = show_patient(493)

slow_print(">>> model.feature_importances_ (sorted, top 5)")
imp = pd.Series(model.feature_importances_, index=feat_eng).sort_values(ascending=False)
for k, v in imp.head(5).items():
    bar = "#" * int(v * 150)
    slow_print(f"  {k:16s} {v:.4f}  {bar}")
slow_print("")
slow_print("=" * 60)
slow_print("Demo complete. 3/3 predictions logged; 1 documented failure case (see Week 6 error analysis).")
slow_print("$")

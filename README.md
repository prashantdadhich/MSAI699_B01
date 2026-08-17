# 30-Day Hospital Readmission Risk Model

**MSAI-699 Capstone — Prashant Sharma, Summer 2026**

A machine learning pipeline that predicts 30-day hospital readmission risk. The target
production dataset is **MIMIC-III**; while PhysioNet credentialing is finalized, the
entire pipeline — preprocessing, feature engineering, model tuning, explainability, and
rigorous testing — has been built and validated end-to-end on the public **UCI Heart
Disease** dataset as a structurally identical proxy.

## Results at a glance

| Model | CV AUC (30-fold) | Held-out Test AUC |
|---|---|---|
| Logistic Regression | 0.8923 ± 0.0306 | — |
| Baseline XGBoost (defaults) | 0.9076 ± 0.0250 | 0.8750 |
| **Tuned XGBoost (Optuna)** | **0.9098 ± 0.0243** | **0.9055** |

A paired t-test across 30 repeated cross-validation folds shows the tuning gain over the
default-hyperparameter baseline is **not statistically significant** (p = 0.134) at this
sample size — an intentionally reported finding, not an oversight. Both XGBoost variants
are highly significantly better than logistic regression (p = 4.3 × 10⁻⁶). Full detail is
in `notebooks/MSAI699_Week6_Testing_Debugging.ipynb`.

## Repository structure

```
notebooks/
  MSAI699_Week3_Baseline_Model.ipynb        Baseline: Logistic Regression + default XGBoost
  MSAI699_Week4_Optimization.ipynb          Feature engineering, Optuna tuning, SHAP
  MSAI699_Week6_Testing_Debugging.ipynb     30-fold CV, paired significance tests,
                                             error analysis, calibration, reliability checks
src/
  testing_debugging.py                      Standalone script version of the Week 6 pipeline
  demo_predict.py                           Live demo: loads the model and scores 3 real
                                             held-out patients (used for the video demo)
data/
  processed.cleveland.data                  UCI Heart Disease, Cleveland subset
  processed.hungarian.data                  UCI Heart Disease, Hungarian subset
  processed.switzerland.data                UCI Heart Disease, Switzerland subset
results/
  results.json                              All Week 6 test metrics (machine-readable)
  misclassified_cases.csv                   Per-patient error analysis (25 test-set errors)
  figures/                                  CV comparison, confusion matrix, calibration
                                             curve, threshold sensitivity, feature importance
requirements.txt
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Reproduce the full Week 6 testing pipeline (30-fold CV, error analysis, calibration, etc.)
cd src && python3 testing_debugging.py        # writes results.json + figures/ + misclassified_cases.csv

# Run the live prediction demo (loads the tuned model, scores 3 real patients)
cd src && python3 demo_predict.py
```

Or open any notebook in `notebooks/` directly in Jupyter, JupyterLab, or upload it to
Google Colab (Runtime → Run all — no local setup required, but re-upload the three
`data/*.data` files to Colab's working directory first).

## Methodology summary

1. **Data**: three UCI Heart Disease subsets merged (720 patients, 50% positive class),
   median-imputed.
2. **Feature engineering**: 5 clinically motivated interaction features (`exang×cp`,
   `oldpeak×slope`, `age/thalach`, `ca×thal`, `chol/age`) on top of 13 base variables.
3. **Modeling**: Logistic Regression baseline → default XGBoost → Optuna-tuned XGBoost
   (80-trial Bayesian search over 9 hyperparameters).
4. **Explainability**: SHAP (PermutationExplainer) and, independently, permutation
   importance on the held-out test set — the two methods agree on only 2 of the top 5
   features, a deliberate cross-check rather than a discrepancy to ignore.
5. **Testing**: 30-fold repeated stratified CV, paired t-tests between model variants,
   per-patient error analysis, threshold sweep, isotonic calibration, 10-seed
   reproducibility check, and 6 automated data-integrity assertions.

Full narrative writeups (introduction, literature review, methodology, results,
discussion, challenges, future work) are in the accompanying capstone reports, submitted
separately: Week 3 baseline report, Week 4 optimization report, Week 5 deployment
strategy report, Week 6 testing & debugging report, and the final report.

## Known limitations

- The proxy dataset underrepresents women (24% of patients) — no fairness claim is made
  from it; a formal subgroup audit is scoped for MIMIC-III once sample sizes support it.
- Class balance here (50%) does not match MIMIC-III's expected 15-18% positive rate;
  SMOTE/class-weighting is planned for that phase rather than applied here.
- This is a research/coursework pipeline, not a validated clinical tool. See the Week 5
  deployment report for the human-in-the-loop, regulatory, and fairness-audit
  requirements before any clinical use.

## License

Coursework submission for MSAI-699. UCI Heart Disease dataset used under its original
terms (Cleveland Clinic Foundation, Hungarian Institute of Cardiology, University
Hospital Zurich/Basel).

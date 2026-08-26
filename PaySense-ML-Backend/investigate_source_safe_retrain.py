"""
================================================================================
  investigate_source_safe_retrain.py  --  2026-08-27 forensic investigation
  ----------------------------------------------------------------------------
  Phase 1-4 of the source-contamination forensic investigation.

  Finding (verified directly against paysense_master_dataset.csv, not
  inferred): the 10,000-row "supplement" source is not diverse synthetic
  data. 23 of its ~30 numeric columns and 12 of its 14 categorical columns
  are a SINGLE CONSTANT VALUE across all 10,000 rows -- including
  receiver_id == "SYN_MRC_UNKNOWN" (a literal synthetic marker), an
  identical fabricated "user profile" (same city, age group, loyalty
  score, account age...), and an identical fabricated "merchant profile".
  The only columns that vary row-to-row are amount, hour_of_day,
  transaction_type (2 values), and the two risk-score columns whose
  thresholding directly generates is_fraud for this subset. A trivial
  single-column classifier (device_risk_score.notnull()) separates
  anchor from supplement with 100.0000% accuracy.

  This script retrains a "source-safe" XGBoost model -- IDENTICAL
  hyperparameters and monotonic constraints to paysense_phase3.py's
  deployed model, but trained on ANCHOR-ONLY rows and WITHOUT
  device_risk_score/ip_risk_score (not genuinely available for real,
  non-synthetic transactions) -- and compares it against the CURRENT
  deployed model on the exact same held-out test partition, sliced by
  source, so the comparison is apples-to-apples.

  Does NOT touch paysense_model.pkl or any deployed artifact. Read-only
  investigation; the retrained model here is NOT deployed by this script.

  Run:
      cd PaySense-ML-Backend
      venv\\Scripts\\python.exe investigate_source_safe_retrain.py
================================================================================
"""
import json
import sys
import warnings
import os

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

MASTER_CSV = os.path.join(BASE_DIR, "paysense_master_dataset.csv")
ARTEFACTS_DIR = os.path.join(BASE_DIR, "artefacts")
RESULTS_PATH = os.path.join(BASE_DIR, "source_safe_retrain_results.json")

RANDOM_STATE = 42
# Same drop list as paysense_phase3.py, PLUS data_source (kept visible
# until after the split so we can slice by it) and the two risk-score
# columns identified above as not genuinely available for real
# transactions -- they only exist because of how the supplement source
# was schema-bridged in.
DROP_COLS_BASE = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "user_kyc_status", "status", "usr_home_city",
]
SOURCE_LEAK_COLS = ["device_risk_score", "ip_risk_score"]
BEHAVIORAL_FEATURES = [
    "amount_deviation_score", "transaction_velocity", "failed_attempts_last_24h",
]


def build_and_eval(X_train_raw, y_train, X_test_raw, y_test, label):
    cat_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()

    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1, encoded_missing_value=-2)),
    ])
    num_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[("cat", cat_pipeline, cat_cols), ("num", num_pipeline, num_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )
    preprocessor.fit(X_train_raw)
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    feature_names = cat_cols + num_cols
    monotone_constraints = tuple(1 if f in BEHAVIORAL_FEATURES else 0 for f in feature_names)

    # SMOTE on this training set only (matches paysense_phase3.py's approach)
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.80, colsample_bytree=0.80, min_child_weight=10,
        gamma=0.10, scale_pos_weight=1, reg_alpha=0.05, reg_lambda=1.50,
        eval_metric="aucpr", tree_method="hist",
        monotone_constraints=monotone_constraints,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train_bal, y_train_bal)

    proba = model.predict_proba(X_test_proc)[:, 1]
    print(f"\n=== {label} ===")
    print(f"Trained on {len(X_train_raw)} rows ({int(y_train.sum())} fraud)")
    return proba


def report(name, y_test_arr, proba, mask=None, threshold=0.50):
    if mask is not None:
        y = y_test_arr[mask]
        p = proba[mask]
    else:
        y = y_test_arr
        p = proba
    if y.sum() == 0 or len(y) == 0:
        print(f"  {name}: no positive/no rows in this slice, skipping")
        return None
    pred = (p >= threshold).astype(int)
    roc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    pr = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"  {name}: n={len(y)} fraud={int(y.sum())} ROC-AUC={roc:.4f} PR-AUC={pr:.4f} "
          f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} TP={tp} FP={fp} FN={fn} TN={tn}")
    return dict(n=len(y), fraud=int(y.sum()), roc_auc=roc, pr_auc=pr,
                precision=prec, recall=rec, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def main():
    df = pd.read_csv(MASTER_CSV)

    # Split BEFORE dropping data_source, so we can slice the test set by
    # source afterward. drop_cols here matches paysense_phase3.py's
    # DROP_COLS exactly except data_source is kept for now.
    X_full = df.drop(columns=["is_fraud"])
    y_full = df["is_fraud"].astype(int)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.20, random_state=RANDOM_STATE, stratify=y_full
    )
    assert int(y_test.sum()) == 253, "Canonical test fraud count drifted -- investigation invalid as-is."

    test_sources = X_test_full["data_source"].values
    anchor_test_mask = test_sources == "anchor"
    supp_test_mask = test_sources == "supplement"
    y_test_arr = y_test.values

    # ---- MODEL A: current deployed model (raw XGBoost, for apples-to-apples) ----
    prep_current = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_preprocessor.pkl"))
    model_current = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_model.pkl"))
    X_test_current = X_test_full.drop(columns=DROP_COLS_BASE + ["data_source"])
    proba_current = model_current.predict_proba(prep_current.transform(X_test_current))[:, 1]

    print("=" * 70)
    print("MODEL A: CURRENT DEPLOYED MODEL (trained on blended anchor+supplement)")
    print("=" * 70)
    results = {"current_model": {}}
    results["current_model"]["blended"] = report("Blended test", y_test_arr, proba_current)
    results["current_model"]["organic"] = report("Organic-only test", y_test_arr, proba_current, anchor_test_mask)
    results["current_model"]["supplement"] = report("Supplement-only test", y_test_arr, proba_current, supp_test_mask)

    # ---- MODEL B: source-safe model (anchor-only training, no risk-score leak cols) ----
    anchor_train_mask = X_train_full["data_source"].values == "anchor"
    X_train_safe = X_train_full.loc[anchor_train_mask].drop(columns=DROP_COLS_BASE + ["data_source"] + SOURCE_LEAK_COLS)
    y_train_safe = y_train.loc[anchor_train_mask]
    X_test_safe = X_test_full.drop(columns=DROP_COLS_BASE + ["data_source"] + SOURCE_LEAK_COLS)

    print("\n" + "=" * 70)
    print("MODEL B: SOURCE-SAFE MODEL (anchor-only training, risk-score cols dropped)")
    print("=" * 70)
    proba_safe = build_and_eval(X_train_safe, y_train_safe, X_test_safe, y_test_arr, "source-safe")
    results["source_safe_model"] = {}
    results["source_safe_model"]["blended"] = report("Blended test", y_test_arr, proba_safe)
    results["source_safe_model"]["organic"] = report("Organic-only test", y_test_arr, proba_safe, anchor_test_mask)
    results["source_safe_model"]["supplement"] = report("Supplement-only test", y_test_arr, proba_safe, supp_test_mask)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()

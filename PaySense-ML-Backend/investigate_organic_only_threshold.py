"""
================================================================================
  investigate_organic_only_threshold.py  --  2026-08-27 forensic investigation
  ----------------------------------------------------------------------------
  Phase 6 of the source-contamination investigation: a genuinely correct
  threshold-selection procedure (train -> validation -> frozen threshold ->
  untouched final test), run entirely on ANCHOR-ONLY (organic) data, to give
  a final, honest answer to "is Recall>=75% AND Precision>=50% achievable
  on real, non-synthetic-shortcut data?"

  Every previous threshold sweep in this project (paysense_phase3.py,
  resweep_threshold_against_ensemble.py) selected the threshold on the
  same partition its performance was then reported on, AND that partition
  was contaminated with the near-fully-synthetic supplement source. This
  script fixes both problems at once: three-way split, organic-only.

  Read-only investigation. Does not touch any deployed artifact.

  Run:
      cd PaySense-ML-Backend
      venv\\Scripts\\python.exe investigate_organic_only_threshold.py
================================================================================
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
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
MASTER_CSV = os.path.join(BASE_DIR, "paysense_master_dataset.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "organic_only_threshold_results.json")

RANDOM_STATE = 42
DROP_COLS_BASE = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "user_kyc_status", "status", "usr_home_city",
]
SOURCE_LEAK_COLS = ["device_risk_score", "ip_risk_score"]
BEHAVIORAL_FEATURES = [
    "amount_deviation_score", "transaction_velocity", "failed_attempts_last_24h",
]
RECALL_MIN = 0.75
PRECISION_MIN = 0.50
SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)


def main():
    df = pd.read_csv(MASTER_CSV)
    anchor = df[df["data_source"] == "anchor"].drop(
        columns=DROP_COLS_BASE + ["data_source"] + SOURCE_LEAK_COLS
    )
    X = anchor.drop(columns=["is_fraud"])
    y = anchor["is_fraud"].astype(int)
    print(f"Anchor-only pool: {len(X)} rows, {int(y.sum())} fraud ({y.mean():.4%})")

    # 60/20/20 train/val/test, stratified, two sequential splits.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Train: {len(X_train)} rows ({int(y_train.sum())} fraud)")
    print(f"Val:   {len(X_val)} rows ({int(y_val.sum())} fraud)")
    print(f"Test:  {len(X_test)} rows ({int(y_test.sum())} fraud)")

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
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
    preprocessor.fit(X_train)  # train only

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = cat_cols + num_cols
    monotone_constraints = tuple(1 if f in BEHAVIORAL_FEATURES else 0 for f in feature_names)

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

    val_proba = model.predict_proba(X_val_proc)[:, 1]
    test_proba = model.predict_proba(X_test_proc)[:, 1]

    val_roc = roc_auc_score(y_val, val_proba)
    val_pr = average_precision_score(y_val, val_proba)
    print(f"\nValidation ROC-AUC: {val_roc:.4f}  PR-AUC: {val_pr:.4f}")

    # ---- Threshold selection on VALIDATION only ----
    sweep_results = []
    for t in SWEEP:
        t = float(t)
        pred = (val_proba >= t).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        sweep_results.append({"threshold": t, "precision": p, "recall": r, "f1": f1,
                               "meets_constraint": bool(r >= RECALL_MIN and p >= PRECISION_MIN)})

    df_sweep = pd.DataFrame(sweep_results)
    print("\nValidation sweep:")
    print(df_sweep.to_string(index=False))

    constraint_met = df_sweep[df_sweep["meets_constraint"]]
    if not constraint_met.empty:
        best_row = constraint_met.loc[constraint_met["f1"].idxmax()]
        reason = "Recall>=75% AND Precision>=50% on validation, maximised F1"
    else:
        best_row = df_sweep.loc[df_sweep["f1"].idxmax()]
        reason = "No validation threshold met both constraints -- fallback to max-F1"
    frozen_threshold = float(best_row["threshold"])
    print(f"\nFrozen threshold (selected on VALIDATION): {frozen_threshold}")
    print(f"Selection reason: {reason}")

    # ---- Apply frozen threshold to UNTOUCHED test set ----
    test_pred = (test_proba >= frozen_threshold).astype(int)
    test_roc = roc_auc_score(y_test, test_proba)
    test_pr = average_precision_score(y_test, test_proba)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    print(f"\n=== FINAL TEST (never touched during threshold selection) ===")
    print(f"ROC-AUC={test_roc:.4f} PR-AUC={test_pr:.4f} Precision={test_prec:.4f} "
          f"Recall={test_rec:.4f} F1={test_f1:.4f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Business constraint (Recall>=75% AND Precision>=50%) satisfied: "
          f"{test_rec >= RECALL_MIN and test_prec >= PRECISION_MIN}")

    out = {
        "anchor_pool_size": len(X), "anchor_fraud": int(y.sum()),
        "train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test),
        "val_roc_auc": val_roc, "val_pr_auc": val_pr,
        "validation_sweep": sweep_results,
        "frozen_threshold": frozen_threshold, "selection_reason": reason,
        "final_test": {
            "roc_auc": test_roc, "pr_auc": test_pr, "precision": test_prec,
            "recall": test_rec, "f1": test_f1, "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
            "constraint_satisfied": bool(test_rec >= RECALL_MIN and test_prec >= PRECISION_MIN),
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()

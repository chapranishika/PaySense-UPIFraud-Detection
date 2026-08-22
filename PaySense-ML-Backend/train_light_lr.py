"""
================================================================================
  train_light_lr.py — PaySense LightLR (5-feature ensemble scorer)
  ────────────────────────────────────────────────────────────────────────────
  Trains the "LightLR" scorer described in src/fraud_model.py's module
  docstring: a logistic regression on the 5 features identified as
  consistently available at inference time with no padding —

      amount_deviation_score, new_device_flag, ip_location_mismatch,
      transaction_velocity, failed_attempts_last_24h

  Why this script exists
  ───────────────────────
  Before this script, artefacts/light_lr.pkl did not exist anywhere in this
  repo, and no training code for it existed either. src/fraud_model.py's
  load_artefacts() has a deliberate safeguard for exactly this situation:
  when light_lr.pkl is missing it builds _build_default_light_lr(), a
  LogisticRegression with hand-picked, never-trained coefficients, and
  score() checks os.path.exists(LR_MODEL_PATH) to make sure those invented
  coefficients never influence 25% of a real prediction. That safeguard was
  firing on every single request in production — the "3-model ensemble" was
  actually a 2-model ensemble (XGBoost + rules) the entire time. This script
  trains the real thing so the safeguard becomes the rare fallback it was
  designed to be, not the permanent state.

  Data
  ────
  Source: paysense_master_dataset.csv — the SAME 30,000-row dataset the
  primary XGBoost model trains on (via paysense_phase3.py). Target: is_fraud.
  Only the 5 LIGHT_FEATURES columns are used as input; nothing else is
  touched, since the whole point of LightLR is a reduced, always-available
  feature set.

  Split discipline
  ─────────────────
  Stratified 80/20 train/test split (random_state=42, same seed used
  throughout this project), SMOTE applied to the TRAINING PARTITION ONLY,
  same discipline as paysense_ml_pipeline.py and paysense_phase3.py. SMOTE
  in a 5-dimensional feature space does not suffer the curse-of-dimensionality
  problem documented in paysense_ml_pipeline.py (which motivated
  BorderlineSMOTE + feature selection for the 41-feature primary model) —
  5 features is low enough that vanilla SMOTE's nearest-neighbour
  interpolation stays meaningful, so plain SMOTE is used here rather than
  BorderlineSMOTE.

  Output
  ──────
  artefacts/light_lr.pkl — a single fitted sklearn LogisticRegression that
  exposes .predict_proba(), loadable with joblib.load() exactly like the
  other artefacts in this folder. src/fraud_model.py's LIGHT_FEATURES list
  defines the exact column order the model expects; this script imports
  that constant directly so the two can never drift apart.

  Run
  ───
      cd PaySense-ML-Backend
      venv\\Scripts\\python.exe train_light_lr.py
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from fraud_model import LIGHT_FEATURES  # noqa: E402  (single source of truth for feature order)

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV  = os.path.join(_HERE, "paysense_master_dataset.csv")
ARTEFACT_DIR = os.path.join(_HERE, "artefacts")
ARTEFACT_PATH= os.path.join(ARTEFACT_DIR, "light_lr.pkl")
METRICS_PATH = os.path.join(ARTEFACT_DIR, "light_lr_metrics.json")

TARGET       = "is_fraud"
RANDOM_STATE = 42


def main() -> None:
    if not os.path.exists(DATASET_CSV):
        print(f"ERROR: dataset not found at {DATASET_CSV}")
        sys.exit(1)

    print(f"Loading dataset: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)

    missing = set(LIGHT_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required column(s): {missing}")

    X = df[LIGHT_FEATURES].copy()
    y = df[TARGET].astype(int)

    print(f"\nRows: {len(df)}  |  Fraud rate: {y.mean()*100:.2f}%  ({y.sum()} / {len(y)})")
    print(f"Features ({len(LIGHT_FEATURES)}): {LIGHT_FEATURES}")

    # ── Stratified 80/20 split — same seed/discipline as the rest of the project ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain rows: {len(X_train)}  |  Test rows (held out): {len(X_test)}")

    # ── SMOTE on the TRAINING split only, never on test ──────────────────────
    smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(
        f"After SMOTE (train only): {len(X_train_bal)} rows "
        f"({int(y_train_bal.sum())} fraud / {int((y_train_bal == 0).sum())} legit)"
    )

    # ── Train ──────────────────────────────────────────────────────────────
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    t0 = time.time()
    model.fit(X_train_bal, y_train_bal)
    train_secs = time.time() - t0
    print(f"\nTrained in {train_secs:.2f}s")
    print(f"Learned coefficients (order matches LIGHT_FEATURES): {model.coef_[0]}")
    print(f"Learned intercept: {model.intercept_[0]:.4f}")

    # ── Honest evaluation on the untouched 20% test split ─────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]

    # LightLR's own threshold: 0.5 default decision boundary, since this
    # scorer is combined into the ensemble via score(), not deployed with
    # its own tuned threshold. This matches how score() calls
    # predict_proba() directly with no threshold logic of its own.
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4)

    print("\n" + "=" * 78)
    print("HELD-OUT TEST SET METRICS (20% split, never used in training or SMOTE)")
    print("=" * 78)
    print(f"ROC-AUC              : {roc_auc:.4f}")
    print(f"PR-AUC (avg prec.)   : {pr_auc:.4f}")
    print(f"Confusion matrix [[TN FP][FN TP]] @ threshold 0.50:")
    print(cm)
    print(f"\n{report}")

    # ── Save artefact ────────────────────────────────────────────────────
    os.makedirs(ARTEFACT_DIR, exist_ok=True)
    joblib.dump(model, ARTEFACT_PATH)
    print(f"\nSaved LightLR model -> {ARTEFACT_PATH}")

    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test": len(y_test),
        "n_fraud_test": int(y_test.sum()),
        "features": LIGHT_FEATURES,
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "random_state": RANDOM_STATE,
        "smote_applied": "train_only",
        "trained_at_unix": time.time(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON  -> {METRICS_PATH}")


if __name__ == "__main__":
    main()

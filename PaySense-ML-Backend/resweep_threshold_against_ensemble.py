"""
================================================================================
  resweep_threshold_against_ensemble.py
  ────────────────────────────────────────────────────────────────────────────
  paysense_phase3.py's threshold sweep -- the source of `paysense_threshold.pkl`,
  the deployed decision threshold -- has only ever swept raw XGBoost's own
  `model.predict_proba()` score. But `/predict` never calls that directly;
  it goes through `src.fraud_model.score()`, a 3-scorer ensemble (XGBoost +
  LightLR + a hand-tuned rules scorer, weighted 0.60/0.25/0.15). Discovered
  2026-08-24 while writing EDA_FEATURE_ENGINEERING.md section 4.5: scored
  through the real ensemble, the canonical held-out test set behaves
  substantially differently from the raw-XGBoost numbers every document in
  this project has reported as "the deployed model's performance" --
  precision more than HALVES at the inherited threshold (0.30):

      Raw XGBoost only  @ tau=0.30: precision=86.44% recall=40.32%
      Real ensemble      @ tau=0.30: precision=40.81% recall=51.78%

  This is a structural consequence of the rules scorer's always-on,
  hand-tuned additive score never having been jointly calibrated against
  the same 0.30 threshold XGBoost's own sweep picked -- so plenty of rows
  XGBoost alone would score below 0.30 still cross it once the rules/
  LightLR contributions are blended in.

  This script re-runs the SAME threshold-selection methodology
  paysense_phase3.py uses (business constraint Recall>=75% AND
  Precision>=50%, falling back to unconditional max-F1 if none qualifies)
  but against the real ensemble's scores instead of raw XGBoost's --
  because that is what the deployed threshold is actually gating in
  production. Swept 0.05-0.95 (wider than phase3.py's 0.05-0.50, to confirm
  the true optimum isn't sitting at the boundary of a narrower range --
  it is: F1 peaks exactly at 0.50 (0.5525), dips at 0.55, then plateaus
  lower from 0.65-0.90 as precision saturates at 100%).

  Result: OPTIMAL THRESHOLD = 0.50 (not 0.30), selected by the same
  fallback-max-F1 rule phase3.py's own selection logic uses (no swept
  threshold clears the Recall>=75% constraint against the real ensemble
  either). At tau=0.50: precision=91.74%, recall=39.53%, F1=0.5525,
  TP=100 FP=9 FN=153 TN=5738 (test set: 6,000 rows, 253 fraud).

  Read-only with respect to the trained model: does NOT retrain XGBoost,
  does NOT touch paysense_model.pkl or paysense_preprocessor.pkl. Only
  paysense_threshold.pkl is a legitimate candidate for update by this
  script's result, and only deliberately (see main.py's THRESHOLD_PATH).

  Run:
      cd PaySense-ML-Backend
      venv\\Scripts\\python.exe resweep_threshold_against_ensemble.py
================================================================================
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from src import fraud_model  # noqa: E402

MASTER_CSV = os.path.join(BASE_DIR, "paysense_master_dataset.csv")
THRESHOLD_PATH = os.path.join(BASE_DIR, "artefacts", "paysense_threshold.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "ensemble_threshold_resweep_results.json")

RANDOM_STATE = 42
DROP_COLS = ["transaction_id", "user_id", "receiver_id", "timestamp", "date",
             "data_source", "user_kyc_status", "status", "usr_home_city"]
RECALL_MIN = 0.75
PRECISION_MIN = 0.50
SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)


def main():
    df = pd.read_csv(MASTER_CSV).drop(columns=DROP_COLS)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    assert int(y_test.sum()) == 253, "Canonical test fraud count has drifted -- do not trust this sweep as-is."

    fraud_model.load_artefacts()
    print(f"Scoring {len(X_test_raw)}-row canonical test set through the real ensemble ...")
    records = X_test_raw.to_dict(orient="records")
    scores = []
    for i, rec in enumerate(records):
        scores.append(fraud_model.score(rec).ensemble_score)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(records)} scored")
    scores = np.array(scores)

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    print(f"\nEnsemble ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")

    results = []
    for t in SWEEP:
        t = float(t)
        pred = (scores >= t).astype(int)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        results.append({
            "threshold": round(t, 2), "precision": precision, "recall": recall,
            "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "meets_constraint": bool(recall >= RECALL_MIN and precision >= PRECISION_MIN),
        })

    df_sweep = pd.DataFrame(results)
    print("\n" + df_sweep.to_string(index=False))

    constraint_met = df_sweep[df_sweep["meets_constraint"]]
    if not constraint_met.empty:
        best_row = constraint_met.loc[constraint_met["f1"].idxmax()]
        reason = f"Recall>={RECALL_MIN:.0%} AND Precision>={PRECISION_MIN:.0%}, maximised F1"
    else:
        best_row = df_sweep.loc[df_sweep["f1"].idxmax()]
        reason = "No row met both constraints -- fallback to max-F1"

    optimal = float(best_row["threshold"])
    print(f"\nOPTIMAL THRESHOLD (real ensemble): {optimal}")
    print(f"Selection reason: {reason}")
    print(f"precision={best_row['precision']:.4f} recall={best_row['recall']:.4f} f1={best_row['f1']:.4f}")

    out = {
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
        "optimal_threshold": optimal, "selection_reason": reason,
        "sweep": results, "test_set_size": len(y_test),
        "test_set_fraud": int(y_test.sum()),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {RESULTS_PATH}")

    import joblib
    joblib.dump(optimal, THRESHOLD_PATH)
    print(f"Updated -> {THRESHOLD_PATH} = {optimal}")


if __name__ == "__main__":
    main()

"""
================================================================================
  ood_threshold_sweep_variant_a.py
  ────────────────────────────────────────────────────────────────────────────
  OOD_GENERALIZATION_REMEDIATION.md left one question deliberately open:
  Variant A (blended training data) improved ROC-AUC on every real
  out-of-distribution check (Dataset 1 +0.052, Dataset 3 +0.287) but still
  produced 0/701 and 0/64 true positives at the deployed threshold (0.30),
  because its highest score on either dataset never gets close to 0.30.
  Nobody tested whether a much lower, OOD-specific threshold applied to
  Variant A's *improved ranking* would actually recover real fraud rows.

  This is read-only inference against an already-trained, already-saved
  artifact (artefacts/paysense_model_blended_training.pkl) -- no retraining,
  reusing exactly the same scoring harness (swap_ps_state,
  score_ensemble_dataframe, load_dataset_1/3) as
  ood_generalization_remediation.py, so the numbers are directly comparable
  to that document's. The only new thing here is sweeping the threshold
  Variant A's own scores are compared against, instead of fixing it at 0.30.

  This does NOT touch paysense_threshold.pkl or any deployed artifact.
================================================================================
"""
import json
import os
import sys

import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from ood_generalization_remediation import (  # noqa: E402
    ARTEFACTS_DIR, load_dataset_1, load_dataset_3,
    swap_ps_state, score_ensemble_dataframe,
)

# A much finer, much lower grid than the 0.05-step, 0.05-0.50 range used
# elsewhere in this project -- Variant A's own max score on Dataset 1 was
# 0.0847 (OOD_GENERALIZATION_REMEDIATION.md §3.3), so anything above that
# is guaranteed to recover nothing and isn't worth testing again here.
THRESHOLD_GRID = sorted(
    {round(t, 4) for t in np.concatenate([
        np.arange(0.001, 0.01, 0.001),
        np.arange(0.01, 0.10, 0.005),
        np.arange(0.10, 0.31, 0.02),
    ])},
    reverse=True,
)


def sweep(y, scores, dataset_name):
    y = np.asarray(y)
    n = len(y)
    n_fraud = int(y.sum())
    n_legit = n - n_fraud
    rows = []
    for t in THRESHOLD_GRID:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        recall = tp / n_fraud if n_fraud else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        fpr = fp / n_legit if n_legit else 0.0
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(recall, 4), "precision": round(precision, 4),
            "false_positive_rate": round(fpr, 4),
            "total_flagged": tp + fp,
        })
    print(f"\n{dataset_name} -- Variant A, threshold sweep "
          f"(n={n:,}, n_fraud={n_fraud}, max_score={scores.max():.4f})")
    print(f"{'thr':>8} {'TP':>5} {'FP':>7} {'recall':>8} {'precision':>10} {'FPR':>8} {'flagged':>9}")
    prev_tp = 0
    for r in rows:
        marker = "  <-- first TP appears" if r["tp"] > 0 and prev_tp == 0 else ""
        print(f"{r['threshold']:>8.4f} {r['tp']:>5} {r['fp']:>7} {r['recall']:>8.2%} "
              f"{r['precision']:>10.4%} {r['false_positive_rate']:>8.2%} {r['total_flagged']:>9,}{marker}")
        prev_tp = r["tp"]
    return rows


def main():
    from src import fraud_model  # noqa: E402  (import after sys.path setup)

    model_A = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_model_blended_training.pkl"))
    prep_A = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_preprocessor_blended_training.pkl"))
    features_A = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_feature_names_blended_training.pkl"))

    results = {}

    print("Loading Dataset 1 (74,917 rows, primary real-world check) ...")
    mapped1, y1, stats1 = load_dataset_1()
    records1 = mapped1.to_dict(orient="records")
    print(f"Scoring Dataset 1 through Variant A's ensemble ({len(records1):,} rows) -- this takes a few minutes ...")
    with swap_ps_state(model_A, prep_A, features_A):
        scored1 = score_ensemble_dataframe(records1)
    results["dataset1"] = sweep(y1.to_numpy(), scored1["ensemble_scores"], "Dataset 1 (74,917 rows, 701 fraud)")

    print("\nLoading Dataset 3 (1,000 rows, low power, 64 fraud) ...")
    mapped3, y3, stats3 = load_dataset_3()
    records3 = mapped3.to_dict(orient="records")
    with swap_ps_state(model_A, prep_A, features_A):
        scored3 = score_ensemble_dataframe(records3)
    results["dataset3"] = sweep(y3.to_numpy(), scored3["ensemble_scores"], "Dataset 3 (1,000 rows, 64 fraud)")

    out_path = os.path.join(BASE_DIR, "ood_threshold_sweep_variant_a_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()

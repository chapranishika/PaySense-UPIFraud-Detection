"""
resweep_threshold_against_ensemble.py moved the deployed threshold from
0.30 to 0.50 (the real ensemble's own optimum, not raw XGBoost's). Every
existing real-dataset ensemble score (GENERALIZATION_CHECK.md's Dataset 1/3,
computed by generalization_check_ensemble.py) was reported at the OLD
threshold (0.30) and needs re-reporting at the new one (0.50).

Re-scores Dataset 1 and Dataset 3 through the real ensemble ONE more time
(reusing the exact honest feature mappings from generalization_check.py,
unchanged) and this time SAVES the raw per-row score arrays to .npy files
so any *future* threshold change doesn't require another ~20-minute
re-scoring pass -- just re-threshold the cached array.
"""
import os
import sys
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from src import fraud_model  # noqa: E402
from generalization_check import load_dataset_1, load_dataset_3  # noqa: E402

SCORES_D1_PATH = os.path.join(BASE_DIR, "ensemble_scores_dataset1_cached.npy")
SCORES_D3_PATH = os.path.join(BASE_DIR, "ensemble_scores_dataset3_cached.npy")


def score_dataset(name, mapped_df, y, cache_path):
    if os.path.exists(cache_path):
        print(f"{name}: loading cached scores from {cache_path}")
        return np.load(cache_path)
    print(f"{name}: scoring {len(mapped_df)} rows through the real ensemble ...")
    records = mapped_df.to_dict(orient="records")
    scores = []
    t0 = time.time()
    for i, rec in enumerate(records):
        scores.append(fraud_model.score(rec).ensemble_score)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(records)} scored ({time.time()-t0:.1f}s elapsed)")
    scores = np.array(scores)
    np.save(cache_path, scores)
    print(f"  scored {len(records)} rows in {time.time()-t0:.1f}s, cached -> {cache_path}")
    return scores


def report(name, y, scores, thresholds=(0.30, 0.50)):
    roc = roc_auc_score(y, scores)
    pr = average_precision_score(y, scores)
    print(f"\n=== {name} ===  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  n={len(y)}  fraud={int(y.sum())}")
    for t in thresholds:
        pred = (scores >= t).astype(int)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        print(f"  tau={t:.2f}: precision={p:.4f} recall={r:.4f} f1={f1:.4f} "
              f"TP={tp} FP={fp} FN={fn} TN={tn}")


def main():
    fraud_model.load_artefacts()

    mapped1, y1, _ = load_dataset_1()
    scores1 = score_dataset("Dataset 1", mapped1, y1, SCORES_D1_PATH)
    report("Dataset 1 (74,917 rows)", y1, scores1)

    mapped3, y3, _ = load_dataset_3()
    scores3 = score_dataset("Dataset 3", mapped3, y3, SCORES_D3_PATH)
    report("Dataset 3 (1,000 rows, low power)", y3, scores3)


if __name__ == "__main__":
    main()

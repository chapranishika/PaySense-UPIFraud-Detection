"""
Companion to rescore_real_datasets_new_threshold.py, for the held-out
synthetic set (seed 918273) used by SYNTHETIC_GROUNDING.md -- its
confusion matrix at the deployed threshold has genuinely non-zero TP
values (unlike Dataset 1/3, where max score < 0.30 < 0.50 means the
confusion matrix is unchanged by the threshold move and needed no re-run).
Caches raw scores for any future threshold change.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from src import fraud_model  # noqa: E402

HELD_OUT_SYNTH_CSV = os.path.join(BASE_DIR, "synthetic_grounded_dataset.csv")
CACHE_PATH = os.path.join(BASE_DIR, "ensemble_scores_synthetic_cached.npy")


def main():
    # Matches generalization_check_synthetic.py exactly: no column dropping --
    # all 40 model features are present, so the full row (including
    # is_fraud/transaction_id, which fraud_model.score() simply ignores via
    # .get()) is passed straight to the real ensemble, the same way a real
    # /predict caller sending the full feature vector would.
    df = pd.read_csv(HELD_OUT_SYNTH_CSV)
    y = df["is_fraud"].astype(int)

    fraud_model.load_artefacts()

    if os.path.exists(CACHE_PATH):
        scores = np.load(CACHE_PATH)
        print(f"Loaded cached scores from {CACHE_PATH}")
    else:
        print(f"Scoring {len(df)} rows through the real ensemble ...")
        records = df.to_dict(orient="records")
        scores = []
        t0 = time.time()
        for i, rec in enumerate(records):
            scores.append(fraud_model.score(rec).ensemble_score)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(records)} scored ({time.time()-t0:.1f}s elapsed)")
        scores = np.array(scores)
        np.save(CACHE_PATH, scores)
        print(f"scored {len(records)} rows in {time.time()-t0:.1f}s, cached -> {CACHE_PATH}")

    roc = roc_auc_score(y, scores)
    pr = average_precision_score(y, scores)
    print(f"\nROC-AUC={roc:.4f} PR-AUC={pr:.4f} n={len(y)} fraud={int(y.sum())}")
    for t in (0.30, 0.50):
        pred = (scores >= t).astype(int)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        print(f"  tau={t:.2f}: precision={p:.4f} recall={r:.4f} f1={f1:.4f} TP={tp} FP={fp} FN={fn} TN={tn}")


if __name__ == "__main__":
    main()

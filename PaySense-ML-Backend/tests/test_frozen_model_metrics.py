"""
================================================================================
  tests/test_frozen_model_metrics.py
  ────────────────────────────────────────────────────────────────────────────
  Guards against exactly the drift found on 2026-08-22: paysense_report.tex,
  README.md, GENERALIZATION_CHECK.md, and SYNTHETIC_GROUNDING.md all cited a
  threshold-sweep table (ROC-AUC 0.8851, 66.14%/52.17% precision/recall @
  t=0.40, 161 fraud rows in the test set) that did not match what the
  currently-frozen artifacts (paysense_model.pkl, paysense_preprocessor.pkl,
  paysense_threshold.pkl) actually produce when scored against
  paysense_master_dataset.csv (ROC-AUC 0.8863, 98.98%/38.34%, 253 fraud rows).
  The stale numbers were internally consistent with each other, which is
  exactly why nobody noticed until someone recomputed precision/recall
  directly from the artifacts instead of trusting a table.

  This test recomputes the same metrics the same way and fails loudly if the
  on-disk artifacts (or the master dataset) ever change without every
  document that cites their metrics being updated to match.
================================================================================
"""

import pathlib

import joblib
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ARTEFACTS_DIR = BASE_DIR / "artefacts"
MASTER_CSV = BASE_DIR / "paysense_master_dataset.csv"

RANDOM_STATE = 42
DROP_COLS = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "data_source", "user_kyc_status", "status", "usr_home_city",
]

# Values currently published in README.md's Key Results table and
# paysense_report.tex's Table~\ref{tab:results} / abstract / conclusion.
# Tight tolerance: these should match what the frozen artifacts produce
# almost exactly, not just "in the right ballpark".
PUBLISHED_ROC_AUC = 0.8863
PUBLISHED_PR_AUC = 0.5339
PUBLISHED_PRECISION_AT_040 = 0.9898
PUBLISHED_RECALL_AT_040 = 0.3834
PUBLISHED_TEST_FRAUD_COUNT = 253


def _skip_if_artefacts_missing():
    required = [
        ARTEFACTS_DIR / "paysense_model.pkl",
        ARTEFACTS_DIR / "paysense_preprocessor.pkl",
        ARTEFACTS_DIR / "paysense_threshold.pkl",
        MASTER_CSV,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(f"Frozen artefacts not present in this environment: {missing}")


@pytest.fixture(scope="module")
def frozen_eval():
    _skip_if_artefacts_missing()

    df = pd.read_csv(MASTER_CSV)
    df = df.drop(columns=DROP_COLS)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)

    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    prep = joblib.load(ARTEFACTS_DIR / "paysense_preprocessor.pkl")
    model = joblib.load(ARTEFACTS_DIR / "paysense_model.pkl")
    threshold = joblib.load(ARTEFACTS_DIR / "paysense_threshold.pkl")

    y_proba = model.predict_proba(prep.transform(X_test_raw))[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "y_test": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def test_frozen_threshold_is_0_40(frozen_eval):
    assert frozen_eval["threshold"] == pytest.approx(0.40)


def test_test_set_fraud_count_matches_published(frozen_eval):
    fraud_count = int(frozen_eval["y_test"].sum())
    assert fraud_count == PUBLISHED_TEST_FRAUD_COUNT, (
        f"paysense_master_dataset.csv's test split now has {fraud_count} fraud "
        f"rows, not the {PUBLISHED_TEST_FRAUD_COUNT} every doc cites — the "
        f"master dataset changed. Every metric derived from it (README.md, "
        f"paysense_report.tex, GENERALIZATION_CHECK.md, SYNTHETIC_GROUNDING.md) "
        f"needs to be recomputed and updated together."
    )


def test_roc_auc_pr_auc_match_published(frozen_eval):
    roc_auc = roc_auc_score(frozen_eval["y_test"], frozen_eval["y_proba"])
    pr_auc = average_precision_score(frozen_eval["y_test"], frozen_eval["y_proba"])
    assert roc_auc == pytest.approx(PUBLISHED_ROC_AUC, abs=0.005), (
        f"Recomputed ROC-AUC ({roc_auc:.4f}) has drifted from the published "
        f"{PUBLISHED_ROC_AUC} — update README.md and paysense_report.tex."
    )
    assert pr_auc == pytest.approx(PUBLISHED_PR_AUC, abs=0.005), (
        f"Recomputed PR-AUC ({pr_auc:.4f}) has drifted from the published "
        f"{PUBLISHED_PR_AUC} — update README.md and paysense_report.tex."
    )


def test_precision_recall_at_frozen_threshold_match_published(frozen_eval):
    precision = precision_score(frozen_eval["y_test"], frozen_eval["y_pred"], zero_division=0)
    recall = recall_score(frozen_eval["y_test"], frozen_eval["y_pred"], zero_division=0)
    assert precision == pytest.approx(PUBLISHED_PRECISION_AT_040, abs=0.01), (
        f"Recomputed precision @ t=0.40 ({precision:.4f}) has drifted from the "
        f"published {PUBLISHED_PRECISION_AT_040} — this is exactly the class of "
        f"drift that made every doc's threshold-sweep table stale on 2026-08-22. "
        f"Recompute and update README.md and paysense_report.tex together."
    )
    assert recall == pytest.approx(PUBLISHED_RECALL_AT_040, abs=0.01), (
        f"Recomputed recall @ t=0.40 ({recall:.4f}) has drifted from the "
        f"published {PUBLISHED_RECALL_AT_040} — update README.md and "
        f"paysense_report.tex together."
    )

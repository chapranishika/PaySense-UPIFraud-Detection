"""
================================================================================
  tests/test_frozen_model_metrics.py
  ────────────────────────────────────────────────────────────────────────────
  Guards against exactly the drift found on 2026-08-22: paysense_report.tex,
  README.md, GENERALIZATION_CHECK.md, and SYNTHETIC_GROUNDING.md all cited a
  threshold-sweep table that did not match what the frozen artifacts actually
  produced. The stale numbers were internally consistent with each other,
  which is exactly why nobody noticed until someone recomputed precision/
  recall directly from the artifacts instead of trusting a table.

  UPDATE (2026-08-23): RECALL_CEILING_REMEDIATION.md diagnosed and tested a
  fix for the model's recall ceiling; the monotone_constraints variant was
  adopted as the new deployed model (paysense_phase3.py retrained with
  monotonic constraints on amount_deviation_score/transaction_velocity/
  failed_attempts_last_24h — see that file's comment for why). Re-running
  Phase 3's own threshold-selection logic against the new model picked a
  DIFFERENT optimal threshold (0.30, not 0.40) — the threshold is a function
  of the model, not a fixed constant, so promoting a new model correctly
  re-derived it rather than carrying the old value forward. The constants
  below were re-pinned to this new frozen model's real, independently
  recomputed numbers.

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
# paysense_report.tex's Table~\ref{tab:results} / abstract / conclusion,
# for the monotonic-constraints model deployed 2026-08-23.
# Tight tolerance: these should match what the frozen artifacts produce
# almost exactly, not just "in the right ballpark".
PUBLISHED_THRESHOLD = 0.30
PUBLISHED_ROC_AUC = 0.8889
PUBLISHED_PR_AUC = 0.5352
PUBLISHED_PRECISION_AT_THRESHOLD = 0.8644
PUBLISHED_RECALL_AT_THRESHOLD = 0.4032
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


def test_frozen_threshold_matches_published(frozen_eval):
    assert frozen_eval["threshold"] == pytest.approx(PUBLISHED_THRESHOLD), (
        f"Frozen threshold is {frozen_eval['threshold']}, not the published "
        f"{PUBLISHED_THRESHOLD} — if the model was retrained, Phase 3's own "
        f"threshold-selection logic must be re-run (it can pick a different "
        f"optimal threshold for a different model, as it did on 2026-08-23), "
        f"and every doc's numbers recomputed and updated together."
    )


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
    assert precision == pytest.approx(PUBLISHED_PRECISION_AT_THRESHOLD, abs=0.01), (
        f"Recomputed precision @ the frozen threshold ({precision:.4f}) has "
        f"drifted from the published {PUBLISHED_PRECISION_AT_THRESHOLD} — this "
        f"is exactly the class of drift that made every doc's threshold-sweep "
        f"table stale on 2026-08-22. Recompute and update README.md and "
        f"paysense_report.tex together."
    )
    assert recall == pytest.approx(PUBLISHED_RECALL_AT_THRESHOLD, abs=0.01), (
        f"Recomputed recall @ the frozen threshold ({recall:.4f}) has drifted "
        f"from the published {PUBLISHED_RECALL_AT_THRESHOLD} — update README.md "
        f"and paysense_report.tex together."
    )

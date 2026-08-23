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

  UPDATE (2026-08-23): the monotone_constraints model was adopted as the
  deployed model (RECALL_CEILING_REMEDIATION.md); Phase 3's threshold
  selection picked 0.30 for it (re-derived, not carried over from 0.40).

  UPDATE (2026-08-24) — a bigger correction than either prior one: this
  file, like every headline metric in README.md/paysense_report.tex until
  now, scored the FROZEN MODEL directly (`model.predict_proba()`) — but
  `/predict` never calls that. It calls `src.fraud_model.score()`, a
  3-scorer ensemble (XGBoost + LightLR + a hand-tuned rules scorer,
  weighted 0.60/0.25/0.15). Scored through the real ensemble, the
  canonical test set behaves substantially differently from the raw-
  XGBoost numbers every document had reported as "the deployed model's
  performance": precision at the (then-)deployed threshold of 0.30 was
  actually 40.81%, not the 86.44% every doc claimed — because the rules
  scorer's always-on additive score was never jointly calibrated against
  that threshold the way XGBoost's own sweep was.

  Fixed properly, not patched: `resweep_threshold_against_ensemble.py`
  re-ran the SAME threshold-selection methodology (business constraint
  Recall>=75%/Precision>=50%, fallback to max-F1) against the REAL
  ensemble's scores instead of raw XGBoost's, swept 0.05-0.95 to confirm
  the optimum wasn't sitting at a range boundary (it isn't — F1 peaks
  exactly at 0.50, dips at 0.55, then plateaus lower from 0.65-0.90 as
  precision saturates at 100%). New deployed threshold: 0.50 (was 0.30).
  This file now scores through the real ensemble too, and every constant
  below is pinned to that — not to raw model.predict_proba() — because
  that is what "the deployed model's metrics" now correctly means.

  This test recomputes the same metrics the same way and fails loudly if
  the on-disk artifacts, the master dataset, or the ensemble's other two
  scorers (LightLR, rules) ever change without every document that cites
  their metrics being updated to match.
================================================================================
"""

import pathlib
import sys

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
sys.path.insert(0, str(BASE_DIR))

RANDOM_STATE = 42
DROP_COLS = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "data_source", "user_kyc_status", "status", "usr_home_city",
]

# Values currently published in README.md's Key Results table and
# paysense_report.tex's Table~\ref{tab:results} / abstract / conclusion,
# for the monotonic-constraints model, scored through the REAL ENSEMBLE
# (not raw XGBoost), deployed threshold 0.50 as of 2026-08-24.
# Tight tolerance: these should match what the frozen artifacts + ensemble
# produce almost exactly, not just "in the right ballpark".
PUBLISHED_THRESHOLD = 0.50
PUBLISHED_ROC_AUC = 0.8969
PUBLISHED_PR_AUC = 0.5498
PUBLISHED_PRECISION_AT_THRESHOLD = 0.9174
PUBLISHED_RECALL_AT_THRESHOLD = 0.3953
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
    from src import fraud_model

    df = pd.read_csv(MASTER_CSV)
    df = df.drop(columns=DROP_COLS)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)

    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    threshold = joblib.load(ARTEFACTS_DIR / "paysense_threshold.pkl")
    fraud_model.load_artefacts()

    records = X_test_raw.to_dict(orient="records")
    y_proba = [fraud_model.score(rec).ensemble_score for rec in records]
    y_pred = [1 if s >= threshold else 0 for s in y_proba]

    return {
        "threshold": threshold,
        "y_test": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def test_frozen_threshold_matches_published(frozen_eval):
    assert frozen_eval["threshold"] == pytest.approx(PUBLISHED_THRESHOLD), (
        f"Frozen threshold is {frozen_eval['threshold']}, not the published "
        f"{PUBLISHED_THRESHOLD} — if the model or ensemble was changed, "
        f"resweep_threshold_against_ensemble.py must be re-run (it can pick "
        f"a different optimal threshold, as it did on 2026-08-24 when the "
        f"sweep target changed from raw XGBoost to the real ensemble), and "
        f"every doc's numbers recomputed and updated together."
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
        f"Recomputed ensemble ROC-AUC ({roc_auc:.4f}) has drifted from the "
        f"published {PUBLISHED_ROC_AUC} — update README.md and "
        f"paysense_report.tex."
    )
    assert pr_auc == pytest.approx(PUBLISHED_PR_AUC, abs=0.005), (
        f"Recomputed ensemble PR-AUC ({pr_auc:.4f}) has drifted from the "
        f"published {PUBLISHED_PR_AUC} — update README.md and "
        f"paysense_report.tex."
    )


def test_precision_recall_at_frozen_threshold_match_published(frozen_eval):
    precision = precision_score(frozen_eval["y_test"], frozen_eval["y_pred"], zero_division=0)
    recall = recall_score(frozen_eval["y_test"], frozen_eval["y_pred"], zero_division=0)
    assert precision == pytest.approx(PUBLISHED_PRECISION_AT_THRESHOLD, abs=0.01), (
        f"Recomputed ensemble precision @ the frozen threshold ({precision:.4f}) "
        f"has drifted from the published {PUBLISHED_PRECISION_AT_THRESHOLD} — "
        f"this is exactly the class of drift that made every doc's "
        f"threshold-sweep table stale before. Recompute and update README.md "
        f"and paysense_report.tex together."
    )
    assert recall == pytest.approx(PUBLISHED_RECALL_AT_THRESHOLD, abs=0.01), (
        f"Recomputed ensemble recall @ the frozen threshold ({recall:.4f}) has "
        f"drifted from the published {PUBLISHED_RECALL_AT_THRESHOLD} — update "
        f"README.md and paysense_report.tex together."
    )


def test_ensemble_differs_materially_from_raw_xgboost_at_threshold(frozen_eval):
    """Regression guard for the 2026-08-24 finding itself: raw XGBoost and
    the real ensemble must NOT be silently treated as interchangeable again.
    If a future change makes them converge, that's fine and this test can
    be relaxed then — but it should be a deliberate observation, not
    something that slips by unnoticed the way the original discrepancy did."""
    import joblib as _joblib
    prep = _joblib.load(ARTEFACTS_DIR / "paysense_preprocessor.pkl")
    model = _joblib.load(ARTEFACTS_DIR / "paysense_model.pkl")

    df = pd.read_csv(MASTER_CSV).drop(columns=DROP_COLS)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    _, X_test_raw, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    raw_proba = model.predict_proba(prep.transform(X_test_raw))[:, 1]
    raw_pred = (raw_proba >= frozen_eval["threshold"]).astype(int)
    raw_precision = precision_score(frozen_eval["y_test"], raw_pred, zero_division=0)

    ensemble_precision = precision_score(
        frozen_eval["y_test"], frozen_eval["y_pred"], zero_division=0
    )
    assert abs(raw_precision - ensemble_precision) > 0.05, (
        f"Raw XGBoost precision ({raw_precision:.4f}) and real ensemble "
        f"precision ({ensemble_precision:.4f}) at the deployed threshold "
        f"are now nearly identical — if this is expected (e.g. rules/LightLR "
        f"were changed to matter less), update this test's rationale rather "
        f"than deleting it silently."
    )

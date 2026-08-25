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


# ── Business-constraint verification, added 2026-08-26 audit ────────────────
# paysense_phase3.py and resweep_threshold_against_ensemble.py both target a
# documented business constraint: Recall >= 75% AND Precision >= 50%. Neither
# script's own sweep output was ever turned into a test asserting whether the
# DEPLOYED threshold actually satisfies it -- an easy thing to silently drift
# out of sync with reality the same way the raw-vs-ensemble metrics did.
#
# Full sweep (ensemble_threshold_resweep_results.json, 0.05-0.95 step 0.05)
# confirms zero thresholds meet both constraints simultaneously. The closest
# candidate, t=0.15, hits recall=75.9% but precision=15.99% -- 84% of alerts
# at that threshold would be false alarms. This is a genuine constraint
# infeasibility on this test set, not a threshold-selection bug: no single
# threshold trades recall for precision favorably enough to clear both bars
# at once, which is exactly why both selection scripts fall back to
# unconditional max-F1 (landing on 0.50) rather than ever finding a
# constraint-satisfying row.
#
# This test pins the CURRENT, VERIFIED, KNOWN state: the deployed threshold
# does not meet the originally-documented recall floor. It exists so that if
# the model is ever retrained well enough to change this, that's a genuine,
# visible event requiring a documentation update (revising the "Recall >=75%"
# claim to match reality, or reporting that the constraint is newly
# achievable) -- not something that silently drifts one way or the other
# unnoticed the way the raw-vs-ensemble metrics did for weeks.
RECALL_CONSTRAINT_MIN = 0.75
PRECISION_CONSTRAINT_MIN = 0.50


def test_deployed_threshold_does_not_meet_documented_recall_constraint(frozen_eval):
    precision = precision_score(frozen_eval["y_test"], frozen_eval["y_pred"], zero_division=0)
    recall = recall_score(frozen_eval["y_test"], frozen_eval["y_pred"], zero_division=0)

    assert recall < RECALL_CONSTRAINT_MIN, (
        f"Recall at the deployed threshold is now {recall:.4f}, which MEETS "
        f"the documented Recall>={RECALL_CONSTRAINT_MIN:.0%} business "
        f"constraint that this test previously verified was NOT met. If this "
        f"is a real improvement (retrained model, more/better features), "
        f"this is good news -- update README.md/PROJECT.md to remove the "
        f"'documented requirement not currently satisfied' finding and note "
        f"the fix. Do not just relax this assertion without checking why it "
        f"changed."
    )
    assert precision == pytest.approx(PUBLISHED_PRECISION_AT_THRESHOLD, abs=0.01), (
        f"Precision at the deployed threshold ({precision:.4f}) drifted from "
        f"the published {PUBLISHED_PRECISION_AT_THRESHOLD} -- recompute the "
        f"full trade-off table (see resweep_threshold_against_ensemble.py) "
        f"before updating any documented business-constraint claim."
    )


def test_no_swept_threshold_meets_both_business_constraints(frozen_eval):
    """Reproduces the full 0.05-0.95 sweep against the frozen artefacts and
    confirms the documented gap directly, rather than trusting the static
    ensemble_threshold_resweep_results.json file to still reflect reality."""
    import numpy as np

    y_test = frozen_eval["y_test"]
    y_proba = np.array(frozen_eval["y_proba"])
    sweep = np.round(np.arange(0.05, 0.96, 0.05), 2)

    meets_both = []
    for t in sweep:
        pred = (y_proba >= float(t)).astype(int)
        p = precision_score(y_test, pred, zero_division=0)
        r = recall_score(y_test, pred, zero_division=0)
        if r >= RECALL_CONSTRAINT_MIN and p >= PRECISION_CONSTRAINT_MIN:
            meets_both.append((float(t), p, r))

    assert meets_both == [], (
        f"Found {len(meets_both)} threshold(s) that now satisfy BOTH "
        f"Recall>={RECALL_CONSTRAINT_MIN:.0%} AND "
        f"Precision>={PRECISION_CONSTRAINT_MIN:.0%}: {meets_both}. This "
        f"contradicts the 2026-08-24 sweep finding that no threshold does. "
        f"If the model genuinely improved, re-run "
        f"resweep_threshold_against_ensemble.py and deploy the new "
        f"constraint-satisfying threshold -- don't leave the model sitting "
        f"on the max-F1 fallback if a real constraint-satisfying option now "
        f"exists."
    )


# ── Source-stratified metrics -- the biggest finding of the 2026-08-26 audit ─
# EDA_FEATURE_ENGINEERING.md documented that the 10K-row "supplement" source
# (schema-bridged from an external synthetic_fraud_dataset.csv) carries a
# near-tautological relationship between new_device_flag/ip_location_mismatch
# and is_fraud (that external dataset's own label-generation formula, not a
# bug introduced by this project). What had not been checked before this
# audit: since the train/test split is a plain stratified random split over
# the FULL blended dataset (data_source plays no role in the split), the test
# set inherits the same ~35% supplement contamination -- and the model's
# performance on that slice versus the organic "anchor" slice is wildly
# different:
#
#   Full blended test set (what every doc has ever reported):
#     ROC-AUC 0.8969   PR-AUC 0.5498   recall@0.50 = 39.53% (TP=100/253)
#   Anchor-only (organic, real-style data, n=3913, 157 fraud):
#     ROC-AUC 0.7465   PR-AUC 0.1138   recall@0.50 =  2.55% (TP=4/157)
#   Supplement-only (tautological-label data, n=2087, 96 fraud):
#     ROC-AUC 1.0000   PR-AUC 1.0000   recall@0.50 = 100.00% (TP=96/96)
#
# The model catches essentially none of the organic fraud in this test set
# (4 of 157) -- its reported 39.53% blended recall is almost entirely the
# supplement subset's trivially-learnable synthetic shortcut, not evidence
# of learned, transferable fraud-detection skill. This does not mean the
# model is worthless (0.7465 ROC-AUC / 2.8x-baseline PR-AUC on organic data
# alone is still real, positive signal) -- but every "13x baseline" /
# "0.90 ROC-AUC" headline claim in this project's docs is computed on the
# contaminated blended set and materially overstates real-world performance.
#
# This test exists to make sure that gap stays visible and quantified rather
# than being silently smoothed over by a future retrain that "fixes" the
# blended number without anyone checking whether organic-only performance
# actually improved.
def test_organic_subset_performance_is_much_weaker_than_blended_headline(frozen_eval):
    import numpy as np
    from src import fraud_model as _fm  # noqa: F401  (ensures artefacts loaded)

    df = pd.read_csv(MASTER_CSV).drop(columns=[c for c in DROP_COLS if c != "data_source"])
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    sources = X_test["data_source"].values
    anchor_mask = sources == "anchor"

    y_test_arr = y_test.values
    y_proba_arr = np.array(frozen_eval["y_proba"])

    assert len(y_proba_arr) == len(anchor_mask), (
        "frozen_eval's scored rows and this test's re-derived source labels "
        "are out of sync -- the split parameters must match exactly "
        "(same DROP_COLS minus data_source, same random_state/test_size)."
    )

    anchor_roc_auc = roc_auc_score(y_test_arr[anchor_mask], y_proba_arr[anchor_mask])
    anchor_pr_auc = average_precision_score(y_test_arr[anchor_mask], y_proba_arr[anchor_mask])

    blended_roc_auc = roc_auc_score(y_test_arr, y_proba_arr)
    blended_pr_auc = average_precision_score(y_test_arr, y_proba_arr)

    assert anchor_roc_auc < blended_roc_auc - 0.10, (
        f"Anchor-only ROC-AUC ({anchor_roc_auc:.4f}) is no longer "
        f"substantially below the blended ROC-AUC ({blended_roc_auc:.4f}). "
        f"If the model genuinely improved on organic data, that's real "
        f"progress worth documenting explicitly in WALKTHROUGH.md/"
        f"PROJECT.md -- but verify it's a real improvement (e.g. via "
        f"feature engineering or more organic training data) and not a "
        f"measurement artifact before updating the headline claims."
    )
    assert anchor_pr_auc < blended_pr_auc - 0.20, (
        f"Anchor-only PR-AUC ({anchor_pr_auc:.4f}) is no longer "
        f"substantially below the blended PR-AUC ({blended_pr_auc:.4f}). "
        f"Same caveat as above -- verify before updating any documented "
        f"'Nx baseline' claim, since that claim is computed on the blended "
        f"set and would need to be recomputed on organic-only data to be "
        f"an honest representation of real-world generalization."
    )

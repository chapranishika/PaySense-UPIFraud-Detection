"""
================================================================================
  tests/test_eda_feature_engineering.py
  ────────────────────────────────────────────────────────────────────────────
  EDA_FEATURE_ENGINEERING.md found that a third of paysense_master_dataset.csv
  (the "supplement" partition) has new_device_flag/ip_location_mismatch
  PERFECTLY separating is_fraud (a near-tautological, non-organic artifact
  inherited from Financial Fraud Dataset/synthetic_fraud_dataset.csv, not
  something this project's own pipeline introduced) and engineered two new
  features to test whether that finding is fixable:

    - stealth_fraud_score: a GATED interaction term, (1-new_device_flag) *
      (1-ip_location_mismatch) * mean(min-max normalized behavioral
      features) -- nonzero only when both hard flags read clean, targeting
      the population RECALL_CEILING_REMEDIATION.md already showed is where
      invisible fraud concentrates.
    - hour_sin / hour_cos: cyclical encoding of hour_of_day, motivated by a
      KS-test-confirmed distributional mismatch between training data's
      hour_of_day and real Dataset 1's.

  This test file guards two things, in the same spirit as
  tests/test_recall_ceiling_remediation.py and
  tests/test_ood_generalization_remediation.py:

    1. compute_engineered_features()'s core behavior -- NaN propagation when
       an underlying input is genuinely absent (so an external dataset that
       never supplied new_device_flag/ip_location_mismatch/behavioral
       features gets an honest NaN, not a fabricated 0), and correct
       cyclical range for hour_sin/hour_cos.
    2. If the saved candidate artifact (paysense_model_feature_engineered.pkl)
       is present, it still loads and produces valid, non-constant
       probabilities on the canonical held-out test set with the engineered
       columns added -- a smoke test, not a full metric re-pin (those live
       in EDA_FEATURE_ENGINEERING.md and are expensive to recompute).

  NO RETRAINING happens in this test file. All artifacts are loaded
  read-only.
================================================================================
"""

import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

ARTEFACTS_DIR = BASE_DIR / "artefacts"
MASTER_CSV = BASE_DIR / "paysense_master_dataset.csv"
RANDOM_STATE = 42
DROP_COLS = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "data_source", "user_kyc_status", "status", "usr_home_city",
]

MODEL_PATH = ARTEFACTS_DIR / "paysense_model_feature_engineered.pkl"
PREP_PATH = ARTEFACTS_DIR / "paysense_preprocessor_feature_engineered.pkl"
FEATURES_PATH = ARTEFACTS_DIR / "paysense_feature_names_feature_engineered.pkl"


def _get_compute_fn():
    from eda_feature_engineering import compute_engineered_features
    return compute_engineered_features


DUMMY_TRAIN_STATS = {
    "amount_deviation_score": {"min": 0.0, "max": 10.0},
    "transaction_velocity": {"min": 0.0, "max": 4.0},
    "failed_attempts_last_24h": {"min": 0.0, "max": 5.0},
}


def test_stealth_fraud_score_is_nan_when_hard_flags_absent():
    """Dataset 1 / Dataset 3 (GENERALIZATION_CHECK.md) never supply
    new_device_flag/ip_location_mismatch/behavioral features -- the derived
    feature must honestly propagate NaN (to be median-imputed like every
    other missing feature), never silently default to 0."""
    compute_engineered_features = _get_compute_fn()
    df = pd.DataFrame({"amount": [10.0, 20.0], "hour_of_day": [5, 13]})
    out = compute_engineered_features(df, DUMMY_TRAIN_STATS)
    assert out["stealth_fraud_score"].isna().all()


def test_stealth_fraud_score_is_gated_by_both_hard_flags():
    """The whole point of this feature vs. RECALL_CEILING_REMEDIATION.md's
    flat composite_feature is that it is ZERO whenever either hard flag is
    1, and only reflects the behavioral mean when BOTH flags are clean."""
    compute_engineered_features = _get_compute_fn()
    df = pd.DataFrame({
        "new_device_flag": [0, 1, 0, 1],
        "ip_location_mismatch": [0, 0, 1, 1],
        "amount_deviation_score": [5.0, 5.0, 5.0, 5.0],
        "transaction_velocity": [2.0, 2.0, 2.0, 2.0],
        "failed_attempts_last_24h": [2.5, 2.5, 2.5, 2.5],
    })
    out = compute_engineered_features(df, DUMMY_TRAIN_STATS)
    # Row 0: both flags clean -> nonzero behavioral-mean contribution.
    assert out["stealth_fraud_score"].iloc[0] > 0
    # Rows 1-3: at least one flag set -> gated to exactly 0.
    assert out["stealth_fraud_score"].iloc[1] == 0
    assert out["stealth_fraud_score"].iloc[2] == 0
    assert out["stealth_fraud_score"].iloc[3] == 0


def test_hour_sin_cos_cyclical_range_and_wraparound():
    """hour_sin/hour_cos must stay in [-1, 1] and correctly treat hour 23
    and hour 0 as adjacent (small Euclidean distance in sin/cos space),
    unlike raw integer hour_of_day where they are maximally far apart."""
    compute_engineered_features = _get_compute_fn()
    df = pd.DataFrame({"hour_of_day": [0, 6, 12, 18, 23]})
    out = compute_engineered_features(df, DUMMY_TRAIN_STATS)
    assert out["hour_sin"].between(-1.0, 1.0).all()
    assert out["hour_cos"].between(-1.0, 1.0).all()

    hour0 = out.iloc[0][["hour_sin", "hour_cos"]].to_numpy(dtype=float)
    hour23 = out.iloc[4][["hour_sin", "hour_cos"]].to_numpy(dtype=float)
    hour12 = out.iloc[2][["hour_sin", "hour_cos"]].to_numpy(dtype=float)
    dist_0_23 = np.linalg.norm(hour0 - hour23)
    dist_0_12 = np.linalg.norm(hour0 - hour12)
    assert dist_0_23 < dist_0_12, (
        "hour=0 and hour=23 should be close in cyclical (sin,cos) space, "
        "closer than hour=0 and hour=12 (opposite side of the clock)."
    )


def test_hour_sin_cos_absent_when_hour_of_day_absent():
    """Dataset 3 (GENERALIZATION_CHECK.md) never maps hour_of_day -- the
    derived cyclical features must honestly propagate NaN there too."""
    compute_engineered_features = _get_compute_fn()
    df = pd.DataFrame({"amount": [10.0, 20.0], "usr_account_age_days": [100.0, 200.0]})
    out = compute_engineered_features(df, DUMMY_TRAIN_STATS)
    assert out["hour_sin"].isna().all()
    assert out["hour_cos"].isna().all()


@pytest.mark.skipif(
    not (MODEL_PATH.exists() and PREP_PATH.exists() and FEATURES_PATH.exists() and MASTER_CSV.exists()),
    reason="Feature-engineered candidate artifacts not present in this environment.",
)
def test_feature_engineered_artifact_loads_and_scores_canonical_test_set():
    """Smoke test: if the candidate artifact from this experiment is
    present, it must load and produce valid, non-constant, better-than-
    chance probabilities on the canonical held-out test set once the two
    engineered features are added -- catches a corrupted/incompatible
    artifact or a schema mismatch, without re-pinning the exact ROC-AUC
    (that lives in EDA_FEATURE_ENGINEERING.md and is expensive to
    recompute -- a full retrain)."""
    compute_engineered_features = _get_compute_fn()

    df = pd.read_csv(MASTER_CSV).drop(columns=DROP_COLS)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    train_stats = {
        c: {"min": float(X_train_raw[c].min()), "max": float(X_train_raw[c].max())}
        for c in ["amount_deviation_score", "transaction_velocity", "failed_attempts_last_24h"]
    }
    X_test_fe = compute_engineered_features(X_test_raw, train_stats)

    model = joblib.load(MODEL_PATH)
    prep = joblib.load(PREP_PATH)
    features = joblib.load(FEATURES_PATH)
    assert "stealth_fraud_score" in features
    assert "hour_sin" in features and "hour_cos" in features

    X_test_proc = prep.transform(X_test_fe)
    proba = model.predict_proba(X_test_proc)[:, 1]

    assert proba.shape[0] == len(y_test)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert not np.any(np.isnan(proba))
    assert proba.std() > 0, "Feature-engineered artifact produces a constant score -- likely broken."

    roc_auc = roc_auc_score(y_test, proba)
    assert 0.5 < roc_auc <= 1.0, (
        f"Feature-engineered artifact ROC-AUC {roc_auc:.4f} on canonical test set is not "
        f"better than chance -- likely a broken artifact or preprocessor mismatch."
    )

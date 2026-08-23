"""
================================================================================
  tests/test_research_grounded_synthetic_dataset.py
  ────────────────────────────────────────────────────────────────────────────
  Covers generate_research_grounded_synthetic_dataset.py (Track B of
  REAL_DATA_AND_RESEARCH_GROUNDING.md) -- the 3-typology mixture-model
  synthetic dataset, structurally different from
  generate_grounded_synthetic_dataset.py's single calibrated-logistic label.

  Mirrors tests/test_synthetic_dataset_generation.py's conventions: all tests
  call generate() directly with a small row count, never reading the real
  15,000-row research_grounded_synthetic_dataset.csv from disk, so the suite
  stays fast and independent of whether that file has been generated yet.
================================================================================
"""

import pathlib

import numpy as np
import pytest

import generate_research_grounded_synthetic_dataset as gen

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ML_PIPELINE_PY = BASE_DIR / "paysense_ml_pipeline.py"
ARTEFACTS_DIR = BASE_DIR / "artefacts"

MASTER_SCHEMA_COLUMNS = {
    "transaction_id", "user_id", "receiver_id", "receiver_type", "amount",
    "timestamp", "date", "hour_of_day", "day_of_week", "is_weekend",
    "is_night_transaction", "time_since_last_txn_min", "transaction_type",
    "payment_app", "device_type", "status", "user_city_tier",
    "user_kyc_status", "user_avg_monthly_txn", "user_avg_txn_value",
    "user_loyalty_score", "new_device_flag", "ip_location_mismatch",
    "failed_attempts_last_24h", "transaction_velocity",
    "amount_deviation_score", "is_fraud", "recurring_payment_flag",
    "balance_after_transaction", "transaction_frequency_score",
    "txn_success_flag", "kyc_verified_flag", "data_source",
    "usr_age_group", "usr_home_city", "usr_home_city_tier",
    "usr_account_age_days", "usr_linked_bank_count",
    "usr_avg_monthly_txn_profile", "usr_avg_txn_value_profile",
    "usr_preferred_app", "usr_preferred_device", "usr_is_high_risk",
    "mrc_category", "mrc_size", "mrc_avg_daily_txn", "mrc_is_registered",
    "mrc_rating", "device_risk_score", "ip_risk_score",
}


@pytest.fixture(scope="module")
def small_df():
    return gen.generate(n_txn=6000, n_users=1200, n_merchants=150, seed=42, target_fraud_rate=0.04)


# ── Structural difference from the seed-918273/445566 generator ───────────
def test_seed_disjoint_from_all_prior_seeds():
    assert gen.SEED not in (42, 918273, 445566)


def test_typology_weights_sum_to_one_and_are_positive():
    assert abs(sum(gen.TYPOLOGY_WEIGHTS) - 1.0) < 1e-9
    assert all(w > 0 for w in gen.TYPOLOGY_WEIGHTS)
    assert len(gen.TYPOLOGY_WEIGHTS) == len(gen.TYPOLOGY_NAMES) == 3


def test_schema_matches_master_dataset_columns(small_df):
    assert set(small_df.columns) == MASTER_SCHEMA_COLUMNS
    assert len(small_df.columns) == 50


def test_drop_cols_then_target_yields_40_model_features(small_df):
    import ast

    tree = ast.parse(ML_PIPELINE_PY.read_text(encoding="utf-8"))
    drop_cols = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DROP_COLS" for t in node.targets
        ):
            drop_cols = ast.literal_eval(node.value)
    assert drop_cols is not None, "DROP_COLS not found in paysense_ml_pipeline.py"
    remaining = small_df.drop(columns=drop_cols + ["is_fraud"])
    assert remaining.shape[1] == 40


# ── The core structural claim: typology-conditioned generation actually
#    produces DIFFERENT feature signatures per typology, not one uniform
#    fraud population. ───────────────────────────────────────────────────
def test_three_typologies_are_all_realised(small_df):
    df = gen.generate(n_txn=20000, n_users=2500, n_merchants=300, seed=11, target_fraud_rate=0.04)
    fraud = df[df["is_fraud"] == 1]
    # Recompute typology assignment is not stored on the returned df (only in
    # df.attrs during generate(), which this call doesn't capture) -- instead
    # verify the STRUCTURAL CONSEQUENCE: fraud rows split into a
    # hard-flags-dirty cluster and a hard-flags-clean cluster, both non-empty
    # and each a substantial share (not one dominating at >95%, which would
    # mean the mixture collapsed to one component in practice).
    both_dirty = (fraud["new_device_flag"] == 1) & (fraud["ip_location_mismatch"] == 1)
    both_clean = (fraud["new_device_flag"] == 0) & (fraud["ip_location_mismatch"] == 0)
    assert both_dirty.mean() > 0.05, "account-takeover-like fraud cluster too small or missing"
    assert both_clean.mean() > 0.30, "social-engineering-like (flags-clean) fraud cluster too small or missing"


def test_account_takeover_signature_new_device_and_ip_mismatch_elevated(small_df):
    """T1 rows are constructed to force new_device_flag/ip_location_mismatch
    high; this must show up as a real, non-trivial elevation in fraud rate
    for both flags (the mixture's dirty-flags component actually exists),
    without being disqualifyingly deterministic (GENERALIZATION_CHECK.md
    §2.2's red flag)."""
    for col in ("new_device_flag", "ip_location_mismatch"):
        rate_1 = small_df.loc[small_df[col] == 1, "is_fraud"].mean()
        rate_0 = small_df.loc[small_df[col] == 0, "is_fraud"].mean()
        assert rate_1 > rate_0, f"{col}=1 should raise fraud rate"
        assert rate_1 < 0.95, f"{col} separates the label too cleanly (near-deterministic)"


def test_mule_network_signature_velocity_elevated(small_df):
    """T3 rows are constructed with elevated transaction_velocity; the
    correlation must be real (positive, non-trivial) but not deterministic."""
    corr = small_df["transaction_velocity"].corr(small_df["is_fraud"].astype(float))
    assert corr > 0.02, f"transaction_velocity correlation with is_fraud too weak: {corr:.4f}"
    assert corr < 0.6, f"transaction_velocity correlation suspiciously high (near-deterministic): {corr:.4f}"


def test_social_engineering_signature_amount_deviation_elevated_with_clean_flags(small_df):
    """The defining structural claim of the social_engineering typology:
    fraud rows with BOTH hard flags clean must still show elevated
    amount_deviation_score relative to legitimate rows -- i.e. there is a
    real, usable signal for this population, even though the two 'hard'
    flags this project's own EDA found dominant are silent for it."""
    clean_fraud = small_df[
        (small_df["is_fraud"] == 1)
        & (small_df["new_device_flag"] == 0)
        & (small_df["ip_location_mismatch"] == 0)
    ]
    legit = small_df[small_df["is_fraud"] == 0]
    if len(clean_fraud) < 5:
        pytest.skip("Too few flags-clean fraud rows in this small draw to compare means meaningfully.")
    assert clean_fraud["amount_deviation_score"].mean() > legit["amount_deviation_score"].mean()


def test_fraud_signal_is_correlated_but_not_deterministic_overall(small_df):
    """Same red-flag guard as test_synthetic_dataset_generation.py's
    equivalent test, applied to this generator's own label."""
    for col in ("new_device_flag", "ip_location_mismatch"):
        rate_1 = small_df.loc[small_df[col] == 1, "is_fraud"].mean()
        rate_0 = small_df.loc[small_df[col] == 0, "is_fraud"].mean()
        assert rate_1 > rate_0
        assert rate_1 < 0.95


def test_fraud_rate_near_target():
    df = gen.generate(n_txn=20000, n_users=2000, n_merchants=300, seed=7, target_fraud_rate=0.04)
    rate = df["is_fraud"].mean()
    assert 0.025 < rate < 0.06, f"fraud rate {rate:.4f} not near target 0.04"


def test_amount_capped_at_100000_and_positive(small_df):
    assert small_df["amount"].max() <= 100000
    assert small_df["amount"].min() > 0


def test_day_of_week_is_numeric_not_string(small_df):
    assert small_df["day_of_week"].dtype.kind in "iu"
    assert small_df["day_of_week"].between(0, 6).all()


def test_city_tier_columns_are_numeric_1_2_3(small_df):
    for col in ("user_city_tier", "usr_home_city_tier"):
        assert small_df[col].dtype.kind in "iu", f"{col} must be numeric"
        assert set(small_df[col].unique()) <= {1, 2, 3}


def test_mrc_fields_null_for_p2p_populated_for_merchant(small_df):
    p2p = small_df["receiver_type"] == "User"
    merchant = small_df["receiver_type"] == "Merchant"
    assert small_df.loc[p2p, "mrc_category"].isna().all()
    assert small_df.loc[merchant, "mrc_category"].notna().all()


@pytest.mark.skipif(
    not (ARTEFACTS_DIR / "paysense_feature_names.pkl").exists(),
    reason="frozen artefacts not present in this environment",
)
def test_drop_cols_result_matches_frozen_feature_names(small_df):
    import ast
    import joblib

    tree = ast.parse(ML_PIPELINE_PY.read_text(encoding="utf-8"))
    drop_cols = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DROP_COLS" for t in node.targets
        ):
            drop_cols = ast.literal_eval(node.value)
    feature_names = joblib.load(ARTEFACTS_DIR / "paysense_feature_names.pkl")
    remaining_cols = set(small_df.drop(columns=drop_cols + ["is_fraud"]).columns)
    assert remaining_cols == set(feature_names)

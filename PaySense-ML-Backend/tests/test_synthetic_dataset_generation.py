"""
================================================================================
  tests/test_synthetic_dataset_generation.py
  ────────────────────────────────────────────────────────────────────────────
  Covers generate_grounded_synthetic_dataset.py — the independently-drawn,
  full-40-feature synthetic dataset built to test the frozen model without
  the "85% of features missing" confound documented in
  GENERALIZATION_CHECK.md. Mirrors this project's existing convention
  (test_pipeline_logic.py) of checking that generated/claimed distributions
  actually hold, not just that the script runs.

  All tests here call generate() directly with a SMALL row count — they do
  not read the real 25,000-row synthetic_grounded_dataset.csv from disk, so
  the suite stays fast regardless of whether that file has been generated
  yet in the environment running pytest.
================================================================================
"""

import ast
import pathlib

import numpy as np
import pytest

import generate_grounded_synthetic_dataset as gen

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


def _extract_drop_cols(pyfile: pathlib.Path) -> list:
    tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DROP_COLS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"No DROP_COLS assignment found in {pyfile}")


@pytest.fixture(scope="module")
def small_df():
    """A small, fast draw — same generator, same logic, modest N (still
    < 1s to generate). n_users=1000 keeps sampling noise on ~87%/~5%-style
    rates within the tolerances used below without needing the full 3,000
    -user production scale."""
    return gen.generate(n_txn=4000, n_users=1000, n_merchants=120, seed=42, target_fraud_rate=0.04)


# ── Schema ───────────────────────────────────────────────────────────────
def test_schema_matches_master_dataset_columns(small_df):
    assert set(small_df.columns) == MASTER_SCHEMA_COLUMNS
    assert len(small_df.columns) == 50


def test_drop_cols_then_target_yields_40_model_features(small_df):
    """After applying the SAME DROP_COLS the real ml pipeline uses, and
    removing the target, exactly 40 columns remain — matching
    artefacts/paysense_feature_names.pkl's length."""
    drop_cols = _extract_drop_cols(ML_PIPELINE_PY)
    remaining = small_df.drop(columns=drop_cols + ["is_fraud"])
    assert remaining.shape[1] == 40


@pytest.mark.skipif(
    not (ARTEFACTS_DIR / "paysense_feature_names.pkl").exists(),
    reason="frozen artefacts not present in this environment",
)
def test_drop_cols_result_matches_frozen_feature_names(small_df):
    import joblib

    feature_names = joblib.load(ARTEFACTS_DIR / "paysense_feature_names.pkl")
    drop_cols = _extract_drop_cols(ML_PIPELINE_PY)
    remaining_cols = set(small_df.drop(columns=drop_cols + ["is_fraud"]).columns)
    assert remaining_cols == set(feature_names)


# ── Exact documented rules (data_dictionary.csv gives these verbatim) ─────
def test_transaction_frequency_score_exact_formula(small_df):
    expected = np.clip(small_df["user_avg_monthly_txn"] / 50.0, 0, 1)
    np.testing.assert_allclose(small_df["transaction_frequency_score"], expected)


def test_is_night_transaction_exact_rule(small_df):
    expected = ((small_df["hour_of_day"] < 6) | (small_df["hour_of_day"] >= 22)).astype(int)
    assert (small_df["is_night_transaction"] == expected).all()


def test_recurring_payment_flag_exact_rule(small_df):
    should_be_1 = small_df["transaction_type"].isin(["Bill Payment", "Subscription", "EMI"])
    assert (small_df.loc[should_be_1, "recurring_payment_flag"] == 1).all()
    assert (small_df.loc[~should_be_1, "recurring_payment_flag"] == 0).all()


def test_txn_success_flag_matches_status(small_df):
    expected = (small_df["status"] == "Success").astype(int)
    assert (small_df["txn_success_flag"] == expected).all()


def test_kyc_verified_flag_matches_user_kyc_status(small_df):
    expected = (small_df["user_kyc_status"] == "Verified").astype(int)
    assert (small_df["kyc_verified_flag"] == expected).all()


def test_amount_capped_at_100000_and_positive(small_df):
    assert small_df["amount"].max() <= 100000
    assert small_df["amount"].min() > 0


# ── Regression tests for the dictionary/frozen-artefact dtype mismatches
#    found while wiring this dataset through fraud_model.score() (see
#    generate_grounded_synthetic_dataset.py's day_of_week / _tier_to_int
#    comments and SYNTHETIC_GROUNDING.md) ───────────────────────────────────
def test_day_of_week_is_numeric_not_string(small_df):
    """data_dictionary.csv documents day_of_week as a string ('Monday',
    'Saturday'), but artefacts/paysense_preprocessor.pkl's fitted
    ColumnTransformer routes it through the NUMERIC pipeline (verified by
    inspecting transformers_ and paysense_master_dataset.csv's int64
    dtype). Must stay numeric or every row silently fails PaySense scoring."""
    assert small_df["day_of_week"].dtype.kind in "iu"
    assert small_df["day_of_week"].between(0, 6).all()


def test_day_of_week_matches_is_weekend(small_df):
    # pandas .dt.dayofweek convention: Monday=0 ... Sunday=6, so 5/6 = weekend
    is_weekend_from_dow = small_df["day_of_week"].isin([5, 6]).astype(int)
    assert (small_df["is_weekend"] == is_weekend_from_dow).all()


def test_city_tier_columns_are_numeric_1_2_3(small_df):
    """Same class of mismatch as day_of_week: data_dictionary.csv documents
    'Tier 1'/'Tier 2'/'Tier 3' strings, but the frozen preprocessor's fitted
    ColumnTransformer treats user_city_tier/usr_home_city_tier as numeric."""
    for col in ("user_city_tier", "usr_home_city_tier"):
        assert small_df[col].dtype.kind in "iu", f"{col} must be numeric"
        assert set(small_df[col].unique()) <= {1, 2, 3}


# ── Distribution checks (statistical, generous tolerance — matches the
#    spirit of the ~N% figures data_dictionary.csv documents) ─────────────
def test_fraud_rate_near_target():
    df = gen.generate(n_txn=20000, n_users=2000, n_merchants=300, seed=7, target_fraud_rate=0.04)
    rate = df["is_fraud"].mean()
    assert 0.025 < rate < 0.06, f"fraud rate {rate:.4f} not near target 0.04"


def test_status_distribution_near_documented_rates(small_df):
    vc = small_df["status"].value_counts(normalize=True)
    assert abs(vc.get("Success", 0) - 0.88) < 0.05
    assert abs(vc.get("Failed", 0) - 0.09) < 0.05
    assert abs(vc.get("Pending", 0) - 0.03) < 0.03


def test_failed_attempts_zero_rate_near_documented(small_df):
    rate = (small_df["failed_attempts_last_24h"] == 0).mean()
    assert abs(rate - 0.72) < 0.06


def test_kyc_verified_rate_near_documented(small_df):
    rate = (small_df["user_kyc_status"] == "Verified").mean()
    assert abs(rate - 0.87) < 0.05


def test_intentional_missingness_near_2pct(small_df):
    for col in ("time_since_last_txn_min", "amount_deviation_score", "transaction_velocity"):
        rate = small_df[col].isna().mean()
        assert 0.0 < rate < 0.06, f"{col} missing rate {rate:.4f} not near documented ~2%"


def test_mrc_fields_null_for_p2p_populated_for_merchant(small_df):
    p2p = small_df["receiver_type"] == "User"
    merchant = small_df["receiver_type"] == "Merchant"
    assert small_df.loc[p2p, "mrc_category"].isna().all()
    assert small_df.loc[merchant, "mrc_category"].notna().all()


def test_fraud_signal_is_correlated_but_not_deterministic(small_df):
    """Regression guard against the exact red flag GENERALIZATION_CHECK.md
    used to reject its Dataset 2 (§2.2): a feature that drives the label to
    a clean 0% or 100%. new_device_flag/ip_location_mismatch must raise the
    fraud rate meaningfully, but never separate the classes perfectly."""
    for col in ("new_device_flag", "ip_location_mismatch"):
        rate_1 = small_df.loc[small_df[col] == 1, "is_fraud"].mean()
        rate_0 = small_df.loc[small_df[col] == 0, "is_fraud"].mean()
        assert rate_1 > rate_0, f"{col}=1 should raise fraud rate"
        assert rate_1 < 0.95, f"{col} separates the label too cleanly (near-deterministic)"

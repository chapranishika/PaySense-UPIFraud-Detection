"""
================================================================================
  tests/test_real_data_and_research_grounding.py
  ────────────────────────────────────────────────────────────────────────────
  Covers real_data_and_research_grounding.py -- Track A (Dataset 5:
  kaggle_vbinh002_fraud_ecommerce/Fraud_Data.csv) and Track B (Variant C,
  trained on the research-grounded synthetic blend) of
  REAL_DATA_AND_RESEARCH_GROUNDING.md.

  Mirrors tests/test_ood_generalization_remediation.py's conventions:
  cheap structural/leakage checks always run; anything that needs the real
  external CSV or a trained candidate artifact is skipped gracefully if not
  present in this environment, rather than failing the whole suite.
================================================================================
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ARTEFACTS_DIR = BASE_DIR / "artefacts"
MASTER_CSV = BASE_DIR / "paysense_master_dataset.csv"
HELD_OUT_SYNTH_CSV = BASE_DIR / "synthetic_grounded_dataset.csv"  # seed 918273
RG_SYNTH_CSV = BASE_DIR / "research_grounded_synthetic_dataset.csv"  # seed 771029
DATASET5_CSV = BASE_DIR / "external_data" / "kaggle_vbinh002_fraud_ecommerce" / "Fraud_Data.csv"

RANDOM_STATE = 42
DROP_COLS = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "data_source", "user_kyc_status", "status", "usr_home_city",
]


def test_research_grounded_seed_disjoint_from_held_out_and_blend_seeds():
    import generate_research_grounded_synthetic_dataset as gen_rg

    assert gen_rg.SEED not in (42, 918273, 445566)


@pytest.mark.skipif(
    not (HELD_OUT_SYNTH_CSV.exists() and RG_SYNTH_CSV.exists()),
    reason="Both synthetic CSVs must be on disk to check for row-level leakage.",
)
def test_no_full_row_duplication_between_research_grounded_and_held_out_synthetic_sets():
    df_rg = pd.read_csv(RG_SYNTH_CSV)
    df_held_out = pd.read_csv(HELD_OUT_SYNTH_CSV)
    compare_cols = [c for c in df_rg.columns if c != "transaction_id"]
    assert compare_cols == [c for c in df_held_out.columns if c != "transaction_id"], (
        "Research-grounded and held-out synthetic sets have different schemas."
    )
    rg_rows = set(map(tuple, df_rg[compare_cols].astype(str).values.tolist()))
    held_out_rows = set(map(tuple, df_held_out[compare_cols].astype(str).values.tolist()))
    overlap = rg_rows & held_out_rows
    assert not overlap, (
        f"{len(overlap)} fully-identical rows appear in BOTH the research-grounded "
        f"training set and the held-out evaluation set -- the seeds are not "
        f"producing independent draws."
    )


@pytest.mark.skipif(
    not RG_SYNTH_CSV.exists(),
    reason="Research-grounded synthetic dataset not present in this environment.",
)
def test_research_grounded_dataset_schema_matches_master_dataset():
    df_rg = pd.read_csv(RG_SYNTH_CSV, nrows=5)
    master_cols = pd.read_csv(MASTER_CSV, nrows=5).columns
    assert set(df_rg.columns) == set(master_cols)


@pytest.mark.parametrize(
    "filename,preprocessor_filename",
    [
        ("paysense_model_research_grounded.pkl", "paysense_preprocessor_research_grounded.pkl"),
    ],
)
def test_candidate_artifact_loads_and_scores_canonical_test_set(filename, preprocessor_filename):
    """Light-touch smoke test mirroring test_ood_generalization_remediation.py's
    equivalent: if Variant C's artifact is present, it must load and produce
    valid, non-constant, better-than-chance probabilities on the canonical
    held-out test set. Does not re-pin exact ROC-AUC (that lives in
    REAL_DATA_AND_RESEARCH_GROUNDING.md and requires a full retrain to
    reproduce) -- this only catches a corrupted/incompatible artifact."""
    model_path = ARTEFACTS_DIR / filename
    prep_path = ARTEFACTS_DIR / preprocessor_filename
    if not (model_path.exists() and prep_path.exists() and MASTER_CSV.exists()):
        pytest.skip(f"{filename} / {preprocessor_filename} not present in this environment.")

    df = pd.read_csv(MASTER_CSV).drop(columns=DROP_COLS)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = joblib.load(model_path)
    prep = joblib.load(prep_path)
    X_test_proc = prep.transform(X_test_raw)
    proba = model.predict_proba(X_test_proc)[:, 1]

    assert proba.shape[0] == len(y_test)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert not np.any(np.isnan(proba))
    assert proba.std() > 0, f"{filename} produces a constant score -- likely a broken artifact."

    roc_auc = roc_auc_score(y_test, proba)
    assert 0.5 < roc_auc <= 1.0, (
        f"{filename}: ROC-AUC {roc_auc:.4f} on the canonical test set is not "
        f"better than chance -- likely a broken artifact or preprocessor mismatch."
    )


# ── Dataset 5 (Fraud_Data.csv) vetting + honest-mapping regression guards ──
pytestmark_dataset5 = pytest.mark.skipif(
    not DATASET5_CSV.exists(), reason="Dataset 5 CSV not present in this environment."
)


@pytest.fixture(scope="module")
def df5():
    if not DATASET5_CSV.exists():
        pytest.skip("Dataset 5 CSV not present in this environment.")
    return pd.read_csv(DATASET5_CSV)


@pytestmark_dataset5
def test_no_duplicate_user_ids_or_full_rows(df5):
    assert df5["user_id"].duplicated().sum() == 0
    assert df5.duplicated().sum() == 0


@pytestmark_dataset5
def test_no_nulls(df5):
    assert df5.isna().sum().sum() == 0


@pytestmark_dataset5
def test_fraud_rate_is_not_a_round_number(df5):
    """Regression guard against the GENERALIZATION_CHECK.md §2.2 red
    flag: a suspiciously round fraud rate (e.g. 20.00%, 50.01%) is a
    signature of pre-balanced/templated data."""
    rate = df5["class"].mean()
    assert 0.01 < rate < 0.30, f"fraud rate {rate:.4f} outside a plausible real range"
    # Checked against the precise value vetted in
    # REAL_DATA_AND_RESEARCH_GROUNDING.md rather than a generic
    # round-number heuristic (which could false-positive on legitimate
    # data that happens to land near a round figure).
    assert abs(rate - 0.09364577267192546) < 1e-6, (
        "Dataset 5's fraud rate drifted from the value vetted in "
        "REAL_DATA_AND_RESEARCH_GROUNDING.md -- re-vet before reusing "
        "this number in the write-up."
    )


@pytestmark_dataset5
def test_categorical_fields_not_near_deterministic(df5):
    """None of source/browser/sex may drive the fraud rate to (near) 0%
    or 100% -- the same check that disqualified GENERALIZATION_CHECK.md's
    Dataset 2."""
    for col in ("source", "browser", "sex"):
        rates = df5.groupby(col)["class"].mean()
        assert rates.min() > 0.03, f"{col} has a near-zero-fraud category: {rates.to_dict()}"
        assert rates.max() < 0.20, f"{col} has a near-deterministic-fraud category: {rates.to_dict()}"


@pytestmark_dataset5
def test_account_age_signal_elevated_but_not_deterministic(df5):
    """The dataset's standout real signal: fraud purchases happen almost
    immediately after signup. Must be a real, large elevation (this is
    expected to be strong) but the two classes' distributions must still
    overlap -- not a clean separation."""
    signup = pd.to_datetime(df5["signup_time"])
    purchase = pd.to_datetime(df5["purchase_time"])
    acct_age_days = (purchase - signup).dt.total_seconds() / 86400.0
    fraud_age = acct_age_days[df5["class"] == 1]
    legit_age = acct_age_days[df5["class"] == 0]
    assert fraud_age.mean() < legit_age.mean(), "fraud rows should skew toward newer accounts"
    assert fraud_age.max() > 5, "fraud age distribution should still have real spread/overlap"


@pytest.mark.skipif(not DATASET5_CSV.exists(), reason="Dataset 5 CSV not present in this environment.")
def test_load_dataset_5_honest_mapping_has_no_leakage_columns():
    """Regression guard: the honest mapping must never include device_id,
    browser, source, sex, or ip_address -- these were deliberately NOT
    mapped (see real_data_and_research_grounding.py's load_dataset_5
    docstring) because they either collide in name with a PaySense field
    while meaning something different (browser vs. device_type), or have no
    honest PaySense counterpart at all."""
    import sys

    sys.path.insert(0, str(BASE_DIR))
    import real_data_and_research_grounding as m

    mapped, y, stats = m.load_dataset_5(sample_n=200, sample_seed=1)
    forbidden = {"device_id", "browser", "source", "sex", "ip_address", "device_type",
                 "new_device_flag", "ip_location_mismatch"}
    assert not (set(mapped.columns) & forbidden), (
        f"load_dataset_5 mapped a column it explicitly documents as unmappable: "
        f"{set(mapped.columns) & forbidden}"
    )
    assert set(mapped.columns) == {
        "amount", "usr_account_age_days", "hour_of_day", "is_night_transaction",
        "day_of_week", "is_weekend", "usr_age_group",
    }
    assert stats["features_mapped"] == 7
    assert len(y) == 200

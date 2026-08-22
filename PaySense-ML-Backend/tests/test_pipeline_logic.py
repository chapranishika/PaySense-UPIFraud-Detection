"""
================================================================================
  tests/test_pipeline_logic.py — PaySense ML Pipeline Unit Test Suite
  ────────────────────────────────────────────────────────────────────────────
  Covers claims made in ../PaySense-Report/paysense_report.tex and in the
  in-repo pipeline documentation that were previously only asserted in prose:

    1. SMOTE discipline    — applied only post-split, only to the training
                              partition; X_test is never touched or resized.
    2. Alert-level bands   — [0,0.20)->none [0.20,0.40)->low
                              [0.40,0.70)->medium [0.70,1]->high, exact
                              boundary values.
    3. Stddev clamps       — max(sigma, 1.0) / max(sigma, 0.5). NOT PRESENT
                              in this Python backend (verified absent from
                              src/fraud_model.py, main.py, paysense_pipeline.py,
                              paysense_ml_pipeline.py, paysense_phase3.py, and
                              static/app.js). The only implementation found in
                              the whole PaySense_Final_Submission tree is
                              Kotlin, client-side:
                              PaySense-Android-Client-New/app/src/main/kotlin/
                              com/paysense/app/layer3/FraudApiService.kt lines
                              204 & 223 (`sqrt(varAmount).coerceAtLeast(1.0)`,
                              `sqrt(vh).coerceAtLeast(0.5)`). No test added
                              here for that reason — there is nothing to
                              import and exercise on the Python side.
    4. DROP_COLS / schema  — the 9 non-predictive columns dropped in
                              paysense_ml_pipeline.py / paysense_phase3.py are
                              actually absent from the model-ready feature
                              matrix, and the resulting feature count matches
                              the frozen artefact.

  WHY THESE TESTS DON'T IMPORT paysense_pipeline.py / paysense_ml_pipeline.py
  / paysense_phase3.py DIRECTLY:
  All three are top-level scripts, not libraries of functions — importing
  any of them executes the entire pipeline immediately (reads a 9.5MB /
  1.3GB CSV from disk, fits a 400-tree XGBoost model, runs BorderlineSMOTE,
  computes SHAP values, writes PNGs...). That is not something a unit test
  should trigger on every run. Instead:
    - DROP_COLS is extracted via `ast` static analysis of the real source
      files (no execution) so the test fails if the *actual* source list
      ever changes, without needing to run the script.
    - The split -> SMOTE discipline is exercised with a small synthetic
      DataFrame using the exact library calls the scripts use
      (sklearn.train_test_split(..., stratify=y) then
      imblearn SMOTE.fit_resample on the train partition only), which is
      sufficient to prove the *pattern* is leak-free; it does not require
      touching the real 30k-row dataset or training a real model.
    - The frozen artefacts under artefacts/*.pkl (already produced by a
      prior real run of the pipeline) are loaded read-only to check the
      real, shipped feature schema and threshold — these tests are skipped
      if the artefacts are not present in the environment running the suite.
================================================================================
"""

import ast
import os
import pathlib

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

BASE_DIR      = pathlib.Path(__file__).resolve().parent.parent
ARTEFACTS_DIR = BASE_DIR / "artefacts"

ML_PIPELINE_PY  = BASE_DIR / "paysense_ml_pipeline.py"
PHASE3_PY       = BASE_DIR / "paysense_phase3.py"

# The 9 columns the report / pipeline docstrings claim are dropped as
# non-predictive (identifiers, temporal leakage vectors, metadata,
# redundant string duplicates of existing binary flags, and a
# high-cardinality geographic string).
DOCUMENTED_DROP_COLS = [
    "transaction_id", "user_id", "receiver_id",
    "timestamp", "date",
    "data_source",
    "user_kyc_status", "status",
    "usr_home_city",
]

# Real paysense_master_dataset.csv header (50 columns), used only to build a
# small *synthetic* schema-shaped DataFrame — the multi-MB CSV itself is
# never read by this test module.
MASTER_SCHEMA_COLUMNS = [
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
]
TARGET_COL = "is_fraud"

assert len(MASTER_SCHEMA_COLUMNS) == 50, "Fixture schema drifted from real CSV header"


def _extract_drop_cols(pyfile: pathlib.Path) -> list:
    """Statically extract the `DROP_COLS = [...]` list literal from a
    pipeline script via AST parsing, without executing the module (see
    module docstring for why these scripts cannot be safely imported)."""
    tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DROP_COLS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"No DROP_COLS assignment found in {pyfile}")


def _synthetic_master_dataframe(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """A tiny DataFrame shaped exactly like paysense_master_dataset.csv
    (same 50 column names/order) filled with cheap synthetic values, so
    schema-shape logic can be tested without touching the real 9.5MB CSV."""
    rng = np.random.RandomState(seed)
    data = {}
    for col in MASTER_SCHEMA_COLUMNS:
        if col in ("timestamp", "date"):
            data[col] = pd.date_range("2024-01-01", periods=n, freq="h")
        elif col in ("is_fraud", "is_weekend", "is_night_transaction",
                     "new_device_flag", "ip_location_mismatch",
                     "recurring_payment_flag", "txn_success_flag",
                     "kyc_verified_flag", "usr_is_high_risk",
                     "mrc_is_registered"):
            data[col] = rng.randint(0, 2, n)
        elif col in ("transaction_id", "user_id", "receiver_id",
                     "status", "user_kyc_status", "data_source",
                     "usr_home_city", "usr_age_group", "usr_preferred_app",
                     "usr_preferred_device", "receiver_type",
                     "transaction_type", "payment_app", "device_type",
                     "mrc_category", "mrc_size"):
            data[col] = [f"{col}_{i}" for i in range(n)]
        else:
            data[col] = rng.uniform(0, 100, n)
    return pd.DataFrame(data)


# ════════════════════════════════════════════════════════════════════════════
#  CLAIM 4 — DROP_COLS / feature schema
# ════════════════════════════════════════════════════════════════════════════
class TestDropColsSchema:

    def test_ml_pipeline_drop_cols_has_nine_entries(self):
        drop_cols = _extract_drop_cols(ML_PIPELINE_PY)
        assert len(drop_cols) == 9

    def test_ml_pipeline_drop_cols_matches_documented_list(self):
        drop_cols = _extract_drop_cols(ML_PIPELINE_PY)
        assert set(drop_cols) == set(DOCUMENTED_DROP_COLS)

    def test_phase3_drop_cols_matches_ml_pipeline(self):
        """Phase 3 re-declares DROP_COLS inline to replay Phase 2's
        preprocessing exactly. If someone edits one list and forgets the
        other, the frozen model and the training script silently diverge."""
        assert _extract_drop_cols(PHASE3_PY) == _extract_drop_cols(ML_PIPELINE_PY)

    def test_synthetic_schema_after_drop_and_target_removal_is_40_features(self):
        df = _synthetic_master_dataframe()
        assert df.shape[1] == 50

        drop_cols = _extract_drop_cols(ML_PIPELINE_PY)
        df = df.drop(columns=drop_cols)
        assert df.shape[1] == 50 - 9 == 41, (
            "This 41-column count (after dropping the 9 non-predictive "
            "columns but BEFORE removing the target) is what the report's "
            "'41 model-ready features' claim actually measures."
        )

        X = df.drop(columns=[TARGET_COL])
        assert X.shape[1] == 40, (
            "The real model-ready feature matrix (X, target excluded) has "
            "40 columns, not 41 as stated in paysense_report.tex — the "
            "report's count includes the is_fraud target column. This is "
            "confirmed independently by the frozen "
            "artefacts/paysense_feature_names.pkl (see "
            "TestFrozenArtefacts below)."
        )

    def test_dropped_columns_absent_from_final_feature_matrix(self):
        df = _synthetic_master_dataframe()
        drop_cols = _extract_drop_cols(ML_PIPELINE_PY)
        X = df.drop(columns=drop_cols).drop(columns=[TARGET_COL])
        for col in drop_cols:
            assert col not in X.columns
        assert TARGET_COL not in X.columns


ARTEFACTS_PRESENT = (ARTEFACTS_DIR / "paysense_feature_names.pkl").exists()


@pytest.mark.skipif(not ARTEFACTS_PRESENT, reason="artefacts/*.pkl not present in this environment")
class TestFrozenArtefacts:
    """Loads the real, already-trained artefacts read-only. These were
    produced by a prior run of paysense_phase3.py and are what main.py
    actually loads at startup (src/fraud_model.py::load_artefacts) — so
    this is the ground truth for what schema is shipped to production."""

    @staticmethod
    def _feature_names():
        return joblib.load(ARTEFACTS_DIR / "paysense_feature_names.pkl")

    def test_feature_count_is_40_not_41(self):
        feats = self._feature_names()
        assert len(feats) == 40

    def test_drop_cols_absent_from_shipped_feature_names(self):
        feats = set(self._feature_names())
        drop_cols = set(_extract_drop_cols(ML_PIPELINE_PY))
        assert feats.isdisjoint(drop_cols)

    def test_target_absent_from_shipped_feature_names(self):
        assert TARGET_COL not in self._feature_names()

    def test_threshold_is_valid_probability(self):
        threshold = joblib.load(ARTEFACTS_DIR / "paysense_threshold.pkl")
        assert 0.0 <= float(threshold) <= 1.0

    def test_preprocessor_transforms_full_schema_row_to_40_columns(self):
        """The frozen ColumnTransformer must accept a row built from the
        shipped feature names and emit exactly one column per feature."""
        preprocessor = joblib.load(ARTEFACTS_DIR / "paysense_preprocessor.pkl")
        feats = self._feature_names()

        row = {}
        for f in feats:
            if f in ("receiver_type", "transaction_type", "payment_app",
                      "device_type", "usr_age_group", "usr_preferred_app",
                      "usr_preferred_device", "mrc_category", "mrc_size"):
                row[f] = "SomeCategory"
            else:
                row[f] = 1.0
        df = pd.DataFrame([row]).reindex(columns=feats)

        transformed = preprocessor.transform(df)
        assert transformed.shape == (1, len(feats))


# ════════════════════════════════════════════════════════════════════════════
#  CLAIM 1 — SMOTE applied only post-split, only to the training partition
# ════════════════════════════════════════════════════════════════════════════
class TestSmoteSplitDiscipline:
    """paysense_ml_pipeline.py / paysense_phase3.py both: (a) stratified
    train_test_split BEFORE any fitting, (b) fit the preprocessor on train
    only, (c) call SMOTE/BorderlineSMOTE.fit_resample(X_train, y_train) —
    X_test/y_test are never arguments to the resampler. We reproduce that
    exact call sequence on a small synthetic imbalanced dataset (rather than
    importing the real scripts — see module docstring) and assert the
    invariant the report claims: the test partition's size and class
    balance are provably unaffected by SMOTE, because it is never given to
    it.
    """

    @staticmethod
    def _make_imbalanced_frame(n=600, fraud_rate=0.05, seed=42):
        rng = np.random.RandomState(seed)
        n_fraud = int(n * fraud_rate)
        n_legit = n - n_fraud
        y = np.array([0] * n_legit + [1] * n_fraud)
        X = pd.DataFrame({
            "amount":                  rng.exponential(500, n),
            "amount_deviation_score":  rng.normal(0, 1, n) + y * 3,
            "transaction_velocity":    rng.uniform(0, 1, n) + y * 0.3,
            "new_device_flag":         rng.binomial(1, np.clip(0.1 + y * 0.4, 0, 1)),
        })
        # shuffle so class order isn't trivially separable by row position
        idx = rng.permutation(n)
        return X.iloc[idx].reset_index(drop=True), pd.Series(y[idx])

    def test_split_is_stratified(self):
        X, y = self._make_imbalanced_frame()
        _, _, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        full_rate = y.mean()
        assert abs(y_test.mean() - full_rate) < 0.03
        assert abs(y_train.mean() - full_rate) < 0.03

    def test_smote_never_touches_or_resizes_test_partition(self):
        X, y = self._make_imbalanced_frame()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        X_test_snapshot = X_test.copy(deep=True)
        y_test_snapshot = y_test.copy(deep=True)
        test_size_before = len(X_test)
        test_fraud_count_before = int(y_test.sum())

        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)  # X_test never passed in

        # X_test / y_test objects themselves must be byte-for-byte identical
        pd.testing.assert_frame_equal(X_test, X_test_snapshot)
        pd.testing.assert_series_equal(y_test, y_test_snapshot)
        assert len(X_test) == test_size_before
        assert int(y_test.sum()) == test_fraud_count_before

    def test_smote_balances_train_partition_only(self):
        X, y = self._make_imbalanced_frame()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        train_fraud_rate_before = y_train.mean()

        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        assert len(X_train_bal) > len(X_train), "SMOTE should add synthetic rows to train"
        assert y_train_bal.mean() > train_fraud_rate_before, (
            "Train partition should be more balanced after SMOTE"
        )
        # sanity: test partition size is untouched by how much train grew
        assert len(X_test) == len(y_test) == int(round(len(X) * 0.20))


# ════════════════════════════════════════════════════════════════════════════
#  CLAIM 2 — alert-level threshold bands (exact boundaries)
# ════════════════════════════════════════════════════════════════════════════
from src.fraud_model import EnsembleResult    # noqa: E402  (after sys.path setup in conftest)


def _ensemble_result(score: float) -> EnsembleResult:
    return EnsembleResult(
        ensemble_score = score,
        paysense_score = score,
        light_lr_score = None,
        rules_score    = score,
        active_scorers = ["rules"],
        weights_used   = {"rules": 1.0},
        threshold      = 0.4,
    )


BOUNDARY_CASES = [
    (0.0,      "none"),
    (0.05,     "none"),
    (0.19,     "none"),
    (0.1999,   "none"),
    (0.19999,  "none"),
    (0.20,     "low"),      # lower bound of "low" is inclusive
    (0.200001, "low"),
    (0.35,     "low"),
    (0.3999,   "low"),
    (0.39999,  "low"),
    (0.40,     "medium"),   # lower bound of "medium" is inclusive
    (0.400001, "medium"),
    (0.55,     "medium"),
    (0.6999,   "medium"),
    (0.69999,  "medium"),
    (0.70,     "high"),     # lower bound of "high" is inclusive
    (0.700001, "high"),
    (0.85,     "high"),
    (1.0,      "high"),
]


class TestAlertLevelThresholds:
    """src/fraud_model.py::EnsembleResult.alert_level is the implementation
    actually used by /predict (main.py's `result = ensemble_score(txn_dict)`
    then `alert_level = result.alert_level`).

    main.py previously had its own duplicate `compute_alert_level()` with the
    same boundary logic, never called anywhere (/predict exclusively used
    `result.alert_level`). It was dead code with a live twin that could have
    silently drifted apart, so it was deleted rather than kept "just in case"
    — this class now only tests the one implementation that's actually live."""

    @pytest.mark.parametrize("score,expected", BOUNDARY_CASES)
    def test_ensemble_result_alert_level_boundary(self, score, expected):
        assert _ensemble_result(score).alert_level == expected

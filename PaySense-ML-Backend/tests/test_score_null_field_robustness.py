"""
================================================================================
  tests/test_score_null_field_robustness.py
  ────────────────────────────────────────────────────────────────────────────
  Closes a real gap: three separate bugs tonight (LightLR's null handling,
  the rules scorer's null handling, and by extension the general pattern)
  were all found by *running scripts against real datasets with intentional
  nulls*, never by an automated test — because every existing test that
  exercises `/predict` goes through Pydantic, and every field the ensemble
  scorers read (`failed_attempts_last_24h`, `amount_deviation_score`,
  `new_device_flag`, `ip_location_mismatch`, `kyc_verified_flag`,
  `usr_is_high_risk`, `usr_account_age_days`, ...) is declared
  `Field(..., ...)` in main.py's TransactionInput -- required, non-nullable.
  Pydantic rejects a null for any of them with a 422 before the request
  ever reaches `fraud_model.score()`. So an HTTP-level test sending nulls
  only proves Pydantic validation works (see TestPublicApiRejectsNulls
  below, which is real and worth having) -- it does NOT exercise the same
  code path the three real bugs lived in.

  The actual vulnerable surface is any INTERNAL caller of
  `fraud_model.score()` that builds its own dict and skips Pydantic --
  exactly what every generalization-check / EDA / OOD-remediation script
  in this project does, and exactly how all three bugs were found. This
  file tests that surface directly, once, for every field the ensemble
  scorers touch -- so this class of bug can't hide behind "the API rejects
  it anyway" reasoning again, since not every caller is the API.
================================================================================
"""
import pathlib
import sys

import pytest

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

ARTEFACTS_DIR = BASE_DIR / "artefacts"

# Every field _score_rules() or _score_light_lr() reads via txn_dict.get(...).
NULLABLE_FIELDS = [
    "failed_attempts_last_24h",
    "amount_deviation_score",
    "new_device_flag",
    "ip_location_mismatch",
    "kyc_verified_flag",
    "usr_is_high_risk",
    "usr_account_age_days",
    "transaction_velocity",
    "is_night_transaction",
]


def _base_record() -> dict:
    """A complete, valid feature dict -- every field a real /predict caller
    would send, used as the baseline before nulling one field at a time."""
    return {
        "receiver_type": "Merchant", "transaction_type": "P2M", "payment_app": "GPay",
        "device_type": "Android", "usr_age_group": "25-34", "usr_preferred_app": "GPay",
        "usr_preferred_device": "Android", "mrc_category": "Grocery", "mrc_size": "Medium",
        "amount": 500.0, "hour_of_day": 14, "day_of_week": 2, "is_weekend": 0,
        "is_night_transaction": 0, "time_since_last_txn_min": 120.0,
        "transaction_velocity": 1.0, "amount_deviation_score": 0.5,
        "failed_attempts_last_24h": 0.0, "recurring_payment_flag": 0,
        "transaction_frequency_score": 0.3, "new_device_flag": 0,
        "ip_location_mismatch": 0, "user_city_tier": 1, "user_avg_monthly_txn": 20.0,
        "user_avg_txn_value": 500.0, "user_loyalty_score": 0.5,
        "balance_after_transaction": 10000.0, "txn_success_flag": 1,
        "kyc_verified_flag": 1, "usr_home_city_tier": 1, "usr_account_age_days": 400.0,
        "usr_linked_bank_count": 1.0, "usr_avg_monthly_txn_profile": 20.0,
        "usr_avg_txn_value_profile": 500.0, "usr_is_high_risk": 0,
        "mrc_avg_daily_txn": 50.0, "mrc_is_registered": 1, "mrc_rating": 4.0,
        "device_risk_score": 0.1, "ip_risk_score": 0.1,
    }


def _fraud_model_or_skip():
    if not (ARTEFACTS_DIR / "paysense_model.pkl").exists():
        pytest.skip("Frozen model artefacts not present in this environment.")
    from src import fraud_model
    fraud_model.load_artefacts()
    return fraud_model


class TestScoreHandlesExplicitNullDirectly:
    """The actual regression guard: fraud_model.score() must never raise
    on an explicit None for any field the ensemble scorers read, since
    internal callers (generalization checks, EDA scripts, future ones)
    build dicts by hand and are not protected by Pydantic."""

    @pytest.mark.parametrize("field", NULLABLE_FIELDS)
    def test_single_field_null_does_not_raise(self, field):
        fraud_model = _fraud_model_or_skip()
        record = _base_record()
        record[field] = None
        try:
            result = fraud_model.score(record)
        except Exception as e:
            pytest.fail(
                f"fraud_model.score() raised {type(e).__name__} when '{field}' "
                f"was explicitly None -- this is the exact bug class already "
                f"found twice tonight (LightLR, rules scorer). Error: {e}"
            )
        assert 0.0 <= result.ensemble_score <= 1.0
        assert result.rules_score is not None  # rules scorer always runs

    def test_all_fields_null_simultaneously_does_not_raise(self):
        """The worst case: every optional-ish field null at once, e.g. a
        caller passing a mostly-empty dict."""
        fraud_model = _fraud_model_or_skip()
        record = _base_record()
        for field in NULLABLE_FIELDS:
            record[field] = None
        result = fraud_model.score(record)
        assert 0.0 <= result.ensemble_score <= 1.0

    def test_all_fields_absent_entirely_does_not_raise(self):
        """The other edge: keys missing entirely (not just null) -- the
        original, always-correctly-handled case, kept as a control so a
        future change can't silently break this while 'fixing' the null case."""
        fraud_model = _fraud_model_or_skip()
        record = {k: v for k, v in _base_record().items() if k not in NULLABLE_FIELDS}
        result = fraud_model.score(record)
        assert 0.0 <= result.ensemble_score <= 1.0


class TestPublicApiRejectsNulls:
    """The complementary check: confirm Pydantic really is the safety net
    it's assumed to be for the public endpoint, rather than assuming it.
    A null for any of these fields must be rejected with 422, not crash
    with a 500 and not silently succeed with a wrong score."""

    client = TestClient(app, raise_server_exceptions=True)

    @classmethod
    @pytest.fixture(scope="class")
    def auth_headers(cls):
        resp = cls.client.post("/auth/token", json={"username": "paysense", "password": "guardian2025"})
        if resp.status_code != 200:
            pytest.skip("Cannot obtain auth token in this environment.")
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    @pytest.mark.parametrize("field", [
        "failed_attempts_last_24h", "amount_deviation_score", "new_device_flag",
        "ip_location_mismatch", "kyc_verified_flag", "usr_is_high_risk",
        "usr_account_age_days",
    ])
    def test_null_field_rejected_with_422_not_500(self, auth_headers, field):
        payload = _base_record()
        payload[field] = None
        resp = self.client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 422, (
            f"Expected /predict to reject a null '{field}' with 422 "
            f"(Pydantic validation), got {resp.status_code}. If this field "
            f"was deliberately made Optional, the internal null-handling "
            f"tests above become the ONLY protection against this bug class "
            f"reaching production -- make sure they still pass."
        )

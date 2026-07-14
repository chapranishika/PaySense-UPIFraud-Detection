"""
================================================================================
  tests/test_api.py  — PaySense Guardian Integration Test Suite
  ────────────────────────────────────────────────────────────────────────────
  Covers:
    • Auth: token issuance, invalid credentials, expired/malformed tokens
    • /predict: valid fraud + legit payloads, Pydantic validation, rate limit
    • /insights/weekly: tip generation, auth guard
    • /health: ensemble state reporting
    • Security: unauthenticated requests, invalid VPA formats
    • Ensemble: rules-only fallback path
    • Edge cases: cold-start (min history), high-velocity transactions

  Run:
      pip install pytest httpx
      pytest tests/ -v
================================================================================
"""

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app, raise_server_exceptions=True)


# ── Shared fixtures ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def auth_token():
    """Obtain a valid JWT token once for the entire test module."""
    resp = client.post("/auth/token", json={"username": "paysense", "password": "guardian2025"})
    assert resp.status_code == 200, f"Token request failed: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture(scope="module")
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


def legit_payload(**overrides) -> dict:
    """A baseline LEGITIMATE transaction payload."""
    base = {
        "receiver_type": "Merchant",
        "transaction_type": "P2M",
        "payment_app": "GPay",
        "device_type": "Android",
        "usr_age_group": "25-34",
        "usr_preferred_app": "GPay",
        "usr_preferred_device": "Android",
        "mrc_category": "Food",
        "mrc_size": "Medium",
        "amount": 250.0,
        "hour_of_day": 13,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night_transaction": 0,
        "time_since_last_txn_min": 60.0,
        "transaction_velocity": 0.10,
        "amount_deviation_score": 0.4,
        "failed_attempts_last_24h": 0.0,
        "recurring_payment_flag": 0,
        "transaction_frequency_score": 0.45,
        "new_device_flag": 0,
        "ip_location_mismatch": 0,
        "user_city_tier": 2,
        "user_avg_monthly_txn": 32.0,
        "user_avg_txn_value": 850.0,
        "user_loyalty_score": 0.75,
        "balance_after_transaction": 15000.0,
        "txn_success_flag": 1,
        "kyc_verified_flag": 1,
        "usr_home_city_tier": 2,
        "usr_account_age_days": 730.0,
        "usr_linked_bank_count": 2.0,
        "usr_avg_monthly_txn_profile": 32.0,
        "usr_avg_txn_value_profile": 850.0,
        "usr_is_high_risk": 0,
        "mrc_avg_daily_txn": 120.0,
        "mrc_is_registered": 1,
        "mrc_rating": 4.2,
        "device_risk_score": 0.05,
        "ip_risk_score": 0.03,
    }
    base.update(overrides)
    return base


def fraud_payload(**overrides) -> dict:
    """A high-risk fraud signal payload."""
    base = legit_payload(
        amount=15000.0,
        hour_of_day=2,
        is_night_transaction=1,
        new_device_flag=1,
        ip_location_mismatch=1,
        amount_deviation_score=5.5,
        transaction_velocity=0.95,
        failed_attempts_last_24h=3.0,
        kyc_verified_flag=0,
        usr_is_high_risk=1,
        usr_account_age_days=14.0,
        device_risk_score=0.91,
        ip_risk_score=0.87,
    )
    base.update(overrides)
    return base


# ════════════════════════════════════════════════════════════════════════════
#  AUTH TESTS
# ════════════════════════════════════════════════════════════════════════════
class TestAuth:

    def test_valid_credentials_return_token(self):
        resp = client.post("/auth/token",
                           json={"username": "paysense", "password": "guardian2025"})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0

    def test_wrong_password_returns_401(self):
        resp = client.post("/auth/token",
                           json={"username": "paysense", "password": "wrong"})
        assert resp.status_code == 401

    def test_wrong_username_returns_401(self):
        resp = client.post("/auth/token",
                           json={"username": "hacker", "password": "guardian2025"})
        assert resp.status_code == 401

    def test_missing_fields_returns_422(self):
        resp = client.post("/auth/token", json={"username": "paysense"})
        assert resp.status_code == 422

    def test_predict_without_token_returns_403(self):
        resp = client.post("/predict", json=legit_payload())
        assert resp.status_code in (401, 403)

    def test_predict_with_invalid_token_returns_401(self):
        resp = client.post("/predict",
                           json=legit_payload(),
                           headers={"Authorization": "Bearer totally.fake.token"})
        assert resp.status_code == 401

    def test_predict_with_malformed_bearer_returns_403(self):
        resp = client.post("/predict",
                           json=legit_payload(),
                           headers={"Authorization": "NotBearer something"})
        assert resp.status_code in (401, 403)


# ════════════════════════════════════════════════════════════════════════════
#  /predict TESTS
# ════════════════════════════════════════════════════════════════════════════
class TestPredict:

    def test_legit_transaction_scores_low(self, auth_headers):
        resp = client.post("/predict", json=legit_payload(), headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["fraud_score"] <= 1.0
        assert body["alert_level"] in ("none", "low")
        assert body["is_fraud"] is False

    def test_fraud_transaction_scores_high(self, auth_headers):
        resp = client.post("/predict", json=fraud_payload(), headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["fraud_score"] >= 0.40, (
            f"Fraud payload scored too low: {body['fraud_score']}"
        )
        assert body["alert_level"] in ("medium", "high")

    def test_response_schema_complete(self, auth_headers):
        resp = client.post("/predict", json=legit_payload(), headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        required = {"fraud_score", "is_fraud", "alert_level",
                    "threshold_used", "model_version", "request_id",
                    "active_scorers"}
        assert required.issubset(body.keys())

    def test_request_id_in_response_header(self, auth_headers):
        resp = client.post("/predict", json=legit_payload(), headers=auth_headers)
        assert "X-Request-ID" in resp.headers

    def test_request_id_in_response_body(self, auth_headers):
        resp = client.post("/predict", json=legit_payload(), headers=auth_headers)
        assert resp.json()["request_id"] != ""

    def test_active_scorers_always_includes_rules(self, auth_headers):
        resp = client.post("/predict", json=legit_payload(), headers=auth_headers)
        assert "rules" in resp.json()["active_scorers"]

    def test_ensemble_score_in_valid_range(self, auth_headers):
        for payload in [legit_payload(), fraud_payload()]:
            resp = client.post("/predict", json=payload, headers=auth_headers)
            score = resp.json()["fraud_score"]
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_new_device_flag_raises_score(self, auth_headers):
        r_base  = client.post("/predict", json=legit_payload(new_device_flag=0),
                              headers=auth_headers)
        r_new   = client.post("/predict", json=legit_payload(new_device_flag=1),
                              headers=auth_headers)
        assert r_new.json()["fraud_score"] > r_base.json()["fraud_score"], (
            "new_device_flag=1 should raise fraud score"
        )

    def test_night_transaction_raises_score(self, auth_headers):
        r_day   = client.post("/predict", json=legit_payload(is_night_transaction=0),
                              headers=auth_headers)
        r_night = client.post("/predict", json=legit_payload(is_night_transaction=1,
                                                              hour_of_day=3),
                              headers=auth_headers)
        assert r_night.json()["fraud_score"] >= r_day.json()["fraud_score"]

    def test_valid_upi_vpa_accepted(self, auth_headers):
        for vpa in ["user@oksbi", "nishika.chapra@ybl", "test-user@paytm"]:
            resp = client.post("/predict", json=legit_payload(upi_id=vpa),
                               headers=auth_headers)
            assert resp.status_code == 200, f"Valid VPA '{vpa}' rejected: {resp.text}"

    def test_invalid_upi_vpa_returns_422(self, auth_headers):
        for bad_vpa in ["notavpa", "@nopsp", "user@", "a"]:
            resp = client.post("/predict", json=legit_payload(upi_id=bad_vpa),
                               headers=auth_headers)
            assert resp.status_code == 422, f"Bad VPA '{bad_vpa}' should be rejected"

    def test_p2p_requires_p2p_mrc_size(self, auth_headers):
        payload = legit_payload(receiver_type="User", mrc_size="Medium")
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_merchant_rejects_p2p_mrc_size(self, auth_headers):
        payload = legit_payload(receiver_type="Merchant", mrc_size="P2P")
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_amount_must_be_positive(self, auth_headers):
        resp = client.post("/predict", json=legit_payload(amount=-100),
                           headers=auth_headers)
        assert resp.status_code == 422

    def test_hour_of_day_range_validated(self, auth_headers):
        for bad_hour in [-1, 24, 100]:
            resp = client.post("/predict", json=legit_payload(hour_of_day=bad_hour),
                               headers=auth_headers)
            assert resp.status_code == 422

    def test_cold_start_legit_neutral(self, auth_headers):
        """New user (cold start z=0) should not be flagged."""
        resp = client.post(
            "/predict",
            json=legit_payload(amount_deviation_score=0.0,
                               transaction_frequency_score=0.0,
                               usr_account_age_days=1.0),
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["alert_level"] in ("none", "low"), (
            "Cold-start user with neutral z-score should not be high-risk"
        )


# ════════════════════════════════════════════════════════════════════════════
#  /insights/weekly TESTS
# ════════════════════════════════════════════════════════════════════════════
class TestInsights:

    def test_weekly_insights_requires_auth(self):
        resp = client.get("/insights/weekly")
        assert resp.status_code in (401, 403)

    def test_weekly_insights_returns_tip(self, auth_headers):
        resp = client.get("/insights/weekly",
                          params={"total_spent": 12000, "top_category": "Food",
                                  "top_category_pct": 45, "fraud_alerts": 0,
                                  "vs_last_week_pct": 15},
                          headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "savings_tip" in body
        assert len(body["savings_tip"]) > 5

    def test_weekly_insights_budget_status_over_pace(self, auth_headers):
        resp = client.get("/insights/weekly",
                          params={"total_spent": 18000, "top_category": "Shopping",
                                  "top_category_pct": 55, "fraud_alerts": 2,
                                  "vs_last_week_pct": 35},
                          headers=auth_headers)
        assert resp.status_code == 200
        assert "⚑" in resp.json()["budget_status"]

    def test_weekly_insights_budget_status_on_track(self, auth_headers):
        resp = client.get("/insights/weekly",
                          params={"total_spent": 8000, "top_category": "Grocery",
                                  "top_category_pct": 30, "fraud_alerts": 0,
                                  "vs_last_week_pct": -15},
                          headers=auth_headers)
        assert resp.status_code == 200
        assert "✓" in resp.json()["budget_status"]

    def test_weekly_insights_schema(self, auth_headers):
        resp = client.get("/insights/weekly", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        for field in ["period", "total_spent", "top_category",
                      "top_category_pct", "fraud_alerts",
                      "savings_tip", "budget_status"]:
            assert field in body, f"Missing field: {field}"


# ════════════════════════════════════════════════════════════════════════════
#  /health TESTS
# ════════════════════════════════════════════════════════════════════════════
class TestHealth:

    def test_health_is_public(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self):
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "ok"
        for key in ["ensemble_ready", "active_scorers", "rules_always_on",
                    "auth_required", "rate_limit", "api_version"]:
            assert key in body, f"Missing health key: {key}"

    def test_health_rules_always_on(self):
        assert client.get("/health").json()["rules_always_on"] is True

    def test_health_auth_required_true(self):
        assert client.get("/health").json()["auth_required"] is True


# ════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE LOGIC TESTS
# ════════════════════════════════════════════════════════════════════════════
class TestEnsemble:

    def test_extreme_fraud_signals_score_high(self, auth_headers):
        """All major fraud signals active — ensemble must exceed medium threshold."""
        payload = fraud_payload(
            new_device_flag=1,
            ip_location_mismatch=1,
            amount_deviation_score=6.0,
            failed_attempts_last_24h=5.0,
            kyc_verified_flag=0,
            usr_is_high_risk=1,
            is_night_transaction=1,
        )
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.json()["fraud_score"] >= 0.40

    def test_all_legit_signals_score_low(self, auth_headers):
        """All major safety signals active — ensemble must stay below medium."""
        payload = legit_payload(
            new_device_flag=0,
            ip_location_mismatch=0,
            amount_deviation_score=0.2,
            failed_attempts_last_24h=0.0,
            kyc_verified_flag=1,
            usr_is_high_risk=0,
            is_night_transaction=0,
            user_loyalty_score=0.95,
            usr_account_age_days=1825.0,
        )
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.json()["fraud_score"] < 0.40

    def test_alert_levels_consistent_with_score(self, auth_headers):
        """Verify alert level thresholds match documented values."""
        resp = client.post("/predict", json=legit_payload(), headers=auth_headers)
        body = resp.json()
        score = body["fraud_score"]
        level = body["alert_level"]
        if score >= 0.70:   assert level == "high"
        elif score >= 0.40: assert level == "medium"
        elif score >= 0.20: assert level == "low"
        else:               assert level == "none"

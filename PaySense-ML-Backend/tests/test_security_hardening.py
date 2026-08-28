"""
================================================================================
  tests/test_security_hardening.py
  ------------------------------------------------------------------------------
  Regression guards for the 2026-08-28 senior-SDE security/production-
  readiness audit. Each test here protects a concrete finding verified
  during that audit via a live local server (uvicorn + real smoke-test
  requests) or a real Docker build -- not a hypothetical.

  Covers:
    - /auth/token had NO rate limit at all (verified: 20 rapid wrong-
      password attempts, zero 429s). Fixed with @limiter.limit("10/minute").
    - A genuinely expired-but-correctly-signed JWT was never regression-
      tested (only garbage/malformed tokens were). Expiry enforcement
      itself needs its own test.
    - An unbounded numeric field (amount_deviation_score has no ge/le
      constraint) must not let an extreme value produce a NaN/Inf response
      or crash the server (verified live: 1e300 and -1e300 both currently
      return a normal, finite, in-range response).

  Does NOT cover (see .private/SECURITY_AUDIT.md and .private/
  PRODUCTION_READINESS.md for the full non-public detail):
    - The Dockerfile bugs (broken COPY syntax; missing src/) -- verified
      via a real `docker build`/`docker run`, not something a pytest suite
      can exercise without a Docker daemon in CI.
    - Live production CORS/deployment status -- not testable without
      Render dashboard access; documented as NOT VERIFIED where relevant.
================================================================================
"""
import math
import time

import pytest
from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def auth_token():
    resp = client.post("/auth/token", json={"username": "paysense", "password": "guardian2025"})
    assert resp.status_code == 200, f"Token request failed: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture(scope="module")
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


def legit_payload(**overrides) -> dict:
    base = {
        "receiver_type": "Merchant", "transaction_type": "P2M", "payment_app": "GPay",
        "device_type": "Android", "usr_age_group": "25-34", "usr_preferred_app": "GPay",
        "usr_preferred_device": "Android", "mrc_category": "Grocery", "mrc_size": "Medium",
        "amount": 577.16, "hour_of_day": 10, "day_of_week": 1, "is_weekend": 0,
        "is_night_transaction": 0, "time_since_last_txn_min": 45.5,
        "transaction_velocity": 0.12, "amount_deviation_score": 1.8,
        "failed_attempts_last_24h": 1.0, "recurring_payment_flag": 0,
        "transaction_frequency_score": 0.45, "new_device_flag": 1,
        "ip_location_mismatch": 0, "user_city_tier": 2, "user_avg_monthly_txn": 32.0,
        "user_avg_txn_value": 850.0, "user_loyalty_score": 0.62,
        "balance_after_transaction": 24500.0, "txn_success_flag": 1,
        "kyc_verified_flag": 1, "usr_home_city_tier": 2, "usr_account_age_days": 720.0,
        "usr_linked_bank_count": 2.0, "usr_avg_monthly_txn_profile": 32.0,
        "usr_avg_txn_value_profile": 850.0, "usr_is_high_risk": 0,
        "mrc_avg_daily_txn": 120.0, "mrc_is_registered": 1,
    }
    base.update(overrides)
    return base


# ── JWT expiry enforcement (not just malformed-token rejection) --------------
# tests/test_api.py already covers a garbage/malformed bearer token, but
# never a token that is CORRECTLY SIGNED with the real secret and simply
# expired -- a materially different code path (jose.jwt.decode raises
# ExpiredSignatureError, a JWTError subclass, for this case specifically).
def test_predict_rejects_genuinely_expired_token(auth_headers):
    # sanity: a valid token works first, so a failure below is about expiry
    # specifically, not some other auth regression.
    ok = client.post("/predict", json=legit_payload(), headers=auth_headers)
    assert ok.status_code == 200, f"Valid token unexpectedly rejected: {ok.text}"

    # create_access_token always sets exp in the future; construct an
    # already-expired one directly with the real secret instead.
    from jose import jwt as jose_jwt
    really_expired = jose_jwt.encode(
        {"sub": "paysense", "exp": time.time() - 3600},
        main.SECRET_KEY, algorithm=main.ALGORITHM,
    )
    resp = client.post("/predict", json=legit_payload(),
                        headers={"Authorization": f"Bearer {really_expired}"})
    assert resp.status_code == 401, (
        f"A correctly-signed but expired JWT was accepted (status "
        f"{resp.status_code}) -- expiry is no longer enforced. This is a "
        f"different failure mode than a malformed/garbage token (which "
        f"tests/test_api.py already covers) and would let an old token "
        f"stay valid forever."
    )


# ── Unbounded numeric field cannot produce NaN/Inf or crash the server -------
# amount_deviation_score has no ge/le/gt constraint in TransactionInput
# (unlike most other numeric fields). Verified live against a running
# server: extreme values (1e300, -1e300) currently return a normal, finite
# response -- this test locks that in.
@pytest.mark.parametrize("extreme_value", [1e300, -1e300, 1e15, -1e15])
def test_predict_extreme_unbounded_field_does_not_crash_or_produce_nan(auth_headers, extreme_value):
    resp = client.post(
        "/predict",
        json=legit_payload(amount_deviation_score=extreme_value),
        headers=auth_headers,
    )
    assert resp.status_code == 200, (
        f"amount_deviation_score={extreme_value} caused a non-200 response "
        f"({resp.status_code}): {resp.text[:300]} -- an unbounded field "
        f"should never crash the server, even if it should arguably be "
        f"rejected by validation."
    )
    body = resp.json()
    score = body["fraud_score"]
    assert math.isfinite(score), (
        f"amount_deviation_score={extreme_value} produced a non-finite "
        f"fraud_score ({score!r}) -- this is exactly the NaN/Inf response "
        f"class of bug the audit checked for. Investigate which scorer "
        f"(_score_paysense, _score_light_lr, _score_rules) let a non-finite "
        f"value through before the final clamp in src/fraud_model.score()."
    )
    assert 0.0 <= score <= 1.0, (
        f"amount_deviation_score={extreme_value} produced fraud_score="
        f"{score}, outside the documented [0, 1] range."
    )


# ── Fail-open config defaults emit a startup warning, not a silent bypass ----
# APP_ENV defaults to "development" (auth bypass active) and ALLOWED_ORIGINS
# has a documented "*" dev default -- both intentional for local-dev
# convenience, but a deployer who forgets to override either should see a
# loud warning, not silence. This test re-imports main's warning logic
# directly (importing the whole module twice in one process would re-run
# side effects unsafely) by calling the same condition checks main.py uses.
def test_non_production_app_env_would_trigger_a_warning():
    # main is already imported with APP_ENV=production (conftest.py pins
    # this before any test module import) -- so we can't observe the
    # warning firing at this module's own import time. Instead, verify the
    # guard condition and message are present in main.py's source, which is
    # what actually matters: that the check exists and hasn't been quietly
    # removed.
    import inspect
    source = inspect.getsource(main)
    assert 'if APP_ENV != "production":' in source, (
        "The startup guard warning for a non-production APP_ENV appears to "
        "have been removed from main.py. This guard exists so a deployment "
        "that forgets to set APP_ENV=production gets a loud log warning "
        "(auth bypass is active) instead of silently running open."
    )
    assert 'ALLOWED_ORIGINS", "") == "*"' in source, (
        "The startup guard warning for ALLOWED_ORIGINS=* appears to have "
        "been removed from main.py."
    )


# ── /auth/token rate limiting -------------------------------------------------
# Found by empirical smoke testing (20 rapid wrong-password requests against a
# live local server, all 20 returned 401 with zero 429s) that /auth/token --
# the only unauthenticated endpoint that accepts a secret -- had no rate limit
# at all, unlike every other endpoint. Fixed by adding @limiter.limit
# ("10/minute"). Deliberately the LAST test in this file: it exhausts
# /auth/token's rate-limit quota for the rest of the pytest process, which
# would break the module-scoped auth_headers fixture (and therefore every
# other test above) if it ran first. Sends a generous burst (not tied to an
# exact count, since slowapi's in-memory limiter state may already be
# partially consumed by other test files sharing the same process) and
# asserts at least one 429 appears.
def test_zz_auth_token_endpoint_is_rate_limited():
    statuses = []
    for _ in range(15):
        resp = client.post("/auth/token", json={"username": "wrong", "password": "wrong"})
        statuses.append(resp.status_code)
    assert 429 in statuses, (
        f"Sent 15 rapid requests to /auth/token and got no 429 (statuses: "
        f"{statuses}). This endpoint accepts a secret (username/password) "
        f"and MUST be rate-limited against brute-force guessing -- if this "
        f"regresses, re-check the @limiter.limit(...) decorator on "
        f"POST /auth/token in main.py."
    )

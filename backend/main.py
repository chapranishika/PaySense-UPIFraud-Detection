"""
================================================================================
  PaySense — FastAPI Inference Backend
  File    : main.py
  Author  : Senior Backend Engineer
  Project : PaySense — ML-Based UPI Fraud Detection

  Run locally:
      pip install fastapi uvicorn[standard] pandas joblib xgboost scikit-learn
      uvicorn main:app --reload --port 8000

  The Android app sends a POST /predict with the raw transaction JSON.
  This server transforms it, scores it, and returns the fraud verdict.
================================================================================
"""

import logging
from contextlib import asynccontextmanager
from typing     import Literal, Optional

import joblib
import numpy  as np
import pandas as pd
import uvicorn
from fastapi              import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic             import BaseModel, Field, field_validator

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  [%(levelname)s]  %(message)s",
)
log = logging.getLogger("paysense")


# ════════════════════════════════════════════════════════════════════════════
#  1. MODEL ARTEFACT PATHS
#  Change these paths when deploying to a server / Docker container.
# ════════════════════════════════════════════════════════════════════════════
MODEL_PATH        = "paysense_model.pkl"
PREPROCESSOR_PATH = "paysense_preprocessor.pkl"
THRESHOLD_PATH    = "paysense_threshold.pkl"
FEATURE_NAMES_PATH= "paysense_feature_names.pkl"


# ════════════════════════════════════════════════════════════════════════════
#  2. GLOBAL MODEL REGISTRY
#  All artefacts are loaded ONCE at startup into this dict and reused for
#  every subsequent request.  Never load from disk inside an endpoint.
# ════════════════════════════════════════════════════════════════════════════
ML: dict = {}


# ════════════════════════════════════════════════════════════════════════════
#  3. LIFESPAN — STARTUP & SHUTDOWN EVENTS
#  The lifespan context manager replaces the deprecated @app.on_event pattern.
#  Code before `yield` runs on startup; after `yield` runs on shutdown.
# ════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────────────────────
    log.info("PaySense API starting — loading model artefacts …")
    try:
        ML["model"]         = joblib.load(MODEL_PATH)
        ML["preprocessor"]  = joblib.load(PREPROCESSOR_PATH)
        ML["threshold"]     = joblib.load(THRESHOLD_PATH)
        ML["feature_names"] = joblib.load(FEATURE_NAMES_PATH)
        log.info(f"  ✓ Model loaded          ({MODEL_PATH})")
        log.info(f"  ✓ Preprocessor loaded   ({PREPROCESSOR_PATH})")
        log.info(f"  ✓ Threshold loaded      (value = {ML['threshold']})")
        log.info(f"  ✓ Feature names loaded  ({len(ML['feature_names'])} features)")
        log.info("PaySense API is ready to accept requests.")
    except FileNotFoundError as e:
        log.critical(f"  ✗ Artefact not found: {e}")
        log.critical("    Ensure all .pkl files are in the same directory as main.py")
        raise RuntimeError("Failed to load model artefacts — server cannot start.") from e

    yield   # ← server is live here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    ML.clear()
    log.info("PaySense API shut down cleanly.")


# ════════════════════════════════════════════════════════════════════════════
#  4. FASTAPI APP INSTANCE
# ════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "PaySense Fraud Detection API",
    description = (
        "ML-powered UPI fraud detection backend for the PaySense Android app.\n\n"
        "POST a transaction to `/predict` and receive a fraud score, "
        "binary decision, and alert level in return."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Allow the Android app and any local dev client to reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # tighten to your domain in production
    allow_methods  = ["POST", "GET"],
    allow_headers  = ["*"],
)


# ════════════════════════════════════════════════════════════════════════════
#  5. PYDANTIC INPUT MODEL
#
#  Every field maps 1-to-1 to a column in paysense_feature_names.pkl.
#  Categorical fields use Literal types — Pydantic will reject any value
#  not in the allowed set BEFORE the request reaches inference logic.
#  Fields that are Optional (nullable in the training data) accept None;
#  the preprocessor's SimpleImputer handles them downstream.
# ════════════════════════════════════════════════════════════════════════════
class TransactionInput(BaseModel):

    # ── Categorical fields (validated against training vocabulary) ────────────
    receiver_type: Literal["Merchant", "User"] = Field(
        ..., example="Merchant",
        description="Whether the receiver is a registered merchant or a peer user (P2P)."
    )
    transaction_type: Literal[
        "P2M", "P2P", "Bill Payment", "EMI", "Recharge", "Subscription", "ATM"
    ] = Field(..., example="P2M")

    payment_app: Literal[
        "GPay", "PhonePe", "Paytm", "Amazon Pay", "BHIM", "WhatsApp Pay"
    ] = Field(..., example="GPay")

    device_type: Literal["Android", "iOS", "Web"] = Field(..., example="Android")

    usr_age_group: Literal["18-24", "25-34", "35-44", "45-54", "55+"] = Field(
        ..., example="25-34"
    )
    usr_preferred_app: Literal[
        "GPay", "PhonePe", "Paytm", "Amazon Pay", "BHIM", "WhatsApp Pay"
    ] = Field(..., example="GPay")

    usr_preferred_device: Literal["Android", "iOS", "Web"] = Field(
        ..., example="Android"
    )
    mrc_category: Literal[
        "Food", "Food & Dining", "Travel", "Grocery", "Electronics",
        "Clothing", "Healthcare", "Entertainment", "Education",
        "Utilities", "Fuel", "Insurance", "Shopping", "Recharge",
        "P2P Transfer",
    ] = Field(..., example="Grocery",
              description="Use 'P2P Transfer' when receiver_type is 'User'.")

    mrc_size: Literal["Small", "Medium", "Enterprise", "P2P"] = Field(
        ..., example="Medium",
        description="Use 'P2P' when receiver_type is 'User'."
    )

    # ── Transaction numeric fields ─────────────────────────────────────────
    amount: float = Field(..., gt=0, example=577.16,
                          description="Transaction amount in INR.")
    hour_of_day: int  = Field(..., ge=0, le=23, example=10)
    day_of_week: int  = Field(..., ge=0, le=6,  example=1,
                              description="0=Monday … 6=Sunday")
    is_weekend: int   = Field(..., ge=0, le=1,  example=0)
    is_night_transaction: int = Field(..., ge=0, le=1, example=0,
                                      description="1 if hour < 6 or hour >= 22.")

    # ── Velocity & behavioural signals ────────────────────────────────────
    time_since_last_txn_min: float = Field(
        ..., example=45.5,
        description="Minutes since the user's previous transaction. "
                    "Use a large positive value (e.g. 10080) for first transaction."
    )
    transaction_velocity: float = Field(
        ..., ge=0, example=0.12,
        description="Normalised count of transactions in the last 1-hour window."
    )
    amount_deviation_score: float = Field(
        ..., example=1.8,
        description="Z-score of this amount vs. user's historical average."
    )
    failed_attempts_last_24h: float = Field(..., ge=0, example=1.0)
    recurring_payment_flag: int     = Field(..., ge=0, le=1, example=0)
    transaction_frequency_score: float = Field(..., ge=0, example=0.45)

    # ── Security flags ────────────────────────────────────────────────────
    new_device_flag: int       = Field(..., ge=0, le=1, example=1,
                                       description="1 if this device has not been seen before for this user.")
    ip_location_mismatch: int  = Field(..., ge=0, le=1, example=0,
                                       description="1 if IP geolocation differs from user's registered city.")

    # ── Account-level fields ──────────────────────────────────────────────
    user_city_tier: int      = Field(..., ge=1, le=3, example=2)
    user_avg_monthly_txn: float  = Field(..., gt=0, example=32.0)
    user_avg_txn_value: float    = Field(..., gt=0, example=850.0)
    user_loyalty_score: float    = Field(..., ge=0, le=1, example=0.62)
    balance_after_transaction: float = Field(..., example=24500.0)
    txn_success_flag: int        = Field(..., ge=0, le=1, example=1,
                                         description="1 if the transaction completed successfully.")
    kyc_verified_flag: int       = Field(..., ge=0, le=1, example=1)

    # ── User profile fields (from user lookup at login) ───────────────────
    usr_home_city_tier: int          = Field(..., ge=1, le=3, example=2)
    usr_account_age_days: float      = Field(..., ge=0, example=720.0)
    usr_linked_bank_count: float     = Field(..., ge=1, example=2.0)
    usr_avg_monthly_txn_profile: float = Field(..., gt=0, example=32.0)
    usr_avg_txn_value_profile: float   = Field(..., gt=0, example=850.0)
    usr_is_high_risk: int              = Field(..., ge=0, le=1, example=0)

    # ── Merchant profile fields (from merchant lookup at payment) ─────────
    mrc_avg_daily_txn: float     = Field(..., ge=0, example=120.0,
                                         description="Set to 0 for P2P transfers.")
    mrc_is_registered: int       = Field(..., ge=0, le=1, example=1)
    mrc_rating: Optional[float]  = Field(None, ge=0, le=5, example=4.1,
                                         description="Leave null / None for P2P transfers.")

    # ── Supplement-source risk scores (optional — set null if unavailable) ─
    device_risk_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, example=None,
        description="Device risk score from your device-fingerprinting service. "
                    "Leave null if not available — model handles it gracefully."
    )
    ip_risk_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, example=None,
        description="IP risk score from your threat-intel service. "
                    "Leave null if not available."
    )

    # ── Cross-field validator: flag P2P consistency errors ──────────────────
    @field_validator("mrc_size")
    @classmethod
    def p2p_size_consistency(cls, v, info):
        receiver = info.data.get("receiver_type")
        if receiver == "User" and v != "P2P":
            raise ValueError(
                "When receiver_type is 'User' (P2P transfer), "
                "mrc_size must be 'P2P'."
            )
        if receiver == "Merchant" and v == "P2P":
            raise ValueError(
                "mrc_size 'P2P' is only valid when receiver_type is 'User'."
            )
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "receiver_type": "Merchant",
            "transaction_type": "P2M",
            "payment_app": "GPay",
            "device_type": "iOS",
            "usr_age_group": "35-44",
            "usr_preferred_app": "GPay",
            "usr_preferred_device": "Android",
            "mrc_category": "Grocery",
            "mrc_size": "Medium",
            "amount": 577.16,
            "hour_of_day": 10,
            "day_of_week": 0,
            "is_weekend": 0,
            "is_night_transaction": 0,
            "time_since_last_txn_min": 45.5,
            "transaction_velocity": 0.12,
            "amount_deviation_score": 1.8,
            "failed_attempts_last_24h": 1.0,
            "recurring_payment_flag": 0,
            "transaction_frequency_score": 0.45,
            "new_device_flag": 1,
            "ip_location_mismatch": 0,
            "user_city_tier": 2,
            "user_avg_monthly_txn": 32.0,
            "user_avg_txn_value": 850.0,
            "user_loyalty_score": 0.62,
            "balance_after_transaction": 24500.0,
            "txn_success_flag": 1,
            "kyc_verified_flag": 1,
            "usr_home_city_tier": 2,
            "usr_account_age_days": 720.0,
            "usr_linked_bank_count": 2.0,
            "usr_avg_monthly_txn_profile": 32.0,
            "usr_avg_txn_value_profile": 850.0,
            "usr_is_high_risk": 0,
            "mrc_avg_daily_txn": 120.0,
            "mrc_is_registered": 1,
            "mrc_rating": 4.1,
            "device_risk_score": None,
            "ip_risk_score": None,
        }
    }}


# ════════════════════════════════════════════════════════════════════════════
#  6. PYDANTIC OUTPUT MODEL
# ════════════════════════════════════════════════════════════════════════════
class PredictionResponse(BaseModel):
    fraud_score : float  = Field(..., description="Raw fraud probability (0.0 – 1.0).")
    is_fraud    : bool   = Field(..., description="True if fraud_score ≥ optimal threshold.")
    alert_level : Literal["none", "low", "medium", "high"] = Field(
        ...,
        description=(
            "none   → fraud_score < 0.20  (very likely legitimate)\n"
            "low    → 0.20 ≤ score < 0.40 (monitor silently)\n"
            "medium → 0.40 ≤ score < 0.70 (soft-block: require PIN re-entry)\n"
            "high   → score ≥ 0.70        (hard-block: freeze and notify user)"
        ),
    )
    threshold_used: float = Field(..., description="The decision threshold applied.")
    model_version : str   = Field("1.0.0", description="Model artefact version tag.")


# ════════════════════════════════════════════════════════════════════════════
#  7. HELPER: ALERT LEVEL LOGIC
#  Decoupled from the endpoint so it can be unit-tested independently.
# ════════════════════════════════════════════════════════════════════════════
def compute_alert_level(fraud_score: float) -> str:
    """
    Maps a continuous fraud probability to a 4-tier action level.

    Tiers are designed for graduated UX responses in the Android app:
        none   → log transaction normally, no user interruption.
        low    → log with internal flag; trigger background review.
        medium → pause payment UI; ask user to confirm with UPI PIN.
        high   → block the payment immediately; push a security notification.
    """
    if fraud_score >= 0.70:
        return "high"
    elif fraud_score >= 0.40:
        return "medium"
    elif fraud_score >= 0.20:
        return "low"
    else:
        return "none"


# ════════════════════════════════════════════════════════════════════════════
#  8. /predict — MAIN INFERENCE ENDPOINT
# ════════════════════════════════════════════════════════════════════════════
@app.post(
    "/predict",
    response_model = PredictionResponse,
    summary        = "Score a UPI transaction for fraud",
    tags           = ["Inference"],
)
async def predict(transaction: TransactionInput) -> PredictionResponse:
    """
    Accepts a raw UPI transaction payload from the Android app and returns
    a fraud score, binary decision, and graduated alert level.

    The five-step inference pipeline:
    ─────────────────────────────────
    1. Convert JSON → single-row DataFrame in the exact column order the
       preprocessor expects (enforced via paysense_feature_names.pkl).
    2. Transform using the frozen ColumnTransformer (OrdinalEncoder + imputer).
    3. Extract fraud probability from XGBoost's predict_proba output.
    4. Compare probability against the frozen optimal threshold.
    5. Return fraud_score, is_fraud, and alert_level.
    """

    # ── Step 1: JSON → DataFrame (enforcing training column order) ───────────
    #
    #  transaction.model_dump() returns a plain Python dict from the Pydantic
    #  model.  We wrap it in a list to create a single-row DataFrame, then
    #  reindex to the exact column order saved in paysense_feature_names.pkl.
    #  If the Android app sends extra fields, reindex silently drops them.
    #  If it omits optional fields, reindex fills with NaN (handled in step 2).
    try:
        raw_dict   = transaction.model_dump()
        input_df   = pd.DataFrame([raw_dict])
        input_df   = input_df.reindex(columns=ML["feature_names"])
    except Exception as exc:
        log.error(f"DataFrame construction failed: {exc}")
        raise HTTPException(status_code=422, detail=f"Input construction error: {exc}")

    # ── Step 2: Transform via frozen ColumnTransformer ────────────────────────
    #
    #  preprocessor.transform() applies OrdinalEncoder to the 9 categorical
    #  columns and SimpleImputer (median strategy) to all numeric columns,
    #  using the exact mappings learned from the training data.
    #  Any unseen categorical value is gracefully encoded as -1 by the
    #  OrdinalEncoder's handle_unknown='use_encoded_value' setting.
    try:
        transformed = ML["preprocessor"].transform(input_df)
    except Exception as exc:
        log.error(f"Preprocessing failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {exc}")

    # ── Step 3: Extract fraud probability from XGBoost ───────────────────────
    #
    #  predict_proba returns [[prob_legitimate, prob_fraud]].
    #  We take index [:, 1] to isolate the fraud class probability.
    #  round() to 6 decimal places avoids floating-point noise in the response.
    try:
        fraud_score = float(round(
            ML["model"].predict_proba(transformed)[0, 1], 6
        ))
    except Exception as exc:
        log.error(f"Model inference failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # ── Step 4: Apply frozen optimal threshold ────────────────────────────────
    threshold = float(ML["threshold"])
    is_fraud  = bool(fraud_score >= threshold)

    # ── Step 5: Build and return response ─────────────────────────────────────
    alert_level = compute_alert_level(fraud_score)

    log.info(
        f"Prediction → score={fraud_score:.4f}  "
        f"is_fraud={is_fraud}  alert={alert_level}  "
        f"amount=₹{transaction.amount}  app={transaction.payment_app}"
    )

    return PredictionResponse(
        fraud_score    = fraud_score,
        is_fraud       = is_fraud,
        alert_level    = alert_level,
        threshold_used = threshold,
        model_version  = "1.0.0",
    )


# ════════════════════════════════════════════════════════════════════════════
#  9. HEALTH CHECK ENDPOINT
#  The Android app can ping this before sending a transaction to confirm
#  the server is up and the model is loaded.
# ════════════════════════════════════════════════════════════════════════════
@app.get("/health", tags=["System"])
async def health_check():
    """Returns server status and confirms model artefacts are loaded."""
    model_ready = "model" in ML and "preprocessor" in ML
    return {
        "status"       : "ok" if model_ready else "degraded",
        "model_loaded" : model_ready,
        "threshold"    : ML.get("threshold", None),
        "feature_count": len(ML.get("feature_names", [])),
        "api_version"  : "1.0.0",
    }


# ════════════════════════════════════════════════════════════════════════════
#  10. ROOT REDIRECT TO DOCS
# ════════════════════════════════════════════════════════════════════════════
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "PaySense API is running. Visit /docs for the interactive UI."}


# ════════════════════════════════════════════════════════════════════════════
#  11. ENTRYPOINT (run directly with: python main.py)
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,    # set False in production
        workers = 1,       # increase to CPU count in production
    )


# ════════════════════════════════════════════════════════════════════════════
#
#  SAMPLE TEST PAYLOADS
#  ─────────────────────────────────────────────────────────────────────────
#
#  1. HEALTH CHECK
#  ───────────────
#  curl http://localhost:8000/health
#
#
#  2. SUSPICIOUS TRANSACTION (high-risk signals: new_device_flag=1,
#     ip_location_mismatch=1, late night hour, high amount deviation)
#  ────────────────────────────────────────────────────────────────────
#  curl -X POST http://localhost:8000/predict \
#    -H "Content-Type: application/json" \
#    -d '{
#      "receiver_type": "Merchant",
#      "transaction_type": "P2M",
#      "payment_app": "GPay",
#      "device_type": "iOS",
#      "usr_age_group": "35-44",
#      "usr_preferred_app": "GPay",
#      "usr_preferred_device": "Android",
#      "mrc_category": "Grocery",
#      "mrc_size": "Medium",
#      "amount": 577.16,
#      "hour_of_day": 2,
#      "day_of_week": 0,
#      "is_weekend": 0,
#      "is_night_transaction": 1,
#      "time_since_last_txn_min": 3.0,
#      "transaction_velocity": 0.95,
#      "amount_deviation_score": 4.2,
#      "failed_attempts_last_24h": 3.0,
#      "recurring_payment_flag": 0,
#      "transaction_frequency_score": 0.88,
#      "new_device_flag": 1,
#      "ip_location_mismatch": 1,
#      "user_city_tier": 1,
#      "user_avg_monthly_txn": 12.0,
#      "user_avg_txn_value": 200.0,
#      "user_loyalty_score": 0.11,
#      "balance_after_transaction": 980.0,
#      "txn_success_flag": 1,
#      "kyc_verified_flag": 0,
#      "usr_home_city_tier": 1,
#      "usr_account_age_days": 14.0,
#      "usr_linked_bank_count": 1.0,
#      "usr_avg_monthly_txn_profile": 12.0,
#      "usr_avg_txn_value_profile": 200.0,
#      "usr_is_high_risk": 1,
#      "mrc_avg_daily_txn": 120.0,
#      "mrc_is_registered": 1,
#      "mrc_rating": 2.1,
#      "device_risk_score": 0.91,
#      "ip_risk_score": 0.87
#    }'
#
#
#  3. NORMAL P2P TRANSFER (all low-risk signals)
#  ─────────────────────────────────────────────
#  curl -X POST http://localhost:8000/predict \
#    -H "Content-Type: application/json" \
#    -d '{
#      "receiver_type": "User",
#      "transaction_type": "P2P",
#      "payment_app": "PhonePe",
#      "device_type": "Android",
#      "usr_age_group": "25-34",
#      "usr_preferred_app": "PhonePe",
#      "usr_preferred_device": "Android",
#      "mrc_category": "P2P Transfer",
#      "mrc_size": "P2P",
#      "amount": 500.0,
#      "hour_of_day": 14,
#      "day_of_week": 2,
#      "is_weekend": 0,
#      "is_night_transaction": 0,
#      "time_since_last_txn_min": 1440.0,
#      "transaction_velocity": 0.05,
#      "amount_deviation_score": 0.3,
#      "failed_attempts_last_24h": 0.0,
#      "recurring_payment_flag": 1,
#      "transaction_frequency_score": 0.22,
#      "new_device_flag": 0,
#      "ip_location_mismatch": 0,
#      "user_city_tier": 2,
#      "user_avg_monthly_txn": 40.0,
#      "user_avg_txn_value": 650.0,
#      "user_loyalty_score": 0.78,
#      "balance_after_transaction": 12000.0,
#      "txn_success_flag": 1,
#      "kyc_verified_flag": 1,
#      "usr_home_city_tier": 2,
#      "usr_account_age_days": 900.0,
#      "usr_linked_bank_count": 2.0,
#      "usr_avg_monthly_txn_profile": 40.0,
#      "usr_avg_txn_value_profile": 650.0,
#      "usr_is_high_risk": 0,
#      "mrc_avg_daily_txn": 0.0,
#      "mrc_is_registered": 1,
#      "mrc_rating": null,
#      "device_risk_score": null,
#      "ip_risk_score": null
#    }'
#
# ════════════════════════════════════════════════════════════════════════════

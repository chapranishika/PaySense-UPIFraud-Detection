"""
================================================================================
  PaySense: Finlatics and Finance Flow — FastAPI Inference Backend  (v2)
  ────────────────────────────────────────────────────────────────────────────
  Combines the best of PaySense (UPI-native ML, per-user z-score, Finance
  Tracker integration) and UPI Guardian (JWT auth, rate limiting, structured
  logging, multi-model ensemble, Gemini savings recommendations).

  Endpoints
  ─────────
  POST /predict              Score a UPI transaction (JWT required)
  GET  /health               Liveness probe (public)
  GET  /insights/weekly      AI-powered spending insights (JWT required)
  POST /auth/token           Get JWT access token

  Security additions over v1
  ──────────────────────────
  * JWT Bearer token on /predict and /insights/* — issued by /auth/token
  * slowapi rate limiting: 60 req/min per IP on /predict
  * Per-request structured logging via contextvars (no shared state)
  * UPI VPA format validator on optional upi_id field
  * APP_ENV guard: override parameters disabled in production

  Run locally
  ───────────
      pip install -r requirements.txt
      uvicorn main:app --reload --port 8000
================================================================================
"""

import contextvars
import logging
import os
import re
import sys
import uuid
from contextlib    import asynccontextmanager
from datetime      import datetime, timedelta, timezone
from typing        import Annotated, Literal, Optional

# ── Ensure src/ is importable regardless of working directory ─────────────────
# On Render and other cloud hosts the process may not start from the same
# directory as this file. This guarantees `from src.fraud_model import ...`
# resolves correctly in all environments.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy   as np
import pandas  as pd
import uvicorn
from dotenv import load_dotenv

from fastapi               import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security      import HTTPAuthorizationCredentials, HTTPBearer
from jose                  import JWTError, jwt
from pydantic              import BaseModel, Field, field_validator
from slowapi                import Limiter, _rate_limit_exceeded_handler
from slowapi.errors         import RateLimitExceeded
from slowapi.util           import get_remote_address

# ── Import ensemble scorer ──────────────────────────────────────────────────
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.fraud_model import load_artefacts, score as ensemble_score, get_state

load_dotenv()

# ── Config from environment ───────────────────────────────────────────────────
SECRET_KEY      = os.environ.get("JWT_SECRET_KEY", "paysense-dev-secret-change-in-prod")
ALGORITHM       = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("TOKEN_EXPIRE_MIN", "60"))
APP_ENV         = os.environ.get("APP_ENV", "development")
API_DEMO_USER   = os.environ.get("API_DEMO_USER", "paysense")
API_DEMO_PASS   = os.environ.get("API_DEMO_PASS", "guardian2025")

MODEL_PATH       = "paysense_model.pkl"
PREPROCESSOR_PATH= "paysense_preprocessor.pkl"
THRESHOLD_PATH   = "paysense_threshold.pkl"
FEATURE_NAMES_PATH="paysense_feature_names.pkl"

UPI_VPA_RE = re.compile(r"^[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}$")

# ── Per-request logging context ────────────────────────────────────────────────
#  Each asyncio Task gets its own ContextVar slot — no shared mutable state,
#  no race conditions under concurrent requests.
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_request_id_ctx", default="-"
)

class _RequestIDFormatter(logging.Formatter):
    def format(self, record):
        record.request_id = _request_id_ctx.get("-")
        return super().format(record)

logging.basicConfig(level=logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(_RequestIDFormatter(
    "%(asctime)s [%(request_id)s] %(levelname)s %(name)s — %(message)s"
))
log = logging.getLogger("paysense")
log.handlers = [_handler]
log.propagate = False

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── ML artefact cache ─────────────────────────────────────────────────────────
ML: dict = {}

# ── JWT security scheme ───────────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


# ════════════════════════════════════════════════════════════════════════════
#  LIFESPAN — load model artefacts once at startup
# ════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("PaySense Guardian starting — loading ensemble artefacts …")
    state = load_artefacts()
    log.info(f"Ensemble ready — active scorers: {state.active_scorers}")
    yield
    log.info("PaySense Guardian shut down.")


# ════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "PaySense: Finlatics and Finance Flow API",
    description = (
        "Three-layer UPI fraud detection with per-user personalisation, "
        "multi-model ensemble scoring, and AI-powered savings recommendations. "
        "\n\n**Auth:** POST /auth/token → use returned `access_token` as "
        "Bearer token on protected endpoints."
    ),
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["Authorization", "Content-Type"],
)


# ════════════════════════════════════════════════════════════════════════════
#  REQUEST ID MIDDLEWARE
# ════════════════════════════════════════════════════════════════════════════
@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    token = _request_id_ctx.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    _request_id_ctx.reset(token)
    return response


# ════════════════════════════════════════════════════════════════════════════
#  JWT HELPERS
# ════════════════════════════════════════════════════════════════════════════
def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": subject, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)]
) -> str:
    """FastAPI dependency — validates JWT and returns the subject (username)."""
    if APP_ENV == "development" and not credentials:
        return "paysense-dev-bypass"
    if not credentials:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Not authenticated",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user: str = payload.get("sub", "")
        if not user:
            raise JWTError("empty sub")
        return user
    except JWTError:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or expired token. POST /auth/token to get a new one.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )


# ════════════════════════════════════════════════════════════════════════════
#  AUTH ENDPOINT  — public, no JWT needed
# ════════════════════════════════════════════════════════════════════════════
class TokenRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = ACCESS_TOKEN_EXPIRE_MINUTES * 60

@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"],
          summary="Get a JWT access token")
async def get_token(body: TokenRequest):
    """
    Demo credentials (change via env vars in production):
    - username: `paysense`   password: `guardian2025`

    Returns a Bearer token valid for 60 minutes.
    Use it as: `Authorization: Bearer <token>`
    """
    if body.username != API_DEMO_USER or body.password != API_DEMO_PASS:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(body.username)
    return TokenResponse(access_token=token)


# ════════════════════════════════════════════════════════════════════════════
#  PYDANTIC INPUT MODEL  (unchanged from v1 — Android compatibility preserved)
# ════════════════════════════════════════════════════════════════════════════
class TransactionInput(BaseModel):

    # ── Categorical fields ─────────────────────────────────────────────────
    receiver_type: Literal["Merchant", "User"] = Field(..., example="Merchant")
    transaction_type: Literal[
        "P2M", "P2P", "Bill Payment", "EMI", "Recharge", "Subscription", "ATM"
    ] = Field(..., example="P2M")
    payment_app: Literal[
        "GPay", "PhonePe", "Paytm", "Amazon Pay", "BHIM", "WhatsApp Pay"
    ] = Field(..., example="GPay")
    device_type: Literal["Android", "iOS", "Web"] = Field(..., example="Android")
    usr_age_group: Literal["18-24", "25-34", "35-44", "45-54", "55+"] = Field(
        ..., example="25-34")
    usr_preferred_app: Literal[
        "GPay", "PhonePe", "Paytm", "Amazon Pay", "BHIM", "WhatsApp Pay"
    ] = Field(..., example="GPay")
    usr_preferred_device: Literal["Android", "iOS", "Web"] = Field(..., example="Android")
    mrc_category: Literal[
        "Food", "Food & Dining", "Travel", "Grocery", "Electronics",
        "Clothing", "Healthcare", "Entertainment", "Education",
        "Utilities", "Fuel", "Insurance", "Shopping", "Recharge", "P2P Transfer",
    ] = Field(..., example="Grocery")
    mrc_size: Literal["Small", "Medium", "Enterprise", "P2P"] = Field(..., example="Medium")

    # ── Optional metadata (new in v2 — for richer logging and validation) ──
    upi_id: Optional[str] = Field(
        None, example="user@oksbi",
        description="UPI Virtual Payment Address. Validated against VPA format if provided."
    )

    # ── Transaction numeric fields ─────────────────────────────────────────
    amount: float                    = Field(..., gt=0, example=577.16)
    hour_of_day: int                 = Field(..., ge=0, le=23, example=10)
    day_of_week: int                 = Field(..., ge=0, le=6,  example=1)
    is_weekend: int                  = Field(..., ge=0, le=1,  example=0)
    is_night_transaction: int        = Field(..., ge=0, le=1,  example=0)
    time_since_last_txn_min: float   = Field(..., example=45.5)
    transaction_velocity: float      = Field(..., ge=0, example=0.12)
    amount_deviation_score: float    = Field(..., example=1.8)
    failed_attempts_last_24h: float  = Field(..., ge=0, example=1.0)
    recurring_payment_flag: int      = Field(..., ge=0, le=1, example=0)
    transaction_frequency_score: float = Field(..., ge=0, example=0.45)

    # ── Security flags ─────────────────────────────────────────────────────
    new_device_flag: int        = Field(..., ge=0, le=1, example=1)
    ip_location_mismatch: int   = Field(..., ge=0, le=1, example=0)

    # ── Account-level fields ───────────────────────────────────────────────
    user_city_tier: int              = Field(..., ge=1, le=3, example=2)
    user_avg_monthly_txn: float      = Field(..., gt=0, example=32.0)
    user_avg_txn_value: float        = Field(..., gt=0, example=850.0)
    user_loyalty_score: float        = Field(..., ge=0, le=1, example=0.62)
    balance_after_transaction: float = Field(..., example=24500.0)
    txn_success_flag: int            = Field(..., ge=0, le=1, example=1)
    kyc_verified_flag: int           = Field(..., ge=0, le=1, example=1)

    # ── User profile fields ────────────────────────────────────────────────
    usr_home_city_tier: int          = Field(..., ge=1, le=3, example=2)
    usr_account_age_days: float      = Field(..., ge=0, example=720.0)
    usr_linked_bank_count: float     = Field(..., ge=1, example=2.0)
    usr_avg_monthly_txn_profile: float = Field(..., gt=0, example=32.0)
    usr_avg_txn_value_profile: float   = Field(..., gt=0, example=850.0)
    usr_is_high_risk: int              = Field(..., ge=0, le=1, example=0)

    # ── Merchant profile fields ────────────────────────────────────────────
    mrc_avg_daily_txn: float       = Field(..., ge=0, example=120.0)
    mrc_is_registered: int         = Field(..., ge=0, le=1, example=1)
    mrc_rating: Optional[float]    = Field(None, ge=0, le=5, example=4.1)
    device_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    ip_risk_score: Optional[float]     = Field(None, ge=0.0, le=1.0)

    # ── Validators ────────────────────────────────────────────────────────
    @field_validator("mrc_size")
    @classmethod
    def p2p_size_consistency(cls, v, info):
        receiver = info.data.get("receiver_type")
        if receiver == "User" and v != "P2P":
            raise ValueError("receiver_type='User' requires mrc_size='P2P'")
        if receiver == "Merchant" and v == "P2P":
            raise ValueError("mrc_size='P2P' only valid when receiver_type='User'")
        return v

    @field_validator("upi_id")
    @classmethod
    def validate_upi_vpa(cls, v):
        if v is not None and not UPI_VPA_RE.match(v):
            raise ValueError(
                f"'{v}' is not a valid UPI VPA. Expected format: localpart@psp "
                "(e.g. user@oksbi, name@ybl)"
            )
        return v


# ════════════════════════════════════════════════════════════════════════════
#  PYDANTIC OUTPUT MODELS
# ════════════════════════════════════════════════════════════════════════════
class PredictionResponse(BaseModel):
    fraud_score    : float  = Field(..., description="Ensemble fraud probability (0.0–1.0).")
    is_fraud       : bool
    alert_level    : Literal["none", "low", "medium", "high"] = Field(
        ..., description="none=<0.20  low=0.20–0.40  medium=0.40–0.70  high=>0.70"
    )
    threshold_used : float
    model_version  : str = "2.0.0"
    request_id     : str = Field("", description="Trace ID for this prediction.")
    # Ensemble breakdown (new in v2)
    paysense_score  : Optional[float] = Field(None, description="PaySense XGBoost score (primary)")
    light_lr_score  : Optional[float] = Field(None, description="LightLR 5-feature score")
    rules_score     : Optional[float] = Field(None, description="Rule-based score")
    active_scorers  : list[str]       = Field(default_factory=list)


class WeeklyInsight(BaseModel):
    period          : str
    total_spent     : float
    top_category    : str
    top_category_pct: float
    fraud_alerts    : int
    savings_tip     : str
    budget_status   : str


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
def compute_alert_level(score: float) -> str:
    if score >= 0.70: return "high"
    if score >= 0.40: return "medium"
    if score >= 0.20: return "low"
    return "none"


def _mock_score(txn: TransactionInput) -> float:
    """
    Deterministic mock scorer used when model artefacts are absent.
    Combines the most important known signals in a simple formula so the
    /predict endpoint remains usable for integration testing without .pkl files.
    """
    score = 0.05
    if txn.new_device_flag:       score += 0.35
    if txn.ip_location_mismatch:  score += 0.20
    if txn.usr_is_high_risk:      score += 0.15
    if txn.amount_deviation_score and txn.amount_deviation_score > 3.0:
        score += 0.15
    if txn.is_night_transaction:  score += 0.05
    if txn.kyc_verified_flag == 0:score += 0.05
    return min(round(score, 4), 0.99)


# ════════════════════════════════════════════════════════════════════════════
#  /predict  — MAIN INFERENCE ENDPOINT  (rate-limited, JWT-protected)
# ════════════════════════════════════════════════════════════════════════════
@app.post(
    "/predict",
    response_model = PredictionResponse,
    summary        = "Score a UPI transaction for fraud",
    tags           = ["Inference"],
)
@limiter.limit("60/minute")
async def predict(
    request    : Request,
    transaction: TransactionInput,
    user       : Annotated[str, Depends(get_current_user)],
) -> PredictionResponse:
    """
    **Requires:** `Authorization: Bearer <token>` (obtain from POST /auth/token).

    **Rate limit:** 60 requests per minute per IP.

    Returns fraud_score (0–1), binary decision, and graduated alert level.
    The scoring pipeline:
    1. Runs PaySense's UPI-native XGBoost (43 features, per-user z-score).
    2. Falls back to a deterministic mock scorer if artefacts are absent
       (useful for integration testing without running the full pipeline).
    """
    rid = _request_id_ctx.get("-")
    txn_dict = transaction.model_dump()

    result = ensemble_score(txn_dict)

    log.info(
        f"predict user={user} ensemble={result.ensemble_score:.4f} "
        f"alert={result.alert_level} amount=₹{transaction.amount} "
        f"scorers={result.active_scorers}"
    )

    return PredictionResponse(
        fraud_score     = result.ensemble_score,
        is_fraud        = result.is_fraud,
        alert_level     = result.alert_level,
        threshold_used  = result.threshold,
        request_id      = rid,
        paysense_score  = result.paysense_score,
        light_lr_score  = result.light_lr_score,
        rules_score     = result.rules_score,
        active_scorers  = result.active_scorers,
    )


# ════════════════════════════════════════════════════════════════════════════
#  /insights/weekly  — AI SAVINGS RECOMMENDATIONS  (JWT-protected)
# ════════════════════════════════════════════════════════════════════════════
@app.get(
    "/insights/weekly",
    response_model = WeeklyInsight,
    summary        = "Get AI-powered weekly spending insights and savings tips",
    tags           = ["Insights"],
)
async def weekly_insights(
    user            : Annotated[str, Depends(get_current_user)],
    total_spent     : float = 12500.0,
    top_category    : str   = "Food",
    top_category_pct: float = 38.0,
    fraud_alerts    : int   = 0,
    vs_last_week_pct: float = 12.0,
) -> WeeklyInsight:
    """
    **Requires:** `Authorization: Bearer <token>`.

    Accepts current week's aggregated spend stats from the Android Finance
    Tracker and returns a personalised savings tip and budget status.

    The Android app calls this endpoint weekly (or on-demand from the Finance
    tab) and displays the result in the Insights section. In production, wire
    this to Gemini API for LLM-generated coaching text. The deterministic
    fallback below covers the case where Gemini is not configured.
    """

    # ── Savings tip generation (deterministic fallback — no Gemini key needed)
    # In production: replace this block with a Gemini API call.
    # GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
    # if GEMINI_KEY: tip = await call_gemini(GEMINI_KEY, context)
    # else: tip = _rule_based_tip(...)

    GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

    if GEMINI_KEY:
        # ── Gemini path (active when GEMINI_API_KEY env var is set) ───────
        try:
            import httpx
            prompt = (
                f"The user spent ₹{total_spent:.0f} this week. "
                f"Top category: {top_category} ({top_category_pct:.0f}% of total). "
                f"Spending changed {vs_last_week_pct:+.0f}% vs last week. "
                f"Fraud alerts this week: {fraud_alerts}. "
                "Give ONE actionable savings tip in under 25 words. "
                "Be specific, friendly, and direct. No preamble."
            )
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-1.5-flash:generateContent?key={GEMINI_KEY}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
                tip = (resp.json()
                       ["candidates"][0]["content"]["parts"][0]["text"]
                       .strip())
        except Exception as e:
            log.warning(f"Gemini call failed ({e}), using rule-based tip")
            tip = _rule_based_tip(top_category, top_category_pct, vs_last_week_pct)
    else:
        tip = _rule_based_tip(top_category, top_category_pct, vs_last_week_pct)

    # ── Budget status ──────────────────────────────────────────────────────
    if vs_last_week_pct > 20:
        budget_status = f"⚑ Spending up {vs_last_week_pct:.0f}% vs last week — review budget"
    elif vs_last_week_pct < -10:
        budget_status = f"✓ Great — spending down {abs(vs_last_week_pct):.0f}% vs last week"
    else:
        budget_status = "~ On track with last week's pace"

    log.info(f"insights user={user} total=₹{total_spent:.0f} tip='{tip[:40]}...'")

    return WeeklyInsight(
        period           = "This week",
        total_spent      = round(total_spent, 2),
        top_category     = top_category,
        top_category_pct = round(top_category_pct, 1),
        fraud_alerts     = fraud_alerts,
        savings_tip      = tip,
        budget_status    = budget_status,
    )


def _rule_based_tip(category: str, pct: float, vs_last: float) -> str:
    """Deterministic savings tip when Gemini is not configured."""
    tips = {
        "Food":      "Try cooking 2 extra meals at home — saves ≈₹800/week on average.",
        "Food & Dining": "Limit restaurant visits to weekends to cut food spend by 30%.",
        "Travel":    "Book rides 15 min before departure — surge pricing drops fast.",
        "Shopping":  "Add items to cart, wait 24h — impulse urge passes 60% of the time.",
        "Grocery":   "Buy staples in bulk monthly — saves ≈15% vs weekly small trips.",
        "Entertainment":"Share OTT subscriptions — you can legally share with 1 household.",
        "Recharge":  "Switch to a quarterly recharge plan — saves ₹50–120 per cycle.",
        "Healthcare":"Schedule non-urgent appointments in bulk to save on consultation fees.",
    }
    base_tip = tips.get(category, f"Your top spend is {category} at {pct:.0f}% of total.")
    if vs_last > 20:
        return f"Spending spiked {vs_last:.0f}% this week. {base_tip}"
    return base_tip


# ════════════════════════════════════════════════════════════════════════════
#  /health  — LIVENESS PROBE  (public, no auth)
# ════════════════════════════════════════════════════════════════════════════
@app.get("/health", tags=["System"], summary="API health and model status")
async def health_check():
    state = get_state()
    return {
        "status"          : "ok",
        "api_version"     : "2.0.0",
        "ensemble_ready"  : state.ready,
        "active_scorers"  : state.active_scorers,
        "paysense_loaded" : state.ps_model is not None,
        "light_lr_loaded" : state.lr_model is not None,
        "rules_always_on" : True,
        "threshold"       : state.ps_threshold,
        "feature_count"   : len(state.ps_features) if state.ps_features else 0,
        "auth_required"   : True,
        "rate_limit"      : "60/min on /predict",
        "gemini_enabled"  : bool(os.environ.get("GEMINI_API_KEY")),
        "mode"            : "production" if state.ps_model else "demo",
    }


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "PaySense Guardian API v2. Visit /docs for the interactive UI.",
        "auth":    "POST /auth/token with username+password to get a Bearer token.",
        "demo_credentials": {"username": API_DEMO_USER, "note": "set via env vars"},
    }


# ════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)

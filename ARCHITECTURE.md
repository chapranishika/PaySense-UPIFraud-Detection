# PaySense — Architecture

Every diagram below reflects components that actually exist and were
verified running this session (backend `main.py`, Android client under
`PaySense-Android-Client-New/`) — nothing invented for shape. Where a box
represents something not yet live (e.g. a real push notification channel),
it's explicitly marked as such.

---

## 1. System architecture

```
                    ┌─────────────────────────────────────┐
                    │        Android Client (Kotlin)        │
                    │   PaySense-Android-Client-New/         │
                    │                                       │
                    │  Layer 1 — SmsReceiver (on-device,     │
                    │            no network, 3-gate regex)   │
                    │  Layer 2 — PayeeCacheRepository (Room   │
                    │            SQLite cache + HITL prompt)  │
                    │  Layer 3 — FraudApiService (Retrofit,   │
                    │            JWT attach, all network I/O) │
                    │  UI — Dashboard / Finance / Assistant /  │
                    │       Profile fragments                 │
                    └───────────────────┬───────────────────┘
                                        │ HTTPS + Bearer JWT
                                        ▼
                    ┌─────────────────────────────────────┐
                    │     FastAPI Backend (main.py)          │
                    │     PaySense-ML-Backend/                │
                    │                                       │
                    │  POST /auth/token      — issue JWT      │
                    │  POST /predict         — fraud score     │
                    │  POST /classify        — category (NLP)  │
                    │  GET  /insights/weekly — savings tip      │
                    │  POST /assistant/chat  — LLM assistant    │
                    │  GET  /health          — public probe     │
                    │  GET  /               — web dashboard UI  │
                    │       (static/index.html + app.js —       │
                    │        same backend, second real client)  │
                    └───────┬─────────────────┬─────────────┘
                            │                 │
                            ▼                 ▼
              ┌───────────────────┐  ┌─────────────────────┐
              │  Ensemble scorer    │  │  Gemini API (optional)│
              │  src/fraud_model.py │  │  generateContent      │
              │                     │  │  system_instruction +  │
              │  rules   (w=0.15)   │  │  safety settings +     │
              │  XGBoost (w=0.60)   │  │  jailbreak pre-filter  │
              │  LightLR (w=0.25)   │  │                        │
              │  + category clf.    │  │  Falls back to          │
              │    (TF-IDF+LinearSVC)│  │  deterministic rules    │
              └───────────────────┘  │  when unset/failed      │
                                      └─────────────────────┘
```

Two real clients hit the same backend: the Android app, and a server-served
web dashboard (`GET /` → `static/index.html` + `app.js`) with its own
login, transaction feed, AI assistant tab, and finance tracker — not a
placeholder page, a second full client sharing one API.

**Not present:** no message queue, no separate microservices, no container
orchestration beyond a single `Dockerfile`/`docker-compose.yml` for local
dev, no distributed cache. This is a monolithic FastAPI process plus a
native Android client — accurately represented above, not padded with
components that don't exist.

---

## 2. Layer 1 — SMS parsing pipeline (Android, on-device)

```
Incoming SMS (system broadcast, SMS_RECEIVED_ACTION)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Gate 1 — Sender ID format                    │
│  ^[A-Z]{2}-[A-Z0-9]{4,6}$  (TRAI DLT format)  │
└───────────────┬───────────────────────────────┘
                │ pass
                ▼
┌─────────────────────────────────────────────┐
│  Gate 2 — Transaction keyword                  │
│  debited | credited | upi | rs\. | inr |        │
│  transaction | payment                          │
└───────────────┬───────────────────────────────┘
                │ pass
                ▼
┌─────────────────────────────────────────────┐
│  Gate 3 — Named-group extraction                │
│  amount · payee · txnId · date                  │
│  (indexed group access, not named — API 24 min) │
└───────────────┬───────────────────────────────┘
                │ amount extracted successfully
                ▼
        Layer 2 — category cache lookup
```

A message failing any gate is dropped silently by design — gates 1–2 are a
cheap filter for "is this even a transaction SMS," and gate 3 failing to
extract an amount routes to a quarantine log path (`applyGate3`'s `null`
return), not a crash.

---

## 3. Layer 2 — categorization pipeline

```
Parsed transaction (payee, amount, txnId)
        │
        ▼
┌─────────────────────────────┐
│ Tier 1 — Room cache lookup     │
│ PayeeCacheRepository            │
│ (has this payee been seen        │
│  and categorized before?)        │
└──────────┬──────────────────┘
   cache HIT │           │ cache MISS
             ▼           ▼
    use cached category   ┌─────────────────────────┐
             │            │ Tier 2 — server NLP call   │
             │            │ POST /classify               │
             │            │ TF-IDF + calibrated LinearSVC │
             │            │ (trained on FinText-6K)        │
             │            └──────────┬──────────────────┘
             │              confidence ≥ 0.65 │  < 0.65 or failure
             │                       ▼         ▼
             │              use NLP result   Tier 3 — HITL prompt
             │                       │        (CategoryBottomSheet,
             │                       │         user picks; cached
             │                       │         for next time)
             ▼                       ▼                 ▼
                    Layer 3 — fraud scoring (dispatched immediately
                    with "Uncategorized" if still pending, so fraud
                    checking is never blocked on categorization)
```

---

## 4. Inference flow — `/predict`

```
Android: ParsedTransaction + device/app/behavioral signals
        │  POST /predict, Authorization: Bearer <JWT>
        ▼
┌─────────────────────────────────────────┐
│  FastAPI: get_current_user (JWT verify)    │
│  slowapi: 60/min rate limit                 │
│  Pydantic: TransactionInput validation      │
│  (every field typed + bounded)               │
└───────────────┬───────────────────────────┘
                ▼
┌─────────────────────────────────────────┐
│  src/fraud_model.py — score()               │
│                                             │
│   rules_score   × 0.15  (always on)          │
│ + xgboost_score × 0.60  (40 features,         │
│                   monotone-constrained on     │
│                   3 behavioral features)       │
│ + light_lr_score × 0.25 (5-feature fallback,   │
│                    also usable standalone      │
│                    if XGBoost artefact is      │
│                    absent)                     │
│  = ensemble_score, compared to deployed         │
│    threshold (0.50)                             │
└───────────────┬───────────────────────────┘
                ▼
   PredictionResponse: fraud_score, is_fraud,
   alert_level (none/low/medium/high),
   per-scorer breakdown, weights_used
                │
                ▼
   Android: dashboard risk %, in-app "AT RISK"
   card + badge if alert_level is high
   (no system push notification — see SECURITY.md)
```

A dropped scorer (e.g. XGBoost artefact missing) renormalizes weights
across whichever scorers are actually active — `active_scorers` and
`weights_used` in the response reflect the real per-request composition,
not a hardcoded nominal value. (`/health`'s `nominal_weights` field is the
full-ensemble case only, used for display purposes on the Profile screen.)

---

## 5. Deployment (as of this audit)

```
Developer machine
   │  git push
   ▼
GitHub (chapranishika/PaySense-UPIFraud-Detection)
   │
   ├─→ GitHub Actions CI (.github/workflows/ci.yml)
   │     • backend pytest (211 tests)
   │     • Android unit tests + assembleDebug
   │
   └─→ Render (paysense-api.onrender.com)
         • uvicorn main:app, single web service
         • NOT RESPONDING as of this audit — see
           PROJECT.md §Known Limitations. Zero bytes
           returned after 90s across DNS-resolves-fine /
           TLS-handshakes-fine / request-sent, request
           never answered. Root cause not diagnosed from
           this environment (no Render dashboard access).

Alternative path prepared, not completed:
Vercel — vercel.json + measured bundle-size finding
(502.4MB minimal deps, over the 500MB standard Python
function limit; documented fix is enabling Large Functions,
which needs account-level dashboard access this
environment doesn't have). See DEPLOY_VERCEL.md.
```

The Android app's production build points at the Render URL
(`FraudApiService.kt`'s `BASE_URL`). All Android live-testing this session
used a temporary, always-reverted swap to a local `10.0.2.2:8010` backend
(verified via `git diff` before every commit) — never a claim that the
production URL was working when it wasn't.

---

## 6. What's deliberately NOT here

- **No API gateway / load balancer** — single Render web service, one
  process.
- **No model registry** — model artifacts (`.pkl` files under
  `PaySense-ML-Backend/artefacts/`) are loaded once at FastAPI startup
  (`lifespan` context manager) and cached in a module-level dict
  (`ML: dict = {}` / `_state` in `fraud_model.py`). Retraining means
  running a training script and replacing the `.pkl` file — there is no
  versioned registry, no A/B rollout, no shadow deployment.
- **No message queue** — every request is synchronous request/response.
- **No separate feature store** — features are computed inline per-request
  from the request payload (e.g. `amount_deviation_score` is a client-side
  z-score computation, `DeviationStatsCalculator.kt`, not a server-side
  lookup against historical data).

These are honest gaps for a production financial system, appropriate for
what this actually is: a personal/academic portfolio project with a real,
working ML pipeline behind it — not a claim of enterprise-scale
infrastructure that isn't there.

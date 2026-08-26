# PaySense — Finlatics and Finance Flow

> A three-layer UPI fraud detection system: an Android app parses incoming
> bank SMS on-device, categorizes the transaction, and scores it for
> fraud risk via a FastAPI + XGBoost-ensemble backend, personalized to
> each user's own spending history. A full web dashboard shares the same
> backend API.

**Author:** Nishika Chapra | KJ Somaiya Institute of Technology | 2025

**Full documentation:** [`WALKTHROUGH.md`](WALKTHROUGH.md) (live screenshots
and results) · [`ARCHITECTURE.md`](ARCHITECTURE.md) ·
[`DATASET.md`](DATASET.md) · [`EXPERIMENTS.md`](EXPERIMENTS.md) ·
[`SECURITY.md`](SECURITY.md) · [`PROJECT.md`](PROJECT.md) (deep technical
narrative)

---

## 1. Problem

UPI fraud detection today runs almost entirely server-side, invisible to
the user until money has already moved. This project parses the
confirmation SMS on-device at the moment it arrives, and scores it using
signals personalized to that user's own spending pattern rather than
population-level rules alone.

## 2. System overview

Two real clients share one backend: the Android app, and a server-served
web dashboard (`GET /` → `static/index.html`). Both call the same
`/predict`, `/classify`, `/insights/weekly`, and `/assistant/chat`
endpoints behind real JWT authentication.

## 3. Architecture

```
Bank SMS
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 1 — SMS Engine  (Android, on-device, no internet)      │
│  Gate 1: TRAI Sender ID regex   Gate 2: keyword match          │
│  Gate 3: named-group extraction (amount/payee/txnId/date)      │
└──────────────────────────────┬────────────────────────────────┘
                                │  ParsedTransaction
              ┌─────────────────┴─────────────────┐
              ▼                                    ▼
┌──────────────────────────┐      ┌────────────────────────────────┐
│  LAYER 2 — Room SQLite    │      │  LAYER 3 — FastAPI + ensemble   │
│  Tier 1: cache lookup     │      │  Rules (0.15) + XGBoost (0.60)  │
│  Tier 2: NLP classifier   │      │  + LightLR (0.25) → threshold   │
│  Tier 3: HITL prompt      │      │  → alert level → response       │
└──────────────────────────┘      └────────────────────────────────┘
```

Full diagram set (inference flow, deployment, categorization pipeline)
in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 4. Dataset

30,000-row fraud-scoring dataset blended from two sources: a 20,000-row
"anchor" set (real transaction-style data) and a 10,000-row "supplement"
set (schema-bridged from an external synthetic dataset). **The supplement
source is severely structurally contaminated and is not representative
of the organic deployment distribution** — see §6 and
[`DATASET.md`](DATASET.md) for the full finding. A separate FinText-6K
dataset (6,000 rows) trains the category classifier. Full detail,
schema, and known limitations in [`DATASET.md`](DATASET.md).

## 5. ML methodology

A 3-scorer ensemble (`src/fraud_model.py`): an always-on rules scorer,
an XGBoost model (40 features, monotonic constraints on three behavioral
features), and a lightweight logistic-regression fallback. PR-AUC is the
primary evaluation metric (not accuracy — see §7). Threshold selection
follows a train→validation→test protocol on organic data, described in
full below.

## 6. Evaluation

**A forensic investigation (2026-08-27,
[`SOURCE_CONTAMINATION_INVESTIGATION.md`](PaySense-ML-Backend/SOURCE_CONTAMINATION_INVESTIGATION.md),
[`EXPERIMENTS.md`](EXPERIMENTS.md)) found that the canonical test set used
for most of this project's history is contaminated by the same source
issue as the training data** — 23 of ~30 numeric columns and 12 of 14
categorical columns in the "supplement" source are a single constant
value across all 10,000 rows (including a literal synthetic marker,
`receiver_id == "SYN_MRC_UNKNOWN"`), and the split doesn't account for
source, so the test set inherits the contamination proportionally.

Tested directly (not assumed): retraining on organic-only data did **not**
materially change organic-subset ROC-AUC (0.7260→0.7261). A proper
train(60%)/validation(20%)/test(20%) split was then run entirely on
organic data, with the threshold selected on validation only and the
final test set touched exactly once. That result — not the older blended
metrics — is this project's primary reported result.

## 7. Results

**Primary result — clean organic evaluation:**

| Metric | Value |
|---|---:|
| Test ROC-AUC | 0.7050 |
| Test PR-AUC | 0.0945 |
| Test precision @ frozen threshold (τ=0.10, selected on validation) | 8.82% |
| Test recall @ frozen threshold | 21.05% (32/152) |
| Documented requirement (Recall≥75%, Precision≥50%) | Not met by the current model/data at this measurement |

**Historical result (blended, source-contaminated test set — superseded
above, kept for record):**

| Metric | Value |
|---|---:|
| ROC-AUC | **0.8969** |
| PR-AUC | **0.5498** |
| Precision @ deployed threshold (t=0.50) | **91.74%** |
| Recall @ deployed threshold (t=0.50) | **39.53%** |

**Other verified results:** 243 tests passing (215 backend pytest + 28
Android unit tests). Category classifier: 78.0% real-world accuracy
(deployed), 83.0% (DistilBERT candidate, validated but not deployed —
see [`EXPERIMENTS.md`](EXPERIMENTS.md)). Android security: 4 findings, 4
fixed.

Accuracy is not reported as a headline metric here on purpose — a model
that predicts "legitimate" for every transaction scores 95.79% accuracy
on this dataset and is useless; PR-AUC and recall/precision at a
validated threshold are the metrics that matter for a 4%-base-rate
problem.

## 8. Security

JWT authentication, rate limiting, Pydantic request validation
throughout. LLM-backed endpoints (`/assistant/chat`, `/insights/weekly`)
use a real `system_instruction` channel plus a regex prompt-injection
pre-filter, with a deterministic fallback if no key is configured or a
call fails. A `pip-audit` scan (2026-08-26) found and fixed CVEs in 4
dependencies; full scan result, what remains, and why in
[`SECURITY.md`](SECURITY.md). No secret has ever been committed to this
repository.

## 9. Testing

```bash
# Backend (215 tests)
cd PaySense-ML-Backend && pytest tests/ -v

# Android (28 tests)
cd PaySense-Android-Client-New && ./gradlew test
```

Both suites run in CI on every push (`.github/workflows/ci.yml`).
Regression tests exist for every real bug this project found and fixed —
including three protecting the source-contamination finding itself, so
it can't silently drift unnoticed.

## 10. Setup

**Backend:**
```bash
cd PaySense-ML-Backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # fill in JWT_SECRET_KEY, API_DEMO_USER, API_DEMO_PASS
uvicorn main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health`. Swagger UI at `/docs`.

**Android:** open `PaySense-Android-Client-New/` in Android Studio
(SDK Platform 34+, JDK 17), let Gradle sync, then Run. The app points at
`https://paysense-api.onrender.com/` by default (`FraudApiService.kt`'s
`BASE_URL`) — see §11 for current deployment status.

## 11. Limitations

- **Organic fraud detection is weak.** Under the clean evaluation
  protocol above, the model catches 21.05% of organic fraud at 8.82%
  precision — real signal above chance, but well short of a production
  bar. Retraining on the existing (cleaned) dataset did not improve this;
  closing the gap needs different or additional organic training data,
  which has not been sourced as part of this work.
- **The documented Recall≥75%/Precision≥50% business requirement is not
  met** by the current model/data under any evaluation protocol tested.
- **Production deployment status is currently unverified from this
  environment** — the Render URL the app points at has not responded in
  recent checks; no dashboard access exists here to diagnose further.
- **A 2026-08-24 finding, corrected in the ensemble scoring code:** every
  metric this project reported before that date used raw XGBoost
  `predict_proba()` instead of the real 3-scorer ensemble `/predict`
  actually serves. Fixed, regression-tested, documented in
  [`EXPERIMENTS.md`](EXPERIMENTS.md).
- Full, unabridged limitations list in [`PROJECT.md`](PROJECT.md) §22 and
  [`WALKTHROUGH.md`](WALKTHROUGH.md)'s "Honest findings."

---

*PaySense — Nishika Chapra · KJ Somaiya Institute of Technology · 2025*

# PaySense — Finlatics and Finance Flow

> A three-layer zero-trust fraud detection system for Indian UPI payments, built as an Android application with a FastAPI + XGBoost ML backend. Every incoming bank SMS is parsed, categorised, and scored for fraud risk in real time — personalised to each individual user's spending habits.

**Author:** Nishika Chapra | KJ Somaiya Institute of Technology | 2025

---

## Key Results

| Metric | Value |
|---|---|
| ROC-AUC *(real 3-scorer ensemble — see note)* | **0.8969** |
| PR-AUC *(primary, real ensemble)* | **0.5498** — 13.05× above random baseline |
| Precision @ deployed threshold (t=0.50) | **91.74%** |
| Recall @ deployed threshold (t=0.50) | **39.53%** |
| Datasets evaluated | 18 |
| Master dataset | 30,000 rows · 40 model-ready features · 4.21% fraud |
| SMOTE applied | Training partition only (24K → 45,980 rows) |
| Tests passing | **165** (28 Android unit tests · 137 backend pytest) |

**Correction, 2026-08-24 — the most important one yet, because it's a
methodology error, not a stale number.** Every metric ever reported in this
table, tonight and before, was computed by calling
`model.predict_proba()` directly on the frozen XGBoost artifact. But
`/predict` never does that — it calls `src.fraud_model.score()`, which
blends XGBoost with two more scorers (LightLR, weight 0.25; a hand-tuned
rules scorer, weight 0.15) that raw `predict_proba()` completely bypasses.
Scored through the real ensemble, the canonical held-out test set behaves
substantially differently: at the threshold that was deployed until today
(t=0.30), the real ensemble's precision was **40.81%**, not the 86.44%
this table claimed — the rules scorer's always-on additive score was
never jointly calibrated against that threshold the way XGBoost's own
sweep was, so plenty of rows XGBoost alone would score below 0.30 still
crossed it once blended in. `resweep_threshold_against_ensemble.py`
re-ran the exact same threshold-selection logic
(`paysense_phase3.py`'s Recall≥75%/Precision≥50% business constraint,
fallback to max-F1) against the real ensemble instead, swept 0.05-0.95 to
confirm the optimum wasn't sitting at a range boundary (it isn't: F1
peaks exactly at t=0.50, dips at 0.55, then plateaus lower from
0.65-0.90 as precision saturates at 100%). **New deployed threshold:
t=0.50** (was 0.30) — `artefacts/paysense_threshold.pkl` updated
accordingly, verified live against a restarted server. Every number in
the table above is now the real ensemble's, not raw XGBoost's, and
`tests/test_frozen_model_metrics.py` now pins (and guards) the ensemble
figures, plus a new regression test asserting raw XGBoost and the real
ensemble stay materially different — so this exact class of "silently
measuring the wrong component" mistake can't recur unnoticed.

**Prior update, 2026-08-23 (still accurate, superseded only in threshold
value by the correction above):** the deployed model trains with
`monotone_constraints` on three behavioral features
(`amount_deviation_score`, `transaction_velocity`,
`failed_attempts_last_24h`) — see `RECALL_CEILING_REMEDIATION.md` for the
diagnosis (XGBoost's trees were gating hard on `new_device_flag`/
`ip_location_mismatch` first) and the comparison that justified adopting
this specific fix over two other candidates: it recovered 10 of 76
previously-invisible fraud rows while *improving* both raw-XGBoost
ROC-AUC and PR-AUC, the only one of three tested variants with no
measurable downside.

---

## Architecture

```
Bank SMS
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 1 — SMS Engine  (Android, on-device, no internet)     │
│                                                              │
│  Gate 1: TRAI Sender ID regex  ^[A-Z]{2}-[A-Z0-9]{4,6}$     │
│  Gate 2: Transaction keyword   debited · credited · UPI      │
│  Gate 3: Named-group regex     amount · payee · txnId · date │
└──────────────────────────────┬───────────────────────────────┘
                               │  ParsedTransaction
              ┌────────────────┴────────────────┐
              ▼                                  ▼
┌─────────────────────────┐      ┌───────────────────────────────┐
│  LAYER 2 — Room SQLite  │      │  LAYER 3 — FastAPI + XGBoost  │
│                         │      │                               │
│  Tier 1: Cache lookup   │      │  1. Query 90-day user stats   │
│  Tier 2: NLP classifier │      │  2. Save transaction to DB    │
│  Tier 3: HITL prompt    │      │  3. Compute z-score deviation │
│  (asked once, forever)  │      │  4. POST /predict → XGBoost   │
└─────────────────────────┘      │  5. Update fraud verdict      │
              │                  └───────────────────────────────┘
              └──────────────────────────┐
                                         ▼
                              ┌──────────────────────┐
                              │  Dashboard UI         │
                              │  White card = safe    │
                              │  Red card  = fraud ⚠  │
                              └──────────────────────┘
```

---

## Repository Structure

```
paysense/
│
├── README.md
├── .gitignore                        ← covers Python + Android + IDE
│
├── PaySense-ML-Backend/              ← Python FastAPI server + ML pipeline
│   ├── main.py                       ← FastAPI: /predict, /health endpoints
│   ├── paysense_pipeline.py          ← Phase 1: dataset merge (30K rows)
│   ├── paysense_ml_pipeline.py       ← Phase 2: XGBoost + SMOTE + SHAP
│   ├── paysense_phase3.py            ← Phase 3: threshold tuning + model freeze
│   ├── requirements.txt
│   └── plots/                        ← evaluation charts (committed, no data)
│       ├── paysense_shap_bar.png
│       ├── paysense_shap_beeswarm.png
│       ├── paysense_threshold_analysis.png
│       ├── paysense_evaluation_report.png
│       ├── paysense_class_imbalance.png
│       └── paysense_feature_engineering.png
│
├── PaySense-Android-Client-New/      ← Kotlin Android app (active — builds & tests green)
│   └── app/src/
│       ├── main/
│       │   ├── AndroidManifest.xml
│       │   ├── res/{layout,values,drawable,color,font}/
│       │   └── kotlin/com/paysense/app/
│       │       ├── layer1/SmsReceiver.kt
│       │       ├── layer2/
│       │       │   ├── PayeeCache.kt · PayeeDao.kt · PayeeCacheRepository.kt
│       │       │   ├── PaySenseDatabase.kt · TransactionHistory.kt
│       │       │   └── Budget.kt · SavingsGoal.kt · PdfReportGenerator.kt · FinanceExportUtil.kt
│       │       ├── layer3/
│       │       │   ├── ApiModels.kt · PaySenseApi.kt · FraudApiService.kt
│       │       └── ui/
│       │           ├── MainActivity.kt · TransactionAdapter.kt
│       │           ├── DashboardContentFragment.kt · FinanceFragment.kt · AssistantFragment.kt · ProfileFragment.kt
│       │           └── AddTransactionDialog.kt · CategoryBottomSheet.kt · BudgetBottomSheet.kt · GoalBottomSheet.kt · DonutChartView.kt
│       └── test/kotlin/com/paysense/app/     ← 28 unit tests (layer1/2/3)
│
│  (PaySense-Android-Client/ is an early, incomplete scaffold — superseded by -New, kept for history)
│
├── screenshots/                      ← 10 real device screenshots of the app
│
└── PaySense-Report/                  ← IEEE LaTeX report (Overleaf ready)
    ├── paysense_report.tex
    └── [6 PNG figures for LaTeX]
```

---

## SHAP Feature Importance

![SHAP Bar](PaySense-ML-Backend/plots/paysense_shap_bar.png)

`new_device_flag` is the dominant fraud signal (mean |SHAP| = 1.17). The personalised `amount_deviation_score` ranks 9th — confirming that raw amount is a weak signal; it only becomes meaningful as a per-user z-score.

![SHAP Beeswarm](PaySense-ML-Backend/plots/paysense_shap_beeswarm.png)

---

## Threshold Analysis

![Threshold Analysis](PaySense-ML-Backend/plots/paysense_threshold_analysis.png)

Recall ceiling: **71.94%** at threshold=0.05 (182 of 253 fraud rows caught, 71 missed) — independently recomputed 2026-08-23 against the current on-disk model. (This specific figure describes raw XGBoost's own ranking behavior in isolation, the same scope `RECALL_CEILING_REMEDIATION.md` and `PLATT_SCALING_RESULT.md` used — not the full 3-scorer ensemble `/predict` actually runs; see the Key Results note above on the 2026-08-24 ensemble-vs-raw correction for that distinction.) This is an improvement over the **69.96%** ceiling of the prior (pre-monotonic) frozen model: the `monotone_constraints` update described in the Key Results note above recovers 10 of the 76 fraud rows that model could never reach at any threshold, while the remaining 45 (59% of the original 76) are still unreachable even under this fix — see `RECALL_CEILING_REMEDIATION.md` §6 for the full honest breakdown of which part of the ceiling is a fixable structural artifact versus a harder data limitation. `PaySense-ML-Backend/PLATT_SCALING_RESULT.md` implements and tests the fix this project used to propose — Platt Scaling — against the pre-monotonic baseline model, and finds it does **not** move that model's ceiling: ROC-AUC, PR-AUC, and recall at every swept threshold are identical before and after calibration (to floating-point precision), because a monotonic 1-D rescaling of scores cannot change which rows a classifier ranks lowest. That finding is a mathematical property of monotonic transforms, not specific to the model tested, so it applies identically to today's monotonic-constraints model — the residual ceiling above is a **ranking/discrimination** limit, not a probability-scale artifact, and closing the rest of it needs better features or a different model, not recalibration. Platt scaling's actual, separate benefit — probability *reliability* — was also mixed on the prior model: on a held-out slice, raw XGBoost's Brier score was consistently as good or better than the Platt-scaled version across 6 resampled calibration draws; that experiment was not repeated against the new model (see `PLATT_SCALING_RESULT.md`'s own update note).

---

## Local Run — Backend

```bash
cd PaySense-ML-Backend

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Step 1: Build the 30,000-row master dataset (~2 min)
python paysense_pipeline.py

# Step 2: Train XGBoost + generate all SHAP/evaluation plots (~5 min)
python paysense_ml_pipeline.py

# Step 3: Threshold sweep + freeze model to .pkl files
python paysense_phase3.py

# Step 4: Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify:**
```bash
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":true,"threshold":0.5,"feature_count":40}
```

**Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

**Test fraud prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "receiver_type":"Merchant","transaction_type":"P2M",
    "payment_app":"GPay","device_type":"iOS",
    "usr_age_group":"35-44","usr_preferred_app":"GPay",
    "usr_preferred_device":"Android","mrc_category":"Grocery",
    "mrc_size":"Medium","amount":15000.00,"hour_of_day":2,
    "day_of_week":0,"is_weekend":0,"is_night_transaction":1,
    "time_since_last_txn_min":3.0,"transaction_velocity":0.95,
    "amount_deviation_score":4.2,"failed_attempts_last_24h":3.0,
    "recurring_payment_flag":0,"transaction_frequency_score":0.88,
    "new_device_flag":1,"ip_location_mismatch":1,
    "user_city_tier":1,"user_avg_monthly_txn":12.0,
    "user_avg_txn_value":200.0,"user_loyalty_score":0.11,
    "balance_after_transaction":980.0,"txn_success_flag":1,
    "kyc_verified_flag":0,"usr_home_city_tier":1,
    "usr_account_age_days":14.0,"usr_linked_bank_count":1.0,
    "usr_avg_monthly_txn_profile":12.0,"usr_avg_txn_value_profile":200.0,
    "usr_is_high_risk":1,"mrc_avg_daily_txn":120.0,
    "mrc_is_registered":1,"mrc_rating":2.1,
    "device_risk_score":0.91,"ip_risk_score":0.87
  }'
# → {"fraud_score":0.98,"is_fraud":true,"alert_level":"high"}
```

---

## Local Run — Android App

### Prerequisites
- Android Studio (Ladybug or newer) with SDK Platform 34 + Build-Tools 36
- JDK 17 (Android Studio's bundled JBR works)

### Setup

1. Open Android Studio → **File → Open** → select `PaySense-Android-Client-New/`. All dependencies (Room 2.8.4 via KSP, Retrofit, OkHttp, Coroutines, Material 1.12) are already declared in `app/build.gradle.kts` — no manual edits needed, just let Gradle sync.

2. By default the app points at the live deployed backend:
```kotlin
// FraudApiService.kt
private const val BASE_URL = "https://paysense-api.onrender.com/"
```
   To hit a locally-running backend instead (`uvicorn main:app` from the steps above), change this to `http://10.0.2.2:8000/` (the emulator's alias for your laptop's localhost) and rebuild.

3. Click **Run ▶**, or from the command line:
```bash
cd PaySense-Android-Client-New
./gradlew assembleDebug   # build the APK
./gradlew test            # run the 28 Layer 1/2/3 unit tests
```

### Test SMS via Emulator

In Android Studio emulator → **⋮** → **Phone** → **SMS**

**Safe transaction:**
- Sender: `AD-HDFCBK`
- Message: `Your a/c XX1234 debited Rs.500.00 on 26-Apr-25 to Zomato India UPI Ref: 512345678901. Bal: Rs.12,000.`

Expected Logcat (filter: `PaySense`):
```
PaySense_Layer1: 🟢 GATE 1 PASS | AD-HDFCBK
PaySense_Layer1: 🟢 GATE 2 PASS | 'debited'
PaySense_Layer1: 🔵 GATE 3 EXTRACT | amount=500.0 | payee=Zomato India
PaySense_Layer2: ⚡ Keyword: 'zomato' → Food (0.99)
PaySense_Layer3: ✅ SAFE | score=0.09 | alert=none
```

**Suspicious transaction:**
- Sender: `AD-ICICIB`
- Message: `ALERT: Rs.15000.00 debited to unknown@upi UPI Ref: 987654321098. Time: 02:14 AM.`

Expected:
```
PaySense_Layer3: 🚨 HIGH ALERT | score=0.98 | alert=high
```
UI: Red card with ⚠ icon and "Score: 98%"

---

## Testing

165 tests, all passing. Also wired into CI (`.github/workflows/ci.yml`) — runs both suites on every push/PR to `main`.

| Suite | Tests | What's covered |
|---|---|---|
| `PaySense-ML-Backend/tests/test_api.py` | 45 | `/predict`, `/health`, `/classify` — auth, request validation, VPA format, P2P consistency, cold start, alert-level consistency, one correctness check per category class |
| `PaySense-ML-Backend/tests/test_pipeline_logic.py` | 32 | SMOTE applied post-split/train-only, alert-level threshold boundaries, `DROP_COLS` schema, frozen preprocessor/feature-count artefacts |
| `PaySense-Android-Client-New/app/src/test/kotlin/.../layer1/SmsReceiverTest.kt` | 12 | Gate 1 (TRAI sender regex), Gate 2 (transaction keywords), Gate 3 (named-group extraction + quarantine on unparseable amount) |
| `.../layer2/NlpKeywordRulesTest.kt` | 9 | Merchant→category keyword table, confidence floor, unknown-payee fallthrough to HITL |
| `.../layer3/DeviationStatsCalculatorTest.kt` | 7 | Cold-start neutral z-score, personalised amount/hour z-scores, stddev clamping (÷0 guard) |

Run them:
```bash
# Backend
cd PaySense-ML-Backend && pytest tests/ -v

# Android
cd PaySense-Android-Client-New && ./gradlew test
```

Two real bugs the suite (and the process of writing it) caught, both now fixed rather than just flagged:
- **Auth-bypass-by-default**: `.env` shipped `APP_ENV=development`, under which `/predict`/`/classify` skipped JWT checks entirely when no `Authorization` header was sent — which had also been silently passing three auth tests for the wrong reason. `.env.example`'s default is now `production`, `conftest.py` pins `production` for the test run, and (see **Authentication** below) the client no longer depends on the bypass anyway.
- **Dead code with a live twin**: `main.py` had its own `compute_alert_level()`, byte-for-byte duplicate logic of the one actually used by `/predict` (`EnsembleResult.alert_level` in `src/fraud_model.py`), never called anywhere. Deleted, along with the 20 tests that existed only to guard it against drifting from its live counterpart.

The Android z-score math (`computeDeviationStats` in `FraudApiService.kt`) and the Layer 2 keyword classifier (`runNlpClassifier` in `PayeeCacheRepository.kt`) both needed an Android `Context`/Room DB to construct — rather than pull in Robolectric, the pure calculation logic was extracted into standalone `internal object DeviationStatsCalculator` and `internal object NlpKeywordRules`, so it's directly unit-testable with plain JUnit.

---

## Authentication

The login screen used to be theater: a local, hardcoded `username == "paysense" && password == "guardian2025"` string comparison in `MainActivity.kt` that never talked to the network, while the backend's real `POST /auth/token` endpoint (and its JWT protection on `/predict`, `/classify`, `/insights/weekly`) sat completely disconnected from it — the client worked at all only because the server's dev-mode bypass let every request through unauthenticated.

That's fixed. The login screen now calls `/auth/token` for real; on success the JWT is persisted and an `OkHttp` `Interceptor` attaches `Authorization: Bearer <token>` to every subsequent request automatically; a 401 (expired/invalid token) clears the stored auth state so the next launch asks you to log in again. The hardcoded credential string is gone from the Kotlin source — the one real check is server-side, in `main.py`, via `API_DEMO_USER`/`API_DEMO_PASS`.

Verified against a locally-running instance of the backend, not just asserted:
```bash
# No token — now genuinely rejected (previously would have passed under the dev bypass):
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
# → 401 {"detail":"Not authenticated"}

# Real token, obtained the same way the app does:
curl -X POST http://localhost:8000/auth/token -H "Content-Type: application/json" \
  -d '{"username":"paysense","password":"guardian2025"}'
# → 200 {"access_token":"...", "token_type":"bearer", "expires_in":3600}
```
And on-device: the login flow was run on a real emulator against this same local backend, both with the correct demo credentials (JWT issued, dashboard loads, a subsequent authenticated call to `/insights/weekly` succeeds with the token attached) and with wrong ones (401, visible "Invalid username or password" error) — confirmed via `adb logcat` and the backend's own request log, not just a compiling build.

---

## Alert Level Logic

| Fraud Score | Alert | Android UI Action |
|---|---|---|
| ≥ 0.70 | `high` | Block payment + push notification |
| 0.40–0.70 | `medium` | Require PIN re-entry |
| 0.20–0.40 | `low` | Silent internal flag |
| < 0.20 | `none` | Log as safe, no user interruption |

---

## Personalised Anomaly Scoring

The core innovation. Instead of flagging a ₹15,000 transaction as suspicious for everyone, PaySense computes how unusual it is **for that specific user**:

```
z_amount = (transaction_amount - user_90day_mean) / user_90day_stddev

Example:
  User always spends ~₹500 (mean=500, std=80)
  New transaction: ₹15,000
  z = (15000 - 500) / 80 = 181.25  ← extreme anomaly

  Same ₹15,000 for a developer who regularly buys cloud servers:
  mean=12000, std=4000
  z = (15000 - 12000) / 4000 = 0.75  ← completely normal
```

**Critical rule:** Statistics are computed **before** saving the current transaction, so the transaction never inflates its own baseline.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Android | Kotlin, Material 3, Room, Retrofit, Coroutines |
| API | FastAPI, Pydantic v2, Uvicorn |
| ML | XGBoost, Scikit-learn, SMOTE, SHAP |
| Data | Pandas, NumPy |

---

## Report

The `PaySense-Report/` folder contains a complete **10-page IEEE-format LaTeX report** ready for Overleaf. Upload `paysense_report.tex` and all 6 PNG files to a new Overleaf project, set compiler to pdfLaTeX, and recompile.

---

## Layer 2 NLP Classifier

The architecture diagram above says "Tier 2: NLP classifier" — until now, that tier didn't actually exist. `runNlpClassifier()` in `PayeeCacheRepository.kt` was a keyword-matching stub, even though the paper's bibliography already cited `FinText-6K` as the dataset "used to train the Layer 2 NLP classifier." That dataset had never been touched by any training code. It now has been: a TF-IDF + linear SVM classifier trained on FinText-6K's 5,000-row split, served via a new `POST /classify` endpoint, and wired into the Android client's Tier 2 (gated by the same 0.65 confidence threshold as before — below it, falls through to the human prompt exactly as it did previously).

**Honest caveat, stated up front — and since actually measured, not just hedged:** the held-out test set scores 100% accuracy because FinText-6K's 5,000 rows are generated from only 40 fixed sentence templates with the amount/reference number swapped, and the test split draws from those same 40 templates. The real question — does it hold up on narration it's never seen the *shape* of — has been tested directly: scored against 200 hand-authored, structurally-verified-novel narrations in real HDFC/SBI/ICICI/Axis SMS and GPay/PhonePe/Paytm formats, the originally-deployed classifier scored **72.5%** accuracy, and only **62.0%** of predictions both classified correctly and cleared the app's 0.65 confidence gate. The cause was traced directly, not guessed: the fitted vocabulary (821 tokens) was entirely mined from the 40 templates, so novel text sharing zero content words with training collapsed to an identical default prediction regardless of the true label (near-random, 30.0% accuracy), while text that happened to reuse one training word scored near the training-time ceiling (86.7%) — but often via single-keyword lookup, not real narration understanding (e.g. "Amazon Pay" as a payment rail got read as the Amazon marketplace and misclassified a coffee-shop payment as Shopping at 96% confidence).

**Deployed 2026-08-24:** retrained on FinText-6K blended with 8,000 rows of hand-built templates covering far more merchants, banks, and sentence structures (`CATEGORY_CLASSIFIER_V3_ATTEMPT.md`) — verified programmatically disjoint from the 200-row novel eval set before training (a prior attempt, v2, was discarded after its templates turned out to reuse the eval set's own sentence skeletons). This model, not the original, is now `artefacts/paysense_category_classifier.pkl`: accuracy on the novel eval set rises to **78.0%**, and correct-and-confident (clears the 0.65 gate) rises to **70.5%**. Still a real gap, not a solved problem — Food/Travel/Shopping still leak into each other on ambiguous vocabulary, and EMI's still the only class that generalizes close to perfectly, because "EMI" is a rare, unambiguous token the other four classes don't have an equivalent of. Full metrics, the template-extraction method, the novel test set itself, and every misclassification are in `PaySense-ML-Backend/CATEGORY_CLASSIFIER_GENERALIZATION.md` and `CATEGORY_CLASSIFIER_V3_ATTEMPT.md` (original training metrics in `CATEGORY_CLASSIFIER.md`).

---

## Generalization Check

The 0.8969 ROC-AUC above is measured on a held-out split of the model's *own* training pipeline — it proves the model didn't memorize its own test rows, not that it works on data it's never seen. `PaySense-ML-Backend/GENERALIZATION_CHECK.md` closes that gap: the frozen model, unmodified, was scored against real UPI/fraud datasets it was never trained on, using the same dataset-vetting rigor as the Trojan Family discovery (one 100K-row candidate with a suspicious flat 20.00% fraud rate and near-deterministic feature→label correlations was rejected outright, the same way the pre-balanced Trojan file was).

On the one dataset that passed vetting (74,917 real rows, 0.94% fraud, only 6 of 40 features honestly mappable to PaySense's schema — the rest imputed), scored through the real 3-scorer ensemble (not raw XGBoost alone — see the Key Results note above on why that distinction matters):

| Metric | Value |
|---|---|
| ROC-AUC | 0.7919 — real ranking signal, using ~15% of the feature vector (raw XGBoost alone: 0.7687) |
| Recall @ deployed threshold (0.50) | **0 / 701** — the model's max output on this dataset was 0.0847, far below its own decision threshold |

**Honest read:** the model learned *something* transferable — it isn't pure memorization — but it is not operationally useful on any transaction stream that can't supply its full 40-feature, personalization-heavy vector (per-user z-scores, device/IP risk, KYC flags), and no such real-world dataset appears to exist outside this project's own synthetic pipeline. That's a real generalization gap, reported rather than hidden. **Recomputed 2026-08-24** against the current model and the corrected deployed threshold — the ensemble ROC-AUC is 0.7919 and the zero-recall finding is unchanged regardless of which threshold (0.30 or 0.50) is applied, since the model's maximum score on this dataset sits below both. A newer, richer real dataset was also tried (`REAL_DATA_AND_RESEARCH_GROUNDING.md`) and scored at chance — traced to a currency-scale mismatch in the rules scorer specific to that USD-denominated dataset, not further evidence against the model. Full numbers, the ensemble comparison, and every secondary dataset are in `GENERALIZATION_CHECK.md` and `REAL_DATA_AND_RESEARCH_GROUNDING.md`.

**Does a lower threshold help — tested, not assumed:** `OOD_GENERALIZATION_REMEDIATION.md` §7 followed up directly rather than leaving "zero recall" as the final word. A candidate model with better real-world ranking (Variant A, blended training data, undeployed) was swept across a much lower threshold range on this same dataset — the deployed threshold isn't the only one that could ever apply. At τ=0.06, it catches **37 of 701** real fraud rows with **zero false positives** — the first threshold this project has ever found, on any model, that catches any real fraud here at all. At τ=0.055, that rises to 45.5% recall, but at a real cost (283 legitimate transactions flagged) reported alongside it, not hidden. This is not a deployment recommendation — the threshold was found by sweeping against this dataset's own labels, not a held-out calibration split, and no OOD-aware threshold-switching policy exists in production — but it changes "the model catches literally zero real fraud, always" to a real, narrow, honestly-caveated exception. Full sweep table and the caveat in full in `OOD_GENERALIZATION_REMEDIATION.md` §7.

---

## Honest Limitations

- All training data is **synthetic**, and the generalization check above confirms the gap directly: **0/701** frauds caught on real out-of-distribution data at the production threshold — a live shadow-mode trial (or a dataset that can supply the full 40-feature vector) is required before this could be trusted on real traffic
- **71.94% recall ceiling** for raw XGBoost at any threshold (recomputed 2026-08-23 against the current monotonic-constraints model; was 69.96% before that update; scoped to the raw model, not the full deployed ensemble — see the Key Results note on the 2026-08-24 ensemble-vs-raw correction) — tested and confirmed to be mostly a ranking limitation, not a calibration one (`PLATT_SCALING_RESULT.md`, run against the prior model, but the ranking-invariance finding is model-independent): Platt Scaling leaves ROC-AUC/PR-AUC/recall completely unchanged under any monotonic transform. `RECALL_CEILING_REMEDIATION.md` went further and found the ceiling is partially — not fully — a fixable structural artifact: forcing the model's trees to give the three behavioral features (`amount_deviation_score`, `transaction_velocity`, `failed_attempts_last_24h`) independent weight recovered 10 of the original 76 invisible fraud rows with no measurable cost (now adopted), and a more aggressive variant recovered 31 of 76 but at a real cost to overall ranking quality and false-positive volume (not adopted). The remaining ~45 rows may need genuinely new, more discriminative features rather than a different arrangement of the ones already available
- `new_device_flag` uses a placeholder default in the demo — production needs device fingerprinting APIs

---

*PaySense — Nishika Chapra · KJ Somaiya Institute of Technology · 2025*

Created with love ❤️ by Nishika Chapra

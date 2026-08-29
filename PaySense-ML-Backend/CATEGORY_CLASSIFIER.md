# PaySense Layer 2 — NLP Category Classifier

This document closes a real gap between what `PaySense-Report/paysense_report.tex`
claimed (bibitem `fintext2024`: "This dataset ... used to train the Layer 2 NLP
classifier in PaySense") and what actually existed in the codebase: `runNlpClassifier()`
in `PayeeCacheRepository.kt` was a keyword-matching stub with a
`// TODO: Replace with actual TFLite / API NLP call here` comment. FinText-6K had
never been touched by any training code. This work trains a real classifier on
that dataset, serves it from the backend, and wires the Android client to call it.

## Data

Source: `E:\Projects\upi\FinText-6K\` (Kaggle, Apache 2.0). Two pre-split CSVs,
`text,label` columns:

- `train_transaction_dataset.csv` — 5,000 rows, used for fitting only.
- `test_transaction_dataset.csv` — 1,000 rows, held out, used for evaluation only
  (never fit on, never re-shuffled with train).

Verified label strings exactly as they appear in the CSVs (case is inconsistent
in the source data, not a typo introduced here): `food`, `travel`, `EMI`,
`investment`, `shopping`. These are normalised to Title Case for the API
(`Food`, `Travel`, `EMI`, `Investment`, `Shopping`) — EMI kept as an acronym.

Class balance (roughly even, ~1000/class in train, ~200/class in test):

| label | train | test |
|---|---|---|
| food | 993 | 205 |
| travel | 1005 | 192 |
| EMI | 957 | 201 |
| investment | 1055 | 172 |
| shopping | 990 | 230 |

**Important caveat about the dataset itself:** it is synthetic and heavily
templated — the entire 5,000-row train split is generated from only **40
unique templates** (e.g. `"Restaurant payment of Rs # via UPI Ref #"`,
`"Mutual fund SIP of Rs # via UPI Ref #"`) with only the amount and reference
number varying. This is why the model below scores a perfect 100% on the held-out
test set: the test split draws from the same fixed template pool as train, so
there is essentially zero linguistic variation left to generalize over. This
is an honest property of the dataset, not an artifact of overfitting or a
leak — train and test are disjoint amount/reference-number instances of the
same 40 templates. In production, real bank SMS narrations will vary more
than this, so treat the 100% figure as a ceiling for "can it read these kinds
of templated narrations," not a promise about arbitrary free-text messages.

## Model

`train_category_classifier.py` builds a single sklearn `Pipeline`:

```
TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)
  -> CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid", cv=5)
```

TF-IDF + linear SVM is the right complexity for ~1,000 short, templated
narration strings per class — a transformer would add real latency/memory
cost on a free-tier Render deployment for no accuracy benefit here.
`CalibratedClassifierCV` wraps the SVM specifically so `predict_proba()`
returns genuine probability estimates (LinearSVC alone only exposes
`decision_function` margins), which the Android client's
`NLP_CONFIDENCE_THRESHOLD = 0.65f` gate needs.

### Held-out test-set results (1,000 rows, never trained on)

```
Accuracy    : 1.0000
Macro F1    : 1.0000
Weighted F1 : 1.0000

              precision    recall  f1-score   support
         EMI     1.0000    1.0000    1.0000       201
        Food     1.0000    1.0000    1.0000       205
  Investment     1.0000    1.0000    1.0000       172
    Shopping     1.0000    1.0000    1.0000       230
      Travel     1.0000    1.0000    1.0000       192
```

Confusion matrix (rows = true, cols = predicted; order `[EMI, Food, Investment, Shopping, Travel]`):

```
[[201   0   0   0   0]
 [  0 205   0   0   0]
 [  0   0 172   0   0]
 [  0   0   0 230   0]
 [  0   0   0   0 192]]
```

Mean top-class confidence on the test set: **0.9939**. 100% of test rows clear
the app's 0.65 confidence gate, and accuracy among those is 1.0 (trivially,
since it's the whole set).

Full metrics are also saved machine-readably at
`artefacts/paysense_category_classifier_metrics.json`.

## The category-vocabulary mismatch — decision made explicitly

FinText-6K only covers 5 classes: Food, Travel, EMI, Investment, Shopping.
The Android app has **two different** pre-existing category vocabularies that
were already inconsistent with each other before this change:

- `NlpKeywordRules` (the Tier-1.5 keyword table in `PayeeCacheRepository.kt`):
  Food, Travel, Shopping, Entertainment, Recharge, Utilities.
- The HITL bottom sheet (`layout_bottom_sheet_category.xml`, what the user
  actually taps): Food, Travel, Shopping, Bills, Grocery, Entertainment,
  Healthcare, Misc.
- Downstream UI (`FinanceFragment.categoryColors`, `TransactionAdapter.categoryIconMap`)
  already had color/icon entries for `EMI` (used nowhere upstream — dead
  entries) but nothing for `Investment`.

Decision taken: **option (a) from the task brief.** The classifier only ever
returns one of its own 5 trained classes — it is never force-mapped onto
category names it wasn't trained on (e.g. mapping "EMI" to "Bills" would be a
real semantic loss: an EMI payment is not a bill in the way electricity is,
and conflating them would produce wrong spending-category analytics). Instead:

1. `POST /classify`'s response schema documents explicitly that it returns
   Food/Travel/EMI/Investment/Shopping and nothing else.
2. The Android client's Tier-2 call (`PayeeCacheRepository.runNlpClassifier`)
   passes the result through unchanged — no re-mapping.
3. The app's category vocabulary is genuinely extended, not just silently
   fed new strings: `layout_bottom_sheet_category.xml` gained `EMI` and
   `Investment` chips (EMI already had dead downstream color/icon support;
   Investment had none and now does — see `FinanceFragment.kt` and
   `TransactionAdapter.kt`), so a user can confirm these categories by hand
   in Tier 3 too, not just receive them silently from Tier 2.
4. Categories the classifier was never trained on (Bills, Grocery,
   Entertainment, Healthcare, Misc, Utilities, Recharge) simply cannot come
   from Tier 2 — they still only ever get assigned via the keyword table or
   Tier-3 human input. This is a real, stated limitation, not a bug.

## Backend: `POST /classify`

Added to `main.py`, following `/predict`'s existing patterns:

- JWT-protected via the same `get_current_user` dependency (bypassed only
  in `APP_ENV=development`, same as `/predict`).
- Rate-limited 60/min per IP via the same `slowapi` limiter.
- Request: `{"text": string}` (1–1000 chars). Response:
  `{"category": string, "confidence": float}`.
- Returns `503` if the artefact isn't present (e.g. training hasn't been run
  yet on a fresh checkout) rather than crashing — same graceful-degradation
  philosophy as the existing ensemble scorer.
- Model loading lives in `src/fraud_model.py` (`classify_category()`), next
  to the existing XGBoost/LightLR loading code, with a lazy-load fallback so
  it also works correctly when FastAPI's lifespan startup event doesn't fire
  (e.g. a bare `TestClient(app)` used without a `with` block, as the existing
  test suite does).

10 new tests added to `tests/test_api.py` (`TestClassify`): auth requirement,
one correctness check per class, response-schema check, and 422 validation
for empty/missing text. Full suite: **97 tests passing** (87 pre-existing + 10
new), run via the project's existing `venv/`.

## Android: Tier 2 wiring

- `PaySenseApi.kt` — new `@POST("/classify") suspend fun classifyCategory(...)`.
- `ApiModels.kt` — new `CategoryRequest`/`CategoryResponse` data classes
  (Gson `@SerializedName` matching the Pydantic field names exactly, same
  convention as `TransactionRequest`/`TransactionResponse`).
- `FraudApiService.kt` — new `classifyCategory(rawBody): Pair<String, Float>?`
  method reusing the existing Retrofit/`PaySenseApi` singleton (no second
  client built). Returns `null` on any network error, non-2xx, or empty body.
- `PayeeCacheRepository.kt` — `runNlpClassifier()` now: (1) tries the
  keyword table (unchanged), then (2) on a miss, calls
  `FraudApiService.classifyCategory(txn.rawBody)`. The existing
  `NLP_CONFIDENCE_THRESHOLD = 0.65f` gate in `resolveCategory()` is
  untouched and applies uniformly to both keyword and API results — below
  threshold or on `null`, it falls through to Tier 3 HITL exactly as before.

`BASE_URL` (`https://paysense-upifraud-detection.onrender.com/`) is unchanged; `/classify`
is a new relative path on the same host and won't be live there until this
backend change is deployed — the code is correct and ready for that.

## Reproducing training

```
cd PaySense-ML-Backend
venv\Scripts\python.exe train_category_classifier.py
```

Requires `E:\Projects\upi\FinText-6K\{train,test}_transaction_dataset.csv` to
exist (override the directory via the `FINTEXT_DIR` env var if moved).
Writes `artefacts/paysense_category_classifier.pkl` and
`artefacts/paysense_category_classifier_metrics.json`.

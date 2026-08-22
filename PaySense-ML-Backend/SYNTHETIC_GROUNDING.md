# PaySense — Grounded Synthetic Dataset & Full-Feature Generalization Check

**Date:** 2026-08-22
**Author's intent:** `GENERALIZATION_CHECK.md` tested the frozen model against
two real external UPI datasets and found they carried only 15.0% and 5.0% of
the model's 40-feature vector — real datasets never carry PaySense's
personalization-heavy fields (per-user z-scores, device/IP risk, KYC/merchant
flags). That check could only ever measure "does it work with 85% of the
input missing," not "does it generalize to a genuinely different but
complete data-generating process." This document builds and scores the
complement: an independently-generated synthetic dataset carrying the full
40-feature schema, grounded in real published statistics rather than
arbitrary numbers, so the missing-feature confound can finally be removed
from the question.

Scripts: `generate_grounded_synthetic_dataset.py` (dataset),
`generalization_check_synthetic.py` (scoring, via the real
`src.fraud_model.score()` ensemble path — not raw `model.predict_proba()`).
No retraining, no fine-tuning, no threshold recalibration anywhere in this
document.

---

## 1. Real-world grounding: what "fraud rate" actually means

### 1.1 Government/regulator figures used (as supplied, not re-derived)

- **RBI Annual Report FY2024-25** (via Business Standard's coverage): digital
  payment frauds — 13,516 cases (56.5% of all reported banking frauds),
  ₹520 crore in that sub-category. Separately, overall bank frauds *by
  amount* rose to ₹36,014 crore in FY25 from ₹12,230 crore in FY24, on
  23,953 cases (down from 36,060 cases). These are different reporting
  categories and are not conflated here.
- **UPI-specific fraud** (Ministry of Finance, Lok Sabha reply, widely
  reported e.g. via The420.in / Nation Press, Dec 2025):

  | Year | Amount (₹ crore) | Incidents (lakh) |
  |---|---:|---:|
  | FY22 | 242 | — |
  | FY23 | 573 | — |
  | FY24 | 1,087 | 13.42 |
  | FY25 | 981 | 12.64 |
  | FY26 (through Nov, partial) | 805 | 10.64 |

- **UPI transaction volume** (RBI/NPCI): FY24 — 131.1 billion transactions;
  FY25 — 185.9 billion transactions (+41.7%), value ₹260.6 lakh crore in
  FY25.

### 1.2 Derived population fraud rate (arithmetic shown)

```
FY25 rate by count = 12.64 lakh / 185.9 billion
                    = 1,264,000 / 185,900,000,000
                    ≈ 0.00068%

FY24 rate by count = 13.42 lakh / 131.1 billion
                    = 1,342,000 / 131,100,000,000
                    ≈ 0.00102%

FY25 avg loss/incident = ₹981 crore / 12.64 lakh
                        = ₹9,810,000,000 / 1,264,000
                        ≈ ₹7,761

FY24 avg loss/incident = ₹1,087 crore / 13.42 lakh
                        ≈ ₹8,100
```

**The real-world UPI fraud rate is approximately 0.0007-0.001% of
transactions — three to four orders of magnitude below the ~3.8-4.2% rate in
every labeled dataset in this project, including the one built below.**

### 1.3 The enrichment gap, stated plainly

Nothing in this project's prior documentation states the real population
rate or where the ~3.8-4.2% training rate comes from — it is presented as if
it were natural. It is not. A dataset sampled at the true ~0.0007% rate would
contain almost no positive examples in any dataset small enough to train or
evaluate on (a 25,000-row dataset at the real rate would contain roughly
**0.17 fraud rows** — not usable for anything). Every fraud-detection dataset
in this project, including `transactions.csv`
(`data_dictionary.csv`: "~3.8% positive rate"), `paysense_master_dataset.csv`
(~4.21%, per `paysense_ml_pipeline.py`'s own Stage-B log message), and the
dataset built here, is **deliberately fraud-enriched by roughly 5,000-6,000x**
the real FY25 population rate, precisely so there is a learnable/measurable
positive class. This is standard, necessary practice in fraud modeling — the
problem is only ever that it went uncited before now.

---

## 2. The dataset: `generate_grounded_synthetic_dataset.py`

### 2.1 Deliberate choice: enrichment, not population realism

Two honest options existed:
(a) build at the real ~0.0007% rate to test "prevalence realism," accepting
that almost no positive examples would exist to measure recall/precision at
all, or (b) enrich the label to a rate comparable to the training set's own
documented rate, explicitly flagged as enrichment (not a real-world claim),
so ROC-AUC/PR-AUC/confusion-matrix/recall are all measurable.

**Choice made: (b), consistently.** `TARGET_FRAUD_RATE = 0.040` (4.0%) — in
the same band as `data_dictionary.csv`'s documented ~3.8% and
`paysense_ml_pipeline.py`'s actual ~4.21%, but not copied from either. This
keeps the comparison to the training distribution's positive-class *density*
apples-to-apples while still being an independently-chosen number, and it is
stated here rather than silently mixed with §1's population-rate figures.

### 2.2 Why this is a genuinely separate draw, not a re-run

| | Original pipeline (`paysense_pipeline.py` / `paysense_ml_pipeline.py`) | This dataset |
|---|---|---|
| RNG seed | `np.random.seed(42)` / `RANDOM_STATE=42` | `SEED = 918273` |
| Row count | 30,000 (20k anchor + 10k supplement) | 25,000 |
| Structure | Blends two pre-existing sources via a rule-based schema bridge | Single unified generative model |
| `new_device_flag`/`ip_location_mismatch` | Derived from continuous `device_risk_score`/`ip_risk_score` via a hard **0.70 threshold** (pipeline lines ~318-324) | Sampled directly as independent low-probability events; continuous `device_risk_score`/`ip_risk_score` derived **from** the flags (opposite causal direction — Beta(8,2) when flagged, Beta(2,8) otherwise) |
| `is_fraud` | Two-source anchor/supplement blend | Single **calibrated logistic risk model**: `logit = Σ(weight_i · signal_i) + noise`, intercept binary-searched so `mean(sigmoid(logit)) == 0.040`, then `is_fraud ~ Bernoulli(p)` per row |
| `amount_deviation_score` | Unknown internal method | Computed as a **real per-row statistic**: `|amount - user_avg_txn_value| / (user_avg_txn_value·0.5 + 50)`, clipped to [0,10] |

### 2.3 Grounding table — every distribution's source

Columns with an exact figure from `data_dictionary.csv` are used as-is.
Columns where the dictionary gives only a range/category list without a
shape are explicitly marked **ASSUMPTION** in the script's own comments and
below — never presented as if cited.

| Field | Grounding | Type |
|---|---|---|
| `amount` — log-normal, cap ₹100,000 | `data_dictionary.csv` (transactions.csv) | Exact |
| `is_night_transaction` = 1 if hour<6 or hour≥22 | `data_dictionary.csv` | Exact rule |
| `is_weekend` = 1 if Sat/Sun | `data_dictionary.csv` | Exact rule |
| `recurring_payment_flag` = 1 for Bill Payment/Subscription/EMI | `data_dictionary.csv` | Exact rule |
| `transaction_frequency_score` = min(avg_monthly_txn/50, 1) | `data_dictionary.csv` | Exact formula |
| `status` split 88/9/3 (Success/Failed/Pending) | `data_dictionary.csv` | Exact |
| `failed_attempts_last_24h` ~72% zero | `data_dictionary.csv` | Exact; nonzero shape (1-5) is an ASSUMPTION |
| `time_since_last_txn_min`/`transaction_velocity`/`amount_deviation_score` ~2% missing | `data_dictionary.csv` | Exact |
| `kyc_status` ~87% Verified | `data_dictionary.csv` (users.csv) | Exact |
| `linked_bank_count` ~50% = 1 | `data_dictionary.csv` | Exact; split of remaining 50% is an ASSUMPTION |
| `avg_monthly_transactions` Tier 1 ≈45/mo | `data_dictionary.csv` | Exact for Tier 1; Tier 2/3 means are ASSUMPTIONs |
| `is_high_risk_user` ~5% | `data_dictionary.csv` | Exact |
| `merchant_size` ~55% Small | `data_dictionary.csv` (merchants.csv) | Exact; Medium/Enterprise split is an ASSUMPTION |
| `is_registered` ~94% | `data_dictionary.csv` | Exact |
| Merchant category count = 11 | `data_dictionary.csv` ("11 categories total") | Exact count, category *names* are an ASSUMPTION (dict gives only examples) |
| Hour-of-day shape, payment-app/device-type shares, city-tier shares, transaction-type shares, `ip_location_mismatch` baseline rate, account-age shape | Not given by the dictionary | **ASSUMPTION** — plausible, documented in-script, not cited to any source |

### 2.4 Verified against the frozen preprocessor — two dictionary/artefact mismatches found

While wiring this dataset through `fraud_model.score()`, two columns caused
every single `PaySense scorer failed: Cannot use median strategy with
non-numeric data` error:

- **`day_of_week`**: `data_dictionary.csv` documents this as a string
  (`"Monday"`, `"Saturday"`). But `artefacts/paysense_preprocessor.pkl`'s
  fitted `ColumnTransformer` routes `day_of_week` through its **31-column
  numeric pipeline**, not the 9-column categorical pipeline (verified via
  `prep.transformers_`). `paysense_master_dataset.csv` itself stores it as
  `int64`, matching pandas' `.dt.dayofweek` convention (Monday=0 ...
  Sunday=6 — confirmed by cross-checking that rows with `day_of_week ∈
  {5,6}` are exactly the rows with `is_weekend=1`).
- **`user_city_tier` / `usr_home_city_tier`**: same story. The dictionary
  documents `"Tier 1"/"Tier 2"/"Tier 3"` strings; the frozen preprocessor's
  `ColumnTransformer` treats both columns as numeric, and
  `paysense_master_dataset.csv` stores them as `int64` `{1,2,3}`.

**This dataset matches the frozen artefact's real expectation (numeric),
not the dictionary's literal documented dtype** — because the goal is
scoring through the actual frozen model, and a dataset that matched the
dictionary instead would fail 100% of PaySense scores. Both mismatches are
recorded here rather than silently "corrected" in either document, and
regression tests (`tests/test_synthetic_dataset_generation.py::
test_day_of_week_is_numeric_not_string`, `::test_city_tier_columns_are_numeric_1_2_3`)
guard against reintroducing the string form.

### 2.5 Generation-time sanity checks (guarding against the Trojan/Dataset-2 red flags)

`GENERALIZATION_CHECK.md` §2.2 rejected a candidate dataset because its
features separated the fraud label near-deterministically (0% fraud in one
category, 85%+ in others; correlations up to 0.68-0.75). The same scrutiny
applied to this dataset, at generation time:

```
new_device_flag        fraud rate | flag=1: 14.69%  flag=0:  3.51%
ip_location_mismatch   fraud rate | flag=1:  9.11%  flag=0:  3.49%
amount_deviation_score correlation with is_fraud: 0.041
```

Real, meaningful elevation (4-5x for the two hard flags) with no category
anywhere near 0% or 100% — the same shape of evidence that made Dataset 1
credible and Dataset 2 not. `amount_deviation_score`'s correlation (0.041) is
notably weak — weaker than the disqualified Dataset 2's 0.66-0.75 range by a
wide margin, and weaker even than the accepted Dataset 3's 0.45. This is a
side effect of the logistic model's Gaussian noise term (σ=0.6) dominating
that particular signal's contribution; it is not tuned to be flattering in
either direction.

### 2.6 Realised distribution check (25,000-row production run, `SEED=918273`)

```
Realised fraud rate: 985/25,000 = 3.940%   (target 4.000%)
status split           : Success 87.7% / Failed 9.2% / Pending 3.1%   (target 88/9/3)
user_kyc_status Verified: 85.52%                                      (target ~87%)
failed_attempts==0      : 71.86%                                      (target ~72%)
time_since_last_txn NaN : 2.05%                                       (target ~2%)
amount_deviation_score NaN: 2.23%                                     (target ~2%)
transaction_velocity NaN: 1.97%                                       (target ~2%)
amount max              : ₹37,541.15                                  (cap ₹100,000)
```

All target rates realised within 1.5 percentage points.

---

## 3. Realism sanity check against `MyTransaction.csv`

`E:\Projects\upi\MyTransaction.csv` (1,470 rows) is a real personal bank
statement export — Date/Category/RefNo/Withdrawal/Deposit/Balance, with
reference numbers in scientific notation (`3.00E+11`), a classic Excel
corruption artifact of a genuine exported statement, not something a
synthetic generator would produce. It has no fraud labels and is not
UPI-specific, so it cannot train the model — it is used here only as an
independent realism check on amount magnitude.

| | `MyTransaction.csv` (real, 1,304 spend rows) | This synthetic dataset (25,000 rows) |
|---|---:|---:|
| Median amount | ₹55 | ₹498.53 |
| Mean amount | ₹438.40 | ₹957.06 |
| IQR | ₹30 - ₹156.52 | ₹235.05 - ₹1,068.98 |
| Max | ₹21,000 | ₹37,541.15 |

**Honest read: same order of magnitude, but the synthetic dataset's central
tendency runs noticeably higher than a real personal account.** The real
data's median (₹55) reflects a spending pattern dominated by small day-to-day
purchases (912 of 1,304 spend rows are `Food`, median well under ₹100); the
synthetic generator's `amount` is drawn log-normal around each synthetic
user's own `avg_transaction_value` profile stat (range ₹50-10,000 per
`data_dictionary.csv`), which pulls the median toward the mid-hundreds rather
than the double digits a real low-value UPI transaction stream shows. Both
datasets stay well under the documented ₹100,000 cap and the real max
(₹21,000) and synthetic max (₹37,541) are the same order of magnitude — this
is not the kind of "wildly off" result that would disqualify the dataset,
but it is a real, reportable gap between synthetic and organic spending
patterns, not glossed over here.

Category comparison: `MyTransaction.csv`'s category mix (Food 62.1%, Misc
34.4%, Shopping 2.2%, Salary 0.7%, Rent 0.5%, Transport 0.2%) is a single
individual's spend pattern, not comparable one-to-one against this dataset's
11-category merchant taxonomy (`data_dictionary.csv` merchant categories) —
noted as a limitation of this comparison rather than forced into a mapping
that would fabricate a correspondence.

---

## 4. Layer-1 side-check: `spam.csv` against Gate 1/Gate 2

**This is a side-finding, not the main task — reported briefly.**

`E:\Projects\upi\spam.csv` (1,000 rows, `v1`=ham/spam, `v2`=text) was checked
against the Android client's Layer-1 gating regexes
(`PaySense-Android-Client-New/app/src/main/kotlin/com/paysense/app/layer1/SmsReceiver.kt`):

- **Gate 1** (`^[A-Z]{2}-[A-Z0-9]{4,6}$`) validates the **SMS sender ID**
  (TRAI DLT format, e.g. `VM-HDFCBK`). `spam.csv` has no sender field — only
  label and message body — so Gate 1 cannot be exercised against this
  dataset at all. This is stated plainly rather than fabricating a sender
  column to force a result.
- **Gate 2** (`(?i)\b(debited|credited|upi|rs\.|inr|transaction|payment)\b`)
  was ported to Python `re` (the pattern needs no Kotlin-specific behavior)
  and run against all 1,000 `v2` texts.

**Result: 0 of 1,000 rows (0 spam, 0 ham) match Gate 2.** None of the seven
keywords appear anywhere in the dataset — confirmed by direct substring
search (`debited`, `credited`, `upi`, `rs.`, `inr`, `transaction`, `payment`
all return zero hits). So there is no false-positive risk from this
particular file at Gate 2 — but the reason is not that the filter is robust,
it's that this file doesn't contain the kind of content Gate 2 is built to
catch or reject.

**A more important finding, in the same spirit as this project's own
"Trojan Family" dataset scrutiny:** this file is **not** the real UCI SMS
Spam Collection dataset it is named after. It contains only **20 unique
message templates**, each repeated ~50 times, with the ham/spam label
assigned essentially at random per repetition — e.g. the identical string
`"Wishing you a great day ahead!"` appears labeled `ham` 26 times and `spam`
26 times; `"WINNER!! As a valued network customer"` appears as `ham` 23 times
and `spam` 32 times. No real SMS spam corpus has this shape (the genuine UCI
dataset has ~5,500 largely-unique messages with real word-level signal
separating the classes). This file is unsuitable for any real spam
classification task and was not used for anything beyond the literal Gate-2
regex check above.

---

## 5. Results: frozen ensemble vs. this dataset

Full script output reproducible via `python generalization_check_synthetic.py`.

### 5.1 Metrics (25,000 rows, 985 fraud = 3.94%)

| Metric | Raw XGBoost only | Full ensemble (XGBoost + rules + LightLR) |
|---|---:|---:|
| ROC-AUC | 0.6811 | **0.6947** |
| PR-AUC (average precision) | 0.0945 | **0.1017** |
| Confusion matrix @ threshold 0.40 | TN=22,816 FP=1,199 / FN=804 TP=181 | TN=22,258 FP=1,757 / FN=735 TP=**250** |
| Precision / Recall / F1 (fraud class) | — | 0.1246 / 0.2538 / 0.1671 |
| Max predicted score | 0.9961 | 0.9660 |

A side-effect worth reporting on its own: **1,035 of 25,000 rows (4.14%)**
triggered a `LightLR scorer failed` warning inside `fraud_model.score()`.
`_score_light_lr()` reads its 5 features via `txn_dict.get(key, 0.0)`, whose
default only applies when a key is **absent**. This dataset intentionally
sends `None` for `amount_deviation_score`/`transaction_velocity` on ~2% of
rows each (matching `data_dictionary.csv`'s documented "~2% missing
intentionally"), as an explicit present-but-null value — a realistic
representation of an optional field a real client sent empty. `.get()`
returns that `None`, and `float(None)` raises, so LightLR silently drops out
of the ensemble for those rows (weight renormalizes to `paysense`+`rules`
only — confirmed via `weights_used`, which took exactly two distinct values
across the run: `{light_lr:0.25, paysense:0.60, rules:0.15}` and
`{paysense:0.80, rules:0.20}`). This is a genuine, minor robustness gap
between "field omitted" and "field sent as null" in the production scorer —
noted here, not fixed, since fixing production code was out of scope for
this check.

### 5.2 Side-by-side with `GENERALIZATION_CHECK.md`'s two external datasets

| Metric | Dataset 1 (6/40 features, real UPI data) | Dataset 3 (2/40 features, real data) | **This dataset (40/40 features, synthetic)** |
|---|---:|---:|---:|
| ROC-AUC (full ensemble) | 0.8107 | n/a (raw XGBoost only, 0.6179) | **0.6947** |
| PR-AUC (full ensemble) | 0.4136 | n/a | **0.1017** |
| Recall @ 0.40 threshold | 0/701 = **0.0%** | 0/64 = **0.0%** | **250/985 = 25.4%** |

### 5.3 What this actually means — the honest verdict

This is the finding the task brief explicitly asked not to be softened, and
it is not flattering, in a specific and interesting way:

**Recall genuinely improves — from 0% on both real external datasets to
25.4% here — confirming that missing features really were part of the
problem in `GENERALIZATION_CHECK.md`.** With the full 40-feature vector
present, the frozen threshold (0.40) is no longer permanently out of reach:
250 of 985 real fraud rows in this dataset score above it, something that
never happened even once across 74,917 + 1,000 real external rows. That is a
genuine, positive, and previously-unmeasurable data point — the earlier
checks literally could not distinguish "the model doesn't work" from "the
model never got enough information to work," and this result resolves that
ambiguity partially in the model's favor: given full information, it does
fire on real positives.

**But ROC-AUC and PR-AUC are both *worse* here than on Dataset 1 — despite
this dataset supplying 40/40 features against Dataset 1's 6/40.** ROC-AUC
0.6947 vs. Dataset 1's 0.8107; PR-AUC 0.1017 vs. Dataset 1's 0.4136. Both are
far below the model's own reported held-out training metrics (ROC-AUC
0.8851, PR-AUC 0.5303). If missing features were the *only* thing limiting
generalization, a full-feature dataset should have outperformed a
15%-feature one by a wide margin — instead it is *ranking fraud above
non-fraud worse*, not better, once every field is present.

**The honest explanation is that this result is doing exactly what it was
built to do: isolate the "different generation process" variable from the
"missing features" variable, and it shows both matter, independently.**
Dataset 1's 6 mapped features (amount, hour, night flag, merchant category,
device, success flag) are all *organically real* — sourced from an actual
observed UPI transaction stream — even though there were few of them. This
dataset's 40 features are all *present*, but every one of them, including
the fraud label itself, was generated by a deliberately different process
(§2.2) from the one the frozen model was trained on: a calibrated logistic
model over standardized signals plus Gaussian noise, versus the training
pipeline's anchor/supplement blend with hard-threshold flag derivation. The
model evidently learned decision boundaries tied to specifics of its own
training pipeline's correlation structure — not just to which fields were
populated — and a full feature vector generated by a different process
degrades ranking quality even as it (correctly) lets the model act on
signals it couldn't see at all before.

**What this proves:** feature completeness measurably helps the model act
(0% → 25.4% recall) — the missing-feature confound in
`GENERALIZATION_CHECK.md` was real and is now partially disentangled from
the deeper question. **What this does not prove, and what this document
will not soften:** the frozen model still ranks and calibrates worse on a
full-feature dataset from a different generative process than it does on a
mostly-empty dataset from a real one. That is evidence of at least partial
overfitting to the training pipeline's specific correlation structure
(the anchor/supplement blend and its 0.70-threshold flag derivation), not
purely to a lack of information. Neither this document nor
`GENERALIZATION_CHECK.md` supports a claim that the frozen model is ready
for production against real-world UPI traffic; this document adds a second,
independent line of evidence for the same conclusion, from the opposite
direction (full features, different process) rather than the same one
(sparse features, real process).

---

## 6. Reproducing this check

```
cd PaySense-ML-Backend
venv\Scripts\python.exe generate_grounded_synthetic_dataset.py      # writes synthetic_grounded_dataset.csv (25,000 rows x 50 cols)
venv\Scripts\python.exe generalization_check_synthetic.py           # scores it through the real ensemble (several minutes; ~67 rows/sec)
```

Unit tests covering the generator's claimed distributions and the two
dictionary/artefact dtype mismatches (§2.4):

```
venv\Scripts\python.exe -m pytest tests/test_synthetic_dataset_generation.py -v
```

No new dependencies were installed; both scripts and the test module use
only what `requirements.txt` / the existing `venv/` already provide.
`synthetic_grounded_dataset.csv` is written under `PaySense-ML-Backend/`
(git-ignored the same way `paysense_master_dataset.csv` is, if applicable —
check `.gitignore` before committing a 25,000-row CSV).

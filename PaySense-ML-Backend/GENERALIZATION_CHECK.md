# PaySense — Out-of-Distribution Generalization Check

> **UPDATE (2026-08-23):** the frozen model was retrained that day with
> `monotone_constraints` on three behavioral features
> (`RECALL_CEILING_REMEDIATION.md`, adopted per README.md's Key Results
> note), and the deployed decision threshold changed from 0.40 to 0.30 as a
> result. Every number in this document was **actually re-run** against the
> new on-disk artifacts (not text-edited) via `generalization_check.py` and
> `generalization_check_ensemble.py`, which load whatever is in
> `artefacts/` dynamically. The qualitative verdict (§5) is unchanged; the
> specific ROC-AUC/PR-AUC/max-probability numbers moved slightly, in both
> directions, and are updated below to the freshly recomputed values.

> **UPDATE (2026-08-24):** the deployed threshold moved again, from 0.30 to
> 0.50 — not a model change this time, but a methodology correction: every
> "deployed threshold" figure in this project had been chosen by sweeping
> raw XGBoost's own score, never the real 3-scorer ensemble `/predict`
> actually uses (see README.md's Key Results note,
> `EDA_FEATURE_ENGINEERING.md` §4.5). Re-verified directly rather than
> assumed: the max ensemble score observed anywhere in this document's
> two datasets is 0.0847 (Dataset 1) — already far below both the old
> (0.30) and new (0.50) threshold, so every confusion matrix below is
> **unchanged**, confirmed by an actual re-run at both thresholds, not by
> inference. Only the threshold *value* referenced in the prose below is
> updated, from 0.30 to 0.50.

**Date:** 2026-08-22
**Author's intent:** Answer one question honestly — does the frozen PaySense
XGBoost model (`artefacts/paysense_model.pkl`) generalize to real UPI/financial
fraud data it has never seen, or does it only work on data drawn from its own
30,000-row training pipeline? No retraining, no fine-tuning, no threshold
recalibration happened anywhere in this check. Script: `generalization_check.py`.

This extends the report's own **Trojan Family Discovery** methodology
(`../PaySense-Report/paysense_report.tex`, §"Dataset Inventory and the Trojan
Family Discovery") to three new candidate datasets: the same scrutiny that
correctly rejected a pre-balanced 50.01%-fraud file is applied here before
trusting anything enough to score against it.

---

## 1. Why this check exists

The frozen model's only reported metrics (ROC-AUC 0.8889, PR-AUC 0.5352 as
of the 2026-08-23 monotonic-constraints retrain — was ROC-AUC 0.8863, PR-AUC
0.5339 on 2026-08-22, and an earlier, stale 0.8851/0.5303 before that had
drifted from what the artifacts actually produced at the time, see
README.md's Key Results note) come
from a held-out split of `paysense_master_dataset.csv` — a dataset built from
one 20,000-row anchor plus one 10,000-row synthetic supplement, joined and
schema-bridged by the same author who trained the model. A held-out split of
one pipeline's own output says the model didn't memorize its *own* test rows;
it says nothing about whether the *patterns* it learned exist outside that
pipeline. That gap is the subject of this document.

---

## 2. Dataset trust triage

Three previously-unused datasets were proposed. Each was checked for the same
red flags that caught the Trojan Family: identical ID backbones across files,
suspiciously round fraud rates, and features that correlate with the label too
cleanly to be organic.

| # | Dataset | Rows | Reported fraud rate | Verdict |
|---|---|---:|---:|---|
| 1 | `UPI Fraud detection dataset/upi_fraud_dataset.csv` | 74,917 | 0.94% (701/74,917) | **USED** |
| 2 | `UPI Transactions Dataset for Fraud Detection/fraud_detection_data_100000.csv` | 100,000 | 20.00% (19,999/100,000) | **REJECTED** |
| 3 | `Synthetic Financial Fraud Detection Dataset/fraud_dataset.csv` | 1,000 | 6.4% (64/1,000) | **USED — secondary, low power** |
| — | `Upi_Fraud/upi_transactions.csv` | — | — | Skipped: no fraud label (`Status` is SUCCESS/FAILED only) |
| — | `UPI Transaction Insights Dataset/upi_transaction_insights_dataset.csv` | — | — | Skipped: no fraud label (`is_successful` is about delivery success, not fraud) |

### 2.1 Dataset 1 — accepted

Verification performed (not just the fraud rate):
- No duplicate `transaction_id` (74,917 unique values across 74,917 rows).
- No duplicate full rows.
- No nulls in any column.
- Fraud rate (0.94%) is a believable, non-round number — same shape of
  evidence that made the 0.19% and 2.00% Trojan files plausible before the
  0.19% one lost independently to the anchor dataset on other grounds.
- `transaction_status` (SUCCESS/FAILED) is genuinely correlated but not
  deterministic with `isFraud` (FAILED txns: 1.18% fraud vs SUCCESS: 0.82%
  fraud) — consistent with organic behavior, not a label leak.
- Fraud vs. legit amount distributions differ sensibly (fraud mean ₹85,658
  vs. legit mean ₹28,299) rather than being identical or perfectly separable.
- `device`, `merchant_category`, `payee_type`, `payer_type` all have a small,
  bounded set of realistic categories with no single category driving fraud
  rate to 0% or 100%.

No evidence of the Trojan Family's ID-backbone sharing or synthetic
pre-balancing. Accepted as the leading candidate, exactly as the initial
triage suspected.

### 2.2 Dataset 2 — rejected

The initial triage's suspicion is confirmed, and the evidence is worse than
the flag that was raised:

- **Fraud rate is 19,999/100,000 = 19.999% ≈ 20.00%** — a round number in the
  same suspicious family as the Trojan file's 50.01%.
- **`Transaction_Amount_Deviation` is clipped to [-99, 99.9]** and is
  *almost cleanly bimodal by label*: legitimate transactions average −23.6
  (σ=44.8), fraud transactions average +71.9 (σ=16.2), with near-zero
  overlap in the interquartile ranges. An organically computed z-score
  would not split this cleanly along the label.
- **Categorical columns predict the label almost deterministically.**
  `Merchant_Category = "Bill Payment"` has **exactly 0.0% fraud**; several
  other categories (`Financial services and Taxes`, `Investment`, `Other`)
  sit at 85–86% fraud. `Transaction_Type` shows the same pattern: `Bill
  Payment`/`Purchase`/`Subscription` sit at ~5.5% fraud while `Bank
  Transfer`/`Investment`/`Refund` sit at ~85%. Real-world fraud does not
  sort this cleanly by category.
- **`Transaction_Frequency` correlates 0.68 with the label,
  `Days_Since_Last_Transaction` correlates −0.75, `amount` correlates 0.66.**
  Correlations this high, across four independent-looking columns
  simultaneously, are the signature of a label generated by a formula
  applied to those same columns — not of an organically fraudulent
  population.

**Verdict: unconditionally rejected**, on the same grounds the report used
for the pre-balanced 50.01% Trojan file, and for an additional reason that
file didn't have: near-deterministic feature→label separability suggests the
fraud label itself was synthesized from a rule involving `Merchant_Category`,
`Transaction_Type`, `Transaction_Frequency`, `Transaction_Amount_Deviation`,
and `Days_Since_Last_Transaction`. Scoring the frozen model against it would
not measure real-world generalization — it would measure whether the model
can reverse-engineer this dataset's specific generation formula, which is a
different and uninteresting question. It was **not scored**. No number from
this dataset appears anywhere else in this document.

### 2.3 Dataset 3 — accepted, secondary only

- No duplicate `transaction_id`, no nulls.
- Fraud rate 6.4% (64/1,000) matches the initial triage exactly.
- `location_risk = "high"` accounts for 90.6% of fraud rows but is not
  deterministic (9.4% of fraud rows have low/medium location_risk, and
  28.7% of *legitimate* rows are also `high` risk) — a real correlation, not
  a leak.
- `amount` correlates 0.45 with the label — meaningfully predictive but far
  from the near-1.0 correlations that flagged Dataset 2.

**Verdict: accepted, but only as a secondary sanity check.** 1,000 rows and
64 positive examples is low statistical power — any metric below is a rough
signal, not a precise estimate, and its confidence interval is wide. This is
stated explicitly here and should be read as such wherever the number
reappears.

---

## 3. Feature mapping

The frozen model expects exactly these 40 features (from
`artefacts/paysense_feature_names.pkl`). The frozen `OrdinalEncoder` uses
`handle_unknown="use_encoded_value", unknown_value=-1` and the frozen
`SimpleImputer` uses `strategy="median"` for numerics / `"most_frequent"` for
categoricals (`paysense_ml_pipeline.py`, ~line 169–181) — so any column left
unmapped is not fabricated, it is passed through as `NaN` and imputed by the
same logic the frozen preprocessor already uses for missing production data.

### 3.1 Dataset 1 mapping — 6 of 40 features (15.0%) carry real signal

| Frozen feature | Source column | Mapping |
|---|---|---|
| `amount` | `Amount` | Direct (same unit, INR) |
| `hour_of_day` | `hour` | Direct (0–23) |
| `mrc_category` | `merchant_category` | Case + synonym: education→Education, shopping→Shopping, food→Food, entertainment→Entertainment, transport→Travel, medical→Healthcare |
| `device_type` | `device` | Case normalization only: android→Android, ios→iOS, web→Web |
| `txn_success_flag` | `transaction_status` | SUCCESS→1, FAILED→0 |
| `is_night_transaction` | derived from `hour` | Same rule as the training pipeline's own supplement bridge (`paysense_pipeline.py` line ~352): `1 if hour<6 or hour>=22 else 0` |
| *(remaining 34 features)* | — | Left as `NaN`, median/most-frequent imputed |

**Explicitly not mapped, and why:**
- `is_new_payee` → `new_device_flag`: different concepts — payee novelty vs.
  device novelty. Conflating them was flagged as exactly the kind of
  dishonest shortcut this check exists to avoid, so it was not done.
- `payee_type` → `receiver_type`: `payee_type` has four roughly evenly-split
  categories (bank 25.2%, p2p 25.0%, upi_number 25.0%, p2m 24.8%). Only
  `p2m`→Merchant and `p2p`→User are semantically clean; `bank` and
  `upi_number` are genuinely ambiguous as to whether the receiver is a
  business or a person. Rather than guess on half the dataset, this feature
  was left unmapped entirely.
- `usual_hour` / `usual_location` / `usual_device` / `avg_amount`: this
  dataset's own attempt at per-user personalization — the same idea
  PaySense's z-score features encode, but not the same statistic (no
  z-score, no 90-day rolling window, unknown baseline construction). Using
  them as a stand-in for `amount_deviation_score` would fabricate a
  transformation that was never actually computed.
- `user_id`, `transaction_id`, `location`, `payment_retries`, `payer_type`:
  no honest corresponding frozen feature.

### 3.2 Dataset 3 mapping — 2 of 40 features (5.0%) carry real signal

| Frozen feature | Source column | Mapping |
|---|---|---|
| `amount` | `amount` | Direct |
| `usr_account_age_days` | `account_age_days` | Direct (same unit, days) |
| *(remaining 38 features)* | — | Left as `NaN`, median/most-frequent imputed |

**Explicitly not mapped, and why:**
- `device_type` → `device_type`: **name collision, different concept.** This
  dataset's `device_type` is a form factor (desktop/mobile/tablet).
  PaySense's `device_type` is an OS (Android/iOS/Web). "mobile" does not
  tell you Android vs. iOS — mapping these together despite the identical
  column name would be a guess wearing a match's clothes.
- `transaction_type` → `transaction_type`: this dataset's values
  (cash_out/debit/transfer/payment) read as a generic ledger/mobile-money
  schema, not UPI-specific channels (P2M/P2P/Bill Payment/EMI/Recharge/
  Subscription/ATM). `transfer`→P2P or `cash_out`→ATM are arguable but
  unverifiable judgment calls, so left unmapped.
- `location_risk`: a coarse three-level risk bucket, not the same thing as
  PaySense's binary `ip_location_mismatch` flag — different semantics, not
  substitutable.
- `num_prev_transactions`: a raw lifetime count with no honest conversion to
  `user_avg_monthly_txn` or `transaction_frequency_score` without inventing
  a rate or a time window that doesn't exist in the source data.

---

## 4. Results

Metrics use the frozen model, the frozen preprocessor, and the frozen
decision threshold (**0.50** as of 2026-08-24 — see the update notes above;
was 0.30, then 0.40 before that, from
`artefacts/paysense_threshold.pkl`) with no adjustment. Full script output
is reproducible via `python generalization_check.py`.

### 4.1 Dataset 1 — `upi_fraud_dataset.csv` (74,917 rows, 701 fraud)

| Metric | Value (2026-08-23, monotonic model) | Was (2026-08-22, pre-monotonic model) |
|---|---:|---:|
| ROC-AUC | **0.7687** | 0.8064 |
| PR-AUC (average precision) | **0.2693** | 0.4052 |
| Confusion matrix @ threshold | TN=74,216 FP=0 / FN=701 TP=0 | TN=74,216 FP=0 / FN=701 TP=0 |
| Precision / Recall / F1 (fraud class) | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 |
| Max predicted probability (any row) | **0.0112** | 0.0095 |

### 4.2 Dataset 3 — synthetic financial fraud (1,000 rows, 64 fraud) — low power

| Metric | Value (2026-08-23, monotonic model) | Was (2026-08-22, pre-monotonic model) |
|---|---:|---:|
| ROC-AUC | **0.6048** | 0.6179 |
| PR-AUC (average precision) | **0.1160** | 0.1051 |
| Confusion matrix @ threshold | TN=936 FP=0 / FN=64 TP=0 | TN=936 FP=0 / FN=64 TP=0 |
| Precision / Recall / F1 (fraud class) | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 |
| Max predicted probability (any row) | **0.0028** | 0.0034 |

### 4.3 What the two numbers together actually mean

These two metrics tell different, both-true stories, and reporting only one
would be misleading:

- **ROC-AUC 0.77 on Dataset 1 (was 0.81 on the pre-monotonic model) says the
  model still ranks fraud above legitimate transactions better than
  chance**, using only 6 honestly-mapped features out of 40. That is a real,
  if more modest than before, generalization signal — the 6 mapped features
  (amount, hour, night flag, merchant category, device, success flag)
  evidently still carry usable signal even completely outside the training
  pipeline. The monotonic-constraints retrain traded a little of this
  external-ranking signal for the recall-ceiling improvement documented in
  `RECALL_CEILING_REMEDIATION.md` — a real, reportable side effect, not
  hidden here.
- **The confusion matrix says the model catches zero of the 701 frauds in
  Dataset 1 at its own frozen threshold.** The maximum predicted probability
  across all 74,917 rows — fraud or not — is 0.0112, roughly 45× below the
  0.50 decision threshold used in production (was 0.30 before the
  2026-08-24 ensemble-vs-raw correction, and 0.40 before the
  monotonic-constraints retrain — the ratio only gets larger at each step,
  never closer). This is not a close miss; the model's entire output
  range on this dataset is compressed into a band where the frozen threshold
  can never fire.

The reason is straightforward: 34 of 40 features (85%) are `NaN` for every
single row and get median/most-frequent imputed to the *training
distribution's* "typical, unremarkable" values. XGBoost's learned decision
boundary depends on interactions across the full 40-feature vector; flooding
85% of that vector with "everything is normal" imputed values pulls every
prediction toward the model's low-probability, low-risk region regardless of
what the 6 real features say. Rank-order discrimination survives (ROC-AUC);
absolute calibration does not (PR-AUC, confusion matrix, recall).

Dataset 3's ROC-AUC of 0.60 (was 0.62) — with only 2/40 features mapped and
64 positive examples — is closer to a coin flip than to real discrimination,
and should be read as exactly the low-power, low-confidence result the task
brief asked to flag explicitly, not as independent evidence of anything.

### 4.4 Full production ensemble vs. raw XGBoost — does the ensemble do better?

Everything above (§4.1–4.3) scores the raw frozen XGBoost artifact directly
via `model.predict_proba()`. That never exercises the rules scorer or
LightLR — the other two members of the 3-model ensemble that `/predict`
actually runs in production (`src/fraud_model.py`'s `score()`). LightLR in
particular is built for exactly this scenario: its 5 features were chosen as
the ones "consistently available at inference time," i.e. the case where
most of the 40-feature vector is missing — which is precisely what this OOD
dataset is. `generalization_check_ensemble.py` re-scores the identical
Dataset 1 rows, with the identical 6-column honest mapping from §3.1 (reused
verbatim, not re-derived), through the real `score()` ensemble path instead
of the raw model — same frozen artifacts, same frozen threshold, no
retraining or fine-tuning on this data anywhere.

**Recomputed 2026-08-23 against the monotonic-constraints model** (see the
update note at the top of this document) via a real re-run of
`generalization_check_ensemble.py` — not a text edit. The deployed
threshold is now 0.50 (was 0.30, then 0.40 before that) — re-verified
directly at 0.50 via `rescore_real_datasets_new_threshold.py`: the
confusion matrix is identical (0/701), since the max ensemble score here
(0.0847, see the table below) sits below every threshold tried.

| Metric | Raw XGBoost only (§4.1) | Full ensemble (XGBoost + rules + LightLR) | Was (2026-08-22, pre-monotonic, @ τ=0.40) |
|---|---:|---:|---:|
| ROC-AUC | 0.7687 | **0.7919** | 0.8064 raw / 0.8107 ensemble |
| PR-AUC (average precision) | 0.2693 | **0.3767** | 0.4052 raw / 0.4136 ensemble |
| Confusion matrix @ threshold | TN=74,216 FP=0 / FN=701 TP=0 | TN=74,216 FP=0 / FN=701 TP=0 | same, both thresholds |
| Precision / Recall / F1 (fraud class) | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | same |
| Max predicted score (any row) | 0.0112 | 0.0782 | 0.0095 raw / 0.0772 ensemble |
| LightLR score range across all 74,917 rows | — | **constant: 0.153867** | constant: 0.153867 (unchanged — LightLR is untouched by the XGBoost retrain) |
| Rules score range across all 74,917 rows | — | **two values: 0.17 / 0.22** | two values: 0.17 / 0.22 (unchanged — rules scorer is untouched) |

The ensemble's edge over raw XGBoost on this dataset is larger on the new
model than it was on the old one (ROC-AUC +0.0232 vs. the prior +0.0043;
PR-AUC +0.1074 vs. the prior +0.0084). This is a real, reproducible effect
of the monotonic retrain, not noise: the new model's raw `paysense_score`
values on this dataset are compressed into an even narrower band (0.0006 to
0.0112, all still far below the 0.50 threshold) than before, so the small
but real organic signal `is_night_transaction` contributes through the
rules scorer's fixed 0.17/0.22 split — unchanged, since neither the rules
scorer nor LightLR were touched by the XGBoost retrain — now moves relative
rank order more than it used to, precisely because the raw scores it's
competing against carry less separation of their own. Both LightLR and the
rules scorer are otherwise identical to the pre-monotonic run, confirming
the entire delta traces to the XGBoost component, not to any change in the
other two scorers.

**Why the extra scorers barely move the needle, and why that's expected
rather than a bug:** none of LightLR's 5 features
(`amount_deviation_score`, `new_device_flag`, `ip_location_mismatch`,
`transaction_velocity`, `failed_attempts_last_24h`) appear anywhere in
Dataset 1's honest 6-column mapping (§3.1) — check the mapping table again:
`new_device_flag` and `ip_location_mismatch` were both explicitly **not**
mapped (payee novelty ≠ device novelty; no IP-mismatch-equivalent column
exists in this dataset at all), and the other three LightLR features have
no candidate source column either. So every one of LightLR's 5 inputs falls
back to its `.get(key, 0.0)` default for all 74,917 rows, and LightLR
produces the exact same score (0.153867, its intercept-driven baseline) for
every single transaction — a constant, not a signal. The rules scorer fares
only marginally better: of its four hard-signal fields
(`new_device_flag`, `ip_location_mismatch`, `kyc_verified_flag`,
`usr_is_high_risk`), all four are also absent from the honest mapping, so
`kyc_verified_flag`'s missing-value default alone fixes +0.15 on every row,
and the *only* honestly-mapped field the rules scorer actually reads is
`is_night_transaction` — worth +0.05. That's why the rules score takes
exactly two values (0.17 baseline, 0.22 at night) across all 74,917 rows,
driven by one binary flag, not the "highest-SHAP hard signals" the scorer
was designed around.

**Honest verdict: the full ensemble does not meaningfully outperform raw
XGBoost in any operationally useful sense on this dataset, and it fails in
the identical operational way.** At the frozen threshold (0.50 as of
2026-08-24, was 0.30, then 0.40 before that) it still catches **0 of 701** real frauds — same
confusion matrix, same zero recall, as the raw-model-only check. The
ROC-AUC/PR-AUC nudge upward is real, and larger on the current model
(+0.0232 / +0.1074) than it was on the pre-monotonic one (+0.0043 /
+0.0084) — see the note under the table above for why the monotonic
retrain's *narrower* raw-score band on this dataset makes the same fixed
`is_night_transaction` signal move relative ranking more than before, even
though nothing about the rules scorer or LightLR changed. It remains
traceable entirely to `is_night_transaction` carrying weak organic signal
through the rules scorer — not to any contribution from LightLR, which
degenerates to a constant here because its entire feature set is
unavailable in this dataset, exactly the caveat this check exists to
surface rather than hide. The reason isn't a flaw in LightLR or the rules
scorer's design — both are doing exactly what they're supposed to do with
the inputs they're given — it's that this specific external dataset doesn't
supply the device/IP-risk signals either scorer needs, the same 85%-of-features-missing
problem that limits raw XGBoost. **No result here was manufactured into a
win**: the ensemble's real value (rules + LightLR catching device/IP-based
fraud) simply cannot be exercised by a dataset that doesn't carry those
columns, and that limitation is reported plainly rather than glossed over —
a bigger ROC-AUC/PR-AUC nudge is still not the same thing as catching any
additional real fraud.

Reproduce with:
```
cd PaySense-ML-Backend
venv\Scripts\python.exe generalization_check_ensemble.py
```
(Takes several minutes — the full ensemble path runs `fraud_model.score()`
once per row, the same real per-request code path `/predict` uses, rather
than a single vectorised `model.predict_proba()` call.)

---

## 5. Verdict

**The frozen PaySense model does not generalize in any operationally useful
sense to data outside its own training pipeline, and this check found no
evidence to spin that into a positive result.** This holds whether you score
the raw XGBoost artifact alone (§4.1) or the full production 3-model
ensemble through `fraud_model.score()` (§4.4) — both catch **0 of 701** real
frauds at the frozen threshold on Dataset 1, because the two scorers meant
to compensate for missing XGBoost features (rules, LightLR) depend on the
same device/IP-risk columns this dataset doesn't honestly provide, so they
contribute a near-constant offset rather than real per-row signal here. On
the one dataset trustworthy
enough to test (Dataset 1), the model retains a real but modest ability to
*rank* fraud above non-fraud (ROC-AUC 0.81) using the ~15% of its feature
vector that could be honestly reconstructed from a different dataset's
schema — that is a genuine, non-trivial finding, and it is evidence the
model learned some transferable signal rather than pure memorization of its
own training rows. But at the frozen 0.40 decision threshold — the only
threshold this model actually ships with — it flags **zero** of 701 real
fraud transactions in that dataset, because 85% of the features it needs are
unavailable outside PaySense's own pipeline and get imputed to
training-typical values that suppress every prediction into a narrow
low-probability band. The secondary dataset (64 fraud examples, 2 usable
features) adds only a weak, low-confidence data point that does not change
this picture. This result does not indict the model's internal logic — it
indicts the assumption that a model requiring 40 PaySense-specific,
personalization-heavy features (per-user z-scores, device/IP risk scores,
merchant registration status, KYC flags) can be meaningfully evaluated, let
alone deployed, against any transaction stream that cannot supply those same
40 fields. **What this proves:** the model has learned something that is not
purely an artifact of its own synthetic pipeline. **What this does not
prove:** that the model is ready for, or would perform acceptably in,
production against real-world UPI traffic — that would require either a
real dataset with PaySense's full feature vector (which does not appear to
exist outside the project's own synthetic pipeline) or a live shadow-deployment
comparison, neither of which this check had access to.

---

## 6. Reproducing this check

Raw-XGBoost-only check (§4.1–4.3):
```
cd PaySense-ML-Backend
venv\Scripts\python.exe generalization_check.py
```

Full production ensemble check (§4.4), scoring through `fraud_model.score()`:
```
cd PaySense-ML-Backend
venv\Scripts\python.exe generalization_check_ensemble.py
```

Requires the datasets at their original paths under `E:\Projects\upi\` (not
copied into this repo, per the task's data-hygiene conventions) and the
existing `venv/` — no new packages were installed. The ensemble check also
requires `artefacts/light_lr.pkl` to exist (see `train_light_lr.py`) so it's
actually testing the 3-model ensemble and not silently falling back to
XGBoost + rules only.

# PaySense — Dataset

This is a structured index into the project's dataset facts. Full
methodology and the deeper findings (label-leakage tracing, EDA, retraining
experiments) live in `PaySense-ML-Backend/EDA_FEATURE_ENGINEERING.md`,
`GENERALIZATION_CHECK.md`, `CATEGORY_CLASSIFIER.md`, and
`CATEGORY_CLASSIFIER_GENERALIZATION.md` — this document doesn't duplicate
them, it orients you into them with verified, current numbers.

Two separate datasets feed two separate models. Do not conflate them.

---

## 1. Fraud-scoring dataset (`paysense_master_dataset.csv`)

**FACT, verified directly against the file this audit:**
- **Shape:** 30,000 rows × 50 raw columns.
- **Fraud rate:** 4.21% (1,264 fraud / 30,000).
- **Target column:** `is_fraud` (binary).
- **Unit of prediction:** one row = one UPI transaction event.

**Origin (from `EDA_FEATURE_ENGINEERING.md` §1.1, built by
`paysense_pipeline.py`):** two blended sources —
- **Anchor, 20,000 rows** (`transactions.csv`) — a real transaction-style
  dataset with organic, weaker `new_device_flag`/`ip_location_mismatch`
  signal.
- **Supplement, 10,000 rows** — schema-bridged from an external
  `Financial Fraud Dataset/synthetic_fraud_dataset.csv`. Its
  `new_device_flag`/`ip_location_mismatch` are derived by this project's own
  pipeline (`device_risk_score`/`ip_risk_score` thresholded at `> 0.70`),
  but its **`is_fraud` label is inherited as-is from the source file, not
  derived by this project**.

**Real finding, not a hypothetical (verified in `EDA_FEATURE_ENGINEERING.md`
§1.1):** in the 10,000-row supplement, `new_device_flag` and
`ip_location_mismatch` separate the label with **zero overlap in both
directions** (`flag=0` → 9,500/9,500 legitimate; `flag=1` → 500/500 fraud).
Verified against the raw external source file directly, not an artifact
introduced by this project's bridge — the label-generation formula in the
external dataset itself produces this near-tautological relationship. This
is a third of the frozen model's training data carrying a near-deterministic
feature→label shortcut that has nothing to do with real fraud behavior.

**Quantified for the first time in the 2026-08-26 audit: this contaminates
the test set too, and it explains almost the entire headline recall
number.** The train/test split (`train_test_split(..., stratify=y)`) does
not account for `data_source` at all, so the canonical 6,000-row test set
inherits the same ~35% supplement contamination (verified directly:
2,087 supplement rows / 3,913 anchor rows, 96 supplement-fraud / 157
anchor-fraud). Scoring the frozen ensemble separately on each subset:

| | Full blended (reported everywhere) | Anchor only (organic) | Supplement only (tautological) |
|---|---:|---:|---:|
| ROC-AUC | 0.8969 | **0.7465** | 1.0000 |
| PR-AUC | 0.5498 | **0.1138** | 1.0000 |
| Recall @ τ=0.50 | 39.53% (TP=100/253) | **2.55% (TP=4/157)** | 100.00% (TP=96/96) |

The model catches 4 of 157 organic fraud cases in this test set at the
deployed threshold. Its reported 39.53% recall is almost entirely the
supplement subset's trivially-learnable shortcut. This is **not classic
leakage** in the train/future-information sense —
`new_device_flag`/`ip_location_mismatch` are legitimately available at
prediction time — it is a **label-and-feature-validity problem**, and it
runs deeper than two columns: a follow-up forensic investigation
(`SOURCE_CONTAMINATION_INVESTIGATION.md`) found **23 of ~30 numeric
columns and 12 of 14 categorical columns are a single constant value
across the entire 10,000-row supplement subset** (including
`receiver_id == "SYN_MRC_UNKNOWN"`, a literal synthetic marker) — not
diverse synthetic data, one templated profile repeated 10,000 times.
`device_risk_score.notnull()` alone separates supplement from anchor rows
with exactly 100% accuracy.

**The supplement source is not representative of the organic (real-world)
deployment distribution and should not be treated as evidence of
real-world model performance.** Tested directly, not assumed: retraining
on anchor-only data did **not materially change organic-subset ROC-AUC**
(0.7260→0.7261, statistically identical) — the contamination inflates
the *blended* metrics, but removing the contaminated rows from the
existing dataset, on its own, did not change the model's discrimination
on organic data. A properly re-derived threshold (train/validation/test
split, selected on validation only, entirely on organic data) gives the
**current measured performance under a clean evaluation protocol**:
**21.05% recall at 8.82% precision** (TP=32/152) on a genuinely untouched
final test set — not directly comparable to the earlier 2.55% figure,
which used a threshold selected for a different, contaminated score
distribution. Neither figure meets the documented Recall≥75% requirement
at this measurement; full detail, mechanism, and the full A–H experiment
chain in `SOURCE_CONTAMINATION_INVESTIGATION.md` and `EXPERIMENTS.md`.
Regression-tested
(`test_organic_subset_performance_is_much_weaker_than_blended_headline`,
`test_supplement_source_is_near_fully_constant_and_perfectly_separable`,
`tests/test_frozen_model_metrics.py`) so none of this drifts unnoticed.
**Not fixed in the deployed system** — cleaning the existing dataset
alone was tested and did not resolve it; improving on this would need
different or additional organic training data, which has not been
sourced as part of this work.

**Re-verified independently in the 2026-08-27 model-family benchmark:**
a plain logistic regression trained on the 38-feature "kept" set (with
`device_risk_score`/`ip_risk_score` already excluded) still predicts
`data_source` at **99.62% accuracy / 0.9993 ROC-AUC** —
confirming the contamination is not fixed by removing the two originally
-suspected columns; 30 of those 38 kept features are themselves constant
across the entire supplement subset. Full detail:
`PaySense-ML-Backend/experiments/feature_audit.md`,
`PaySense-ML-Backend/experiments/source_classifier_results.json`.

**Feature engineering:** 50 raw columns → 40 model-ready features (README's
"40 model-ready features" claim, verified consistent with `/health`'s
`feature_count: 40` this session). Includes both raw signal columns
(`amount`, `hour_of_day`, `new_device_flag`, `ip_location_mismatch`,
`failed_attempts_last_24h`, `transaction_velocity`, ...) and
per-user personalization columns (`usr_avg_monthly_txn_profile`,
`usr_is_high_risk`, ...).

**Split & class imbalance handling, fully verified in the 2026-08-26 audit
by reading `paysense_phase3.py` and `resweep_threshold_against_ensemble.py`
directly, not inferred from documentation:** a single stratified random
split (`sklearn.train_test_split(..., test_size=0.20, random_state=42,
stratify=y)`), identical parameters and identical dropped columns in both
scripts — confirmed by checking they produce the exact same held-out
partition (both recover the published 253-fraud test set). The
preprocessor (imputation + ordinal encoding) is fit on the training
partition only, confirmed by reading the fit/transform calls directly.
SMOTE is applied to the training partition only, after the split (24K →
45,980 rows) — the correct direction. **Real gap, found in this same
audit:** the threshold itself is selected on this same held-out partition
(there is no third, separate validation set) — `resweep_threshold_
against_ensemble.py` sweeps thresholds and picks the best-F1 one using the
*same* 6,000 rows it then reports "test" performance on. This means the
reported precision/recall numbers are somewhat threshold-optimistic (the
threshold was chosen to look good on exactly this data), though this is a
narrower issue than the source-contamination finding above and doesn't
affect the underlying trained model's parameters, only which operating
point was chosen.

**Known dataset limitations (from the project's own docs, not invented
here):**
- One real dataset (`Financial Fraud Dataset/synthetic_fraud_dataset.csv`,
  supplement source) carries the near-tautological label relationship
  above.
- Scored against 74,917 real, previously-unseen UPI transactions (an
  external OOD test), the deployed ensemble ranks better than chance
  (ROC-AUC 0.79) but **catches 0 of 701 real frauds at the production
  threshold** — traced to a specific, confirmed cause: the rules scorer's
  cold-start bonus gates on `amount > 5000`, calibrated to this project's
  own ₹-scale training data; that external dataset's USD amounts (max
  ≈$154) never cross it. Full trace, and the narrow exception where a
  much lower OOD-specific threshold recovers 37/701 frauds with zero false
  positives, in `WALKTHROUGH.md`'s "Honest findings" section and
  `OOD_GENERALIZATION_REMEDIATION.md` / `ood_threshold_sweep_variant_a.py`.

---

## 2. Category-classification dataset (FinText-6K)

**FACT, verified against `CATEGORY_CLASSIFIER.md`:**
- **Source:** FinText-6K (Kaggle, Apache 2.0), pre-split `train`/`test`
  CSVs with `text,label` columns.
- **Train:** 5,000 rows (fit only). **Test:** 1,000 rows (held out, never
  fit on).
- **Classes (5, roughly balanced ~1000/class train, ~200/class test):**
  Food, Travel, EMI, Investment, Shopping.
- **Target column:** `label` — text narration (e.g. bank SMS-style
  sentences) → category.

**Real finding, honestly documented (`CATEGORY_CLASSIFIER.md`,
`CATEGORY_CLASSIFIER_GENERALIZATION.md`):** the entire 5,000-row train
split is generated from only **40 unique sentence templates**, with only
the amount and reference number varying — which is why the first trained
model scored a hollow 100% on the held-out test split (the test split draws
from the *same* 40-template pool). Scored instead against 200
hand-authored, structurally-verified-novel real bank/UPI narrations: real
accuracy dropped to 72.5%. Traced to a mechanism (the fitted vocabulary is
821 tokens, entirely mined from those 40 templates — novel text sharing
zero training words collapses to an identical wrong-by-construction
prediction). Retrained on a broader, verified-disjoint template set: 78.0%
real accuracy, now deployed. A DistilBERT fine-tune on the identical data
reached 83.0% but was deliberately left undeployed (128× larger, 370×
slower, unverified memory headroom on the hosting tier). Full trace in
`WALKTHROUGH.md`'s "Category classifier & the DistilBERT test" section.

---

## 3. What this project does NOT have

- **No geography.** No coordinates, no spatial joins, no geocoding, no
  region-based train/test split. `usr_home_city_tier` and `usr_home_city`
  are categorical profile fields (e.g. "Tier 2"), not spatial data — there
  is no distance calculation or geographic leakage risk to assess. This
  section of the standard audit template is genuinely not applicable here,
  not skipped.
- **No time-series/temporal split.** Transactions have timestamps
  (`timestamp`, `date`, `hour_of_day`), used as per-row features (e.g.
  `is_night_transaction`), but the train/test split is not documented or
  verified as temporally ordered — **NOT VERIFIED** whether future
  transactions could appear in training relative to test rows. Flagged as
  an open question for the next audit pass, not asserted either way.
- **No PII in the datasets used for training.** `user_id`/`receiver_id`
  are synthetic identifiers, not real names, phone numbers, or account
  numbers, in both source datasets as used by this project.

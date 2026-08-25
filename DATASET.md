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
Documented, not silently absorbed — see the full writeup for how this
informed the retrain decisions in `RECALL_CEILING_REMEDIATION.md`.

**Feature engineering:** 50 raw columns → 40 model-ready features (README's
"40 model-ready features" claim, verified consistent with `/health`'s
`feature_count: 40` this session). Includes both raw signal columns
(`amount`, `hour_of_day`, `new_device_flag`, `ip_location_mismatch`,
`failed_attempts_last_24h`, `transaction_velocity`, ...) and
per-user personalization columns (`usr_avg_monthly_txn_profile`,
`usr_is_high_risk`, ...).

**Split & class imbalance handling:** SMOTE applied to the training
partition only (24K → 45,980 rows after oversampling) — the correct
direction (never oversample before splitting, or synthetic neighbors of a
test-set row can leak into training). **NOT independently re-verified this
audit** which exact script performs the split point vs. the SMOTE call, or
confirmed the split is not itself contaminated some other way — this is
the single highest-value thing for a future audit pass to re-derive from
`paysense_phase3.py` directly rather than trust the existing claim.

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

# PaySense — Experiments

Chronological record of the project's real methodology work — including
the corrections. A wrong result caught and fixed is useful engineering
history; this document doesn't hide that in favor of a cleaner-looking
narrative. Full technical detail for each experiment lives in its own
`.md` file in `PaySense-ML-Backend/`, linked below — this document is the
index and the throughline connecting them.

---

## Category classifier contamination (2026-08-23/24)

**Hypothesis:** the category classifier's reported 100% test accuracy
reflects genuine generalization to real-world bank/UPI text.

**Method:** inspected the training data (FinText-6K) and its relationship
to the held-out test split.

**Result:** the entire 5,000-row training split is generated from only 40
unique sentence templates, and the test split draws from the same fixed
template pool — a perfect score measures template memorization, not
generalization. Built a 200-row, hand-authored, structurally-verified-
novel test set instead: real accuracy measured at **72.5%**, not 100%.

**Interpretation:** a perfect score on a real-world text task is itself a
reason to check the evaluation methodology before trusting the number.

**Conclusion:** retrained on FinText-6K blended with 8,000 hand-built
template rows, verified programmatically disjoint from the evaluation
set. New accuracy: **78.0%** (deployed). A first retrain attempt (v2) was
discarded after its own templates were found to be the eval set's
sentence skeletons with only the merchant name swapped — a real
contamination bug in the fix itself, caught before shipping.

Full detail: `PaySense-ML-Backend/CATEGORY_CLASSIFIER.md`,
`CATEGORY_CLASSIFIER_GENERALIZATION.md`.

---

## DistilBERT architecture ablation (2026-08-23)

**Hypothesis:** the 78.0% accuracy ceiling from the experiment above is a
data-diversity limit, not an architecture limit.

**Method:** fine-tuned `distilbert-base-uncased` on the *exact same*
training data as the 78.0% retrain — same rows, same evaluation set, only
the model architecture changed.

**Result:** 83.0% accuracy, 82.0% correct-and-confident (vs. 78.0%/70.5%
for the deployed TF-IDF+LinearSVC model) — a larger gain than the data-
diversity retrain produced, on identical data.

**Interpretation:** real evidence the remaining gap is at least partly
architectural, not purely a data-diversity problem.

**Conclusion:** not deployed. Measured trade-off: 267.8MB vs ~2.1MB
(128× larger), 369ms vs sub-millisecond inference (370× slower), and no
way to verify the live hosting tier's memory headroom for
`torch`+`transformers` alongside the existing model stack. An
out-of-memory crash on the live service would be a worse outcome than
staying on the faster, less accurate classifier.

Full detail: `PaySense-ML-Backend/CATEGORY_CLASSIFIER_V3_ATTEMPT.md`.

---

## The fraud-model investigation: Experiments A–H (2026-08-24 to 2026-08-27)

### Experiment A — Initial blended evaluation

**Hypothesis:** the deployed 3-scorer ensemble's reported metrics
(ROC-AUC 0.8969, PR-AUC 0.5498, 91.74% precision / 39.53% recall at
τ=0.50) reflect its real-world fraud-detection performance.

**Method:** standard held-out test evaluation on
`paysense_master_dataset.csv`'s canonical 20%-test split (30,000 rows
total, both "anchor" and "supplement" sources blended together, no
source-aware stratification).

**Result:** the headline numbers above, computed and reported as the
project's key results for most of its history.

**Interpretation, at the time:** these looked like strong, usable
numbers for a UPI fraud classifier.

**Conclusion:** correct arithmetic, but — established later by
Experiments C onward — computed on a test set with a significant, then-
undetected data-quality problem. Superseded, not retracted as wrong math;
the numbers are real outputs of the pipeline as it existed, just not
representative of organic (real-world-style) performance.

### Experiment B — Scoring-path correction

**Hypothesis:** the model's documented metrics come from the same code
path that actually serves predictions in production.

**Method:** cross-checked which function the metrics-reporting scripts
called (`model.predict_proba()` directly) against which function the live
`/predict` endpoint calls (`src.fraud_model.score()`, a weighted 3-scorer
ensemble).

**Result:** they were different. Re-scoring the canonical test set
through the real ensemble instead of raw XGBoost, at the
then-deployed threshold (0.30): precision dropped from a claimed 86.44%
to a real 40.81%.

**Interpretation:** the rules scorer's always-on additive score had never
been jointly calibrated against the same threshold XGBoost's own sweep
had picked, so rows XGBoost alone would score below 0.30 still crossed it
once blended in.

**Conclusion:** re-ran threshold selection against the real ensemble;
threshold moved to 0.50. Regression test added
(`test_ensemble_differs_materially_from_raw_xgboost_at_threshold`) to
prevent this exact "measuring the wrong component" mistake recurring
silently. Full detail: `PaySense-ML-Backend/EDA_FEATURE_ENGINEERING.md` §4.5.

### Experiment C — Supplement/source contamination discovery

**Hypothesis:** the training data's known label-generation quirk (see
`EDA_FEATURE_ENGINEERING.md` §1.1: `new_device_flag`/
`ip_location_mismatch` separate the "supplement" source's fraud label
with zero overlap) is confined to two columns and doesn't affect the
test set.

**Method:** checked whether the train/test split accounts for
`data_source`, and audited every column in the dataset for source-
dependent patterns, not just the two originally suspected.

**Result:** the split does not account for source — the canonical test
set inherits the same ~35% contamination proportionally. Far beyond two
columns: **23 of ~30 numeric columns and 12 of 14 categorical columns are
a single constant value across the entire 10,000-row supplement subset**,
including `receiver_id == "SYN_MRC_UNKNOWN"` (a literal synthetic
marker).

**Interpretation:** the supplement source is not diverse synthetic data —
it is one templated synthetic profile repeated 10,000 times, with only
`amount`, `hour_of_day`, `transaction_type` (2 values), and the two risk-
score columns actually varying row to row.

**Conclusion:** source-stratified re-scoring of the deployed model shows
the two subsets behave completely differently (organic ROC-AUC 0.7465 /
PR-AUC 0.1138 / recall 2.55%, vs. supplement ROC-AUC 1.0 / PR-AUC 1.0 /
recall 100%) — the blended headline numbers are dominated by the
supplement subset. Full detail:
`PaySense-ML-Backend/SOURCE_CONTAMINATION_INVESTIGATION.md` §1.

### Experiment D — Source-only separability test

**Hypothesis:** the organic/supplement distinction found in Experiment C
is a real but modest statistical difference, not a trivial shortcut.

**Method:** built the simplest possible classifier —
`device_risk_score.notnull()` — and measured how well it alone predicts
`data_source`.

**Result:** **100.0000% accuracy.** A single column perfectly separates
the two sources.

**Interpretation:** any model with access to this column (or any of the
~20 other near-constant supplement-only columns) can trivially identify
which source a row came from, independent of any fraud reasoning.

**Conclusion:** this is not a subtle statistical artifact — it's a
structural property of how the two data sources were combined. Full
detail: `PaySense-ML-Backend/SOURCE_CONTAMINATION_INVESTIGATION.md` §1.

### Experiment E — Organic-only retraining

**Hypothesis:** retraining the fraud model on anchor-only (organic) data,
excluding the contaminated supplement rows, will improve the model's
ability to detect organic fraud.

**Method:** retrained XGBoost with identical hyperparameters and
monotonic constraints to the deployed model, on anchor-only training rows
(16,087 rows, 606 fraud), with `device_risk_score`/`ip_risk_score`
explicitly excluded. Evaluated on the same held-out test partition used
throughout this project, sliced by source.

**Result:** organic-subset ROC-AUC was 0.7260 before, 0.7261 after —
statistically identical. PR-AUC likewise essentially unchanged (0.0971
vs 0.0993).

**Interpretation:** the hypothesis was wrong. The contamination inflates
blended metrics, but removing the contaminated rows from the existing
dataset did not change the model's discrimination on organic data —
XGBoost appears to solve the (trivially separable) supplement subset
largely independently of how it learns organic patterns, so training on
one doesn't meaningfully cost or benefit the other. (A further,
unexpected observation: the anchor-only-trained model, having never seen
a single supplement row or the two risk-score columns, still scored a
perfect 1.0/1.0 ROC-AUC/PR-AUC on the supplement-only test subset — the
~20 other near-constant columns were sufficient on their own.)

**Conclusion:** cleaning the existing dataset, on its own, is not a fix
for organic-fraud-detection capability. This does not establish that
better organic performance is unreachable with different or additional
data — only that this specific remediation (drop the contaminated rows)
did not produce it. Full detail:
`PaySense-ML-Backend/investigate_source_safe_retrain.py`,
`SOURCE_CONTAMINATION_INVESTIGATION.md` §2.

### Experiment F — Clean 60/20/20 evaluation

**Hypothesis:** a properly-structured train/validation/test split, on
organic-only data, is needed before any further claim about this model's
real recall/precision can be trusted — every prior evaluation in this
project selected its threshold on the same partition it reported
performance on.

**Method:** split the 20,000-row anchor-only pool into train (60%,
12,000 rows, 458 fraud), validation (20%, 4,000 rows, 153 fraud), and
test (20%, 4,000 rows, 152 fraud), stratified. Trained XGBoost (same
hyperparameters/constraints) on the training partition only.

**Result:** validation ROC-AUC 0.6506, PR-AUC 0.0727.

**Interpretation:** a real, separate validation set now exists for this
model for the first time in the project's history.

**Conclusion:** proceed to threshold selection on this validation set
(Experiment G). Full detail:
`PaySense-ML-Backend/investigate_organic_only_threshold.py`.

### Experiment G — Validation-only threshold selection

**Hypothesis:** some threshold in the 0.05–0.95 range satisfies the
documented Recall≥75% AND Precision≥50% business constraint on the
validation set.

**Method:** swept thresholds 0.05–0.95 (step 0.05) against validation
predictions only; selected the constraint-satisfying threshold with
maximum F1, or the unconditional max-F1 threshold as a fallback if none
satisfied both constraints.

**Result:** no threshold in the swept range satisfied both constraints
simultaneously on the validation set. Frozen threshold (max-F1 fallback):
**0.10**.

**Interpretation:** consistent with every prior sweep on this project
(raw XGBoost, real ensemble, and now organic-only data) — the business
constraint has not been met at any threshold tested so far.

**Conclusion:** freeze τ=0.10, proceed to a single, final evaluation on
the untouched test set (Experiment H). Full detail:
`SOURCE_CONTAMINATION_INVESTIGATION.md` §3.

### Experiment H — Final untouched test

**Hypothesis:** none — this is a single, final measurement, performed
exactly once, on data untouched by any prior step in Experiments F–G.

**Method:** applied the frozen threshold (0.10, selected in Experiment G)
to the held-out test partition (4,000 rows, 152 fraud) for the first and
only time.

**Result:**

```
ROC-AUC=0.7050  PR-AUC=0.0945
Precision=8.82%  Recall=21.05%  (TP=32, FP=331, FN=120, TN=3517)
Recall>=75% AND Precision>=50%: not met at this measurement
```

**Interpretation:** this is the current measured performance of this
model under a clean evaluation protocol — not a theoretical maximum, not
a claim about what any future model or dataset could achieve. It differs
from the earlier 2.55% organic-recall figure because that figure used a
threshold (0.50) selected for a different, contaminated score
distribution; the two are not directly comparable.

**Conclusion:** the current model and available dataset do not meet the
stated Recall≥75% requirement under this clean organic evaluation
protocol. Regression-tested implicitly via
`test_organic_subset_performance_is_much_weaker_than_blended_headline`
and `test_supplement_source_is_near_fully_constant_and_perfectly_
separable` (`tests/test_frozen_model_metrics.py`), which protect the
underlying source-contamination finding from silently drifting. The
deployed model and threshold are unchanged by this investigation — this
document reports what was found, not a change made to production.

---

## Model-family benchmark (2026-08-27)

**Question:** does an alternative model family (RandomForest, LightGBM,
CatBoost) materially improve fraud detection on the clean organic
evaluation established in Experiments F–H, over the currently deployed
XGBoost model?

**Feature audit performed first:** every one of the 48 non-target raw
columns was classified VALID/SUSPICIOUS/INVALID before any model was
trained (`PaySense-ML-Backend/experiments/feature_audit.md`). 30 of the
38 kept features are constant across the entire supplement subset —
kept anyway because they vary naturally in organic data, not because the
contamination was overlooked. A source classifier trained on the kept
feature set with `device_risk_score`/`ip_risk_score` already excluded
still separated organic from supplement rows at **99.62% accuracy /
0.9993 ROC-AUC** — direct, independent confirmation that removing the
two originally-suspected columns comes nowhere close to fixing the
underlying contamination (systemic across ~30 columns, not a two-column
problem). This is a diagnostic finding only, not a fraud model.

**Method:** all four models trained on the identical anchor-only
60/20/20 split from Experiment F (`random_state=42`), preprocessing fit
on train only, threshold selected on validation only (same sweep/
selection rule as Experiment G), evaluated on the same untouched test
set exactly once per model. RandomForest and LightGBM got a small
validation-selected hyperparameter grid (4 configs each); CatBoost got
the same grid plus native categorical handling (no ordinal encoding) and
`auto_class_weights='Balanced'` instead of SMOTE (SMOTE is not natural
on raw categorical text without SMOTENC — a disclosed pipeline
difference, not an inconsistency in the split or test set). Text-based
model families (TF-IDF+LogReg, linear SVM, text-only, hybrid, DistilBERT)
were **not implemented** — `paysense_master_dataset.csv` has no text/SMS
field at all, verified directly against its 50 raw columns; there is no
text signal for those model families to use on this task. (DistilBERT
already exists in this project for the unrelated, text-bearing category-
classification task — see above.)

**Result:**

| Model | Test ROC-AUC | Test PR-AUC | Precision | Recall | 5-fold CV PR-AUC (train+val only) |
|---|---:|---:|---:|---:|---:|
| XGBoost (current, unmodified) | 0.7050 | 0.0945 | 8.82% | 21.05% | 0.0827 ± 0.0053 |
| RandomForest | 0.7210 | 0.0860 | 8.61% | 27.63% | 0.0963 ± 0.0053 |
| LightGBM | 0.6904 | 0.0980 | 10.34% | 13.82% | 0.0806 ± 0.0048 |
| CatBoost | 0.6628 | 0.0785 | 12.56% | 18.42% | not run (see below) |

None of the four satisfies Recall≥75% AND Precision≥50% at its own
validation-selected threshold. RandomForest and CatBoost *can* reach
Recall≥75% by lowering the threshold far enough (validation threshold
0.05), but only at 6.06% and 4.80% test precision respectively — nowhere
near the 50% floor. Full table, PR-curve comparison, and per-model
feature-importance breakdown:
`PaySense-ML-Backend/experiments/model_benchmark.md`,
`PaySense-ML-Backend/experiments/plots/precision_recall_comparison.png`.

CatBoost's CV was **not run**: its grid-search fits took roughly an hour
each in this environment (vs. single-digit minutes for the other three
model families), and it already had the lowest validation PR-AUC of the
four — a disclosed compute-budget decision, not a silent gap. Its
single-split test result and grid search are complete and reported
above.

**Interpretation:** the four models' test PR-AUC values (0.0785–0.0980)
span a narrower range than the 5-fold cross-validation standard
deviation measured for the three that were cross-validated (~0.005–0.008
each). LightGBM's point estimate is highest, but by less than one CV
standard deviation over XGBoost — not a distinguishable improvement,
just where four noisy estimates happened to land. Feature importance is
also inconsistent across model families (tree-split-based importances
for XGBoost/RandomForest highlight `is_night_transaction`/
`transaction_velocity`/`new_device_flag`; gain-based importances for
LightGBM/CatBoost highlight `balance_after_transaction`/
`usr_account_age_days`/`amount` instead) — consistent with a dataset
that has real but weak signal, not a strong pattern any of these model
families is failing to find.

**Conclusion:** no evaluated model family currently satisfies the
business requirement, and none materially improves on the deployed
XGBoost model's organic-data performance — the differences observed are
not distinguishable from cross-validation noise. Recommendation:
**keep the current model** — none of the alternatives justify a
replacement, and the underlying gap to the Recall≥75%/Precision≥50%
requirement is a data problem (established in Experiments E and above),
not a model-family problem this benchmark's results would suggest fixing
by switching architectures. Regression-tested in
`tests/test_model_benchmark.py`. Full detail:
`PaySense-ML-Backend/experiments/run_model_benchmark.py`,
`PaySense-ML-Backend/experiments/model_benchmark.md`,
`PaySense-ML-Backend/experiments/feature_audit.md`.

---

## Source-domain forensics (2026-08-27)

**Question:** the model benchmark's source classifier separated organic
from supplement rows at 99.62% accuracy using the 38 kept features, even
with the two previously-known leak columns already excluded. Why? Is
this hidden leakage, a synthetic-generation artifact, real domain shift,
or a mixture?

**Method:** classified every one of the 38 kept features into
VALID_SIGNAL / DOMAIN_SHIFT / COLLECTION_ARTIFACT / SYNTHETIC_ARTIFACT /
LEAKAGE / UNKNOWN, using computed statistics rather than assumption for
each: constant/near-constant analysis per source, missingness per
source, a standardized effect size (Cohen's d for numeric, Cramér's V
for categorical) between organic and supplement, and — critically — a
SEPARATE effect size measuring each feature's association with
`is_fraud` *within anchor-only (organic) rows*, so a feature that is
constant in supplement isn't automatically judged worthless for fraud
modelling. `legitimate_at_inference` was checked directly against
`main.py`'s real `/predict` request schema (every one of the 38 is a
literal field there — 37 required, `mrc_rating` optional) rather than
assumed.

**Result:**

- **30 of 38 features: SYNTHETIC_ARTIFACT.** Constant in the supplement
  source specifically — the supplement's own generation process simply
  didn't vary these fields. This mechanically explains most of the
  99.62% separability.
- **5 of 38 features: DOMAIN_SHIFT** — vary naturally in *both* sources,
  but with a large, real distributional difference: `mrc_category`
  (Cramér's V=0.848 — organic spans 10 merchant categories dominated by
  P2P Transfer, supplement is dominated by Food/Clothing/Electronics,
  categories neither source shares), `transaction_type` (Cramér's
  V=0.676 — supplement introduces an "ATM" transaction type organic data
  never has), `amount` (Cohen's d=0.818 — organic mean ₹876.85 vs.
  supplement mean ₹178.14, a ~5× scale difference), `is_night_transaction`
  (d=0.499), `hour_of_day` (d=0.436).
- **3 of 38 features: VALID_SIGNAL** — `new_device_flag`,
  `ip_location_mismatch`, `transaction_frequency_score` show no material
  source-separability at all.
- **0 features: COLLECTION_ARTIFACT or LEAKAGE.** No hidden hard-leakage
  or missingness-pattern-based source tell was found among these 38
  (the two real leak columns, `device_risk_score`/`ip_risk_score`, were
  already excluded before this analysis — see `SOURCE_CONTAMINATION_
  INVESTIGATION.md`).
- Of the 30 SYNTHETIC_ARTIFACT features, **5 also show real organic
  fraud-signal** (`transaction_velocity`, `failed_attempts_last_24h`,
  `amount_deviation_score` — the three monotonic-constraint behavioral
  features — plus `kyc_verified_flag`, `usr_is_high_risk`) and were kept
  rather than removed; removing genuinely useful signal to fix a
  dataset-generation problem elsewhere would be the wrong trade.
- **Feature-set ablation on the frozen organic 60/20/20 split:** SET A
  (all 38) reproduces the established baseline exactly (ROC-AUC 0.7050,
  PR-AUC 0.0945). SET B (13 features — the 30 pure artifacts removed,
  the 8 naturally-varying features plus the 5 fraud-signal-bearing
  artifact-tainted ones kept) scores ROC-AUC 0.7018, PR-AUC 0.0867 —
  **not better**, marginally lower on both threshold-independent
  metrics. Its validation-selected threshold happened to land much lower
  (0.05 vs. 0.10), producing a very different-looking operating point
  (65.79% recall at 6.78% precision) — a different point on a
  similar-quality curve, not a real improvement; reported here
  specifically so it isn't misread as one. SET C (deployment-available-
  only) is **bit-for-bit identical to SET A** — verified, not assumed:
  every one of the 38 kept features is already a required/optional field
  in the real `/predict` schema, so there is no stricter
  deployment-availability boundary to draw here.
- **Source classifier retrained on SET B (post-cleaning):** accuracy
  drops from 99.62%/0.9993 ROC-AUC to **90.73%/0.9624 ROC-AUC** —
  substantially lower, but still very high. Removing the confirmed pure
  artifacts does not come close to eliminating source separability.

**Interpretation:** the evidence supports a **combination of causes**,
not a single tidy answer. The dominant mechanism (30/38 features) is a
synthetic-generation artifact — the supplement source is a templated
profile that fails to vary most fields, confirming and quantifying the
2026-08-27 model-benchmark's finding further. But a second, independent
cause is also real and evidenced: 5 of the 8 naturally-varying features
show large, genuine distributional shift, consistent with the two
sources representing different transaction populations (smaller
amounts, different hours, non-overlapping merchant-category and
transaction-type vocabularies) rather than one dataset simply being a
noisier version of the other. No evidence of hidden leakage was found —
every feature is legitimately available at real inference time, checked
directly. Critically, removing the confirmed artifacts does **not**
improve fraud-detection performance on organic data (PR-AUC 0.0867 vs.
0.0945, marginally lower, not higher) — consistent with, and extending,
the earlier finding that cleaning this dataset does not by itself fix
organic fraud-detection capability (Experiment E). 99.62% source-
separability is not, on its own, proof of fraud-relevant leakage — three
of the most fraud-relevant features (`new_device_flag`,
`ip_location_mismatch`, `transaction_frequency_score`) show no source
separability at all, and the features that do separate strongly are
mostly ones a real fraud model would legitimately need regardless of
this dataset's specific generation quirks.

**Conclusion:** the remaining evidence indicates the dominant limitation
is representation/domain coverage — a real, severe difference between
the two data sources' transaction populations, compounded by a
synthetic-generation artifact in the supplement source — rather than a
model-family choice (Experiments A–H, model-family benchmark above) or
a small, fixable set of leaking columns (this analysis found none).
Regression-tested in `tests/test_source_forensics.py`. Full detail and
every feature's individual classification:
`PaySense-ML-Backend/experiments/source_forensics/feature_validity_audit.md`,
`source_feature_importance.csv`, `distribution_shift.csv`,
`fraud_feature_ablation.csv`, `source_classifier_before_after.json`.

# PaySense — Experiments

Chronological record of the project's real methodology work — including
the corrections. A wrong result caught and fixed is useful engineering
history; this document doesn't hide that in favor of a cleaner-looking
narrative. Full technical detail for each experiment lives in its own
`.md` file in `PaySense-ML-Backend/`, linked below — this document is the
index and the throughline connecting them.

---

## Experiment 1 — Ensemble scoring correction (2026-08-24)

**Initial state:** every fraud-model metric this project had ever
reported — in `README.md`, in `paysense_report.tex` — was computed by
calling the frozen XGBoost model's `predict_proba()` directly.

**Suspicion raised:** while writing up feature-engineering work
(`EDA_FEATURE_ENGINEERING.md` §4.5), a discrepancy surfaced between the
model's documented behavior and what the live `/predict` endpoint
actually does — `/predict` calls `src.fraud_model.score()`, a 3-scorer
ensemble (XGBoost weighted 0.60, LightLR 0.25, a hand-tuned rules scorer
0.15), never the raw model directly.

**Investigation:** `resweep_threshold_against_ensemble.py` re-scored the
canonical held-out test set through the real ensemble instead of raw
XGBoost, at the threshold that was deployed at the time (0.30).

**Result:**

```
Raw XGBoost only  @ τ=0.30: precision=86.44%  recall=40.32%
Real ensemble     @ τ=0.30: precision=40.81%  recall=51.78%
```

Precision more than halved once measured correctly. Root cause: the
rules scorer's always-on additive score was never jointly calibrated
against the same threshold XGBoost's own sweep had picked, so rows
XGBoost alone would score below 0.30 still crossed it once the rules/
LightLR contributions were blended in.

**Fix:** re-ran the same threshold-selection methodology against the
real ensemble, swept 0.05–0.95. F1 peaks exactly at τ=0.50. New deployed
threshold: 0.50 (was 0.30).

**Regression test added:** `test_ensemble_differs_materially_from_raw_
xgboost_at_threshold` (`tests/test_frozen_model_metrics.py`) — asserts
raw XGBoost and the real ensemble stay materially different at the
deployed threshold, so this exact class of "measuring the wrong
component" mistake can't recur silently.

Full detail: `PaySense-ML-Backend/EDA_FEATURE_ENGINEERING.md` §4.5.

---

## Experiment 2 — Category classification contamination (2026-08-23/24)

**Initial state:** the category classifier (TF-IDF + LinearSVC, trained
on FinText-6K) reported 100% accuracy on its held-out test split.

**Suspicion raised:** a perfect score on a real-world text-classification
task is itself a reason to look closer, not a reason to stop looking.

**Investigation:** FinText-6K's entire 5,000-row training split turned
out to be generated from only 40 unique sentence templates — and the test
split draws from the same fixed template pool, so a perfect score
measures template memorization, not generalization.

**First correction attempt (v2), later retracted:** a broader synthetic
template set was generated, but its own templates turned out to be the
evaluation set's sentence skeletons with only the merchant name swapped —
a real contamination bug in the fix itself, caught and discarded rather
than shipped.

**Second, verified correction (v3/v4):** built a 200-row, hand-authored,
structurally-verified-novel test set (real HDFC/SBI/ICICI/Axis SMS and
GPay/PhonePe/Paytm formats, checked disjoint from every training
template). Retrained on FinText-6K blended with 8,000 verified-disjoint
template rows.

**Result:**

```
Original (v1), against its own contaminated test split: 100.0% accuracy
Original (v1), against the 200-row novel test set:       72.5% accuracy
Retrained (v3/v4), against the 200-row novel test set:   78.0% accuracy
                    (70.5% both correct and confident enough to deploy)
```

**Regression tests added:** `test_category_training_v2_disjointness.py`,
`test_category_training_v3_disjointness.py` — verify the training
templates are programmatically disjoint from the evaluation set, so the
exact mechanism behind the original hollow 100% can't recur unnoticed.

Full detail: `PaySense-ML-Backend/CATEGORY_CLASSIFIER.md`,
`CATEGORY_CLASSIFIER_GENERALIZATION.md`.

---

## Experiment 3 — DistilBERT architecture ablation (2026-08-23)

**Why tested:** Experiment 2 fixed the *data* (broader, verified-disjoint
templates). The open question it left: was the resulting 78.0% ceiling a
data-diversity limit, or an architecture limit?

**Methodology:** fine-tuned `distilbert-base-uncased` on the *exact same*
training data as the v3/v4 retrain — same rows, same evaluation set, only
the model architecture changed. This isolates architecture as the single
variable.

**Result:**

```
TF-IDF + LinearSVC (deployed):  78.0% accuracy, 70.5% correct-and-confident
DistilBERT (this experiment):   83.0% accuracy, 82.0% correct-and-confident
```

A larger gain than the data-diversity retrain produced, on identical
data — real evidence the remaining ceiling was architectural, not just a
data problem.

**Measured trade-off, and why it was not deployed:**

```
                   Deployed (TF-IDF+LinearSVC)   DistilBERT candidate
Model size         ~2.1 MB                        267.8 MB (128× larger)
Inference latency  sub-millisecond                369 ms (370× slower)
```

No way to verify, from this environment, that the live hosting tier has
the memory headroom for `torch` + `transformers` alongside the existing
model stack. An out-of-memory crash on the live service would be a worse
outcome than staying on the faster, 5-points-less-accurate classifier —
a deliberate, reasoned trade-off, not an oversight.

Full detail: `PaySense-ML-Backend/CATEGORY_CLASSIFIER_V3_ATTEMPT.md`.

---

## Experiment 4 — Source contamination and the threshold ceiling (2026-08-26/27)

**Initial state:** a routine audit found the canonical test set inherits
the same ~35% "supplement"-source contamination already documented in
training data (a third of rows come from an external synthetic dataset
schema-bridged into this project's data, whose label was generated by
thresholding two risk-score features).

**Investigation, Phase 1 — how deep does it go:** checked every column in
the dataset for source-dependent patterns, not just the two originally
suspected. Found **23 of ~30 numeric columns and 12 of 14 categorical
columns are a single constant value across the entire 10,000-row
supplement subset** — including `receiver_id == "SYN_MRC_UNKNOWN"`, a
literal synthetic marker. A single column (`device_risk_score.notnull()`)
separates the two sources with exactly 100% accuracy.

**Quantified impact:** source-stratified re-scoring of the deployed
model:

```
                    Blended (every prior headline)   Organic-only   Supplement-only
ROC-AUC             0.8969                            0.7465          1.0000
PR-AUC              0.5498                             0.1138          1.0000
Recall @ τ=0.50     39.53% (100/253)                   2.55% (4/157)   100% (96/96)
```

**Investigation, Phase 2 — does retraining on clean data fix it:** tested
directly rather than assumed. Retrained XGBoost on anchor-only (organic)
rows, identical hyperparameters and monotonic constraints, risk-score
columns dropped. Organic-subset ROC-AUC: 0.7260 → 0.7261 — statistically
identical. **Removing the contamination did not improve organic
performance**, because it was never suppressing real capability to begin
with; the model appears to solve the (trivially separable) supplement
third almost independently of how it learns the organic two-thirds.

**Investigation, Phase 3 — what's the honest ceiling on clean data:**
every threshold this project had ever selected was chosen on the same
partition its performance was reported on. Ran a proper train (60%) /
validation (20%) / test (20%) split, entirely on organic data, threshold
selected on validation only, applied once to an untouched final test set:

```
Frozen threshold (selected on validation): 0.10
Final test (never touched during selection):
  ROC-AUC=0.7050  PR-AUC=0.0945  Precision=8.82%  Recall=21.05%
```

More honest than the earlier 2.55% figure, which used a threshold
calibrated for a different (contaminated) score distribution. Still far
from the documented Recall≥75% business constraint — confirmed
unachievable a third time, now via the cleanest methodology available on
this dataset.

**Regression tests added:** `test_organic_subset_performance_is_much_
weaker_than_blended_headline`, `test_supplement_source_is_near_fully_
constant_and_perfectly_separable`, `test_no_swept_threshold_meets_both_
business_constraints` (`tests/test_frozen_model_metrics.py`).

**What this means going forward:** this is not a fixable methodology bug
— it's a genuine data-quantity/quality ceiling. Improving real-world
organic fraud detection requires new, genuinely organic training data,
not further cleanup of what already exists.

Full detail: `PaySense-ML-Backend/SOURCE_CONTAMINATION_INVESTIGATION.md`.

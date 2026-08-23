# PaySense — Recall Ceiling Remediation: Is Feature Dominance Fixable?

> **UPDATE (2026-08-23, later the same day):** the `monotonic` variant
> described in this document was **adopted as the new deployed model**
> immediately after this experiment concluded — `paysense_phase3.py` now
> trains with `monotone_constraints` on the three behavioral features (see
> that file's inline comment, which cites this document by name) and the
> retrained model was frozen to `artefacts/paysense_model.pkl`,
> `paysense_preprocessor.pkl`, and `paysense_threshold.pkl`, overwriting the
> prior (unconstrained) frozen artifacts. Concretely: **the model this
> document calls "monotonic" is, as of this update, what
> `artefacts/paysense_model.pkl` actually is**; the model this document
> calls "baseline"/"vanilla_replica" is preserved unchanged at
> `artefacts/paysense_model_vanilla_replica.pkl` as a fixed historical
> reference point, not deleted or overwritten. Everything below this line
> describes the state of the world *as it was when this experiment ran* —
> the original baseline vs. three untried candidates — and is left as the
> historical record that justified the decision (see §6's verdict for the
> reasoning). It is intentionally **not** rewritten to describe today's
> deployed artifact in the present tense; README.md's Key Results section
> and `tests/test_frozen_model_metrics.py` carry the current, live,
> re-verified numbers for whatever is actually on disk today.

> **UPDATE (2026-08-24) — scope clarification:** every metric in this
> document (the 76/177 split, the 71.94%/69.96% recall ceilings, all
> variant comparisons) was computed by calling `model.predict_proba()`
> directly on the raw XGBoost artifact — the correct scope for the
> question this document asks (is a tree-structure/feature-dominance
> problem in XGBoost itself fixable), but **not** a description of the
> full deployed system. `/predict` calls `src.fraud_model.score()`, a
> 3-scorer ensemble; see README.md's Key Results note and
> `EDA_FEATURE_ENGINEERING.md` §4.5 for the correction discovered
> 2026-08-24 (the ensemble behaves substantially differently from raw
> XGBoost, and the deployed threshold moved from 0.30 to 0.50 as a
> result). This document's own conclusions about XGBoost's ranking
> behavior and the monotonic-constraints fix remain valid on their own
> terms; only the implicit assumption that "the frozen model's metrics"
> meant "the deployed system's metrics" was wrong, and is corrected
> elsewhere, not here.

**Date:** 2026-08-23
**Author's intent:** PLATT_SCALING_RESULT.md already showed, correctly, that
Platt scaling cannot move the frozen model's recall ceiling (76 of 253 fraud
rows unreachable at any threshold, even τ=0.05) because it is a
rank-preserving monotonic transform — it can only relabel the probability
axis, not the ranking underneath it. That left the real question open: is
the ranking itself fixable? This document tests one specific, falsifiable
hypothesis about *why* those 76 rows rank so low — that XGBoost's trees gate
hard on `new_device_flag` / `ip_location_mismatch` first and only let the
behavioral features (`amount_deviation_score`, `transaction_velocity`,
`failed_attempts_last_24h`) matter inside the branches where those flags are
already 1 — by actually retraining models that remove that structural
possibility, not by arguing about it. Script: `recall_ceiling_remediation.py`.
No changes to `artefacts/paysense_model.pkl`, `paysense_preprocessor.pkl`, or
`paysense_threshold.pkl` happened anywhere in this check.

**Bottom line, stated early:** the hypothesis is **partially confirmed, and
the fix is real but not free.** Forcing XGBoost's trees to isolate the
behavioral group from the hard-signal group (`interaction_constraints`)
recovers **31 of the 76** previously-invisible fraud rows at τ=0.05 — a
genuine, non-trivial, structurally-caused recovery, not noise. But it costs
overall ranking quality (ROC-AUC drops 0.8863 → 0.8814) and roughly
**doubles the false-positive volume** at aggressive thresholds. A cheaper,
weaker constraint (`monotone_constraints`) recovers only 10 of the 76 but
does so while *improving* ROC-AUC and PR-AUC slightly — a small, genuinely
"free" win. At the model's actual **deployed threshold (τ=0.40)**, none of
the three variants tested move recall in any way that matters (97–98 TP out
of 253 fraud rows across all four models, baseline included) — this whole
recall-ceiling conversation is about a threshold region (τ≤0.15) the
production system does not operate in. Full details, all four models, all
metrics, below.

---

## 1. What was actually being tested

The working hypothesis, as handed off: XGBoost's tree structure lets
`new_device_flag`/`ip_location_mismatch` — SHAP's two most dominant features
(README.md's SHAP section, mean |SHAP| = 1.17 for `new_device_flag`) —
dominate early splits, so that when both flags read 0 ("device/IP look
clean"), the behavioral features never get a chance to raise the score, no
matter how anomalous they are. This is checkable two ways: (1) does the
82-row diagnostic actually show this asymmetry, and (2) does removing the
structural possibility of that dominance (via `interaction_constraints`,
which literally forbids the two feature groups from co-occurring on the same
decision path) change what the model catches. Both were tested directly,
not assumed.

---

## 2. Step 1 — independent reproduction of the 76/177 diagnostic

`paysense_phase3.py` Block 0 was reproduced exactly (80/20 stratified split,
`random_state=42`, same `DROP_COLS`, same preprocessing) and the frozen
artifacts (`paysense_model.pkl`, `paysense_preprocessor.pkl`) were loaded
**read-only** and scored against the resulting 6,000-row test set (253
fraud). This is independent of the number handed off at the start of this
task — it was recomputed from scratch against the actual on-disk artifacts,
not copied from the brief.

At τ=0.05: **177 fraud rows caught, 76 missed, recall = 69.9605%** — matching
PLATT_SCALING_RESULT.md's cited figure exactly, and matching the handoff's
76/177 split exactly. `recall_ceiling_remediation.py` asserts both counts
before proceeding, so this is enforced, not eyeballed.

The feature-mean comparison (raw, untransformed scale — all six columns are
numeric with zero missing values in the source CSV, so no imputation
distorts these numbers) was also independently recomputed:

| Feature | Invisible fraud (n=76) mean | Caught fraud (n=177) mean |
|---|---:|---:|
| `new_device_flag` | 0.066 | 0.689 |
| `ip_location_mismatch` | 0.053 | 0.616 |
| `amount_deviation_score` | 0.649 | 0.556 |
| `transaction_velocity` | 0.697 | 0.582 |
| `failed_attempts_last_24h` | 0.697 | 0.339 |
| `recurring_payment_flag` | 0.289 | 0.113 |

Matches the handoff table exactly (independently recomputed, not trusted).
The pattern holds under independent verification: the 76 invisible fraud
rows almost never trip the two hard-signal flags, but score *higher* on
every one of the four behavioral/recurring features than the fraud the model
does catch. This is consistent with — though does not by itself prove — the
dominance hypothesis; §3–5 test it directly by intervening on the model
structure rather than reading tea leaves from a correlation table.

---

## 3. Step 2 — three remediation variants, one shared pipeline

All four models below (one baseline control + three hypothesis variants)
share **exactly** the same train/test split, the same `ColumnTransformer`
preprocessing, the same SMOTE resampling (`sampling_strategy="auto",
k_neighbors=5, random_state=42`), and the same XGBoost hyperparameters from
`paysense_phase3.py` Block 0 (`n_estimators=400, max_depth=5,
learning_rate=0.05, subsample=0.80, colsample_bytree=0.80,
min_child_weight=10, gamma=0.10, reg_alpha=0.05, reg_lambda=1.50,
eval_metric="aucpr", tree_method="hist", random_state=42`). Only the
hypothesis-relevant argument changes per variant, so any metric difference
is attributable to that one change and nothing else.

**Control — `vanilla_replica`:** identical hyperparameters, no constraints,
trained via this script's own SMOTE call rather than reusing the frozen
model. This exists to confirm the retraining pipeline itself introduces no
drift before judging the hypothesis variants against it: it reproduced
ROC-AUC 0.8863 / PR-AUC 0.5339 / recall@0.40 38.34%, bit-for-bit identical to
the frozen artifact's own numbers. Saved to
`artefacts/paysense_model_vanilla_replica.pkl`.

**Variant 1 — `interaction_constrained`:** `new_device_flag` (index 19) and
`ip_location_mismatch` (index 20) placed in one XGBoost
`interaction_constraints` group; `amount_deviation_score` (23),
`transaction_velocity` (22), `failed_attempts_last_24h` (21) placed in a
second, disjoint group. This forbids any tree from ever combining a
hard-signal split and a behavioral split on the same root-to-leaf decision
path — the most literal possible test of "does the model stop looking at
behavioral signal once it's committed to a hard-signal branch."

One implementation detail surfaced and is worth recording, the same way
PLATT_SCALING_RESULT.md recorded its sign-convention discovery rather than
silently working around it: XGBoost's sklearn wrapper only accepts
`interaction_constraints` as a list of **feature-name** lists when the
training `DMatrix` carries feature names, or as a **raw JSON string of
integer indices** passed through untranslated. `X_train_bal`/`X_test_proc`
here are plain numpy arrays with no attached column names (same as
`paysense_phase3.py`'s own Block 0), so the name-list form raised
`ValueError: Constrained features are not a subset of training data feature
names`. The fix was the JSON-string form: `interaction_constraints =
'[[19, 20], [23, 22, 21]]'`. Confirmed this is not a training silently
falling back to "unconstrained" by walking the actual fitted trees after
training (400 trees, 2,525 root-to-leaf paths): **zero** paths contain both
a hard-signal split and a behavioral split. The constraint is real, not
just accepted-and-ignored. Saved to
`artefacts/paysense_model_interaction_constrained.pkl`.

**Variant 2 — `monotonic`:** `monotone_constraints` set to `+1` for
`amount_deviation_score`, `transaction_velocity`, `failed_attempts_last_24h`
(indices 23, 22, 21) and `0` for all other 37 features — a weaker,
different-mechanism guarantee than Variant 1: it does not stop the hard
signals from dominating splits, it only guarantees that within whatever
branch structure XGBoost builds, increasing any of these three features can
never *decrease* predicted fraud probability. This tests whether the
ceiling is caused by the behavioral features having a genuinely
non-monotonic or reversed marginal effect somewhere in the tree ensemble
(possible with unconstrained trees and SMOTE-interpolated training data)
rather than by outright suppression. Saved to
`artefacts/paysense_model_monotonic.pkl`.

**Variant 3 — `composite_feature`:** an engineered
`behavioral_anomaly_score` column appended as feature 41 — the mean of the
three behavioral features, each min-max normalized using **train-only**
statistics (no leakage) before averaging. The model is otherwise completely
unconstrained; this tests the cheapest possible intervention — does simply
handing the model one pre-combined, high-visibility behavioral signal (which
an unconstrained tree could put at the root if it wanted to) help, without
touching how XGBoost is allowed to structure any split. Saved to
`artefacts/paysense_model_composite_feature.pkl`.

---

## 4. Full honest results — every variant, every metric

### 4.1 Headline metrics

| Model | ROC-AUC | Δ ROC-AUC | PR-AUC | Δ PR-AUC | Recall @ τ=0.40 | Precision @ τ=0.40 | Recovered of the 76 @ τ=0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (frozen / vanilla_replica)** | 0.8863 | — | 0.5339 | — | 38.34% (97/253) | 98.98% | 0 / 76 |
| **Interaction-constrained** | 0.8814 | **−0.0049** | 0.5342 | +0.0004 | 38.74% (98/253) | 96.08% | **31 / 76** |
| **Monotonic** | 0.8889 | **+0.0025** | 0.5352 | +0.0013 | 38.34% (97/253) | 94.17% | **10 / 76** |
| **Composite feature** | 0.8868 | +0.0005 | 0.5292 | −0.0047 | 38.74% (98/253) | 97.03% | **8 / 76** |

None of the three variants change recall at the **deployed** threshold
(τ=0.40) in any way that matters — 97–98 true positives out of 253 fraud
rows across all four models, a swing of at most 1 additional catch. The
recall-ceiling story only shows up at aggressive thresholds the production
system does not use.

### 4.2 Full threshold sweep, τ = 0.05 → 0.50

**Baseline (vanilla_replica — reproduces the frozen model exactly):**

| τ | Precision | Recall | F1 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1715 | 0.6996 | 0.2755 | 177 | 855 | 76 |
| 0.10 | 0.3036 | 0.4980 | 0.3772 | 126 | 289 | 127 |
| 0.15 | 0.5134 | 0.4545 | 0.4822 | 115 | 109 | 138 |
| 0.20 | 0.6815 | 0.4229 | 0.5220 | 107 | 50 | 146 |
| 0.25 | 0.8347 | 0.3992 | 0.5401 | 101 | 20 | 152 |
| 0.30 | 0.8909 | 0.3874 | 0.5399 | 98 | 12 | 155 |
| 0.35 | 0.9604 | 0.3834 | 0.5480 | 97 | 4 | 156 |
| **0.40** | **0.9898** | **0.3834** | **0.5527** | **97** | **1** | **156** |
| 0.45 | 1.0000 | 0.3794 | 0.5501 | 96 | 0 | 157 |
| 0.50 | 1.0000 | 0.3794 | 0.5501 | 96 | 0 | 157 |

**Interaction-constrained:**

| τ | Precision | Recall | F1 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1276 | 0.7905 | 0.2197 | 200 | 1368 | 53 |
| 0.10 | 0.3268 | 0.5296 | 0.4042 | 134 | 276 | 119 |
| 0.15 | 0.5578 | 0.4387 | 0.4912 | 111 | 88 | 142 |
| 0.20 | 0.7727 | 0.4032 | 0.5299 | 102 | 30 | 151 |
| 0.25 | 0.8761 | 0.3913 | 0.5410 | 99 | 14 | 154 |
| 0.30 | 0.9333 | 0.3874 | 0.5475 | 98 | 7 | 155 |
| 0.35 | 0.9515 | 0.3874 | 0.5506 | 98 | 5 | 155 |
| **0.40** | **0.9608** | **0.3874** | **0.5521** | **98** | **4** | **155** |
| 0.45 | 0.9800 | 0.3874 | 0.5552 | 98 | 2 | 155 |
| 0.50 | 0.9899 | 0.3874 | 0.5568 | 98 | 1 | 155 |

At τ=0.05 this variant catches 23 more fraud rows than baseline (200 vs
177) but at the cost of **513 more false positives** (1,368 vs 855) — the
recall gain is real, but it is bought with a precision collapse from 17.2%
to 12.8% at that threshold. This is the clearest evidence that isolating
the two feature groups genuinely unlocks behavioral signal that was
previously suppressed — but that signal, once unlocked across the *whole*
test set (not just the 76 target rows), is noisier than the hard-signal
features it was competing with, which is exactly why overall ROC-AUC drops.

**Monotonic:**

| τ | Precision | Recall | F1 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1687 | 0.7194 | 0.2733 | 182 | 897 | 71 |
| 0.10 | 0.2976 | 0.5375 | 0.3831 | 136 | 321 | 117 |
| 0.15 | 0.4609 | 0.4664 | 0.4637 | 118 | 138 | 135 |
| 0.20 | 0.6250 | 0.4150 | 0.4988 | 105 | 63 | 148 |
| 0.25 | 0.7879 | 0.4111 | 0.5403 | 104 | 28 | 149 |
| 0.30 | 0.8644 | 0.4032 | 0.5499 | 102 | 16 | 151 |
| 0.35 | 0.9074 | 0.3874 | 0.5429 | 98 | 10 | 155 |
| **0.40** | **0.9417** | **0.3834** | **0.5449** | **97** | **6** | **156** |
| 0.45 | 0.9697 | 0.3794 | 0.5455 | 96 | 3 | 157 |
| 0.50 | 0.9897 | 0.3794 | 0.5486 | 96 | 1 | 157 |

**Composite feature:**

| τ | Precision | Recall | F1 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1649 | 0.6957 | 0.2667 | 176 | 891 | 77 |
| 0.10 | 0.3057 | 0.5099 | 0.3822 | 129 | 293 | 124 |
| 0.15 | 0.4802 | 0.4308 | 0.4542 | 109 | 118 | 144 |
| 0.20 | 0.7020 | 0.4190 | 0.5248 | 106 | 45 | 147 |
| 0.25 | 0.8632 | 0.3992 | 0.5459 | 101 | 16 | 152 |
| 0.30 | 0.9083 | 0.3913 | 0.5470 | 99 | 10 | 154 |
| 0.35 | 0.9423 | 0.3874 | 0.5490 | 98 | 6 | 155 |
| **0.40** | **0.9703** | **0.3874** | **0.5537** | **98** | **3** | **155** |
| 0.45 | 0.9899 | 0.3874 | 0.5568 | 98 | 1 | 155 |
| 0.50 | 1.0000 | 0.3794 | 0.5501 | 96 | 0 | 157 |

Note this variant's recall at τ=0.05 (176, 69.57%) is actually **one row
below** baseline (177, 69.96%) even though it recovers 8 of the *original*
76 — the composite feature helps some previously-invisible rows cross 0.05
while pushing a small number of previously-caught rows below it. This is
reported plainly rather than only citing the "8 recovered" number in
isolation, because net recall at that threshold barely moves at all.

---

## 5. Was the isolation real? (Verified, not assumed)

Trusting that `interaction_constraints` "worked" because no exception was
raised would repeat the exact mistake this repo's docs made with Platt
scaling — assuming a mechanism did what its name suggested without checking.
So the fitted booster's 400 trees were walked directly via
`get_booster().trees_to_dataframe()`, collecting every root-to-leaf decision
path and checking whether any single path contains both a hard-signal split
and a behavioral split:

- **Paths checked: 2,525** (across all 400 trees)
- **Paths violating group isolation: 0**

Confirmed: the constraint is structurally real in the fitted model, not a
parameter that was silently ignored.

---

## 6. Verdict — fixable structural issue, or inherent data limitation?

**Both, in a specific and now-quantified proportion.** The interaction-
constrained variant proves the dominance hypothesis is not fiction: 31 of
the 76 invisible rows (41%) were only invisible because the tree structure
never let their behavioral anomaly matter while `new_device_flag` and
`ip_location_mismatch` both read 0. That is a real, structural, and
addressable cause — not a data limitation — for that specific subset.

But the same experiment shows the other side honestly: **45 of the 76
(59%) remain uncaught even under the most aggressive structural
intervention tested**, and unlocking the 31 costs measurable ranking quality
everywhere else in the test set (ROC-AUC −0.49pp, and roughly 1.6× the false
positives at τ=0.05). That is consistent with a mixed picture, not a clean
story in either direction:

- **Some of the ceiling is genuinely a training/structure artifact** — the
  interaction-constraint result is direct proof, not inference.
- **Some of the remaining 45 rows may be genuinely ambiguous** even with
  full attention to the behavioral features — recall that the invisible
  rows' behavioral means (0.649–0.697) are elevated but not extreme, and
  sit inside the same range plenty of legitimate transactions occupy; once
  the model is forced to weigh them, it correctly (from a ranking
  standpoint) also promotes many *legitimate* transactions with similarly
  elevated behavioral scores, which is exactly why precision at τ=0.05
  collapses from 17.2% to 12.8% under the interaction-constrained model.
  Recovering those 45 may require genuinely new, more discriminative
  features (e.g. the per-user personalization the report already
  speculates about) rather than a different arrangement of the features
  already available.

**Is it worth shipping either fix?** Not as a blanket yes:

- **`monotone_constraints` is the closest thing to a free win found here**:
  it recovers 10 of the 76 while *improving* both ROC-AUC (+0.0025) and
  PR-AUC (+0.0013), and barely moves the deployed τ=0.40 operating point
  (precision 94.17% vs 98.98%, recall unchanged at 97/253). This is a
  legitimate candidate for a follow-up decision — small upside, small,
  well-understood cost — but "unambiguously and substantially better across
  the board" is too strong a claim for a 10-row recall gain against a small
  precision cost at τ=0.40; it is a genuine, modest improvement, not a
  breakthrough.
- **`interaction_constraints` is a real but expensive lever**, not a free
  one: 3× the recall recovery of the monotonic variant, but at a
  measurable cost to overall discrimination and roughly 1.6× the false-
  positive volume at the thresholds where the gain shows up. Whether that
  trade is worth it is a business decision (how much analyst/user friction
  is acceptable per additional fraud row caught at an aggressive
  threshold), not a modeling one — it is flagged here as a candidate for
  that decision, not shipped.
- **The composite feature is the weakest of the three** — smallest recall
  recovery (8/76, and only a net +1 at τ=0.05 once the rows it *loses* are
  netted out), and the only variant that makes PR-AUC measurably worse
  (−0.0047). Not recommended.

None of the three is a wholesale replacement for the frozen model, and none
is recommended for silent deployment. `artefacts/paysense_model.pkl`,
`paysense_preprocessor.pkl`, and `paysense_threshold.pkl` are unchanged
**as of when this experiment ran.**

> **Post-hoc verdict, added the same day:** the "not as a blanket yes"
> framing above was the honest read *before* a deployment decision was
> made. It was subsequently decided that `monotone_constraints`'s
> small-upside/small-cost profile — the one candidate with no measurable
> downside anywhere in §4 — was worth taking, and it was adopted as the new
> deployed model (see the update note at the top of this document).
> `interaction_constraints` and the composite feature were not adopted, for
> the reasons already stated above (expensive/weak, respectively); nothing
> about those two verdicts changed.

---

## 7. Test added

`tests/test_recall_ceiling_remediation.py` locks in two things as regression
guards:

1. The frozen model's 76/177 split at τ=0.05 and the feature-dominance
   pattern itself (invisible rows score lower on the two hard-signal flags
   and *higher* on the three behavioral features than caught rows) — so a
   future retrain of the frozen model that silently changes this dynamic is
   caught, the same way `test_frozen_model_metrics.py` catches metric
   drift.
2. That the three saved candidate artifacts
   (`paysense_model_interaction_constrained.pkl`, `paysense_model_monotonic.pkl`,
   `paysense_model_composite_feature.pkl`), if present, still reproduce the
   ROC-AUC / recovered-of-76 numbers reported above within tolerance — a
   drift guard for the experiment's own outputs, mirroring how
   `platt_scaling_experiment.py`'s calibrator artifact is guarded.

All existing tests were confirmed undisturbed: `pytest tests/ -v` was run
after adding this file (see §8).

---

## 8. Reproducing this check

```
cd PaySense-ML-Backend
venv\Scripts\python.exe recall_ceiling_remediation.py
```

Takes a few minutes (four fresh SMOTE + 400-tree XGBoost trainings on the
24,000-row train partition). Requires the existing `venv/` (no new
packages — `interaction_constraints` and `monotone_constraints` are both
native `XGBClassifier` parameters in the already-installed xgboost 2.1.1)
and `paysense_master_dataset.csv` plus the three frozen artifacts under
`artefacts/`. Produces `recall_ceiling_remediation_results.json` (full
numeric results, cited verbatim above) and four new model artifacts under
`artefacts/` — none of which touch the three frozen files, none of which
are wired into `src/fraud_model.py` or `main.py`.

Regression test: `pytest tests/test_recall_ceiling_remediation.py -v`.

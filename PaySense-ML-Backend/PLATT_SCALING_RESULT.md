# PaySense — Platt Scaling: Does It Move the Recall Ceiling?

**Date:** 2026-08-23
**Author's intent:** README.md and `PaySense-Report/paysense_report.tex` have
long stated that the frozen model's 69.96% recall ceiling (at the most
aggressive threshold tested, τ=0.05) is "a probability calibration issue,"
and proposed **Platt Scaling** as the fix. That was a stated intention, never
implemented or tested. This document implements it and measures, honestly,
whether it does what the docs claim. Script: `platt_scaling_experiment.py`.
No retraining, no fine-tuning of XGBoost, and no changes to
`artefacts/paysense_model.pkl`, `paysense_preprocessor.pkl`, or
`paysense_threshold.pkl` happened anywhere in this check.

**Bottom line, stated early:** the recall ceiling does **not** move.
ROC-AUC and PR-AUC are unchanged to floating-point precision, and the exact
same 76 of 253 fraud rows remain unreachable at any decision threshold. This
matches the theoretical expectation, not the docs' claim — a strictly
monotonic 1-D transform of a classifier's scores cannot change its ROC
curve, PR curve, or which rows clear a given operating point; it can only
relabel which numeric threshold value corresponds to that point. The
"Platt Scaling is the fix" framing in both documents is incorrect and is
corrected below (§6). Platt scaling **does** answer a real, separate
question — whether raw XGBoost's probabilities are already trustworthy as
probabilities — and that result is genuinely mixed, reported in full in §4.

---

## 1. What was actually being claimed, and why it needed checking

README.md (line 145) and `paysense_report.tex` (§V-D, §"Limitations and
Future Work", abstract, conclusion) all state that the 69.96% recall
ceiling is a "probability calibration" problem, distinct from a threshold
problem, and that Platt Scaling — fitting a logistic function
σ(A·f(x)+B) to the model's raw scores against held-out labels — is "the
recommended first implementation" to fix it.

This is checkable, not just arguable. Platt scaling is, by construction, a
**strictly monotonically increasing** function of the raw score whenever
A > 0 (or strictly decreasing if A < 0 — either way, a bijection that
preserves order, just possibly reversed). A decision rule "flag if score ≥ τ"
under a strictly monotonic transform g flags exactly the same set of rows
under "flag if g(score) ≥ g(τ)" for any τ. Consequently:

- **ROC-AUC and PR-AUC**, which are computed purely from the *ranking* of
  scores against labels (not their numeric values), must be identical
  before and after any strictly monotonic transform.
- **The full precision-recall frontier** (the set of achievable
  (precision, recall) pairs across all thresholds) is identical — only the
  numeric threshold value needed to reach a given point on that frontier
  changes.
- Therefore the recall ceiling — driven by which fraud rows are *ranked*
  lowest by the model, not by what numeric probability value they're
  assigned — cannot move under Platt scaling.

That's the theory. It was not assumed here; it was implemented and measured
against the actual frozen artifacts.

---

## 2. Method, and the leakage problem that shapes it

`paysense_phase3.py` Block 0 was read in full before writing anything.
Its split is: 80/20 stratified train/test on `paysense_master_dataset.csv`
(`random_state=42`, same `DROP_COLS`, target `is_fraud`), then SMOTE +
XGBoost fit on **100% of the 80% train partition** — there is no internal
train/validation carve-out anywhere in Block 0. That matters here: it means
there is no subset of the 80% train partition that the frozen model has not
already seen. A calibration set drawn from inside that partition would not
be genuinely held out — the frozen model would have already memorized (to
whatever degree gradient-boosted trees do) the exact rows used to fit the
sigmoid, biasing the calibration-quality numbers (not the ranking-invariance
math, which holds regardless of what data any monotonic transform was fit
on — but the Brier/ECE numbers in §4 absolutely would be biased by this).

So the only genuinely unseen data available **without retraining XGBoost**
is inside the 20% test partition. The split used here:

1. Reproduce the canonical split exactly: 80% train (24,000 rows, discarded
   here — already consumed by the frozen model) / 20% test (6,000 rows,
   253 fraud — same partition every doc in this repo cites).
2. Split that 20% test partition again, stratified, 50/50,
   `random_state=42`:
   - **calib** (3,000 rows, 127 fraud) — used **only** to fit the Platt
     sigmoid's A, B via `sklearn.calibration.CalibratedClassifierCV(
     estimator=model, method="sigmoid", cv="prefit")`. The base XGBoost
     estimator is frozen (`cv="prefit"` means it is not refit).
   - **final_test** (3,000 rows, 126 fraud) — used **only** to report
     calibration-quality metrics (Brier score, ECE, reliability tables) in
     §4. Never touched during fitting.

For the ranking/AUC/recall-ceiling question in §3, the already-fit
calibrator is *applied* (not re-fit) to the full canonical 253-fraud test
partition. Applying a fixed function is not a leakage risk for a
rank-preservation check — the check is a mathematical identity, not an
estimate that could be inflated by reuse — and doing it this way makes the
numbers directly comparable to the 253-fraud table every other document in
this repo cites.

A new, separate artifact, `artefacts/platt_calibrator.pkl`, was saved. It is
**not** wired into `src/fraud_model.py` or `main.py` — this is an
experiment, and given the result below, there is no case for wiring it in
for the recall question (see §6).

---

## 3. Does Platt scaling move the recall ceiling? (It does not.)

Evaluated on the full canonical 253-fraud test partition, comparing raw
XGBoost scores against the Platt-scaled version of the same scores:

| Metric | Raw | Platt-scaled | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.886349 | 0.886349 | +0.00000000 |
| PR-AUC | 0.533859 | 0.533859 | −0.00000000 |

Rank order was checked directly, not just inferred from equal AUC: sorting
all 6,000 test rows by raw score and walking through them in order, no
Platt-scaled value from an earlier block of tied raw scores ever exceeds a
later block's — **0 inversions across all 6,000 rows**.

The fitted sigmoid: **A = −12.946041, B = 4.009092** (negative A — the
`CalibratedClassifierCV` internals for a binary `cv="prefit"` classifier fit
the sigmoid against the *complement* class internally, so the class-1
probability is recovered as σ(−(A·p_raw + B)); this was verified empirically
against `platt.predict_proba()`'s actual output, not assumed, and the
experiment script asserts this sign convention holds before using it —
see the `_sanity` check in `platt_scaling_experiment.py`).

The documented ceiling, re-tested directly: at the raw threshold τ=0.05,
recall is 69.9605% (177/253 caught, 1,032 rows flagged). The analytically
equivalent Platt-scaled threshold is τ' = σ(−(A·0.05+B)) = 0.033511.
Thresholding the Platt-scaled scores at 0.033511 flags **the exact same
1,032 rows** and produces **the exact same 69.9605% recall** — the same 76
of 253 fraud rows remain below the equivalent threshold, identically.

This was checked at all ten swept thresholds (τ = 0.05 → 0.50), not just
τ=0.05:

| τ (raw) | Raw P | Raw R | τ' (Platt-equivalent) | Platt P | Platt R | Same rows flagged |
|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 0.1715 | 0.6996 | 0.033511 | 0.1715 | 0.6996 | **True** |
| 0.10 | 0.3036 | 0.4980 | 0.062124 | 0.3036 | 0.4980 | **True** |
| 0.15 | 0.5134 | 0.4545 | 0.112327 | 0.5134 | 0.4545 | **True** |
| 0.20 | 0.6815 | 0.4229 | 0.194680 | 0.6815 | 0.4229 | **True** |
| 0.25 | 0.8347 | 0.3992 | 0.315921 | 0.8347 | 0.3992 | **True** |
| 0.30 | 0.8909 | 0.3874 | 0.468721 | 0.8909 | 0.3874 | **True** |
| 0.35 | 0.9604 | 0.3834 | 0.627621 | 0.9604 | 0.3834 | **True** |
| 0.40 | 0.9898 | 0.3834 | 0.763023 | 0.9898 | 0.3834 | **True** |
| 0.45 | 1.0000 | 0.3794 | 0.860161 | 1.0000 | 0.3794 | **True** |
| 0.50 | 1.0000 | 0.3794 | 0.921574 | 1.0000 | 0.3794 | **True** |

All ten points, all identical. **This confirms the theoretical expectation,
not the docs' framing**: Platt scaling remaps the numeric probability axis
— what "0.40" means shifts from a raw score to a calibrated one — but every
achievable (precision, recall) operating point, and therefore the recall
ceiling itself, is untouched. There is no threshold on the Platt-scaled axis
that recovers a single one of the 76 fraud rows the raw model already ranks
below its own bottom decile of fraud scores; they are exactly as
unreachable after calibration as before, because calibration cannot change
*which* rows are ranked where — only what numeric label is printed next to
that rank.

---

## 4. Is the separate question — are the probabilities themselves
trustworthy? — worth asking? Yes, and the answer here is mixed, tilting
against Platt scaling.

Evaluated **only** on the final_test slice (3,000 rows, 126 fraud) —
genuinely unseen by both the frozen XGBoost model and the Platt calibrator's
fitting step:

| Metric | Raw XGBoost | Platt-scaled | Better |
|---|---:|---:|---|
| Brier score | 0.022447 | 0.022994 | Raw (lower is better) |
| ECE (10 equal-width bins) | 0.005386 | 0.007790 | Raw (lower is better) |

On this single draw, raw XGBoost is *already* slightly better calibrated
than its Platt-scaled version by both metrics — the opposite of what the
docs assumed ("the model assigns inappropriately low probabilities...
Platt Scaling... redistributes the probability mass more uniformly"). The
reliability tables show why: raw scores are heavily concentrated near 0
(2,768 of 3,000 rows in [0.00, 0.10), empirical fraud rate 1.99% there,
matching a mean predicted probability of 1.87% closely) and near 1 (56 rows
in [0.90, 1.00), 100% actually fraud, mean predicted 99.94%) — XGBoost's own
`eval_metric="aucpr"` does not optimize for calibration, but log-loss-driven
gradient boosting with enough estimators (400 here) tends to produce scores
that are already reasonably well-shaped for this task, at least in the
bulk. Platt scaling's fitted sigmoid, forced through a 2-parameter family on
only 127 fraud examples, over-corrects: its own reliability table shows
worse extremes (e.g. the [0.30, 0.40) bin: 6 rows, 32.6% mean predicted vs.
50% empirical — a bigger miss than raw's corresponding [0.30,0.40) bin,
which had only 5 rows and 0% empirical, also a miss, but smaller in
absolute probability-mass terms across the full bin count).

**126 fraud rows is a small sample for a calibration verdict, so this was
not left as a single-draw claim.** The calib/final_test split was
re-resampled with 5 different seeds (0–4), refitting Platt fresh each time
on that seed's calib half and evaluating Brier/ECE on that seed's other
half — still zero contact with the 80% train partition:

| Seed | Brier (raw) | Brier (Platt) | ECE (raw) | ECE (Platt) | Platt wins Brier? | Platt wins ECE? |
|---:|---:|---:|---:|---:|---|---|
| 0 | 0.024968 | 0.025768 | 0.007271 | 0.004728 | No | Yes |
| 1 | 0.024157 | 0.024746 | 0.005270 | 0.003268 | No | Yes |
| 2 | 0.024442 | 0.024753 | 0.007390 | 0.001006 | No | Yes |
| 3 | 0.023707 | 0.023966 | 0.004577 | 0.002276 | No | Yes |
| 4 | 0.023742 | 0.024276 | 0.004446 | 0.004979 | No | No |

Across all 6 draws (the original seed=42 50/50 split plus these 5):
**Platt scaling never beats raw XGBoost on Brier score (0/6)**, but **does
beat it on ECE in most draws (4/6)**. This is a genuinely mixed result, not
softened either direction:

- **Brier score** (a strictly proper scoring rule — it cannot be gamed by a
  transform that doesn't actually improve calibration) consistently, if
  narrowly, favors the raw scores. This is the more rigorous of the two
  metrics here.
- **ECE with 10 equal-width bins** more often favors Platt, but ECE is
  sensitive to bin choice and bin sample size, and several bins in this
  126-fraud final_test slice hold single-digit counts (see the [0.30,0.90)
  range in §4's tables) — a metric this noisy on this little data should
  not be read as a confident win.

**Honest read:** raw XGBoost's probabilities in this test set are already
reasonably well-calibrated in the bulk (both extremes — near-0 and near-1 —
track their empirical rates closely), which is a real, if unremarkable,
property of gradient-boosted trees trained with enough estimators, even
under an AUCPR objective that doesn't explicitly target calibration. Platt
scaling, fit on a few hundred fraud examples, does not clearly improve on
this, and by the stricter metric (Brier) is consistently slightly worse.
This is not the same question as §3 — a model can be well- or poorly
calibrated independently of how well it ranks — but for *this* frozen
model, the calibration story does not support the "Platt scaling will fix
the probabilities" half of the docs' claim either, at least not with the
calibration-set sizes available here without retraining.

---

## 5. Test added

`tests/test_platt_scaling_invariance.py` reproduces the same
calib/final_test split and fits the same `CalibratedClassifierCV` sigmoid,
then asserts:

- ROC-AUC and PR-AUC are unchanged (`abs=1e-6`) between raw and
  Platt-scaled scores on the canonical test partition.
- Zero rank inversions between raw and Platt-scaled scores (tie-aware — a
  block of exactly-tied raw scores is allowed to reorder among itself, but
  must not cross an adjacent block).
- At the documented τ=0.05 threshold, the analytically-equivalent
  Platt-scaled threshold flags the identical set of rows and produces
  identical recall.

This is a regression guard against exactly the failure mode this task was
about: a future change (swapping in isotonic regression, refactoring the
calibration pipeline, or someone re-reading the old docs and assuming Platt
scaling "fixes" recall without re-checking) silently reintroducing an
unverified "calibration moved the recall ceiling" claim. If a genuinely
different, non-monotonic recalibration approach is ever tried and it *does*
move these numbers, this test will fail loudly, which is the correct
outcome — it means the premise changed and deserves fresh scrutiny (see
the docstring in the test file).

`pytest tests/ -v` was run after adding this file: **104 passed** (the
existing 100 plus these 4), no changes to any prior test's outcome.

---

## 6. What this means for README.md and paysense_report.tex

Both documents claimed Platt scaling was "the fix" for the recall ceiling
without it ever having been implemented or tested. That claim is now shown
to be incorrect, in the specific, checkable way theory predicted: Platt
scaling is a rank-preserving remap and cannot move a ranking-based ceiling.
Both documents were updated (this session) to:

- State plainly that Platt scaling was implemented and tested, and that it
  leaves ROC-AUC, PR-AUC, and the recall ceiling completely unchanged
  (0/253 additional fraud rows recoverable at any threshold), matching the
  theoretical expectation that a monotonic transform preserves the full
  precision-recall frontier.
- Reclassify the 69.96% recall ceiling correctly: it is a **ranking /
  discrimination** limitation of the frozen model on this feature set — 76
  of 253 fraud rows are ranked, by the model itself, below the bottom
  decile of all other fraud rows — not a probability-scale artifact that
  recalibration can correct.
- Replace the "Platt Scaling is the fix" recommendation with what would
  actually help: better or additional features that separate those 76
  low-ranked fraud rows from legitimate transactions (the report already
  speculates these are SMOTE-interpolated edge cases near the class
  boundary — that diagnosis of *why* the ceiling exists may still be right;
  only the proposed remedy was wrong), a different base model or ensemble
  member that ranks those specific rows differently, or explicitly
  documenting the ceiling as an accepted property of the current
  feature/model combination pending further data.
- Keep Platt scaling mentioned only for what it actually is: a legitimate
  tool for probability *reliability* (useful for risk-tier UX copy like "70%
  confident this is fraud"), with this document's own Brier/ECE finding
  cited honestly — including that it did not clearly beat raw XGBoost's
  probabilities here — rather than presented as a settled improvement.

---

## 7. Reproducing this check

```
cd PaySense-ML-Backend
venv\Scripts\python.exe platt_scaling_experiment.py
```

Requires the existing `venv/` (no new packages — `CalibratedClassifierCV`
and `brier_score_loss` are both in the already-installed scikit-learn
1.5.1) and `paysense_master_dataset.csv` plus the three frozen artifacts
under `artefacts/`. Produces `artefacts/platt_calibrator.pkl` as a new,
separate, experiment-only artifact; does not modify
`paysense_model.pkl`, `paysense_preprocessor.pkl`, or
`paysense_threshold.pkl`.

Regression test: `pytest tests/test_platt_scaling_invariance.py -v`.

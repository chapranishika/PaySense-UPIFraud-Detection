# PaySense — Out-of-Distribution Generalization: Can It Be Improved, Not Just Diagnosed?

**Date:** 2026-08-23
**Author's intent:** `GENERALIZATION_CHECK.md` and `SYNTHETIC_GROUNDING.md` diagnosed —
but did not attempt to fix — the frozen model's deepest limitation: it scores
*worse* on an independently-generated, full-40-feature synthetic dataset
(ROC-AUC ~0.70) than on a real dataset carrying only 15% of its features
(ROC-AUC ~0.79), evidence the model has partially overfit to
`paysense_pipeline.py`/`paysense_ml_pipeline.py`'s own specific
anchor/supplement correlation structure, not just to which fields happen to
be populated. Everything tried before this document (Platt scaling,
interaction/monotonic constraints) targeted the model's *within-distribution*
recall ceiling. This is the first attempt to address the OOD finding itself.

Script: `ood_generalization_remediation.py`. Results:
`ood_generalization_remediation_results.json`. No retraining or fine-tuning
of the frozen artifacts happened anywhere in this document —
`artefacts/paysense_model.pkl`, `paysense_preprocessor.pkl`, and
`paysense_threshold.pkl` are untouched throughout.

---

## 1. Hypothesis

If the model overfits to one generative process's specific correlation
structure, training on a **blend** of that process's data plus data from a
*differently-structured* generative process should force it to learn
patterns that hold across generation processes, rather than artifacts
specific to one — at some possible in-distribution cost, which must be
reported honestly regardless of direction.

## 2. Methodology, and the leakage discipline that shapes it

Two candidate variants were trained, both starting from the same Block-0
hyperparameters as the currently-frozen model (`paysense_phase3.py`,
including `monotone_constraints` — this experiment does not regress the
already-deployed recall-ceiling fix):

- **Variant A — blended training data.** The original 80% train partition
  (24,000 rows) concatenated with a **newly generated, independently-seeded**
  synthetic dataset (`generate_grounded_synthetic_dataset.py`, `SEED=445566`,
  12,000 rows, 3.81% fraud) — 36,000 rows total (4.08% fraud) before SMOTE.
- **Variant B — heavier regularization, original data only.** Same training
  data as the baseline, but `max_depth=3` (was 5), `reg_lambda=3.0` (was
  1.5), `min_child_weight=20` (was 10) — a control for whether generalization
  is (also, or instead) a variance problem rather than a data-diversity one.

**The leakage risk that had to be designed around**: the pre-existing
`synthetic_grounded_dataset.csv` (seed 918273, built for
`SYNTHETIC_GROUNDING.md`) was kept as a **held-out evaluation set only** —
never touched during training here. A **third**, distinct seed (445566) was
used for the blend-training data specifically so training and evaluation
draw from different generator runs, not the same one relabeled. Verified,
not assumed: `tests/test_ood_generalization_remediation.py` checks
programmatically that no fully-identical row (every generated column except
the purely sequential, seed-independent `transaction_id`) appears in both
files.

All three models (baseline + both variants) were evaluated identically on
four held-out sets, through the real ensemble path
(`src.fraud_model.score()`, via an in-process monkeypatch of the loaded
model/preprocessor — not a permanent code change, and not touching disk
artifacts):

1. The canonical 6,000-row held-out test split (in-distribution — same
   pipeline as training).
2. The held-out synthetic set (seed 918273, full 40 features, never trained
   on here or before).
3. Real Dataset 1 (74,917 rows, 15% of features honestly mappable — the
   primary real-world check).
4. Real Dataset 3 (1,000 rows, 5% of features honestly mappable, 64 fraud —
   low statistical power, secondary only, per `GENERALIZATION_CHECK.md`
   §2.3's own caveat).

**A real bug was found while building this experiment, and fixed.**
Reindexing Dataset 1/3's honestly-mapped columns to the full 40-feature
schema (to build a DataFrame `score()` could consume) turned every unmapped
column into an explicit `None`, which crashed `_score_rules()`'s
`txn_dict.get("failed_attempts_last_24h", 0) > 2` check — `None > 2` raises
a `TypeError` in Python 3. This is the same `.get()`-default-only-fires-on-
absent-key class of bug already fixed for `_score_light_lr()` in an earlier
commit, but that fix hadn't been extended to `_score_rules()`. The
experiment itself worked around it locally by not reindexing (matching
`generalization_check_ensemble.py`'s original, correct approach), but the
underlying bug was real and live in production — any caller sending an
explicit `null` for `failed_attempts_last_24h` (present but empty, not
omitted) would crash `_score_rules()`, which unlike `_score_paysense`/
`_score_light_lr` has no `try/except` around it at all. Note the *other*
three hard-signal checks in that function (`new_device_flag`,
`ip_location_mismatch`, `kyc_verified_flag`, `usr_is_high_risk`) are
actually safe even for `None` — `if txn_dict.get(...):` treats `None` as
falsy rather than crashing, so only the `> 2` numeric comparison was at
risk. **Fixed directly** in `src/fraud_model.py`: `(txn_dict.get(key, 0) or
0) > 2` coalesces both "absent" and "present but null" the same way,
matching the pattern already used two lines above for
`amount_deviation_score`.

---

## 3. Full honest results, every evaluation set, both variants

### 3.1 Canonical held-out test (in-distribution — the cost side of the ledger)

| Model | ROC-AUC | PR-AUC | Recall @ τ=0.30 | Precision @ τ=0.30 |
|---|---:|---:|---:|---:|
| Baseline | 0.8889 | 0.5352 | 40.32% | 86.44% |
| Variant A (blended) | 0.8857 (−0.0032) | 0.5413 (+0.0061) | 38.34% (−1.98pp) | 90.65% (+4.21pp) |
| Variant B (regularized) | 0.8925 (+0.0036) | 0.5404 (+0.0052) | 38.74% (−1.58pp) | 89.09% (+2.65pp) |

Both variants cost a little in-distribution recall and gain a little
precision — a small, unremarkable trade either way. Neither is a
same-pipeline regression worth worrying about on its own.

### 3.2 Held-out synthetic dataset (seed 918273 — different generative process, never trained on)

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Baseline *(cited from `SYNTHETIC_GROUNDING.md` §5.1, recomputed against the monotonic model)* | 0.7000 | 0.1015 |
| Variant A (blended) | **0.7304** (+0.0304) | **0.1309** (+0.0294) |
| Variant B (regularized) | 0.6952 (−0.0048) | 0.1001 (−00014) |

### 3.3 Real Dataset 1 (74,917 rows, primary real-world check)

| Model | ROC-AUC | PR-AUC | TP / 701 @ τ=0.30 | Max score (any row) |
|---|---:|---:|---:|---:|
| Baseline *(cited from `GENERALIZATION_CHECK.md` §4.4, monotonic model)* | 0.7919 | 0.3767 | 0 | 0.0782 |
| Variant A (blended) | **0.8442** (+0.0523) | **0.4764** (+0.0997) | 0 | 0.0847 |
| Variant B (regularized) | 0.8031 (+0.0112) | 0.2250 (**−0.1517**) | 0 | 0.0849 |

### 3.4 Real Dataset 3 (1,000 rows, low power — 64 fraud, directional signal only)

| Model | ROC-AUC | PR-AUC | TP / 64 @ τ=0.30 |
|---|---:|---:|---:|
| Baseline | 0.6046 | 0.1157 | 0 |
| Variant A (blended) | **0.8913** (+0.2867) | **0.3295** (+0.2138) | 0 |
| Variant B (regularized) | 0.6929 (+0.0883) | 0.2207 (+0.1050) | 0 |

---

## 4. Verdict — a real, consistent improvement in ranking, and an honest limit to what that buys

**The hypothesis is confirmed, and the result is not manufactured into
more than it is.** Variant A (blended training) improves ROC-AUC on
**every single held-out evaluation that isn't the canonical same-pipeline
split** — the held-out synthetic set (+0.030), real Dataset 1 (+0.052), and
real Dataset 3 (+0.287, though that one is low-power and should be read
directionally, not precisely). This is a consistent pattern across three
independent checks, two of them real data, not a fluke on one dataset — the
strongest evidence yet that blending generative processes during training
genuinely reduces the model's dependence on `paysense_pipeline.py`'s
specific correlation structure, exactly as hypothesized. PR-AUC improves
alongside ROC-AUC on the two datasets that matter most (held-out synthetic,
Dataset 1), which rules out the improvement being a degenerate ROC artifact
on a wildly imbalanced set.

**But this does not translate into catching more real fraud at the deployed
operating point, and that has to be said as plainly as the good news
above.** Confusion matrices are **identical to the baseline everywhere**:
still 0/701 on Dataset 1, still 0/64 on Dataset 3, regardless of which model
scores them. The reason is visible in the max-score column: Variant A's
highest predicted probability on any of Dataset 1's 74,917 rows is 0.0847 —
nowhere near the deployed threshold of 0.30. **Better ranking is not the
same as a higher absolute score.** A model can correctly rank fraud above
legitimate transactions more often (higher ROC-AUC/PR-AUC) while still never
producing a probability large enough to cross a fixed, in-distribution-tuned
threshold — this is the same lesson `PLATT_SCALING_RESULT.md` established
from the calibration side: ranking quality and operating-point behavior are
genuinely separate properties, and improving one doesn't automatically move
the other.

**Variant B (regularization) is not a good candidate.** It helps marginally
on Dataset 1's ROC-AUC (+0.011) but **hurts its PR-AUC substantially
(−0.152)** and is flat-to-worse on the held-out synthetic set. Heavier
regularization alone does not address this problem — the data-diversity
lever (Variant A) is doing real work that variance reduction alone does not
replicate.

**What this means going forward, stated without softening either
direction:** blending generative processes during training is a genuinely
promising, evidence-backed direction for closing the ranking gap this
project's own OOD checks diagnosed — not a dead end. But it is **not
sufficient on its own to make this model catch more real out-of-distribution
fraud today**, because the deployed threshold was tuned for the
in-distribution score scale, and no variant tested here changes that scale
enough to matter. Two honest follow-up questions this document deliberately
leaves open rather than answering with an unearned confidence: (1) would a
*much* lower, OOD-specific operating threshold applied to Variant A actually
recover real fraud rows, given its improved ranking — untested here, since
every confusion matrix above uses the single deployed threshold on purpose,
to answer "would this help today" rather than "could this help under a
different policy"; (2) does the ranking improvement compound with more
blend data (12,000 rows was one arbitrary choice, not a swept parameter).

**Not recommended for deployment as-is.** Unlike the monotonic-constraints
adoption, this variant shows no improvement at the actual deployed operating
point on the primary real-world check — only in a metric (ranking) that
does not, by itself, produce more caught fraud under the current threshold
policy. Saved as `artefacts/paysense_model_blended_training.pkl` /
`paysense_preprocessor_blended_training.pkl` for any follow-up work; not
wired into `src/fraud_model.py` or `main.py`.

---

## 5. Bug fixed along the way (in `src/fraud_model.py`, not this experiment's own code)

See §2 above for the full account: `_score_rules()`'s
`failed_attempts_last_24h` check crashed on an explicit `null` value
(`None > 2` raises), and had no `try/except` to fail soft the way
`_score_paysense`/`_score_light_lr` do. Fixed with the same `or 0`
coalescing pattern already used elsewhere in that function.

## 6. Reproducing this check

```
cd PaySense-ML-Backend
venv\Scripts\python.exe ood_generalization_remediation.py
```

Takes on the order of an hour — trains two model variants, then scores
baseline + both variants through the real ensemble path against Dataset 1
(74,917 rows, several minutes per model) and Dataset 3. Results are written
to `ood_generalization_remediation_results.json`.

## 7. Following up on the open question — does a lower threshold recover real fraud?

§4 left two honest follow-up questions open rather than answered with
unearned confidence. This closes the first one: **would a much lower,
OOD-specific operating threshold applied to Variant A actually recover real
fraud rows, given its improved ranking?**

`ood_threshold_sweep_variant_a.py` re-scores Dataset 1 and Dataset 3 through
Variant A's already-saved artifact (no retraining — read-only inference,
same `swap_ps_state`/`score_ensemble_dataframe` harness as above) and sweeps
a fine, low threshold grid instead of fixing it at the deployed 0.30. Full
tables in `ood_threshold_sweep_variant_a_results.json`; key rows:

**Dataset 1 (74,917 rows, 701 fraud, the primary real-world check):**

| Threshold | TP | FP | Recall | Precision | FPR | Total flagged |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 (deployed) | 0 | 0 | 0.00% | — | 0.00% | 0 |
| 0.060 | **37** | **0** | 5.28% | **100.00%** | 0.00% | 37 |
| 0.055 | 319 | 283 | 45.51% | 52.99% | 0.38% | 602 |
| 0.050 | 464 | 3,096 | 66.19% | 13.03% | 4.17% | 3,560 |
| 0.035 and below | 701 | 74,216 | 100.00% | 0.94% | 100.00% | 74,917 |

**This is a real, if modest, answer — the first time any check tonight (or
any prior night) found a threshold on any model that catches *any* real
fraud on this dataset.** At τ=0.06, Variant A catches 37 of 701 real fraud
rows with zero false positives — a small slice of real capability that
simply does not exist at the deployed threshold, where every single check
this project has ever run reports exactly 0/701, no exceptions. A more
aggressive threshold (0.055) recovers ten times more fraud (45.5%) but the
cost has to be reported in the same breath: 283 legitimate transactions
would also be flagged. Below 0.035, the model's score distribution on this
dataset stops discriminating anything at all — literally every row, fraud
or not, scores above the threshold, so "100% recall" there is worthless
(0.94% precision, the base fraud rate, i.e. flagging the entire dataset).

**Dataset 3 (1,000 rows, 64 fraud, only 5% of features honestly mappable —
much sparser than Dataset 1's 15%) shows no such usable middle ground.**
The transition is a cliff, not a ramp: 0 recall at threshold 0.04, then
straight to 64/64 (100%) recall with 936/936 (100% FPR) at threshold 0.035
— nothing in between. The remediation that works on Dataset 1 does not
generalize to a dataset with less feature coverage; this is consistent with
every other finding in this project tying generalization quality to how
much of the 40-feature vector a real dataset can honestly supply.

**The necessary caveat, stated as plainly as the finding itself:** this
threshold was found by sweeping directly against Dataset 1's own labels,
not a separate calibration split held out from the sweep itself. The
honest reading of "37/701 @ 100% precision, threshold 0.06" is *"this
specific dataset, examined in retrospect,"* not *"a validated operating
point that would reproduce on the next unseen external dataset."* Reporting
the full curve (not just the best point) is the mitigation available here —
a reader can see exactly how narrow the useful band is (0.050–0.062, a
razor's edge in a model whose scores are otherwise flat) and judge for
themselves rather than take a single cherry-picked number on faith.

**What this changes, and what it doesn't:** this is not a deployment
recommendation. Variant A remains undeployed, for the same reason §4 gave —
no OOD-specific threshold policy exists in `main.py`/`src/fraud_model.py`,
and building one (detecting "this looks like an out-of-distribution
transaction, switch to threshold X instead of the deployed 0.30/0.50")
is a materially different, larger piece of engineering than this document
set out to do. What *does* change: "the model catches literally zero real
out-of-distribution fraud, always, at any threshold anyone has tried" was
true of every prior check in this project and is no longer true — a real,
narrow, quantified, honestly-caveated exception now exists, and it's
documented rather than left as an unexplored dead end.

## 8. Reproducing the threshold sweep

```
cd PaySense-ML-Backend
venv\Scripts\python.exe ood_threshold_sweep_variant_a.py
```

Read-only inference against the already-trained
`artefacts/paysense_model_blended_training.pkl` (no retraining). Takes
about 11 minutes — Dataset 1's 74,917 rows scored row-by-row through the
real ensemble at ~119 rows/sec is the only slow part; Dataset 3 (1,000
rows) takes seconds. Results written to
`ood_threshold_sweep_variant_a_results.json`; pinned in
`tests/test_ood_threshold_sweep_variant_a.py`.

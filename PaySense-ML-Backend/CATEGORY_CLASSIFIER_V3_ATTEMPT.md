# PaySense — Category Classifier v3: Fixing v2's Contamination, Then Actually Measuring the Gap

**Date:** 2026-08-23

## 0. A prior attempt was invalidated — here's exactly what went wrong, so it isn't repeated

An earlier attempt tonight (`generate_category_training_v2.py`,
`artefacts/paysense_category_classifier_v2.pkl`) reported **97.5% accuracy**
on `category_generalization_test_set.csv` (the 200-row gold, hand-authored,
held-out evaluation set from `CATEGORY_CLASSIFIER_GENERALIZATION.md`) — up
from the deployed classifier's 72.5%. **That number is invalid and must not
be cited.** The agent that built it, and the prompt that briefed it, both
read the eval set's actual content (the briefing prompt quoted three of its
sentences verbatim as "example failures to check"). The resulting v2
training templates turned out to be the eval set's own sentence skeletons
with only the merchant name swapped — e.g. the eval file's `"amt rs.499
dedcuted a/c XX7788 for BARBEQUE NATION on 05-08-26 upi ref
309112233445"` versus v2's `"amt rs.78 dedcuted a/c XX7978 for FRESHMENU
ORDER on 20-08-20 upi ref 164785176038"`. Confirmed by direct comparison of
multiple rows, not an isolated coincidence. The existing digit-collapsing
disjointness check (`tests/test_category_generalization.py`) did **not**
catch this, because it only masks digits, not merchant-name substitutions —
a real gap in that check, now understood.

**This attempt (v3) avoided the same failure by construction, not by
willpower alone:**
- `generate_category_training_v3.py`'s templates were written from general
  knowledge of Indian bank/UPI SMS conventions, without opening
  `category_generalization_test_set.csv` during design.
- The orchestrating session had, by this point, already seen several exact
  eval-set sentences during the investigation that caught v2's problem —
  so instead of relying on memory discipline alone, **two independent
  programmatic checks** were run before any training happened: (1) the
  existing digit-collapsing check, and (2) a stronger check that also masks
  sequences of capitalized words (merchant/proper-noun tokens) and
  account-number patterns before comparing. Both returned **zero overlap**
  between the 8,000 v3 training rows and the 200 eval rows (1,157 unique
  v3 skeletons vs. 116 unique eval skeletons under the stronger mask, 0
  shared). The generator script itself refuses to write its output CSV if
  either check finds any overlap — a hard gate, not a warning.

## 1. Construction method

`generate_category_training_v3.py` builds 8,000 rows (1,600/class) from:
- 5 merchant/vendor pools (~25-40 distinct real brand/service names per
  class — Blinkit, Zepto, Licious, Milkbasket for Food; Zerodha, Groww,
  Sovereign Gold Bond, NPS for Investment; NACH mandate, ECS housing loan,
  Bajaj Finserv for EMI; IRCTC, Ola, OYO, FASTag for Travel; Myntra, Croma,
  Netflix, Urban Company for Shopping — deliberately broader than the
  handful of brands FinText-6K's 40 templates use).
- 7-9 sentence-connector *structures* per class (different word order,
  verb choice, and formatting conventions — bank-first vs. merchant-first
  vs. amount-first phrasing, NACH/SI/standing-instruction language for
  EMI, application/premium/folio language for Investment), each filled
  with a randomly chosen bank name (13 real Indian banks), payment app,
  account number, reference number, date, and amount.

This is templated (unlike the 200-row eval set, which is entirely
hand-written), so it inherits some of the same "fixed shape, varying
values" limitation FinText-6K has — but with roughly 4x the distinct
connector structures per class and far more vendor/bank variety, it is
meaningfully more diverse than FinText-6K's single 40-template family, and
verified structurally disjoint from the *evaluation* set specifically,
which is what matters for an honest measurement.

## 2. Results

Trained on FinText-6K's 5,000-row train split blended with the 8,000-row
v3 synthetic set (13,000 rows total). Evaluated on two held-out sets, both
untouched during training:

| Evaluation set | Accuracy | Correct & confidence≥0.65 | Gate pass rate |
|---|---:|---:|---:|
| FinText-6K's own held-out test (in-distribution) | **100.0%** (no regression) | 100.0% | 100.0% |
| Gold novel eval set (`category_generalization_test_set.csv`) | **78.0%** (145→156/200) | **70.5%** (was 62.0%) | 86.0% |

Compare against the documented baseline (deployed v1 classifier, same eval
set): **72.5% → 78.0%** accuracy (+5.5pp), **62.0% → 70.5%**
correct-and-confident (+8.5pp). This is a real, modest, believable
improvement — nowhere near v2's invalidated 97.5%, which is itself strong
supporting evidence that v2's number was almost entirely a contamination
artifact rather than a genuine effect of more training diversity.

### Per-class (gold eval set)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| EMI | 0.85 | **1.00** | 0.92 |
| Food | 0.54 | 0.80 | 0.65 |
| Investment | **1.00** | 0.75 | 0.86 |
| Shopping | 0.90 | 0.68 | 0.77 |
| Travel | 0.79 | 0.68 | 0.73 |

EMI still generalizes best (a rare, distinctive token — unchanged from the
v1 finding). Food/Travel/Shopping still leak into each other, though less
than before; Food in particular still over-predicts (recall 0.80 but
precision only 0.54 — it's now a bigger false-positive sink than before,
trading one imbalance for a milder one).

### Specific documented failures, re-checked directly

Three specific misclassifications cited in `CATEGORY_CLASSIFIER_GENERALIZATION.md`
§4.2 were re-run against v3:

| Text (true label) | v1 (deployed) | v3 (this attempt) |
|---|---|---|
| "Sent Rs.1250 to THE COFFEE HOUSE using Amazon Pay UPI" (Food) | Shopping (0.962) | **Food (0.930)** — fixed |
| "IPO application blocked amount Rs 15000 - ZOMATO IPO ASBA..." (Investment) | Food (0.957) | **Investment (0.972)** — fixed |
| "EazyDiner reservation advance Rs 500 debited via UPI..." (Food) | Travel (0.977) | Travel (0.488) — still wrong, but now **below the 0.65 confidence gate** |

Two of three flip to correct. The third is still wrong but meaningfully
different in kind: v1 was confidently wrong (0.977, would silently
misclassify in production); v3 is uncertain (0.488, below the app's gate),
which means it would now correctly fall through to Tier-3 human
confirmation instead of confidently getting it wrong. That's a real
improvement in production safety even where the raw prediction is still
incorrect.

## 3. Verdict

**More training diversity helps, genuinely, but only modestly — the
underlying model is still a word-level lookup, not a narration-understanding
system.** The 5.5pp accuracy gain and 8.5pp confidence-gate gain are real
and traceable to broader vocabulary (more merchants, more banks, more
sentence structures), not to eval-set memorization (verified by construction
this time). But EMI's perfect recall versus Food/Travel/Shopping's
continued mutual confusion shows the same underlying limitation persists:
a linear model over TF-IDF n-grams generalizes well when a category has a
rare, unambiguous keyword (EMI) and poorly when categories share
vocabulary and the distinguishing signal is closer to semantic context
than literal words (a "reservation" for a restaurant vs. a bus; "Amazon
Pay" as a payment rail vs. a marketplace). Materially closing the rest of
this gap likely needs a representation that captures some of that context
— `paysense_report.tex`'s existing recommendation (a small on-device
DistilBERT model) rather than more TF-IDF training data, though more data
in the meantime is a real, free, low-risk improvement worth keeping.

Not wired into `main.py`/`src/fraud_model.py` at the time this section was
written — saved as `artefacts/paysense_category_classifier_v3.pkl` for a
follow-up deployment decision, same as every other candidate artifact
produced tonight.

**Deployed 2026-08-24.** `src/fraud_model.py` loads
`artefacts/paysense_category_classifier.pkl` by a fixed filename, so
deployment was a file swap, not a code change: the original (v1) artifact
was archived as
`artefacts/paysense_category_classifier_v1_deployed_until_2026-08-24.pkl`,
and this v3 pipeline was copied into the canonical filename in its place.
Verified before and after via `tests/test_category_generalization.py`
(pinned numbers updated to v3's: 78.0% accuracy / 0.7849 macro F1 / 86.0%
confidence-gate pass rate on the 200-row novel eval set) and
`tests/test_api.py`'s `TestClassify` class, which exercises the deployed
model through the real `/classify` endpoint — both pass. v4 (§3.5,
marginally better gate-pass-rate but dependent on re-downloading external
Kaggle data to reproduce, with no accuracy gain over v3) was considered and
not chosen, to keep the deployed model's training pipeline self-contained.

## 3.5. v4 — adding real Kaggle data on top of v3

The invalidated v2 attempt had, before its templates went wrong, done
legitimate and valuable work finding and vetting real Kaggle datasets — that
vetting is separate from the contaminated templates and is reused here (see
`external_data/NOTICE.md` for the full record). Two datasets survived vetting:

- **`coderanand/indian-banking-transaction-text-dataset`** (Apache 2.0,
  11,000 rows, Food/Travel/EMI/Investment/Shopping) — its own description
  states it is synthetic; independently verified only 29 unique
  digit-collapsed phrases exist across all 11,000 rows. Used for its
  distinct vocabulary (Zerodha, Groww, IRCTC, IndiGo, McDonald's, Bajaj
  Finserv, PPF — none in FinText-6K's 40 templates), not represented as
  real free text.
- **`bhavyasingh25/financial-transaction-description-dataset`** (MIT, 9
  classes) — only the 5 rows matching PaySense's taxonomy kept (3,280 of
  6,000; entertainment/healthcare/education/utilities dropped, not forced
  into a mapping nobody asked for). Also self-evidently templated (28
  unique phrases).
- One dataset found during vetting (`apoorvwatsky/bank-transaction-data`,
  116,201 real corporate transfer rows) was rejected — no row carries any
  consumer spending-category signal. Another
  (`engreemali/bank-transactions-sms-datasetss`, genuinely real UAE bank
  SMS with real merchant narrations) was also rejected — zero category
  labels in the shipped file; hand-labeling it ourselves would be
  self-authored labels dressed up as a real dataset. Both re-verified as
  correctly rejected, not just cited secondhand.

Both usable datasets were re-downloaded fresh (Kaggle's public,
unauthenticated download endpoint works in this environment — confirmed
directly, not assumed) and checked for exact-text overlap against the gold
eval set (0 in both). Blended with FinText-6K + v3's synthetic templates:
27,280 total training rows.

| Evaluation set | v3 (FinText-6K + own templates) | v4 (+ 2 real Kaggle sources) |
|---|---:|---:|
| FinText-6K's own test (in-distribution) | 100.0% | 100.0% (no regression) |
| Gold novel eval set — accuracy | 78.0% | 78.0% (unchanged) |
| Gold novel eval set — correct & confident | 70.5% | 72.5% (+2pp) |
| Gate pass rate | 86.0% | 86.5% |

**Honest reading: adding real-download Kaggle data bought almost nothing
beyond what v3's own broader templates already provided, and the reason is
diagnosable, not mysterious** — both Kaggle sources are, by their own
description and independently verified, themselves narrowly templated
(29 and 28 unique phrases respectively) rather than genuine free-form real
narrations. "Real dataset" and "genuinely diverse text" turned out to be
different properties in this search: a dataset can be authentically
downloaded from a real source and still be synthetic-and-narrow underneath.
The two datasets that *were* genuinely real free text found during vetting
(`apoorvwatsky`, `engreemali`) were both unusable for a different reason
each (no category signal; no labels at all) — real, diverse, *and*
correctly-labeled UPI narration text simply was not found in this search.
v3's own hand-built template diversity remains the larger lever of the two
tried tonight.

Saved as `artefacts/paysense_category_classifier_v4.pkl` — not deployed,
same as v3.

## 4. Reproducing this check

```
cd PaySense-ML-Backend
venv\Scripts\python.exe generate_category_training_v3.py
venv\Scripts\python.exe train_category_classifier_v3.py

REM v4 additionally re-downloads the two vetted Kaggle datasets into
REM external_data/ (requires internet access) before blending:
venv\Scripts\python.exe train_category_classifier_v4.py
```

The generator refuses to write output if it finds any overlap (exact or
merchant-masked-structural) with `category_generalization_test_set.csv`.
`train_category_classifier_v4.py` requires the two Kaggle CSVs to already
be present under `external_data/kaggle_coderanand_indian_banking/` and
`external_data/kaggle_bhavya_financial_transaction/` (downloaded via
Kaggle's public dataset-download endpoint, see `external_data/NOTICE.md`).

# PaySense — Category Classifier Generalization Check

**Date:** 2026-08-23
**Author's intent:** `CATEGORY_CLASSIFIER.md` reports 100% accuracy on FinText-6K's
held-out test set and honestly discloses why that number is hollow: the
entire 6,000-row dataset (train + test) is generated from only 40 fixed
sentence templates, so train and test share the exact same 40 shapes and
there is no linguistic variation left to fail on. Nobody had ever tested the
frozen classifier (`artefacts/paysense_category_classifier.pkl`, loaded
read-only here — no retraining, no fine-tuning, no threshold recalibration
anywhere in this document) against text it wasn't trained to recognize the
*shape* of. This document builds that test and reports the real number,
whatever it turned out to be.

Scripts: `build_category_generalization_test_set.py` (dataset, hand-authored),
`score_category_generalization.py` (scoring, via the same
`predict_proba()` / `named_steps["clf"].classes_` call `src/fraud_model.py`'s
`classify_category()` uses).

---

## 1. Confirming exactly how templated FinText-6K is

`CATEGORY_CLASSIFIER.md` states the dataset is "generated from only 40
unique templates." This was verified directly, not taken on faith: every
row's `text` field (train + test, 6,000 rows total) was normalized by
collapsing digit runs to `#`, and the unique normalized strings were
counted.

```
Total rows (train+test): 6,000
Unique normalized structures: 40
```

The 40 templates, by class (extracted verbatim, digits replaced with `#`):

| Class | Count | Templates |
|---|---:|---|
| EMI | 7 | Personal loan EMI of Rs # via UPI Ref # · Loan EMI deduction of Rs # via UPI Ref # · Credit card EMI payment of Rs # via UPI Ref # · Mobile EMI deduction of Rs # via UPI Ref # · Car loan EMI of Rs # via UPI Ref # · Appliance EMI payment of Rs # via UPI Ref # · Home loan installment of Rs # via UPI Ref # |
| Food | 9 | Restaurant payment of Rs # via UPI Ref # · Lunch combo purchase of Rs # via UPI Ref # · Pizza outlet transaction of Rs # via UPI Ref # · Dinner at hotel of Rs # via UPI Ref # · Zomato food delivery of Rs # via UPI Ref # · Bakery payment of Rs # via UPI Ref # · Cafe bill of Rs # via UPI Ref # · Grocery store purchase of Rs # via UPI Ref # · Swiggy order of Rs # via UPI Ref # |
| Investment | 7 | Mutual fund SIP of Rs # via UPI Ref # · Demat account funding of Rs # via UPI Ref # · FD deposit of Rs # via UPI Ref # · Gold bond investment of Rs # via UPI Ref # · Crypto purchase of Rs # via UPI Ref # · Insurance premium investment of Rs # via UPI Ref # · Stock market investment of Rs # via UPI Ref # |
| Shopping | 8 | Supermarket shopping of Rs # via UPI Ref # · Electronics store purchase of Rs # via UPI Ref # · Online shopping transaction of Rs # via UPI Ref # · Clothing store payment of Rs # via UPI Ref # · Mall shopping bill of Rs # via UPI Ref # · Fashion outlet bill of Rs # via UPI Ref # · Flipkart purchase of Rs # via UPI Ref # · Amazon order payment of Rs # via UPI Ref # |
| Travel | 9 | Train ticket purchase of Rs # via UPI Ref # · Hotel booking of Rs # via UPI Ref # · Bus reservation of Rs # via UPI Ref # · Flight ticket booking of Rs # via UPI Ref # · Metro recharge of Rs # via UPI Ref # · Travel agency payment of Rs # via UPI Ref # · Petrol pump payment of Rs # via UPI Ref # · Ola cab booking of Rs # via UPI Ref # · Uber ride payment of Rs # via UPI Ref # |

**The finding is more extreme than "40 templates" alone suggests: every
single one of the 6,000 rows, with no exceptions, matches the exact regex**

```
^[A-Za-z ]+ of Rs [0-9]+ via UPI Ref [0-9]+$
```

**— i.e. the *only* thing that varies anywhere in the entire dataset is (a)
which of 40 fixed leading noun phrases is used, and (b) the two numbers.**
The suffix `" of Rs # via UPI Ref #"` is identical, word-for-word, on every
row of every class. This is the ceiling `CATEGORY_CLASSIFIER.md` already
warned about, quantified precisely.

---

## 2. Building a genuinely novel test set

### 2.1 Construction method

`build_category_generalization_test_set.py` hand-authors 200 examples (40
per class) of realistic Indian bank-SMS / UPI-app narration text. None are
templated or formula-generated — each was written individually, modeled on
real narration conventions this project's other documents already
established as realistic (`SYNTHETIC_GROUNDING.md` §3 used a real personal
bank-statement export, `MyTransaction.csv`, as a grounding reference; the
formats below are the same family of real SMS/app text):

- **Bank debit-SMS formats** from HDFC, SBI, ICICI, Axis, Kotak — each bank
  has a different real narration convention (`UPI/DR/<ref>/<merchant>/<bank>`,
  `Dear Customer, Rs.X debited ... Refno ...`, `INR X debited ... Info:
  UPI/<ref>/<merchant>/UPI`, `Acct XXNNNN debited INR X ... to <merchant>
  -Axis Bank`, etc.)
- **UPI-app in-app notification text** — Google Pay ("You paid ₹X to
  `<merchant>` using Google Pay"), PhonePe ("Paid Rs X to `<merchant>` via
  PhonePe. Txn ID T..."), Paytm ("Rs X paid to `<merchant>` via Paytm UPI.
  Order ID ...")
- **IMPS / NACH / ECS / standing-instruction narrations** — `IMPS-P2A-<ref>-
  <merchant>-<bank>-Rs.X`, `NACH-DR-<biller>-EMI-XXXXXX<acct>-Rs.X`,
  `SI EXECUTED-<purpose>-A/c XX<acct>-Rs.X-<bank>`, `ECS-DR-<ref>-<biller>-
  <bank>`
- **Merchant names not in the 40 templates**: Barbeque Nation, Behrouz
  Biryani, Country Delight, Bikanervala, EazyDiner, Zoomcar, RedBus, Vistara,
  IRCTC, Cleartrip, Bajaj Finserv, Tata Capital, IIFL, HDB Financial,
  Muthoot, Zerodha, Groww, Upstox, WazirX, CoinDCX, Angel One, Myntra,
  Croma, Nykaa, Lenskart, Tanishq, Decathlon, Pepperfry, and ~150 others.
- **Real-world SMS noise**: typos and shorthand actually seen in bank SMS
  and casual UPI-app copy (`dedcuted`, `paymnet`, `recieved`, `oder`,
  `paymnt`, `dr`/`cr` abbreviations, inconsistent capitalization,
  `A/c`/`Acct`/`a/c`, `4` for "for", `2` for "to").

Every example's category is unambiguous to a human reader on inspection —
this test was built to measure real generalization, not to manufacture
failures with deliberately ambiguous text. One labeling convention was
carried over from FinText-6K's own (slightly inconsistent) grouping rather
than re-litigated: FinText-6K puts "Grocery store purchase" under **Food**
but "Supermarket shopping" under **Shopping** — the new test set's grocery/
kirana-store examples are labeled Food and its department-store/electronics/
clothing examples are labeled Shopping, matching the source dataset's own
convention so the comparison is apples-to-apples.

### 2.2 Verified structurally disjoint from the 40 templates

Both checks below are load-bearing and are also encoded as regression tests
in `tests/test_category_generalization.py`:

```
FinText-6K rows matching the 40-template suffix pattern:  6,000 / 6,000
New test set rows matching that same pattern:                  0 / 200
New rows whose normalized structure exactly matches
  a FinText-6K row (digit-collapsed):                          0 / 200
```

Zero overlap, either by the exact suffix pattern or by the same
digit-collapse normalization used to extract the 40 templates in §1.

### 2.3 Representative sample (10 of 200 — full set in `category_generalization_test_set.csv`)

| text | label |
|---|---|
| `Dear Customer, Rs.245.00 debited from A/c XX3456 on 14Aug26 trf to ZOMATO ONLINE Refno 430987654321. If not done by u, fwd this sms to 9223008333 -SBI` | food |
| `IMPS-P2A-398217364501-FASTAG RECHARGE NHAI-HDFC BANK-Rs.500.00` | travel |
| `EMI DEBIT-BAJAJ FINSERV-LOANID BJ4512367-Rs 3450.00-HDFC0001234` | EMI |
| `NACH mandate executed: Rs 5000 debited towards NPS CONTRIBUTION, Ref NACH8891299` | investment |
| `You paid ₹15600 towards Samsung Mobile Store Purchase using PhonePe UPI` | shopping |
| `Milk & dairy subscription auto-debit Rs 780 - COUNTRY DELIGHT, UPI Ref 391827364500` | food |
| `Rs 500 paid to Parking Lot Attendant via Paytm, Order#PK9012` | travel |
| `Auto-debit failed: EMI Rs 2300 could not be processed - insufficient balance, HDB FINANCIAL` | EMI |
| `stok purchse paymnt rs 8900 debited, ref no 391827364522 UPSTOX SECURITIES` | investment |
| `jwelry purchse rs 25000 tanishq showrom paymnt thru card linked upi` | shopping |

Full CSV (`text,label`, 200 rows, same raw label casing as FinText-6K):
`category_generalization_test_set.csv`.

---

## 3. Results

Full script output reproducible via `python score_category_generalization.py`.

### 3.1 Headline number

```
Accuracy on FinText-6K held-out test set (CATEGORY_CLASSIFIER.md):  1.0000  (1,000/1,000)
Accuracy on this novel, structurally disjoint test set:              0.7250  (145/200)
```

**A 27.5-point drop.** This is a real, meaningful, diagnosable gap — not
noise, and not softened here.

### 3.2 Per-class precision / recall / F1

```
              precision    recall  f1-score   support

         EMI     0.8889    1.0000    0.9412        40
        Food     0.4828    0.7000    0.5714        40
  Investment     0.6735    0.8250    0.7416        40
    Shopping     0.8750    0.5250    0.6562        40
      Travel     0.9583    0.5750    0.7188        40

    accuracy                         0.7250       200
   macro avg     0.7757    0.7250    0.7258       200
weighted avg     0.7757    0.7250    0.7258       200
```

### 3.3 Confusion matrix

Rows = true, cols = predicted, class order `[EMI, Food, Investment, Shopping, Travel]`:

```
              EMI  Food  Inv  Shop  Trav
EMI         [  40    0    0    0    0 ]
Food        [   0   28    8    3    1 ]
Investment  [   0    7   33    0    0 ]
Shopping    [   4   13    2   21    0 ]
Travel      [   1   10    6    0   23 ]
```

**EMI generalizes perfectly (100% recall, 0 false negatives)** — the token
"EMI" is distinctive, rare outside the EMI class in the training vocabulary,
and appears literally in almost every real-world EMI narration, so this
holds up regardless of surrounding sentence structure.

**Every other class leaks into Food or Investment.** Reading the Food
column: on top of its own 28 correctly-classified rows, Food absorbs 7 true
Investment rows, 13 true Shopping rows, and 10 true Travel rows (30
misattributed rows total) — the largest false-positive sink of the five.
Investment similarly absorbs 8 true Food rows and 6 true Travel rows.
Shopping and Travel have the highest *precision* (0.875, 0.958 — when the
model says Shopping or Travel it's usually right) but the worst *recall*
(0.525, 0.575 — it frequently fails to recognize genuine Shopping/Travel
text as such, defaulting to Food or Investment instead).

### 3.4 Confidence-gate pass rate (Android's `NLP_CONFIDENCE_THRESHOLD = 0.65`)

A correct-but-low-confidence prediction still falls through to Tier-3 HITL
in production — it is not a Tier-2 success. Both numbers matter separately:

```
Mean top-class confidence                              : 0.7537
Fraction of predictions clearing the 0.65 gate          : 0.6700  (134/200)
Accuracy among predictions that clear the gate          : 0.9254  (124/134)
Correct AND clears the gate (real production Tier-2 rate): 0.6200  (124/200)
```

So of 200 realistic narrations, only **62%** would actually be resolved
correctly by Tier 2 in production; the other 38% either get the wrong
category outright, or get gated out to Tier-3 human review even when the
underlying prediction happened to be right (21 of the 66 below-gate
predictions were correct but not confident enough to count).

---

## 4. Diagnosis: why this happens, quantified

This is not a vague "it overfit" claim — the mechanism was traced directly
by inspecting the fitted `TfidfVectorizer`'s vocabulary against the failing
examples.

### 4.1 The vocabulary is tiny and entirely template-derived

```
Fitted TF-IDF vocabulary size: 821 tokens (unigrams + bigrams)
```

821 tokens is what `min_df=2` extracts from 5,000 training rows built out of
only 40 sentence shapes — every token in the vocabulary is either a content
word from one of the 40 templates (`restaurant`, `zomato`, `mutual`, `fund`,
`sip`, `flight`, `amazon`, ...) or one of the small set of connective words
shared by *all* 40 templates (`of`, `rs`, `via`, `upi`, `ref`, and the
bigrams `upi ref`, `via upi`).

### 4.2 Two distinct, quantified failure modes

Every one of the 200 novel-test-set rows was bucketed by whether its
TF-IDF vector, after transformation, contains **only** those shared
connective tokens (i.e., zero real content-word overlap with the training
vocabulary) or contains at least one genuine content word:

```
Rows with ONLY generic connectives (rs/ref/upi/via) or zero overlap: 50/200 (25.0%)  -> accuracy 0.300, mean confidence 0.297
Rows with at least one training-vocabulary content word matched:    150/200 (75.0%)  -> accuracy 0.867, mean confidence 0.906
```

(0.75 × 0.867 + 0.25 × 0.300 = 0.725 — this exactly reconstructs the
headline accuracy, confirming the bucketing explains the result rather than
being a post-hoc coincidence.)

**Failure mode A — vocabulary collapse to a fixed default.** When a
sentence's real content words (bank codes, ref numbers, narration
boilerplate like "debited", "dedcuted", "A/c XX9021 dr with") don't overlap
the 821-token training vocabulary at all, the TF-IDF vector degenerates to
just the shared connective tokens, and — critically — **every such sentence
produces the identical predicted-probability vector, regardless of its true
category**:

```
{'EMI': 0.090, 'Food': 0.293, 'Investment': 0.306, 'Shopping': 0.085, 'Travel': 0.225}
```

This was confirmed directly: four different rows from three different true
classes (Food, Food, Travel, Shopping) — e.g. `"amt rs.499 dedcuted a/c
XX7788 for BARBEQUE NATION on 05-08-26 upi ref 309112233445"` (true: Food)
and `"A/c XX9021 dr with Rs 999 towards LENSKART EYEWEAR on 12-Aug UPI Ref
309887712322"` (true: Shopping) — all produced this exact same probability
vector and were both predicted **Investment** at confidence 0.306, because
Investment happens to be whichever class's sigmoid-calibration curve sits
highest at a near-zero decision-function margin. The model isn't guessing
based on content here; it has no content to go on, and collapses to a fixed
default that has nothing to do with the true label.

**Failure mode B — single-keyword lookup, not narration understanding.**
When a sentence *does* contain one training-vocabulary content word, that
one word can dominate the prediction with high (>0.95) confidence even when
it's semantically the wrong cue for that sentence:

- `"Sent Rs.1250 to THE COFFEE HOUSE using Amazon Pay UPI"` (true: **Food**
  — a coffee shop payment) → predicted **Shopping**, confidence **0.962**.
  The word "Amazon" — trained exclusively on "Amazon order payment of Rs #"
  (Shopping) — completely dominates, even though "Amazon Pay" here is just
  the payment rail, not the merchant.
- `"IPO application blocked amount Rs 15000 - ZOMATO IPO ASBA, Ref
  391827364511"` (true: **Investment** — a stock-market IPO application) →
  predicted **Food**, confidence **0.957**. "Zomato" was trained exclusively
  as a food-delivery keyword ("Zomato food delivery of Rs #"), so the model
  has no way to recognize the same brand name in an investment context.
- `"EazyDiner reservation advance Rs 500 debited via UPI, ref
  398271650912"` (true: **Food** — a restaurant table booking) → predicted
  **Travel**, confidence **0.977**. The word "reservation" was trained
  exclusively via "Bus reservation of Rs #" (Travel), so any reservation
  language gets pulled toward Travel regardless of what's being reserved.

Both failure modes trace to the same root cause: with only 40 training
sentence shapes, the fitted decision boundary is a lookup table over ~821
literal tokens, not a semantic model of transaction narration. It performs
essentially at its documented ceiling (86.7% accuracy, mean confidence 0.91)
whenever a novel sentence happens to reuse one of those tokens, and collapses
to a near-random, wrong-by-construction default (30.0% accuracy — barely
above the 20% random baseline for 5 classes) whenever it doesn't.

---

## 5. Verdict — no win or failure manufactured

**The documented 100% accuracy is real for what it measures (FinText-6K's
own 40 template shapes) and does not transfer to realistic novel phrasing.**
On a 200-row, hand-authored, structurally disjoint test set built from real
Indian bank-SMS and UPI-app narration conventions, the frozen classifier
scores **72.5% accuracy**, and only **62.0%** of predictions are both
correct and confident enough to clear the app's Tier-2 gate in production —
meaning **38% of realistic transactions would not be resolved correctly by
Tier 2 today**, either misclassified outright or correctly-but-not-
confidently falling through to Tier-3 human review.

This is not uniformly bad news, and that is reported plainly too: **EMI
generalizes essentially perfectly** (100% recall, 88.9% precision) because
"EMI" is a distinctive, rare token that survives into real-world phrasing
almost by definition. The failure is concentrated in the four classes whose
training vocabulary consists of common words and common brand names that
collide with other classes' real-world usage (grocery/food-adjacent
merchant names, travel-adjacent verbs like "reservation", payment-rail names
like "Amazon Pay" that aren't the actual merchant) — a genuinely diagnosable,
specific mechanism (§4), not a vague "the model is bad" claim.

### 5.1 Comparable to `SYNTHETIC_GROUNDING.md`'s finding for the fraud model?

**Yes — the same qualitative pattern holds across both models in this
project, which makes it a credible, consistent finding rather than a
one-off.** `SYNTHETIC_GROUNDING.md` found the frozen XGBoost fraud model
ranks and calibrates *worse* on a full-feature dataset from a different
generative process than on a sparse-feature dataset from a real one — direct
evidence the model overfit to its own training pipeline's specific
correlation structure, not just to which features were present. The category
classifier shows the text-domain analogue: it overfit to its own training
pipeline's specific *sentence structure and 821-token vocabulary*, not to a
genuine understanding of what these five categories mean. In both cases, a
model trained on a narrow, formulaically-generated synthetic distribution
learned a decision surface that only reliably fires inside that
distribution's specific shape — high held-out accuracy on data drawn from
the same generator, and a real, quantified degradation the moment the
generator's specific structure is no longer present. The category
classifier's degradation (100% → 72.5%, −27.5 points) is measured on a
different axis than the fraud model's (ROC-AUC 0.89 training vs. 0.60–0.70
external, a ranking-quality metric, not accuracy), so the two numbers are
not directly comparable in magnitude — but the *shape* of the finding is the
same: synthetic, narrowly-templated training data produces a model that
looks excellent on its own held-out split and measurably underperforms the
moment it meets data generated by a genuinely different process, even
within the same problem domain.

**What this proves:** the classifier has learned real, useful signal for at
least one class (EMI) and partial signal for the others (75% of novel rows
that happen to share vocabulary with training score at 86.7% — well above
chance). **What this does not prove, and what this document will not
soften:** the classifier is not ready to be trusted as a silent Tier-2
resolver for arbitrary real-world UPI narration text — 25% of realistic
transactions in this check contained no recognizable training vocabulary at
all and were scored close to a random guess. The app's existing three-tier
design (keyword table → NLP classifier → human-in-the-loop Tier 3, gated by
the 0.65 confidence threshold) is exactly the right architecture for a model
with this profile: the confidence gate does correctly catch some of the
damage (67% of predictions clear it, and those are 92.5% accurate), but it
does not catch all of it — Failure Mode B's high-confidence wrong answers
(0.96–0.98 confidence, wrong label) sail straight through the gate and would
reach the user's transaction history mislabeled.

---

## 6. Reproducing this check

```
cd PaySense-ML-Backend
venv\Scripts\python.exe build_category_generalization_test_set.py   # writes category_generalization_test_set.csv (200 rows)
venv\Scripts\python.exe score_category_generalization.py            # scores it via the frozen artifact, writes artefacts/category_generalization_metrics.json
```

Regression tests (structural disjointness + pinned accuracy/confidence-gate
numbers):

```
venv\Scripts\python.exe -m pytest tests/test_category_generalization.py -v
```

Requires `E:\Projects\upi\FinText-6K\{train,test}_transaction_dataset.csv`
(for the disjointness check against the source templates) and the existing
`venv/` — no new dependencies were installed and
`artefacts/paysense_category_classifier.pkl` was not modified anywhere in
this document.

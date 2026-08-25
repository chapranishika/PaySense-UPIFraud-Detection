# PaySense — Interview Preparation

Every answer below is grounded in something verifiable in this repository —
a file, a commit, a test, a number from `WALKTHROUGH.md` or this audit.
Where the honest answer is "I don't know" or "not measured," that's what's
written, because a confident wrong answer is worse than an honest gap in a
real interview.

---

## Beginner questions

**Q: What does the project do?**
A real-time UPI fraud detection system. An Android app reads incoming bank
SMS on-device, extracts the transaction details, categorizes it, and sends
it to a backend that scores it for fraud risk using a 3-model ensemble
personalized to that user's own spending history.

**Q: Why did you build it?**
UPI fraud detection today is mostly server-side and invisible to the user
until money is already gone. This tries to catch it at confirmation-SMS
time, and to use per-user personalization (what's normal for *you*) rather
than only population-level rules.

**Q: What is the input?**
Two kinds: (1) a bank confirmation SMS on the Android side, parsed into
amount/payee/transaction ID/date; (2) a JSON transaction payload on the
backend side (40 features — amount, device flags, behavioral z-scores,
user profile fields).

**Q: What is the output?**
A fraud score (0–1), a binary decision, and a graduated alert level
(none/low/medium/high) — plus, separately, a spending category and an AI-
generated savings tip.

**Q: What dataset did you use?**
Two, for two different models. Fraud scoring: a 30,000-row blend of a
real-style 20K-row anchor and a 10K-row synthetic supplement. Category
classification: FinText-6K, a 6,000-row labeled bank-narration text
dataset. Full detail in `DATASET.md`.

**Q: What model did you use?**
Not one model — a weighted ensemble: an always-on rules scorer (weight
0.15), an XGBoost model trained on all 40 features with monotonic
constraints on three behavioral features (weight 0.60), and a lightweight
5-feature logistic regression that can also run standalone if the XGBoost
artifact is unavailable (weight 0.25).

---

## Intermediate questions

**Q: Why this architecture (ensemble of three, not one model)?**
Redundancy and interpretability. LightLR can serve alone if XGBoost's
artifact is missing (verified in `main.py`'s `_mock_score()` fallback path
and `fraud_model.py`'s active-scorer renormalization). The always-on rules
scorer encodes known, auditable fraud signals (e.g. a cold-start bonus for
new payees above ₹5,000) that don't depend on the ML model having seen
anything like this pattern in training.

**Q: Why this specific model (XGBoost) over alternatives?**
**NOT DIRECTLY DOCUMENTED** in the repo as an explicit alternatives-
considered comparison — I'd be honest about that rather than invent a
justification. What *is* documented: XGBoost with monotonic constraints on
`amount_deviation_score`/`transaction_velocity`/`failed_attempts_last_24h`
recovered 10 previously-missed fraud rows while *improving* both raw ROC-
AUC and PR-AUC, the only one of three tested remediation variants with no
measurable downside (`RECALL_CEILING_REMEDIATION.md`).

**Q: How did you preprocess the data?**
50 raw columns engineered down to 40 model-ready features
(`eda_feature_engineering.py`/`paysense_phase3.py`), SMOTE applied to the
training partition only (24K → 45,980 rows) — after the split, not before,
which is the direction that avoids synthetic-neighbor leakage into test
data.

**Q: How did you split the data?**
**PARTIALLY VERIFIED.** SMOTE-after-split is confirmed. The exact split
mechanism (random vs. stratified, and whether it's temporally ordered) was
**not independently re-verified** in this audit — I'd say exactly that in
an interview rather than guess, and point to `paysense_phase3.py` as where
to look.

**Q: What metrics did you use, and why?**
ROC-AUC and PR-AUC as primary (PR-AUC specifically because the fraud rate
is only 4.21% — accuracy would be misleading here: a model predicting
"legitimate" for everything scores 95.79% accuracy and is useless).
Precision/recall at the deployed threshold for the actual business
decision. Accuracy was computed but explicitly never used to select or
justify the model.

**Q: What were the biggest challenges?**
Two stand out because they were genuine, embarrassing-if-hidden methodology
bugs, caught and fixed rather than shipped: (1) every fraud metric reported
for most of the project's life was computed on raw XGBoost `predict_proba()`,
but the real `/predict` endpoint blends three scorers — re-scoring through
the real ensemble dropped precision at the then-deployed threshold from a
claimed 86.44% to a real 40.81%, forcing a threshold re-tune. (2) The
category classifier's reported 100% accuracy was hollow — train and test
splits shared the same 40 sentence templates; a hand-built, genuinely-novel
200-row test set measured real accuracy at 72.5%.

**Q: How did you debug failures?**
Concretely, from this session: a "SMS auto-detection isn't working" bug
turned out to be a false alarm — a `logcat -c` clear, issued while chasing
an unrelated dead end, had wiped the evidence that the SMS pipeline had
already succeeded. Caught by re-checking the actual on-screen app state
(a category prompt was live) rather than trusting the log's absence of
evidence as evidence of absence.

**Q: How did you make it reproducible?**
`.env.example` for config, `requirements.txt` pinned versions, a CI
workflow that runs the real test suite on every push. **Real gap, found
this audit:** that CI workflow was completely broken (100% failure rate)
because it never supplied the env vars `main.py` requires to even start —
fixed as part of this review, verified by reproducing the exact failure
locally before and after the fix.

---

## Advanced questions

**Q: Where could data leakage occur?**
The clearest documented case isn't leakage in the train/test sense — it's
a near-tautological feature→label relationship *inside the training data
itself*: in the 10K synthetic supplement, `new_device_flag` and
`ip_location_mismatch` separate `is_fraud` with zero overlap in both
directions, traced to the external source dataset's own label-generation
formula (`EDA_FEATURE_ENGINEERING.md` §1.1). That's a third of the training
data carrying a shortcut that has nothing to do with organic fraud
behavior. Separately: SMOTE-after-split is correct practice and was
verified; whether the split itself is entity/time-aware was **not**
independently re-verified this audit.

**Q: What happens if the dataset grows 100x?**
**NOT MEASURED.** No load or scale testing exists. Reasoned answer: model
artifacts are `.pkl` files loaded once into memory at FastAPI startup —
training-time scaling (100x more rows) is a separate concern from serving-
time scaling (100x more requests); the repo has evidence for neither.

**Q: What is the bottleneck?**
For serving: **not measured**. For the category classifier specifically,
there *is* a measured number — the DistilBERT candidate's 369ms per-request
CPU inference vs. sub-millisecond for the deployed TF-IDF+LinearSVC model
(370× slower) is exactly why it was left undeployed.

**Q: How would you scale inference?**
Reasoned, not implemented: increase `uvicorn` worker count (currently
`workers=1`), move model loading to a shared cache if multi-process, add a
proper load balancer in front of multiple instances. None of this exists
today — the current deployment is single-process.

**Q: How would you deploy this?**
It already has two real, worked deployment paths: Render (currently not
responding — diagnosed but not root-caused from this environment, no
dashboard access) and a prepared-but-incomplete Vercel path (bundle size
measured at 502.4MB against Vercel's 500MB standard limit, fix identified
as enabling Large Functions, blocked on account-level dashboard access).

**Q: How would you monitor the model / detect drift?**
**NOT IMPLEMENTED.** No monitoring or drift-detection exists in this repo.
Honest answer for an interview: I'd want to log the ensemble score
distribution over time and alert on a shift, and I'd want ground-truth
fraud labels fed back eventually (which UPI fraud detection genuinely
struggles with — confirmed fraud often isn't known for weeks).

**Q: How would you handle bad/malformed input?**
Already real, not hypothetical: every field on `TransactionInput` is a
typed, bounded Pydantic model — malformed input gets a 422 before it ever
reaches the scoring code. `/assistant/chat`'s message field is length-
bounded (1–500 chars) for the same reason.

**Q: What happens if the model service is unavailable?**
For the fraud model specifically: `_mock_score()` is a deterministic
fallback formula in `main.py`, used when model artifacts are absent — the
API stays usable for integration testing even without the real pickle
files. For the Gemini-backed endpoints: verified graceful degradation —
timeout, non-2xx, or missing key all fall through to a deterministic
rule-based reply, never a blank response or a 500.

**Q: How would you secure the API further?**
Already real: JWT auth, rate limiting, Pydantic validation, and — new this
session — a real `system_instruction`-based LLM guardrail plus a regex
prompt-injection pre-filter. Gaps I'd name honestly: no CORS audit against
the actual production config (**NOT VERIFIED** what `ALLOWED_ORIGINS`
is set to on Render), no dependency CVE scan has been run.

**Q: What would you redesign with another month?**
Fix the Render deployment first — nothing else matters if the production
API isn't answering. Then: delete the three dead directories cluttering
the repo root, move 30 loose experiment scripts into a proper
`experiments/` folder, and either push recall past 39.53% or make an
explicit, documented business call that the current precision/recall trade-
off is acceptable and stop treating it as an open problem.

---

## Adversarial senior-engineer questions

**Q: Why is your train/test split actually valid?**
Honest answer: SMOTE-after-split is verified correct. The split mechanism
itself (random/stratified/temporal) was not re-derived from
`paysense_phase3.py` in this specific audit — I would open that file
before claiming more than that in a real interview.

**Q: What evidence says your model is better than the baseline?**
PR-AUC 0.5498 against a random-baseline PR-AUC of ~0.0421 (the fraud
rate) — 13.05× above baseline. Accuracy is explicitly *not* the evidence:
the "always predict legitimate" baseline scores 95.79% accuracy on the
same test set and is documented as useless.

**Q: Where can leakage occur?**
Answered directly above — the near-tautological supplement-data finding is
the real, documented one. I would not claim the rest of the pipeline is
leakage-free; the split methodology itself is an open verification gap.

**Q: What happens when this input is missing?**
For `/predict`: Pydantic rejects it with 422 if a required field is absent.
For optional fields with real business meaning (like
`amount_deviation_score`): **not fully traced this audit** whether a
missing-but-optional field silently defaults to a value that changes the
score in a way that was intended vs. accidental — a real thing to verify,
not something I'd claim confidence about without checking `TransactionInput`
's field defaults directly.

**Q: Why does this abstraction exist?** *(e.g. the shared `_call_gemini()`
helper)*
Because two endpoints (`/insights/weekly` and `/assistant/chat`) both need
a guardrailed LLM call, and duplicating the system-instruction/safety-
settings/fallback logic across both would mean fixing a guardrail bug in
one place and forgetting the other. One abstraction, two callers — the
minimum bar for "this abstraction earns its existence."

**Q: Why did you choose this dependency?** *(e.g. `slowapi` for rate
limiting)*
**Partially answerable, partially not.** `slowapi` is a real, working
choice (verified: 60/min and 30/min limits are live and tested). Why
*this* library specifically over alternatives — not documented in the
repo, and I wouldn't invent a justification for it.

**Q: What happens when the dataset distribution changes?**
No drift detection exists (answered above). The OOD testing that *does*
exist (external real-world datasets scored against the frozen model)
already demonstrates the failure mode directly: 0/701 real frauds caught
on one external dataset at the production threshold, root-caused to a
currency-scale mismatch, not a mystery.

**Q: What is your biggest scalability problem?**
Single `uvicorn` worker, no load testing ever performed, no measured
latency or throughput number exists anywhere in this repo. I would say
exactly that rather than claim a number I don't have.

**Q: Which part of this system would you delete?**
The three dead directories (`android/`, `backend/`, old
`PaySense-Android-Client/`) — zero functional value, real confusion cost
for anyone reviewing the repo. Concretely identifiable: none has a commit
in the last month-plus of active work.

**Q: What engineering decision are you least confident about?**
The deployed fraud threshold (0.50) — it was re-derived correctly after
catching the raw-vs-ensemble methodology bug, but the underlying business
constraint it optimizes for (Recall≥75%/Precision≥50%, falling back to
max-F1) isn't hit at 0.50 either (actual recall there is 39.53%, well
under 75%) — meaning the *fallback* branch (max-F1) is what's actually
governing the deployed threshold, and I'd want to re-examine whether that
fallback is still the right choice given the real recall number it
produces.

**Q: If I gave you 100x more data, what breaks first?**
Reasoned, not measured: training time for XGBoost on 3M rows with SMOTE
applied would grow substantially; more concretely, the current pipeline
loads the full dataset into a pandas DataFrame in memory
(`pd.read_csv`) — that's the first thing I'd expect to need rework (batched/
chunked loading, or a real data warehouse) well before 100x, not at it.

**Q: If production predictions become wrong tomorrow, how would you
investigate?**
There's a real precedent for exactly this in the project's own history:
the raw-XGBoost-vs-ensemble bug was found by re-checking which function
`/predict` actually calls versus which function the metrics scripts called
— i.e., verify the served code path matches the measured code path before
trusting any other explanation. I'd start there again.

**Q: What metric would tell you the model is degrading?**
Not implemented today. I'd want the ensemble score distribution tracked
over time (a shift in the mean/variance of scores on live traffic without
a corresponding shift in real fraud reports is a drift signal), plus
eventual ground-truth fraud labels fed back — acknowledging that UPI fraud
confirmation typically lags by weeks, which is a real constraint on how
fast this could ever be.

**Q: What assumptions does your system make?**
That an incoming SMS matching the TRAI sender-ID format and containing a
transaction keyword really is a transaction SMS (Gates 1–2 are heuristic,
not guaranteed-correct); that the demo/training data's ₹-denominated
amount thresholds generalize (they demonstrably don't to USD-denominated
external data — the 0/701 finding); that a user's own transaction history
is a stable baseline for "normal" (no explicit handling for a user whose
spending pattern genuinely changes, e.g. a new job or a big one-time
purchase, versus fraud).

**Q: Which assumption is most dangerous?**
The currency-scale assumption, because it fails silently and specifically
in the direction that matters most — it doesn't produce an error, it
produces a *confident wrong non-alert* on data denominated differently
than what the rules scorer was calibrated against. Documented and
regression-tested in this project (`test_rules_scorer_currency_scale.py`),
which is the right response to finding it, but the underlying assumption
is still load-bearing in production.

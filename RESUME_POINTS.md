# PaySense — Resume Points

Every bullet below passes: **truth test** (repo can prove it), **ownership
test** (actually implemented, not just described), **technical depth
test**, and **interview test** (defensible for 5–10 minutes). Where no real
measured metric exists, the bullet says `[METRIC NEEDED]` instead of
inventing one — no accuracy/latency/throughput/user/scale/cost number
below is fabricated.

Each bullet includes its evidence and interview risk so you can see exactly
what backs it and what a follow-up question would probe.

---

## A. Software Development Engineer

**Bullet:** Diagnosed and fixed a CI pipeline with a 100% failure rate by
tracing a missing-environment-variable startup guard through the GitHub
Actions REST API and a local repro, restoring backend test execution on
every push.
- **Evidence:** `.github/workflows/ci.yml` commit `7754d87`; verified via
  `GET /repos/.../actions/runs` showing 5/5 prior runs failing at the
  pytest step; reproduced locally by removing `.env` and running with the
  workflow's exact declared env.
- **Interview Q:** "How did you know it was that specific cause and not
  something else?" **A:** Confirmed via the failing step name first
  ("Run pytest," not the Android job), then reproduced the exact
  `RuntimeError` locally before writing the fix, then re-ran locally with
  the fix's env vars to confirm 62/62 tests passed under the same
  conditions CI uses.
- **Weak point:** Only verified the fix locally, not by watching the
  actual next CI run complete (that would need a follow-up push and
  observation).

**Bullet:** Built a real JWT authentication flow end-to-end (backend issue/
verify, Android client login, encrypted on-device token storage via
Android Keystore), replacing an earlier client-side credential check.
- **Evidence:** `main.py`'s `get_current_user`/`POST /auth/token`;
  Android's `FraudApiService.login()`; live-verified this session via a
  real 401 (wrong credentials) → 200 (correct) round-trip against a
  running backend, and a force-stop + relaunch test proving the encrypted
  token survives a cold process restart.
- **Interview Q:** "What would happen if the JWT secret leaked?" **A:**
  Every issued token becomes forgeable until the secret is rotated and
  `JWT_SECRET_KEY` redeployed — no key rotation mechanism exists today,
  a real gap I'd name directly.

**Bullet:** Designed a two-layer prompt-injection defense for an LLM-backed
API endpoint — a deterministic regex pre-filter plus a real
`system_instruction` channel — reducing exposure to a class of attack that
a single-layer defense (the codebase's own earlier implementation) didn't
fully address.
- **Evidence:** `main.py`'s `_JAILBREAK_PATTERNS` + `_call_gemini()`;
  `tests/test_api.py::TestAssistantChat` — 4 jailbreak phrasings blocked
  pre-LLM, 1 false-positive sanity test.
- **Interview Q:** "Is the regex exhaustive?" **A:** No — explicitly
  documented as defense-in-depth, not a guarantee; a sufficiently novel
  phrasing relies entirely on the system instruction holding, which was
  tested against 4 phrasings, not adversarially red-teamed at scale.

**Bullet:** `[METRIC NEEDED]` — "Reduced API response time by X%" or similar
latency claims. **Not measured.** No load testing exists in this repo. Do
not use a latency/throughput bullet until this is actually benchmarked.

---

## B. ML / AI Engineer

**Bullet:** Found and corrected a metrics-reporting methodology error where
every previously-reported fraud-model metric had been computed on raw
XGBoost `predict_proba()` instead of the real 3-scorer ensemble the API
actually serves — re-scored the full evaluation, dropping a previously
reported 86.44% precision figure to a verified 40.81% at the
then-deployed threshold, and re-tuned the deployed threshold (0.30→0.50)
against the corrected numbers.
- **Evidence:** `WALKTHROUGH.md`'s README correction, dated 2026-08-24;
  `resweep_threshold_against_ensemble.py`; `test_frozen_model_metrics.py`
  (new regression test explicitly asserting raw XGBoost and the real
  ensemble stay materially different, guarding against recurrence).
- **Interview Q:** "Why didn't this get caught earlier?" **A:** The
  ensemble-blending code was added after the metrics-reporting habit was
  established; nobody had re-checked which function the reported numbers
  actually traced back to versus which function `/predict` calls. Root
  cause was a code-path mismatch, not a math error.

**Bullet:** Diagnosed a category classifier's reported 100% test accuracy
as an artifact of template contamination (train and test drawn from the
same 40 fixed sentence templates), built an independent 200-row,
structurally-verified-novel evaluation set, and retrained to a real
78.0% accuracy on genuinely unseen phrasing.
- **Evidence:** `CATEGORY_CLASSIFIER.md`, `CATEGORY_CLASSIFIER_GENERALIZATION.md`;
  vocabulary-overlap mechanism confirmed (821 tokens, entirely mined from
  the 40 training templates).
- **Interview Q:** "How did you verify the new eval set was actually
  novel, not just superficially different?" **A:** Structurally verified
  disjoint from training templates before use — not just visually
  different wording.

**Bullet:** Ran a controlled architecture-vs-data ablation for the category
classifier — fine-tuned DistilBERT on the *identical* training data as the
deployed TF-IDF+LinearSVC retrain — isolating architecture as the variable
and measuring an 83.0% accuracy ceiling (5 points above the deployed
model), then made and documented the deliberate call not to deploy it
given a measured 128× size and 370× inference-latency cost.
- **Evidence:** `train_category_classifier_distilbert.py`;
  `WALKTHROUGH.md`'s DistilBERT section — 267.8MB vs ~2.1MB, 369ms vs
  sub-millisecond measured inference.
- **Interview Q:** "Why not deploy it anyway if it's more accurate?" **A:**
  No verified memory headroom on the hosting tier for torch+transformers
  alongside the existing model stack — an OOM crash on the live service is
  a worse outcome than a slightly-less-accurate model that stays up. A
  real engineering trade-off, made and documented, not avoided.

**Bullet:** Built a real out-of-distribution evaluation against 74,917
external, previously-unseen UPI transactions, found the deployed ensemble
caught 0 of 701 real frauds at the production threshold, and traced the
cause to a specific, confirmed root: a rules-scorer cold-start bonus
calibrated for ₹-denominated amounts silently never triggering on the
external dataset's $-denominated values.
- **Evidence:** `OOD_GENERALIZATION_REMEDIATION.md`,
  `ood_threshold_sweep_variant_a.py`,
  `test_rules_scorer_currency_scale.py` (regression test locking in the
  finding).
- **Interview Q:** "Did you fix it?" **A:** Confirmed the root cause and
  regression-tested it, but left the production threshold unchanged —
  production only ever sees real ₹ traffic, so this is a documented,
  understood limitation of external-dataset generalization, not a live
  bug. A follow-up threshold sweep found a much lower, OOD-specific
  threshold recovers 37/701 frauds with zero false positives on that
  specific dataset — not deployed, since it was found by sweeping that
  dataset's own labels rather than a held-out calibration set.

---

## C. Data Scientist

**Bullet:** Traced a near-tautological feature→label relationship in a
third of a fraud model's training data to its origin — an external
synthetic dataset whose own label-generation formula, not this project's
pipeline, produced zero-overlap separation between two risk flags and the
fraud label — and documented it as a real, unresolved training-data
limitation rather than silently accepting the resulting metrics.
- **Evidence:** `EDA_FEATURE_ENGINEERING.md` §1.1, verified independently
  against the raw external source file (`device_risk_score > 0.70` → exact
  500/500 fraud, 9500/9500 legitimate split).
- **Interview Q:** "Why does this matter if the model still performs
  well?" **A:** Because a model that performs well partly by exploiting a
  formula-generated shortcut in synthetic data won't generalize the same
  way to organic real-world fraud that doesn't share that shortcut — which
  is exactly what the separate OOD evaluation (0/701 real frauds caught)
  demonstrates.

**Bullet:** Selected PR-AUC over accuracy as the primary evaluation metric
for a 4.21%-base-rate fraud classification problem, explicitly computing
and reporting the accuracy of a trivial "always predict legitimate"
baseline (95.79%) to make the metric choice defensible rather than
assumed.
- **Evidence:** `WALKTHROUGH.md`'s "Honest findings" — "Accuracy reads
  97.38%... but a model that predicts legitimate for every transaction
  would score 95.79%... Accuracy was computed, never used, to pick this
  model."
- **Interview Q:** "What metric would you use if false positives were
  extremely costly?" **A:** Precision at a fixed recall floor, or a cost-
  weighted metric reflecting the real business cost of a false fraud
  block versus a missed fraud — not implemented in this project, but the
  reasoning generalizes from the PR-AUC choice already made.

**Bullet:** `[METRIC NEEDED]` — any claim of "improved model accuracy by
X%" as a single headline number. The real, honest story is multi-metric
and includes a corrected-downward number (precision 86.44%→40.81% after
the methodology fix) alongside genuine improvements (recall ceiling
71.94% via monotonic constraints, up from a lower baseline) — collapsing
this into one clean percentage would misrepresent the actual work. Use the
specific, correctly-attributed numbers above instead of a single summary
figure.

---

## Notes on what NOT to claim

- **No production usage numbers exist.** Do not write "used by X users" or
  "processed X transactions in production" — the Render deployment is
  currently not even responding (see `PROJECT.md` §23). This is a
  portfolio/academic project, not a live production service with users.
- **No measured latency/throughput numbers exist.** Do not write "handles
  X requests/second" or "reduced latency by X%."
- **No test coverage percentage exists.** 211/211 backend tests and 3
  Android unit test files are real and countable; a coverage *percentage*
  (lines/branches covered) was never measured, so don't state one.

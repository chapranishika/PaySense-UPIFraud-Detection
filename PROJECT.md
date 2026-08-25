# PaySense — Project

The deep technical narrative: what this is, why it exists, how it evolved,
what it's actually made of, and what's still wrong with it. This document
is the connective tissue between the other docs in this repo — it doesn't
re-derive what `WALKTHROUGH.md` (results + honest findings), `DATASET.md`,
`ARCHITECTURE.md`, and `SECURITY.md` already cover well; it points into
them and covers what they don't: history, requirements, repo structure,
design decisions, and the "how to run this" instructions.

Status labels: **FACT** (verified this audit by reading/running code),
**INFERENCE** (reasoned conclusion, not independently proven), **NOT
VERIFIED**, **MISSING**.

---

## 1. Executive summary

PaySense is a three-layer UPI (India's real-time payments rail) fraud
detection system: an Android client that parses incoming bank SMS on-device,
categorizes the transaction, and sends it to a FastAPI + XGBoost-ensemble
backend for a real-time fraud score. A second, full-featured web dashboard
is served by the same backend. Built solo by one author (Nishika Chapra,
KJ Somaiya Institute of Technology, 2025) as an academic/portfolio project,
with substantial LLM assistance during development (this document exists
because of that fact, not despite it).

**What actually works, verified this session, not just claimed:** real
login against a live backend, on-device SMS parsing through three
regex-based gates, category resolution through a real cache→NLP→human-in-
the-loop pipeline, an XGBoost+LightLR+rules ensemble that correctly
distinguishes a normal ₹15,000 purchase (`alert=none`) from a ₹185,000
new-device/IP-mismatch transaction (`alert=high`, ensemble score 0.95),
and — as of the latest work — a real Gemini-backed AI assistant with
genuine prompt-injection guardrails, falling back gracefully when no key is
configured.

**What's currently broken:** the production Render deployment is not
responding (see §23). CI was fully broken (every run failing) until this
audit found and fixed it.

## 2. Problem statement

**Plain language:** UPI fraud in India is real and growing — someone tricks
you into approving a payment, or your phone is compromised, and money moves
before you notice. Most fraud detection lives entirely server-side, invisible
to the end user until the money is already gone. PaySense's premise: catch
it earlier, on the device, the moment the confirmation SMS arrives, using
personalization (this user's own spending pattern) rather than only
population-level rules.

**Technical formulation:** binary classification (fraud / not-fraud) on a
UPI transaction event, scored by a weighted ensemble of three independently-
trained/tuned components, gated by an app-side confirmation flow that
mirrors real-world "does this look like your normal spending" reasoning.

## 3. Motivation

**FACT.** The README frames this explicitly as solving a personalization
gap: population-level fraud rules miss the fact that a ₹50,000 transaction
is completely normal for one user and wildly anomalous for another. The
`amount_deviation_score` feature and per-user profile fields
(`usr_avg_txn_value_profile`, `usr_is_high_risk`) are the technical
expression of that idea.

## 4. Project history (reconstructed from `git log`, not invented)

**FACT.** 60 commits total. Verified date distribution:

```
2026-04-29  1 commit   — initial scaffold: "PaySense full system (Android + ML backend)"
2026-07-15  2 commits  — "PaySense Guardian" rename, Finance Tracker feature added
2026-07-16  1 commit   — docs
2026-07-23  2 commits  — theme/crash fixes, web dashboard + AI Assistant added
2026-07-24  2 commits  — UI redesign, Finance tab charts
2026-07-25  2 commits  — seeded data, Cash Flow chart, AI insight tone
2026-08-04  1 commit   — point Android client at live Render URL
2026-08-22  13 commits — Android/backend test suites added; real JWT auth
                         replaces a hardcoded login check; real NLP category
                         classifier trained and wired in; CI added
2026-08-23  7 commits  — recall-ceiling remediation experiments; monotonic-
                         constraints model adopted; category classifier's
                         hollow 100% replaced with a real 72.5% measurement
2026-08-24  23 commits — v2 category classifier retracted after finding
                         template contamination, real v3/v4 shipped; a
                         methodology error (raw XGBoost vs. real ensemble)
                         found and corrected, threshold re-tuned 0.30→0.50;
                         OOD/generalization work; test-coverage hardening
2026-08-25  5 commits  — Android emulator debugging + live on-device
                         verification; real-LLM-backed AI Assistant added
2026-08-26  1 commit   — this audit's CI fix
```

**INFERENCE.** The concentration of 49/60 commits (82%) in the Aug 22–26
window, following a 2.5-week gap after Aug 4, reads as: an initial
prototype built over April–August, then an intensive audit-and-harden pass
in the final week before submission — consistent with the commit messages
themselves ("fix... default APP_ENV to production; add CI",
"correct my own error", "retract invalidated classifier v2 result").
**This self-correction pattern is a real, positive signal**, not a defect
to hide: `git log` shows genuine methodology errors caught and fixed by the
same person/process that introduced them, with the wrong result never
quietly replaced — see e.g. `08-24 fix(ml): correct raw-XGBoost-vs-real-
ensemble methodology error, re-tune threshold to 0.50`, where a previously-
reported 86.44% precision number is explicitly retracted in the same commit
that reports the real 40.81%.

**MISSING.** No evidence in the repo of what happened between the April 29
scaffold and the July 15 "Guardian" rename (2.5 months) — could be offline
design work, coursework timing, or simply not committed. Not fabricated
here.

## 5. Requirements

**Functional (verified against actual endpoints/screens, not aspirational):**
- Parse incoming bank SMS on-device and extract amount/payee/txnId/date.
- Categorize a transaction (cache → NLP → human-in-the-loop).
- Score a transaction for fraud risk in real time, personalized per user.
- Show a fraud/safety dashboard, transaction history, and finance
  analytics (Android + web).
- Provide savings recommendations and a conversational assistant, scoped to
  the user's own data.
- Real authentication (JWT), not a client-side check.

**Non-functional, honestly assessed:**
- **Performance:** NOT VERIFIED under load. No load testing evidence in
  the repo. Single-process `uvicorn`, single worker (`workers=1` in
  `main.py`'s entrypoint) — this will not scale past one process without
  operator changes.
- **Reliability:** rate limiting present (60/min, 30/min); no retry/circuit-
  breaker logic on the Gemini calls beyond the built-in fallback; no
  documented uptime target.
- **Security:** see `SECURITY.md` — real JWT auth, no committed secrets,
  guardrailed LLM calls.
- **Reproducibility:** see §22 below — mostly reproducible, with real gaps
  (no pinned random seeds verified, dataset files are committed directly
  rather than versioned externally).
- **Maintainability:** mixed — see §9 repo map. The active components
  (`PaySense-ML-Backend/`, `PaySense-Android-Client-New/`) are reasonably
  organized; the root directory carries real clutter (§9, §23).

## 6. Dataset

See `DATASET.md` for the full treatment. Summary: 30,000-row fraud-scoring
dataset (4.21% fraud, blended from a real-style 20K anchor + a 10K
synthetic supplement whose label carries a documented near-tautological
relationship to two of its own features), and a separate 6,000-row
FinText-6K text dataset for category classification (40-template synthetic
generation, documented and worked around).

## 7. Geography

**Not applicable.** No coordinates, spatial joins, or geographic train/test
split exist in this project. `usr_home_city_tier` is a categorical profile
field, not spatial data. Not fabricated to fill this section.

## 8. Data pipeline

```
paysense_pipeline.py
  → blends anchor (20K, transactions.csv) + supplement (10K, external
    synthetic_fraud_dataset.csv, schema-bridged)
  → paysense_master_dataset.csv (30,000 rows × 50 columns)

eda_feature_engineering.py / paysense_phase3.py
  → 50 raw columns → 40 model-ready features
  → train/test split
  → SMOTE on training partition only (24K → 45,980 rows)

training scripts (paysense_ml_pipeline.py, monotonic-constraints retrain)
  → paysense_model.pkl (XGBoost, 40 features)
  → paysense_preprocessor.pkl
  → paysense_threshold.pkl  (currently 0.50)
  → paysense_feature_names.pkl

train_light_lr.py
  → light_lr.pkl (5-feature LogisticRegression, standalone-capable fallback)

train_category_classifier.py / _v3.py / _v4.py
  → category classifier artefact (TF-IDF + calibrated LinearSVC)

main.py's lifespan() context manager
  → loads all artefacts once at FastAPI startup into src/fraud_model.py's
    module-level _state
```

Every arrow above is a real, separately-runnable script in
`PaySense-ML-Backend/` — verified present via directory listing, not
inferred from documentation alone.

## 9. Repository map (what's real, what's dead)

**FACT, verified via `git log -1 -- <dir>` for each candidate:**

| Path | Status | Last touched | Verdict |
|---|---|---|---|
| `PaySense-ML-Backend/` | **ACTIVE** | 2026-08-26 (this audit) | The real backend. |
| `PaySense-Android-Client-New/` | **ACTIVE** | 2026-08-25 | The real Android client. |
| `PaySense-Android-Client/` (no "-New") | **DEAD, kept deliberately** | 2026-07-23 | Superseded by `-New`, ~207KB. README.md already documents this explicitly: "early, incomplete scaffold — superseded by -New, kept for history." A conscious choice by the project owner, not an oversight — this audit did not override it. |
| `android/` | **REMOVED, 2026-08-26** | was 2026-04-29 only | First-draft scaffold, ~172KB, never touched again in 4 months, never mentioned anywhere. Verified unreferenced by any config/CI/docs before removal via `git rm -r`; full history remains in git. |
| `backend/` | **REMOVED, 2026-08-26** | was 2026-04-29 only | First-draft scaffold, ~2.1MB, same verification and removal as `android/`. |
| `PaySense-Report/` | Static assets | — | LaTeX report + generated plots (SHAP, evaluation figures). Not code. |
| `screenshots/` | Documentation assets | — | Real captures referenced by `WALKTHROUGH.md`, not decorative. |
| `.github/workflows/ci.yml` | **ACTIVE, now fixed** | 2026-08-26 | Was fully broken (100% failure rate) — see `SECURITY.md` §1. |

**Resolution, 2026-08-26 audit:** `android/` and `backend/` were removed
via `git rm -r` after verifying (a) no CI/build config references either
path, (b) no documentation anywhere frames them as intentional (unlike the
old Android client, below), and (c) their full history remains permanently
retrievable via `git log --all`/`git show` — deletion from the working
tree loses nothing. `PaySense-Android-Client` (no "-New") was deliberately
**left in place**: README.md already documents it as a conscious "kept for
history" decision by the project owner, and overriding that unilaterally
would replace one person's judgment call with this audit's own, which
isn't the same kind of "safe, verified, unreferenced" deletion the other
two were.

**Inside `PaySense-ML-Backend/` — genuinely large experiment-script sprawl,
by design, not by accident:** 30 loose `.py` scripts at the top level
(`generate_category_training_v2.py`, `ood_threshold_sweep_variant_a.py`,
`platt_scaling_experiment.py`, `recall_ceiling_remediation.py`, etc.), each
paired with a `.md` writeup and often a `.log`/`.json` result file. This
looks messy at first glance but is **not** the "abandoned experimental
code" smell the audit was asked to hunt for — cross-checking a sample
against `WALKTHROUGH.md`'s fix log confirms every one of these represents
a real, documented, referenced experiment (e.g. `platt_scaling_experiment.py`
↔ `PLATT_SCALING_RESULT.md` ↔ the "Recall plateaued... Platt Scaling...
wasn't [the fix]" finding in `WALKTHROUGH.md`). **Recommendation (P2):**
move these into an `experiments/` subdirectory to signal "these are
research scripts, not the served application" at a glance — `main.py` and
`src/` are the only code that actually runs in production. Not done in
this audit (a mechanical move, real but low-value relative to the P0/P1
items).

**`.pytest_cache/`, `__pycache__/`, `.vercel/`, `distilbert_train_tmp/`,
and the many `*_run.log` files at the backend root** are exactly what
they look like — local run artifacts. `.gitignore` catches `.pyc` and
`__pycache__/` but **not** the `*.log`/`*.json` result files or
`.pytest_cache`/`.vercel`/`distilbert_train_tmp` — **NOT VERIFIED** whether
these are currently tracked in git or just present untracked locally;
worth a `git status`/`git ls-files` check before assuming either way.

## 10. Model / algorithm / methodology

See `WALKTHROUGH.md`'s "Key results" and "Honest findings" sections — this
is the single most rigorous part of the existing documentation and is not
duplicated here. Summary: 3-scorer ensemble (rules 0.15 / XGBoost 0.60 /
LightLR 0.25), ROC-AUC 0.8969, PR-AUC 0.5498 (13.1× random baseline),
precision 91.74% / recall 39.53% at the deployed threshold (0.50). Baseline
comparison exists and is explicit: "a model that predicts legitimate for
every transaction would score 95.79% [accuracy] on this same test set" —
accuracy was computed but never used to select the model, precisely because
of this.

## 11. Evaluation

Metrics used: ROC-AUC, PR-AUC (primary, given 4.21% base rate — accuracy
would be misleading, and the docs say so explicitly), precision/recall at
the deployed threshold. Threshold selection: `paysense_phase3.py`'s
business constraint (Recall≥75%/Precision≥50%, fallback to max-F1),
re-swept against the real ensemble (not raw XGBoost) after the methodology
correction on 2026-08-24. **This is real, defensible evaluation
methodology** — not "we used accuracy because it's common."

## 12. Failure analysis

**FACT, from `WALKTHROUGH.md`, not restated in full here:**
- 0/701 real frauds caught on one external OOD dataset at the production
  threshold — root cause traced to a currency-scale mismatch in the rules
  scorer's cold-start bonus (calibrated for ₹, tested against $).
- Category classifier: 22–28 percentage points of real accuracy lost
  between "hollow 100%" and "measured against genuinely novel input,"
  depending on which retrain generation is compared.
- Recall plateaus at 39.53% (deployed) / 71.94% (best achieved with
  monotonic constraints) — roughly 45 rows resist every remediation tried.

**New this audit:** the production API endpoint itself is currently
returning nothing at all (see §23) — a different class of failure than
anything ML-related: the model could be perfect and no request would ever
reach it right now.

## 13. Challenges (selected, from real git history + docs)

```
Problem:  Category classifier's documented 100% accuracy was hollow.
Why:      Train and test splits shared the same 40 sentence templates.
Options:  (a) trust the number, (b) build an independent test set.
Decision: (b) — 200 hand-authored, structurally-verified-novel narrations.
Result:   72.5% real accuracy — 27.5 points below the reported figure.
Trade-off: Took real effort to build a proper eval set; worth it because
           the false number would have shipped into a resume/interview.
```

```
Problem:  Every fraud metric ever reported used raw XGBoost predict_proba(),
          but /predict calls the real 3-scorer ensemble.
Why:      The ensemble blending code was added after the metrics-reporting
          habit was established, and nobody re-checked which function the
          numbers actually came from.
Options:  (a) leave it, it's "close enough", (b) re-score everything
          through the real ensemble and accept whatever changes.
Decision: (b).
Result:   Precision at the previously-deployed threshold (0.30) dropped
          from a claimed 86.44% to a real 40.81%. Threshold re-tuned to
          0.50 against the real ensemble.
Trade-off: A worse-looking number, reported anyway, plus a new regression
           test (test_frozen_model_metrics.py) specifically guarding
           against this exact class of mistake recurring silently.
```

More in `WALKTHROUGH.md`'s "Full fix log" — not re-derived here.

## 14. Architecture

See `ARCHITECTURE.md` for the full diagram set (system, SMS pipeline,
categorization pipeline, inference flow, deployment).

## 15. Security

See `SECURITY.md`.

## 16. Performance

**A modest, reproducible local benchmark — not a load test — run 2026-08-26**
to establish basic evidence where none existed. Not claimed to represent
production behavior (different hardware, single local request stream, no
concurrency).

*Environment:* this development machine (Intel Core Ultra 9 185H, Windows
11 Home), local `uvicorn main:app` process, Python 3.11, sequential
(non-concurrent) requests over `localhost`.

*Cold start* (process launch to first successful `GET /health`, includes
loading all ML artefacts): **1.77 seconds**, one run.

*Warm inference latency*, `POST /predict`, 50 sequential requests against
an already-running server, real JWT auth on every request, using the test
suite's own `legit_payload()` fixture:

| | ms |
|---|---:|
| Mean | 14.58 |
| Median | 11.98 |
| P95 | 35.23 |
| P99 | 39.23 |
| Min | 6.94 |
| Max | 39.23 |

**NOT MEASURED reliably: process memory (RSS).** An attempted measurement
returned an implausible 4MB for a process with XGBoost/pandas/scikit-learn
loaded — almost certainly the wrong process handle, not a real number.
Rather than report it, this is flagged as a failed measurement attempt.

**NOT MEASURED at all:** concurrent request throughput, behavior under
load, production-hardware latency (Render's actual instance size is
unknown from this environment), and the category classifier's separately-
measured 369ms (DistilBERT) vs. sub-millisecond (deployed) single-request
inference numbers already documented in `WALKTHROUGH.md` are the only
other real performance evidence this project has. This benchmark is
explicitly a floor for "basic engineering evidence exists," not a
production capacity claim.

## 17. Testing

**FACT, run this audit:** 211/211 backend pytest tests passing (`pytest
tests/ -q`), covering auth, `/predict` (legit + fraud payloads, schema,
validation, new-device/night-transaction score deltas), `/classify`,
`/insights/weekly`, `/assistant/chat` (13 tests, including 4 jailbreak
phrasings and a false-positive sanity check), `/health`, plus a large body
of ML-methodology regression tests (`test_frozen_model_metrics.py`,
`test_ood_generalization_remediation.py`, `test_recall_ceiling_remediation.py`,
etc.) that pin specific numeric results so a future change can't silently
drift them. 3 Android JUnit4 unit test files exist
(`SmsReceiverTest.kt`, `NlpKeywordRulesTest.kt`,
`DeviationStatsCalculatorTest.kt`) covering pure regex/keyword/z-score
logic — **NOT VERIFIED this audit** whether they currently pass (the CI
job that runs them was, until this fix, blocked from ever completing the
backend job in the same workflow run, though the Android job is a separate
job and per the GitHub API check above, it *has* been passing).

**No coverage percentage is claimed anywhere in this repo, and none is
invented here.**

## 18. Reproducibility

```bash
# Backend
cd PaySense-ML-Backend
python -m venv venv && venv/Scripts/activate  # or source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
cp .env.example .env   # fill in JWT_SECRET_KEY, API_DEMO_USER, API_DEMO_PASS
uvicorn main:app --reload --port 8000

# Android
cd PaySense-Android-Client-New
./gradlew assembleDebug
```

**FACT.** Both commands above were run successfully this session (backend
serving real predictions; Android APK builds and installs). **NOT
VERIFIED:** whether `requirements.txt`'s pinned versions resolve cleanly
on a machine other than this one, or whether training scripts (as opposed
to serving the frozen artifacts) are fully reproducible — no random-seed
audit was performed across the training scripts this session.

## 19. Deployment

See `ARCHITECTURE.md` §5 and §23 below.

## 20. Timeline

See §4 above — merged rather than duplicated, per the "avoid documentation
fragmentation" principle this audit was asked to follow.

## 21. Design decisions log (selected)

```
Decision: Deploy the data-diversity-retrained category classifier (78.0%),
          not the more-accurate DistilBERT fine-tune (83.0%).
Context:  Both were trained on identical data; DistilBERT scored higher.
Options:  Deploy the better model; keep the faster/smaller one.
Chosen:   Keep the smaller, faster, already-deployed model.
Why:      267.8MB model vs ~2.1MB (128× larger); 369ms vs sub-millisecond
          inference (370× slower); no way to verify the live hosting
          tier's memory headroom for torch+transformers alongside the
          existing stack.
Trade-off: Leaves 5 percentage points of real accuracy on the table.
Consequence: An OOM crash on the live service would be strictly worse than
          keeping the faster, slightly-less-accurate classifier running.
```

```
Decision: Assistant LLM guardrails as two separate layers (regex pre-filter
          + system_instruction), not one.
Context:  A single-layer defense (only a system_instruction, the way the
          original savings-tip Gemini call worked) is real but can
          sometimes be argued past by a sufficiently clever rephrase.
Options:  System-instruction only; regex-only; both.
Chosen:   Both, regex first.
Why:      The regex layer is free (no LLM call, no latency, no cost) and
          categorically cannot be argued past — it either matches known
          phrasing or it doesn't. The system instruction is what has to
          hold for everything the regex doesn't catch.
Trade-off: A finite, hand-authored pattern list is not exhaustive — a
          novel injection phrasing relies entirely on layer 2 holding,
          which was tested against 4 phrasings, not adversarially
          red-teamed at scale.
```

```
Decision: Keep the deployed threshold at 0.50 despite it not satisfying
          the documented Recall>=75% business constraint.
Context:  A full 0.05-0.95 sweep against the real ensemble found ZERO
          thresholds meeting Recall>=75% AND Precision>=50% simultaneously.
          The closest candidate (t=0.15) clears 75.9% recall at 15.99%
          precision -- 84% of its alerts would be false alarms.
Options:  (a) deploy t=0.15 to technically satisfy the recall floor;
          (b) keep 0.50 (best F1); (c) revise the documented requirement.
Chosen:   (b) and (c) together -- keep 0.50, document that the original
          Recall>=75% requirement is not currently achievable.
Why:      A system that fires false fraud alerts 84% of the time destroys
          user trust faster than missing fraud does, and doesn't actually
          satisfy the *spirit* of the requirement even if it satisfies the
          letter. Choosing a threshold to make a number technically true
          is exactly the kind of "prettier metric" selection that caused
          the raw-XGBoost-vs-ensemble bug in the first place.
Trade-off: The deployed system misses most organic fraud (2.55% recall on
          the organic test subset). This is now precisely quantified and
          regression-tested, not softened.
Consequence: Whoever owns this project's requirements needs to either
          accept a revised, lower recall target, or fund the larger work
          (more/better organic training data, or a materially different
          model) that could make Recall>=75% genuinely achievable.
```

## 22. Known limitations — brutally honest

**Technical:**
- Production Render deployment is not responding (§23) — this is the
  single most severe current issue.
- CI was completely broken until this audit (fixed, but was broken through
  at least 5 recent commits including today's work before the fix).
- Single-process, single-worker deployment — no horizontal scaling path
  documented or built.

**Data — the single most important finding of the 2026-08-26 audit:**
- A third of the fraud training data (the 10K supplement) carries a
  near-tautological feature→label relationship inherited from its source
  — and this contaminates the test set in the same proportion, because the
  split doesn't account for `data_source`. Quantified precisely: on the
  organic (anchor-only) subset of the canonical test set, ROC-AUC is
  0.7465 and PR-AUC is 0.1138 (vs. 0.8969 / 0.5498 blended) — real,
  positive signal, but nowhere near the headline numbers. Full breakdown
  in `DATASET.md` and `WALKTHROUGH.md`'s honest findings.
- The threshold is selected on the same held-out partition its performance
  is then reported on — no separate validation set exists. Narrower issue
  than the above, but means reported precision/recall are somewhat
  threshold-optimistic.
- The category classifier's original training data is 40 fixed sentence
  templates — worked around, not eliminated (the deployed model's 78.0%
  accuracy is real, but still template-influenced relative to true
  open-domain text).

**Model:**
- Recall caps at 39.53% at the deployed threshold on the blended test
  set — but only 2.55% (4/157) on the organic subset alone (see above).
  The majority of real, organic fraud in this system's own held-out test
  data is essentially undetected.
- **The documented business requirement (Recall≥75% AND Precision≥50%) is
  confirmed, by a full threshold sweep, to be unachievable by this model on
  this test set at any threshold** — not a threshold-selection oversight.
  The deployed 0.50 is the best available F1 operating point, not a
  constraint-satisfying one. The honest resolution is revising the
  documented requirement, not changing the deployed threshold (see design
  decision below).
- 0/701 real frauds caught on one specific external OOD dataset at the
  production threshold (root-caused, not mysterious, but not fixed either
  — fixing it would require either a currency-aware rules scorer or
  accepting the false-positive cost of a much lower threshold).

**Scalability:**
- No load testing performed; no evidence this handles concurrent traffic
  at any specific scale.

**Deployment:**
- Two deployment paths exist (Render, prepared-but-incomplete Vercel);
  neither is currently fully working from outside this development
  environment.
- Repository carries ~2.5MB and three full directories of dead,
  unmaintained duplicate code (§9).

## 23. Production deployment status (verified this audit, current as of
### 2026-08-26)

**FACT.** `https://paysense-api.onrender.com/health` was tested directly
this audit: DNS resolves correctly, TLS handshake succeeds, the HTTP
request is sent — and zero bytes come back even after a full 90-second
wait. This is not a Render free-tier cold-start (those resolve in
30–50 seconds); this pattern (connection accepted, then silence) is
consistent with the application process having crashed after accepting
the connection, or the service being suspended server-side. **Root cause
NOT VERIFIED** from this environment — no Render dashboard access exists
here. This is the highest-priority action item for whoever owns the Render
account: check the dashboard's deploy logs directly.

## 24. Future improvements

**High impact:**
1. Diagnose and fix the Render deployment (§23) — without this, the
   production `BASE_URL` the shipped app points at serves nothing.
2. **~~Delete the dead directories (§9)~~ — DONE, 2026-08-26** (`android/`,
   `backend/`; `PaySense-Android-Client` deliberately kept, see §9).
3. Retrain on organic-only data (or a genuinely new organic dataset) to
   close the gap found in §22/`DATASET.md` — 2.55% real recall on organic
   fraud is the actual number that matters, not the blended 39.53%. This
   is now the single highest-value remaining item in the entire project.
4. Formally revise the "Recall≥75%" documented requirement, or fund the
   model/data work that could make it achievable — confirmed unachievable
   at any threshold on the current model (§21 design decision).

**Medium impact:**
5. **~~Remove the unused `aiosqlite` dependency~~ — DONE, 2026-08-26.**
6. Move the 30 loose experiment scripts under `PaySense-ML-Backend/` into
   an `experiments/` subdirectory.
7. Add a real load test and publish actual latency/throughput numbers
   instead of leaving §16 empty.
8. Upgrade `starlette`/`fastapi` together to clear the remaining CVEs
   (`SECURITY.md` §5) — needs real regression testing, not a blind bump.

**Nice to have:**
9. System-level push notifications for high-risk fraud alerts (currently
   in-app only).
10. **~~A CVE/dependency scan~~ — DONE, 2026-08-26** (`pip-audit`; 4 of 8
    flagged packages fully fixed, remainder documented in `SECURITY.md`).
11. Deploy the DistilBERT category classifier once the hosting tier's
    memory headroom is actually confirmed, recovering ~5 points of real
    accuracy.

## 25–28. How to run / train / evaluate / run inference

**Run the API:** §18 above.

**Train:** each model has its own script at `PaySense-ML-Backend/` root —
`paysense_ml_pipeline.py` (fraud ensemble), `train_light_lr.py`,
`train_category_classifier.py` (or `_v3.py`/`_v4.py` for the deployed
generation). **NOT VERIFIED this audit** that a full clean-environment
retrain was actually re-run end-to-end — the frozen `.pkl` artefacts under
`artefacts/` were used as-is for all serving/testing this session.

**Evaluate:** `pytest tests/` runs the full regression suite, including
the ML-methodology pinning tests. `generalization_check*.py` scripts
re-score against the external OOD datasets referenced throughout
`WALKTHROUGH.md`.

**Run inference:** `POST /predict` with a JWT (see `README.md`'s Swagger
screenshots, or `tests/test_api.py`'s `legit_payload()`/`fraud_payload()`
helpers for exact request shapes) is the real inference path — this is
what the Android app and web dashboard both call.

## 29. Repository structure

See §9 above.

## 30. Glossary

- **Ensemble** — the weighted blend of rules (0.15) + XGBoost (0.60) +
  LightLR (0.25) scorers that produces the final fraud score.
- **Gate** — one of three sequential regex checks Layer 1 (Android)
  applies to an incoming SMS before treating it as a transaction.
- **HITL** — human-in-the-loop; the category confirmation prompt shown
  when the NLP classifier's confidence is below 0.65.
- **OOD** — out-of-distribution; testing the model against data unlike
  what it trained on, to check whether it generalizes or just memorized.
- **System instruction** — the Gemini API's dedicated channel for
  constraining a model's behavior, kept separate from the user's own
  message (as opposed to blending both into one prompt string).
- **TRAI** — Telecom Regulatory Authority of India; the DLT sender-ID
  format (`^[A-Z]{2}-[A-Z0-9]{4,6}$`) Gate 1 checks against.

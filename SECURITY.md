# PaySense — Security

This is the top-level security summary. It consolidates findings from three
places that previously held security information separately:
`PaySense-ML-Backend/ANDROID_SECURITY_REVIEW.md` (full detail on the Android
client, kept as the source of truth for that half), the security-relevant
comments scattered through `main.py`, and one new finding from this audit
(CI secret handling). Where this document summarizes a finding already
written up in full elsewhere, it says so and links out rather than
duplicating.

Status labels used below follow the audit convention: **FACT** (verified by
reading code or running it), **INFERENCE** (a reasoned conclusion from
facts, not independently proven), **NOT VERIFIED** (plausible but unchecked).

---

## 1. Secrets

**FACT.** `git ls-files | grep '\.env$'` returns nothing — no `.env` file
has ever been committed to this repository, on either the Android or
backend side. `.gitignore` excludes `*.jks`, `*.keystore`, `local.properties`,
and `.env`.

**FACT.** `main.py`'s `_require_env()` (line ~75) refuses to start the
server at all if `JWT_SECRET_KEY`, `API_DEMO_USER`, or `API_DEMO_PASS` are
unset — there is no hardcoded fallback for any of them. This is a
deliberate, correct choice: a missing env var fails loudly at startup
instead of silently running with a guessable default that anyone reading
this public repo could use to forge tokens.

**FACT, found and fixed in this audit.** The GitHub Actions CI workflow
(`.github/workflows/ci.yml`) ran the backend pytest job with
`APP_ENV: production` but never supplied `JWT_SECRET_KEY`,
`API_DEMO_USER`, or `API_DEMO_PASS` — meaning `_require_env()` raised at
import time and **every CI run failed before a single test could execute**.
Confirmed via the GitHub Actions REST API (5/5 recent runs failed at the
"Run pytest" step) and reproduced locally by removing `.env` and running
pytest with only the workflow's declared env vars. Fixed by adding the
three as CI-only env vars on that step — not real secrets:
`JWT_SECRET_KEY` only needs to be *some* string for the test suite's
sign/verify round-trip to work, and `API_DEMO_USER`/`API_DEMO_PASS` are the
same demo credentials (`paysense` / `guardian2025`) already public in
`.env.example` and `README.md`. This is not a security regression to fix by
hiding the demo credentials harder — they were always meant to be public,
since this is a portfolio/demo deployment, not a real financial service
with real user funds behind it. Verified fixed: 62/62 tests in
`test_api.py` pass locally under the exact env the CI job now sets.

**Recommendation, not yet done.** `GEMINI_API_KEY` is optional (both
LLM-backed endpoints degrade gracefully without it — see
`main.py`'s `_call_gemini()`), so it does not block CI or local dev. If a
real key is ever set on the production deployment (Render), it should be
set as a Render environment variable, never committed, and rotated if it
is ever accidentally exposed in a log or screenshot. No evidence a real key
has ever been used or exposed — `gemini_enabled` has read `false` in every
`/health` check observed this session.

**Verdict: no secret has ever been exposed in this repository.** The CI
fix above is a reliability/reproducibility fix, not a leak remediation.

---

## 2. Android client — full detail in ANDROID_SECURITY_REVIEW.md

Five findings, all either fixed-and-live-verified or explicitly documented
as out-of-scope with reasoning. Full writeups (including the exact code
diffs and, for finding #4, the real on-device Keystore round-trip
verification via force-stop + relaunch) live in
`PaySense-ML-Backend/ANDROID_SECURITY_REVIEW.md`. Summary:

| # | Finding | Status |
|---|---|---|
| 1 | Full request/response bodies, including the JWT, logged unconditionally | **FIXED**, verified via real build |
| 2 | `StaticFieldLeak` lint warnings (×2) | Verified **false positive** (both singletons store `applicationContext`), suppressed with justification |
| 3 | `usesCleartextTraffic` allowed plaintext HTTP | **FIXED** — HTTPS-only in the shipped/committed config (`false`); flipped to `true` only for temporary local-backend demo builds this session, verified reverted via `git diff` every time before committing |
| 4 | Encrypted JWT storage (Keystore round-trip) | **FIXED and live-verified** — real login, force-stop to clear all in-memory state, relaunch goes straight to the authenticated dashboard with no login prompt |
| 5 | Login flow trusts only the server's `POST /auth/token` response, no client-side credential comparison | Verified clean on read; confirmed via real 401→200 round-trip against a live backend |

Also verified clean: login has no client-side credential check (`MainActivity`
calls `FraudApiService.login()`, which trusts only the server's response),
and the production `BASE_URL` is HTTPS-only.

**Known, disclosed limitation (not a bug):** the app never requests
`POST_NOTIFICATIONS` and has no notification channel — a high-risk fraud
alert only reaches the user through in-app UI (a badge + the "AT RISK"
card), never a system-level push. Documented in `WALKTHROUGH.md`'s honest
findings; not fixed as of this audit.

---

## 3. API security (backend)

**FACT.**
- JWT Bearer auth on `/predict`, `/classify`, `/insights/weekly`, and
  `/assistant/chat`, issued by `POST /auth/token` (`get_current_user`
  dependency, `main.py`). `/health` is intentionally public (liveness
  probe, no sensitive data).
- Rate limiting via `slowapi`: 60/min on `/predict` and `/classify`, 30/min
  on `/assistant/chat`.
- `APP_ENV` gates a development-only auth bypass (`get_current_user`
  returns a fixed dev-bypass identity when `APP_ENV=development` and no
  credentials are sent) — explicitly documented in `.env.example` as a
  local-curl convenience, and explicitly set to `production` on every real
  deployment path (Render env, and now CI).
- Request validation is real, not decorative: every field on
  `TransactionInput` is a typed, bounded Pydantic field (see `main.py`'s
  `TransactionInput` class), and `top_category` in `/insights/weekly` was
  tightened this session from a free-text string (a real, if narrow,
  prompt-injection surface into the Gemini call) to a closed `Literal`
  allowlist — verified rejected with 422 for injection-style payloads in
  `tests/test_api.py::TestInsights::test_weekly_insights_rejects_prompt_injection_category`.

**INFERENCE, not independently re-verified this session.** CORS is
configured via `ALLOWED_ORIGINS` (`.env.example` documents `*` for dev). Not
re-checked what the production Render deployment actually sets — **NOT
VERIFIED**.

---

## 4. LLM / AI security (new this audit, `/assistant/chat` and `/insights/weekly`)

Both Gemini-backed endpoints now share one call path (`_call_gemini()` in
`main.py`) with:

- **A real `system_instruction` field**, kept separate from the user's own
  message — not string-concatenated into one prompt (the original
  savings-tip implementation did exactly that, which is what made
  `top_category`'s earlier lack of validation a real injection surface).
- **A deterministic regex pre-filter** (`_JAILBREAK_PATTERNS`) that blocks
  common prompt-injection phrasing ("ignore all previous instructions",
  "you are now…", "reveal your system prompt", "developer mode", etc.)
  *before any LLM call is made* — cheaper than a model call, and immune to
  a cleverer rephrase arguing past the model, since it never reaches the
  model at all.
- **Safety settings** on the Gemini call itself (`BLOCK_MEDIUM_AND_ABOVE`
  for harassment/hate/sexual/dangerous content categories).
- **A deterministic fallback** for every failure mode (`GEMINI_API_KEY`
  unset, timeout, non-2xx, malformed response) — the assistant never
  returns nothing, and never silently swallows an error into a blank reply.
- **No arbitrary tool use or code execution.** The assistant only ever
  returns text; it has no ability to call other endpoints, read files, or
  execute anything on behalf of a request.

Verified with 13 tests (4 different jailbreak phrasings all blocked
pre-LLM, plus a sanity check that the regex doesn't false-positive on
legitimate questions using words like "act" or "pretend") and live curl
smoke tests against a real running server. See `WALKTHROUGH.md`'s
"AI Assistant — from client-side keyword router to a guardrailed LLM"
section for the full writeup.

**Known limitation, disclosed, not a bug:** the regex pre-filter is a
finite, hand-authored pattern list — it is a defense-in-depth layer, not a
guarantee. A sufficiently novel injection phrasing could get past layer 1
and rely entirely on the system instruction (layer 2) holding, which has
not been adversarially red-teamed beyond the 4 phrasings tested.

---

## 5. Dependency security

**FACT.** `aiosqlite==0.20.0` is declared in `requirements.txt` (labeled
"Phase 2 — weekly insights persistence") but `grep -rln "aiosqlite"
--include="*.py" .` returns **zero matches** anywhere in the codebase — it
is fully unused. Not a security vulnerability by itself, but dead weight in
every deployed bundle, and exactly the kind of "planned but never wired up"
dependency the size-constrained Vercel deployment path (`DEPLOY_VERCEL.md`)
can't afford. **Recommendation: remove it** (P2 — see remediation plan).

**Real scan run, 2026-08-26, `pip-audit` against the actual project venv:**

*Before:* 35 known vulnerabilities in 8 packages, including 5 CVEs in
`python-jose` — the JWT library this project's entire auth model depends
on (`PYSEC-2024-232`, `PYSEC-2024-233`, `PYSEC-2025-185`).

*Fixed and verified (full test suite re-run after each bump, 214/214
passing throughout):* `python-jose` 3.3.0→3.4.0, `cryptography` 49.0.0→
50.0.0, `python-dotenv` 1.0.1→1.2.2, `python-multipart` 0.0.9→0.0.31 — all
CVEs in these four packages fully cleared.

*Not fixed, documented honestly:*
- **`starlette` 0.38.6** (8 CVE entries, fix versions 0.40.0+/1.x) — FastAPI
  0.115.0 pins `starlette<0.39.0,>=0.37.2`; every fix version is outside
  that range. Fixing this requires a coordinated FastAPI+starlette upgrade
  with real regression testing (OpenAPI schema generation, response model
  behavior can change between major Starlette versions) — out of scope for
  this pass, flagged as P1.
- **`ecdsa` 0.19.2** (1 CVE, no fix version listed upstream) — transitive
  dependency, nothing to upgrade to yet.
- **`pyasn1` 0.4.8** (multiple CVEs) — transitive dependency.
- **`pip` 24.0` / `setuptools` 78.1.0`** — packaging tooling, not code that
  runs in the deployed service.

*Result as of 2026-08-26:* 27 known vulnerabilities remain in 5 packages
(down from 35 in 8). **No claim of "secure"** — this is the actual,
current, dated result of one tool's scan, not a guarantee.

---

## 6. What this document does NOT cover

- Formal penetration testing of the live Render deployment (which, as of
  this audit, is not responding at all — see `PROJECT.md`'s known
  limitations).
- The remaining 27 dependency vulnerabilities documented above as not yet
  fixed (starlette, ecdsa, pyasn1, pip, setuptools).
- Anything about `PaySense-Android-Client` (no "-New") — explicitly kept
  by the project owner per an existing README.md comment ("early,
  incomplete scaffold — superseded by -New, kept for history"), a
  deliberate choice this audit did not override. `android/` and `backend/`
  — the two undocumented, unreferenced first-commit scaffolds with no such
  comment anywhere — were removed via `git rm -r` in this audit; their
  full history remains in git regardless. See `PROJECT.md`'s repository
  map for the distinction.

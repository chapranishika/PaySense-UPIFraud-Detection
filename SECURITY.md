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

CORS is configured via `ALLOWED_ORIGINS` — see §6 for the 2026-08-28
live-verified finding (wildcard behavior confirmed locally, production
value NOT VERIFIED) and the startup-warning fix.

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

**FACT, corrected in the 2026-08-28 audit.** `aiosqlite` is **not** in
`requirements.txt` — it was removed in an earlier pass (this document
previously said otherwise; that was stale).

**Real scan run, 2026-08-26, `pip-audit` against the actual project venv:**

*Before:* 35 known vulnerabilities in 8 packages, including 5 CVEs in
`python-jose` — the JWT library this project's entire auth model depends
on (`PYSEC-2024-232`, `PYSEC-2024-233`, `PYSEC-2025-185`).

*Fixed and verified (full test suite re-run after each bump, 214/214
passing throughout):* `python-jose` 3.3.0→3.4.0, `cryptography` 49.0.0→
50.0.0, `python-dotenv` 1.0.1→1.2.2, `python-multipart` 0.0.9→0.0.31 — all
CVEs in these four packages fully cleared.

*Result as of 2026-08-26:* 27 known vulnerabilities remain in 5 packages
(down from 35 in 8).

**Re-scanned 2026-08-28, same tool, fresh run: 16 vulnerabilities in 3
packages** (`pyasn1==0.4.8`: 6, `starlette==0.38.6`: 9, `ecdsa==0.19.2`: 1)
— fewer packages/CVEs than the 2026-08-26 number above; the vulnerability
database itself changes over time (entries added/reclassified), so this is
a fresh count, not a claim that 11 CVEs were silently fixed between scans.

**Fixed in this pass:** `python-jose` 3.4.0→3.5.0. This was not primarily
about python-jose's own CVE history — it was pinned specifically because
**3.5.0 relaxes python-jose's own `pyasn1` constraint from `<0.5.0` to
`>=0.5.0`**, which had been blocking every pyasn1 fix version. `pyasn1`
bumped 0.4.8→0.6.4 in the same pass, clearing all 6 pyasn1 CVEs. Verified
safe via the full test suite (230/230 passing) before pinning either
version. **Result: 10 vulnerabilities remain in 2 packages** (`starlette`:
9, `ecdsa`: 1).

*Not fixed, documented honestly:*
- **`starlette` 0.38.6`** (9 CVE entries in the current scan) — FastAPI
  0.115.0 pins `starlette<0.39.0,>=0.37.2` (re-verified 2026-08-28 via
  `importlib.metadata.requires('fastapi')`); every fix version pip-audit
  lists is outside that range. Fixing this requires a coordinated
  FastAPI+starlette upgrade with real regression testing (OpenAPI schema
  generation, response model behavior can change between major Starlette
  versions) — out of scope for this pass, flagged as P1. Some of
  pip-audit's reported `fix_versions` for starlette (e.g. `1.0.1`, `1.3.0`)
  don't match Starlette's actual release history (which has never reached
  a 1.x version) — reported here exactly as the tool returned them, not
  edited, but flagged as a possible vulnerability-database data-quality
  issue rather than asserted as fact either way.
- **`ecdsa` 0.19.2`** (1 CVE, `fix_versions=[]` — pip-audit reports no
  patched version exists upstream at all) — transitive dependency of
  `python-jose`, nothing to upgrade to.
- **`pip`/`setuptools`** — packaging tooling, not code that runs in the
  deployed service; not re-scanned this pass.

**No claim of "secure"** — this is the actual, current, dated result of
one tool's scan against one file, not a guarantee, and not a claim that no
other vulnerability classes exist.

---

## 6. Configuration: fail-open defaults (found and hardened 2026-08-28)

**FACT.** Two config values default to their MORE PERMISSIVE state when
unset, rather than failing closed:
- `APP_ENV` defaults to `"development"` if unset — in that mode,
  `get_current_user()` accepts requests to `/predict`, `/classify`, and
  `/insights/weekly` with **no** `Authorization` header at all (a
  documented local-curl convenience). A deployment that forgets to set
  `APP_ENV=production` would silently run with this bypass active.
- `ALLOWED_ORIGINS` defaults to no origins if unset, but the documented
  `.env.example` value for local dev is `*` (any origin). **Verified live**
  against a local server: a CORS preflight from `Origin:
  https://evil.example.com` got back `Access-Control-Allow-Origin: *`
  when `ALLOWED_ORIGINS=*` was set. The API uses Bearer-token auth (not
  cookies) with no `allow_credentials=True`, so this is not a classic
  cookie-based CSRF vector — but it does mean any website's JavaScript
  could call `/health` (non-sensitive) or attempt credential-stuffing
  against `/auth/token` from a visitor's browser if this were ever left at
  `*` on a real deployment.

**Fixed:** neither default was changed (both have genuine local-dev value,
and changing either risks breaking existing setups) — instead, `main.py`
now logs a loud `WARNING` at startup for either condition, so a deployer
who forgot to override one sees it immediately in their logs instead of
finding out never or with a network-security-scan noticing a live gap.
Regression-tested: `tests/test_security_hardening.py::
test_non_production_app_env_would_trigger_a_warning` guards that these
checks stay in `main.py`.

**Live production CORS/APP_ENV configuration is NOT VERIFIED** — no
dashboard access to the Render deployment exists in this environment. The
finding above is about the CODE's behavior and the documented default, not
a claim about what is currently set on the live service.

---

## 7. `/auth/token` had no rate limit (found and fixed 2026-08-28)

**FACT, verified live against a running server.** Every other endpoint in
this API has a `@limiter.limit(...)` decorator; `/auth/token` — the only
unauthenticated endpoint that accepts a secret (username/password) — had
none. Confirmed empirically: 20 rapid wrong-password requests against a
local server all returned `401`, zero `429`s.

Practical severity is low today: the demo credentials are already
intentionally public (printed in the OpenAPI docs and this repo's own
README), so there is nothing secret left to brute-force. This is flagged
and fixed anyway because the code pattern — an auth endpoint with no rate
limit — would be a real brute-force vector if this project (or anyone
copying its structure) ever added real, non-public per-user credentials
without revisiting this endpoint.

**Fixed:** added `@limiter.limit("10/minute")` to `POST /auth/token`.
Regression-tested: `tests/test_security_hardening.py::
test_zz_auth_token_endpoint_is_rate_limited`.

---

## 8. Unbounded numeric input cannot crash the server or produce NaN

**FACT, verified live.** `amount_deviation_score` (and a few other
numeric fields) have no `ge`/`le` bound in `TransactionInput`, unlike most
other fields. Sent extreme values (`1e300`, `-1e300`) directly to a
running `/predict` endpoint: the response was `200`, with a finite
`fraud_score` in `[0, 1]` both times — `src/fraud_model.score()`'s final
`min(max(ensemble, 0.0), 1.0)` clamp, plus each individual scorer's own
`try/except`, holds up under this input. No fix was needed here; this is
recorded as a verified-safe finding, not a gap. Regression-tested:
`tests/test_security_hardening.py::
test_predict_extreme_unbounded_field_does_not_crash_or_produce_nan`
(parametrized over `1e300`, `-1e300`, `1e15`, `-1e15`).

---

## 9. Dockerfile was broken (found and fixed 2026-08-28)

**FACT, verified via a real `docker build`/`docker run` — Docker is
installed and was actually exercised, not just read.** `render.yaml` uses
Render's native Python buildpack (`env: python`), **not Docker** — the
Dockerfile/`docker-compose.yml` are an optional local-dev path only, never
used by the live deployment. That path was nonetheless completely broken,
two different ways:

1. `COPY artefacts/ artefacts/ 2>/dev/null || true` — Dockerfile `COPY`
   has no shell/conditional syntax; this was parsed as literal extra
   arguments and failed every build (`"/||": not found`). Fixed: plain
   `COPY artefacts/ artefacts/` (the directory is tracked in git and
   always present at build time, so no conditional was ever needed).
2. Once the build succeeded, the container **crashed on every startup**:
   the Dockerfile copied a hand-picked list of files that never included
   `src/`, and `main.py` does `from src.fraud_model import ...`
   (`ModuleNotFoundError: No module named 'src'`). Fixed by switching to
   `COPY . .` with a new `.dockerignore` (excludes `venv/`, datasets,
   logs, `.env`, `.git/`) — avoids this whole class of "forgot to list a
   file" bug rather than patching the specific omission.

**Verified fixed end-to-end:** a real `docker build` now succeeds, and a
real `docker run` (with the required env vars set) starts the full
ensemble (`GET /health` returns `ensemble_ready: true`,
`active_scorers: ["rules","paysense","light_lr"]`) — checked directly,
not inferred from the build succeeding alone.

---

## 10. Production deployment status (re-verified 2026-08-27/28)

**FACT, re-checked fresh (not assumed from an earlier session's finding).**
`https://paysense-api.onrender.com/health` and the root path both
returned `503 Service Unavailable` (with a `Retry-After: 5` header) across
3 separate requests. This environment has no Render dashboard access, so
the cause (suspended free-tier instance, a crash, or an intentional pause)
is **NOT VERIFIED** — only the symptom (currently down) is confirmed.
**Production deployment status: DOWN as of this check, cause unknown.**

**Update, 2026-08-29:** the actual cause turned out to be simpler than
either guess above — the Render dashboard showed the Production
environment had **zero services in it at all**; the service behind the
old URL no longer existed. A new Web Service was created
(`https://paysense-upifraud-detection.onrender.com`) from the same repo.
Its first deploy also failed, for an unrelated reason: Render's default
build image used Python 3.14, and `pandas==2.2.2` (pinned in
`requirements.txt`) has no prebuilt wheel for that version, so pip tried
to compile it from source and hit a Cython/C++ incompatibility. Fixed by
setting `PYTHON_VERSION=3.11.0` as an environment variable (the value
`render.yaml` already specified, but which only auto-applies via a
Blueprint deploy, not a manually-created service) and redeploying.
**Re-verified live:** `GET /health` on the new URL now returns
`status: ok`, `mode: production`, all three scorers active. Every
reference to the old URL in this repo (Android's `FraudApiService.kt`,
`README.md`, `DEPLOY.md`, `ARCHITECTURE.md`, and elsewhere) has been
updated to the new one.

---

## 11. What this document does NOT cover

- Formal penetration testing of the live Render deployment (currently
  down — see §10).
- Live production CORS/`APP_ENV` configuration (§6) — no dashboard access.
- The remaining 10 dependency vulnerabilities documented in §5 as not yet
  fixed (starlette, ecdsa).
- Anything about `PaySense-Android-Client` (no "-New") — explicitly kept
  by the project owner per an existing README.md comment ("early,
  incomplete scaffold — superseded by -New, kept for history"), a
  deliberate choice this audit did not override. `android/` and `backend/`
  — the two undocumented, unreferenced first-commit scaffolds with no such
  comment anywhere — were removed via `git rm -r` in an earlier audit;
  their full history remains in git regardless. See `PROJECT.md`'s
  repository map for the distinction.

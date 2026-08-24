# PaySense API — Deploying to Vercel

An alternative to `DEPLOY.md`'s Render setup. Read the size finding below
before deploying — it determines which Vercel plan setting is required, and
skipping it will produce a build that fails with a bundle-size error.

**This is not just an API deployment.** `main.py` serves a full web
dashboard (`static/index.html` + `app.js`) at `/`, in addition to the JSON
endpoints — real login, a transaction feed, an AI assistant tab, and a
finance tracker, all wired to the same live backend. Deploying this
function to Vercel puts that whole app online at `https://<project>.vercel.app/`,
not just an API surface — see `WALKTHROUGH.md`'s "The web app" section for
what it actually looks like.

---

## The size finding — measured, not estimated

Vercel Functions bundle *all* files reachable at build time with no
automatic tree-shaking for Python, and the standard limit for a Python
function is **500 MB uncompressed**.

This backend's runtime dependencies alone — fastapi, uvicorn, pydantic,
joblib, numpy, pandas, scikit-learn, xgboost, python-jose, slowapi,
python-dotenv, passlib, httpx, aiosqlite — were installed into a clean,
minimal virtual environment and measured directly: **502.4 MB**, already
over the standard limit before adding a single line of application code or
a model file. XGBoost's native binaries (170 MB) and scipy (117 MB, a hard
dependency of scikit-learn) account for most of it — there is no realistic
way to trim this stack under 500 MB while still serving the actual XGBoost
fraud model and the scikit-learn-based components (`light_lr.pkl`).

**The fix: enable Vercel's Large Functions.** Documented, GA-adjacent (beta)
support for uncompressed bundles up to **5 GB**, on the same runtimes
(Python included). New Vercel projects get this by default; existing
projects opt in with one setting:

```
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
```

as a project environment variable (Vercel dashboard → Project Settings →
Environment Variables, or `vercel env add`). Requires Fluid Compute with
Active CPU enabled, which is also the default for new projects.

## What's already prepared in this repo

- **`vercel.json`** — excludes everything not needed at runtime from the
  function bundle. `artefacts/` on its own is 272 MB, but only 6 files in
  it are actually loaded by `main.py`/`src/fraud_model.py`
  (`paysense_model.pkl`, `paysense_preprocessor.pkl`,
  `paysense_threshold.pkl`, `paysense_feature_names.pkl`, `light_lr.pkl`,
  `paysense_category_classifier.pkl` — 2.4 MB total). Every other artefact
  version (v1/v2/v4 category classifiers, every experimental fraud-model
  variant, the DistilBERT candidate, all `*_metrics.json` files), plus
  `external_data/`, cached `.npy`/`.csv` files, and `tests/`, are excluded.
  The exclude patterns were verified directly against the actual file list
  (every real bloat file matched, none of the 6 needed files matched) —
  not just written and assumed correct.
- **Path resolution already works unmodified.** `fraud_model.py`'s
  `_BASE`/`_ARTEFACTS` and `main.py`'s `static_dir` are both derived from
  `__file__`, not the current working directory — Vercel's execution model
  doesn't require any code change here.
- **Zero-config entrypoint detection.** Vercel looks for `main.py` with a
  top-level `app` variable — this file already has exactly that
  (`app = FastAPI(...)`), and the `if __name__ == "__main__": uvicorn.run(...)`
  block at the bottom is never triggered by Vercel's import-based execution,
  so no changes were needed there either.

## What still needs your Vercel account (not something this environment can do)

1. **Import the GitHub repo** into a new Vercel project.
2. **Set the project's Root Directory** to `PaySense-ML-Backend` — the repo
   is a monorepo with the Android client alongside the backend; Vercel
   needs to be told to build from this subdirectory.
3. **Set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`** as a project environment
   variable if the project doesn't already default to large functions.
4. **Set the same environment variables `.env` defines locally** — Vercel's
   dashboard, not a committed file (this repo's `.gitignore` already
   excludes `.env`, same as the Render setup): `JWT_SECRET_KEY`,
   `TOKEN_EXPIRE_MIN`, `API_DEMO_USER`, `API_DEMO_PASS`, `APP_ENV`,
   `ALLOWED_ORIGINS`, and optionally `GEMINI_API_KEY`.
5. **Deploy**, and update the Android client's `BASE_URL` in
   `FraudApiService.kt` to the resulting `*.vercel.app` domain (or a custom
   domain), the same step `DEPLOY.md` §5 describes for Render.

## Verifying after deploy

```bash
curl https://<your-project>.vercel.app/health
# Expected: ensemble_ready: true, paysense_loaded: true, light_lr_loaded: true,
#           category_classifier_loaded: true, mode: "production"
```

If `mode` comes back `"demo"` instead of `"production"`, the artefact files
didn't make it into the bundle — check that `vercel.json`'s `excludeFiles`
pattern wasn't accidentally matching the 6 needed files (verify with the
same fnmatch check used to build this config, or temporarily remove
`excludeFiles` entirely to confirm the base case works, then reintroduce it).

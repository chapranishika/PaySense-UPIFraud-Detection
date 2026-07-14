# PaySense API — Deployment Guide

Deploy the FastAPI fraud-detection backend to **Render** for free in ~10 minutes.

---

## Step 0 — Run the pipeline locally first (one time only)

The trained model (.pkl artefacts) must be generated on your machine and committed
to the repo — the pipeline scripts are not run on Render's free tier due to
memory/time limits.

```bash
cd PaySense-ML-Backend/

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Download your dataset CSVs into this folder, then:
python paysense_pipeline.py        # builds master dataset (~2 min)
python paysense_ml_pipeline.py     # trains XGBoost, saves to artefacts/ (~5 min)
python paysense_phase3.py          # tunes threshold, freezes model

# Verify artefacts/ now exists
ls artefacts/
# Expected: paysense_model.pkl  threshold.json  (or similar names)
```

---

## Step 1 — Push to GitHub (including artefacts/)

```bash
cd PaySense-ML-Backend/

# Edit .gitignore — temporarily ALLOW the artefacts folder for deployment
# Comment out or remove the line:  artefacts/

git add artefacts/
git add Dockerfile render.yaml main.py requirements.txt
git add paysense_pipeline.py paysense_ml_pipeline.py paysense_phase3.py
git commit -m "deploy: add trained model artefacts for Render deployment"
git push
```

> **Note:** The artefacts are synthetic-dataset model files, NOT sensitive.
> Committing them is fine. After deployment, you can re-add `artefacts/`
> to .gitignore — Render will use the committed version already.

---

## Step 2 — Create a Render account

Go to **https://render.com** → Sign up free with GitHub.

---

## Step 3 — Create a new Web Service

1. Dashboard → **New** → **Web Service**
2. Connect your GitHub account → select your `paysense` repo
3. Render will auto-detect `render.yaml` — settings should pre-fill:

   | Setting | Value |
   |---|---|
   | Name | `paysense-api` |
   | Region | Singapore (closest to India) |
   | Branch | `main` |
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | Plan | **Free** |

4. Click **Create Web Service**

Render will build and deploy — first build takes ~3–5 minutes.

---

## Step 4 — Get your live URL

After deployment completes, Render shows:

```
https://paysense-api.onrender.com
```

(The exact subdomain is auto-generated from your service name.)

**Verify it's working:**
```bash
curl https://paysense-api.onrender.com/health
# Expected: {"status":"ok","model_loaded":true,"threshold":0.5,"feature_count":41}
```

**Swagger UI:** https://paysense-api.onrender.com/docs

---

## Step 5 — Update the Android app

In `FraudApiService.kt`, change:

```kotlin
// BEFORE (local emulator only)
private const val BASE_URL = "http://10.0.2.2:8000/"

// AFTER (live Render deployment)
private const val BASE_URL = "https://paysense-api.onrender.com/"
```

Rebuild and run the Android app — it now calls the live deployed API.

---

## Important: Render Free Tier Behaviour

- **Spins down after 15 minutes of inactivity.** The first request after
  inactivity takes ~30 seconds to wake up (a "cold start"). This is normal
  on the free tier.
- **512 MB RAM.** Our model is small (XGBoost on 30K rows) — well within limits.
- **750 free hours/month.** More than enough for a demo/portfolio project.
- **No credit card required** for the free tier.

---

## Troubleshooting

**Build fails with `ModuleNotFoundError`:**
```bash
# Make sure requirements.txt includes all dependencies
pip freeze > requirements.txt   # from your local venv
```

**`model_loaded: false` in /health:**
```
The artefacts/ directory is missing from the repo.
Go back to Step 1 and make sure artefacts/ is committed.
```

**Cold start taking too long:**
```
Normal on free tier. Use UptimeRobot (free) to ping /health every 10 minutes
and keep the service warm:
https://uptimerobot.com → New Monitor → HTTP → URL: your /health endpoint
```

---

## Portfolio URL

Once live, add this to:
- Your GitHub repo description: `https://paysense-api.onrender.com/docs`
- Your resume: "Live demo: https://paysense-api.onrender.com/docs"
- Your LinkedIn project: same URL

The Swagger UI at `/docs` lets anyone explore the 43-field API and make
test predictions — it's the best demo you can give in an interview.

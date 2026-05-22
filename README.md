# FraudGuard — Two-Stage Fraud Detection Demo
### BRAC University · CSE Thesis 2025

Real Hybrid SVM running live. Stage 2 DistilBERT pre-cached.
Supports the 83× GPU cost reduction claim from the thesis.

---

## Project Structure

```
fraudguard/
├── backend/                   ← Deploy to Render
│   ├── app.py                 ← Flask API (real SVM inference)
│   ├── requirements.txt
│   └── artifacts/
│       ├── svm_hybrid.pkl     ← Trained on EMSCAD (17,880 records)
│       ├── tfidf.pkl          ← TF-IDF (50K features, unigrams+bigrams)
│       ├── meta_preprocessor.pkl
│       ├── lr_terms.json      ← Real LR coefficients for fraud terms
│       ├── stage2_cache.json  ← 30 pre-computed DistilBERT examples
│       ├── model_meta.json    ← All metrics + cost figures
│       └── top_metadata.json  ← RF feature importances
│
└── frontend/                  ← Deploy to Vercel / GitHub Pages
    └── index.html             ← Complete single-file app
```

---

## Real Numbers (from training on EMSCAD)

| Metric | Value |
|--------|-------|
| SVM throughput | 198,742 preds/sec |
| Escalation rate | 1.2% |
| Cost reduction | **83.3×** |
| SVM F1 | 0.8817 |
| SVM ROC-AUC | 0.993 |
| SVM Brier score | 0.0061 |
| False Positives | 2 (out of 3,576) |
| GPU hrs/year (BERT) | 23.5h |
| GPU hrs/year (two-stage) | 0.282h |

---

## STEP 1 — One-time setup: create two GitHub repos

Go to github.com and create two NEW empty repos:
- `fraudguard-backend`
- `fraudguard-frontend`

---

## STEP 2 — Open terminal, navigate to this folder

```bash
cd path/to/fraudguard
```

Replace `path/to/fraudguard` with wherever you unzipped this file.
On Windows use `cd C:\Users\YourName\Downloads\fraudguard`

---

## STEP 3 — Push backend to GitHub

Copy and run these commands ONE BY ONE in your terminal:

```bash
cd backend
git init
git add .
git commit -m "initial: real trained SVM + artifacts"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fraudguard-backend.git
git push -u origin main
cd ..
```

⚠ Replace `YOUR_USERNAME` with your actual GitHub username.

---

## STEP 4 — Deploy backend to Render

1. Go to https://render.com → sign in with GitHub
2. Click **New** → **Web Service**
3. Connect repo: `fraudguard-backend`
4. Fill in settings:
   - **Name**: `fraudguard-api` (or anything)
   - **Region**: Singapore (closest to BD)
   - **Branch**: `main`
   - **Root Directory**: *(leave blank)*
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 1 --timeout 120`
   - **Instance Type**: `Free`
5. Click **Create Web Service**
6. Wait ~3 minutes for build to finish
7. Copy your URL — looks like: `https://fraudguard-api-xxxx.onrender.com`

---

## STEP 5 — Put your Render URL into the frontend

Open `frontend/index.html` in WebStorm.

Find this line (around line 256):
```js
const API_URL = window.BACKEND_URL || 'https://fraudguard-api.onrender.com';
```

Replace `https://fraudguard-api.onrender.com` with your actual Render URL from Step 4.

Save the file.

---

## STEP 6 — Push frontend to GitHub

```bash
cd frontend
git init
git add .
git commit -m "initial: fraud detection demo frontend"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fraudguard-frontend.git
git push -u origin main
cd ..
```

⚠ Replace `YOUR_USERNAME` with your actual GitHub username.

---

## STEP 7 — Deploy frontend to Vercel

1. Go to https://vercel.com → sign in with GitHub
2. Click **Add New** → **Project**
3. Import repo: `fraudguard-frontend`
4. Settings:
   - **Framework Preset**: `Other`
   - **Root Directory**: *(leave blank — it's already in the root)*
   - **Build Command**: *(leave blank)*
   - **Output Directory**: *(leave blank)*
5. Click **Deploy**
6. Your live URL: `https://fraudguard-frontend.vercel.app`

---

## STEP 8 — Test everything before the defence

```bash
# 1. Wake the Render backend (free tier sleeps after 15 min)
curl https://YOUR-RENDER-URL.onrender.com/health

# Expected response:
# {"loaded":true,"status":"ok","svm_tps":198742}

# 2. Test a prediction
curl -X POST https://YOUR-RENDER-URL.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"fields":{"title":"Data Entry Work From Home","description":"Apply using the link below earn money","has_company_logo":0,"telecommuting":1}}'
```

---

## If you make changes later

### Update backend:
```bash
cd backend
git add .
git commit -m "update: describe what changed"
git push
```
Render auto-redeploys on every push.

### Update frontend:
```bash
cd frontend
git add .
git commit -m "update: describe what changed"
git push
```
Vercel auto-redeploys on every push.

---

## Defence day checklist

- [ ] 5 minutes before: visit `https://YOUR-RENDER-URL.onrender.com/health` to wake the server
- [ ] Open `https://fraudguard-frontend.vercel.app` on your laptop
- [ ] Try the 🚨 Fraud example — should route HIGH_RISK in ~50ms
- [ ] Try the ✅ Legit example — should route LOW_RISK instantly
- [ ] Try the ⚠ Review example — should escalate to Stage 2

## What to say to the panel

> "Stage 1 handles 98.8% of all inputs on a free CPU server — no GPU
> needed — at 198,742 predictions per second. DistilBERT needs a GPU
> and runs at 118 predictions per second. Running it universally would
> cost 83× more in GPU compute. The two-stage pipeline escalates only
> the uncertain 1.2% to Stage 2. This website proves the claim — it is
> running on Render's free tier with 512MB RAM and zero GPU attached,
> and Stage 1 responds in real time."

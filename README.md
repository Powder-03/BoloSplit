# BoloSplit Backend (Voice-First Splitwise)

Production-ready FastAPI + PostgreSQL backend for a voice-first Splitwise-style expense tracker.
Supports direct ledger execution, Hare-Niemeyer exact percentage splits, settlement recording, bilateral friend feeds, and Firebase Google/Phone authentication.

---

## 🚀 Quick Deploy to Render (Recommended for MVP)

Render gives you free HTTPS, automated CI/CD from GitHub (`git push` to deploy), and a managed PostgreSQL database with zero server maintenance.

### Option 1: Automatic Blueprint (1-Click)
1. Push this repository to a GitHub repository:
   ```bash
   git add .
   git commit -m "Initial commit for BoloSplit backend"
   git remote add origin https://github.com/<your-username>/BoloSplit_backend.git
   git push -u origin main
   ```
2. Go to [Render Dashboard](https://dashboard.render.com/) $\rightarrow$ **New +** $\rightarrow$ **Blueprint**.
3. Select your repository. Render reads `render.yaml` automatically.
4. In the environment prompts, supply:
   - `DATABASE_URL`: Your Supabase PostgreSQL connection string (see Supabase section below).
   - `GEMINI_API_KEY`: Your Google Gemini API key.
   - `FIREBASE_CREDENTIALS_JSON`: Paste the raw text of `firebase-service-account.json`.
5. Click **Apply**. Render will build and deploy your service with a public HTTPS URL (e.g. `https://bolosplit-backend.onrender.com`).

---

## ⚡ Using Supabase for Free Tier PostgreSQL

Supabase offers a free managed PostgreSQL database with 500MB storage and extensions included:

1. Create a free project at [supabase.com](https://supabase.com).
2. Go to **Project Settings** (⚙️) $\rightarrow$ **Database**.
3. Under **Connection String**, select **URI** (or **Session Pooler**):
   - Example: `postgresql://postgres:[YOUR-PASSWORD]@db.<project-ref>.supabase.co:5432/postgres`
   - Or Pooler: `postgresql://postgres.<project-ref>:[YOUR-PASSWORD]@aws-0-<region>.pooler.supabase.com:6543/postgres`
4. Replace `[YOUR-PASSWORD]` with the database password you chose during project creation.
5. Paste this connection string into `DATABASE_URL` in Render (or local `.env`).

> **Built-in Supabase Optimizations in this code**:
> - Disables prepared statement caching (`statement_cache_size=0`) when using Supabase poolers (port 6543).
> - Uses `pool_pre_ping=True` and `pool_recycle=300` to handle Supabase's idle connection drop gracefully without throwing 500 errors.
> - Automatically runs `CREATE EXTENSION IF NOT EXISTS pg_trgm` and creates all SQLModel tables on startup.

### Option 2: Standard Web Service
1. **New +** $\rightarrow$ **PostgreSQL** $\rightarrow$ name it `bolosplit-db` $\rightarrow$ copy the **Internal Database URL**.
2. **New +** $\rightarrow$ **Web Service** $\rightarrow$ link your GitHub repo:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**:
     - `DATABASE_URL`: (paste your Render PostgreSQL Internal Database URL)
     - `GEMINI_API_KEY`: (your Gemini API key)
     - `FIREBASE_PROJECT_ID`: `bolosplit-274e0`
     - `FIREBASE_CREDENTIALS_JSON`: (paste contents of `firebase-service-account.json`)
     - `ENVIRONMENT`: `production`
     - `DEBUG`: `false`

---

## 🐳 Container Deployment (GCP Cloud Run / AWS EC2)

The included `Dockerfile` is cloud-agnostic:

### Google Cloud Run
```bash
gcloud run deploy bolosplit-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ENVIRONMENT=production,FIREBASE_PROJECT_ID=bolosplit-274e0
```

### AWS EC2 Migration (Future)
When ready to migrate to an EC2 instance:
```bash
# On your EC2 Ubuntu/Amazon Linux instance:
sudo docker build -t bolosplit-backend .
sudo docker run -d -p 80:8000 --env-file .env --name bolosplit bolosplit-backend
```

---

## 💻 Local Development

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Run test suite
uv run pytest -v

# 3. Start local development server
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API documentation will be available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/health`

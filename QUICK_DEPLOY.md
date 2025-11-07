# ⚡ Quick Deploy Guide - 5 Minutes

## Fastest Path: Vercel + Railway

### Step 1: Push to GitHub (2 min)
```bash
cd /Users/nevinselby/Documents/Projects/AlgorithmicTrading
git init
git add .
git commit -m "Ready for deployment"
# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy Backend on Railway (2 min)
1. Go to [railway.app](https://railway.app) → Sign up with GitHub
2. **New Project** → **Deploy from GitHub repo** → Select your repo
3. **Add Service** → **Empty Service**
4. **Settings**:
   - **Root Directory**: `backend`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Variables** → Add: `ALLOWED_ORIGINS` = `*` (or your Vercel URL later)
6. Wait for deploy → Copy the URL (e.g., `https://your-app.up.railway.app`)

### Step 3: Deploy Frontend on Vercel (1 min)
1. Go to [vercel.com](https://vercel.com) → Sign up with GitHub
2. **New Project** → Import your GitHub repo
3. **Configure Project**:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Next.js (auto)
4. **Environment Variables**:
   - `NEXT_PUBLIC_API_URL` = Your Railway backend URL
5. **Deploy** → Done! ✅

### Step 4: Update CORS (30 sec)
1. Go back to Railway
2. **Variables** → Update `ALLOWED_ORIGINS` = Your Vercel URL (e.g., `https://your-app.vercel.app`)
3. Railway will auto-redeploy

---

## Alternative: Render (All-in-One)

### Deploy Backend
1. [render.com](https://render.com) → Sign up
2. **New** → **Web Service** → Connect repo
3. Settings:
   - **Root Directory**: `backend`
   - **Build**: `pip install -r requirements.txt`
   - **Start**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy → Copy URL

### Deploy Frontend
1. **New** → **Static Site** → Connect repo
2. Settings:
   - **Root Directory**: `frontend`
   - **Build**: `cd frontend && npm install && npm run build`
   - **Publish**: `frontend/.next`
3. **Environment**: `NEXT_PUBLIC_API_URL` = Backend URL
4. Deploy → Done! ✅

---

## 🎯 That's It!

Your site will be live at:
- Frontend: `https://your-app.vercel.app` (or Render URL)
- Backend: `https://your-app.up.railway.app` (or Render URL)

**Total time: ~5 minutes** 🚀


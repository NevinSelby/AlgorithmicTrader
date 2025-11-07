# Quick Deployment Guide - IterAI Platform

## 🚀 Fastest Free Deployment Options

### Option 1: Vercel (Frontend) + Railway (Backend) ⭐ RECOMMENDED

#### Frontend Deployment (Vercel - FREE)
1. **Push to GitHub** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub
   - Click "New Project"
   - Import your repository
   - **Root Directory**: Set to `frontend`
   - **Framework Preset**: Next.js (auto-detected)
   - Click "Deploy"
   - ✅ Done! You'll get a URL like `your-app.vercel.app`

#### Backend Deployment (Railway - FREE)
1. **Go to [railway.app](https://railway.app)**
2. **Sign up with GitHub**
3. **New Project** → **Deploy from GitHub repo**
4. **Select your repository**
5. **Add Service** → **Empty Service**
6. **Settings** → **Root Directory**: Set to `backend`
7. **Settings** → **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
8. **Variables** tab → Add environment variables if needed
9. **Deploy** → Railway will auto-detect Python and install dependencies
10. ✅ You'll get a URL like `your-app.up.railway.app`

#### Update Frontend API URLs
After backend is deployed, update frontend to use Railway URL:

1. **Create `frontend/.env.local`**:
   ```env
   NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app
   ```

2. **Update API calls in frontend** (or use environment variable):
   - Change `/api/stock/...` to `${process.env.NEXT_PUBLIC_API_URL}/stock/...`
   - Or keep relative paths if using Vercel rewrites (see below)

3. **Add Vercel Rewrites** (Optional - to keep `/api/*` paths):
   Create `frontend/vercel.json`:
   ```json
   {
     "rewrites": [
       {
         "source": "/api/:path*",
         "destination": "https://your-backend.up.railway.app/:path*"
       }
     ]
   }
   ```

---

### Option 2: Render (Both Frontend & Backend) - SIMPLER

#### Deploy Both on Render
1. **Go to [render.com](https://render.com)**
2. **Sign up with GitHub**

#### Backend (Web Service)
1. **New** → **Web Service**
2. **Connect GitHub repo**
3. **Settings**:
   - **Name**: `iterai-backend`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Deploy**

#### Frontend (Static Site)
1. **New** → **Static Site**
2. **Connect GitHub repo**
3. **Settings**:
   - **Name**: `iterai-frontend`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `frontend/.next`
4. **Environment Variables**:
   - `NEXT_PUBLIC_API_URL`: `https://your-backend.onrender.com`
5. **Deploy**

---

### Option 3: Vercel (Frontend) + Fly.io (Backend)

#### Backend on Fly.io
1. **Install Fly CLI**: `curl -L https://fly.io/install.sh | sh`
2. **Login**: `fly auth login`
3. **In `backend/` directory**:
   ```bash
   fly launch
   ```
4. **Follow prompts** (auto-detects Python)
5. **Deploy**: `fly deploy`

---

## 🔧 Quick Fixes Needed Before Deployment

### 1. Update API URLs in Frontend
If not using rewrites, update all API calls:

**File**: `frontend/components/TradingPlatform.tsx`
```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Then use: `${API_BASE}/stock/${symbol}`
```

### 2. CORS Configuration (Backend)
Update `backend/main.py` to allow your frontend domain:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. Environment Variables
Create `.env` files:

**Backend** (Railway/Render will use these):
- No secrets needed (using free APIs)

**Frontend** (Vercel):
- `NEXT_PUBLIC_API_URL`: Your backend URL

---

## 📝 Deployment Checklist

- [ ] Push code to GitHub
- [ ] Deploy backend (Railway/Render/Fly.io)
- [ ] Get backend URL
- [ ] Deploy frontend (Vercel/Render)
- [ ] Set frontend environment variable with backend URL
- [ ] Test all endpoints
- [ ] Update CORS in backend with frontend URL
- [ ] Test on mobile/desktop

---

## 🎯 Recommended: Vercel + Railway

**Why?**
- ✅ Vercel: Best Next.js support, instant deployments
- ✅ Railway: Easy Python deployment, free tier generous
- ✅ Both: GitHub integration, auto-deploy on push
- ✅ Free tier: More than enough for this project

**Time to deploy**: ~10 minutes total

---

## 🔗 Custom Domain Setup

After deployment:

1. **Vercel**: Settings → Domains → Add `iterai.net`
2. **Railway**: Settings → Networking → Add custom domain
3. **Update DNS**: Point `iterai.net` to Vercel, `api.iterai.net` to Railway

---

## 💡 Pro Tips

1. **Use GitHub Actions** for automated testing before deploy
2. **Monitor** Railway/Render usage (free tier limits)
3. **Cache** API responses to reduce backend calls
4. **CDN** is automatic with Vercel

---

## 🆘 Troubleshooting

**Backend not connecting?**
- Check CORS settings
- Verify backend URL is correct
- Check Railway/Render logs

**Frontend build fails?**
- Check Node version (Vercel auto-detects)
- Verify all dependencies in `package.json`

**API calls failing?**
- Check network tab in browser
- Verify environment variables are set
- Check backend logs


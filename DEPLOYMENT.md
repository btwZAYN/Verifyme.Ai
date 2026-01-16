# Deployment Guide - Verify Application

## 🌐 Deployment Options

This guide covers deploying the Verify face recognition application to production.

---

## ⚠️ Important Considerations

Your application has specific requirements that affect deployment:

1. **WebSocket Support** - Required for real-time face verification
2. **Computational Resources** - Face recognition needs CPU/memory
3. **Persistent Storage** - Face images and database need to persist
4. **Long-running Processes** - Backend needs to stay active
5. **Camera Access** - Frontend needs HTTPS for camera permissions

---

## 🎯 Recommended: Split Deployment

### Frontend → Vercel
### Backend → Railway (or Render/DigitalOcean)

This is the **best approach** for your application.

---

## 📦 Option 1: Frontend on Vercel + Backend on Railway

### Part A: Deploy Backend to Railway

#### 1. Prepare Backend for Deployment

Create `railway.json` in the root directory:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn backend.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

Create `Procfile` in the root directory:

```
web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Update `requirements.txt` to include all dependencies:

```txt
opencv-python-headless>=4.5.0
face_recognition>=1.3.0
numpy>=1.19.0
Pillow>=8.0.0
pytesseract>=0.3.8
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
websockets>=11.0
```

#### 2. Deploy to Railway

1. Go to [railway.app](https://railway.app)
2. Sign up/Login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect Python and deploy
6. Note your backend URL (e.g., `https://your-app.railway.app`)

#### 3. Configure Environment Variables (if needed)

In Railway dashboard:
- Go to Variables tab
- Add any environment variables

---

### Part B: Deploy Frontend to Vercel

#### 1. Update Frontend API URL

Create `.env.production` in `frontend/` directory:

```env
VITE_API_URL=https://your-app.railway.app
VITE_WS_URL=wss://your-app.railway.app
```

Update `frontend/src/pages/Auth.jsx`:

```javascript
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
```

Update `frontend/src/pages/Verification.jsx`:

```javascript
const WS_BASE = import.meta.env.VITE_WS_URL || `ws://${window.location.hostname}:8000`;
const SOCKET_URL = `${WS_BASE}/ws/verify`;
```

#### 2. Create `vercel.json` in `frontend/` directory:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

#### 3. Deploy to Vercel

**Option A: Using Vercel CLI**

```bash
cd frontend
npm install -g vercel
vercel login
vercel --prod
```

**Option B: Using Vercel Dashboard**

1. Go to [vercel.com](https://vercel.com)
2. Sign up/Login with GitHub
3. Click "Add New" → "Project"
4. Import your repository
5. Set root directory to `frontend`
6. Add environment variables:
   - `VITE_API_URL` = `https://your-app.railway.app`
   - `VITE_WS_URL` = `wss://your-app.railway.app`
7. Click "Deploy"

---

## 📦 Option 2: Full Stack on Render

Deploy both frontend and backend together.

### 1. Create `render.yaml` in root directory:

```yaml
services:
  # Backend Service
  - type: web
    name: verify-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
    
  # Frontend Service
  - type: web
    name: verify-frontend
    env: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: frontend/dist
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
```

### 2. Deploy to Render

1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub
3. Click "New" → "Blueprint"
4. Connect your repository
5. Render will use `render.yaml` to deploy both services
6. Note your URLs

---

## 📦 Option 3: Full Stack on Railway

### 1. Create separate services for frontend and backend

**Backend Service:**
- Deploy as shown in Option 1, Part A

**Frontend Service:**
- Create new service in same project
- Set build command: `cd frontend && npm install && npm run build`
- Set start command: `npx serve -s frontend/dist -p $PORT`
- Add environment variables for API URLs

---

## 🔧 Backend CORS Configuration

Update `backend/main.py` to allow your frontend domain:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://your-frontend.vercel.app",  # Add your Vercel domain
        "https://*.vercel.app",  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📁 File Storage Considerations

### Problem
Your app stores face images in local folders (`faces/`, `id_cards/`), which won't persist on serverless platforms.

### Solutions

**Option 1: Use Cloud Storage (Recommended)**
- AWS S3
- Cloudinary
- Google Cloud Storage
- Azure Blob Storage

**Option 2: Use Database with BLOB storage**
- PostgreSQL with bytea
- MongoDB GridFS

**Option 3: Use Platform Storage**
- Railway Volumes
- Render Disks

---

## 🗄️ Database Migration

Replace `users_database.json` with a real database:

### Option 1: PostgreSQL (Recommended)

```bash
# Install psycopg2
pip install psycopg2-binary sqlalchemy
```

Update `backend/core.py` to use PostgreSQL instead of JSON file.

### Option 2: MongoDB

```bash
pip install pymongo motor
```

### Free Database Hosting:
- **Supabase** (PostgreSQL) - Free tier
- **MongoDB Atlas** - Free tier
- **Railway** - PostgreSQL included
- **Render** - PostgreSQL included

---

## ✅ Pre-Deployment Checklist

- [ ] Update CORS origins in `backend/main.py`
- [ ] Add environment variables for API URLs
- [ ] Update `requirements.txt` with all dependencies
- [ ] Test WebSocket connections work over WSS (secure WebSocket)
- [ ] Implement proper database (not JSON file)
- [ ] Set up cloud storage for images
- [ ] Add error logging (Sentry, LogRocket)
- [ ] Implement rate limiting
- [ ] Add authentication tokens (JWT)
- [ ] Enable HTTPS for camera access
- [ ] Test on mobile devices
- [ ] Set up monitoring and alerts

---

## 🚀 Quick Start: Deploy to Railway + Vercel

### 1. Backend to Railway (5 minutes)

```bash
# In root directory
git add .
git commit -m "Prepare for deployment"
git push

# Go to railway.app
# New Project → Deploy from GitHub
# Select your repo
# Done! Note the URL
```

### 2. Frontend to Vercel (5 minutes)

```bash
cd frontend

# Create .env.production
echo "VITE_API_URL=https://your-app.railway.app" > .env.production
echo "VITE_WS_URL=wss://your-app.railway.app" >> .env.production

# Deploy
npx vercel --prod
```

### 3. Update Backend CORS

Add your Vercel URL to allowed origins in `backend/main.py`.

---

## 🐛 Common Deployment Issues

### Issue: "WebSocket connection failed"
- **Solution**: Make sure backend URL uses `wss://` (not `ws://`)
- Check CORS settings
- Verify WebSocket endpoint is accessible

### Issue: "Camera not working"
- **Solution**: Frontend must be served over HTTPS
- Vercel automatically provides HTTPS

### Issue: "Face images not persisting"
- **Solution**: Implement cloud storage (S3, Cloudinary)
- Don't rely on local file system

### Issue: "Backend timeout"
- **Solution**: Increase timeout limits
- Optimize face recognition code
- Use background tasks for heavy processing

---

## 💰 Cost Estimates

### Free Tier Options:
- **Vercel**: Free for frontend (100GB bandwidth/month)
- **Railway**: $5/month credit (enough for small apps)
- **Render**: Free tier available (with limitations)
- **Supabase**: Free PostgreSQL database

### Recommended for Production:
- **Frontend (Vercel)**: Free - $20/month
- **Backend (Railway)**: $5 - $20/month
- **Database (Supabase)**: Free - $25/month
- **Storage (Cloudinary)**: Free - $99/month

**Total**: $0 - $164/month depending on usage

---

## 📞 Need Help?

If you encounter issues during deployment:
1. Check deployment logs in platform dashboard
2. Verify environment variables are set correctly
3. Test API endpoints using Postman
4. Check CORS configuration
5. Verify WebSocket connections

---

**Good luck with your deployment! 🚀**

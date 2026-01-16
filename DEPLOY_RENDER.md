# Step-by-Step Guide: Deploy to Render

## 🎯 Overview

We'll deploy your Verify app to Render with:
- **Backend**: Python web service (supports WebSockets ✅)
- **Frontend**: Static site
- **Database**: PostgreSQL (optional, for now we'll keep JSON)

**Total Time**: ~20 minutes  
**Cost**: FREE (Render free tier)

---

## 📋 Prerequisites

Before starting:
- ✅ GitHub account
- ✅ Your project code
- ✅ Git installed on your computer

---

## 🚀 STEP 1: Prepare Your Project for Deployment

### 1.1 Update Backend CORS Settings

We need to allow your frontend domain to connect to the backend.

**File to edit**: `backend/main.py`

Find the CORS middleware section and update it:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://*.onrender.com",  # Allow all Render domains
        "*"  # For development - remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 1.2 Update Frontend API URLs

**Create new file**: `frontend/.env.production`

```env
VITE_API_URL=https://verify-backend.onrender.com
VITE_WS_URL=wss://verify-backend.onrender.com
```

**Note**: Replace `verify-backend` with your actual backend service name (we'll create this in Step 3)

### 1.3 Update Frontend Code to Use Environment Variables

**File to edit**: `frontend/src/pages/Auth.jsx`

Change line 5 from:
```javascript
const API_URL = "http://localhost:8000";
```

To:
```javascript
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
```

**File to edit**: `frontend/src/pages/Verification.jsx`

Change lines 5-6 from:
```javascript
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const SOCKET_URL = `${protocol}//${window.location.hostname}:8000/ws/verify`;
```

To:
```javascript
const WS_BASE = import.meta.env.VITE_WS_URL || `ws://${window.location.hostname}:8000`;
const SOCKET_URL = `${WS_BASE}/ws/verify`;
```

### 1.4 Update requirements.txt

Make sure your `requirements.txt` has all dependencies:

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

**Note**: We use `opencv-python-headless` instead of `opencv-python` for server deployment.

---

## 🔧 STEP 2: Push Your Code to GitHub

### 2.1 Initialize Git (if not already done)

```bash
cd e:\Intelli\Tracker

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for Render deployment"
```

### 2.2 Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click the **+** icon → **New repository**
3. Name it: `verify-app` (or any name you like)
4. **Don't** initialize with README (we already have code)
5. Click **Create repository**

### 2.3 Push to GitHub

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/verify-app.git

# Push code
git branch -M main
git push -u origin main
```

---

## 🌐 STEP 3: Deploy Backend to Render

### 3.1 Create Render Account

1. Go to [render.com](https://render.com)
2. Click **Get Started**
3. Sign up with GitHub (recommended)
4. Authorize Render to access your repositories

### 3.2 Create Backend Web Service

1. Click **New +** → **Web Service**
2. Connect your GitHub repository (`verify-app`)
3. Configure the service:

**Basic Settings:**
- **Name**: `verify-backend` (or your choice)
- **Region**: Choose closest to you
- **Branch**: `main`
- **Root Directory**: Leave empty (root of repo)
- **Runtime**: `Python 3`

**Build & Deploy:**
- **Build Command**: 
  ```bash
  pip install -r requirements.txt
  ```

- **Start Command**: 
  ```bash
  uvicorn backend.main:app --host 0.0.0.0 --port $PORT
  ```

**Instance Type:**
- Select **Free** (for testing)

4. Click **Create Web Service**

### 3.3 Wait for Deployment

- Render will start building your backend
- This takes 5-10 minutes
- Watch the logs for any errors
- Once you see "Application startup complete", it's ready!

### 3.4 Note Your Backend URL

- At the top of the page, you'll see your URL
- Example: `https://verify-backend.onrender.com`
- **Copy this URL** - you'll need it for the frontend!

---

## 🎨 STEP 4: Deploy Frontend to Render

### 4.1 Update Frontend Environment Variables

**Edit**: `frontend/.env.production`

Replace with your actual backend URL:
```env
VITE_API_URL=https://verify-backend.onrender.com
VITE_WS_URL=wss://verify-backend.onrender.com
```

**Commit and push this change:**
```bash
git add frontend/.env.production
git commit -m "Update production API URLs"
git push
```

### 4.2 Create Frontend Static Site

1. In Render dashboard, click **New +** → **Static Site**
2. Select your repository again
3. Configure:

**Basic Settings:**
- **Name**: `verify-frontend`
- **Branch**: `main`
- **Root Directory**: `frontend`

**Build Settings:**
- **Build Command**: 
  ```bash
  npm install && npm run build
  ```

- **Publish Directory**: 
  ```
  dist
  ```

**Environment Variables:**
Click **Add Environment Variable** and add:
- Key: `VITE_API_URL`, Value: `https://verify-backend.onrender.com`
- Key: `VITE_WS_URL`, Value: `wss://verify-backend.onrender.com`

4. Click **Create Static Site**

### 4.3 Wait for Deployment

- Frontend builds faster (2-3 minutes)
- Once complete, you'll get a URL
- Example: `https://verify-frontend.onrender.com`

---

## ✅ STEP 5: Test Your Deployed App

### 5.1 Open Your Frontend URL

Visit: `https://verify-frontend.onrender.com`

### 5.2 Test the Flow

1. **Home Page** - Should load with animations
2. **Sign Up** - Create a test account
3. **Login** - Login with your account
4. **Dashboard** - Should show your name
5. **Verification** - Allow camera, test face recognition

### 5.3 Check for Errors

Open browser console (F12) and check for:
- ❌ CORS errors
- ❌ WebSocket connection failures
- ❌ API errors

---

## 🐛 STEP 6: Troubleshooting Common Issues

### Issue 1: "CORS Error"

**Fix**: Update `backend/main.py` CORS settings:
```python
allow_origins=["https://verify-frontend.onrender.com", "*"]
```

Then redeploy backend:
```bash
git add backend/main.py
git commit -m "Fix CORS"
git push
```

Render will auto-redeploy.

### Issue 2: "WebSocket Connection Failed"

**Check**:
1. Backend URL uses `wss://` (not `ws://`)
2. Backend is running (check Render logs)
3. Environment variables are set correctly

### Issue 3: "Backend Timeout"

**Fix**: Render free tier has limitations
- Upgrade to paid tier ($7/month)
- Or optimize your face recognition code

### Issue 4: "Images Not Saving"

**Expected**: Render's free tier doesn't persist files
**Fix**: Implement cloud storage (we can do this later)

---

## 🎉 STEP 7: Your App is Live!

Congratulations! Your app is now deployed:

- **Frontend**: `https://verify-frontend.onrender.com`
- **Backend**: `https://verify-backend.onrender.com`
- **API Docs**: `https://verify-backend.onrender.com/docs`

### Share Your App:
- Send the frontend URL to anyone
- They can access it from any device
- Camera will work (HTTPS enabled)

---

## 📝 Important Notes

### Free Tier Limitations:
- ⚠️ **Spins down after 15 min of inactivity** (first request takes 30-60s to wake up)
- ⚠️ **No persistent storage** (uploaded images won't persist after restart)
- ⚠️ **750 hours/month** (enough for testing)

### Upgrade to Paid ($7/month per service):
- ✅ Always on (no spin down)
- ✅ Persistent storage
- ✅ Better performance
- ✅ Custom domains

---

## 🔄 Making Updates

Whenever you make changes:

```bash
# Make your changes
# Then commit and push
git add .
git commit -m "Your update message"
git push
```

Render will **automatically redeploy** both services! 🚀

---

## 📞 Need Help?

If something goes wrong:
1. Check Render logs (click on service → Logs tab)
2. Check browser console for errors
3. Verify environment variables are set
4. Make sure backend is running before testing frontend

---

## 🎯 Next Steps

After successful deployment:

1. **Add Custom Domain** (optional)
   - Go to Settings → Custom Domain
   - Add your domain (e.g., verify.yourdomain.com)

2. **Set Up Database** (recommended)
   - Add PostgreSQL database in Render
   - Migrate from JSON to database

3. **Add Cloud Storage** (recommended)
   - Use Cloudinary or AWS S3
   - Store face images in cloud

4. **Monitor Your App**
   - Set up error tracking (Sentry)
   - Monitor performance

---

**You're all set! Happy deploying! 🚀**

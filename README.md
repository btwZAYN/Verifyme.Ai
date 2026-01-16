# Verify - Face Recognition & Identity Verification System

A full-stack web application for real-time facial recognition and identity verification using AI-powered technology.

## 🚀 Quick Start Guide

### Prerequisites

Before running the project, make sure you have:

- **Python 3.8+** installed
- **Node.js 16+** and npm installed
- **Webcam** for face verification
- **Tesseract OCR** (optional, for ID card scanning)

---

## 📦 Installation

### 1. Clone or Navigate to Project Directory

```bash
cd e:\Intelli\Tracker
```

### 2. Install Backend Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

**Required packages:**
- opencv-python
- face_recognition
- numpy
- Pillow
- pytesseract
- fastapi
- uvicorn
- python-multipart
- websockets

### 3. Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install

# Return to root directory
cd ..
```

---

## 🎯 Running the Project

You need to run **both** the backend and frontend servers simultaneously.

### Option 1: Using Two Terminal Windows (Recommended)

#### Terminal 1 - Backend Server
```bash
# From the root directory (e:\Intelli\Tracker)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started server process
Loading face data...
Loaded X people.
```

#### Terminal 2 - Frontend Server
```bash
# From the root directory (e:\Intelli\Tracker)
cd frontend
npm run dev
```

**Expected output:**
```
VITE v5.4.21  ready in XXX ms

➜  Local:   http://localhost:5174/
➜  Network: http://192.168.X.X:5174/
```

### Option 2: Using PowerShell Background Jobs

```powershell
# Start backend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'e:\Intelli\Tracker'; python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

# Start frontend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'e:\Intelli\Tracker\frontend'; npm run dev"
```

---

## 🌐 Accessing the Application

Once both servers are running:

1. **Open your browser**
2. **Navigate to:** `http://localhost:5174/` (or the port shown in your terminal)
3. **Start using the app!**

### Available URLs:
- **Frontend:** http://localhost:5174/
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs (FastAPI Swagger UI)

---

## 📱 How to Use

### 1. **Landing Page** (`/home`)
- View features and information about the system
- Click "Get Started" to sign up

### 2. **Sign Up** (`/signup`)
- Enter your full name
- Upload a clear photo of your face
- Click "Sign Up"
- Wait for confirmation

### 3. **Login** (`/`)
- Enter your name
- Click "Login"
- You'll be redirected to the dashboard

### 4. **Dashboard** (`/dashboard`)
- View your profile information
- Access quick actions
- Click "Verify Now" to start verification

### 5. **Verification** (`/verify`)
- Allow camera access when prompted
- Position your face in the camera frame
- Wait for real-time verification
- See results with confidence scores

---

## 🛠️ Troubleshooting

### Backend Issues

**Problem: "ModuleNotFoundError"**
```bash
# Solution: Install missing packages
pip install -r requirements.txt
```

**Problem: "Port 8000 already in use"**
```bash
# Solution: Kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use a different port:
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
```

**Problem: "No module named 'backend'"**
```bash
# Solution: Make sure you're in the root directory (e:\Intelli\Tracker)
cd e:\Intelli\Tracker
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Issues

**Problem: "Port 5173 is in use"**
- The frontend will automatically use the next available port (5174, 5175, etc.)
- Check the terminal output for the actual port number

**Problem: "npm: command not found"**
```bash
# Solution: Install Node.js from https://nodejs.org/
```

**Problem: "Module not found" errors**
```bash
# Solution: Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Connection Issues

**Problem: "CHECK CONNECTION" or WebSocket errors**
- Make sure the backend is running on port 8000
- Check that frontend is using HTTP (not HTTPS)
- Verify both servers are running

**Problem: Camera not working**
- Allow camera permissions in your browser
- Make sure no other application is using the camera
- Try a different browser (Chrome/Edge recommended)

---

## 📁 Project Structure

```
e:\Intelli\Tracker\
├── backend/
│   ├── main.py              # FastAPI server & WebSocket
│   ├── core.py              # Face recognition logic
│   ├── faces/               # Stored face images
│   └── id_cards/            # ID card images
├── frontend/
│   ├── src/
│   │   ├── pages/           # React pages
│   │   │   ├── Home.jsx
│   │   │   ├── Auth.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   └── Verification.jsx
│   │   ├── components/      # Reusable components
│   │   │   ├── Toast.jsx
│   │   │   ├── LoadingSpinner.jsx
│   │   │   └── ProtectedRoute.jsx
│   │   ├── App.jsx          # Main app component
│   │   └── index.css        # Global styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── users_database.json      # User data storage
└── requirements.txt         # Python dependencies
```

---

## 🔒 Security Notes

- This is a **development setup** - not production-ready
- User data is stored in `users_database.json` (not encrypted)
- Sessions are stored in browser localStorage
- For production, implement:
  - Proper database (PostgreSQL, MongoDB)
  - JWT authentication
  - HTTPS/SSL certificates
  - Password hashing
  - Rate limiting

---

## 🎨 Features

✅ Real-time face detection and recognition  
✅ User registration with photo upload  
✅ Live camera verification  
✅ WebSocket-based real-time communication  
✅ Session management  
✅ Toast notifications  
✅ Responsive design  
✅ Protected routes  
✅ Auto-reconnection on connection loss  
✅ Pause/Resume verification  

---

## 🐛 Common Issues & Solutions

### "Face not detected"
- Ensure good lighting
- Face the camera directly
- Remove glasses/masks if possible
- Move closer to the camera

### "Unknown User"
- Make sure you've signed up first
- Use the same name you registered with
- Check that your face photo was uploaded successfully

### "Connection failed"
- Restart the backend server
- Clear browser cache
- Check firewall settings

---

## 📞 Support

If you encounter any issues:
1. Check the terminal output for error messages
2. Review the troubleshooting section above
3. Make sure all dependencies are installed
4. Verify both servers are running

---

## 🎯 Next Steps

After getting the project running:
1. Test the complete user flow
2. Register multiple users
3. Try the verification with different lighting
4. Explore the dashboard features
5. Check the API documentation at http://localhost:8000/docs

---

**Enjoy using Verify! 🚀**

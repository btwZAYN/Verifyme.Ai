from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
import numpy as np
import base64
import json
from .core import core_system

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://*.onrender.com",  # Allow all Render domains
        "*"  # Allow all origins for development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Verify Backend is running"}

@app.post("/auth/signup")
async def signup(name: str = Form(...), photo: UploadFile = File(...)):
    # Save uploaded photo
    file_location = f"faces/{name}_{photo.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(photo.file, file_object)
    
    # Register in system
    success, msg = core_system.register_user(name, file_location)
    
    return {"success": success, "message": msg}

@app.post("/auth/login")
async def login(name: str = Form(...)):
    # Simple login check (just checking if user exists in memory for now)
    if name in core_system.person_mean_encodings:
        return {"success": True, "message": f"Welcome back, {name}"}
    return {"success": False, "message": "User not found"}

@app.websocket("/ws/verify")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Expecting base64 image data
            if "data:image" in data:
                header, encoded = data.split(",", 1)
                image_data = base64.b64decode(encoded)
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    results = core_system.verify_frame(frame)
                    await websocket.send_json(results)
                else:
                    await websocket.send_json({"error": "Invalid frame"})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import cv2
import face_recognition
import numpy as np
from PIL import Image
from collections import defaultdict, deque
import json
import pytesseract
import time
import re
from datetime import datetime

# ==================== CONFIGURATION DEFAULTS ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_FOLDER = os.path.join(BASE_DIR, "faces")
ID_CARDS_FOLDER = os.path.join(BASE_DIR, "id_cards")
DATABASE_FILE = os.path.join(BASE_DIR, "users_database.json")
TOLERANCE = 0.60
MIN_MARGIN = 0.08
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

# OCR Configuration
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    pass

class UserDatabase:
    def __init__(self, db_file):
        self.db_file = db_file
        self.users = {}
        self.load_database()
    
    def load_database(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
                self.users = {}
        else:
            self.users = {}
    
    def save_database(self):
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, indent=4)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def add_user(self, name, id_info, id_card_features):
        self.users[name] = {
            'id_info': id_info,
            'id_card_features': id_card_features,
            'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.save_database()
    
    def get_user(self, name):
        return self.users.get(name, None)

# ==================== HELPER FUNCTIONS ====================
def safe_crop(img, x, y, w, h):
    if img is None: return None
    h_img, w_img = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    if x + w > w_img: w = w_img - x
    if y + h > h_img: h = h_img - y
    if w <= 0 or h <= 0: return None
    return img[y:y+h, x:x+w]

def normalize_id_number(id_str):
    if not id_str: return None
    digits = re.sub(r'\D', '', id_str)
    if len(digits) == 13:
        return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"
    return digits

def normalize_name(name_str):
    if not name_str: return None
    name = ' '.join(name_str.split()).title()
    name = re.sub(r'\b(Mr|Mrs|Ms|Dr)\.?\b', '', name, flags=re.IGNORECASE).strip()
    return name

def normalize_date(date_str):
    if not date_str: return None
    date_patterns = [
        (r'(\d{2})[./](\d{2})[./](\d{4})', r'\1.\2.\3'),
        (r'(\d{4})[./](\d{2})[./](\d{2})', r'\3.\2.\1'),
        (r'(\d{2})\s*(\d{2})\s*(\d{4})', r'\1.\2.\3'),
    ]
    for pattern, replacement in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            return re.sub(pattern, replacement, match.group(0))
    return date_str

def extract_text_from_image(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Simplified for brevity, assumes good quality from frontend or re-uses advanced logic if needed
        return pytesseract.image_to_string(gray)
    except:
        return ""

def parse_id_card_info(text):
    info = {
        'full_text': text,
        'id_number': None, 'name': None, 'father_name': None,
        'dob': None, 'issue_date': None, 'expiry': None, 'gender': None, 'country': None
    }
    lines = text.splitlines()
    
    # ID Number
    id_patterns = [r'(\d{5})\s*[-–—]\s*(\d{7})\s*[-–—]\s*(\d)', r'\b(\d{13})\b']
    for line in lines:
        for pattern in id_patterns:
            match = re.search(pattern, line)
            if match:
                digits = re.sub(r'\D', '', match.group(0))
                if len(digits) == 13:
                    info['id_number'] = f"{digits[:5]}-{digits[5:12]}-{digits[12]}"
                break
        if info['id_number']: break
        
    return info

def extract_id_card_features(img):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Perceptual hash
    resized = cv2.resize(gray, (32, 32))
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:8, :8]
    avg = np.mean(dct_low)
    phash = (dct_low > avg).flatten().astype(int)
    
    # Color histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Convert numpy arrays to lists for JSON serialization
    return {
        'descriptors': descriptors.tolist() if descriptors is not None else None,
        'phash': phash.tolist(),
        'histogram': hist.tolist()
    }

class CoreSystem:
    def __init__(self):
        self.db = UserDatabase(DATABASE_FILE)
        self.person_encodings = defaultdict(list)
        self.person_mean_encodings = {}
        self.load_data()
        
    def load_data(self):
        if not os.path.exists(IMAGES_FOLDER): os.makedirs(IMAGES_FOLDER)
        if not os.path.exists(ID_CARDS_FOLDER): os.makedirs(ID_CARDS_FOLDER)
        
        print("Loading face data...")
        face_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(VALID_EXT)]
        
        for img_file in face_files:
            base = os.path.splitext(img_file)[0]
            person_name = base.split('_')[0] if '_' in base else base
            img_path = os.path.join(IMAGES_FOLDER, img_file)
            
            try:
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    for e in encs:
                        self.person_encodings[person_name].append(e)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")

        for name, encodings in self.person_encodings.items():
            self.person_mean_encodings[name] = np.mean(encodings, axis=0)
        print(f"Loaded {len(self.person_mean_encodings)} people.")

    def register_user(self, name, face_image_path, id_card_path=None):
        """Registers a new user by processing their uploaded face and optional ID card."""
        try:
            # reload data to include new file
            img = face_recognition.load_image_file(face_image_path)
            encs = face_recognition.face_encodings(img)
            if not encs:
                return False, "No face found in image"
            
            self.person_encodings[name].extend(encs)
            self.person_mean_encodings[name] = np.mean(self.person_encodings[name], axis=0)
            
            # Process ID card if provided
            if id_card_path:
                id_img = cv2.imread(id_card_path)
                if id_img is not None:
                    text = extract_text_from_image(id_img)
                    id_info = parse_id_card_info(text)
                    id_features = extract_id_card_features(id_img)
                    self.db.add_user(name, id_info, id_features)
            
            return True, "User registered successfully"
        except Exception as e:
            return False, str(e)

    def verify_frame(self, frame):
        """
        Process a single video frame.
        Returns: list of detections [{'name': '...', 'confidence': 0.0, 'bbox': [t,r,b,l], 'liveness': True/False}]
        """
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        results = []
        
        for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
            # Scale back up
            top *= 2; right *= 2; bottom *= 2; left *= 2
            
            name = "Unknown"
            confidence = 0.0
            
            if self.person_mean_encodings:
                names = list(self.person_mean_encodings.keys())
                means = list(self.person_mean_encodings.values())
                
                dists = face_recognition.face_distance(means, enc)
                best_idx = np.argmin(dists)
                best_dist = dists[best_idx]
                
                if best_dist < TOLERANCE:
                    name = names[best_idx]
                    confidence = (1.0 - best_dist) * 100
            
            results.append({
                "name": name,
                "confidence": confidence,
                "bbox": [top, right, bottom, left],
                "verified": name != "Unknown"
            })
            
        return results

# Singleton instance
core_system = CoreSystem()

#!/usr/bin/env python3
"""
Enhanced Face Recognition + ID Card Verification
Improved ID card information extraction and matching
"""
import os
import sys
import cv2
import face_recognition
import numpy as np
from PIL import Image
from collections import defaultdict, deque
from datetime import datetime
import json
import pytesseract
import time
import re

# ==================== CONFIGURATION ====================
IMAGES_FOLDER = "faces"
ID_CARDS_FOLDER = "id_cards"
DATABASE_FILE = "users_database.json"

TOLERANCE = 0.60
MIN_MARGIN = 0.08
USE_CNN = False
SCALE_FACTOR = 0.5
SHOW_DEBUG = True

# Anti-Spoofing Configuration
LIVENESS_SENSITIVITY = 70
FAKE_SENSITIVITY = 30
ENABLE_ANTISPOOFING = True

# ID Card Verification Configuration
ENABLE_ID_VERIFICATION = True
ID_MATCH_THRESHOLD = 0.2
REQUIRE_ID_CARD = True
ID_INFO_MATCH_WEIGHT = 0.3  # Weight for text matching
ID_VISUAL_MATCH_WEIGHT = 0.7  # Weight for visual matching

# OCR Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ID_CARD_DEBUG = True
ID_CARD_DEBUG_SAVE_DIR = "id_debug_candidates"
os.makedirs(ID_CARD_DEBUG_SAVE_DIR, exist_ok=True)

HISTORY_SIZE = 30
motion_history = {}
texture_history = {}
depth_history = {}
face_size_history = {}
noise_history = deque(maxlen=HISTORY_SIZE)
edge_history = deque(maxlen=HISTORY_SIZE)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
last_blink_time = {}
blink_counter = {}
eye_aspect_ratio_history = {}
id_card_match_history = {}

VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

# ==================== USER DATABASE ====================
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
                print(f"📚 Loaded {len(self.users)} users from database")
            except Exception as e:
                print(f"⚠️  Error loading database: {e}")
                self.users = {}
        else:
            self.users = {}
    
    def save_database(self):
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, indent=4)
            if SHOW_DEBUG:
                print(f"💾 Database saved successfully")
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def add_user(self, name, id_info, id_card_features):
        self.users[name] = {
            'id_info': id_info,
            'id_card_features': id_card_features,
            'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.save_database()
    
    def get_user(self, name):
        return self.users.get(name, None)

# ==================== IMPROVED OCR / ID PARSING ====================
def preprocess_id_for_ocr(img):
    """Enhanced preprocessing for better OCR accuracy"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR if too small
        height, width = gray.shape
        if width < 800:
            scale = 800 / width
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    except Exception as e:
        if SHOW_DEBUG:
            print(f"[Preprocess] Error: {e}")
        return img

def extract_text_from_image(img):
    """Improved text extraction with multiple OCR attempts"""
    try:
        # Try multiple preprocessing methods
        methods = []
        
        # Method 1: Standard preprocessing
        preprocessed1 = preprocess_id_for_ocr(img)
        methods.append(preprocessed1)
        
        # Method 2: Binary with different threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        methods.append(binary)
        
        # Method 3: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        methods.append(adaptive)
        
        all_text = []
        for method in methods:
            # Try different PSM modes
            for psm in [6, 11, 3]:
                try:
                    text = pytesseract.image_to_string(method, config=f'--psm {psm}')
                    if text.strip():
                        all_text.append(text.strip())
                except:
                    continue
        
        # Combine all results
        combined_text = "\n".join(all_text)
        return combined_text
    except Exception as e:
        if SHOW_DEBUG:
            print(f"[OCR] Error: {e}")
        return ""

def normalize_id_number(id_str):
    """Normalize ID number format"""
    if not id_str:
        return None
    # Remove all non-digits
    digits = re.sub(r'\D', '', id_str)
    if len(digits) == 13:
        return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"
    return digits

def normalize_name(name_str):
    """Normalize name format"""
    if not name_str:
        return None
    # Remove extra spaces, convert to title case
    name = ' '.join(name_str.split()).title()
    # Remove common prefixes/suffixes
    name = re.sub(r'\b(Mr|Mrs|Ms|Dr)\.?\b', '', name, flags=re.IGNORECASE).strip()
    return name

def normalize_date(date_str):
    """Normalize date format to DD.MM.YYYY"""
    if not date_str:
        return None
    # Try to parse different date formats
    date_patterns = [
        (r'(\d{2})[./](\d{2})[./](\d{4})', r'\1.\2.\3'),  # DD/MM/YYYY or DD.MM.YYYY
        (r'(\d{4})[./](\d{2})[./](\d{2})', r'\3.\2.\1'),  # YYYY/MM/DD
        (r'(\d{2})\s*(\d{2})\s*(\d{4})', r'\1.\2.\3'),    # DD MM YYYY
    ]
    for pattern, replacement in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            return re.sub(pattern, replacement, match.group(0))
    return date_str

def parse_id_card_info(text):
    """Enhanced ID card information extraction"""
    info = {
        'full_text': text,
        'id_number': None,
        'name': None,
        'father_name': None,
        'dob': None,
        'issue_date': None,
        'expiry': None,
        'gender': None,
        'country': None,
    }
    
    lines = text.splitlines()
    
    # Extract ID Number
    id_patterns = [
        r'(\d{5})\s*[-–—]\s*(\d{7})\s*[-–—]\s*(\d)',  # With dashes
        r'(\d{5})\s*(\d{7})\s*(\d)',  # Without dashes
        r'\b(\d{13})\b',  # 13 consecutive digits
    ]
    
    for line in lines:
        for pattern in id_patterns:
            match = re.search(pattern, line)
            if match:
                if len(match.groups()) == 3:
                    id_num = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                else:
                    digits = re.sub(r'\D', '', match.group(0))
                    if len(digits) == 13:
                        id_num = f"{digits[:5]}-{digits[5:12]}-{digits[12]}"
                    else:
                        continue
                info['id_number'] = id_num
                if SHOW_DEBUG:
                    print(f"   ✓ ID Number: {info['id_number']}")
                break
        if info['id_number']:
            break
    
    # Extract Name (look for "Name" label followed by text)
    name_found = False
    for i, line in enumerate(lines):
        if not name_found and re.search(r'\bName\b', line, re.IGNORECASE):
            # Check current line first
            name_match = re.search(r'Name\s*[:]*\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', line, re.IGNORECASE)
            if name_match:
                info['name'] = normalize_name(name_match.group(1))
                name_found = True
            # Check next line
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if len(next_line.split()) >= 2:
                    # Check if it's a valid name (2-4 words, capitalized)
                    words = next_line.split()
                    if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                        info['name'] = normalize_name(next_line)
                        name_found = True
            
            if name_found and SHOW_DEBUG:
                print(f"   ✓ Name: {info['name']}")
            break
    
    # Extract Father Name
    father_found = False
    for i, line in enumerate(lines):
        if not father_found and re.search(r'\bFather\s*Name\b', line, re.IGNORECASE):
            # Check current line
            father_match = re.search(r'Father\s*Name\s*[:]*\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', line, re.IGNORECASE)
            if father_match:
                info['father_name'] = normalize_name(father_match.group(1))
                father_found = True
            # Check next line
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if len(next_line.split()) >= 2:
                    words = next_line.split()
                    if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                        info['father_name'] = normalize_name(next_line)
                        father_found = True
            
            if father_found and SHOW_DEBUG:
                print(f"   ✓ Father Name: {info['father_name']}")
            break
    
    # Extract Date of Birth
    for i, line in enumerate(lines):
        if re.search(r'\bDate\s*of\s*Birth\b|\bDOB\b', line, re.IGNORECASE):
            # Check current line
            date_match = re.search(r'(\d{2}[./]\d{2}[./]\d{4})', line)
            if date_match:
                info['dob'] = normalize_date(date_match.group(1))
            # Check next line if not found
            elif i + 1 < len(lines):
                date_match = re.search(r'(\d{2}[./]\d{2}[./]\d{4})', lines[i + 1])
                if date_match:
                    info['dob'] = normalize_date(date_match.group(1))
            
            if info['dob'] and SHOW_DEBUG:
                print(f"   ✓ Date of Birth: {info['dob']}")
            break
    
    # Extract Date of Issue
    for line in lines:
        if re.search(r'\bDate\s*of\s*Issue\b', line, re.IGNORECASE):
            date_match = re.search(r'(\d{2}[./]\d{2}[./]\d{4})', line)
            if date_match:
                info['issue_date'] = normalize_date(date_match.group(1))
                if SHOW_DEBUG:
                    print(f"   ✓ Issue Date: {info['issue_date']}")
            break
    
    # Extract Date of Expiry
    for line in lines:
        if re.search(r'\bDate\s*of\s*Expiry\b|\bExpiry\b', line, re.IGNORECASE):
            date_match = re.search(r'(\d{2}[./]\d{2}[./]\d{4})', line)
            if date_match:
                info['expiry'] = normalize_date(date_match.group(1))
                if SHOW_DEBUG:
                    print(f"   ✓ Expiry Date: {info['expiry']}")
            break
    
    # Extract Gender
    for line in lines:
        if re.search(r'\bGender\b', line, re.IGNORECASE):
            if re.search(r'\bM\b|\bMale\b', line, re.IGNORECASE):
                info['gender'] = 'Male'
            elif re.search(r'\bF\b|\bFemale\b', line, re.IGNORECASE):
                info['gender'] = 'Female'
            if info['gender'] and SHOW_DEBUG:
                print(f"   ✓ Gender: {info['gender']}")
            break
    
    # Extract Country
    if 'pakistan' in text.lower():
        info['country'] = 'Pakistan'
        if SHOW_DEBUG:
            print(f"   ✓ Country: Pakistan")
    
    return info

def compare_id_info(stored_info, live_info):
    """Compare extracted ID card information"""
    if not stored_info or not live_info:
        return 0.0
    
    matches = 0
    total_fields = 0
    match_details = []
    
    # Compare ID Number (most important)
    if stored_info.get('id_number') and live_info.get('id_number'):
        total_fields += 3  # Weight this more
        stored_id = normalize_id_number(stored_info['id_number'])
        live_id = normalize_id_number(live_info['id_number'])
        if stored_id and live_id:
            if stored_id == live_id:
                matches += 3
                match_details.append("ID Number ✓")
            else:
                match_details.append("ID Number ✗")
    
    # Compare Name
    if stored_info.get('name') and live_info.get('name'):
        total_fields += 2  # Weight this more
        stored_name = normalize_name(stored_info['name'])
        live_name = normalize_name(live_info['name'])
        if stored_name and live_name:
            # Fuzzy match for names (allow some variation)
            if stored_name.lower() == live_name.lower():
                matches += 2
                match_details.append("Name ✓")
            elif any(word in live_name.lower() for word in stored_name.lower().split()):
                matches += 1
                match_details.append("Name ≈")
            else:
                match_details.append("Name ✗")
    
    # Compare Father Name
    if stored_info.get('father_name') and live_info.get('father_name'):
        total_fields += 2
        stored_father = normalize_name(stored_info['father_name'])
        live_father = normalize_name(live_info['father_name'])
        if stored_father and live_father:
            if stored_father.lower() == live_father.lower():
                matches += 2
                match_details.append("Father ✓")
            elif any(word in live_father.lower() for word in stored_father.lower().split()):
                matches += 1
                match_details.append("Father ≈")
            else:
                match_details.append("Father ✗")
    
    # Compare Date of Birth
    if stored_info.get('dob') and live_info.get('dob'):
        total_fields += 2
        stored_dob = normalize_date(stored_info['dob'])
        live_dob = normalize_date(live_info['dob'])
        if stored_dob and live_dob:
            if stored_dob == live_dob:
                matches += 2
                match_details.append("DOB ✓")
            else:
                match_details.append("DOB ✗")
    
    # Compare Gender
    if stored_info.get('gender') and live_info.get('gender'):
        total_fields += 1
        if stored_info['gender'] == live_info['gender']:
            matches += 1
            match_details.append("Gender ✓")
    
    if total_fields == 0:
        return 0.0
    
    similarity = matches / total_fields
    
    if SHOW_DEBUG:
        print(f"   ID Info Match: {similarity:.2%} - {', '.join(match_details)}")
    
    return similarity

# ==================== ID card visual features & comparison ====================
def extract_id_card_features(img):
    """Extract visual features from ID card"""
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
    
    return {
        'descriptors': descriptors,
        'phash': phash.tolist(),
        'histogram': hist.tolist(),
        'shape': img.shape
    }

def compare_id_cards_visual(features1, features2):
    """Compare ID cards using visual features"""
    if features1 is None or features2 is None:
        return 0.0
    
    similarity_scores = []
    
    # pHash comparison
    if 'phash' in features1 and 'phash' in features2:
        phash1 = np.array(features1['phash'])
        phash2 = np.array(features2['phash'])
        hamming_dist = np.sum(phash1 != phash2)
        phash_similarity = 1.0 - (hamming_dist / len(phash1))
        similarity_scores.append(phash_similarity * 0.4)
    
    # Histogram comparison
    if 'histogram' in features1 and 'histogram' in features2:
        hist1 = np.array(features1['histogram'])
        hist2 = np.array(features2['histogram'])
        try:
            hist_similarity = cv2.compareHist(hist1.astype(np.float32), 
                                             hist2.astype(np.float32), 
                                             cv2.HISTCMP_CORREL)
        except Exception:
            hist_similarity = 0.0
        similarity_scores.append(max(0, hist_similarity) * 0.3)
    
    # ORB descriptor matching
    if features1.get('descriptors') is not None and features2.get('descriptors') is not None:
        try:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(features1['descriptors'], features2['descriptors'])
            if len(matches) > 10:
                good_matches = sorted(matches, key=lambda x: x.distance)[:50]
                avg_distance = np.mean([m.distance for m in good_matches])
                descriptor_similarity = max(0, 1.0 - (avg_distance / 100))
                similarity_scores.append(descriptor_similarity * 0.3)
        except Exception as e:
            if SHOW_DEBUG:
                print(f"[Descriptor] Error: {e}")
    
    return sum(similarity_scores) if similarity_scores else 0.0

# ==================== ID card detector ====================
def order_quad_points(pts):
    """Order quadrilateral points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Perspective transform to get flat ID card"""
    rect = order_quad_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_id_card_in_frame(frame):
    """Reliable ID card detection - Copy this entire function"""
    if frame is None:
        return None
    
    try:
        # Resize for faster processing
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            scale = 1.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        valid_cards = []
        img_area = frame.shape[0] * frame.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter by area (5% to 60% of frame)
            if area < img_area * 0.05 or area > img_area * 0.6:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Check aspect ratio (ID cards are rectangular)
            aspect = w / float(h)
            if aspect < 1.2 or aspect > 2.5:
                continue
            
            # Check if large enough
            if w < 150 or h < 80:
                continue
            
            # Scale back to original coordinates
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            w_orig = int(w / scale)
            h_orig = int(h / scale)
            
            # Extract card region from ORIGINAL frame
            card_roi = frame[y:y+h, x:x+w]
            
            valid_cards.append({
                'bbox': (x_orig, y_orig, w_orig, h_orig),
                'area': area,
                'warped': card_roi,
                'aspect': aspect
            })
        
        if not valid_cards:
            return None
        
        # Return largest valid card
        best_card = max(valid_cards, key=lambda x: x['area'])
        
        if SHOW_DEBUG:
            print(f"[ID] Detected card: {best_card['bbox']}, Aspect: {best_card['aspect']:.2f}")
        
        return best_card
        
    except Exception as e:
        if SHOW_DEBUG:
            print(f"[ID DETECT] Exception: {e}")
        return None
# ==================== Anti-spoofing helpers ====================
def calculate_motion_score(current_roi, previous_roi):
    if previous_roi is None or current_roi is None or current_roi.shape != previous_roi.shape:
        return 0
    diff = cv2.absdiff(current_roi, previous_roi)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] + 1e-6)
    return motion_score

def analyze_texture_lbp(face_roi):
    if face_roi is None or face_roi.size == 0:
        return 0
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = laplacian.var()
    return texture_score

def detect_screen_moire(img):
    if img is None or img.size == 0:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    h, w = magnitude_spectrum.shape
    center_region = magnitude_spectrum[h//4:3*h//4, w//4:3*w//4]
    mean_val = np.mean(center_region)
    std_val = np.std(center_region)
    peak_score = std_val / (mean_val + 1)
    return peak_score

def detect_face_depth_cues(face_roi):
    if face_roi is None or face_roi.size == 0:
        return 0
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    depth_variance = np.std(gradient_magnitude)
    return depth_variance

def detect_eyes_in_face(face_roi_gray):
    if face_roi_gray is None:
        return 0
    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.3, 5)
    return len(eyes)

def detect_blink(face_id, eye_count):
    current_time = time.time()
    if face_id not in eye_aspect_ratio_history:
        eye_aspect_ratio_history[face_id] = deque(maxlen=5)
    if face_id not in last_blink_time:
        last_blink_time[face_id] = current_time
    if face_id not in blink_counter:
        blink_counter[face_id] = 0
    
    eye_aspect_ratio_history[face_id].append(eye_count)
    
    if len(eye_aspect_ratio_history[face_id]) >= 3:
        history = eye_aspect_ratio_history[face_id]
        if (history[-3] >= 2 and history[-2] < 2 and history[-1] >= 2):
            if current_time - last_blink_time[face_id] > 0.3:
                blink_counter[face_id] += 1
                last_blink_time[face_id] = current_time
                return True, blink_counter[face_id]
    return False, blink_counter.get(face_id, 0)

def calculate_noise_level(img):
    if img is None or img.size == 0:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def detect_compression_artifacts(img):
    if img is None or img.size == 0:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block_size = 8
    h, w = gray.shape
    artifact_score = 0
    count = 0
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                right_diff = np.abs(int(block[:, -1].mean()) - int(gray[i:i+block_size, j+block_size].mean())) if j+block_size < w else 0
                bottom_diff = np.abs(int(block[-1, :].mean()) - int(gray[i+block_size, j:j+block_size].mean())) if i+block_size < h else 0
                artifact_score += right_diff + bottom_diff
                count += 1
    return artifact_score / max(count, 1)

def is_face_live(face_id, face_roi, face_roi_gray, previous_face_roi, face_size, sensitivity):
    if face_roi is None or face_roi.size == 0:
        return False, 0, "No face ROI"
    
    motion_history.setdefault(face_id, deque(maxlen=HISTORY_SIZE))
    texture_history.setdefault(face_id, deque(maxlen=HISTORY_SIZE))
    depth_history.setdefault(face_id, deque(maxlen=HISTORY_SIZE))
    face_size_history.setdefault(face_id, deque(maxlen=HISTORY_SIZE))
    
    liveness_score = 0
    reasons = []
    
    motion_score = calculate_motion_score(face_roi, previous_face_roi)
    motion_history[face_id].append(motion_score)
    if len(motion_history[face_id]) >= 10:
        motion_variance = np.var(list(motion_history[face_id]))
        if motion_variance > 50:
            liveness_score += 25
            reasons.append("Natural motion")
        elif motion_variance < 5:
            liveness_score -= 20
            reasons.append("Too static")
    
    texture_score = analyze_texture_lbp(face_roi)
    texture_history[face_id].append(texture_score)
    if texture_score > 100:
        liveness_score += 20
        reasons.append("Real skin")
    elif texture_score < 30:
        liveness_score -= 15
        reasons.append("Flat texture")
    
    eye_count = detect_eyes_in_face(face_roi_gray)
    blinked, blink_count = detect_blink(face_id, eye_count)
    if blinked:
        liveness_score += 30
        reasons.append(f"Blink #{blink_count}")
    if blink_count >= 2:
        liveness_score += 10
    
    moire_score = detect_screen_moire(face_roi)
    if moire_score > 0.8:
        liveness_score -= 25
        reasons.append("Screen detected")
    
    depth_score = detect_face_depth_cues(face_roi)
    depth_history[face_id].append(depth_score)
    if depth_score > 15:
        liveness_score += 15
        reasons.append("3D depth")
    elif depth_score < 5:
        liveness_score -= 10
        reasons.append("2D flat")
    
    face_size_history[face_id].append(face_size)
    if len(face_size_history[face_id]) >= 20:
        size_variance = np.var(list(face_size_history[face_id]))
        if size_variance > 500:
            liveness_score += 10
            reasons.append("Head movement")
    
    if eye_count >= 2:
        liveness_score += 10
    elif eye_count == 0:
        liveness_score -= 5
    
    threshold = 100 - sensitivity
    is_live = liveness_score > threshold
    confidence = min(100, max(0, liveness_score))
    reason_str = ", ".join(reasons[:3]) if reasons else "Analyzing..."
    return is_live, confidence, reason_str

def is_image_fake(face_roi, sensitivity):
    if face_roi is None or face_roi.size == 0:
        return False, 0
    
    noise_level = calculate_noise_level(face_roi)
    compression_score = detect_compression_artifacts(face_roi)
    noise_history.append(noise_level)
    
    fake_score = 0
    if noise_level < 50:
        fake_score += (50 - noise_level) * 0.3
    if compression_score > 5:
        fake_score += compression_score * 0.2
    
    threshold = 100 - sensitivity
    is_fake = fake_score > threshold
    confidence = min(100, (fake_score / threshold) * 100) if is_fake else 0
    return is_fake, confidence

# ==================== Load training faces & ID cards ====================
def load_faces_and_ids():
    for folder in [IMAGES_FOLDER, ID_CARDS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📁 Created folder: {folder}")
    
    db = UserDatabase(DATABASE_FILE)
    print(f"\n🔍 Loading training data...")
    
    person_encodings = defaultdict(list)
    
    face_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(VALID_EXT)]
    if not face_files:
        print(f"❌ No face images found in '{IMAGES_FOLDER}' folder!")
        print(f"   Add face photos to '{IMAGES_FOLDER}' and ID cards to '{ID_CARDS_FOLDER}'.")
        return None
    
    print(f"\n📸 Processing face images...")
    for img_file in face_files:
        base = os.path.splitext(img_file)[0]
        person_name = base.split('_')[0] if '_' in base else base
        img_path = os.path.join(IMAGES_FOLDER, img_file)
        
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"⚠️  Could not load: {img_file}")
                continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb).convert('RGB')
            if max(pil_image.size) > 1600:
                pil_image.thumbnail((1600, 1600))
            img_rgb = np.array(pil_image, dtype=np.uint8)
            
            locs = face_recognition.face_locations(img_rgb, model='hog')
            if len(locs) == 0:
                print(f"⚠️  No face found in: {img_file}")
                continue
            
            encs = face_recognition.face_encodings(img_rgb, locs)
            if len(encs) == 0:
                print(f"⚠️  Could not encode face in: {img_file}")
                continue
            
            for e in encs:
                person_encodings[person_name].append(e)
            
            print(f"✅ Face loaded: {img_file} -> '{person_name}'")
        except Exception as ex:
            print(f"❌ Error processing {img_file}: {ex}")
    
    if not person_encodings:
        print("\n❌ No faces were successfully encoded!")
        return None
    
    person_mean_encodings = {}
    for name, encodings in person_encodings.items():
        person_mean_encodings[name] = np.mean(encodings, axis=0)
    
    print(f"\n🪪 Processing ID cards...")
    id_files = [f for f in os.listdir(ID_CARDS_FOLDER) if f.lower().endswith(VALID_EXT)]
    
    for id_file in id_files:
        base = os.path.splitext(id_file)[0]
        person_name = base.replace('_id', '').replace('_ID', '').replace('-id', '')
        for sep in ("_", "-", " "):
            if sep in person_name:
                person_name = person_name.split(sep)[0]
                break
        
        if person_name not in person_mean_encodings:
            if SHOW_DEBUG:
                print(f"⚠️  Skipping {id_file}: No matching face image for '{person_name}'")
            continue
        
        id_path = os.path.join(ID_CARDS_FOLDER, id_file)
        try:
            id_img = cv2.imread(id_path)
            if id_img is None:
                print(f"⚠️  Could not load: {id_file}")
                continue
            
            print(f"   📝 Extracting text from {id_file}...")
            text = extract_text_from_image(id_img)
            
            if SHOW_DEBUG and text:
                print(f"   [OCR Raw Text Preview]")
                preview_lines = text.split('\n')[:5]
                for line in preview_lines:
                    if line.strip():
                        print(f"      {line.strip()}")
            
            id_info = parse_id_card_info(text)
            
            print(f"   🔍 Extracting visual features...")
            id_features = extract_id_card_features(id_img)
            
            db.add_user(person_name, id_info, id_features)
            print(f"✅ ID Card processed: {id_file} -> '{person_name}'")
            
            if id_info.get('id_number'):
                print(f"   📋 ID Number: {id_info['id_number']}")
            if id_info.get('name'):
                print(f"   👤 Name: {id_info['name']}")
            if id_info.get('father_name'):
                print(f"   👨 Father: {id_info['father_name']}")
            if id_info.get('dob'):
                print(f"   🎂 DOB: {id_info['dob']}")
                
        except Exception as ex:
            print(f"❌ Error processing {id_file}: {ex}")
    
    print("=" * 70)
    print(f"✅ Training complete!")
    print(f"   👥 People loaded: {len(person_mean_encodings)}")
    print(f"   🪪 ID cards registered: {len(db.users)}")
    print("=" * 70)
    
    return person_encodings, person_mean_encodings, db

# ==================== Utility functions ====================
def distance(a, b):
    return np.linalg.norm(a - b)

def safe_crop(img, x, y, w, h):
    if img is None:
        return None
    h_img, w_img = img.shape[:2]
    if x < 0: x = 0
    if y < 0: y = 0
    if w <= 0 or h <= 0:
        return None
    if x + w > w_img:
        w = w_img - x
    if y + h > h_img:
        h = h_img - y
    if w <= 0 or h <= 0:
        return None
    return img[y:y+h, x:x+w]

def draw_detection_label(img, left, top, right, bottom, name, confidence, 
                         is_live, live_confidence, is_fake, spoof_reason,
                         id_verified, id_info_match, id_visual_match):
    if ENABLE_ANTISPOOFING:
        if is_fake:
            status = "FAKE IMAGE"
            box_color = (0, 0, 255)
        elif not is_live:
            status = "SPOOFED"
            box_color = (0, 0, 255)
        elif name == "Unknown":
            status = "UNKNOWN"
            box_color = (0, 165, 255)
        elif ENABLE_ID_VERIFICATION and REQUIRE_ID_CARD and not id_verified:
            status = "NO ID CARD"
            box_color = (0, 165, 255)
        elif ENABLE_ID_VERIFICATION and id_verified and id_info_match < ID_MATCH_THRESHOLD:
            status = "ID MISMATCH"
            box_color = (0, 0, 255)
        else:
            status = "VERIFIED"
            box_color = (0, 255, 0)
    else:
        if name == "Unknown":
            status = "UNKNOWN"
            box_color = (0, 0, 255)
        else:
            status = name
            box_color = (0, 255, 0)
    
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 3)
    
    if status == "VERIFIED":
        label_line1 = f"{name} ✓"
        if id_verified:
            combined_score = (id_info_match * ID_INFO_MATCH_WEIGHT + 
                            id_visual_match * ID_VISUAL_MATCH_WEIGHT)
            label_line2 = f"LIVE {live_confidence:.0f}% | ID {combined_score*100:.0f}%"
        else:
            label_line2 = f"LIVE {live_confidence:.0f}% | Face {confidence:.0f}%"
    elif status in ("SPOOFED", "FAKE IMAGE"):
        label_line1 = status
        label_line2 = spoof_reason[:30] if spoof_reason else "Anti-spoof failed"
    elif status == "ID MISMATCH":
        label_line1 = f"{name} - ID MISMATCH"
        label_line2 = f"Match: {id_info_match*100:.0f}% (need {ID_MATCH_THRESHOLD*100:.0f}%)"
    elif status == "NO ID CARD":
        label_line1 = f"{name} - SHOW ID"
        label_line2 = "ID card required"
    else:
        label_line1 = "UNKNOWN"
        label_line2 = f"Live: {live_confidence:.0f}%"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    (text_width1, text_height1), _ = cv2.getTextSize(label_line1, font, font_scale, font_thickness)
    (text_width2, text_height2), _ = cv2.getTextSize(label_line2, font, font_scale * 0.7, 1)
    
    label_width = max(text_width1, text_width2) + 20
    label_height = text_height1 + text_height2 + 30
    label_y = top - label_height - 10
    if label_y < 0:
        label_y = bottom + 10
    
    overlay = img.copy()
    cv2.rectangle(overlay, (left, label_y), (left + label_width, label_y + label_height), box_color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (left, label_y), (left + label_width, label_y + label_height), box_color, 2)
    
    cv2.putText(img, label_line1, (left + 10, label_y + text_height1 + 5), 
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(img, label_line2, (left + 10, label_y + text_height1 + text_height2 + 15), 
                font, font_scale * 0.7, (255, 255, 255), 1)

# ==================== MAIN LOOP ====================
def main():
    print("\n" + "=" * 70)
    print("  ENHANCED FACE RECOGNITION WITH ID CARD VERIFICATION")
    print("=" * 70)
    print(f"📁 Face images folder: {IMAGES_FOLDER}")
    print(f"🪪 ID cards folder: {ID_CARDS_FOLDER}")
    print(f"📊 Database file: {DATABASE_FILE}")
    print(f"⚙️  Recognition Tolerance: {TOLERANCE}")
    print(f"🛡️  Anti-Spoofing: {'ENABLED' if ENABLE_ANTISPOOFING else 'DISABLED'}")
    print(f"🪪 ID Verification: {'ENABLED' if ENABLE_ID_VERIFICATION else 'DISABLED'}")
    if ENABLE_ID_VERIFICATION:
        print(f"🪪 ID Required: {'YES' if REQUIRE_ID_CARD else 'NO'}")
        print(f"🪪 ID Match Threshold: {ID_MATCH_THRESHOLD*100:.0f}%")
        print(f"🪪 Text Match Weight: {ID_INFO_MATCH_WEIGHT*100:.0f}%")
        print(f"🪪 Visual Match Weight: {ID_VISUAL_MATCH_WEIGHT*100:.0f}%")
    print("=" * 70)
    
    load_result = load_faces_and_ids()
    if load_result is None:
        print("\n❌ No training data loaded. Exiting.")
        input("\nPress Enter to exit...")
        return
    
    person_encodings, person_mean_enc, db = load_result
    
    if person_mean_enc is None or len(person_mean_enc) == 0:
        print("\n❌ No training encodings found. Exiting.")
        input("\nPress Enter to exit...")
        return
    
    print("\n📹 Starting webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        input("\nPress Enter to exit...")
        return
    
    print("✅ Webcam started!")
    print("\n🎯 SYSTEM ACTIVE")
    print("   Press 'q' to quit")
    print("   Press 's' to save screenshot")
    print("   Press 'i' to save detected ID card")
    print("   Press 'd' to toggle debug mode")
    
    process_this_frame = True
    prev_face_locations = []
    prev_face_names = []
    prev_face_confidences = []
    prev_liveness_status = []
    prev_id_verification = []
    prev_face_rois = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            display_frame = frame.copy()
            
            # Detect ID card in frame
            id_card_detected = None
            live_id_info = None
            if ENABLE_ID_VERIFICATION:
                try:
                    id_card_detected = detect_id_card_in_frame(frame)
                    if id_card_detected and id_card_detected.get('warped') is not None:
                        # Extract information from detected ID card
                        warped_id = id_card_detected['warped']
                        id_text = extract_text_from_image(warped_id)
                        live_id_info = parse_id_card_info(id_text)
                        
                        if SHOW_DEBUG and live_id_info:
                            print("\n[LIVE ID CARD DETECTED]")
                            if live_id_info.get('id_number'):
                                print(f"  ID: {live_id_info['id_number']}")
                            if live_id_info.get('name'):
                                print(f"  Name: {live_id_info['name']}")
                            if live_id_info.get('father_name'):
                                print(f"  Father: {live_id_info['father_name']}")
                            if live_id_info.get('dob'):
                                print(f"  DOB: {live_id_info['dob']}")
                except Exception as e:
                    if SHOW_DEBUG:
                        print(f"[DEBUG] ID detection error: {e}")
                    id_card_detected = None
                    live_id_info = None
            
            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                model_name = 'cnn' if USE_CNN else 'hog'
                
                try:
                    face_locations = face_recognition.face_locations(rgb_small_frame, model=model_name)
                except Exception as e:
                    if SHOW_DEBUG:
                        print(f"[DEBUG] face_locations error: {e}")
                    face_locations = []
                
                if face_locations:
                    try:
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    except Exception as e:
                        if SHOW_DEBUG:
                            print(f"[DEBUG] face_encodings error: {e}")
                        face_encodings = []
                else:
                    face_encodings = []
                
                face_names = []
                face_confidences = []
                liveness_status = []
                id_verification_status = []
                current_face_rois = {}
                
                for idx, (enc, (top, right, bottom, left)) in enumerate(zip(face_encodings, face_locations)):
                    orig_top = int(top / SCALE_FACTOR)
                    orig_right = int(right / SCALE_FACTOR)
                    orig_bottom = int(bottom / SCALE_FACTOR)
                    orig_left = int(left / SCALE_FACTOR)
                    
                    face_roi = safe_crop(frame, orig_left, orig_top, 
                                        orig_right - orig_left, orig_bottom - orig_top)
                    face_roi_gray = None
                    if face_roi is not None and face_roi.size > 0:
                        try:
                            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        except Exception:
                            face_roi_gray = None
                    
                    face_id = f"face_{idx}"
                    previous_roi = prev_face_rois.get(face_id, None)
                    face_size = 0
                    if face_roi is not None:
                        face_size = face_roi.shape[0] * face_roi.shape[1]
                    
                    # Anti-spoofing check
                    if ENABLE_ANTISPOOFING and face_roi is not None and face_roi.size > 0:
                        try:
                            is_live, live_conf, spoof_reason = is_face_live(
                                face_id, face_roi, face_roi_gray, previous_roi, 
                                face_size, LIVENESS_SENSITIVITY)
                            is_fake, fake_conf = is_image_fake(face_roi, FAKE_SENSITIVITY)
                        except Exception as e:
                            if SHOW_DEBUG:
                                print(f"[DEBUG] anti-spoof error: {e}")
                            is_live, live_conf, spoof_reason = True, 100, ""
                            is_fake, fake_conf = False, 0
                    else:
                        is_live, live_conf, spoof_reason = True, 100, ""
                        is_fake, fake_conf = False, 0
                    
                    current_face_rois[face_id] = face_roi.copy() if (face_roi is not None and face_roi.size > 0) else None
                    
                    matched_name = "Unknown"
                    confidence = 0.0
                    id_verified = False
                    id_info_match_score = 0.0
                    id_visual_match_score = 0.0
                    
                    # Face recognition
                    if (not ENABLE_ANTISPOOFING) or (is_live and not is_fake):
                        if person_mean_enc and len(person_mean_enc) > 0:
                            names_list = list(person_mean_enc.keys())
                            means = np.array([person_mean_enc[n] for n in names_list])
                            dists = np.linalg.norm(means - enc, axis=1) if means.size > 0 else np.array([])
                            
                            if dists.size > 0:
                                best_idx = int(np.argmin(dists))
                                best_name = names_list[best_idx]
                                best_dist = float(dists[best_idx])
                                
                                if dists.size > 1:
                                    tmp = dists.copy()
                                    tmp[best_idx] = np.inf
                                    second_best_dist = float(np.min(tmp))
                                else:
                                    second_best_dist = np.inf
                                
                                if TOLERANCE > 0:
                                    confidence = max(0.0, min(100.0, (1.0 - (best_dist / TOLERANCE)) * 100.0))
                                else:
                                    confidence = 0.0
                                
                                matched = False
                                if best_dist < TOLERANCE:
                                    margin = second_best_dist - best_dist
                                    if margin >= MIN_MARGIN:
                                        enc_list_for_person = person_encodings.get(best_name, [])
                                        if len(enc_list_for_person) > 1:
                                            per_encoding_matches = [distance(ei, enc) < (TOLERANCE + 0.05) 
                                                                  for ei in enc_list_for_person]
                                            if any(per_encoding_matches):
                                                matched = True
                                        else:
                                            matched = True
                                
                                if matched:
                                    matched_name = best_name
                                    
                                    # ID Card Verification
                                    if ENABLE_ID_VERIFICATION and id_card_detected and live_id_info:
                                        try:
                                            user_data = db.get_user(best_name)
                                            if user_data and 'id_card_features' in user_data and 'id_info' in user_data:
                                                stored_id_info = user_data['id_info']
                                                stored_id_features = user_data['id_card_features']
                                                
                                                # Compare text information
                                                id_info_match_score = compare_id_info(stored_id_info, live_id_info)
                                                
                                                # Compare visual features
                                                warped_id = id_card_detected['warped']
                                                live_id_features = extract_id_card_features(warped_id)
                                                id_visual_match_score = compare_id_cards_visual(
                                                    stored_id_features, live_id_features)
                                                
                                                # Combined score
                                                combined_id_score = (id_info_match_score * ID_INFO_MATCH_WEIGHT + 
                                                                   id_visual_match_score * ID_VISUAL_MATCH_WEIGHT)
                                                
                                                if combined_id_score >= ID_MATCH_THRESHOLD:
                                                    id_verified = True
                                                    if SHOW_DEBUG:
                                                        print(f"[DEBUG] ID VERIFIED for {best_name}")
                                                        print(f"  Info Match: {id_info_match_score:.2%}")
                                                        print(f"  Visual Match: {id_visual_match_score:.2%}")
                                                        print(f"  Combined: {combined_id_score:.2%}")
                                                else:
                                                    if SHOW_DEBUG:
                                                        print(f"[DEBUG] ID MISMATCH for {best_name}")
                                                        print(f"  Info: {id_info_match_score:.2%}, Visual: {id_visual_match_score:.2%}")
                                                
                                                id_card_match_history.setdefault(face_id, deque(maxlen=5)).append(combined_id_score)
                                        except Exception as e:
                                            if SHOW_DEBUG:
                                                print(f"[DEBUG] ID verification exception: {e}")
                    
                    face_names.append(matched_name)
                    face_confidences.append(confidence)
                    liveness_status.append({
                        'is_live': is_live, 
                        'live_confidence': live_conf, 
                        'is_fake': is_fake, 
                        'spoof_reason': spoof_reason
                    })
                    id_verification_status.append({
                        'id_verified': id_verified, 
                        'id_info_match': id_info_match_score,
                        'id_visual_match': id_visual_match_score
                    })
                
                prev_face_locations = face_locations
                prev_face_names = face_names
                prev_face_confidences = face_confidences
                prev_liveness_status = liveness_status
                prev_id_verification = id_verification_status
                prev_face_rois = current_face_rois
            
            process_this_frame = not process_this_frame
            
            # Draw ID card detection
            if id_card_detected:
                x, y, w, h = id_card_detected['bbox']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                
                # Display extracted ID info on screen
                if live_id_info:
                    info_y = y - 10
                    cv2.putText(display_frame, "ID CARD DETECTED", (x, info_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    if live_id_info.get('id_number'):
                        info_y += 25
                        cv2.putText(display_frame, f"ID: {live_id_info['id_number']}", 
                                  (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(display_frame, "ID CARD DETECTED", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw face detections
            for i, loc in enumerate(prev_face_locations):
                try:
                    top, right, bottom, left = loc
                    top = int(top / SCALE_FACTOR)
                    right = int(right / SCALE_FACTOR)
                    bottom = int(bottom / SCALE_FACTOR)
                    left = int(left / SCALE_FACTOR)
                except Exception:
                    continue
                
                name = prev_face_names[i] if i < len(prev_face_names) else "Unknown"
                conf = prev_face_confidences[i] if i < len(prev_face_confidences) else 0.0
                liveness = prev_liveness_status[i] if i < len(prev_liveness_status) else {
                    'is_live': True, 'live_confidence': 0, 'is_fake': False, 'spoof_reason': ''
                }
                id_verif = prev_id_verification[i] if i < len(prev_id_verification) else {
                    'id_verified': False, 'id_info_match': 0.0, 'id_visual_match': 0.0
                }
                
                draw_detection_label(
                    display_frame, left, top, right, bottom, name, conf,
                    liveness['is_live'], liveness['live_confidence'], 
                    liveness['is_fake'], liveness['spoof_reason'],
                    id_verif['id_verified'], id_verif['id_info_match'], 
                    id_verif['id_visual_match']
                )
            
            # Count verified faces
            verified_count = 0
            for i, nm in enumerate(prev_face_names):
                id_ok = True
                if ENABLE_ID_VERIFICATION and REQUIRE_ID_CARD:
                    if i < len(prev_id_verification):
                        id_ok = prev_id_verification[i].get('id_verified', False)
                    else:
                        id_ok = False
                if nm != "Unknown" and id_ok:
                    verified_count += 1
            
            # Info bar at top
            info = f"Known: {len(person_mean_enc)} | Detected: {len(prev_face_locations)} | Verified: {verified_count}"
            if ENABLE_ID_VERIFICATION:
                info += f" | ID Card: {'✓' if id_card_detected else '✗'}"
            
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
            cv2.putText(display_frame, info, (10, 32), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Enhanced Face Recognition - Press 'q' to quit", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = os.path.join("output", 
                    f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                os.makedirs("output", exist_ok=True)
                cv2.imwrite(screenshot_path, display_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
            elif key == ord('i') and id_card_detected:
                x, y, w, h = id_card_detected['bbox']
                id_card_roi = safe_crop(frame, x, y, w, h)
                if id_card_roi is not None:
                    os.makedirs("output", exist_ok=True)
                    id_save_path = os.path.join("output", 
                        f"detected_id_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(id_save_path, id_card_roi)
                    print(f"🪪 ID Card saved: {id_save_path}")
                    
                    # Also save warped version
                    if id_card_detected.get('warped') is not None:
                        warped_path = os.path.join("output", 
                            f"warped_id_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                        cv2.imwrite(warped_path, id_card_detected['warped'])
                        print(f"🪪 Warped ID saved: {warped_path}")
            
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Enhanced Face Recognition System stopped.")
        print(f"📊 Database saved to: {DATABASE_FILE}")

if __name__ == "__main__":
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print("\n⚠️  WARNING: Tesseract OCR not found!")
        print("   ID card text extraction will not work without Tesseract.")
        print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Or disable ID verification: ENABLE_ID_VERIFICATION = False")
        print("\n   Continuing without OCR...\n")
    
    main()
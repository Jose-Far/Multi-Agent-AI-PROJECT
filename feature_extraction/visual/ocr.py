import os
import logging
import re
from difflib import SequenceMatcher
from typing import Dict, Any

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Predefined dictionary of high-value brands targeted in phishing
TARGET_BRANDS = [
    "google", "amazon", "microsoft", "apple", "paypal", 
    "netflix", "facebook", "instagram", "chase", "adobe", "docusign"
]

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Applies image processing techniques to improve OCR accuracy on noisy or 
    low-contrast backgrounds (common in phishing sites).
    """
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter())
    img = img.point(lambda p: 255 if p > 128 else 0)
    return img

def normalize_and_match_brand(text: str) -> tuple:
    """
    Normalizes OCR text (removing punctuation/casing) and performs fuzzy 
    dictionary matching to correct OCR errors (e.g., 'Googie' -> 'google').
    
    Returns:
        tuple: (best_matched_brand, visual_brand_score)
    """
    normalized_text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    words = normalized_text.split()
    
    best_brand = "unknown"
    highest_score = 0.0

    for brand in TARGET_BRANDS:
        for word in words:
            # Fuzzy string matching (Levenshtein distance proxy)
            score = SequenceMatcher(None, brand, word).ratio()
            if score > highest_score:
                highest_score = score
                best_brand = brand

    # Threshold for fuzzy matching. 'googie' vs 'google' yields ~0.83 similarity.
    if highest_score >= 0.75:
        return best_brand, round(highest_score, 4)
    return "unknown", 0.0

def extract_ocr_text(screenshot_path: str) -> Dict[str, Any]:
    """
    Extracts text from a webpage screenshot using Tesseract OCR, applies 
    normalization, and computes a fuzzy-matched Visual Brand Score.
    """
    features = {
        "ocr_word_count": 0,
        "ocr_text_snippet": "",
        "contains_login_keywords": False,
        "contains_urgency_keywords": False,
        "raw_text_length": 0,
        "ocr_detected_brand": "unknown",
        "ocr_brand_confidence": 0.0
    }

    if not OCR_AVAILABLE:
        logger.warning("pytesseract or PIL not installed. Skipping OCR extraction.")
        return features

    if not screenshot_path or not os.path.exists(screenshot_path):
        logger.warning(f"Screenshot not found at {screenshot_path}. Skipping OCR.")
        return features

    try:
        img = Image.open(screenshot_path)
        processed_img = preprocess_image_for_ocr(img)

        # PSM 11 assumes sparse text with no specific order (ideal for UI layouts)
        custom_config = r'--oem 3 --psm 11'
        extracted_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        cleaned_text = re.sub(r'\s+', ' ', extracted_text).strip()
        if not cleaned_text:
            return features

        # 1. Base OCR Metrics
        words = cleaned_text.split()
        features["ocr_word_count"] = len(words)
        features["raw_text_length"] = len(cleaned_text)
        features["ocr_text_snippet"] = " ".join(words[:50])

        # 2. Phishing Heuristics (Keywords)
        login_keywords = {'login', 'sign in', 'password', 'username', 'email', 'authenticate', 'verify'}
        features["contains_login_keywords"] = any(kw in cleaned_text.lower() for kw in login_keywords)

        urgency_keywords = {'urgent', 'suspend', 'restricted', 'verify now', 'action required'}
        features["contains_urgency_keywords"] = any(kw in cleaned_text.lower() for kw in urgency_keywords)

        # 3. FIX: Text Normalization & Brand Dictionary Matching
        brand_match, brand_score = normalize_and_match_brand(cleaned_text)
        features["ocr_detected_brand"] = brand_match
        features["ocr_brand_confidence"] = brand_score

    except Exception as e:
        logger.error(f"Failed to execute OCR on {screenshot_path}: {str(e)}")

    return features
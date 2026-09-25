import os
import logging
from pathlib import Path

# Set up graceful degradation for heavy Machine Learning libraries
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Cache the model globally so it doesn't reload into memory on every single HTTP request
_LOGO_MODEL = None
_USING_FALLBACK = False

def load_logo_model(model_path: str = "models/brand_logo_yolov8.pt"):
    """
    Singleton pattern to load the YOLO model into memory once.
    Includes an auto-fallback to download a generic YOLOv8 model if the custom one is missing.
    """
    global _LOGO_MODEL, _USING_FALLBACK
    if not YOLO_AVAILABLE:
        return None
        
    if _LOGO_MODEL is None:
        # 1. Ensure the models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if os.path.exists(model_path):
            try:
                # Load the custom pre-trained YOLOv8 network
                _LOGO_MODEL = YOLO(model_path)
                logger.info(f"Successfully loaded custom YOLO logo detection model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize custom YOLO model: {e}")
        else:
            # 2. THE FIX: Auto-Download Fallback Model for Testing
            logger.warning(f"Custom weights not found at {model_path}. Auto-downloading fallback 'yolov8n.pt' for pipeline testing.")
            try:
                # Passing 'yolov8n.pt' triggers Ultralytics to automatically download the base model
                _LOGO_MODEL = YOLO("yolov8n.pt")
                _USING_FALLBACK = True
                logger.info("Successfully loaded fallback YOLOv8n model. Note: This will detect general objects, not specific brands.")
            except Exception as e:
                logger.error(f"Failed to load fallback YOLO model: {e}")
            
    return _LOGO_MODEL

def detect_logo(screenshot_path: str, model_path: str = "models/brand_logo_yolov8.pt") -> dict:
    """
    Detects known brand logos within the webpage screenshot using a Convolutional Neural Network (YOLOv8).
    
    Args:
        screenshot_path (str): Path to the 1080p Playwright screenshot.
        model_path (str): Path to the trained YOLO .pt weights file.
        
    Returns:
        dict: A dictionary containing detection status, brand name, confidence, and spatial coordinates.
    """
    features = {
        "logo_detected": False,
        "primary_brand_name": "unknown",
        "confidence": 0.0,
        "bounding_boxes": [],
        "total_logos_found": 0
    }
    
    if not YOLO_AVAILABLE:
        logger.warning("ultralytics (YOLO) not installed. Skipping deep learning logo detection.")
        return features
        
    if not screenshot_path or not os.path.exists(screenshot_path):
        logger.warning(f"Screenshot not found at {screenshot_path}. Skipping logo detection.")
        return features

    try:
        model = load_logo_model(model_path)
        if model is None:
            return features

        # Run inference on the screenshot. 
        results = model(screenshot_path, conf=0.45, verbose=False)
        
        detected_brands = []
        highest_confidence = 0.0
        best_brand = "unknown"
        
        # Parse YOLO Results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Extract classification confidence
                conf = float(box.conf[0])
                
                # Extract class index and map it to the name
                class_id = int(box.cls[0])
                brand_name = model.names[class_id]
                
                # If using the fallback model, prepend a tag so the AI Agent knows it's a generic object
                if _USING_FALLBACK:
                    brand_name = f"generic_object_{brand_name}"
                
                box_data = {
                    "brand": brand_name,
                    "confidence": round(conf, 4),
                    "coordinates": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                }
                detected_brands.append(box_data)
                
                # Track the most prominent detection on the page
                if conf > highest_confidence:
                    highest_confidence = conf
                    best_brand = brand_name

        # Populate final feature vector
        if detected_brands:
            features["logo_detected"] = True
            features["primary_brand_name"] = best_brand
            features["confidence"] = round(highest_confidence, 4)
            features["total_logos_found"] = len(detected_brands)
            features["bounding_boxes"] = detected_brands

    except Exception as e:
        logger.error(f"Failed to execute YOLO logo detection on {screenshot_path}: {str(e)}")

    return features

# ---------------------------------------------------------
# Standalone Local Testing Hook
# ---------------------------------------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     test_screenshot = "storage/screenshots/test_screenshot.png"
    
#     if os.path.exists(test_screenshot):
#         print("Testing CNN Extractor with Auto-Fallback...")
#         result = detect_logo(test_screenshot)
#         import json
#         print(json.dumps(result, indent=4))
#     else:
#         print(f"Please place a test image at {test_screenshot}")
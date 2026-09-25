import os
import logging

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

def analyze_layout(screenshot_path: str) -> dict:
    """
    Analyzes the structural geometry and UI layout of the webpage screenshot.
    
    Uses OpenCV to perform edge detection, morphological dilation, and polygon 
    approximation to heuristically detect UI elements like buttons and input fields.
    This helps the downstream AI Agent identify credential-harvesting login forms 
    even if the HTML DOM is heavily obfuscated.
    
    Args:
        screenshot_path (str): Path to the rendered webpage screenshot.
        
    Returns:
        dict: Geometrical layout features including button counts, input field 
              counts, form heuristics, and visual complexity scores.
    """
    features = {
        "layout_complexity_score": 0.0,
        "button_count": 0,
        "input_field_count": 0,
        "has_login_form_visual": False
    }
    
    if not CV2_AVAILABLE:
        logger.warning("OpenCV (cv2) or NumPy not installed. Skipping layout geometry extraction.")
        return features
        
    if not screenshot_path or not os.path.exists(screenshot_path):
        logger.warning(f"Screenshot not found at {screenshot_path}. Skipping layout analysis.")
        return features

    try:
        # 1. Load Image
        image = cv2.imread(screenshot_path)
        if image is None:
            logger.error(f"OpenCV failed to read the image at {screenshot_path}")
            return features
            
        # Optional: Standardize image width to 1280px to normalize area thresholds
        # This ensures our bounding box math works regardless of monitor resolution
        height, width = image.shape[:2]
        if width > 1280:
            scale = 1280.0 / width
            image = cv2.resize(image, (1280, int(height * scale)))

        # 2. Preprocessing (Grayscale & Blur)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to remove text noise/artifacts that create false edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Edge Detection (Canny)
        edges = cv2.Canny(blurred, 40, 120)
        
        # Calculate Layout Complexity (Percentage of screen covered by edges)
        # Highly complex sites have high edge density; empty sites have near 0.
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = cv2.countNonZero(edges)
        features["layout_complexity_score"] = round((edge_pixels / total_pixels) * 100, 4)

        # 4. Morphological Transformations
        # Dilate the edges to connect broken lines around buttons/forms
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 5. Contour Extraction
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        button_count = 0
        input_count = 0
        
        # 6. Geometrical Bounding Box Analysis
        for cnt in contours:
            # Approximate the contour to a polygon to smooth out rough edges
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # If the polygon has roughly 4 corners, it's likely a UI element
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                
                # Ignore tiny noise (e.g., icons) and massive containers (e.g., banners)
                if 800 < area < 60000 and h > 0:
                    aspect_ratio = w / float(h)
                    
                    # Heuristic A: Text Input Fields (Typically wide and short)
                    # Example: Username/Password boxes
                    if 3.0 <= aspect_ratio <= 12.0 and 20 <= h <= 70:
                        input_count += 1
                        
                    # Heuristic B: Buttons (Typically slightly wider than tall, but thicker than inputs)
                    # Example: "Sign In" or "Submit"
                    elif 1.0 <= aspect_ratio <= 4.5 and 25 <= h <= 90:
                        button_count += 1

        # Cap the maximums to prevent vector overflow for the AI agent
        features["button_count"] = min(button_count, 50)
        features["input_field_count"] = min(input_count, 50)
        
        # 7. Credential Harvesting Heuristic
        # A visual login form usually consists of at least 1 input field and 1 button
        if features["input_field_count"] >= 1 and features["button_count"] >= 1:
            features["has_login_form_visual"] = True
            
    except Exception as e:
        logger.error(f"Failed to process layout geometry for {screenshot_path}: {str(e)}")
        
    return features

# ---------------------------------------------------------
# Standalone Local Testing Hook
# ---------------------------------------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     # Point this to a real screenshot in your storage to test
#     test_screenshot = "storage/screenshots/test_screenshot.png"
    
#     print("Testing Visual Layout Extractor...")
#     result = analyze_layout(test_screenshot)
    
#     import json
#     print(json.dumps(result, indent=4))
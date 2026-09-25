import os
import logging

try:
    from PIL import Image
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

logger = logging.getLogger(__name__)

def extract_color_palette(screenshot_path: str, num_colors: int = 5) -> dict:
    """
    Extracts the top dominant RGB and HEX colors from a webpage screenshot.
    
    Upgraded to use Fast Octree Quantization which intelligently clusters 
    similar pixel shades together. This is highly resilient to image 
    compression artifacts and anti-aliasing, ensuring accurate brand 
    color detection for downstream AI agents.
    
    Args:
        screenshot_path (str): The local file path to the screenshot PNG.
        num_colors (int): The maximum number of dominant colors to extract.
        
    Returns:
        dict: A dictionary containing 'dominant_colors' (RGB arrays) and 
              'dominant_hex' (Hex strings).
    """
    # Default fallback skeleton
    features = {
        "dominant_colors": [],
        "dominant_hex": []
    }

    if not COLORS_AVAILABLE:
        logger.warning("PIL (Pillow) is not installed. Skipping color extraction.")
        return features
        
    if not screenshot_path or not os.path.exists(screenshot_path):
        logger.warning(f"Screenshot not found at path: {screenshot_path}. Skipping colors.")
        return features

    try:
        # Context manager ensures the file is safely closed after reading
        with Image.open(screenshot_path) as img:
            
            # 1. Handle Transparency (RGBA -> RGB)
            # If the image has an alpha channel, paste it over a pure white background.
            # This prevents transparent pixels from corrupting the color counts.
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                alpha = img.convert('RGBA').split()[-1]
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=alpha)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # 2. Downsample for Performance
            # Reduce resolution drastically to speed up memory processing, 
            # using LANCZOS to maintain color integrity during the shrink.
            img.thumbnail((150, 150), Image.Resampling.LANCZOS)

            # 3. Image Quantization (Clustering)
            # Method 2 is Fast Octree. It reduces the entire image down to exactly 'num_colors'
            # grouping similar pixels (e.g., light blue and dark blue) into a single dominant bucket.
            quantized = img.quantize(colors=num_colors, method=2)

            # 4. Extract Color Palette
            palette = quantized.getpalette()
            color_counts = quantized.getcolors()

            if not color_counts or not palette:
                return features

            # Sort the clustered colors by their frequency (pixel count)
            color_counts.sort(key=lambda x: x[0], reverse=True)

            dominant_rgb = []
            dominant_hex = []

            for count, index in color_counts[:num_colors]:
                # The palette is returned as a flat list: [R, G, B, R, G, B...]
                r = palette[index * 3]
                g = palette[index * 3 + 1]
                b = palette[index * 3 + 2]
                
                # Append standard RGB array (maintains compatibility with your existing JSON)
                dominant_rgb.append([r, g, b])
                
                # Append HEX string (Crucial for downstream LLM Brand Identity matching)
                hex_code = f"#{r:02x}{g:02x}{b:02x}".upper()
                dominant_hex.append(hex_code)

            features["dominant_colors"] = dominant_rgb
            features["dominant_hex"] = dominant_hex

    except Exception as e:
        logger.error(f"Failed to extract color palette from {screenshot_path}: {str(e)}")
        # Return empty lists to prevent pipeline crashes
        features["dominant_colors"] = []
        features["dominant_hex"] = []

    return features

# ---------------------------------------------------------
# Standalone Local Testing Hook
# ---------------------------------------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     # Create a dummy image for testing if none exists
#     test_path = "test_color_block.png"
#     if not os.path.exists(test_path) and COLORS_AVAILABLE:
#         img = Image.new('RGB', (100, 100), color = '#3B5998') # Facebook Blue
#         img.save(test_path)
        
#     print("Testing Color Extractor...")
#     result = extract_color_palette(test_path)
    
#     import json
#     print(json.dumps(result, indent=4))
    
#     # Cleanup test file
#     if os.path.exists(test_path):
#         os.remove(test_path)
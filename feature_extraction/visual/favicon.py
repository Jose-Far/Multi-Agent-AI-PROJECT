import logging
import hashlib
import requests
from typing import Dict, Any

# Configure logger
logger = logging.getLogger(__name__)

def analyze_favicon(favicon_url: str, screenshot_path: str = None) -> Dict[str, Any]:
    """
    Evaluates the presence, validity, and cryptographic signatures of a website's favicon.
    
    This extractor attempts to download the favicon to compute its MD5 and SHA-256 hashes.
    These hashes are critical for downstream Brand Impersonation AI Agents to detect 
    if a phishing site is perfectly cloning a legitimate brand's icon.
    
    Args:
        favicon_url (str): The absolute URL pointing to the website's favicon.
        screenshot_path (str, optional): Unused in this module, kept for pipeline signature compatibility.
        
    Returns:
        dict: A dictionary containing favicon presence, hashes, and size metrics.
    """
    features = {
        "favicon_present": False,
        "favicon_md5": None,
        "favicon_sha256": None,
        "favicon_size_bytes": 0,
        "favicon_download_success": False
    }
    
    # 1. Basic Validation
    if not favicon_url or not isinstance(favicon_url, str) or not favicon_url.startswith("http"):
        logger.warning(f"Invalid or missing favicon URL provided: {favicon_url}")
        return features
        
    features["favicon_present"] = True
    
    # 2. Cryptographic Hash Extraction
    try:
        # Set a short timeout. We don't want the whole pipeline hanging on a dead favicon link.
        # Use a generic User-Agent as some WAFs block default Python-requests.
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(favicon_url, headers=headers, timeout=5, stream=True)
        
        # Only process if we get a valid HTTP 200 OK response
        if response.status_code == 200:
            content = response.content
            
            # Prevent processing massive files if the server lied about it being an icon
            if len(content) > 5 * 1024 * 1024: # 5MB limit
                logger.warning(f"Favicon at {favicon_url} is too large (>5MB). Skipping hash.")
                return features
                
            features["favicon_size_bytes"] = len(content)
            
            # Compute MD5 Hash (Commonly used in legacy Threat Intel feeds)
            features["favicon_md5"] = hashlib.md5(content).hexdigest()
            
            # Compute SHA-256 Hash (Modern standard for Agentic matching)
            features["favicon_sha256"] = hashlib.sha256(content).hexdigest()
            
            features["favicon_download_success"] = True
            logger.info(f"Successfully hashed favicon from {favicon_url}")
            
        else:
            logger.warning(f"Failed to download favicon from {favicon_url}. Status Code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while attempting to fetch favicon from {favicon_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during favicon analysis: {str(e)}")
        
    return features

# ---------------------------------------------------------
# Standalone Local Testing Hook
# ---------------------------------------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     print("Testing Valid Google Favicon...")
#     valid_test = analyze_favicon("https://www.google.com/favicon.ico")
    
#     import json
#     print(json.dumps(valid_test, indent=4))
    
#     print("\nTesting Invalid/Fake Favicon...")
#     invalid_test = analyze_favicon("https://this-domain-does-not-exist.com/favicon.ico")
#     print(json.dumps(invalid_test, indent=4))
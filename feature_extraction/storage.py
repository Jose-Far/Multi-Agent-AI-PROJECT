import json
import os
import re
import logging
from datetime import datetime
from urllib.parse import urlparse

class FeatureStorage:
    """
    Handles the secure and organized persistence of extracted feature vectors.
    Saves outputs as structured JSON files for AI Agents to consume.
    """
    def __init__(self, base_dir: str = "data/features"):
        self.base_dir = base_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure the output directory structure exists
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create storage directory '{self.base_dir}': {str(e)}")

    def _sanitize_domain(self, url: str) -> str:
        """
        Extract and strictly sanitize the domain from a URL for clean file naming.
        Removes invalid path characters to prevent Directory Traversal vulnerabilities.
        """
        try:
            # Extract network location (e.g., www.example.com:8080)
            netloc = urlparse(url).netloc
            
            # Strip port number if it exists
            domain = netloc.split(':')[0]
            
            if not domain:
                return "unknown_domain"
            
            # Regex: Keep only alphanumeric characters, dots, and hyphens
            # Replaces anything else (like slashes, wildcards) with an underscore
            safe_domain = re.sub(r'[^a-zA-Z0-9.-]', '_', domain)
            
            return safe_domain
            
        except Exception as e:
            self.logger.warning(f"Domain sanitization failed for URL '{url}': {str(e)}")
            return "unknown_domain"

    def save_unified_features(self, url: str, features: dict) -> str:
        """
        Saves the complete unified feature vector to disk as a JSON file.
        
        Args:
            url (str): The target URL scanned.
            features (dict): The complete compiled feature dictionary.
            
        Returns:
            str: The absolute path to the saved JSON file, or an empty string if failed.
        """
        domain = self._sanitize_domain(url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_{domain}_{timestamp}.json"
        
        # Construct safe absolute path
        filepath = os.path.join(self.base_dir, filename)
        filepath = os.path.abspath(filepath)

        try:
            # Write features to disk with clean indentation for human readability
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=4, ensure_ascii=False)
            
            self.logger.debug(f"Features successfully written to: {filepath}")
            return filepath
            
        except IOError as e:
            self.logger.error(f"IOError while saving features to '{filepath}': {str(e)}")
            return ""
        except TypeError as e:
            self.logger.error(f"TypeError (Unserializable data) saving features for '{url}': {str(e)}")
            return ""
        except Exception as e:
            self.logger.error(f"Unexpected error saving features for '{url}': {str(e)}")
            return ""

# # --- Standalone Execution Test ---
# if __name__ == "__main__":
#     # Setup basic logging for testing
#     logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
#     storage = FeatureStorage()
    
#     mock_url = "https://www.spotify.com/us/login/"
#     mock_features = {
#         "metadata": {"target_url": mock_url},
#         "url_features": {"entropy": 4.38, "https": True},
#         "status": "success"
#     }
    
#     saved_path = storage.save_unified_features(mock_url, mock_features)
#     if saved_path:
#         print(f"\n[TEST SUCCESS] File created at: {saved_path}")
#     else:
#         print("\n[TEST FAILED] File was not created.")
import os
import json
import base64
import re
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Production-grade Storage Manager for the Multi-Agent Cybersecurity Analyst.
    
    Security & Architecture Enhancements:
    - Path Traversal Defenses: Strictly bounds all writes to the intended base directory.
    - Atomic File Writes: Writes data to temporary files first, then renames, preventing 
      corrupted artifacts if the process is terminated mid-write.
    - Safe Filename Lengths: Implements URL truncation and MD5 hashing for highly 
      obfuscated, massive phishing URLs to prevent OS-level MAX_PATH crashes.
    """

    def __init__(self, base_dir: str = "storage"):
        # Use pathlib for robust, cross-platform path resolution
        self.base_dir = (Path(__file__).parent / ".." / base_dir).resolve()
        
        # Define structured subdirectories
        self.dirs = {
            "screenshots": self.base_dir / "screenshots",
            "reports": self.base_dir / "reports",
            "html": self.base_dir / "html",
            "temp": self.base_dir / ".tmp"  # Hidden directory for atomic writes
        }
        
        self._init_directories()

    def _init_directories(self) -> None:
        """Securely provisions the required directory structures."""
        try:
            for name, dir_path in self.dirs.items():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Storage directory initialized: {dir_path}")
        except PermissionError as e:
            logger.critical(f"Permission denied creating storage directories: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error initializing storage: {e}")
            raise

    def _sanitize_filename(self, url: str, max_length: int = 150) -> str:
        """
        Strips schemes and normalizes a URL into a safe, OS-compliant filename.
        Includes defenses against excessively long URLs used in phishing evasion.
        """
        parsed = urlparse(url)
        # Combine netloc and path, replacing non-alphanumeric chars with underscores
        raw_name = f"{parsed.netloc}{parsed.path}"
        sanitized = re.sub(r'[^a-zA-Z0-9.\-]', '_', raw_name).strip('_')

        # Prevent OS MAX_PATH errors from massive obfuscated URLs
        if len(sanitized) > max_length:
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
            # Truncate and append the short hash to maintain uniqueness
            sanitized = f"{sanitized[:max_length - 9]}_{url_hash}"

        # Fallback for completely malformed inputs
        if not sanitized:
            sanitized = f"unknown_host_{uuid.uuid4().hex[:8]}"

        return sanitized

    def _ensure_safe_path(self, target_path: Path, expected_parent: Path) -> Path:
        """
        CRITICAL SECURITY CHECK: Prevents Path Traversal Attacks.
        Ensures the resolved final path strictly lives inside the designated storage folder.
        """
        resolved_path = target_path.resolve()
        if not str(resolved_path).startswith(str(expected_parent.resolve())):
            logger.error(f"Path Traversal Attempt Blocked: {resolved_path}")
            raise PermissionError("Target path escapes the allowed storage boundary.")
        return resolved_path

    def _atomic_write(self, data: Any, final_path: Path, mode: str = 'w') -> bool:
        """
        Writes data to a temporary file first, then atomically replaces the target.
        Prevents file corruption if the AI agent crashes or times out during disk I/O.
        """
        temp_name = f"tmp_{uuid.uuid4().hex}"
        temp_path = self.dirs["temp"] / temp_name
        
        try:
            # Write to temp file
            if 'b' in mode:
                with open(temp_path, mode) as f:
                    f.write(data)
            else:
                with open(temp_path, mode, encoding='utf-8') as f:
                    if isinstance(data, dict) or isinstance(data, list):
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    else:
                        f.write(data)
            
            # Atomically move temp file to final destination
            temp_path.replace(final_path)
            return True
            
        except Exception as e:
            logger.error(f"Atomic write failed for {final_path.name}: {e}")
            if temp_path.exists():
                temp_path.unlink()  # Cleanup on failure
            return False

    def save_scan_report(self, url: str, data: Dict[str, Any]) -> str:
        """
        Saves the aggregated JSON multi-vector telemetry report safely.
        
        Args:
            url (str): The target URL scanned.
            data (Dict[str, Any]): The aggregated JSON payload.
            
        Returns:
            str: The absolute filepath where the report was saved, or empty string on failure.
        """
        try:
            safe_name = self._sanitize_filename(url)
            timestamp = int(datetime.now(timezone.utc).timestamp())
            filename = f"report_{safe_name}_{timestamp}.json"
            
            target_path = self.dirs["reports"] / filename
            safe_target_path = self._ensure_safe_path(target_path, self.dirs["reports"])

            if self._atomic_write(data, safe_target_path, mode='w'):
                logger.info(f"Successfully saved JSON report: {filename}")
                return str(safe_target_path)
            return ""
            
        except Exception as e:
            logger.error(f"Failed to save JSON report for {url}: {str(e)}")
            return ""

    def save_screenshot(self, url: str, base64_data: str) -> str:
        """
        Decodes a base64 string and saves it securely as a PNG image.
        
        Args:
            url (str): The target URL scanned.
            base64_data (str): The base64 string returned by the Playwright/Selenium collector.
            
        Returns:
            str: The absolute filepath where the image was saved, or empty string on failure.
        """
        if not base64_data:
            logger.warning(f"No base64 data provided for screenshot storage: {url}")
            return ""

        try:
            safe_name = self._sanitize_filename(url)
            timestamp = int(datetime.now(timezone.utc).timestamp())
            filename = f"screenshot_{safe_name}_{timestamp}.png"
            
            target_path = self.dirs["screenshots"] / filename
            safe_target_path = self._ensure_safe_path(target_path, self.dirs["screenshots"])

            # Strip the data URI protocol header if it was accidentally passed down
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
                
            # Pad the base64 string if necessary to prevent decoding errors
            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)
                
            image_bytes = base64.b64decode(base64_data)
            
            if self._atomic_write(image_bytes, safe_target_path, mode='wb'):
                logger.info(f"Successfully saved Screenshot: {filename}")
                return str(safe_target_path)
            return ""
            
        except base64.binascii.Error as e:
            logger.error(f"Base64 decoding failed for {url}: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Failed to save screenshot for {url}: {str(e)}")
            return ""

    def save_html(self, url: str, html_content: str) -> str:
        """
        Saves the raw DOM structure extracted by the HTML collector.
        
        Args:
            url (str): The target URL scanned.
            html_content (str): The raw HTML string.
            
        Returns:
            str: The absolute filepath where the HTML was saved, or empty string on failure.
        """
        if not html_content:
            logger.warning(f"No HTML content provided to save for: {url}")
            return ""

        try:
            safe_name = self._sanitize_filename(url)
            timestamp = int(datetime.now(timezone.utc).timestamp())
            filename = f"dom_{safe_name}_{timestamp}.html"
            
            target_path = self.dirs["html"] / filename
            safe_target_path = self._ensure_safe_path(target_path, self.dirs["html"])

            if self._atomic_write(html_content, safe_target_path, mode='w'):
                logger.debug(f"Successfully saved raw DOM artifact: {filename}")
                return str(safe_target_path)
            return ""
            
        except Exception as e:
            logger.error(f"Failed to save HTML for {url}: {str(e)}")
            return ""
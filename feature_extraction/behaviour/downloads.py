import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DownloadBehaviorFeatureExtractor:
    """
    Advanced Drive-by Download and HTML Smuggling Feature Extractor.
    
    Identifies automatic drive-by downloads, suspicious file attachments, and 
    HTML Smuggling techniques (e.g., dynamically creating blobs or data URIs) 
    triggered via scripts, HTML5 attributes, or meta-refresh tags.
    """
    def __init__(self):
        # High-risk executables and scripts (Immediate severe threat)
        self.critical_extensions = [
            '.exe', '.scr', '.vbs', '.bat', '.cmd', '.ps1', 
            '.msi', '.msp', '.jar', '.sh', '.elf'
        ]
        
        # Archive and disk image formats frequently used to bypass initial scanning
        self.suspicious_extensions = [
            '.zip', '.rar', '.7z', '.iso', '.img', '.dmg', 
            '.docm', '.xlsm', '.wsf', '.apk', '.cab'
        ]

        # MIME types associated with HTML Smuggling (embedded base64 payloads)
        self.smuggling_mimes = [
            'application/octet-stream',
            'application/x-msdownload',
            'application/x-dosexec',
            'text/vbs',
            'application/java-archive',
            'application/vnd.microsoft.portable-executable'
        ]

        # JS functions chained together to force a silent download
        self.js_download_apis = [
            "navigator.mssaveblob", 
            "navigator.mssaveoropenblob",
            "url.createobjecturl", 
            "document.createelement('a')"
        ]

        # Pre-compiled Regex Patterns
        # Catches <a href="..." download="malware.exe">
        self.html5_download_regex = re.compile(r'<a\s+[^>]*\bdownload(?:\s*=\s*["\'][^"\']*["\'])?[^>]*>', re.IGNORECASE)
        
        # Catches <meta http-equiv="refresh" content="0; url=malware.exe">
        self.meta_refresh_regex = re.compile(r'<meta\s+[^>]*http-equiv\s*=\s*["\']refresh["\'][^>]*>', re.IGNORECASE)

    def extract(self, script_sources: List[str], page_html: str) -> Dict[str, Any]:
        """
        Scans DOM, links, and script strings for automated payload delivery mechanisms.
        
        Args:
            script_sources (List[str]): External script or iframe source URLs.
            page_html (str): Full HTML source content string.
            
        Returns:
            Dict[str, Any]: Download behavior feature vector with risk scoring.
        """
        features = {
            "has_automatic_download_trigger": False,
            "has_html_smuggling_signatures": False,
            "suspicious_download_extension_detected": False,
            "critical_payload_extension_detected": False,
            "detected_dangerous_extensions": [],
            "download_execution_risk_score": 0  # 0 to 100 based on severity
        }

        try:
            html_lower = (page_html or "").lower()
            found_extensions = set()
            risk_score = 0

            # 1. Inspect HTML source and script sources for raw extension presence
            combined_sources = " ".join(script_sources).lower() if script_sources else ""
            inspection_text = f"{html_lower} {combined_sources}"

            for ext in self.critical_extensions:
                if ext in inspection_text:
                    found_extensions.add(ext)
                    features["critical_payload_extension_detected"] = True
                    risk_score += 40

            for ext in self.suspicious_extensions:
                if ext in inspection_text:
                    found_extensions.add(ext)
                    features["suspicious_download_extension_detected"] = True
                    risk_score += 20

            if found_extensions:
                features["detected_dangerous_extensions"] = list(found_extensions)

            # 2. Detect HTML Smuggling & Client-Side File Generation
            # Attackers construct files in the browser memory using Base64/Blobs to bypass network filters
            if any(mime in html_lower for mime in self.smuggling_mimes) and "base64," in html_lower:
                features["has_html_smuggling_signatures"] = True
                risk_score += 60

            # 3. Detect Forced JavaScript Download Triggers
            # Check for API usage commonly used to silently dump the smuggled payload to disk
            if any(api in html_lower for api in self.js_download_apis):
                if "click()" in html_lower or "dispatch" in html_lower:
                    features["has_automatic_download_trigger"] = True
                    risk_score += 30

            # 4. Detect HTML5/DOM Forced Downloads
            if self.html5_download_regex.search(html_lower):
                # An HTML5 download attribute combined with a suspicious extension is highly malicious
                if features["detected_dangerous_extensions"]:
                    features["has_automatic_download_trigger"] = True
                    risk_score += 40

            # Detect rapid meta-refresh redirects pointing to file downloads
            if self.meta_refresh_regex.search(html_lower) and features["detected_dangerous_extensions"]:
                features["has_automatic_download_trigger"] = True
                risk_score += 50

            features["download_execution_risk_score"] = min(100, risk_score)

        except Exception as e:
            logger.error(f"Error during Download Behavior Feature Extraction: {str(e)}")

        return features
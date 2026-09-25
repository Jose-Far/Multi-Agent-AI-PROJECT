from bs4 import BeautifulSoup
import re
import logging

class JavaScriptFeatureExtractor:
    """
    Analyzes JavaScript usage in the DOM, tracking script volumes, 
    external script dependencies, and suspicious obfuscation indicators 
    (e.g., eval(), unescape(), disabled right-clicks).
    """
    def __init__(self, soup: BeautifulSoup, base_domain: str):
        self.soup = soup
        self.base_domain = base_domain.lower()
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_features(self) -> dict:
        """
        Scans script tags and inline handlers for malicious or evasive indicators.
        """
        if not self.soup:
            return {
                "script_count": 0,
                "external_scripts_count": 0,
                "suspicious_inline_scripts": 0,
                "has_evasion_tactics": False
            }

        scripts = self.soup.find_all('script')
        script_count = len(scripts)
        
        external_scripts_count = 0
        suspicious_inline_scripts = 0
        has_evasion_tactics = False

        # Keywords commonly used in obfuscation or anti-analysis scripts
        evasion_keywords = ['eval', 'unescape', 'escape', 'settimeout', 'document.write', 'oncontextmenu', 'onkeydown']

        for script in scripts:
            src = script.get('src', '').strip().lower()
            
            # Check if script is loaded from an external source
            if src:
                external_scripts_count += 1
                if src.startswith('http://') or src.startswith('https://'):
                    if self.base_domain not in src:
                        # External third-party script tracking
                        pass

            # Check inline script text content for obfuscation or evasion
            script_text = script.string or script.text or ""
            if script_text:
                script_lower = script_text.lower()
                
                # Check for suspicious evasion or obfuscation indicators
                if any(keyword in script_lower for keyword in evasion_keywords):
                    suspicious_inline_scripts += 1
                    has_evasion_tactics = True

        return {
            "script_count": script_count,
            "external_scripts_count": external_scripts_count,
            "suspicious_inline_scripts": suspicious_inline_scripts,
            "has_evasion_tactics": has_evasion_tactics
        }
from bs4 import BeautifulSoup
import re
import logging

class IframeFeatureExtractor:
    """
    Analyzes <iframe> tags within the HTML DOM.
    Specifically targets hidden iframes (often used for clickjacking, 
    drive-by downloads, or invisible credential harvesting) and tracks 
    external frame sources.
    """
    def __init__(self, soup: BeautifulSoup, base_domain: str):
        self.soup = soup
        self.base_domain = base_domain.lower()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _is_hidden(self, iframe) -> bool:
        """
        Evaluates if an iframe is intentionally hidden from the user 
        via HTML attributes or inline CSS.
        """
        # 1. Check HTML dimension attributes (0 or 1 pixel sizes are invisible to users)
        width = iframe.get('width', '').strip()
        height = iframe.get('height', '').strip()
        
        if width in ['0', '0px', '1', '1px', '0%'] or height in ['0', '0px', '1', '1px', '0%']:
            return True
            
        # 2. Check inline CSS styles for hidden properties
        style = iframe.get('style', '').lower()
        if style:
            # Regex patterns for common CSS cloaking techniques
            hidden_patterns = [
                r'display:\s*none',
                r'visibility:\s*hidden',
                r'opacity:\s*0',
                r'width:\s*0',
                r'height:\s*0'
            ]
            for pattern in hidden_patterns:
                if re.search(pattern, style):
                    return True
                    
        return False

    def extract_features(self) -> dict:
        """
        Scans all iframes to extract counts, visibility states, and origin boundaries.
        """
        if not self.soup:
            return {
                "iframe_count": 0,
                "hidden_iframe_count": 0,
                "external_iframe_count": 0,
                "has_hidden_iframes": False
            }

        iframes = self.soup.find_all('iframe')
        iframe_count = len(iframes)
        hidden_iframe_count = 0
        external_iframe_count = 0

        for iframe in iframes:
            # 1. Evaluate Visibility (Clickjacking/Evasion check)
            if self._is_hidden(iframe):
                hidden_iframe_count += 1
            
            # 2. Evaluate Source (Is it loading third-party content?)
            src = iframe.get('src', '').strip().lower()
            if src and (src.startswith('http://') or src.startswith('https://')):
                if self.base_domain not in src:
                    external_iframe_count += 1

        return {
            "iframe_count": iframe_count,
            "hidden_iframe_count": hidden_iframe_count,
            "external_iframe_count": external_iframe_count,
            "has_hidden_iframes": hidden_iframe_count > 0
        }
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging

# Robust imports for the sub-modules
try:
    from .forms import FormFeatureExtractor
    from .javascript import JavaScriptFeatureExtractor
    from .iframe import IframeFeatureExtractor
    from .links import LinkFeatureExtractor
    from .metadata import MetadataFeatureExtractor
except ImportError:
    from forms import FormFeatureExtractor
    from javascript import JavaScriptFeatureExtractor
    from iframe import IframeFeatureExtractor
    from links import LinkFeatureExtractor
    from metadata import MetadataFeatureExtractor

class HTMLExtractor:
    """
    Master Extractor for all HTML-based features. 
    Parses the DOM and orchestrates specialized sub-modules 
    (forms, javascript, iframe, links, metadata).
    """
    def __init__(self, html_content: str, target_url: str):
        self.html_content = html_content or ""
        self.target_url = target_url
        self.parsed_url = urlparse(target_url)
        self.base_domain = self.parsed_url.netloc.split(':')[0]
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if self.html_content:
            # Initialize BeautifulSoup parser once, efficiently sharing it with sub-modules
            self.soup = BeautifulSoup(self.html_content, 'html.parser')
        else:
            self.soup = None

    def _extract_general_dom_features(self) -> dict:
        """Extracts high-level document byte metrics."""
        return {
            "html_length_bytes": len(self.html_content)
        }

    def extract(self) -> dict:
        """
        Compiles all HTML and DOM features into a comprehensive dictionary.
        """
        combined_features = {}

        if not self.soup:
            self.logger.warning("No HTML content provided or parsing failed.")
            return {"error": "Empty or invalid HTML content"}

        # 1. General DOM Structural Metrics
        try:
            combined_features.update(self._extract_general_dom_features())
        except Exception as e:
            self.logger.error(f"Error extracting general DOM features: {str(e)}")

        # 2. Form & Input Intelligence
        try:
            form_extractor = FormFeatureExtractor(self.soup, self.base_domain)
            combined_features.update(form_extractor.extract_features())
        except Exception as e:
            self.logger.error(f"Error extracting form features: {str(e)}")

        # 3. JavaScript & Evasion Intelligence
        try:
            js_extractor = JavaScriptFeatureExtractor(self.soup, self.base_domain)
            combined_features.update(js_extractor.extract_features())
        except Exception as e:
            self.logger.error(f"Error extracting JavaScript features: {str(e)}")

        # 4. Iframe & Clickjacking Intelligence
        try:
            iframe_extractor = IframeFeatureExtractor(self.soup, self.base_domain)
            combined_features.update(iframe_extractor.extract_features())
        except Exception as e:
            self.logger.error(f"Error extracting iframe features: {str(e)}")

        # 5. Link & Anchor Intelligence
        try:
            link_extractor = LinkFeatureExtractor(self.soup, self.base_domain)
            combined_features.update(link_extractor.extract_features())
        except Exception as e:
            self.logger.error(f"Error extracting link features: {str(e)}")

        # 6. Metadata Intelligence
        try:
            metadata_extractor = MetadataFeatureExtractor(self.soup)
            combined_features.update(metadata_extractor.extract_features())
        except Exception as e:
            self.logger.error(f"Error extracting metadata features: {str(e)}")

        return combined_features
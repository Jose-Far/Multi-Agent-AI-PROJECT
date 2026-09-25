from bs4 import BeautifulSoup
import logging

class LinkFeatureExtractor:
    """
    Analyzes hyperlink (<a> tag) distribution across the DOM.
    Tracks internal vs. external anchor links, null/empty references, 
    and computes the external link ratio.
    """
    def __init__(self, soup: BeautifulSoup, base_domain: str):
        self.soup = soup
        self.base_domain = base_domain.lower()
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_features(self) -> dict:
        """
        Scans anchor tags to extract link density, external routing, and null reference metrics.
        """
        if not self.soup:
            return {
                "total_links": 0,
                "external_links_count": 0,
                "null_links_count": 0,
                "external_link_ratio": 0.0
            }

        links = self.soup.find_all('a', href=True)
        total_links = len(links)
        
        external_links_count = 0
        null_links_count = 0

        # Common empty/dummy link patterns used in phishing templates
        null_patterns = ['#', 'javascript:void(0)', 'javascript:;', '']

        for link in links:
            href = link['href'].strip().lower()
            
            # Check for null or empty links
            if not href or href in null_patterns:
                null_links_count += 1
                continue

            # Check if link points to an external domain
            if href.startswith('http://') or href.startswith('https://'):
                # It's an absolute URL; check if it belongs to the base domain
                if self.base_domain not in href:
                    external_links_count += 1
            elif href.startswith('//'):
                # Protocol-relative URL check (e.g., //external-cdn.com/script.js)
                if self.base_domain not in href:
                    external_links_count += 1

        # Calculate the external link ratio (safeguarded against division by zero)
        external_link_ratio = round(external_links_count / total_links, 4) if total_links > 0 else 0.0

        return {
            "total_links": total_links,
            "external_links_count": external_links_count,
            "null_links_count": null_links_count,
            "external_link_ratio": external_link_ratio
        }
from bs4 import BeautifulSoup
import logging

class MetadataFeatureExtractor:
    """
    Extracts high-level document metadata from the <head> section of the DOM.
    Captures the page title, favicon links, and standard meta descriptions 
    frequently used for brand impersonation detection.
    """
    def __init__(self, soup: BeautifulSoup):
        self.soup = soup
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_features(self) -> dict:
        """
        Scans the <head> block to pull out titles, meta descriptions, and favicons.
        """
        if not self.soup:
            return {
                "page_title": "",
                "has_favicon": False,
                "meta_description_length": 0
            }

        # Extract Page Title
        title_tag = self.soup.find('title')
        page_title = title_tag.text.strip() if title_tag else ""

        # Extract Favicon Presence
        # Favicons can be declared in several ways (icon, shortcut icon, etc.)
        favicon_link = self.soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        has_favicon = bool(favicon_link)

        # Extract Meta Description Length
        meta_desc_tag = self.soup.find('meta', attrs={'name': 'description'})
        meta_description_length = len(meta_desc_tag['content'].strip()) if meta_desc_tag and meta_desc_tag.get('content') else 0

        return {
            "page_title": page_title,
            "has_favicon": has_favicon,
            "meta_description_length": meta_description_length
        }
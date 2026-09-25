import re
from urllib.parse import urlparse

class StatisticalFeatureExtractor:
    """
    Extracts statistical ratios and sequential character patterns from the URL.
    These features are highly effective at identifying Domain Generation Algorithms (DGAs)
    and subdomain nesting commonly used in phishing campaigns.
    """
    def __init__(self, url: str):
        self.url = url.strip()
        self.parsed_url = urlparse(self.url)
        self.host = self.parsed_url.netloc.split(':')[0]

    def _get_subdomain_count(self) -> int:
        """
        Estimates the number of subdomains.
        A typical domain (example.com) has 2 parts. 
        Nested domains (secure.login.example.com) have > 2 parts.
        """
        if not self.host or re.match(r"^\d{1,3}(\.\d{1,3}){3}$", self.host):
            # If it's an IP address or empty, subdomain count is irrelevant
            return 0
            
        parts = self.host.split('.')
        # Subtracting 2 to account for the Root Domain + TLD
        return max(0, len(parts) - 2)

    def _get_vowel_consonant_ratio(self, text: str) -> float:
        """
        Calculates the ratio of vowels to consonants in the string.
        DGA domains often have highly unnatural ratios compared to human language.
        """
        text = text.lower()
        vowels = sum(1 for c in text if c in 'aeiou')
        consonants = sum(1 for c in text if c in 'bcdfghjklmnpqrstvwxyz')
        
        if consonants == 0:
            return round(float(vowels), 4)
        return round(vowels / consonants, 4)

    def _longest_consecutive_digits(self, text: str) -> int:
        """
        Finds the longest uninterrupted sequence of numbers in the URL.
        """
        matches = re.findall(r'\d+', text)
        if not matches:
            return 0
        return max(len(m) for m in matches)
        
    def _longest_consecutive_consonants(self, text: str) -> int:
        """
        Finds the longest uninterrupted sequence of consonants in the domain.
        A long string of consonants (e.g., 'zxcvbnm') strongly indicates a DGA or obfuscation.
        """
        matches = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]+', text)
        if not matches:
            return 0
        return max(len(m) for m in matches)

    def extract_features(self) -> dict:
        """
        Compiles the statistical properties into a dictionary.
        """
        features = {
            "subdomain_count": self._get_subdomain_count(),
            "vowel_consonant_ratio": self._get_vowel_consonant_ratio(self.host),
            "max_consecutive_digits": self._longest_consecutive_digits(self.url),
            "max_consecutive_consonants": self._longest_consecutive_consonants(self.host)
        }
        
        return features
import math
from urllib.parse import urlparse

class EntropyFeatureExtractor:
    """
    Analyzes the mathematical randomness (Shannon Entropy) of a URL and its components.
    High entropy indicates highly randomized strings, often associated with 
    Domain Generation Algorithms (DGAs), obfuscation, or encoded malware payloads.
    """
    def __init__(self, url: str):
        self.url = url.strip()
        self.parsed_url = urlparse(self.url)

    def _calculate_shannon_entropy(self, text: str) -> float:
        """
        Computes the Shannon entropy of a given string.
        Returns 0.0 if the string is empty.
        """
        if not text:
            return 0.0
            
        entropy = 0.0
        length = len(text)
        
        # Calculate the probability of each unique character and apply the formula
        for x in set(text):
            p_x = float(text.count(x)) / length
            entropy -= p_x * math.log2(p_x)
            
        return round(entropy, 4)

    def extract_features(self) -> dict:
        """
        Calculates entropy for the full URL, domain, path, and query parameters.
        Returns a dictionary of these statistical features.
        """
        # Isolate the domain (host) without the port number
        host = self.parsed_url.netloc.split(':')[0]
        path = self.parsed_url.path
        query = self.parsed_url.query

        features = {
            "url_entropy": self._calculate_shannon_entropy(self.url),
            "domain_entropy": self._calculate_shannon_entropy(host),
            "path_entropy": self._calculate_shannon_entropy(path),
            "query_entropy": self._calculate_shannon_entropy(query)
        }
        
        return features
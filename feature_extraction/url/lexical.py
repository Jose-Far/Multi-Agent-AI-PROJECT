import re
from urllib.parse import urlparse

class LexicalFeatureExtractor:
    """
    Analyzes the lexical (string-based) properties of a URL.
    Extracts lengths, character distributions, and structural anomalies 
    commonly found in phishing links.
    """
    def __init__(self, url: str):
        self.url = url.strip()
        self.parsed_url = urlparse(self.url)
        
        # Expanded list of target words often used in social engineering attacks
        self.suspicious_keywords = [
            'login', 'verify', 'secure', 'update', 'account', 
            'banking', 'confirm', 'free', 'wallet', 'service',
            'support', 'signin', 'auth', 'billing', 'admin'
        ]

    def _contains_ip(self, host: str) -> bool:
        """
        Check if the domain resolves to a raw IPv4 address, 
        a common technique used to hide malicious domains.
        """
        ip_pattern = re.compile(
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        )
        return bool(ip_pattern.match(host))

    def _get_tld(self, host: str) -> str:
        """
        Extract the Top-Level Domain (TLD) from the host.
        Returns 'none' if the host is an IP address.
        """
        if self._contains_ip(host):
            return "none"
            
        parts = host.split('.')
        if len(parts) > 1:
            return f".{parts[-1].lower()}"
        return "none"

    def _count_special_chars(self, text: str) -> int:
        """
        Count occurrences of special characters commonly abused in deeply 
        nested phishing query strings. Matches @, %, =, _, ?, &, -
        """
        special_chars = r"[@%=_?&\-]"
        return len(re.findall(special_chars, text))

    def _count_suspicious_words(self, text: str) -> int:
        """
        Count how many distinct social engineering keywords are in the URL.
        """
        text_lower = text.lower()
        count = sum(1 for word in self.suspicious_keywords if word in text_lower)
        return count

    def extract_features(self) -> dict:
        """
        Compiles all lexical features into a structured dictionary.
        """
        # Strip port numbers from the netloc if they exist (e.g., example.com:8080)
        host = self.parsed_url.netloc.split(':')[0]
        path = self.parsed_url.path

        features = {
            "url_length": len(self.url),
            "domain_length": len(host),
            "path_length": len(path),
            "num_dots": self.url.count('.'),
            "num_hyphens": self.url.count('-'),
            "num_digits": sum(c.isdigit() for c in self.url),
            "special_chars_count": self._count_special_chars(self.url),
            "num_parameters": len(self.parsed_url.query.split('&')) if self.parsed_url.query else 0,
            "is_https": True if self.parsed_url.scheme == 'https' else False,
            "contains_ip": self._contains_ip(host),
            "tld_type": self._get_tld(host),
            "suspicious_word_count": self._count_suspicious_words(self.url),
            "has_suspicious_words": self._count_suspicious_words(self.url) > 0
        }
        
        return features

# # --- Quick Test Block ---
# if __name__ == "__main__":
#     import json
    
#     test_urls = [
#         "https://www.github.com/login",
#         "http://192.168.1.100/secure-login-account/update.php?user=test&verify=true"
#     ]
    
#     print("=== Lexical Feature Extraction Test ===\n")
#     for test_url in test_urls:
#         print(f"Target URL: {test_url}")
#         extractor = LexicalFeatureExtractor(test_url)
#         print(json.dumps(extractor.extract_features(), indent=4))
#         print("-" * 40)
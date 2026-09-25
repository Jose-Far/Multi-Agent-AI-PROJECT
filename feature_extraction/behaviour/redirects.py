import logging
import re
from typing import Dict, Any, List
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class RedirectFeatureExtractor:
    """
    Advanced Behavioral Redirect Extractor.
    
    Analyzes navigation dynamics and redirect chains to identify evasive routing, 
    open loops, multi-hop obfuscation, and Open Redirect abuse (CWE-601).
    """
    def __init__(self):
        # Thresholds based on typical benign web traffic profiles
        self.max_normal_redirects = 3
        
        # Common query parameters abused in Open Redirect attacks (e.g., legit.com/login?next=bad.com)
        self.open_redirect_params = {'url', 'next', 'redirect', 'goto', 'out', 'return', 'dest', 'target'}
        
        # Regex to strip port numbers from netloc (e.g., example.com:443 -> example.com)
        self.port_stripper = re.compile(r':\d+$')

    def _get_base_domain(self, netloc: str) -> str:
        """
        Safely extracts a proxy for the base domain to accurately detect cross-domain jumps.
        (In a full production environment, consider replacing with the 'tldextract' library).
        """
        clean_netloc = self.port_stripper.sub('', netloc.lower())
        parts = clean_netloc.split('.')
        # Return at least the last two parts (e.g., 'sub.example.com' -> 'example.com')
        if len(parts) > 2:
            # Simple heuristic for ccTLDs like .co.uk
            if parts[-2] in ['co', 'com', 'org', 'net', 'gov', 'edu'] and len(parts[-1]) == 2:
                return '.'.join(parts[-3:])
            return '.'.join(parts[-2:])
        return clean_netloc

    def extract(self, redirect_history: List[Dict[str, Any]], final_url: str) -> Dict[str, Any]:
        """
        Extracts deep behavioral metrics from the redirect history array.
        
        Args:
            redirect_history (List[Dict]): List of redirect hops (e.g., [{'url': '...', 'status_code': 301, 'latency_sec': 0.2}]).
            final_url (str): The final landing URL after all redirects.
            
        Returns:
            Dict[str, Any]: Granular redirect analytics feature vector.
        """
        features = {
            "redirect_count": 0,
            "has_excessive_redirects": False,
            "has_cross_domain_redirect": False,
            "has_open_redirect_pattern": False,
            "is_redirect_loop_detected": False,
            "has_suspicious_status_codes": False,
            "unique_hop_domains_count": 0,
            "average_hop_latency_sec": 0.0
        }

        # If no redirects occurred, return the safe baseline
        if not redirect_history or not isinstance(redirect_history, list):
            return features

        try:
            features["redirect_count"] = len(redirect_history)
            features["has_excessive_redirects"] = len(redirect_history) > self.max_normal_redirects

            hop_base_domains = []
            raw_hop_urls = []
            total_latency = 0.0
            latency_count = 0
            has_suspicious_code = False

            # 1. Analyze Initial URL for Open Redirect Patterns
            initial_url = redirect_history[0].get("url", "")
            if initial_url:
                parsed_initial = urlparse(initial_url)
                query_params = parse_qs(parsed_initial.query)
                # Check if any common open redirect parameter is present in the first requested URL
                if any(param in query_params for param in self.open_redirect_params):
                    features["has_open_redirect_pattern"] = True

            # 2. Process the Redirect Chain
            for hop in redirect_history:
                hop_url = hop.get("url", "")
                status_code = hop.get("status_code")
                latency = hop.get("latency_sec")
                
                raw_hop_urls.append(hop_url)

                if hop_url:
                    parsed_hop = urlparse(hop_url)
                    base_domain = self._get_base_domain(parsed_hop.netloc)
                    if base_domain:
                        hop_base_domains.append(base_domain)

                # Track non-standard HTTP redirects (e.g., 307 Temporary Redirects are heavily abused)
                if status_code in [307, 308]:
                    has_suspicious_code = True

                # Aggregate latency if provided by the collector
                if isinstance(latency, (int, float)):
                    total_latency += latency
                    latency_count += 1

            # 3. Compute Domain & Evasion Analytics
            unique_domains = list(set(hop_base_domains))
            features["unique_hop_domains_count"] = len(unique_domains)
            features["has_suspicious_status_codes"] = has_suspicious_code

            # Calculate Average Hop Latency (Phishers using compromised infra often have slow hops)
            if latency_count > 0:
                features["average_hop_latency_sec"] = round(total_latency / latency_count, 4)

            # 4. Cross-Domain Verification
            # Phishing attacks usually bounce from a compromised legit site -> traffic broker -> actual malicious payload
            final_base_domain = self._get_base_domain(urlparse(final_url).netloc)
            
            for domain in unique_domains:
                if domain != final_base_domain:
                    features["has_cross_domain_redirect"] = True
                    break

            # 5. Redirect Loop / Sandbox Evasion Detection
            # If the chain visits A -> B -> A -> C, it's a loop designed to exhaust automated malware scanners
            if len(hop_base_domains) != len(unique_domains) and len(hop_base_domains) > 2:
                # We have duplicate domains. Are they sequential (A->A) or looping (A->B->A)?
                for i in range(len(hop_base_domains) - 2):
                    if hop_base_domains[i] in hop_base_domains[i+2:]:
                        features["is_redirect_loop_detected"] = True
                        break

        except Exception as e:
            logger.error(f"Error during Redirect Feature Extraction: {str(e)}")

        return features
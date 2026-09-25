import os
import base64
import logging
import requests
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# THREAT INTELLIGENCE CACHING (DAY 27)
# ----------------------------------------------------------------------------
THREAT_CACHE = {}

try:
    from performance.config import PerformanceConfig
    CACHE_TTL = PerformanceConfig.THREAT_CACHE_TTL_SEC
except ImportError:
    CACHE_TTL = 3600

class ThreatCollector:
    """
    Advanced Threat Intelligence Gatherer using the VirusTotal v3 API.
    
    Security & Architecture Enhancements:
    - Adaptive Rate Limiting: Handles HTTP 429 (Too Many Requests) and HTTP 204 (Quota Exceeded) 
      with exponential backoff, preventing pipeline crashes during bulk analysis.
    - Deep Context Extraction: Pulls specific engine verdicts and categorization tags 
      (e.g., "phishing") to feed the downstream Explainable AI (SHAP/LIME) engine.
    - Timezone Normalized: Converts raw Unix epoch timestamps into standardized ISO 8601 UTC formats.
    """
    def __init__(self, timeout: int = 15, max_retries: int = 3, backoff_factor: float = 2.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Secure credential loading
        self.api_key = os.getenv("VIRUSTOTAL_API_KEY")
        self.base_url = "https://www.virustotal.com/api/v3/urls"

    def collect(self, target_url: str) -> Dict[str, Any]:
        current_time = time.time()
        if target_url in THREAT_CACHE:
            cached_data, timestamp = THREAT_CACHE[target_url]
            if current_time - timestamp < CACHE_TTL:
                import logging
                logging.getLogger(__name__).info(f"Threat Intel Cache HIT for {target_url}")
                return cached_data

        result = self._collect_internal(target_url)
        
        if result.get("success"):
            THREAT_CACHE[target_url] = (result, current_time)
            
        return result

    def _collect_internal(self, target_url: str) -> Dict[str, Any]:
        """
        Cross-references the target URL against VirusTotal threat feeds.
        
        Args:
            target_url (str): The URL to analyze.
            
        Returns:
            Dict[str, Any]: Standardized threat telemetry payload.
        """
        if not self.api_key or self.api_key.strip() == "" or self.api_key == "your_actual_api_key_here":
            logger.error("VirusTotal API key is missing or unconfigured.")
            return self._build_error_payload("Missing or unconfigured VIRUSTOTAL_API_KEY in environment variables.")

        # VirusTotal v3 requires Base64 URL-safe encoding without the padding '='
        try:
            url_bytes = target_url.encode("utf-8")
            url_id = base64.urlsafe_b64encode(url_bytes).decode("utf-8").replace("=", "")
        except Exception as e:
            logger.error(f"Failed to encode URL for VirusTotal: {e}")
            return self._build_error_payload(f"URL Encoding Error: {str(e)}")

        headers = {
            "accept": "application/json",
            "x-apikey": self.api_key
        }

        # Implement Retry and Backoff Logic for Rate Limits
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}/{url_id}",
                    headers=headers,
                    timeout=self.timeout
                )

                # Handle standard success
                if response.status_code == 200:
                    return self._parse_vt_response(response.json())
                
                # Handle Zero-Day / Unscanned URLs
                elif response.status_code == 404:
                    logger.info(f"URL not found in VirusTotal (Potential Zero-Day): {target_url}")
                    return self._build_zero_day_payload()
                
                # Handle Unauthorized
                elif response.status_code == 401:
                    logger.error("VirusTotal API key rejected (401 Unauthorized).")
                    return self._build_error_payload("401 Unauthorized: Invalid or expired VirusTotal API key.")
                
                # Handle Rate Limiting (429 or 204)
                elif response.status_code in [429, 204]:
                    if attempt < self.max_retries - 1:
                        sleep_time = self.backoff_factor ** attempt
                        logger.warning(f"VirusTotal rate limit hit. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        logger.error("VirusTotal rate limit exhausted after max retries.")
                        return self._build_error_payload("Rate limit exceeded. Try again later.")
                
                # Handle other unexpected errors
                else:
                    return self._build_error_payload(f"VirusTotal API returned HTTP {response.status_code}")

            except requests.exceptions.Timeout:
                logger.warning(f"VirusTotal API request timed out for {target_url} (Attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    return self._build_error_payload("VirusTotal API request timed out repeatedly.")
                time.sleep(self.backoff_factor)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error communicating with VirusTotal: {e}")
                return self._build_error_payload(f"VirusTotal connection error: {str(e)}")
            except Exception as e:
                logger.exception(f"Unexpected error in ThreatCollector: {e}")
                return self._build_error_payload(f"Unexpected processing error: {str(e)}")

        return self._build_error_payload("Max retries exceeded without a successful response.")

    def _parse_vt_response(self, result_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep parses the VT v3 JSON payload to extract high-value features for AI agents.
        """
        attributes = result_json.get("data", {}).get("attributes", {})
        last_analysis_stats = attributes.get("last_analysis_stats", {})
        last_analysis_results = attributes.get("last_analysis_results", {})

        # Extract baseline statistics
        malicious_count = last_analysis_stats.get("malicious", 0)
        suspicious_count = last_analysis_stats.get("suspicious", 0)
        harmless_count = last_analysis_stats.get("harmless", 0)
        undetected_count = last_analysis_stats.get("undetected", 0)

        is_malicious = (malicious_count + suspicious_count) > 0

        # Extract specific engines that flagged the URL (Critical for Explainable AI / SOC Analysts)
        flagged_by: List[str] = []
        for engine, data in last_analysis_results.items():
            category = data.get("category", "")
            if category in ["malicious", "suspicious"]:
                result_name = data.get("result", "unknown")
                flagged_by.append(f"{engine} ({result_name})")

        # Extract categories assigned by various vendors (e.g., "phishing", "malware", "spam")
        categories_dict = attributes.get("categories", {})
        unique_categories = list(set(categories_dict.values()))

        # Normalize the last analysis date (VT returns Unix Epoch integer)
        raw_date = attributes.get("last_analysis_date")
        normalized_date = None
        if raw_date:
            try:
                normalized_date = datetime.fromtimestamp(raw_date, tz=timezone.utc).isoformat()
            except (ValueError, TypeError):
                normalized_date = None

        return {
            "success": True,
            "error": None,
            "data": {
                "is_listed": True,
                "threat_feed": "VirusTotal (v3)",
                "blacklisted": is_malicious,
                "reputation_score": attributes.get("reputation", 0),
                "stats": {
                    "malicious": malicious_count,
                    "suspicious": suspicious_count,
                    "harmless": harmless_count,
                    "undetected": undetected_count
                },
                "flagged_by": flagged_by,
                "categories": unique_categories,
                "last_analysis_date": normalized_date,
                "first_submission_date": self._normalize_epoch(attributes.get("first_submission_date"))
            }
        }

    def _normalize_epoch(self, epoch_time: Any) -> str:
        """Helper to safely convert optional epoch timestamps to ISO strings."""
        if isinstance(epoch_time, (int, float)):
            try:
                return datetime.fromtimestamp(epoch_time, tz=timezone.utc).isoformat()
            except Exception:
                pass
        return ""

    def _build_zero_day_payload(self) -> Dict[str, Any]:
        """Payload for URLs that have never been seen by VT (High Anomaly Indicator)."""
        return {
            "success": True,
            "error": None,
            "data": {
                "is_listed": False,
                "threat_feed": "VirusTotal (v3)",
                "blacklisted": False,
                "reputation_score": 0,
                "stats": {
                    "malicious": 0,
                    "suspicious": 0,
                    "harmless": 0,
                    "undetected": 0
                },
                "flagged_by": [],
                "categories": [],
                "last_analysis_date": None,
                "first_submission_date": None,
                "message": "URL not found in VirusTotal database (potential new/zero-day infrastructure)."
            }
        }

    def _build_error_payload(self, message: str) -> Dict[str, Any]:
        """Standardized failure response."""
        return {
            "success": False,
            "error": message,
            "data": {
                "is_listed": False,
                "threat_feed": "VirusTotal (v3)",
                "blacklisted": None,
                "stats": {},
                "categories": [],
                "flagged_by": []
            }
        }
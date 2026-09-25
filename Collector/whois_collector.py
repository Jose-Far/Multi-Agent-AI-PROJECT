import logging
import tldextract
import whois
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

class WHOISCollector:
    """
    Advanced WHOIS Telemetry Collector for Phishing Detection.
    
    Extracts domain registration provenance, calculates lifecycle metrics (Domain Age, 
    Days Until Expiry), and sanitizes notoriously inconsistent WHOIS server outputs 
    to feed structured data to the Machine Learning Feature Extraction Layer.
    """
    def __init__(self):
        # Initialize tldextract with a cached public suffix list to ensure 
        # accurate parsing of complex ccTLDs (e.g., 'co.uk', 'com.au')
        self.extract = tldextract.TLDExtract(cache_dir='.tld_set_cache')

    def collect(self, target_url: str) -> Dict[str, Any]:
        """
        Orchestrates the domain extraction and WHOIS lookup process.
        
        Args:
            target_url (str): The raw URL or hostname to analyze.
            
        Returns:
            Dict[str, Any]: Standardized telemetry payload containing WHOIS structural metrics.
        """
        try:
            domain = self._extract_apex_domain(target_url)
            if not domain:
                return self._build_error_payload(f"Could not extract a valid apex domain from: {target_url}")

            logger.info(f"WHOISCollector initiating query for domain: {domain}")
            return self._analyze_whois(domain)

        except Exception as e:
            logger.error(f"Unexpected error in WHOIS collection for {target_url}: {str(e)}")
            return self._build_error_payload(f"WHOIS collection failed: {str(e)}")

    def _extract_apex_domain(self, url: str) -> Optional[str]:
        """
        Extracts the root/apex domain using the Public Suffix List.
        Crucial because WHOIS lookups fail on subdomains (e.g., 'login.google.com' -> 'google.com').
        """
        extracted = self.extract(url)
        if not extracted.suffix or not extracted.domain:
            return None
        return f"{extracted.domain}.{extracted.suffix}".lower()

    def _normalize_date(self, date_obj: Union[datetime, str, list, None]) -> Optional[datetime]:
        """
        Aggressively sanitizes WHOIS date outputs. WHOIS servers are notoriously 
        inconsistent, frequently returning lists of dates, raw strings, or naive datetimes.
        """
        if not date_obj:
            return None
        
        # If the registrar returns multiple dates (e.g., creation dates across transfers), grab the first
        if isinstance(date_obj, list):
            date_obj = date_obj[0]

        # If it's a string, python-whois failed to parse it. We skip to prevent pipeline crashes.
        if isinstance(date_obj, str):
            logger.debug(f"Unparseable WHOIS date string encountered: {date_obj}")
            return None

        if isinstance(date_obj, datetime):
            # Enforce strict UTC timezone awareness for accurate mathematical calculations
            if date_obj.tzinfo is None:
                return date_obj.replace(tzinfo=timezone.utc)
            return date_obj.astimezone(timezone.utc)

        return None

    def _parse_list_attribute(self, attr: Union[str, list, None], lowercase: bool = False) -> List[str]:
        """
        Safely flattens WHOIS attributes (like Name Servers or Status codes) 
        that fluctuate randomly between strings and lists depending on the registrar.
        """
        if not attr:
            return []
        if isinstance(attr, str):
            return [attr.lower() if lowercase else attr]
        if isinstance(attr, list):
            return [str(item).lower() if lowercase else str(item) for item in attr if item]
        return []

    def _analyze_whois(self, domain: str) -> Dict[str, Any]:
        """
        Executes the WHOIS query, normalizes the payload, and calculates temporal features.
        """
        try:
            # Note: python-whois performs a blocking socket connection to Port 43
            w = whois.whois(domain)
            
            # 1. Normalize temporal data
            creation_date = self._normalize_date(w.creation_date)
            expiration_date = self._normalize_date(w.expiration_date)
            updated_date = self._normalize_date(w.updated_date)
            
            now_utc = datetime.now(timezone.utc)
            
            # 2. Calculate Lifecycle Metrics (High-Weight ML Features)
            domain_age_days = 0
            days_until_expiry = 0
            is_expired = False

            if creation_date:
                # Use max(0) to prevent negative age on microseconds timezone overlaps of brand new domains
                domain_age_days = max(0, (now_utc - creation_date).days)
                
            if expiration_date:
                days_until_expiry = (expiration_date - now_utc).days
                is_expired = days_until_expiry < 0

            # 3. Extract and sanitize Identity and Infrastructure parameters
            registrar = w.registrar[0] if isinstance(w.registrar, list) else w.registrar
            registrant_country = w.country[0] if isinstance(w.country, list) else w.country
            registrant_org = w.org[0] if isinstance(w.org, list) else w.org

            # 4. Compile the Data Payload
            whois_data = {
                "domain": domain,
                "registrar": str(registrar).strip() if registrar else "Unknown",
                "registrant_country": str(registrant_country).strip().upper() if registrant_country else "Unknown",
                "registrant_org": str(registrant_org).strip() if registrant_org else "Unknown",
                
                # Temporal ISO Strings
                "creation_date": creation_date.isoformat() if creation_date else None,
                "expiration_date": expiration_date.isoformat() if expiration_date else None,
                "updated_date": updated_date.isoformat() if updated_date else None,
                
                # Computed Threat Heuristics
                "domain_age_days": domain_age_days,
                "days_until_expiry": days_until_expiry,
                "is_expired": is_expired,
                
                # Infrastructure Context
                "name_servers": self._parse_list_attribute(w.name_servers, lowercase=True),
                "status_codes": self._parse_list_attribute(w.status)
            }

            # 🔴 PHISHING HEURISTIC TRIGGER: Null WHOIS Record (Domain not registered or completely cloaked)
            if not whois_data["creation_date"] and not whois_data["name_servers"]:
                return self._build_error_payload(f"WHOIS query returned empty or masked record for {domain}.")

            return {
                "success": True,
                "error": None,
                "data": whois_data
            }

        except whois.parser.PywhoisError as e:
            logger.warning(f"Domain '{domain}' not found in WHOIS databases (May be unregistered or protected): {e}")
            return self._build_error_payload(f"Domain not found or WHOIS blocked: {str(e)}")
        except Exception as e:
            logger.error(f"WHOIS socket/parsing failure for {domain}: {str(e)}")
            return self._build_error_payload(f"WHOIS lookup failed: {str(e)}")

    def _build_error_payload(self, message: str) -> Dict[str, Any]:
        """Constructs standardized error response structure."""
        return {
            "success": False,
            "error": message,
            "data": {
                "domain": None,
                "domain_age_days": 0,
                "is_expired": True
            }
        }
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class BlacklistFeatureExtractor:
    """
    Advanced Threat Intelligence Blacklist and Feed Correlation Extractor.
    
    Evaluates target domains and IPs against global threat intelligence feeds 
    (PhishTank, OpenPhish, Google Safe Browsing, VirusTotal indicators) to provide 
    weighted consensus scores, tier-based vendor attribution, and false-positive guards.
    """
    def __init__(self):
        # Tier 1: High-Authority Phishing & Fraud Intelligence Feeds
        self.high_authority_feeds = {
            "phishtank", "google safe browsing", "openphish", 
            "netcraft", "adminuslabs", "alphamountain.ai"
        }
        
        # High-Risk Threat Categories that demand immediate escalation
        self.critical_threat_categories = {
            "phishing and other frauds", "phishing and fraud", 
            "malware", "malicious", "fraud", "scam"
        }

    def extract(self, threat_analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts granular blacklist status, vendor weighting, and categorization metrics.
        
        Args:
            threat_analysis_data (Dict): Raw threat intelligence telemetry block.
            
        Returns:
            Dict[str, Any]: ML-ready blacklist feature vector.
        """
        features = {
            "is_blacklisted": False,
            "blacklist_vendors_count": 0,
            "high_authority_vendor_flagged": False,
            "matched_blacklist_categories": [],
            "flagging_vendors_list": [],
            "blacklist_risk_severity": "clean"  # clean, low, high, critical
        }

        if not threat_analysis_data or not isinstance(threat_analysis_data, dict):
            return features

        try:
            # 1. Base Extraction from Telemetry Payload
            # FIX: Only flag as blacklisted if the vendor explicitly marks it "blacklisted"
            # Do NOT use 'is_listed', as legitimate sites like Google are "listed" in databases for categorization.
            is_listed = bool(threat_analysis_data.get("blacklisted", False)) 
            
            categories = threat_analysis_data.get("categories", [])
            flagging_vendors = threat_analysis_data.get("flagged_by", [])

            # Normalize data types safely
            clean_categories = [str(cat).lower().strip() for cat in categories] if isinstance(categories, list) else [str(categories).lower().strip()]
            clean_vendors = [str(v).lower().strip() for v in flagging_vendors] if isinstance(flagging_vendors, list) else [str(flagging_vendors).lower().strip()]

            features["is_blacklisted"] = is_listed
            features["blacklist_vendors_count"] = len(clean_vendors)
            features["matched_blacklist_categories"] = clean_categories
            features["flagging_vendors_list"] = clean_vendors

            if not is_listed and len(clean_vendors) == 0:
                return features

            # 2. High-Authority Vendor Attribution
            # Check if specialized phishing/fraud feeds flagged the domain
            has_high_auth_flag = any(
                any(feed in vendor for feed in self.high_authority_feeds) 
                for vendor in clean_vendors
            )
            features["high_authority_vendor_flagged"] = has_high_auth_flag

            # 3. Threat Severity Tiering
            # Correlates category types and vendor volume to classify risk severity
            is_critical_category = any(
                any(crit in cat for crit in self.critical_threat_categories) 
                for cat in clean_categories
            )

            if is_listed or len(clean_vendors) >= 5 or (has_high_auth_flag and is_critical_category):
                features["blacklist_risk_severity"] = "critical"
            elif len(clean_vendors) >= 2 or has_high_auth_flag:
                features["blacklist_risk_severity"] = "high"
            elif len(clean_vendors) == 1:
                features["blacklist_risk_severity"] = "low"
            else:
                features["blacklist_risk_severity"] = "clean"

        except Exception as e:
            logger.error(f"Error during Blacklist Feature Extraction: {str(e)}")

        return features
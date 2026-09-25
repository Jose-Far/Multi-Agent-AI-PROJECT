import logging
from typing import Dict, Any

from .blacklist import BlacklistFeatureExtractor
from .malware import MalwareFeatureExtractor
from .reputation import ReputationFeatureExtractor

logger = logging.getLogger(__name__)

class ThreatFeatureExtractor:
    """
    Master Orchestrator for the Threat Intelligence Feature Extraction Layer.
    
    Synthesizes discrete OSINT feeds, malware family indicators, zero-day heuristics, 
    and multi-engine reputation ratios into a unified, highly-dimensional threat vector.
    Serves as the external validation boundary for the AI reasoning engine.
    """
    def __init__(self):
        self.blacklist_extractor = BlacklistFeatureExtractor()
        self.malware_extractor = MalwareFeatureExtractor()
        self.reputation_extractor = ReputationFeatureExtractor()

    def extract_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the comprehensive threat intelligence feature extraction pipeline.
        
        Args:
            raw_data (Dict): The aggregated scan payload containing 'threat_analysis'.
            
        Returns:
            Dict[str, Any]: Unified Threat Feature Vector.
        """
        if not raw_data or not isinstance(raw_data, dict):
            logger.warning("Threat Feature Extractor received empty telemetry. Returning safe baseline.")
            return self._get_empty_features()

        try:
            # Safely navigate to the threat_analysis data block
            threat_analysis = raw_data.get("threat_analysis", {})
            threat_data = threat_analysis.get("data", threat_analysis) if isinstance(threat_analysis, dict) else {}

            if not threat_data:
                logger.debug("No 'threat_analysis' data found in payload. Applying baseline.")
                return self._get_empty_features()

            # 1. Execute Sub-Modules with Fault Isolation
            blacklist_features = self._safe_extract(self.blacklist_extractor.extract, threat_data)
            malware_features = self._safe_extract(self.malware_extractor.extract, threat_data)
            reputation_features = self._safe_extract(self.reputation_extractor.extract, threat_data)

            # 2. Merge into Unified Vector
            unified_threat_features = {
                **blacklist_features,
                **malware_features,
                **reputation_features
            }

            # 3. Compute Composite Threat Assessment & Triage Verdict
            unified_threat_features = self._compute_threat_risk(unified_threat_features)

            return unified_threat_features

        except Exception as e:
            logger.error(f"Critical error during Threat Feature Extraction pipeline: {str(e)}")
            return self._get_empty_features()

    def _safe_extract(self, extractor_func, *args, **kwargs) -> Dict[str, Any]:
        """Isolates sub-module exceptions to prevent full pipeline failure."""
        try:
            return extractor_func(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Sub-module execution failed in {extractor_func.__qualname__}: {e}")
            return {}

    def _compute_threat_risk(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesizes indicators across Blacklist, Malware, and Reputation modules 
        to calculate a master threat score and operational verdict.
        """
        overall_score = 0.0

        # Base Reputation Weights
        weighted_rep = features.get("weighted_threat_score", 0.0)
        overall_score += weighted_rep * 0.40  # 40% weight to engine consensus

        # Blacklist Severity Penalties
        bl_severity = features.get("blacklist_risk_severity", "clean")
        if bl_severity == "critical":
            overall_score += 60.0
        elif bl_severity == "high":
            overall_score += 40.0
        elif bl_severity == "low":
            overall_score += 15.0

        # Malware & C2 Penalties
        mal_score = features.get("malware_threat_score", 0.0)
        overall_score += (mal_score * 0.60)  # 60% weight to specific malware/C2 findings

        # Immediate Critical Overrides
        if features.get("is_c2_node", False) or features.get("high_authority_vendor_flagged", False):
            overall_score = max(overall_score, 90.0)

        # Zero-Day Heuristic Amplification
        # If the domain is exceptionally young/untested but has a low volume of suspicious hits,
        # elevate the risk to ensure the local AI agents scrutinize it heavily.
        if features.get("is_zero_day_candidate", False) and overall_score > 10.0:
            overall_score = min(100.0, overall_score * 1.5)

        # Final Score Normalization
        overall_score = round(min(100.0, overall_score), 2)
        
        # FIX: Enforce clean verdict if 0 malicious vendors flagged it, regardless of other noise
        if features.get("malicious_engines_count", 0) == 0 and features.get("malicious_ratio", 0.0) == 0.0:
            overall_score = 0.0
            features["is_blacklisted"] = False
            features["blacklist_risk_severity"] = "clean"
            
        features["overall_threat_score"] = overall_score

        # Generate Human-Readable / Agent-Readable Verdict
        if overall_score >= 65.0:
            features["threat_verdict"] = "malicious"
        elif overall_score >= 25.0 or features.get("is_zero_day_candidate", False):
            features["threat_verdict"] = "suspicious"
        else:
            features["threat_verdict"] = "clean"

        # Boolean Flag for high-confidence AI triggers
        features["has_high_threat_consensus"] = features["threat_verdict"] == "malicious"

        return features

    def _get_empty_features(self) -> Dict[str, Any]:
        """Returns a safe, standardized baseline feature vector if extraction fails."""
        return {
            # Blacklist Features
            "is_blacklisted": False,
            "blacklist_vendors_count": 0,
            "high_authority_vendor_flagged": False,
            "matched_blacklist_categories": [],
            "flagging_vendors_list": [],
            "blacklist_risk_severity": "clean",
            
            # Malware Features
            "is_malware_associated": False,
            "matched_malware_families": [],
            "primary_malware_category": "none",
            "malware_threat_score": 0,
            "is_c2_node": False,
            "is_exploit_distributor": False,
            
            # Reputation Features
            "reputation_score": 0,
            "weighted_threat_score": 0.0,
            "harmless_engines_count": 0,
            "malicious_engines_count": 0,
            "suspicious_engines_count": 0,
            "undetected_engines_count": 0,
            "malicious_ratio": 0.0,
            "suspicion_ratio": 0.0,
            "is_zero_day_candidate": False,
            "days_since_first_submission": -1,
            "days_since_last_analysis": -1,
            
            # Composite Scoring
            "overall_threat_score": 0.0,
            "threat_verdict": "clean",
            "has_high_threat_consensus": False
        }
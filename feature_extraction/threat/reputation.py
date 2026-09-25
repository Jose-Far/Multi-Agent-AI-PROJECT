import logging
from typing import Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ReputationFeatureExtractor:
    """
    Advanced Threat Intelligence Reputation Extractor.
    
    Computes normalized reputation scores, consensus ratios, threat staleness, 
    and Zero-Day indicators derived from multi-engine threat analysis platforms 
    (e.g., VirusTotal, URLhaus).
    """
    def extract(self, threat_analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts complex statistical engine consensus and temporal metrics.
        
        Args:
            threat_analysis_data (Dict): Threat analysis block containing 'stats', 
                                         'reputation_score', and timestamp metadata.
            
        Returns:
            Dict[str, Any]: ML-ready reputation feature vector.
        """
        features = {
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
            "days_since_last_analysis": -1
        }

        if not threat_analysis_data or not isinstance(threat_analysis_data, dict):
            return features

        try:
            # 1. Base Reputation Scores
            features["reputation_score"] = int(threat_analysis_data.get("reputation_score", 0))
            
            # 2. Engine Consensus Extraction
            stats = threat_analysis_data.get("stats", {})
            if isinstance(stats, dict):
                harmless = int(stats.get("harmless", 0))
                malicious = int(stats.get("malicious", 0))
                suspicious = int(stats.get("suspicious", 0))
                undetected = int(stats.get("undetected", 0))

                features["harmless_engines_count"] = harmless
                features["malicious_engines_count"] = malicious
                features["suspicious_engines_count"] = suspicious
                features["undetected_engines_count"] = undetected

                total_engines = harmless + malicious + suspicious + undetected
                total_evaluated = harmless + malicious + suspicious

                if total_engines > 0:
                    features["malicious_ratio"] = round(malicious / total_engines, 4)
                    features["suspicion_ratio"] = round(suspicious / total_engines, 4)

                # 3. Weighted Threat Score (0.0 to 100.0)
                # Malicious flags carry a heavy weight (10x), suspicious flags carry a minor weight (2x)
                if total_evaluated > 0:
                    weighted_score = ((malicious * 10) + (suspicious * 2)) / (total_evaluated * 10) * 100
                    features["weighted_threat_score"] = round(min(100.0, weighted_score), 2)

                # 4. Zero-Day / Novelty Heuristic
                # If a large number of engines scan it but return "undetected", and there is no "harmless" consensus
                if total_engines > 0 and harmless < 3 and malicious == 0 and undetected > (total_engines * 0.7):
                    features["is_zero_day_candidate"] = True

            # 5. Temporal / Staleness Metrics
            # Phishing attacks are ephemeral. An analysis from 2 years ago is useless today.
            first_seen_str = threat_analysis_data.get("first_submission_date")
            last_seen_str = threat_analysis_data.get("last_analysis_date")
            
            current_time = datetime.now(timezone.utc)

            if first_seen_str:
                try:
                    # Handle typical ISO 8601 strings from Threat Intel APIs
                    first_seen = datetime.fromisoformat(str(first_seen_str).replace('Z', '+00:00'))
                    delta = current_time - first_seen
                    features["days_since_first_submission"] = max(0, delta.days)
                except ValueError:
                    pass

            if last_seen_str:
                try:
                    last_seen = datetime.fromisoformat(str(last_seen_str).replace('Z', '+00:00'))
                    delta = current_time - last_seen
                    features["days_since_last_analysis"] = max(0, delta.days)
                except ValueError:
                    pass

        except Exception as e:
            logger.error(f"Error during Reputation Feature Extraction: {str(e)}")

        return features
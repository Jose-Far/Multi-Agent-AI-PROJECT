import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

def analyze_certificate(ssl_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes certificate timing properties, validity, age, and expiry windows.
    
    Extracts deep heuristic features for the AI Agent, such as identifying 
    short-lived certificates (typical of free CAs used in phishing) and 
    newly minted certificates.
    
    Args:
        ssl_data (dict): Raw SSL telemetry from the data collection layer.
        
    Returns:
        dict: Processed numerical and boolean certificate feature metrics.
    """
    # 1. Initialize default skeleton (Secure defaults)
    features = {
        "has_ssl": False,
        "is_expired": True,
        "cert_age_days": 0,
        "days_until_expiry": 0,
        "total_lifespan_days": 0,
        "is_self_signed": False,
        "is_recently_issued": False,  # Flag for certs < 14 days old
        "is_short_lived": False       # Flag for 90-day (or less) certs
    }

    if not ssl_data or not ssl_data.get("has_ssl", False):
        logger.debug("No valid SSL data provided to certificate extractor. Returning defaults.")
        return features

    try:
        # 2. Extract Base Metrics
        features["has_ssl"] = True
        features["is_expired"] = ssl_data.get("is_expired", True)
        features["cert_age_days"] = ssl_data.get("cert_age_days", 0)
        features["days_until_expiry"] = ssl_data.get("days_until_expiry", 0)

        # 3. Time-Based Phishing Heuristics
        # Calculate total lifespan if not explicitly provided
        not_before_str = ssl_data.get("not_before")
        not_after_str = ssl_data.get("not_after")
        
        if not_before_str and not_after_str:
            try:
                # Parse ISO 8601 timestamps (e.g., "2026-07-20T18:08:36")
                not_before = datetime.fromisoformat(not_before_str.replace("Z", ""))
                not_after = datetime.fromisoformat(not_after_str.replace("Z", ""))
                
                lifespan = (not_after - not_before).days
                features["total_lifespan_days"] = max(0, lifespan)
                
                # Phishing Indicator A: Short-lived certificates
                # Free automated certs (Let's Encrypt, ZeroSSL) are typically exactly 90 days.
                if features["total_lifespan_days"] <= 90:
                    features["is_short_lived"] = True
                    
            except ValueError as ve:
                logger.warning(f"Could not parse certificate dates for lifespan calculation: {ve}")

        # Phishing Indicator B: Recently issued certificates
        # Phishers constantly burn through domains and spin up new certs just hours before an attack.
        if 0 <= features["cert_age_days"] <= 14:
            features["is_recently_issued"] = True

        # 4. Self-Signed Heuristic
        # If the entity that issued the cert is the exact same as the entity receiving it, 
        # it is self-signed (meaning there is zero third-party trust validation).
        subject = ssl_data.get("subject_common_name", "").strip().lower()
        issuer = ssl_data.get("issuer_common_name", "").strip().lower()
        
        if subject and issuer and subject == issuer:
            features["is_self_signed"] = True
        else:
            features["is_self_signed"] = ssl_data.get("is_self_signed", False)

    except Exception as e:
        logger.error(f"Unexpected error during certificate feature extraction: {str(e)}")

    return features

# ---------------------------------------------------------
# Standalone Local Testing Hook
# ---------------------------------------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     # Mock payload simulating a Let's Encrypt 90-day cert created 5 days ago
#     sample_ssl_data = {
#         "has_ssl": True,
#         "is_expired": False,
#         "cert_age_days": 5,
#         "days_until_expiry": 85,
#         "issuer_common_name": "R3",
#         "subject_common_name": "paypal-secure-update.com",
#         "not_before": "2026-08-06T10:00:00",
#         "not_after": "2026-11-04T10:00:00"
#     }
    
#     print("Testing Certificate Feature Extractor...")
#     result = analyze_certificate(sample_ssl_data)
    
#     import json
#     print(json.dumps(result, indent=4))
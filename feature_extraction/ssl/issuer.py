import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Threat Intelligence: Known Certificate Authority Profiles
# ---------------------------------------------------------

# Highly trusted, enterprise-grade Commercial CAs
# Phishers rarely use these because they cost money and require identity verification.
TRUSTED_CA_ORGS = {
    "digicert", 
    "globalsign", 
    "sectigo", 
    "entrust", 
    "symantec",
    "amazon", 
    "google trust services", 
    "microsoft", 
    "godaddy",
    "network solutions",
    "apple inc.",
    "identrust"
}

# Free and Automated CAs
# These are perfectly legitimate services, but they are HEAVILY abused by phishers 
# because they can be provisioned via API in seconds for free.
FREE_CA_ORGS = {
    "let's encrypt", 
    "zerossl", 
    "trustasia", 
    "cpanel", 
    "cloudflare", # Cloudflare provides free edge certs heavily used to mask phishing IPs
    "buypass"
}

# Suspicious / Local / Default CAs
# A public website should never use these.
SUSPICIOUS_CA_ORGS = {
    "localhost", 
    "snakeoil", 
    "default company ltd", 
    "someorganization", 
    "kubernetes"
}

def analyze_issuer(ssl_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the trustworthiness of the Certificate Authority (CA) that issued the SSL cert.
    
    This module categorizes the issuer into Trusted, Free/Automated, or Suspicious tiers
    and assigns a heuristic trust score to help the AI Agent determine if the SSL 
    implementation aligns with the expected behavior of a legitimate enterprise brand.
    
    Args:
        ssl_data (dict): Raw SSL telemetry from the data collection layer.
        
    Returns:
        dict: Processed numerical and boolean issuer feature metrics.
    """
    # 1. Initialize default skeleton (Secure/Neutral defaults)
    features = {
        "issuer_org": "missing",
        "issuer_cn": "missing",
        "is_trusted_ca": False,
        "is_free_ca": False,
        "is_automated_ca": False,
        "is_suspicious_ca": False,
        "issuer_trust_score": 0  # Scale of 0 (Bad) to 100 (Enterprise Trust)
    }

    if not ssl_data or not ssl_data.get("has_ssl", False):
        logger.debug("No valid SSL data provided to issuer extractor. Returning defaults.")
        return features

    try:
        # 2. Extract and Normalize Raw Metadata
        # We use .lower() to ensure our substring matching works regardless of case formatting.
        raw_org = ssl_data.get("issuer_organization", "")
        raw_cn = ssl_data.get("issuer_common_name", "")
        
        # Handle cases where the fields might be explicitly None in the JSON
        org = raw_org.strip().lower() if isinstance(raw_org, str) else ""
        cn = raw_cn.strip().lower() if isinstance(raw_cn, str) else ""

        features["issuer_org"] = raw_org if raw_org else "missing"
        features["issuer_cn"] = raw_cn if raw_cn else "missing"

        # If we have literally no issuer data on an active SSL connection, that's highly suspicious
        if not org and not cn:
            features["is_suspicious_ca"] = True
            features["issuer_trust_score"] = 0
            return features

        # Combine org and cn for broader heuristic matching 
        # (Sometimes the CA name is only in the Common Name, not the Organization field)
        issuer_signature = f"{org} {cn}"

        # 3. Categorization & Trust Scoring Logic
        is_trusted = any(ca in issuer_signature for ca in TRUSTED_CA_ORGS)
        is_free = any(ca in issuer_signature for ca in FREE_CA_ORGS)
        is_suspicious = any(ca in issuer_signature for ca in SUSPICIOUS_CA_ORGS)

        # Apply flags based on strict precedence
        if is_suspicious:
            features["is_suspicious_ca"] = True
            features["issuer_trust_score"] = 10
            
        elif is_free:
            features["is_free_ca"] = True
            features["is_automated_ca"] = True  # Explicitly map both flags for clarity
            # Free CAs are common, so we don't score them as 0, but they lack enterprise trust
            features["issuer_trust_score"] = 50 
            
        elif is_trusted:
            features["is_trusted_ca"] = True
            features["issuer_trust_score"] = 100
            
        else:
            # Unknown Commercial CA
            features["issuer_trust_score"] = 40

    except Exception as e:
        logger.error(f"Unexpected error during SSL issuer feature extraction: {str(e)}")

    return features
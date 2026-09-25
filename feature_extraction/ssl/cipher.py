import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Threat Intelligence: Cryptographic Protocol & Cipher Baselines
# ---------------------------------------------------------

# Protocols considered deprecated or fundamentally broken by the IETF
# BUG FIX: Formatted for EXACT matching to prevent false positives on TLS 1.3
WEAK_PROTOCOLS = {
    "sslv2", 
    "sslv3", 
    "tls/1.0", 
    "tlsv1.0",
    "tlsv1", 
    "tls/1.1", 
    "tlsv1.1"
}

# Cipher suites known to be vulnerable to specific attacks (e.g., SWEET32, POODLE)
VULNERABLE_CIPHER_FRAGMENTS = {
    "rc4",      # Broken stream cipher
    "des",      # Broken block cipher
    "3des",     # Vulnerable to SWEET32
    "md5",      # Collision-prone hash
    "anon",     # Anonymous authentication (no identity verification)
    "null",     # No encryption
    "export"    # Intentionally weakened export-grade ciphers
}

def analyze_cipher(ssl_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the cryptographic strength of the negotiated SSL/TLS connection.
    
    Extracts protocol versions and cipher suites, evaluating them against modern 
    cryptographic standards to identify weak or deprecated configurations often 
    associated with poorly maintained or malicious infrastructure.
    
    Args:
        ssl_data (dict): Raw SSL telemetry from the data collection layer.
        
    Returns:
        dict: Cryptographic strength, protocol security, and vulnerability indicators.
    """
    features = {
        "tls_version": "unknown",
        "cipher_suite": "unknown",
        "cipher_strength_bits": 0,
        "is_weak_protocol": False,
        "is_vulnerable_cipher": False,
        "crypto_health_score": 0  # Scale 0 (Broken) to 100 (Perfect Forward Secrecy)
    }

    if not ssl_data or not ssl_data.get("has_ssl", False):
        logger.debug("No valid SSL data provided to cipher extractor. Returning defaults.")
        return features

    try:
        # Extract base values
        raw_version = str(ssl_data.get("tls_version", "unknown")).lower().strip()
        raw_cipher = str(ssl_data.get("cipher_suite", "unknown")).lower().strip()
        
        features["tls_version"] = raw_version if raw_version else "unknown"
        features["cipher_suite"] = raw_cipher if raw_cipher else "unknown"
        
        # Safely extract cipher bit strength (Default to 0 if missing or unparseable)
        try:
            features["cipher_strength_bits"] = int(ssl_data.get("cipher_strength_bits", 0))
        except (ValueError, TypeError):
            features["cipher_strength_bits"] = 0

        health_score = 50  # Start neutral

        # 1. Protocol Version Analysis (BUG FIX: Exact Match Logic)
        if raw_version in WEAK_PROTOCOLS:
            features["is_weak_protocol"] = True
            health_score -= 40
        else:
            features["is_weak_protocol"] = False
            health_score += 25
            
            # Bonus points for modern TLS standards
            if "1.3" in raw_version:
                health_score += 15 

        # 2. Cipher Suite Vulnerability Analysis (Substring match is still correct here)
        if raw_cipher != "unknown":
            if any(vuln in raw_cipher for vuln in VULNERABLE_CIPHER_FRAGMENTS):
                features["is_vulnerable_cipher"] = True
                health_score -= 30
            
            # Check for Forward Secrecy (ECDHE/DHE) and strong authenticated encryption (GCM/CCM/CHACHA20)
            if "ecdhe" in raw_cipher or "dhe" in raw_cipher:
                health_score += 10
            if "gcm" in raw_cipher or "chacha20" in raw_cipher:
                health_score += 10

        # 3. Bit Strength Verification
        if features["cipher_strength_bits"] < 128 and features["cipher_strength_bits"] > 0:
            health_score -= 20 # Penalize weak encryption keys
            features["is_vulnerable_cipher"] = True # Force flag for <128 bit

        # Ensure score stays within 0-100 bounds
        features["crypto_health_score"] = max(0, min(100, health_score))

    except Exception as e:
        logger.error(f"Unexpected error during SSL cipher extraction: {str(e)}")

    return features

# ---------------------------------------------------------
# Standalone Local Testing Hook
# ---------------------------------------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     # Test 1: Modern Secure Configuration (TLS 1.3 + AEAD)
#     test_modern = {
#         "has_ssl": True,
#         "tls_version": "TLSv1.3",
#         "cipher_suite": "TLS_AES_256_GCM_SHA384",
#         "cipher_strength_bits": 256
#     }
    
#     # Test 2: Vulnerable Legacy Configuration (TLS 1.0 + RC4)
#     test_legacy = {
#         "has_ssl": True,
#         "tls_version": "TLSv1.0",
#         "cipher_suite": "TLS_RSA_WITH_RC4_128_MD5",
#         "cipher_strength_bits": 128
#     }

#     print("--- Testing Modern Cipher Extractor ---")
#     mod_res = analyze_cipher(test_modern)
#     import json
#     print(json.dumps(mod_res, indent=4))
    
#     print("\n--- Testing Vulnerable Cipher Extractor ---")
#     leg_res = analyze_cipher(test_legacy)
#     print(json.dumps(leg_res, indent=4))
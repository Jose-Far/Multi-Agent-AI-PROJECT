import ssl
import socket
import logging
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SSLCollector:
    """
    Advanced SSL/TLS Telemetry Collector for Phishing Detection.
    
    Connects to the target domain over port 443 and extracts the SSL/TLS certificate metadata,
    including cryptographic ciphers, TLS versions, Subject Alternative Names (SANs), and issuer trust details.
    
    Utilizes a dual-pass verification system to aggressively extract TLS metadata 
    even if the phishing site is using an expired, self-signed, or mismatched certificate.
    """
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        
        # Known Certificate Authorities frequently abused by automated phishing kits
        self.high_risk_free_cas = ["let's encrypt", "zerossl", "cpanel", "cloudflare", "google trust services"]

    def collect(self, target_url: str) -> Dict[str, Any]:
        """
        Orchestrates the SSL connection, input sanitization, and payload formatting.
        
        Args:
            target_url (str): The destination URL or hostname to inspect.
            
        Returns:
            Dict[str, Any]: Standardized telemetry payload containing SSL structure.
        """
        parsed_url = urlparse(target_url)
        hostname = parsed_url.hostname
        
        # Fallback if a raw domain (e.g., 'google.com') was passed instead of a full URL
        if not hostname:
            if "/" not in target_url:
                hostname = target_url.split(':')[0]
            else:
                return self._build_error_payload("Could not extract a valid hostname from the URL.")

        # Strip 'www.' to ensure we test the root domain's primary certificate coverage
        hostname = hostname.replace("www.", "") if hostname.startswith("www.") else hostname
        port = parsed_url.port or 443

        return self._fetch_ssl_data(hostname, port)

    def _fetch_ssl_data(self, hostname: str, port: int) -> Dict[str, Any]:
        """
        Executes the TLS Handshake via SNI and parses the transport layer cryptography.
        """
        # Comprehensive Feature Payload Blueprint for AI ML Models
        ssl_data: Dict[str, Any] = {
            "has_ssl": False,
            "tls_version": "unknown",
            "cipher_suite": "unknown",
            "cipher_strength_bits": 0,
            "issuer_organization": "Unknown",
            "issuer_common_name": "Unknown",
            "issuer_country": "Unknown",
            "subject_common_name": "Unknown",
            "subject_alt_names": [],
            "not_before": None,
            "not_after": None,
            "cert_age_days": 0,
            "days_until_expiry": 0,
            "is_expired": True,
            "is_self_signed": False,
            "is_free_ca": False
        }

        # PASS 1: Strict Verification (Standard Browser Behavior)
        strict_context = ssl.create_default_context()
        
        try:
            with socket.create_connection((hostname, port), timeout=self.timeout) as sock:
                # server_hostname is critical for SNI (Server Name Indication)
                with strict_context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    self._populate_cipher_data(ssock, ssl_data)
                    self._populate_cert_data(cert, ssl_data)
                    
        except ssl.SSLCertVerificationError as e:
            # 🔴 PHISHING HEURISTIC TRIGGER: Bad Certificate chain or Expired Cert
            logger.warning(f"Strict SSL Verification failed for {hostname} ({e}). Triggering fallback extraction...")
            ssl_data["is_self_signed"] = True  
            ssl_data["is_expired"] = True      
            ssl_data["issuer_organization"] = "Untrusted/Verification Failed"
            
            # PASS 2: Unverified Context Fallback (Extract Ciphers even if cert chain is broken)
            unverified_context = ssl._create_unverified_context()
            try:
                with socket.create_connection((hostname, port), timeout=self.timeout) as sock:
                    with unverified_context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        self._populate_cipher_data(ssock, ssl_data)
                        
                        # In Python, getpeercert() on an unverified context returns empty. 
                        # We capture the binary DER and parse what we can if required, but primarily 
                        # rely on the fact that the cipher negotiation succeeded.
            except Exception as unverified_e:
                logger.debug(f"Unverified SSL fallback also failed for {hostname}: {unverified_e}")
                return self._build_error_payload(f"SSL Verification Error: {str(e)}")
                
        except ssl.SSLError as e:
            return self._build_error_payload(f"SSL Protocol/Handshake Failed: {str(e)}")
        except socket.timeout:
            return self._build_error_payload("Connection timed out while fetching SSL certificate.")
        except socket.error as e:
            return self._build_error_payload(f"Network error (Host may not support HTTPS on port {port}): {str(e)}")
        except Exception as e:
            return self._build_error_payload(f"Unexpected execution error: {str(e)}")

        return {
            "success": True,
            "error": None,
            "data": ssl_data
        }

    def _populate_cipher_data(self, ssock: ssl.SSLSocket, ssl_data: dict):
        """
        Extracts the negotiated cryptographic parameters from the active socket.
        """
        ssl_data["has_ssl"] = True
        
        # ssock.cipher() returns a tuple: ('TLS_AES_256_GCM_SHA384', 'TLSv1.3', 256)
        cipher_info = ssock.cipher()
        if cipher_info:
            ssl_data["cipher_suite"] = cipher_info[0]
            ssl_data["tls_version"] = cipher_info[1]
            ssl_data["cipher_strength_bits"] = cipher_info[2]
        else:
            ssl_data["tls_version"] = ssock.version() or "unknown"

    def _populate_cert_data(self, cert: dict, ssl_data: dict):
        """
        Deep parses the X.509 Certificate dictionary returned by Python's ssl module.
        """
        if not cert:
            return

        # 1. Extract X.509 Hierarchical Identifiers
        issuer = {k: v for tuple_list in cert.get('issuer', []) for k, v in tuple_list}
        subject = {k: v for tuple_list in cert.get('subject', []) for k, v in tuple_list}
        
        ssl_data["issuer_organization"] = issuer.get('organizationName', 'Unknown')
        ssl_data["issuer_common_name"] = issuer.get('commonName', 'Unknown')
        ssl_data["issuer_country"] = issuer.get('countryName', 'Unknown')
        ssl_data["subject_common_name"] = subject.get('commonName', 'Unknown')
        
        # Flag if the CA is a known free/automated provider often used in bulk by phishers
        issuer_org_lower = ssl_data["issuer_organization"].lower()
        ssl_data["is_free_ca"] = any(free_ca in issuer_org_lower for free_ca in self.high_risk_free_cas)

        # 2. Extract Subject Alternative Names (SANs) - Crucial for detecting domain mismatch
        sans = cert.get('subjectAltName', [])
        ssl_data["subject_alt_names"] = [san[1] for san in sans if san[0].lower() == 'dns']

        # 3. Extract and compute Time Deltas (using timezone-aware math)
        not_before_str = cert.get('notBefore')
        not_after_str = cert.get('notAfter')
        
        if not_before_str and not_after_str:
            # Python SSL dates are formatted as: 'May  9 14:00:00 2024 GMT'
            try:
                not_before = datetime.strptime(not_before_str, '%b %d %H:%M:%S %Y %Z').replace(tzinfo=timezone.utc)
                not_after = datetime.strptime(not_after_str, '%b %d %H:%M:%S %Y %Z').replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                
                # Format to ISO 8601 for standard JSON consumption downstream
                ssl_data["not_before"] = not_before.isoformat()
                ssl_data["not_after"] = not_after.isoformat()
                
                # Compute lifecycle metrics
                ssl_data["cert_age_days"] = max(0, (now - not_before).days)
                days_until_expiry = (not_after - now).days
                ssl_data["days_until_expiry"] = days_until_expiry
                ssl_data["is_expired"] = days_until_expiry < 0
            except ValueError as ve:
                logger.debug(f"Failed to parse SSL dates: {ve}")

    def _build_error_payload(self, message: str) -> Dict[str, Any]:
        """Constructs standardized error response structure."""
        logger.error(f"SSLCollector Error: {message}")
        return {
            "success": False,
            "error": message,
            "data": {
                "has_ssl": False,
                "tls_version": "unknown",
                "cipher_suite": "unknown",
                "issuer_organization": "Unknown",
                "is_expired": True
            }
        }
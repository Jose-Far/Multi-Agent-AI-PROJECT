import ipaddress
import socket
from urllib.parse import urlparse
from typing import Tuple, Optional
import logging
from .config import SecurityConfig

logger = logging.getLogger(__name__)

class SSRFValidator:
    @classmethod
    def validate_url(cls, url: str) -> Tuple[bool, str, Optional[str]]:
        try:
            parsed = urlparse(url)
            if parsed.scheme.lower() not in SecurityConfig.ALLOWED_SCHEMES:
                return False, url, f"Prohibited scheme: {parsed.scheme}://"
                
            hostname = parsed.hostname
            if not hostname:
                return False, url, "Invalid URL format: missing hostname"
                
            if parsed.username or parsed.password:
                return False, url, "Credentials in URL are prohibited for security"
                
            if SecurityConfig.BLOCK_PRIVATE_NETWORKS or SecurityConfig.BLOCK_LOCALHOST:
                is_safe_ip, ip_err = cls._is_safe_hostname(hostname)
                if not is_safe_ip:
                    return False, url, ip_err
                    
            return True, url, None
            
        except Exception as e:
            logger.error(f"URL parsing error: {e}")
            return False, url, "Malformed URL"

    @classmethod
    def _is_safe_hostname(cls, hostname: str) -> Tuple[bool, Optional[str]]:
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            ips = [info[4][0] for info in addr_info]
            for ip_str in ips:
                ip = ipaddress.ip_address(ip_str)
                if SecurityConfig.BLOCK_LOCALHOST and ip.is_loopback:
                    return False, f"Destination resolves to loopback address: {ip_str}"
                if SecurityConfig.BLOCK_PRIVATE_NETWORKS:
                    if ip.is_private:
                        return False, f"Destination resolves to private network: {ip_str}"
                    if ip.is_link_local:
                        return False, f"Destination resolves to link-local network: {ip_str}"
                    if ip.is_multicast:
                        return False, f"Destination resolves to multicast address: {ip_str}"
                    if ip.is_reserved or ip.is_unspecified:
                        return False, f"Destination resolves to reserved/unspecified address: {ip_str}"
            return True, None
        except socket.gaierror:
            return False, f"DNS resolution failed for hostname: {hostname}"
        except ValueError:
            return False, f"Invalid IP address format derived from hostname: {hostname}"

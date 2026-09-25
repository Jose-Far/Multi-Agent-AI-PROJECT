import re
import logging
import ipaddress
import unicodedata
from urllib.parse import urlparse, urlunparse
from typing import Tuple, Optional

# Import the centralized security module!
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from security.validator import SSRFValidator

logger = logging.getLogger(__name__)

class URLValidator:
    DOMAIN_REGEX = re.compile(
        r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}$'
    )

    @staticmethod
    def sanitize_and_validate(raw_url: str, prevent_ssrf: bool = True, max_length: int = 2048) -> Tuple[bool, str]:
        if not raw_url or not isinstance(raw_url, str):
            return False, ""

        if len(raw_url) > max_length:
            return False, raw_url

        sanitized = "".join(ch for ch in raw_url if unicodedata.category(ch)[0] != "C")
        sanitized = sanitized.strip()

        scheme_match = re.match(r'^([a-zA-Z0-9+.-]+):', sanitized)
        if scheme_match:
            scheme = scheme_match.group(1).lower()
            if scheme not in ['http', 'https']:
                return False, sanitized
        else:
            sanitized = "https://" + sanitized

        try:
            parsed = urlparse(sanitized)
            if parsed.username or parsed.password:
                safe_netloc = parsed.hostname
                if parsed.port:
                    safe_netloc += f":{parsed.port}"
                parsed = parsed._replace(netloc=safe_netloc)
                sanitized = urlunparse(parsed)

            # Central SSRF validation (Handles IP literals AND DNS resolution checks)
            if prevent_ssrf:
                is_safe, clean_url, msg = SSRFValidator.validate_url(sanitized)
                if not is_safe:
                    logger.warning(f"Central SSRF blocked: {msg}")
                    return False, sanitized
            else:
                host = parsed.hostname
                if not host:
                    return False, sanitized
                try:
                    host.encode('idna').decode('ascii')
                except:
                    return False, sanitized

            return True, sanitized

        except Exception as e:
            return False, sanitized

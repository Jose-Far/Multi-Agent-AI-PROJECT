import time
import logging
import requests
import urllib3

from urllib.parse import urlparse
from typing import Dict, Any, List


# Suppress warnings because phishing sites may use invalid/self-signed
# certificates and we intentionally disable TLS certificate verification
# for telemetry collection.
urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

logger = logging.getLogger(__name__)


class URLCollector:
    """
    Advanced URL Telemetry Collector for Phishing Detection.

    Responsibilities:
        - Normalize target URLs
        - Perform HTTP/HTTPS requests
        - Follow and record redirect chains
        - Capture response metadata
        - Sanitize sensitive response headers
        - Analyze cookie security metadata without storing values
        - Measure response latency
        - Provide standardized error responses

    Privacy:
        - Raw Set-Cookie values are never stored
        - Cookie values are never returned
        - Only cookie security metadata is retained
    """

    def __init__(
        self,
        timeout: int = 15,
        max_redirects: int = 10
    ):
        self.timeout = timeout
        self.max_redirects = max_redirects

        # -------------------------------------------------------------
        # Browser-like request headers
        # -------------------------------------------------------------

        self.default_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,image/avif,"
                "image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        }

    # =================================================================
    # MAIN COLLECTION METHOD
    # =================================================================

    def collect(
        self,
        target_url: str
    ) -> Dict[str, Any]:
        """
        Executes the network request and collects URL telemetry.

        Args:
            target_url:
                URL to analyze.

        Returns:
            Standardized URL telemetry dictionary.
        """

        # -------------------------------------------------------------
        # Validate input
        # -------------------------------------------------------------

        if not isinstance(
            target_url,
            str
        ) or not target_url.strip():

            logger.error(
                "URLCollector received an empty or invalid URL."
            )

            return self._handle_error(
                str(target_url),
                "InvalidURL",
                "Target URL is empty or invalid."
            )

        target_url = target_url.strip()

        # -------------------------------------------------------------
        # Normalize URL scheme
        # -------------------------------------------------------------

        if not target_url.startswith(
            (
                "http://",
                "https://"
            )
        ):
            target_url = "https://" + target_url

        session = requests.Session()
        session.max_redirects = self.max_redirects

        # SSRF Redirect Protection Hook
        def _ssrf_redirect_hook(res, *args, **kwargs):
            if res.is_redirect:
                next_url = res.headers.get('Location')
                if next_url:
                    from urllib.parse import urljoin
                    full_next = urljoin(res.url, next_url)
                    import sys, os
                    if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from security.validator import SSRFValidator
                    is_safe, _, msg = SSRFValidator.validate_url(full_next)
                    if not is_safe:
                        raise requests.exceptions.RequestException(f"SSRF Blocked on redirect: {msg}")

        session.hooks['response'].append(_ssrf_redirect_hook)

        start_time = time.time()

        try:

            logger.info(
                "URLCollector initiating network trace for: %s",
                target_url
            )

            # =========================================================
            # HTTP REQUEST
            # =========================================================

            response = session.get(
                target_url,
                allow_redirects=True,
                timeout=self.timeout,
                verify=False,
                stream=True  # Phase 9: Do not download the body!
            )
            response.close() # Phase 9: Close connection immediately after headers

            latency = time.time() - start_time

            # =========================================================
            # REDIRECT HISTORY
            # =========================================================

            redirects: List[Dict[str, Any]] = []

            for hop in response.history:

                hop_url = str(
                    hop.url
                )

                redirects.append({

                    "url": hop_url,

                    "status_code": hop.status_code,

                    "headers": self._sanitize_headers(
                        hop.headers
                    ),

                    "is_https": hop_url.lower().startswith(
                        "https://"
                    ),

                    "domain": self._extract_domain(
                        hop_url
                    )
                })

            # =========================================================
            # FINAL RESPONSE HEADER SANITIZATION
            # =========================================================

            safe_headers = self._sanitize_headers(
                response.headers
            )

            # =========================================================
            # COOKIE SECURITY METADATA
            # =========================================================

            cookie_analysis = (
                self._analyze_cookie_metadata(
                    session
                )
            )

            # =========================================================
            # FINAL DOMAIN
            # =========================================================

            final_url = str(
                response.url
            )

            domain_resolved = self._extract_domain(
                final_url
            )

            # =========================================================
            # SUCCESS RESPONSE
            # =========================================================

            return {

                "success": True,

                "status": "success",

                "initial_url": target_url,

                "final_url": final_url,

                "domain_resolved": domain_resolved,

                "response_time_sec": round(
                    latency,
                    4
                ),

                "status_code": response.status_code,

                "redirect_count": len(
                    redirects
                ),

                "redirect_history": redirects,

                "headers": safe_headers,

                "cookie_analysis": cookie_analysis,

                "error": None
            }

        # =============================================================
        # REDIRECT LIMIT
        # =============================================================

        except requests.exceptions.TooManyRedirects:

            latency = time.time() - start_time

            return self._handle_error(
                target_url,
                "TooManyRedirects",
                (
                    f"Exceeded maximum allowed redirects "
                    f"({self.max_redirects})."
                ),
                latency
            )

        # =============================================================
        # TIMEOUT
        # =============================================================

        except requests.exceptions.Timeout:

            latency = time.time() - start_time

            return self._handle_error(
                target_url,
                "Timeout",
                "Network request timed out.",
                latency
            )

        # =============================================================
        # SSL ERROR
        # =============================================================

        except requests.exceptions.SSLError as e:

            latency = time.time() - start_time

            return self._handle_error(
                target_url,
                "SSLError",
                f"Severe TLS protocol failure: {str(e)}",
                latency
            )

        # =============================================================
        # CONNECTION ERROR
        # =============================================================

        except requests.exceptions.ConnectionError as e:

            latency = time.time() - start_time

            return self._handle_error(
                target_url,
                "ConnectionError",
                f"Failed to resolve or connect to host: {str(e)}",
                latency
            )

        # =============================================================
        # GENERAL REQUEST ERROR
        # =============================================================

        except requests.exceptions.RequestException as e:

            latency = time.time() - start_time

            return self._handle_error(
                target_url,
                "RequestException",
                f"Unexpected network error: {str(e)}",
                latency
            )

        # =============================================================
        # UNEXPECTED ERROR
        # =============================================================

        except Exception as e:

            latency = time.time() - start_time

            logger.error(
                "Unexpected URLCollector failure: %s",
                str(e),
                exc_info=True
            )

            return self._handle_error(
                target_url,
                type(e).__name__,
                str(e),
                latency
            )

        finally:

            session.close()

    # =================================================================
    # HEADER SANITIZATION
    # =================================================================

    @staticmethod
    def _sanitize_headers(
        headers: Any
    ) -> Dict[str, Any]:
        """
        Remove sensitive Set-Cookie headers.

        Raw cookie values must never enter the telemetry vector.
        """

        if not hasattr(
            headers,
            "items"
        ):
            return {}

        safe_headers = dict(
            headers
        )

        # Case-insensitive removal
        sensitive_headers = {
            "set-cookie",
            "cookie",
            "authorization",
            "proxy-authorization"
        }

        for key in list(
            safe_headers.keys()
        ):

            if key.lower() in sensitive_headers:

                safe_headers.pop(
                    key,
                    None
                )

        return safe_headers

    # =================================================================
    # COOKIE METADATA ANALYSIS
    # =================================================================

    @staticmethod
    def _analyze_cookie_metadata(
        session: requests.Session
    ) -> Dict[str, Any]:
        """
        Analyze cookie security attributes without storing
        actual cookie names or values.
        """

        secure_count = 0
        httponly_count = 0
        samesite_count = 0

        try:

            for cookie in session.cookies:

                # -----------------------------------------------------
                # Secure attribute
                # -----------------------------------------------------

                if getattr(
                    cookie,
                    "secure",
                    False
                ):
                    secure_count += 1

                # -----------------------------------------------------
                # Requests stores non-standard cookie attributes
                # inside the private _rest dictionary.
                # -----------------------------------------------------

                rest = getattr(
                    cookie,
                    "_rest",
                    {}
                )

                if not isinstance(
                    rest,
                    dict
                ):
                    rest = {}

                rest_keys = {
                    str(key).lower()
                    for key in rest.keys()
                }

                # -----------------------------------------------------
                # HttpOnly
                # -----------------------------------------------------

                if (
                    "httponly" in rest_keys
                    or any(
                        str(key).lower() == "httponly"
                        for key in rest
                    )
                ):
                    httponly_count += 1

                # -----------------------------------------------------
                # SameSite
                # -----------------------------------------------------

                if (
                    "samesite" in rest_keys
                    or any(
                        str(key).lower() == "samesite"
                        for key in rest
                    )
                ):
                    samesite_count += 1

        except Exception as e:

            logger.debug(
                "Cookie metadata analysis failed: %s",
                str(e)
            )

        return {

            "cookie_count": len(
                session.cookies
            ),

            "secure_count": secure_count,

            "httponly_count": httponly_count,

            "samesite_count": samesite_count,

            # Mandatory privacy flag
            "raw_values_stored": False
        }

    # =================================================================
    # DOMAIN EXTRACTION
    # =================================================================

    @staticmethod
    def _extract_domain(
        url: str
    ) -> str:
        """
        Safely extract hostname/netloc from a URL.
        """

        try:

            parsed = urlparse(
                url
            )

            return (
                parsed.hostname
                or ""
            ).lower()

        except Exception:

            return ""

    # =================================================================
    # STANDARDIZED ERROR HANDLER
    # =================================================================

    def _handle_error(
        self,
        url: str,
        error_type: str,
        message: str,
        latency: float = 0.0
    ) -> Dict[str, Any]:
        """
        Construct a standardized URL collector error response.
        """

        logger.error(
            "URLCollector %s: %s",
            error_type,
            message
        )

        return {

            "success": False,

            "status": "error",

            "initial_url": url,

            "final_url": None,

            "domain_resolved": None,

            "response_time_sec": round(
                latency,
                4
            ),

            "status_code": None,

            "redirect_count": 0,

            "redirect_history": [],

            "headers": {},

            "cookie_analysis": {

                "cookie_count": 0,

                "secure_count": 0,

                "httponly_count": 0,

                "samesite_count": 0,

                "raw_values_stored": False
            },

            "error": {

                "type": error_type,

                "message": message
            }
        }
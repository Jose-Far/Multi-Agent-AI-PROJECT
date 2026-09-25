import re
import urllib.parse
import logging
from typing import Dict, Any, Tuple, Optional

from bs4 import BeautifulSoup, Comment

try:
    from playwright.sync_api import (
        sync_playwright,
        TimeoutError as PlaywrightTimeoutError
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


logger = logging.getLogger(__name__)


class HTMLCollector:
    """
    Production-grade HTML/DOM collector for the HTML AI Agent.

    Responsibilities
    ----------------
    1. Launch a controlled Chromium browser.
    2. Navigate to the target URL.
    3. Avoid relying on `networkidle`.
    4. Accept pages that reach DOM/content readiness.
    5. Extract the rendered DOM using page.content().
    6. Gracefully handle navigation timeouts.
    7. Preserve partially rendered HTML when possible.
    8. Return a standardized collection payload.
    9. Never convert collection failure into legitimate evidence.

    Important design decision
    -------------------------
    Dynamic websites can continuously perform network requests.

    Therefore this collector does NOT use:

        wait_until="networkidle"

    as the primary page-readiness condition.

    Instead it uses:

        wait_until="domcontentloaded"

    followed by a controlled short stabilization wait.

    This prevents a continuously active JavaScript application from
    causing unnecessary HTML collection failures.
    """

    # ======================================================================
    # CONFIGURATION
    # ======================================================================

    DEFAULT_TIMEOUT_SECONDS = 20

    DEFAULT_STABILIZATION_MS = 1500

    MAX_HTML_LENGTH = 10_000_000

    MIN_VALID_HTML_LENGTH = 50

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        stabilization_ms: int = DEFAULT_STABILIZATION_MS
    ):
        """
        Initialize HTML collector.

        Args:
            timeout:
                Maximum navigation/operation timeout in seconds.

            stabilization_ms:
                Controlled wait after DOMContentLoaded to allow
                JavaScript-rendered content to appear.
        """

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        if stabilization_ms < 0:
            raise ValueError(
                "stabilization_ms cannot be negative."
            )

        self.timeout_seconds = int(timeout)

        self.timeout_ms = (
            self.timeout_seconds * 1000
        )

        self.stabilization_ms = int(
            stabilization_ms
        )

        # --------------------------------------------------------------
        # JavaScript obfuscation signatures
        # --------------------------------------------------------------

        self.obfuscation_patterns = [
            r'eval\(',
            r'unescape\(',
            r'atob\(',
            r'btoa\(',
            r'document\.write\(',
            r'String\.fromCharCode\(',
            r'%[0-9a-fA-F]{2}',
            r'\\x[0-9a-fA-F]{2}',
            r'\\u[0-9a-fA-F]{4}',
            r'window\[".*"\]'
        ]

        self.compiled_obfuscation_patterns = [
            re.compile(
                pattern,
                re.IGNORECASE
            )
            for pattern in self.obfuscation_patterns
        ]

    # ======================================================================
    # PUBLIC COLLECTION METHOD
    # ======================================================================

    def collect(
        self,
        target_url: str
    ) -> Dict[str, Any]:
        """
        Collect rendered HTML from a target URL.

        The collector follows this pipeline:

            URL
             ↓
            Chromium
             ↓
            DOMContentLoaded
             ↓
            Controlled stabilization
             ↓
            page.content()
             ↓
            HTML validation
             ↓
            feature analysis

        Returns:
            Standardized dictionary:

            {
                "success": bool,
                "error": Optional[str],
                "data": {...}
            }
        """

        # --------------------------------------------------------------
        # Validate URL
        # --------------------------------------------------------------

        normalized_url = self._normalize_url(
            target_url
        )

        if not normalized_url:
            return self._build_error_payload(
                "Invalid or empty target URL."
            )

        # --------------------------------------------------------------
        # Check Playwright availability
        # --------------------------------------------------------------

        if not PLAYWRIGHT_AVAILABLE:

            logger.error(
                "Playwright is not installed."
            )

            return self._build_error_payload(
                "Playwright library missing from environment."
            )

        browser = None
        context = None
        page = None

        try:

            logger.info(
                "HTMLCollector starting browser for: %s",
                normalized_url
            )

            with sync_playwright() as playwright:

                # ======================================================
                # Launch Chromium
                # ======================================================

                browser = playwright.chromium.launch(

                    headless=True,

                    args=[
                        """--no-sandbox", # Removed for Phase 10: Sandboxing enabled"""
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-gpu",
                        "--disable-background-networking",
                        "--disable-background-timer-throttling",
                        "--disable-renderer-backgrounding"
                    ]
                )

                # ======================================================
                # Browser Context
                # ======================================================

                context = browser.new_context(

                    viewport={
                        "width": 1920,
                        "height": 1080
                    },

                    user_agent=self.USER_AGENT,

                    # Important for security research:
                    # invalid certificates should not stop collection.
                    ignore_https_errors=True,

                    java_script_enabled=True
                )

                # ======================================================
                # Default timeouts
                # ======================================================

                context.set_default_timeout(
                    self.timeout_ms
                )

                context.set_default_navigation_timeout(
                    self.timeout_ms
                )

                # ======================================================
                # Create Page
                # ======================================================

                page = context.new_page()

                # ------------------------------------------------------
                # SECURITY: SSRF & Resource Isolation
                # ------------------------------------------------------
                def _ssrf_route_handler(route):
                    import sys, os
                    if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from security.validator import SSRFValidator
                    
                    is_safe, _, msg = SSRFValidator.validate_url(route.request.url)
                    if not is_safe:
                        logger.warning(f"Playwright SSRF Blocked: {route.request.url} - {msg}")
                        route.abort("blockedbyclient")
                    else:
                        route.continue_()
                
                page.route("**/*", _ssrf_route_handler)

                # ------------------------------------------------------
                # Prevent unnecessary popup windows from interfering
                # ------------------------------------------------------

                page.on(
                    "popup",
                    self._handle_popup
                )

                # ------------------------------------------------------
                # Track console messages for debugging only
                # ------------------------------------------------------

                page.on(
                    "console",
                    lambda msg: logger.debug(
                        "Browser console [%s]: %s",
                        msg.type,
                        msg.text
                    )
                )

                # ======================================================
                # NAVIGATION
                # ======================================================

                logger.info(
                    "Navigating using DOMContentLoaded: %s",
                    normalized_url
                )

                response = None

                try:

                    response = page.goto(

                        normalized_url,

                        timeout=self.timeout_ms,

                        # ------------------------------------------------
                        # IMPORTANT FIX
                        # ------------------------------------------------
                        #
                        # Do NOT use:
                        #
                        #     wait_until="networkidle"
                        #
                        # because modern applications such as Google
                        # Forms can continue network activity indefinitely.
                        #
                        # DOMContentLoaded is sufficient to begin
                        # extracting the rendered document.
                        #
                        wait_until="domcontentloaded"
                    )

                except PlaywrightTimeoutError as navigation_error:

                    # --------------------------------------------------
                    # IMPORTANT:
                    #
                    # A navigation timeout does not necessarily mean
                    # there is no HTML.
                    #
                    # Some pages have already rendered useful content
                    # when the timeout occurs.
                    #
                    # Therefore we attempt page.content() before
                    # declaring collection failure.
                    # --------------------------------------------------

                    logger.warning(
                        "Navigation timeout for %s: %s",
                        normalized_url,
                        navigation_error
                    )

                    if page:

                        try:

                            partial_html = page.content()

                            if self._is_valid_html(
                                partial_html
                            ):

                                logger.info(
                                    "Using partially rendered HTML "
                                    "after navigation timeout."
                                )

                                final_url = page.url

                                status_code = (
                                    response.status
                                    if response
                                    else 0
                                )

                                return self._analyze_html(
                                    partial_html,
                                    final_url,
                                    status_code,
                                    collection_mode=(
                                        "partial_after_timeout"
                                    ),
                                    navigation_timeout=True
                                )

                        except Exception as content_error:

                            logger.debug(
                                "Unable to extract partial HTML "
                                "after navigation timeout: %s",
                                content_error
                            )

                    return self._build_error_payload(
                        (
                            "Browser navigation timed out before "
                            "usable HTML content could be collected."
                        )
                    )

                # ======================================================
                # RESPONSE INFORMATION
                # ======================================================

                status_code = (
                    response.status
                    if response
                    else 0
                )

                final_url = page.url

                logger.info(
                    "Navigation completed | "
                    "status=%s | final_url=%s",
                    status_code,
                    final_url
                )

                # ======================================================
                # CONTROLLED STABILIZATION
                # ======================================================

                self._controlled_stabilization(
                    page
                )

                # ======================================================
                # EXTRACT RENDERED DOM
                # ======================================================

                try:

                    rendered_html = page.content()

                except PlaywrightTimeoutError:

                    logger.warning(
                        "page.content() timed out."
                    )

                    return self._build_error_payload(
                        "Unable to extract rendered DOM content."
                    )

                except Exception as content_error:

                    logger.error(
                        "DOM extraction error: %s",
                        content_error
                    )

                    return self._build_error_payload(
                        f"DOM extraction failed: {content_error}"
                    )

                # ======================================================
                # VALIDATE HTML
                # ======================================================

                if not self._is_valid_html(
                    rendered_html
                ):

                    return self._build_error_payload(
                        "Rendered DOM content was empty or invalid."
                    )

                # ======================================================
                # ANALYZE HTML
                # ======================================================

                result = self._analyze_html(

                    rendered_html,

                    final_url,

                    status_code,

                    collection_mode="full",

                    navigation_timeout=False
                )

                return result

        except PlaywrightTimeoutError as timeout_error:

            logger.warning(
                "Playwright timeout while collecting HTML: %s",
                timeout_error
            )

            return self._build_error_payload(
                (
                    "Browser operation timed out while "
                    "collecting HTML content."
                )
            )

        except Exception as error:

            logger.exception(
                "Unexpected HTML collection error."
            )

            return self._build_error_payload(
                str(error)
            )

        finally:

            # ==========================================================
            # SAFE RESOURCE CLEANUP
            # ==========================================================

            try:

                if page:

                    page.close()

            except Exception:

                pass

            try:

                if context:

                    context.close()

            except Exception:

                pass

            try:

                if browser:

                    browser.close()

            except Exception:

                pass

    # ======================================================================
    # URL NORMALIZATION
    # ======================================================================

    @staticmethod
    def _normalize_url(
        target_url: str
    ) -> str:
        """
        Normalize and validate the target URL.
        """

        if not isinstance(
            target_url,
            str
        ):

            return ""

        target_url = target_url.strip()

        if not target_url:

            return ""

        if not target_url.startswith(
            (
                "http://",
                "https://"
            )
        ):

            target_url = (
                "https://"
                +
                target_url
            )

        try:

            parsed = urllib.parse.urlparse(
                target_url
            )

            if not parsed.netloc:

                return ""

            return target_url

        except Exception:

            return ""

    # ======================================================================
    # CONTROLLED PAGE STABILIZATION
    # ======================================================================

    def _controlled_stabilization(
        self,
        page
    ) -> None:
        """
        Allow dynamic JavaScript content to stabilize without waiting
        for network-idle.

        This is intentionally bounded.

        We do NOT wait indefinitely for all network requests to finish.
        """

        if self.stabilization_ms <= 0:

            return

        try:

            page.wait_for_timeout(
                self.stabilization_ms
            )

        except Exception as error:

            logger.debug(
                "Controlled stabilization wait failed: %s",
                error
            )

    # ======================================================================
    # POPUP HANDLER
    # ======================================================================

    @staticmethod
    def _handle_popup(
        popup
    ) -> None:
        """
        Close unexpected popup pages.

        The primary page remains the page being analyzed.
        """

        try:

            popup.close()

        except Exception:

            pass

    # ======================================================================
    # HTML VALIDATION
    # ======================================================================

    def _is_valid_html(
        self,
        html_content: Optional[str]
    ) -> bool:
        """
        Validate that collected content represents meaningful HTML.

        We intentionally do not require a large page.

        A legitimate minimal HTML document can be small.
        """

        if not html_content:

            return False

        if not isinstance(
            html_content,
            str
        ):

            return False

        cleaned = html_content.strip()

        if len(cleaned) < self.MIN_VALID_HTML_LENGTH:

            return False

        # --------------------------------------------------------------
        # Prevent pathological memory usage
        # --------------------------------------------------------------

        if len(cleaned) > self.MAX_HTML_LENGTH:

            logger.warning(
                "HTML content exceeds configured maximum."
            )

            return False

        # --------------------------------------------------------------
        # Basic DOM markers
        # --------------------------------------------------------------

        lowered = cleaned.lower()

        has_html_marker = (
            "<html" in lowered
            or
            "<body" in lowered
            or
            "<head" in lowered
            or
            "<form" in lowered
            or
            "<input" in lowered
        )

        return has_html_marker

    # ======================================================================
    # FAVICON
    # ======================================================================

    def _extract_favicon_url(
        self,
        soup: BeautifulSoup,
        base_url: str
    ) -> str:
        """
        Extract and resolve favicon URL.
        """

        try:

            icon_tag = soup.find(
                "link",
                rel=lambda r:
                    r
                    and
                    "icon"
                    in
                    str(r).lower()
            )

            if (
                icon_tag
                and
                icon_tag.get("href")
            ):

                raw_href = (
                    icon_tag
                    .get("href")
                    .strip()
                )

                return urllib.parse.urljoin(
                    base_url,
                    raw_href
                )

        except Exception as error:

            logger.debug(
                "Favicon parsing error: %s",
                error
            )

        try:

            parsed_url = urllib.parse.urlparse(
                base_url
            )

            if parsed_url.netloc:

                return (
                    f"{parsed_url.scheme}://"
                    f"{parsed_url.netloc}"
                    "/favicon.ico"
                )

        except Exception:

            pass

        return ""

    # ======================================================================
    # LINK ANALYSIS
    # ======================================================================

    def _analyze_links(
        self,
        soup: BeautifulSoup,
        base_domain: str
    ) -> Tuple[int, int, int]:
        """
        Analyze anchor links.
        """

        links = soup.find_all(
            "a",
            href=True
        )

        total_links = len(
            links
        )

        external_links = 0

        empty_links = 0

        normalized_base = (
            base_domain
            .lower()
            .split(":")[0]
        )

        for link in links:

            href = (
                link
                .get("href", "")
                .strip()
            )

            if href in {
                "",
                "#",
                "javascript:void(0)",
                "javascript:;"
            }:

                empty_links += 1

                continue

            try:

                parsed_href = urllib.parse.urlparse(
                    href
                )

                hostname = (
                    parsed_href.hostname
                    or ""
                ).lower()

                if (
                    hostname
                    and
                    hostname != normalized_base
                    and
                    not hostname.endswith(
                        "." + normalized_base
                    )
                ):

                    external_links += 1

            except Exception:

                pass

        return (
            total_links,
            external_links,
            empty_links
        )

    # ======================================================================
    # FORM ANALYSIS
    # ======================================================================

    def _analyze_forms(
        self,
        soup: BeautifulSoup,
        base_domain: str
    ) -> Dict[str, Any]:
        """
        Analyze form structures and destinations.
        """

        forms = soup.find_all(
            "form"
        )

        suspicious_actions = 0

        has_password_field = False

        for form in forms:

            if form.find(
                "input",
                type="password"
            ):

                has_password_field = True

            action = (
                form
                .get(
                    "action",
                    ""
                )
                .strip()
            )

            if not action:

                continue

            try:

                absolute_action = (
                    urllib.parse.urljoin(
                        "https://" + base_domain,
                        action
                    )
                )

                parsed_action = urllib.parse.urlparse(
                    absolute_action
                )

                action_domain = (
                    parsed_action.hostname
                    or ""
                ).lower()

                base_domain_normalized = (
                    base_domain
                    .lower()
                    .split(":")[0]
                )

                if (
                    action_domain
                    and
                    action_domain != base_domain_normalized
                    and
                    not action_domain.endswith(
                        "." + base_domain_normalized
                    )
                ):

                    suspicious_actions += 1

            except Exception:

                pass

        return {

            "form_count":
                len(forms),

            "has_password_field":
                has_password_field,

            "hidden_input_count":
                len(
                    soup.find_all(
                        "input",
                        type="hidden"
                    )
                ),

            "cross_domain_form_actions":
                suspicious_actions
        }

    # ======================================================================
    # MAIN HTML ANALYSIS
    # ======================================================================

    def _analyze_html(
        self,
        html_content: str,
        final_url: str,
        status_code: int,
        collection_mode: str = "full",
        navigation_timeout: bool = False
    ) -> Dict[str, Any]:
        """
        Parse HTML and extract structural features.
        """

        try:

            soup = BeautifulSoup(
                html_content,
                "html.parser"
            )

        except Exception as error:

            logger.error(
                "BeautifulSoup parsing failed: %s",
                error
            )

            return self._build_error_payload(
                f"HTML parsing failed: {error}"
            )

        parsed_final = urllib.parse.urlparse(
            final_url
        )

        base_domain = (
            parsed_final.hostname
            or
            parsed_final.netloc
            or
            ""
        )

        # ==============================================================
        # Forms
        # ==============================================================

        form_data = self._analyze_forms(
            soup,
            base_domain
        )

        # ==============================================================
        # Links
        # ==============================================================

        (
            total_links,
            external_links,
            empty_links
        ) = self._analyze_links(
            soup,
            base_domain
        )

        # ==============================================================
        # Iframes
        # ==============================================================

        iframes = soup.find_all(
            "iframe"
        )

        iframe_sources = [
            iframe.get("src")
            for iframe in iframes
            if iframe.get("src")
        ]

        invisible_iframes = 0

        for iframe in iframes:

            style = (
                iframe
                .get(
                    "style",
                    ""
                )
                .replace(
                    " ",
                    ""
                )
                .lower()
            )

            width = str(
                iframe.get(
                    "width",
                    ""
                )
            ).lower()

            height = str(
                iframe.get(
                    "height",
                    ""
                )).lower()

            if (
                "display:none" in style
                or
                "visibility:hidden" in style
                or
                width in {
                    "0",
                    "0px"
                }
                or
                height in {
                    "0",
                    "0px"
                }
            ):

                invisible_iframes += 1

        # ==============================================================
        # JavaScript
        # ==============================================================

        scripts = soup.find_all(
            "script"
        )

        external_scripts_count = 0

        inline_scripts = []

        for script in scripts:

            src = (
                script.get(
                    "src",
                    ""
                )
                .strip()
            )

            if src:

                external_scripts_count += 1

            if script.string:

                inline_scripts.append(
                    script.string
                )

        obfuscation_indicators = 0

        for script in inline_scripts:

            try:

                if any(
                    pattern.search(
                        script
                    )
                    for pattern
                    in self.compiled_obfuscation_patterns
                ):

                    obfuscation_indicators += 1

            except Exception:

                pass

        # ==============================================================
        # Evasion
        # ==============================================================

        body = soup.find(
            "body"
        )

        right_click_disabled = False

        if body:

            context_menu = (
                body.get(
                    "oncontextmenu",
                    ""
                )
                .lower()
            )

            if (
                "return false"
                in context_menu
            ):

                right_click_disabled = True

        hidden_elements = len(
            soup.find_all(
                style=re.compile(
                    r"display:\s*none|"
                    r"visibility:\s*hidden",
                    re.IGNORECASE
                )
            )
        )

        # ==============================================================
        # Meta refresh
        # ==============================================================

        meta_refresh = bool(
            soup.find(
                "meta",
                attrs={
                    "http-equiv":
                        re.compile(
                            r"refresh",
                            re.IGNORECASE
                        )
                }
            )
        )

        # ==============================================================
        # Title
        # ==============================================================

        page_title = None

        if soup.title:

            page_title = (
                soup.title
                .get_text(
                    strip=True
                )
                or None
            )

        # ==============================================================
        # Comments
        # ==============================================================

        html_comments = len(
            soup.find_all(
                string=lambda text:
                    isinstance(
                        text,
                        Comment
                    )
            )
        )

        # ==============================================================
        # Build response
        # ==============================================================

        return {

            "success":
                True,

            "error":
                None,

            "signal_available":
                True,

            "collection_mode":
                collection_mode,

            "navigation_timeout":
                navigation_timeout,

            "data": {

                "page_title":
                    page_title,

                "favicon_url":
                    self._extract_favicon_url(
                        soup,
                        final_url
                    ),

                "html_length_bytes":
                    len(
                        html_content
                    ),

                "status_code":
                    status_code,

                "final_url":
                    final_url,

                # ------------------------------------------------------
                # Raw HTML
                # ------------------------------------------------------

                "raw_html":
                    html_content,

                # ------------------------------------------------------
                # Forms
                # ------------------------------------------------------

                **form_data,

                # ------------------------------------------------------
                # Iframes
                # ------------------------------------------------------

                "iframe_count":
                    len(iframes),

                "invisible_iframes":
                    invisible_iframes,

                "iframe_sources":
                    iframe_sources,

                # ------------------------------------------------------
                # Scripts
                # ------------------------------------------------------

                "script_count":
                    len(scripts),

                "external_scripts_count":
                    external_scripts_count,

                "suspicious_inline_scripts":
                    obfuscation_indicators,

                # ------------------------------------------------------
                # Links
                # ------------------------------------------------------

                "total_links":
                    total_links,

                "external_links_count":
                    external_links,

                "empty_links_count":
                    empty_links,

                # ------------------------------------------------------
                # Evasion
                # ------------------------------------------------------

                "hidden_elements_count":
                    hidden_elements,

                "right_click_disabled":
                    right_click_disabled,

                "has_meta_refresh":
                    meta_refresh,

                "html_comments_count":
                    html_comments
            }
        }

    # ======================================================================
    # ERROR PAYLOAD
    # ======================================================================

    def _build_error_payload(
        self,
        message: str
    ) -> Dict[str, Any]:
        """
        Standardized failure payload.

        IMPORTANT:

        Failure does NOT produce legitimate evidence.

        signal_available = False
        data = {}
        """

        return {

            "success":
                False,

            "signal_available":
                False,

            "error":
                message,

            "collection_mode":
                "failed",

            "navigation_timeout":
                "timeout"
                in message.lower(),

            "data":
                {}
        }
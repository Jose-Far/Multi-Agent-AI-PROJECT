"""
HTML AI Agent - Feature Extractor
=================================

Extracts security-relevant features from raw HTML.

The extractor focuses on contextual phishing evidence rather than simplistic
rules.

Example:

    hidden_input_count = 23

does NOT automatically mean phishing.

Instead, the extractor distinguishes:

    - normal hidden fields
    - credential-related hidden fields
    - suspicious hidden fields
    - credential collection
    - payment collection
    - OTP collection
    - form destination
    - same-domain submission
    - cross-domain submission
    - subdomain relationships
    - registrable-domain relationships

This allows the ML model to learn combinations of evidence.
"""

import logging
import re
from typing import Dict, Any, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ============================================================================
# FINALIZED 50-FEATURE HTML MODEL SCHEMA
# ============================================================================

HTML_FEATURE_NAMES = [
    "html_length_bytes",
    "form_count",
    "forms_with_action",
    "empty_form_action_count",
    "password_field_count",
    "email_field_count",
    "username_field_count",
    "payment_field_count",
    "otp_field_count",
    "credential_field_count",
    "has_password_field",
    "cross_domain_form_actions",
    "same_domain_form_actions",
    "http_form_actions",
    "https_form_actions",
    "external_form_action_ratio",
    "subdomain_match_count",
    "registrable_domain_match_count",
    "cross_domain_count",
    "hidden_input_count",
    "credential_hidden_input_count",
    "suspicious_hidden_input_count",
    "hidden_input_ratio",
    "iframe_count",
    "invisible_iframes",
    "script_count",
    "external_scripts_count",
    "suspicious_inline_scripts",
    "total_links",
    "external_links_count",
    "empty_links_count",
    "hidden_elements_count",
    "right_click_disabled",
    "has_meta_refresh",
    "html_comments_count",
    "title_length",
    "visible_text_length",
    "heading_count",
    "button_count",
    "input_count",
    "image_count",
    "anchor_count",
    "login_keyword_count",
    "credential_keyword_count",
    "urgency_keyword_count",
    "security_keyword_count",
    "eval_count",
    "document_write_count",
    "javascript_redirect_count",
    "external_images_count",
]

EXPECTED_FEATURE_COUNT = 50

LOGIN_KEYWORDS = {
    "login",
    "log in",
    "signin",
    "sign in",
    "sign-in",
    "authenticate",
    "authentication",
}

URGENCY_KEYWORDS = {
    "urgent",
    "urgently",
    "immediately",
    "immediate",
    "act now",
    "verify now",
    "limited time",
    "expires",
    "expired",
    "suspended",
    "suspend",
    "warning",
}

SECURITY_KEYWORDS = {
    "security",
    "secure",
    "verification",
    "verify",
    "protected",
    "protection",
    "fraud",
    "alert",
    "identity",
    "authentication",
}


class HTMLFeatureExtractor:
    """
    Extract security-oriented features from HTML.

    Parameters
    ----------
    base_url:
        URL of the page being analyzed.

    Example
    -------
    extractor = HTMLFeatureExtractor(
        base_url="https://login.example.com/login"
    )

    features = extractor.extract(html_content)
    """

    # =====================================================================
    # FIELD KEYWORDS
    # =====================================================================

    PASSWORD_KEYWORDS: Set[str] = {
        "password",
        "passwd",
        "pass",
        "pwd",
    }

    USERNAME_KEYWORDS: Set[str] = {
        "username",
        "user_name",
        "user-name",
        "userid",
        "user_id",
        "user-id",
        "user",
        "login",
        "login_id",
        "loginid",
    }

    EMAIL_KEYWORDS: Set[str] = {
        "email",
        "e-mail",
        "mail",
        "email_address",
        "emailaddress",
    }

    PAYMENT_KEYWORDS: Set[str] = {
        "card",
        "card_number",
        "cardnumber",
        "credit_card",
        "creditcard",
        "debit_card",
        "debitcard",
        "cvv",
        "cvc",
        "security_code",
        "securitycode",
        "expiry",
        "expiration",
        "expiration_date",
        "exp_date",
        "cardholder",
        "card_holder",
    }

    OTP_KEYWORDS: Set[str] = {
        "otp",
        "one_time_password",
        "one-time-password",
        "verification_code",
        "verificationcode",
        "security_code",
        "securitycode",
        "auth_code",
        "authcode",
        "authentication_code",
        "pin",
    }

    CREDENTIAL_KEYWORDS: Set[str] = (
        PASSWORD_KEYWORDS
        | USERNAME_KEYWORDS
        | EMAIL_KEYWORDS
        | PAYMENT_KEYWORDS
        | OTP_KEYWORDS
    )

    # =====================================================================
    # LEGITIMATE HIDDEN FIELD KEYWORDS
    # =====================================================================

    LEGITIMATE_HIDDEN_KEYWORDS: Set[str] = {
        "csrf",
        "csrf_token",
        "xsrf",
        "xsrf_token",
        "session",
        "session_id",
        "sessionid",
        "nonce",
        "form_token",
        "form_id",
        "timestamp",
        "redirect",
        "return_url",
        "returnurl",
        "language",
        "locale",
        "page",
        "action",
        "wpnonce",
        "_wpnonce",
    }

    # =====================================================================
    # SUSPICIOUS JAVASCRIPT PATTERNS
    # =====================================================================

    SUSPICIOUS_SCRIPT_PATTERNS = [
        r"\beval\s*\(",
        r"\bunescape\s*\(",
        r"\batob\s*\(",
        r"\bbtoa\s*\(",
        r"document\.write\s*\(",
        r"fromcharcode",
        r"\bwindow\.location\s*=",
        r"\blocation\.href\s*=",
        r"location\.replace",
        r"location\.assign",
    ]

    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    def __init__(
        self,
        base_url: str = ""
    ):
        """
        Initialize the HTML feature extractor.

        Parameters
        ----------
        base_url:
            URL of the page from which the HTML was collected.
        """

        self.base_url = str(
            base_url or ""
        ).strip()

        self.page_domain = self._extract_hostname(
            self.base_url
        )

        self.registrable_page_domain = (
            self._get_registrable_domain(
                self.page_domain
            )
        )

    # =====================================================================
    # MAIN EXTRACTION METHOD
    # =====================================================================

    def extract(
        self,
        html_content: str
    ) -> Dict[str, Any]:
        """
        Extract the complete HTML feature vector.

        Parameters
        ----------
        html_content:
            Raw HTML source.

        Returns
        -------
        Dict[str, Any]
            HTML feature dictionary.
        """

        if not html_content:

            logger.warning(
                "HTMLFeatureExtractor received empty HTML."
            )

            return self._get_empty_features()

        try:

            html_content = str(
                html_content
            )

            soup = BeautifulSoup(
                html_content,
                "html.parser"
            )

            features = {
                "html_length_bytes": len(
                    html_content.encode(
                        "utf-8",
                        errors="ignore"
                    )
                ),

                # ---------------------------------------------------------
                # Domain metadata
                #
                # These are retained for explanation/reporting and are
                # deliberately NOT part of the numerical ML schema.
                # ---------------------------------------------------------

                "page_domain": self.page_domain,

                "registrable_page_domain":
                    self.registrable_page_domain,

                "form_action_domains": [],
            }

            # -------------------------------------------------------------
            # Form analysis
            # -------------------------------------------------------------

            form_features = (
                self._extract_form_features(
                    soup
                )
            )

            features.update(
                form_features
            )

            # -------------------------------------------------------------
            # Hidden input analysis
            # -------------------------------------------------------------

            features.update(
                self._analyze_hidden_inputs(
                    soup
                )
            )

            # -------------------------------------------------------------
            # Iframes
            # -------------------------------------------------------------

            features.update(
                self._extract_iframe_features(
                    soup
                )
            )

            # -------------------------------------------------------------
            # JavaScript
            # -------------------------------------------------------------

            features.update(
                self._extract_script_features(
                    soup
                )
            )

            # -------------------------------------------------------------
            # Links
            # -------------------------------------------------------------

            features.update(
                self._extract_link_features(
                    soup
                )
            )

            # -------------------------------------------------------------
            # DOM/evasion
            # -------------------------------------------------------------

            features.update(
                self._extract_dom_features(
                    soup
                )
            )

            # -------------------------------------------------------------
            # Backward compatibility
            # -------------------------------------------------------------

            features["has_password_field"] = (
                features.get(
                    "password_field_count",
                    0
                ) > 0
            )

            # -------------------------------------------------------------
            # FINAL PRODUCTION MODEL VECTOR
            #
            # The extractor historically returned 35 ML features plus
            # three reporting metadata fields. The tuned XGBoost model
            # requires exactly 50 numerical features.
            # -------------------------------------------------------------

            final_features = (
                self._finalize_50_features(
                    features,
                    soup
                )
            )

            logger.info(
                "HTMLFeatureExtractor produced %d/%d "
                "production features.",
                len(final_features),
                EXPECTED_FEATURE_COUNT
            )

            return final_features

        except Exception as e:

            logger.error(
                "HTML feature extraction failed: %s",
                str(e),
                exc_info=True
            )

            return self._get_empty_features()

    # =====================================================================
    # FORM ANALYSIS
    # =====================================================================

    def _extract_form_features(
        self,
        soup: BeautifulSoup
    ) -> Dict[str, Any]:
        """
        Analyze HTML forms and their input fields.

        This method specifically captures what the form is attempting
        to collect and where the collected information is being sent.
        """

        forms = soup.find_all(
            "form"
        )

        form_count = len(
            forms
        )

        # -----------------------------------------------------------------
        # Credential counters
        # -----------------------------------------------------------------

        password_field_count = 0
        email_field_count = 0
        username_field_count = 0

        payment_field_count = 0
        otp_field_count = 0

        credential_field_count = 0

        # -----------------------------------------------------------------
        # Form action counters
        # -----------------------------------------------------------------

        forms_with_action = 0
        empty_form_action_count = 0

        cross_domain_form_actions = 0
        same_domain_form_actions = 0

        http_form_actions = 0
        https_form_actions = 0

        # -----------------------------------------------------------------
        # Domain relationship counters
        # -----------------------------------------------------------------

        subdomain_match_count = 0
        registrable_domain_match_count = 0

        cross_domain_count = 0

        # -----------------------------------------------------------------
        # Metadata for reporting/explanation
        # -----------------------------------------------------------------

        form_action_domains = []

        for form in forms:

            # =============================================================
            # Resolve form action
            # =============================================================

            raw_action = str(
                form.get(
                    "action",
                    ""
                )
            ).strip()

            if raw_action:

                forms_with_action += 1

            else:

                empty_form_action_count += 1

            # -------------------------------------------------------------
            # Resolve relative action URLs against the page URL.
            # -------------------------------------------------------------

            resolved_action = self._resolve_url(
                raw_action
            )

            action_domain = self._extract_hostname(
                resolved_action
            )

            action_registrable_domain = (
                self._get_registrable_domain(
                    action_domain
                )
            )

            # Save domain information for explanation/reporting.
            form_action_domains.append(
                {
                    "raw_action": raw_action,
                    "resolved_action": resolved_action,
                    "action_domain": action_domain,
                    "page_domain": self.page_domain,
                }
            )

            # =============================================================
            # Form destination analysis
            # =============================================================

            if action_domain:

                # ---------------------------------------------------------
                # Protocol
                # ---------------------------------------------------------

                action_scheme = (
                    urlparse(
                        resolved_action
                    )
                    .scheme
                    .lower()
                )

                if action_scheme == "http":

                    http_form_actions += 1

                elif action_scheme == "https":

                    https_form_actions += 1

                # ---------------------------------------------------------
                # Exact hostname comparison
                # ---------------------------------------------------------

                same_domain = (
                    bool(self.page_domain)
                    and
                    action_domain == self.page_domain
                )

                if same_domain:

                    same_domain_form_actions += 1

                else:

                    cross_domain_form_actions += 1
                    cross_domain_count += 1

                # ---------------------------------------------------------
                # Subdomain relationship
                #
                # Example:
                #
                # page:
                # login.microsoft.com
                #
                # action:
                # account.microsoft.com
                #
                # exact domain differs,
                # but both belong to microsoft.com.
                # ---------------------------------------------------------

                if (
                    self.page_domain
                    and
                    action_domain != self.page_domain
                    and
                    self._is_subdomain_match(
                        self.page_domain,
                        action_domain
                    )
                ):

                    subdomain_match_count += 1

                # ---------------------------------------------------------
                # Registrable domain relationship
                #
                # Example:
                #
                # login.microsoft.com
                # account.microsoft.com
                #
                # registrable domain:
                # microsoft.com
                # ---------------------------------------------------------

                if (
                    self.registrable_page_domain
                    and
                    action_registrable_domain
                    and
                    action_registrable_domain
                    ==
                    self.registrable_page_domain
                ):

                    registrable_domain_match_count += 1

            # =============================================================
            # Analyze form input fields
            # =============================================================

            inputs = form.find_all(
                "input"
            )

            # Include textarea/select because modern forms may collect
            # credentials through them as well.
            textareas = form.find_all(
                "textarea"
            )

            selects = form.find_all(
                "select"
            )

            for input_element in inputs:

                input_type = str(
                    input_element.get(
                        "type",
                        "text"
                    )
                ).lower()

                name = str(
                    input_element.get(
                        "name",
                        ""
                    )
                ).lower()

                input_id = str(
                    input_element.get(
                        "id",
                        ""
                    )
                ).lower()

                placeholder = str(
                    input_element.get(
                        "placeholder",
                        ""
                    )
                ).lower()

                autocomplete = str(
                    input_element.get(
                        "autocomplete",
                        ""
                    )
                ).lower()

                aria_label = str(
                    input_element.get(
                        "aria-label",
                        ""
                    )
                ).lower()

                combined = " ".join(
                    [
                        name,
                        input_id,
                        placeholder,
                        autocomplete,
                        aria_label,
                    ]
                )

                # ---------------------------------------------------------
                # Password
                # ---------------------------------------------------------

                is_password = (
                    input_type == "password"
                    or
                    self._contains_keyword(
                        combined,
                        self.PASSWORD_KEYWORDS
                    )
                )

                if is_password:

                    password_field_count += 1

                    credential_field_count += 1

                # ---------------------------------------------------------
                # Email
                # ---------------------------------------------------------

                is_email = (
                    input_type == "email"
                    or
                    self._contains_keyword(
                        combined,
                        self.EMAIL_KEYWORDS
                    )
                )

                if is_email:

                    email_field_count += 1

                    credential_field_count += 1

                # ---------------------------------------------------------
                # Username
                # ---------------------------------------------------------

                is_username = (
                    self._contains_keyword(
                        combined,
                        self.USERNAME_KEYWORDS
                    )
                )

                if is_username:

                    username_field_count += 1

                    credential_field_count += 1

                # ---------------------------------------------------------
                # Payment
                # ---------------------------------------------------------

                is_payment = (
                    self._contains_keyword(
                        combined,
                        self.PAYMENT_KEYWORDS
                    )
                )

                if is_payment:

                    payment_field_count += 1

                    credential_field_count += 1

                # ---------------------------------------------------------
                # OTP
                # ---------------------------------------------------------

                is_otp = (
                    self._contains_keyword(
                        combined,
                        self.OTP_KEYWORDS
                    )
                )

                if is_otp:

                    otp_field_count += 1

                    credential_field_count += 1

            # -------------------------------------------------------------
            # Textareas
            # -------------------------------------------------------------

            for textarea in textareas:

                combined = " ".join(
                    [
                        str(
                            textarea.get(
                                "name",
                                ""
                            )
                        ).lower(),

                        str(
                            textarea.get(
                                "id",
                                ""
                            )
                        ).lower(),

                        str(
                            textarea.get(
                                "placeholder",
                                ""
                            )
                        ).lower(),
                    ]
                )

                if self._contains_keyword(
                    combined,
                    self.CREDENTIAL_KEYWORDS
                ):

                    credential_field_count += 1

            # -------------------------------------------------------------
            # Select fields
            # -------------------------------------------------------------

            for select in selects:

                combined = " ".join(
                    [
                        str(
                            select.get(
                                "name",
                                ""
                            )
                        ).lower(),

                        str(
                            select.get(
                                "id",
                                ""
                            )
                        ).lower(),
                    ]
                )

                if self._contains_keyword(
                    combined,
                    self.PAYMENT_KEYWORDS
                ):

                    payment_field_count += 1

                    credential_field_count += 1

        # =================================================================
        # FORM RATIOS
        # =================================================================

        if form_count > 0:

            external_form_action_ratio = (
                cross_domain_form_actions /
                form_count
            )

        else:

            external_form_action_ratio = 0.0

        return {

            # -------------------------------------------------------------
            # Form counts
            # -------------------------------------------------------------

            "form_count":
                form_count,

            "forms_with_action":
                forms_with_action,

            "empty_form_action_count":
                empty_form_action_count,

            # -------------------------------------------------------------
            # Credential intelligence
            # -------------------------------------------------------------

            "password_field_count":
                password_field_count,

            "email_field_count":
                email_field_count,

            "username_field_count":
                username_field_count,

            "payment_field_count":
                payment_field_count,

            "otp_field_count":
                otp_field_count,

            "credential_field_count":
                credential_field_count,

            # -------------------------------------------------------------
            # Destination intelligence
            # -------------------------------------------------------------

            "cross_domain_form_actions":
                cross_domain_form_actions,

            "same_domain_form_actions":
                same_domain_form_actions,

            "http_form_actions":
                http_form_actions,

            "https_form_actions":
                https_form_actions,

            "external_form_action_ratio":
                round(
                    external_form_action_ratio,
                    4
                ),

            # -------------------------------------------------------------
            # Domain relationship features
            # -------------------------------------------------------------

            "subdomain_match_count":
                subdomain_match_count,

            "registrable_domain_match_count":
                registrable_domain_match_count,

            "cross_domain_count":
                cross_domain_count,

            # -------------------------------------------------------------
            # Reporting metadata
            # -------------------------------------------------------------

            "form_action_domains":
                form_action_domains,
        }

    # =====================================================================
    # HIDDEN INPUT ANALYSIS
    # =====================================================================

    def _analyze_hidden_inputs(
        self,
        soup: BeautifulSoup
    ) -> Dict[str, Any]:
        """
        Analyze hidden inputs without treating all hidden inputs as malicious.
        """

        hidden_inputs = soup.find_all(
            "input",
            attrs={
                "type": re.compile(
                    r"^\s*hidden\s*$",
                    re.IGNORECASE
                )
            }
        )

        hidden_input_count = len(
            hidden_inputs
        )

        credential_hidden_input_count = 0
        suspicious_hidden_input_count = 0

        for element in hidden_inputs:

            name = str(
                element.get(
                    "name",
                    ""
                )
            ).lower()

            element_id = str(
                element.get(
                    "id",
                    ""
                )
            ).lower()

            value = str(
                element.get(
                    "value",
                    ""
                )
            ).strip()

            combined = (
                f"{name} {element_id}"
            )

            # -------------------------------------------------------------
            # Credential-related hidden field
            # -------------------------------------------------------------

            if self._contains_keyword(
                combined,
                self.CREDENTIAL_KEYWORDS
            ):

                credential_hidden_input_count += 1

                strong_terms = {
                    "password",
                    "passwd",
                    "pwd",
                    "cvv",
                    "cvc",
                    "card_number",
                    "otp",
                    "pin",
                    "secret",
                    "access_token",
                    "auth_token",
                }

                if any(
                    term in combined
                    for term in strong_terms
                ):

                    suspicious_hidden_input_count += 1

                continue

            # -------------------------------------------------------------
            # Legitimate security/session fields
            # -------------------------------------------------------------

            if self._contains_keyword(
                combined,
                self.LEGITIMATE_HIDDEN_KEYWORDS
            ):

                continue

            # -------------------------------------------------------------
            # Conservative obfuscation detection
            # -------------------------------------------------------------

            if self._looks_obfuscated(
                name,
                value
            ):

                suspicious_hidden_input_count += 1

        total_inputs = len(
            soup.find_all(
                "input"
            )
        )

        if total_inputs > 0:

            hidden_input_ratio = (
                hidden_input_count /
                total_inputs
            )

        else:

            hidden_input_ratio = 0.0

        return {

            "hidden_input_count":
                hidden_input_count,

            "credential_hidden_input_count":
                credential_hidden_input_count,

            "suspicious_hidden_input_count":
                suspicious_hidden_input_count,

            "hidden_input_ratio":
                round(
                    hidden_input_ratio,
                    4
                )
        }

    # =====================================================================
    # IFRAME FEATURES
    # =====================================================================

    def _extract_iframe_features(
        self,
        soup: BeautifulSoup
    ) -> Dict[str, Any]:

        iframes = soup.find_all(
            "iframe"
        )

        invisible_count = 0

        for iframe in iframes:

            style = str(
                iframe.get(
                    "style",
                    ""
                )
            ).lower()

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
                )
            ).lower()

            if (
                "display:none" in style
                or
                "visibility:hidden" in style
                or
                width == "0"
                or
                height == "0"
            ):

                invisible_count += 1

        return {

            "iframe_count":
                len(iframes),

            "invisible_iframes":
                invisible_count
        }

    # =====================================================================
    # SCRIPT FEATURES
    # =====================================================================

    def _extract_script_features(
        self,
        soup: BeautifulSoup
    ) -> Dict[str, Any]:

        scripts = soup.find_all(
            "script"
        )

        external_scripts_count = 0
        suspicious_inline_scripts = 0

        for script in scripts:

            src = script.get(
                "src"
            )

            if src:

                external_scripts_count += 1

            else:

                script_text = script.get_text(
                    " ",
                    strip=True
                )

                if any(
                    re.search(
                        pattern,
                        script_text,
                        re.IGNORECASE
                    )
                    for pattern
                    in self.SUSPICIOUS_SCRIPT_PATTERNS
                ):

                    suspicious_inline_scripts += 1

        return {

            "script_count":
                len(scripts),

            "external_scripts_count":
                external_scripts_count,

            "suspicious_inline_scripts":
                suspicious_inline_scripts
        }

    # =====================================================================
    # LINK FEATURES
    # =====================================================================

    def _extract_link_features(
        self,
        soup: BeautifulSoup
    ) -> Dict[str, Any]:

        links = soup.find_all(
            "a"
        )

        external_links_count = 0
        empty_links_count = 0

        for link in links:

            href = str(
                link.get(
                    "href",
                    ""
                )
            ).strip()

            if (
                not href
                or
                href == "#"
                or
                href.lower().startswith(
                    "javascript:void"
                )
            ):

                empty_links_count += 1

            resolved_url = self._resolve_url(
                href
            )

            link_domain = self._extract_hostname(
                resolved_url
            )

            if (
                link_domain
                and
                self.page_domain
                and
                link_domain != self.page_domain
            ):

                external_links_count += 1

        return {

            "total_links":
                len(links),

            "external_links_count":
                external_links_count,

            "empty_links_count":
                empty_links_count
        }

    # =====================================================================
    # DOM FEATURES
    # =====================================================================

    def _extract_dom_features(
        self,
        soup: BeautifulSoup
    ) -> Dict[str, Any]:

        hidden_elements_count = 0

        for element in soup.find_all(
            style=True
        ):

            style = str(
                element.get(
                    "style",
                    ""
                )
            ).lower()

            if (
                "display:none" in style
                or
                "visibility:hidden" in style
            ):

                hidden_elements_count += 1

        html_text = str(
            soup
        ).lower()

        right_click_disabled = (
            "oncontextmenu" in html_text
            and
            "return false" in html_text
        )

        meta_refresh = soup.find(
            "meta",
            attrs={
                "http-equiv": re.compile(
                    r"refresh",
                    re.IGNORECASE
                )
            }
        )

        # BeautifulSoup comments are represented by Comment objects.
        from bs4 import Comment

        comment_count = len(
            soup.find_all(
                string=lambda text:
                isinstance(
                    text,
                    Comment
                )
            )
        )

        return {

            "hidden_elements_count":
                hidden_elements_count,

            "right_click_disabled":
                right_click_disabled,

            "has_meta_refresh":
                meta_refresh is not None,

            "html_comments_count":
                comment_count
        }

    # =====================================================================
    # URL RESOLUTION
    # =====================================================================

    def _resolve_url(
        self,
        url: str
    ) -> str:
        """
        Resolve relative form/link URLs against the page URL.
        """

        if not url:

            return self.base_url

        try:

            return urljoin(
                self.base_url,
                url
            )

        except Exception:

            return url

    # =====================================================================
    # HOSTNAME EXTRACTION
    # =====================================================================

    @staticmethod
    def _extract_hostname(
        url: str
    ) -> str:

        if not url:

            return ""

        try:

            parsed = urlparse(
                url
            )

            hostname = (
                parsed.hostname
                or ""
            )

            return hostname.lower().strip()

        except Exception:

            return ""

    # =====================================================================
    # REGISTRABLE DOMAIN
    # =====================================================================

    @staticmethod
    def _get_registrable_domain(
        hostname: str
    ) -> str:
        """
        Obtain an approximate registrable domain.

        For production-grade public suffix handling, install:

            pip install tldextract

        and replace this helper with tldextract.
        """

        if not hostname:

            return ""

        parts = hostname.split(
            "."
        )

        if len(parts) < 2:

            return hostname

        # Basic fallback:
        #
        # login.microsoft.com
        #        ↓
        # microsoft.com
        #
        return ".".join(
            parts[-2:]
        )

    # =====================================================================
    # SUBDOMAIN RELATIONSHIP
    # =====================================================================

    @staticmethod
    def _is_subdomain_match(
        page_domain: str,
        action_domain: str
    ) -> bool:
        """
        Determine whether action_domain is a subdomain relationship
        with page_domain.

        Example:

            login.example.com
            account.example.com

        Both belong to example.com.
        """

        if not page_domain or not action_domain:

            return False

        if page_domain == action_domain:

            return False

        return (
            action_domain.endswith(
                "." + page_domain
            )
            or
            page_domain.endswith(
                "." + action_domain
            )
        )

    # =====================================================================
    # KEYWORD MATCHING
    # =====================================================================

    @staticmethod
    def _contains_keyword(
        text: str,
        keywords: Set[str]
    ) -> bool:

        text = str(
            text or ""
        ).lower()

        return any(
            keyword in text
            for keyword in keywords
        )

    # =====================================================================
    # OBFUSCATION CHECK
    # =====================================================================

    @staticmethod
    def _looks_obfuscated(
        name: str,
        value: str
    ) -> bool:

        if len(name) >= 80:

            return True

        if len(value) >= 150:

            if re.fullmatch(
                r"[A-Za-z0-9+/=_-]+",
                value
            ):

                return True

        return False

    # =====================================================================
    # FINALIZED 50-FEATURE DOM FALLBACKS
    # =====================================================================

    @staticmethod
    def _count_keyword_occurrences(
        text: str,
        keywords: Set[str]
    ) -> int:
        text = str(text or "").lower()
        return sum(text.count(keyword) for keyword in keywords)

    def _extract_final_dom_features(
        self,
        soup: BeautifulSoup
    ) -> Dict[str, Any]:
        """
        Calculate the 15 features that were missing from the previous
        38-value extractor.

        Existing specialized extractor values remain authoritative.
        These values are used only for features that the specialized
        extractors do not provide.
        """

        # ---------------------------------------------------------------
        # Visible text
        # ---------------------------------------------------------------

        text_soup = BeautifulSoup(
            str(soup),
            "html.parser"
        )

        for element in text_soup.find_all(
            ["script", "style", "noscript", "template"]
        ):
            element.decompose()

        visible_text = text_soup.get_text(
            separator=" ",
            strip=True
        )

        # ---------------------------------------------------------------
        # Title
        # ---------------------------------------------------------------

        title_text = ""

        if soup.title:
            title_text = soup.title.get_text(
                separator=" ",
                strip=True
            )

        # ---------------------------------------------------------------
        # Basic DOM counts
        # ---------------------------------------------------------------

        heading_count = len(
            soup.find_all(
                ["h1", "h2", "h3", "h4", "h5", "h6"]
            )
        )

        button_count = len(
            soup.find_all("button")
        )

        for element in soup.find_all("input"):
            input_type = str(
                element.get("type", "")
            ).strip().lower()

            if input_type in {
                "button",
                "submit",
                "reset",
            }:
                button_count += 1

        input_count = len(
            soup.find_all("input")
        )

        image_count = len(
            soup.find_all("img")
        )

        anchor_count = len(
            soup.find_all("a")
        )

        # ---------------------------------------------------------------
        # JavaScript source/text
        # ---------------------------------------------------------------

        script_text_parts = []

        for script in soup.find_all("script"):
            script_text_parts.append(
                script.get_text(
                    separator=" ",
                    strip=True
                )
            )

        script_text = " ".join(
            script_text_parts
        )

        script_lower = script_text.lower()

        eval_count = len(
            re.findall(
                r"\beval\s*\(",
                script_lower
            )
        )

        document_write_count = len(
            re.findall(
                r"\bdocument\.write\s*\(",
                script_lower
            )
        )

        javascript_redirect_count = 0

        redirect_patterns = [
            # Count navigation operations, not harmless references to a
            # location object in application code or documentation strings.
            r"\bwindow\.location\s*=",
            r"\blocation\.href\s*=",
            r"\blocation\.replace\s*\(",
            r"\blocation\.assign\s*\(",
            r"\bdocument\.location\s*=",
        ]

        for pattern in redirect_patterns:
            javascript_redirect_count += len(
                re.findall(
                    pattern,
                    script_lower
                )
            )

        # ---------------------------------------------------------------
        # Keyword counts
        # ---------------------------------------------------------------

        # Keyword features describe user-visible page content. JavaScript
        # source contains implementation terms such as "password", "verify"
        # and "security" on many legitimate applications and must not be
        # treated as displayed phishing copy.
        combined_text = (
            visible_text
            + " "
            + title_text
        ).lower()

        login_keyword_count = (
            self._count_keyword_occurrences(
                combined_text,
                LOGIN_KEYWORDS
            )
        )

        credential_keyword_count = (
            self._count_keyword_occurrences(
                combined_text,
                self.CREDENTIAL_KEYWORDS
            )
        )

        urgency_keyword_count = (
            self._count_keyword_occurrences(
                combined_text,
                URGENCY_KEYWORDS
            )
        )

        security_keyword_count = (
            self._count_keyword_occurrences(
                combined_text,
                SECURITY_KEYWORDS
            )
        )

        # ---------------------------------------------------------------
        # External image count
        # ---------------------------------------------------------------

        external_images_count = 0

        for image in soup.find_all("img"):

            source = (
                image.get("src")
                or image.get("data-src")
                or image.get("data-lazy-src")
                or ""
            )

            if not source:
                continue

            try:
                resolved = self._resolve_url(
                    source
                )

                image_domain = (
                    self._extract_hostname(
                        resolved
                    )
                )

                if (
                    image_domain
                    and self.page_domain
                    and image_domain != self.page_domain
                ):
                    external_images_count += 1

            except Exception:
                continue

        return {
            "title_length": len(title_text),
            "visible_text_length": len(visible_text),
            "heading_count": heading_count,
            "button_count": button_count,
            "input_count": input_count,
            "image_count": image_count,
            "anchor_count": anchor_count,
            "login_keyword_count": login_keyword_count,
            "credential_keyword_count": credential_keyword_count,
            "urgency_keyword_count": urgency_keyword_count,
            "security_keyword_count": security_keyword_count,
            "eval_count": eval_count,
            "document_write_count": document_write_count,
            "javascript_redirect_count": javascript_redirect_count,
            "external_images_count": external_images_count,
        }

    def _finalize_50_features(
        self,
        features: Dict[str, Any],
        soup: BeautifulSoup
    ) -> Dict[str, Any]:
        """
        Produce exactly the 50 numerical model features.

        Reporting metadata such as page_domain and form_action_domains
        is deliberately excluded from the ML vector.
        """

        dom_features = (
            self._extract_final_dom_features(
                soup
            )
        )

        # Existing specialized features have priority.
        merged = dict(dom_features)
        merged.update(features)

        # Explicit backward-compatible password flag.
        merged["has_password_field"] = int(
            merged.get(
                "password_field_count",
                0
            ) > 0
        )

        final_features = {}

        for feature_name in HTML_FEATURE_NAMES:

            value = merged.get(
                feature_name,
                0
            )

            if isinstance(value, bool):
                value = int(value)

            if value is None:
                value = 0

            final_features[
                feature_name
            ] = value

        if len(final_features) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "HTML feature schema failure: "
                f"expected {EXPECTED_FEATURE_COUNT}, "
                f"found {len(final_features)}"
            )

        return final_features

    # =====================================================================
    # FALLBACK FEATURE VECTOR
    # =====================================================================

    def _get_empty_features(
        self
    ) -> Dict[str, Any]:
        """
        Return a schema-correct empty 50-feature vector.

        This is intentionally numeric-only so it can never be mistaken
        for a valid partial ML vector.
        """

        return {
            feature_name: 0
            for feature_name in HTML_FEATURE_NAMES
        }


__all__ = [
    "HTMLFeatureExtractor",
    "HTML_FEATURE_NAMES",
    "EXPECTED_FEATURE_COUNT",
]

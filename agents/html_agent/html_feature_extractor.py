from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse

import tldextract
from bs4 import BeautifulSoup


# ============================================================
# HTML FEATURE SCHEMA
# ============================================================
#
# IMPORTANT:
# This is the authoritative order of HTML ML features.
#
# Original features:
#     01 - 35
#
# New content/behavior features:
#     36 - 50
#
# Total:
#     50
# ============================================================

HTML_FEATURES = [
    # ========================================================
    # ORIGINAL 35 FEATURES
    # ========================================================

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

    # ========================================================
    # NEW 15 FEATURES
    # ========================================================

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


# ============================================================
# KEYWORD DEFINITIONS
# ============================================================

EMAIL_KEYWORDS = {
    "email",
    "e-mail",
    "email_address",
    "emailaddress",
    "user_email",
    "useremail",
}


USERNAME_KEYWORDS = {
    "username",
    "user_name",
    "userid",
    "user_id",
    "user",
    "login",
    "login_id",
    "loginid",
    "account_name",
    "accountname",
}


PAYMENT_KEYWORDS = {
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
    "expiry_date",
    "account_number",
    "accountnumber",
    "routing_number",
    "routingnumber",
    "iban",
    "swift",
}


OTP_KEYWORDS = {
    "otp",
    "one_time_password",
    "one-time-password",
    "one time password",
    "verification_code",
    "verificationcode",
    "verify_code",
    "verification",
    "auth_code",
    "authcode",
    "authentication_code",
    "security_code",
    "2fa",
    "mfa",
    "two_factor",
    "two-factor",
}


CREDENTIAL_KEYWORDS = {
    "username",
    "user_name",
    "userid",
    "user_id",
    "user",
    "login",
    "login_id",
    "loginid",
    "email",
    "e-mail",
    "email_address",
    "emailaddress",
    "user_email",
    "password",
    "passwd",
    "pass",
    "credential",
    "credentials",
}


SUSPICIOUS_HIDDEN_KEYWORDS = {
    "password",
    "passwd",
    "pass",
    "username",
    "user",
    "email",
    "credential",
    "credentials",
    "login",
    "auth",
    "authentication",
    "token",
    "session",
    "sessionid",
    "session_id",
    "redirect",
    "redirect_url",
    "redirecturl",
    "verification",
    "verify",
    "otp",
    "mfa",
    "2fa",
}


# ============================================================
# NEW CONTENT KEYWORDS
# ============================================================

LOGIN_TEXT_KEYWORDS = [
    "login",
    "log in",
    "signin",
    "sign in",
    "sign-in",
    "authenticate",
    "authentication",
]


CREDENTIAL_TEXT_KEYWORDS = [
    "username",
    "user name",
    "userid",
    "user id",
    "password",
    "passcode",
    "credential",
    "credentials",
    "email address",
    "card number",
    "credit card",
    "debit card",
    "cvv",
    "cvc",
    "security code",
    "account number",
    "pin",
]


URGENCY_TEXT_KEYWORDS = [
    "urgent",
    "urgently",
    "immediately",
    "immediate",
    "verify now",
    "act now",
    "action required",
    "account suspended",
    "account locked",
    "limited time",
    "expire",
    "expired",
    "warning",
    "alert",
]


SECURITY_TEXT_KEYWORDS = [
    "security",
    "secure",
    "verification",
    "verify",
    "identity",
    "authentication",
    "authorization",
    "fraud",
    "suspicious activity",
    "protect your account",
]


# ============================================================
# JAVASCRIPT SUSPICIOUS PATTERNS
# ============================================================

SUSPICIOUS_JS_PATTERNS = [
    r"\beval\s*\(",
    r"\bFunction\s*\(",
    r"\bdocument\.write\s*\(",
    r"\bdocument\.writeln\s*\(",
    r"\bwindow\.location\s*=",
    r"\blocation\.href\s*=",
    r"\blocation\.replace\s*\(",
    r"\blocation\.assign\s*\(",
    r"\bwindow\.open\s*\(",
    r"\batob\s*\(",
    r"\bunescape\s*\(",
    r"\bfromCharCode\s*\(",
]


RIGHT_CLICK_PATTERNS = [
    r"\boncontextmenu\s*=",
    r"\bcontextmenu\b",
    r"\.preventDefault\s*\(",
]


# ============================================================
# URL / DOMAIN HELPERS
# ============================================================

def normalize_url(
    url: Any,
) -> str:

    if url is None:
        return ""

    return str(url).strip()


def resolve_url(
    value: str,
    base_url: str,
) -> str:

    value = normalize_url(
        value
    )

    if not value:
        return ""

    try:

        return urljoin(
            base_url,
            value,
        )

    except Exception:

        return value


def get_hostname(
    url: str,
) -> str:

    try:

        return (
            urlparse(url)
            .hostname
            or ""
        ).lower().strip(".")

    except Exception:

        return ""


def get_registrable_domain(
    url: str,
) -> str:

    hostname = get_hostname(
        url
    )

    if not hostname:
        return ""

    try:

        extracted = (
            tldextract.extract(
                hostname
            )
        )

        if not extracted.domain:
            return ""

        if extracted.suffix:

            return (
                f"{extracted.domain}."
                f"{extracted.suffix}"
            )

        return extracted.domain

    except Exception:

        return hostname


def get_subdomain(
    url: str,
) -> str:

    hostname = get_hostname(
        url
    )

    if not hostname:
        return ""

    try:

        extracted = (
            tldextract.extract(
                hostname
            )
        )

        return extracted.subdomain.lower()

    except Exception:

        return ""


def is_same_registrable_domain(
    first_url: str,
    second_url: str,
) -> bool:

    first_domain = (
        get_registrable_domain(
            first_url
        )
    )

    second_domain = (
        get_registrable_domain(
            second_url
        )
    )

    if (
        not first_domain
        or not second_domain
    ):
        return False

    return (
        first_domain
        == second_domain
    )


def is_subdomain_match(
    page_url: str,
    target_url: str,
) -> bool:

    page_hostname = get_hostname(
        page_url
    )

    target_hostname = get_hostname(
        target_url
    )

    if (
        not page_hostname
        or not target_hostname
    ):
        return False

    if page_hostname == target_hostname:
        return False

    if not is_same_registrable_domain(
        page_url,
        target_url,
    ):
        return False

    return target_hostname.endswith(
        "." + page_hostname
    )


def is_cross_domain(
    page_url: str,
    target_url: str,
) -> bool:

    page_domain = (
        get_registrable_domain(
            page_url
        )
    )

    target_domain = (
        get_registrable_domain(
            target_url
        )
    )

    if (
        not page_domain
        or not target_domain
    ):
        return False

    return (
        page_domain
        != target_domain
    )


# ============================================================
# TEXT / ATTRIBUTE HELPERS
# ============================================================

def get_attribute_text(
    tag,
    attribute_names,
) -> str:

    values = []

    for attribute in attribute_names:

        value = tag.get(
            attribute
        )

        if value is None:
            continue

        if isinstance(
            value,
            list,
        ):

            value = " ".join(
                str(item)
                for item in value
            )

        values.append(
            str(value)
        )

    return (
        " ".join(values)
        .strip()
        .lower()
    )


def contains_keyword(
    text: str,
    keywords: set[str],
) -> bool:

    normalized = re.sub(
        r"[-\s]+",
        "_",
        text.lower(),
    )

    for keyword in keywords:

        keyword_normalized = re.sub(
            r"[-\s]+",
            "_",
            keyword.lower(),
        )

        if (
            keyword_normalized
            in normalized
        ):
            return True

    return False


def count_text_keywords(
    text: str,
    keywords: list[str],
) -> int:

    if not text:
        return 0

    normalized_text = (
        text.lower()
    )

    count = 0

    for keyword in keywords:

        count += normalized_text.count(
            keyword.lower()
        )

    return count


# ============================================================
# INPUT CLASSIFICATION
# ============================================================

def is_email_field(
    tag,
) -> bool:

    input_type = str(
        tag.get(
            "type",
            "",
        )
    ).strip().lower()

    if input_type == "email":
        return True

    text = get_attribute_text(
        tag,
        [
            "name",
            "id",
            "placeholder",
            "autocomplete",
        ],
    )

    return contains_keyword(
        text,
        EMAIL_KEYWORDS,
    )


def is_username_field(
    tag,
) -> bool:

    text = get_attribute_text(
        tag,
        [
            "name",
            "id",
            "placeholder",
            "autocomplete",
        ],
    )

    return contains_keyword(
        text,
        USERNAME_KEYWORDS,
    )


def is_payment_field(
    tag,
) -> bool:

    text = get_attribute_text(
        tag,
        [
            "name",
            "id",
            "placeholder",
            "autocomplete",
            "aria-label",
        ],
    )

    return contains_keyword(
        text,
        PAYMENT_KEYWORDS,
    )


def is_otp_field(
    tag,
) -> bool:

    text = get_attribute_text(
        tag,
        [
            "name",
            "id",
            "placeholder",
            "autocomplete",
            "aria-label",
        ],
    )

    return contains_keyword(
        text,
        OTP_KEYWORDS,
    )


def is_credential_field(
    tag,
) -> bool:

    input_type = str(
        tag.get(
            "type",
            "",
        )
    ).strip().lower()

    if input_type == "password":
        return True

    text = get_attribute_text(
        tag,
        [
            "name",
            "id",
            "placeholder",
            "autocomplete",
            "aria-label",
        ],
    )

    return contains_keyword(
        text,
        CREDENTIAL_KEYWORDS,
    )


# ============================================================
# HIDDEN ELEMENT DETECTION
# ============================================================

def has_hidden_attribute(
    tag,
) -> bool:

    if tag.has_attr(
        "hidden"
    ):
        return True

    aria_hidden = str(
        tag.get(
            "aria-hidden",
            "",
        )
    ).strip().lower()

    if aria_hidden == "true":
        return True

    return False


def style_contains_hidden_value(
    style: str,
) -> bool:

    style = (
        style.lower()
        .replace(
            " ",
            "",
        )
    )

    hidden_patterns = [
        "display:none",
        "visibility:hidden",
        "opacity:0",
    ]

    return any(
        pattern in style
        for pattern in hidden_patterns
    )


def is_hidden_element(
    tag,
) -> bool:

    if has_hidden_attribute(
        tag
    ):
        return True

    style = str(
        tag.get(
            "style",
            "",
        )
    )

    if (
        style
        and style_contains_hidden_value(
            style
        )
    ):
        return True

    return False


def is_invisible_iframe(
    tag,
) -> bool:

    if is_hidden_element(
        tag
    ):
        return True

    style = (
        str(
            tag.get(
                "style",
                "",
            )
        )
        .lower()
        .replace(
            " ",
            "",
        )
    )

    width = str(
        tag.get(
            "width",
            "",
        )
    ).strip().lower()

    height = str(
        tag.get(
            "height",
            "",
        )
    ).strip().lower()

    if width in {
        "0",
        "0px",
    }:
        return True

    if height in {
        "0",
        "0px",
    }:
        return True

    if (
        "width:0" in style
        or "width:0px" in style
    ):
        return True

    if (
        "height:0" in style
        or "height:0px" in style
    ):
        return True

    return False


# ============================================================
# JAVASCRIPT DETECTION
# ============================================================

def is_suspicious_inline_script(
    script_text: str,
) -> bool:

    if not script_text:
        return False

    for pattern in (
        SUSPICIOUS_JS_PATTERNS
    ):

        if re.search(
            pattern,
            script_text,
            flags=re.IGNORECASE,
        ):

            return True

    long_hex_strings = re.findall(
        r"(?:\\x[0-9a-fA-F]{2}){6,}",
        script_text,
    )

    if long_hex_strings:
        return True

    long_unicode_sequences = re.findall(
        r"(?:\\u[0-9a-fA-F]{4}){5,}",
        script_text,
    )

    if long_unicode_sequences:
        return True

    return False


def has_right_click_disabled(
    html: str,
    soup: BeautifulSoup,
) -> int:

    for pattern in (
        RIGHT_CLICK_PATTERNS
    ):

        if re.search(
            pattern,
            html,
            flags=re.IGNORECASE,
        ):
            return 1

    for tag in soup.find_all():

        handler = str(
            tag.get(
                "oncontextmenu",
                "",
            )
        )

        if handler:
            return 1

    return 0


# ============================================================
# FORM FEATURES
# ============================================================

def extract_form_features(
    soup: BeautifulSoup,
    page_url: str,
) -> dict[str, Any]:

    forms = soup.find_all(
        "form"
    )

    form_count = len(
        forms
    )

    forms_with_action = 0
    empty_form_action_count = 0

    password_field_count = 0
    email_field_count = 0
    username_field_count = 0
    payment_field_count = 0
    otp_field_count = 0
    credential_field_count = 0

    cross_domain_form_actions = 0
    same_domain_form_actions = 0
    http_form_actions = 0
    https_form_actions = 0
    subdomain_match_count = 0
    registrable_domain_match_count = 0

    for form in forms:

        action = form.get(
            "action"
        )

        if (
            action is None
            or not str(action).strip()
        ):

            empty_form_action_count += 1

        else:

            forms_with_action += 1

            resolved_action = (
                resolve_url(
                    str(action),
                    page_url,
                )
            )

            parsed_action = urlparse(
                resolved_action
            )

            scheme = (
                parsed_action.scheme
                .lower()
            )

            if scheme == "http":

                http_form_actions += 1

            elif scheme == "https":

                https_form_actions += 1

            if is_cross_domain(
                page_url,
                resolved_action,
            ):

                cross_domain_form_actions += 1

            elif is_same_registrable_domain(
                page_url,
                resolved_action,
            ):

                same_domain_form_actions += 1

                registrable_domain_match_count += 1

                if is_subdomain_match(
                    page_url,
                    resolved_action,
                ):

                    subdomain_match_count += 1

        inputs = form.find_all(
            [
                "input",
                "textarea",
                "select",
            ]
        )

        for field in inputs:

            field_type = str(
                field.get(
                    "type",
                    "",
                )
            ).strip().lower()

            if field_type == "password":

                password_field_count += 1

            if is_email_field(
                field
            ):

                email_field_count += 1

            if is_username_field(
                field
            ):

                username_field_count += 1

            if is_payment_field(
                field
            ):

                payment_field_count += 1

            if is_otp_field(
                field
            ):

                otp_field_count += 1

            if is_credential_field(
                field
            ):

                credential_field_count += 1

    if forms_with_action > 0:

        external_form_action_ratio = (
            cross_domain_form_actions
            / forms_with_action
        )

    else:

        external_form_action_ratio = 0.0

    return {
        "form_count": form_count,

        "forms_with_action":
            forms_with_action,

        "empty_form_action_count":
            empty_form_action_count,

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

        "has_password_field":
            int(
                password_field_count > 0
            ),

        "cross_domain_form_actions":
            cross_domain_form_actions,

        "same_domain_form_actions":
            same_domain_form_actions,

        "http_form_actions":
            http_form_actions,

        "https_form_actions":
            https_form_actions,

        "external_form_action_ratio":
            float(
                external_form_action_ratio
            ),

        "subdomain_match_count":
            subdomain_match_count,

        "registrable_domain_match_count":
            registrable_domain_match_count,
    }


# ============================================================
# HIDDEN INPUT FEATURES
# ============================================================

def extract_hidden_input_features(
    soup: BeautifulSoup,
) -> dict[str, Any]:

    inputs = soup.find_all(
        "input"
    )

    total_input_count = len(
        inputs
    )

    hidden_inputs = []

    for tag in inputs:

        input_type = str(
            tag.get(
                "type",
                "",
            )
        ).strip().lower()

        if input_type == "hidden":

            hidden_inputs.append(
                tag
            )

    hidden_input_count = len(
        hidden_inputs
    )

    credential_hidden_input_count = 0
    suspicious_hidden_input_count = 0

    for tag in hidden_inputs:

        text = get_attribute_text(
            tag,
            [
                "name",
                "id",
                "value",
                "autocomplete",
                "aria-label",
            ],
        )

        if contains_keyword(
            text,
            CREDENTIAL_KEYWORDS,
        ):

            credential_hidden_input_count += 1

        if contains_keyword(
            text,
            SUSPICIOUS_HIDDEN_KEYWORDS,
        ):

            suspicious_hidden_input_count += 1

    if total_input_count > 0:

        hidden_input_ratio = (
            hidden_input_count
            / total_input_count
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
            float(
                hidden_input_ratio
            ),
    }


# ============================================================
# IFRAME FEATURES
# ============================================================

def extract_iframe_features(
    soup: BeautifulSoup,
) -> dict[str, int]:

    iframes = soup.find_all(
        "iframe"
    )

    invisible_iframes = sum(
        1
        for iframe in iframes
        if is_invisible_iframe(
            iframe
        )
    )

    return {
        "iframe_count":
            len(iframes),

        "invisible_iframes":
            invisible_iframes,
    }


# ============================================================
# SCRIPT FEATURES
# ============================================================

def extract_script_features(
    soup: BeautifulSoup,
) -> dict[str, int]:

    scripts = soup.find_all(
        "script"
    )

    external_scripts_count = 0
    suspicious_inline_scripts = 0

    for script in scripts:

        src = str(
            script.get(
                "src",
                "",
            )
        ).strip()

        if src:

            external_scripts_count += 1

        else:

            script_text = (
                script.string
                or script.get_text()
                or ""
            )

            if is_suspicious_inline_script(
                script_text
            ):

                suspicious_inline_scripts += 1

    return {
        "script_count":
            len(scripts),

        "external_scripts_count":
            external_scripts_count,

        "suspicious_inline_scripts":
            suspicious_inline_scripts,
    }


# ============================================================
# LINK FEATURES
# ============================================================

def extract_link_features(
    soup: BeautifulSoup,
    page_url: str,
) -> dict[str, int]:

    links = soup.find_all(
        "a"
    )

    total_links = len(
        links
    )

    external_links_count = 0
    empty_links_count = 0

    for link in links:

        href = link.get(
            "href"
        )

        if href is None:

            empty_links_count += 1
            continue

        href = str(
            href
        ).strip()

        if not href:

            empty_links_count += 1
            continue

        normalized_href = (
            href.lower()
        )

        if normalized_href in {
            "#",
            "javascript:void(0)",
            "javascript:void(0);",
        }:

            empty_links_count += 1
            continue

        resolved_url = resolve_url(
            href,
            page_url,
        )

        if is_cross_domain(
            page_url,
            resolved_url,
        ):

            external_links_count += 1

    return {
        "total_links":
            total_links,

        "external_links_count":
            external_links_count,

        "empty_links_count":
            empty_links_count,
    }


# ============================================================
# CROSS-DOMAIN FEATURES
# ============================================================

def extract_cross_domain_count(
    soup: BeautifulSoup,
    page_url: str,
) -> int:

    cross_domain_count = 0

    # --------------------------------------------------------
    # Forms
    # --------------------------------------------------------

    for form in soup.find_all(
        "form"
    ):

        action = str(
            form.get(
                "action",
                "",
            )
        ).strip()

        if not action:
            continue

        resolved = resolve_url(
            action,
            page_url,
        )

        if is_cross_domain(
            page_url,
            resolved,
        ):

            cross_domain_count += 1

    # --------------------------------------------------------
    # Anchors
    # --------------------------------------------------------

    for link in soup.find_all(
        "a"
    ):

        href = str(
            link.get(
                "href",
                "",
            )
        ).strip()

        if not href:
            continue

        resolved = resolve_url(
            href,
            page_url,
        )

        if is_cross_domain(
            page_url,
            resolved,
        ):

            cross_domain_count += 1

    # --------------------------------------------------------
    # Scripts
    # --------------------------------------------------------

    for script in soup.find_all(
        "script"
    ):

        src = str(
            script.get(
                "src",
                "",
            )
        ).strip()

        if not src:
            continue

        resolved = resolve_url(
            src,
            page_url,
        )

        if is_cross_domain(
            page_url,
            resolved,
        ):

            cross_domain_count += 1

    # --------------------------------------------------------
    # Iframes
    # --------------------------------------------------------

    for iframe in soup.find_all(
        "iframe"
    ):

        src = str(
            iframe.get(
                "src",
                "",
            )
        ).strip()

        if not src:
            continue

        resolved = resolve_url(
            src,
            page_url,
        )

        if is_cross_domain(
            page_url,
            resolved,
        ):

            cross_domain_count += 1

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    for image in soup.find_all(
        "img"
    ):

        src = str(
            image.get(
                "src",
                "",
            )
        ).strip()

        if not src:
            continue

        resolved = resolve_url(
            src,
            page_url,
        )

        if is_cross_domain(
            page_url,
            resolved,
        ):

            cross_domain_count += 1

    # --------------------------------------------------------
    # Stylesheets
    # --------------------------------------------------------

    for link in soup.find_all(
        "link"
    ):

        href = str(
            link.get(
                "href",
                "",
            )
        ).strip()

        if not href:
            continue

        resolved = resolve_url(
            href,
            page_url,
        )

        if is_cross_domain(
            page_url,
            resolved,
        ):

            cross_domain_count += 1

    return cross_domain_count


# ============================================================
# NEW CONTENT FEATURES
# ============================================================

def extract_content_features(
    soup: BeautifulSoup,
    html: str,
    page_url: str,
) -> dict[str, Any]:
    """
    Extract the additional 15 HTML/content features.

    These features are intentionally kept numeric so they can
    be passed directly to XGBoost.
    """

    # --------------------------------------------------------
    # Create a separate soup for visible text.
    #
    # We do NOT modify the original soup because the original
    # 35 features still need the complete parsed HTML.
    # --------------------------------------------------------

    text_soup = BeautifulSoup(
        html,
        "lxml",
    )

    # Remove non-visible-content elements.
    for element in text_soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "template",
        ]
    ):

        element.decompose()

    visible_text = text_soup.get_text(
        separator=" ",
        strip=True,
    )

    # --------------------------------------------------------
    # 36. title_length
    # --------------------------------------------------------

    title_length = 0

    title = soup.find(
        "title"
    )

    if title is not None:

        title_text = title.get_text(
            strip=True
        )

        title_length = len(
            title_text
        )

    # --------------------------------------------------------
    # 37. visible_text_length
    # --------------------------------------------------------

    visible_text_length = len(
        visible_text
    )

    # --------------------------------------------------------
    # 38. heading_count
    # --------------------------------------------------------

    heading_count = len(
        soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            ]
        )
    )

    # --------------------------------------------------------
    # 39. button_count
    # --------------------------------------------------------

    button_count = len(
        soup.find_all(
            "button"
        )
    )

    # Also count input buttons.
    button_inputs = soup.find_all(
        "input",
        attrs={
            "type": re.compile(
                r"^(submit|button|reset)$",
                re.IGNORECASE,
            )
        },
    )

    button_count += len(
        button_inputs
    )

    # --------------------------------------------------------
    # 40. input_count
    # --------------------------------------------------------

    input_count = len(
        soup.find_all(
            "input"
        )
    )

    # --------------------------------------------------------
    # 41. image_count
    # --------------------------------------------------------

    image_count = len(
        soup.find_all(
            "img"
        )
    )

    # --------------------------------------------------------
    # 42. anchor_count
    # --------------------------------------------------------

    anchor_count = len(
        soup.find_all(
            "a"
        )
    )

    # --------------------------------------------------------
    # 43. login_keyword_count
    # --------------------------------------------------------

    login_keyword_count = (
        count_text_keywords(
            visible_text,
            LOGIN_TEXT_KEYWORDS,
        )
    )

    # --------------------------------------------------------
    # 44. credential_keyword_count
    # --------------------------------------------------------

    credential_keyword_count = (
        count_text_keywords(
            visible_text,
            CREDENTIAL_TEXT_KEYWORDS,
        )
    )

    # --------------------------------------------------------
    # 45. urgency_keyword_count
    # --------------------------------------------------------

    urgency_keyword_count = (
        count_text_keywords(
            visible_text,
            URGENCY_TEXT_KEYWORDS,
        )
    )

    # --------------------------------------------------------
    # 46. security_keyword_count
    # --------------------------------------------------------

    security_keyword_count = (
        count_text_keywords(
            visible_text,
            SECURITY_TEXT_KEYWORDS,
        )
    )

    # --------------------------------------------------------
    # Collect JavaScript
    # --------------------------------------------------------

    javascript_parts = []

    for script in soup.find_all(
        "script"
    ):

        script_text = (
            script.string
            or script.get_text()
            or ""
        )

        javascript_parts.append(
            script_text
        )

    javascript = "\n".join(
        javascript_parts
    )

    # --------------------------------------------------------
    # 47. eval_count
    # --------------------------------------------------------

    eval_count = len(
        re.findall(
            r"\beval\s*\(",
            javascript,
            flags=re.IGNORECASE,
        )
    )

    # --------------------------------------------------------
    # 48. document_write_count
    # --------------------------------------------------------

    document_write_count = len(
        re.findall(
            r"\bdocument\s*\.\s*write(?:ln)?\s*\(",
            javascript,
            flags=re.IGNORECASE,
        )
    )

    # --------------------------------------------------------
    # 49. javascript_redirect_count
    # --------------------------------------------------------

    javascript_redirect_patterns = [
        r"\bwindow\s*\.\s*location\b",
        r"\blocation\s*\.\s*href\b",
        r"\blocation\s*=",
        r"\blocation\s*\.\s*replace\s*\(",
        r"\blocation\s*\.\s*assign\s*\(",
        r"\bwindow\s*\.\s*open\s*\(",
    ]

    javascript_redirect_count = 0

    for pattern in (
        javascript_redirect_patterns
    ):

        javascript_redirect_count += len(
            re.findall(
                pattern,
                javascript,
                flags=re.IGNORECASE,
            )
        )

    # --------------------------------------------------------
    # 50. external_images_count
    # --------------------------------------------------------

    external_images_count = 0

    for image in soup.find_all(
        "img"
    ):

        src = str(
            image.get(
                "src",
                "",
            )
        ).strip()

        if not src:
            continue

        resolved = resolve_url(
            src,
            page_url,
        )

        if is_cross_domain(
            page_url,
            resolved,
        ):

            external_images_count += 1

    return {
        "title_length":
            title_length,

        "visible_text_length":
            visible_text_length,

        "heading_count":
            heading_count,

        "button_count":
            button_count,

        "input_count":
            input_count,

        "image_count":
            image_count,

        "anchor_count":
            anchor_count,

        "login_keyword_count":
            login_keyword_count,

        "credential_keyword_count":
            credential_keyword_count,

        "urgency_keyword_count":
            urgency_keyword_count,

        "security_keyword_count":
            security_keyword_count,

        "eval_count":
            eval_count,

        "document_write_count":
            document_write_count,

        "javascript_redirect_count":
            javascript_redirect_count,

        "external_images_count":
            external_images_count,
    }


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_html_features(
    html: str,
    page_url: str = "",
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Normalize input
    # --------------------------------------------------------

    if html is None:

        html = ""

    if not isinstance(
        html,
        str,
    ):

        html = str(
            html
        )

    page_url = normalize_url(
        page_url
    )

    # --------------------------------------------------------
    # Parse HTML
    # --------------------------------------------------------

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    # --------------------------------------------------------
    # Basic
    # --------------------------------------------------------

    features = {
        "html_length_bytes":
            len(
                html.encode(
                    "utf-8",
                    errors="replace",
                )
            ),
    }

    # --------------------------------------------------------
    # Original form features
    # --------------------------------------------------------

    features.update(
        extract_form_features(
            soup,
            page_url,
        )
    )

    # --------------------------------------------------------
    # Original hidden-input features
    # --------------------------------------------------------

    features.update(
        extract_hidden_input_features(
            soup
        )
    )

    # --------------------------------------------------------
    # Original iframe features
    # --------------------------------------------------------

    features.update(
        extract_iframe_features(
            soup
        )
    )

    # --------------------------------------------------------
    # Original script features
    # --------------------------------------------------------

    features.update(
        extract_script_features(
            soup
        )
    )

    # --------------------------------------------------------
    # Original link features
    # --------------------------------------------------------

    features.update(
        extract_link_features(
            soup,
            page_url,
        )
    )

    # --------------------------------------------------------
    # Original cross-domain feature
    # --------------------------------------------------------

    features[
        "cross_domain_count"
    ] = extract_cross_domain_count(
        soup,
        page_url,
    )

    # --------------------------------------------------------
    # Original hidden-elements feature
    # --------------------------------------------------------

    hidden_elements_count = 0

    for tag in soup.find_all():

        if is_hidden_element(
            tag
        ):

            hidden_elements_count += 1

    features[
        "hidden_elements_count"
    ] = hidden_elements_count

    # --------------------------------------------------------
    # Original right-click feature
    # --------------------------------------------------------

    features[
        "right_click_disabled"
    ] = has_right_click_disabled(
        html,
        soup,
    )

    # --------------------------------------------------------
    # Original meta-refresh feature
    # --------------------------------------------------------

    has_meta_refresh = 0

    for meta in soup.find_all(
        "meta"
    ):

        http_equiv = str(
            meta.get(
                "http-equiv",
                "",
            )
        ).strip().lower()

        if http_equiv == "refresh":

            has_meta_refresh = 1
            break

    features[
        "has_meta_refresh"
    ] = has_meta_refresh

    # --------------------------------------------------------
    # Original HTML comments feature
    # --------------------------------------------------------

    from bs4 import Comment

    html_comments_count = len(
        soup.find_all(
            string=lambda text:
            isinstance(
                text,
                Comment,
            )
        )
    )

    features[
        "html_comments_count"
    ] = html_comments_count

    # ========================================================
    # NEW 15 CONTENT FEATURES
    # ========================================================

    features.update(
        extract_content_features(
            soup,
            html,
            page_url,
        )
    )

    # ========================================================
    # ENSURE EXACT 50-FEATURE SCHEMA
    # ========================================================

    final_features = {}

    for feature_name in (
        HTML_FEATURES
    ):

        value = features.get(
            feature_name,
            0,
        )

        final_features[
            feature_name
        ] = value

    # --------------------------------------------------------
    # Final schema safety check
    # --------------------------------------------------------

    if len(
        final_features
    ) != len(
        HTML_FEATURES
    ):

        raise RuntimeError(
            "HTML feature extraction schema "
            "validation failed."
        )

    if list(
        final_features.keys()
    ) != HTML_FEATURES:

        raise RuntimeError(
            "HTML feature ordering does not "
            "match HTML_FEATURES."
        )

    return final_features
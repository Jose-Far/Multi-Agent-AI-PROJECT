"""
HTML AI Agent - Feature Schema
================================

Single source of truth for the HTML AI Agent ML feature vector.

IMPORTANT
---------
The current production HTML model uses exactly 50 features.

The feature order defined here MUST match:

    1. html_feature_extractor.py
    2. dataset extraction
    3. train/validation/test datasets
    4. XGBoost training
    5. preprocessing
    6. model inference
    7. prediction
    8. explanation/XAI
"""

import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class HTMLFeatureSchema:
    """
    Centralized 50-feature contract for the HTML AI Agent.
    """

    # =====================================================================
    # 1. EXACT 50-FEATURE ORDER
    # =====================================================================

    EXPECTED_FEATURE_ORDER: List[str] = [

        # ================================================================
        # BASIC HTML / DOM
        # ================================================================

        "html_length_bytes",

        # ================================================================
        # FORM STRUCTURE
        # ================================================================

        "form_count",
        "forms_with_action",
        "empty_form_action_count",

        # ================================================================
        # CREDENTIAL FIELDS
        # ================================================================

        "password_field_count",
        "email_field_count",
        "username_field_count",
        "payment_field_count",
        "otp_field_count",
        "credential_field_count",
        "has_password_field",

        # ================================================================
        # FORM DESTINATION / EXFILTRATION
        # ================================================================

        "cross_domain_form_actions",
        "same_domain_form_actions",
        "http_form_actions",
        "https_form_actions",
        "external_form_action_ratio",

        # ================================================================
        # DOMAIN RELATIONSHIP
        # ================================================================

        "subdomain_match_count",
        "registrable_domain_match_count",
        "cross_domain_count",

        # ================================================================
        # HIDDEN INPUTS
        # ================================================================

        "hidden_input_count",
        "credential_hidden_input_count",
        "suspicious_hidden_input_count",
        "hidden_input_ratio",

        # ================================================================
        # IFRAMES
        # ================================================================

        "iframe_count",
        "invisible_iframes",

        # ================================================================
        # JAVASCRIPT
        # ================================================================

        "script_count",
        "external_scripts_count",
        "suspicious_inline_scripts",

        # ================================================================
        # LINKS
        # ================================================================

        "total_links",
        "external_links_count",
        "empty_links_count",

        # ================================================================
        # DOM / EVASION
        # ================================================================

        "hidden_elements_count",
        "right_click_disabled",
        "has_meta_refresh",
        "html_comments_count",

        # ================================================================
        # CONTENT / TEXT
        # ================================================================

        "title_length",
        "visible_text_length",
        "heading_count",

        # ================================================================
        # UI / INPUT / MEDIA
        # ================================================================

        "button_count",
        "input_count",
        "image_count",
        "anchor_count",

        # ================================================================
        # SECURITY KEYWORD FEATURES
        # ================================================================

        "login_keyword_count",
        "credential_keyword_count",
        "urgency_keyword_count",
        "security_keyword_count",

        # ================================================================
        # JAVASCRIPT BEHAVIOR
        # ================================================================

        "eval_count",
        "document_write_count",
        "javascript_redirect_count",

        # ================================================================
        # EXTERNAL MEDIA
        # ================================================================

        "external_images_count",
    ]

    # =====================================================================
    # 2. FEATURE DESCRIPTIONS
    # =====================================================================

    FEATURE_METADATA: Dict[str, str] = {

        # ---------------------------------------------------------------
        # BASIC
        # ---------------------------------------------------------------

        "html_length_bytes":
            "Total size of the HTML document in bytes.",

        # ---------------------------------------------------------------
        # FORMS
        # ---------------------------------------------------------------

        "form_count":
            "Total number of HTML forms present in the page.",

        "forms_with_action":
            "Number of forms explicitly specifying an action destination.",

        "empty_form_action_count":
            "Number of forms with empty or missing action attributes.",

        # ---------------------------------------------------------------
        # CREDENTIALS
        # ---------------------------------------------------------------

        "password_field_count":
            "Number of password or password-like input fields.",

        "email_field_count":
            "Number of email or email-like input fields.",

        "username_field_count":
            "Number of username, user ID, login, or identity fields.",

        "payment_field_count":
            "Number of payment-related fields such as card number, CVV, CVC, or expiry.",

        "otp_field_count":
            "Number of OTP, verification-code, PIN, or authentication-code fields.",

        "credential_field_count":
            "Total number of credential or sensitive-information fields.",

        "has_password_field":
            "Indicates whether at least one password field is present.",

        # ---------------------------------------------------------------
        # FORM DESTINATION
        # ---------------------------------------------------------------

        "cross_domain_form_actions":
            "Number of forms submitting data to a different hostname.",

        "same_domain_form_actions":
            "Number of forms submitting data to the same hostname.",

        "http_form_actions":
            "Number of form actions using unencrypted HTTP.",

        "https_form_actions":
            "Number of form actions using HTTPS.",

        "external_form_action_ratio":
            "Ratio of forms submitting data to external domains.",

        # ---------------------------------------------------------------
        # DOMAIN
        # ---------------------------------------------------------------

        "subdomain_match_count":
            "Number of form destinations belonging to a related subdomain hierarchy.",

        "registrable_domain_match_count":
            "Number of form destinations sharing the same registrable domain.",

        "cross_domain_count":
            "Number of forms whose submission destination differs from the page hostname.",

        # ---------------------------------------------------------------
        # HIDDEN INPUTS
        # ---------------------------------------------------------------

        "hidden_input_count":
            "Total number of hidden input elements.",

        "credential_hidden_input_count":
            "Number of hidden inputs associated with credential or sensitive information.",

        "suspicious_hidden_input_count":
            "Number of hidden inputs showing potentially suspicious characteristics.",

        "hidden_input_ratio":
            "Ratio of hidden inputs relative to all input elements.",

        # ---------------------------------------------------------------
        # IFRAMES
        # ---------------------------------------------------------------

        "iframe_count":
            "Total number of iframe elements.",

        "invisible_iframes":
            "Number of hidden or invisible iframe elements.",

        # ---------------------------------------------------------------
        # JAVASCRIPT
        # ---------------------------------------------------------------

        "script_count":
            "Total number of JavaScript script elements.",

        "external_scripts_count":
            "Number of scripts loaded from external sources.",

        "suspicious_inline_scripts":
            "Number of inline scripts containing suspicious or obfuscation-related patterns.",

        # ---------------------------------------------------------------
        # LINKS
        # ---------------------------------------------------------------

        "total_links":
            "Total number of hyperlinks.",

        "external_links_count":
            "Number of hyperlinks pointing to external domains.",

        "empty_links_count":
            "Number of empty, '#' or javascript:void links.",

        # ---------------------------------------------------------------
        # DOM / EVASION
        # ---------------------------------------------------------------

        "hidden_elements_count":
            "Number of elements hidden through CSS display or visibility properties.",

        "right_click_disabled":
            "Indicates whether right-click/context-menu inspection has been disabled.",

        "has_meta_refresh":
            "Indicates the presence of an automatic meta refresh redirect.",

        "html_comments_count":
            "Number of HTML comments present in the document.",

        # ---------------------------------------------------------------
        # CONTENT
        # ---------------------------------------------------------------

        "title_length":
            "Length of the HTML page title.",

        "visible_text_length":
            "Length of visible text extracted from the HTML document.",

        "heading_count":
            "Total number of heading elements.",

        # ---------------------------------------------------------------
        # UI / MEDIA
        # ---------------------------------------------------------------

        "button_count":
            "Total number of button elements.",

        "input_count":
            "Total number of input elements.",

        "image_count":
            "Total number of image elements.",

        "anchor_count":
            "Total number of anchor elements.",

        # ---------------------------------------------------------------
        # KEYWORDS
        # ---------------------------------------------------------------

        "login_keyword_count":
            "Count of login-related keywords in visible page content.",

        "credential_keyword_count":
            "Count of credential-related keywords in visible page content.",

        "urgency_keyword_count":
            "Count of urgency-related keywords in visible page content.",

        "security_keyword_count":
            "Count of security-related keywords in visible page content.",

        # ---------------------------------------------------------------
        # JAVASCRIPT BEHAVIOR
        # ---------------------------------------------------------------

        "eval_count":
            "Number of JavaScript eval() occurrences.",

        "document_write_count":
            "Number of JavaScript document.write() occurrences.",

        "javascript_redirect_count":
            "Number of JavaScript-based redirect patterns.",

        # ---------------------------------------------------------------
        # EXTERNAL IMAGES
        # ---------------------------------------------------------------

        "external_images_count":
            "Number of images loaded from external domains.",
    }

    # =====================================================================
    # 3. BOOLEAN FEATURES
    # =====================================================================

    BOOLEAN_FEATURES = {
        "right_click_disabled",
        "has_meta_refresh",
        "has_password_field",
    }

    # =====================================================================
    # 4. RATIO FEATURES
    # =====================================================================

    RATIO_FEATURES = {
        "external_form_action_ratio",
        "hidden_input_ratio",
    }

    # =====================================================================
    # 5. SCHEMA ACCESS
    # =====================================================================

    @classmethod
    def get_schema(cls) -> List[str]:

        return list(
            cls.EXPECTED_FEATURE_ORDER
        )

    @classmethod
    def get_descriptions(cls) -> Dict[str, str]:

        return dict(
            cls.FEATURE_METADATA
        )

    @classmethod
    def get_boolean_features(cls) -> set:

        return set(
            cls.BOOLEAN_FEATURES
        )

    @classmethod
    def get_ratio_features(cls) -> set:

        return set(
            cls.RATIO_FEATURES
        )

    # =====================================================================
    # 6. DEFAULT VALUE
    # =====================================================================

    @classmethod
    def _default_value(
        cls,
        feature_name: str
    ) -> float:

        return 0.0

    # =====================================================================
    # 7. SAFE FLOAT
    # =====================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0
    ) -> float:

        try:

            if value is None:
                return default

            if isinstance(
                value,
                bool
            ):

                return (
                    1.0
                    if value
                    else 0.0
                )

            converted = float(
                value
            )

            if not np.isfinite(
                converted
            ):

                return default

            return converted

        except (
            TypeError,
            ValueError,
            OverflowError
        ):

            return default

    # =====================================================================
    # 8. ALIGN AND VALIDATE
    # =====================================================================

    @classmethod
    def align_and_validate(
        cls,
        raw_features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Convert raw 50-feature dictionary into a one-row DataFrame.
        """

        if not isinstance(
            raw_features,
            dict
        ):

            logger.warning(
                "Invalid HTML feature payload. "
                "Using zero-valued feature vector."
            )

            raw_features = {}

        normalized = {}

        for feature_name in (
            cls.EXPECTED_FEATURE_ORDER
        ):

            value = raw_features.get(
                feature_name,
                cls._default_value(
                    feature_name
                )
            )

            value = cls._safe_float(
                value
            )

            # -----------------------------------------------------------
            # Boolean normalization
            # -----------------------------------------------------------

            if feature_name in (
                cls.BOOLEAN_FEATURES
            ):

                value = (
                    1.0
                    if value != 0.0
                    else 0.0
                )

            # -----------------------------------------------------------
            # Ratio normalization
            # -----------------------------------------------------------

            elif feature_name in (
                cls.RATIO_FEATURES
            ):

                value = max(
                    0.0,
                    min(
                        1.0,
                        value
                    )
                )

            normalized[
                feature_name
            ] = value

        dataframe = pd.DataFrame(
            [
                normalized
            ],
            columns=cls.EXPECTED_FEATURE_ORDER
        )

        dataframe = dataframe.replace(
            [np.inf, -np.inf],
            0.0
        )

        dataframe = dataframe.fillna(
            0.0
        )

        for column in dataframe.columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            ).fillna(
                0.0
            ).astype(float)

        return dataframe

    # =====================================================================
    # 9. DATAFRAME VALIDATION
    # =====================================================================

    @classmethod
    def validate_dataframe(
        cls,
        dataframe: pd.DataFrame
    ) -> bool:

        if not isinstance(
            dataframe,
            pd.DataFrame
        ):

            return False

        return (
            list(
                dataframe.columns
            )
            ==
            cls.EXPECTED_FEATURE_ORDER
        )

    # =====================================================================
    # 10. EMPTY VECTOR
    # =====================================================================

    @classmethod
    def get_empty_feature_vector(
        cls
    ) -> Dict[str, float]:

        return {
            feature: 0.0
            for feature
            in cls.EXPECTED_FEATURE_ORDER
        }

    # =====================================================================
    # 11. SCHEMA VALIDATION
    # =====================================================================

    @classmethod
    def validate_schema(cls) -> bool:

        if len(
            cls.EXPECTED_FEATURE_ORDER
        ) != 50:

            raise ValueError(
                "HTML feature schema must contain exactly "
                f"50 features, found "
                f"{len(cls.EXPECTED_FEATURE_ORDER)}."
            )

        if len(
            set(
                cls.EXPECTED_FEATURE_ORDER
            )
        ) != 50:

            raise ValueError(
                "HTML feature schema contains duplicate feature names."
            )

        metadata_missing = [
            feature
            for feature
            in cls.EXPECTED_FEATURE_ORDER
            if feature not in cls.FEATURE_METADATA
        ]

        if metadata_missing:

            raise ValueError(
                "Missing feature metadata:\n"
                + "\n".join(
                    metadata_missing
                )
            )

        return True


# =====================================================================
# VALIDATE SCHEMA WHEN MODULE IS IMPORTED
# =====================================================================

HTMLFeatureSchema.validate_schema()
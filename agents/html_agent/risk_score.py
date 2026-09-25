"""
HTML AI Agent - Risk Scoring Engine
===================================

Converts the HTML AI Agent's phishing probability into an
HTML-agent-local risk assessment.

IMPORTANT
---------
This module does NOT produce the final system-wide phishing verdict.

It produces only the HTML Agent's local risk assessment.

The final phishing decision must be produced by the
Multi-Agent Decision Fusion Engine.
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class HTMLRiskScorer:
    """
    Risk scoring engine for the HTML AI Agent.

    The ML phishing probability is the primary signal.
    HTML structural indicators provide supporting evidence.
    """

    # =====================================================================
    # RISK THRESHOLDS
    # =====================================================================

    LOW_THRESHOLD = 0.30
    MEDIUM_THRESHOLD = 0.60
    HIGH_THRESHOLD = 0.80

    MIN_SCORE = 0
    MAX_SCORE = 100

    MIN_PROBABILITY = 0.0
    MAX_PROBABILITY = 1.0

    # =====================================================================
    # INDICATOR WEIGHTS
    # =====================================================================

    INDICATOR_WEIGHTS = {
        "credential_hidden_input": 15,
        "cross_domain_credential_form": 15,
        "password_field": 3,
        "email_field": 2,
        "otp_field": 5,
        "payment_field": 5,

        "cross_domain_form": 10,
        "http_form_action": 8,
        "empty_form_action": 4,

        "hidden_input": 5,
        "suspicious_hidden_input": 8,
        "hidden_elements": 5,

        "invisible_iframe": 12,
        "iframe": 3,

        "suspicious_javascript": 12,
        "high_external_script_count": 4,

        "meta_refresh": 8,
        "right_click_disabled": 3,
        "excessive_empty_links": 3,

        # New 50-feature signals
        "login_keywords": 4,
        "credential_keywords": 5,
        "urgency_keywords": 5,
        "security_keywords": 3,
        "javascript_redirect": 8,
        "external_images": 3,
    }

    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    def __init__(
        self,
        probability_weight: float = 0.70,
        indicator_weight: float = 0.30,
    ):
        self.probability_weight = float(probability_weight)
        self.indicator_weight = float(indicator_weight)

        if not np.isfinite(self.probability_weight):
            raise ValueError(
                "probability_weight must be finite."
            )

        if not np.isfinite(self.indicator_weight):
            raise ValueError(
                "indicator_weight must be finite."
            )

        if (
            self.probability_weight < 0.0
            or self.indicator_weight < 0.0
        ):
            raise ValueError(
                "Risk-scoring weights cannot be negative."
            )

        total = (
            self.probability_weight
            + self.indicator_weight
        )

        if total <= 0.0:
            raise ValueError(
                "Risk-scoring weights must have a positive total."
            )

        self.probability_weight /= total
        self.indicator_weight /= total

        logger.info(
            "HTMLRiskScorer initialized. "
            "ML weight=%.2f indicator weight=%.2f",
            self.probability_weight,
            self.indicator_weight,
        )

    # =====================================================================
    # MAIN RISK CALCULATION
    # =====================================================================

    def calculate_risk(
        self,
        probability: float,
        features: Optional[Any] = None,
    ) -> Dict[str, Any]:

        phishing_probability = self._normalize_probability(
            probability
        )

        feature_map = self._normalize_features(
            features
        )

        indicators = self._detect_indicators(
            feature_map
        )

        indicator_score = self._calculate_indicator_score(
            indicators
        )

        # ---------------------------------------------------------------
        # FINAL LOCAL RISK PROBABILITY
        # ---------------------------------------------------------------
        #
        # XGBoost phishing probability remains the primary signal.
        # HTML indicators provide supporting contextual evidence.
        #
        # final = (ML probability * 0.70)
        #       + (indicator score * 0.30)
        #
        # This is an HTML-agent-local risk calculation. The final
        # system-wide verdict is produced by the Decision Fusion Engine.
        #

        base_probability = phishing_probability

        final_probability = (
            base_probability
            * self.probability_weight
            +
            indicator_score
            * self.indicator_weight
        )

        final_probability = self._normalize_probability(
            final_probability
        )

        risk_score = int(
            round(
                final_probability * self.MAX_SCORE
            )
        )

        risk_score = max(
            self.MIN_SCORE,
            min(
                self.MAX_SCORE,
                risk_score,
            ),
        )

        risk_level = self._get_risk_level(
            final_probability
        )

        # ---------------------------------------------------------------
        # EXPLANATION / CONTROLLER COMPATIBILITY FIELDS
        # ---------------------------------------------------------------

        contextual_adjustment = (
            final_probability - base_probability
        )

        severity_summary = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for indicator in indicators:
            severity = str(
                indicator.get(
                    "severity",
                    "low",
                )
            ).lower()

            if severity in severity_summary:
                severity_summary[severity] += 1

        scoring_method = (
            "weighted_ml_probability_plus_html_indicators"
        )

        scoring_explanation = (
            "HTML risk uses the XGBoost phishing probability "
            "as the primary signal and HTML structural indicators "
            "as supporting evidence. "
            f"ML weight={self.probability_weight:.2f}; "
            f"indicator weight={self.indicator_weight:.2f}."
        )

        result = {
            # Canonical risk fields
            "risk_score": risk_score,

            "risk_level": risk_level,

            "phishing_probability": round(
                base_probability,
                4,
            ),

            "indicator_score": round(
                indicator_score,
                4,
            ),

            "combined_probability": round(
                final_probability,
                4,
            ),

            # Explicit probability fields expected by html_agent.py
            "base_probability": round(
                base_probability,
                4,
            ),

            "final_probability": round(
                final_probability,
                4,
            ),

            "contextual_adjustment": round(
                contextual_adjustment,
                4,
            ),

            # Scoring configuration
            "probability_weight": round(
                self.probability_weight,
                4,
            ),

            "indicator_weight": round(
                self.indicator_weight,
                4,
            ),

            "scoring_method": scoring_method,

            "scoring_explanation": scoring_explanation,

            # Indicator evidence
            "triggered_indicators": indicators,

            "risk_indicators": indicators,

            "indicator_details": indicators,

            "indicator_count": len(
                indicators
            ),

            "severity_summary": severity_summary,

            # Signal availability
            "signal_available": True,

            "reason": None,
        }

        logger.info(
            "HTML risk calculation completed. "
            "Probability=%.4f IndicatorScore=%.4f "
            "RiskScore=%d RiskLevel=%s",
            phishing_probability,
            indicator_score,
            risk_score,
            risk_level,
        )

        return result

    # =====================================================================
    # INDICATOR DETECTION
    # =====================================================================

    def _detect_indicators(
        self,
        features: Dict[str, float],
    ) -> List[Dict[str, Any]]:

        indicators: List[Dict[str, Any]] = []

        # -----------------------------------------------------------------
        # Password field
        # -----------------------------------------------------------------

        password_count = self._get_numeric(
            features,
            "password_field_count",
        )

        if password_count > 0:
            indicators.append(
                self._build_indicator(
                    code="password_field",
                    name="Password Field Present",
                    severity="low",
                    description=(
                        "The webpage contains a password input field."
                    ),
                    value=password_count,
                )
            )

        # -----------------------------------------------------------------
        # Email field
        # -----------------------------------------------------------------

        email_count = self._get_numeric(
            features,
            "email_field_count",
        )

        if email_count > 0:
            indicators.append(
                self._build_indicator(
                    code="email_field",
                    name="Email Field Present",
                    severity="low",
                    description=(
                        "The webpage contains an email input field."
                    ),
                    value=email_count,
                )
            )

        # -----------------------------------------------------------------
        # OTP field
        # -----------------------------------------------------------------

        otp_count = self._get_numeric(
            features,
            "otp_field_count",
        )

        if otp_count > 0:
            indicators.append(
                self._build_indicator(
                    code="otp_field",
                    name="OTP Field Present",
                    severity="medium",
                    description=(
                        "The webpage contains an OTP-related input field."
                    ),
                    value=otp_count,
                )
            )

        # -----------------------------------------------------------------
        # Payment field
        # -----------------------------------------------------------------

        payment_count = self._get_numeric(
            features,
            "payment_field_count",
        )

        if payment_count > 0:
            indicators.append(
                self._build_indicator(
                    code="payment_field",
                    name="Payment Field Present",
                    severity="medium",
                    description=(
                        "The webpage contains a payment-related input field."
                    ),
                    value=payment_count,
                )
            )

        # -----------------------------------------------------------------
        # Credential field
        # -----------------------------------------------------------------

        credential_count = self._get_numeric(
            features,
            "credential_field_count",
        )

        if credential_count > 0:
            indicators.append(
                self._build_indicator(
                    code="credential_field",
                    name="Credential Field Detected",
                    severity="medium",
                    description=(
                        "Credential-related input fields were detected."
                    ),
                    value=credential_count,
                )
            )

        # -----------------------------------------------------------------
        # Hidden inputs
        # -----------------------------------------------------------------

        hidden_input_count = self._get_numeric(
            features,
            "hidden_input_count",
        )

        if hidden_input_count > 0:
            indicators.append(
                self._build_indicator(
                    code="hidden_input",
                    name="Hidden Input Detected",
                    severity="medium",
                    description=(
                        f"{int(hidden_input_count)} hidden input "
                        "element(s) were detected."
                    ),
                    value=hidden_input_count,
                )
            )

        # -----------------------------------------------------------------
        # Credential hidden input
        # -----------------------------------------------------------------

        credential_hidden_count = self._get_numeric(
            features,
            "credential_hidden_input_count",
        )

        if credential_hidden_count > 0:
            indicators.append(
                self._build_indicator(
                    code="credential_hidden_input",
                    name="Credential-Related Hidden Input",
                    severity="high",
                    description=(
                        "Credential-related hidden input elements "
                        "were detected."
                    ),
                    value=credential_hidden_count,
                )
            )

        # -----------------------------------------------------------------
        # Suspicious hidden input
        # -----------------------------------------------------------------

        suspicious_hidden_count = self._get_numeric(
            features,
            "suspicious_hidden_input_count",
        )

        if suspicious_hidden_count > 0:
            indicators.append(
                self._build_indicator(
                    code="suspicious_hidden_input",
                    name="Suspicious Hidden Input",
                    severity="high",
                    description=(
                        "Suspicious hidden input elements were detected."
                    ),
                    value=suspicious_hidden_count,
                )
            )

        # -----------------------------------------------------------------
        # Cross-domain form
        # -----------------------------------------------------------------

        cross_domain_forms = self._get_numeric(
            features,
            "cross_domain_form_actions",
        )

        if cross_domain_forms > 0:
            indicators.append(
                self._build_indicator(
                    code="cross_domain_form",
                    name="Cross-Domain Form Submission",
                    severity="high",
                    description=(
                        "One or more forms submit data to another domain."
                    ),
                    value=cross_domain_forms,
                )
            )

        # -----------------------------------------------------------------
        # Cross-domain credential form
        # -----------------------------------------------------------------

        has_credentials = (
            password_count > 0
            or email_count > 0
            or credential_count > 0
        )

        if (
            cross_domain_forms > 0
            and has_credentials
        ):
            indicators.append(
                self._build_indicator(
                    code="cross_domain_credential_form",
                    name="Cross-Domain Credential Submission",
                    severity="critical",
                    description=(
                        "Credential-related fields are present while "
                        "form data is submitted to another domain."
                    ),
                )
            )

        # -----------------------------------------------------------------
        # HTTP form action
        # -----------------------------------------------------------------

        http_forms = self._get_numeric(
            features,
            "http_form_actions",
        )

        if http_forms > 0:
            indicators.append(
                self._build_indicator(
                    code="http_form_action",
                    name="Insecure HTTP Form Action",
                    severity="medium",
                    description=(
                        "One or more forms submit data using HTTP."
                    ),
                    value=http_forms,
                )
            )

        # -----------------------------------------------------------------
        # Empty form action
        # -----------------------------------------------------------------

        empty_form_actions = self._get_numeric(
            features,
            "empty_form_action_count",
        )

        if empty_form_actions > 0:
            indicators.append(
                self._build_indicator(
                    code="empty_form_action",
                    name="Empty Form Action",
                    severity="medium",
                    description=(
                        "One or more forms contain an empty action attribute."
                    ),
                    value=empty_form_actions,
                )
            )

        # -----------------------------------------------------------------
        # Invisible iframe
        # -----------------------------------------------------------------

        invisible_iframes = self._get_numeric(
            features,
            "invisible_iframes",
        )

        if invisible_iframes > 0:
            indicators.append(
                self._build_indicator(
                    code="invisible_iframe",
                    name="Invisible Iframe",
                    severity="high",
                    description=(
                        "Invisible iframe elements were detected."
                    ),
                    value=invisible_iframes,
                )
            )

        # -----------------------------------------------------------------
        # Iframe
        # -----------------------------------------------------------------

        iframe_count = self._get_numeric(
            features,
            "iframe_count",
        )

        if (
            iframe_count > 0
            and invisible_iframes <= 0
        ):
            indicators.append(
                self._build_indicator(
                    code="iframe",
                    name="Iframe Detected",
                    severity="low",
                    description=(
                        "The webpage contains iframe elements."
                    ),
                    value=iframe_count,
                )
            )

        # -----------------------------------------------------------------
        # Suspicious JavaScript
        # -----------------------------------------------------------------

        suspicious_scripts = self._get_numeric(
            features,
            "suspicious_inline_scripts",
        )

        if suspicious_scripts > 0:
            indicators.append(
                self._build_indicator(
                    code="suspicious_javascript",
                    name="Suspicious JavaScript",
                    severity="high",
                    description=(
                        "Suspicious inline JavaScript patterns were detected."
                    ),
                    value=suspicious_scripts,
                )
            )

        # -----------------------------------------------------------------
        # JavaScript redirects
        # -----------------------------------------------------------------

        javascript_redirects = self._get_numeric(
            features,
            "javascript_redirect_count",
        )

        if javascript_redirects > 0:
            indicators.append(
                self._build_indicator(
                    code="javascript_redirect",
                    name="JavaScript Redirect Detected",
                    severity="high",
                    description=(
                        "JavaScript-based redirect behavior was detected."
                    ),
                    value=javascript_redirects,
                )
            )

        # -----------------------------------------------------------------
        # External scripts
        # -----------------------------------------------------------------

        external_scripts = self._get_numeric(
            features,
            "external_scripts_count",
        )

        if external_scripts > 10:
            indicators.append(
                self._build_indicator(
                    code="high_external_script_count",
                    name="High External Script Count",
                    severity="low",
                    description=(
                        "The webpage loads a relatively large number "
                        "of external scripts."
                    ),
                    value=external_scripts,
                )
            )

        # -----------------------------------------------------------------
        # Hidden elements
        # -----------------------------------------------------------------

        hidden_elements = self._get_numeric(
            features,
            "hidden_elements_count",
        )

        if hidden_elements > 5:
            indicators.append(
                self._build_indicator(
                    code="hidden_elements",
                    name="Multiple Hidden HTML Elements",
                    severity="medium",
                    description=(
                        "Multiple hidden HTML elements were detected."
                    ),
                    value=hidden_elements,
                )
            )

        # -----------------------------------------------------------------
        # Meta refresh
        # -----------------------------------------------------------------

        if self._is_positive(
            features,
            "has_meta_refresh",
        ):
            indicators.append(
                self._build_indicator(
                    code="meta_refresh",
                    name="Meta Refresh Detected",
                    severity="medium",
                    description=(
                        "The webpage contains a meta-refresh mechanism."
                    ),
                )
            )

        # -----------------------------------------------------------------
        # Right-click disabled
        # -----------------------------------------------------------------

        if self._is_positive(
            features,
            "right_click_disabled",
        ):
            indicators.append(
                self._build_indicator(
                    code="right_click_disabled",
                    name="Right-Click Disabled",
                    severity="low",
                    description=(
                        "The webpage disables the browser right-click "
                        "context menu."
                    ),
                )
            )

        # -----------------------------------------------------------------
        # Empty links
        # -----------------------------------------------------------------

        empty_links = self._get_numeric(
            features,
            "empty_links_count",
        )

        if empty_links > 10:
            indicators.append(
                self._build_indicator(
                    code="excessive_empty_links",
                    name="Excessive Empty Links",
                    severity="low",
                    description=(
                        "The webpage contains a relatively large number "
                        "of empty links."
                    ),
                    value=empty_links,
                )
            )

        # -----------------------------------------------------------------
        # New keyword features
        # -----------------------------------------------------------------

        login_keywords = self._get_numeric(
            features,
            "login_keyword_count",
        )

        if login_keywords >= 3:
            indicators.append(
                self._build_indicator(
                    code="login_keywords",
                    name="Multiple Login Keywords",
                    severity="medium",
                    description=(
                        "Multiple login-related keywords were detected "
                        "in the webpage."
                    ),
                    value=login_keywords,
                )
            )

        credential_keywords = self._get_numeric(
            features,
            "credential_keyword_count",
        )

        if credential_keywords >= 2:
            indicators.append(
                self._build_indicator(
                    code="credential_keywords",
                    name="Credential-Related Keywords",
                    severity="medium",
                    description=(
                        "Multiple credential-related keywords were detected."
                    ),
                    value=credential_keywords,
                )
            )

        urgency_keywords = self._get_numeric(
            features,
            "urgency_keyword_count",
        )

        if urgency_keywords > 0:
            indicators.append(
                self._build_indicator(
                    code="urgency_keywords",
                    name="Urgency Keywords Detected",
                    severity="medium",
                    description=(
                        "Urgency-related language was detected in the page."
                    ),
                    value=urgency_keywords,
                )
            )

        security_keywords = self._get_numeric(
            features,
            "security_keyword_count",
        )

        if security_keywords >= 4:
            indicators.append(
                self._build_indicator(
                    code="security_keywords",
                    name="Security-Related Keywords",
                    severity="medium",
                    description=(
                        "Multiple security-related keywords were detected."
                    ),
                    value=security_keywords,
                )
            )

        # -----------------------------------------------------------------
        # External images
        # -----------------------------------------------------------------

        external_images = self._get_numeric(
            features,
            "external_images_count",
        )

        if external_images > 10:
            indicators.append(
                self._build_indicator(
                    code="external_images",
                    name="Many External Images",
                    severity="low",
                    description=(
                        "The webpage references a relatively large number "
                        "of external images."
                    ),
                    value=external_images,
                )
            )

        return indicators

    # =====================================================================
    # INDICATOR SCORE
    # =====================================================================

    def _calculate_indicator_score(
        self,
        indicators: List[Dict[str, Any]],
    ) -> float:

        if not indicators:
            return 0.0

        total_weight = 0.0

        for indicator in indicators:
            code = indicator.get("code")

            total_weight += float(
                self.INDICATOR_WEIGHTS.get(
                    code,
                    0,
                )
            )

        return self._normalize_probability(
            total_weight / 100.0
        )

    # =====================================================================
    # RISK LEVEL
    # =====================================================================

    def _get_risk_level(
        self,
        probability: float,
    ) -> str:

        probability = self._normalize_probability(
            probability
        )

        if probability >= self.HIGH_THRESHOLD:
            return "Critical"

        if probability >= self.MEDIUM_THRESHOLD:
            return "High"

        if probability >= self.LOW_THRESHOLD:
            return "Medium"

        return "Low"

    # =====================================================================
    # INDICATOR BUILDER
    # =====================================================================

    def _build_indicator(
        self,
        code: str,
        name: str,
        severity: str,
        description: str,
        value: Optional[float] = None,
    ) -> Dict[str, Any]:

        indicator = {
            "code": code,
            "name": name,
            "severity": severity,
            "description": description,
        }

        if value is not None:
            indicator["value"] = self._safe_number(
                value
            )

        return indicator

    # =====================================================================
    # FEATURE NORMALIZATION
    # =====================================================================

    @staticmethod
    def _normalize_features(
        features: Optional[Any],
    ) -> Dict[str, float]:

        if features is None:
            return {}

        if isinstance(features, dict):

            return {
                str(key): HTMLRiskScorer._safe_number(value)
                for key, value in features.items()
            }

        if isinstance(features, pd.DataFrame):

            if features.empty:
                return {}

            row = features.iloc[0]

            return {
                str(key): HTMLRiskScorer._safe_number(value)
                for key, value in row.items()
            }

        logger.warning(
            "Unsupported HTML risk feature type: %s",
            type(features).__name__,
        )

        return {}

    # =====================================================================
    # FEATURE CHECK
    # =====================================================================

    @staticmethod
    def _is_positive(
        features: Dict[str, float],
        feature_name: str,
    ) -> bool:

        return (
            HTMLRiskScorer._get_numeric(
                features,
                feature_name,
            )
            > 0.0
        )

    # =====================================================================
    # NUMERIC FEATURE ACCESS
    # =====================================================================

    @staticmethod
    def _get_numeric(
        features: Dict[str, float],
        feature_name: str,
    ) -> float:

        if not isinstance(features, dict):
            return 0.0

        return HTMLRiskScorer._safe_number(
            features.get(
                feature_name,
                0.0,
            )
        )

    # =====================================================================
    # SAFE NUMBER
    # =====================================================================

    @staticmethod
    def _safe_number(
        value: Any,
    ) -> float:

        try:
            number = float(value)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return 0.0

        if not np.isfinite(number):
            return 0.0

        return number

    # =====================================================================
    # PROBABILITY NORMALIZATION
    # =====================================================================

    @staticmethod
    def _normalize_probability(
        probability: Any,
    ) -> float:

        probability = HTMLRiskScorer._safe_number(
            probability
        )

        return max(
            HTMLRiskScorer.MIN_PROBABILITY,
            min(
                HTMLRiskScorer.MAX_PROBABILITY,
                probability,
            ),
        )

    # =====================================================================
    # STATUS
    # =====================================================================

    def get_status(self) -> Dict[str, Any]:

        return {
            "status": "ready",

            "probability_weight": self.probability_weight,

            "indicator_weight": self.indicator_weight,

            "risk_thresholds": {
                "low": self.LOW_THRESHOLD,
                "medium": self.MEDIUM_THRESHOLD,
                "high": self.HIGH_THRESHOLD,
            },

            "indicator_count": len(
                self.INDICATOR_WEIGHTS
            ),

            "feature_schema": "50 HTML features",
        }
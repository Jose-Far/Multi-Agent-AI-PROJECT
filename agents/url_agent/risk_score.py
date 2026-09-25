"""
URL AI Agent - Risk Scoring Engine
==================================

Converts URL phishing probability and URL security indicators into
a standardized risk assessment.

IMPORTANT DESIGN PRINCIPLES
---------------------------

1. ML probability remains the PRIMARY risk signal.

2. Structural URL indicators provide supporting evidence.

3. Global URL entropy is NOT treated as strong standalone phishing
   evidence.

4. Domain entropy is more important than path entropy.

5. Path/query entropy are contextual indicators because legitimate
   services can generate random identifiers.

6. Risk-scoring failures must NEVER become:
       legitimate
       risk_score = 0

   An unavailable/error signal must remain distinguishable from a
   genuine low-risk prediction.

7. Output is designed for the Multi-Agent Fusion Engine.
"""

import logging

from typing import Dict, Any, List, Optional


logger = logging.getLogger(__name__)


class URLRiskScorer:
    """
    Production-grade URL risk scoring engine.

    The scorer combines:

        ML phishing probability
                +
        structural URL indicators
                ↓
        URL-specific risk assessment

    The final score remains probability-driven so that the
    Decision Fusion Engine can safely combine this agent with
    HTML, SSL, DNS, Visual, Brand, Behaviour and Threat agents.
    """

    # =================================================================
    # RISK THRESHOLDS
    # =================================================================

    CRITICAL_THRESHOLD = 0.80

    HIGH_THRESHOLD = 0.60

    MEDIUM_THRESHOLD = 0.30

    LOW_THRESHOLD = 0.00

    # =================================================================
    # INDICATOR SEVERITIES
    # =================================================================

    INDICATOR_SEVERITY = {

        "ip_address_in_domain":
            "high",

        "missing_https":
            "medium",

        "high_domain_entropy":
            "high",

        "high_url_entropy_context":
            "low",

        "high_path_entropy_context":
            "low",

        "high_query_entropy_context":
            "low",

        "excessive_special_characters":
            "medium",

        "suspicious_keywords_detected":
            "high",

        "excessive_hyphens_brand_spoofing_risk":
            "medium",

        "abnormal_url_length":
            "low",

        "excessive_subdomains":
            "medium",

        "punycode_domain":
            "medium"
    }

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __init__(
        self,
        critical_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        medium_threshold: Optional[float] = None
    ):
        """
        Initialize the URL risk scorer.

        Args:
            critical_threshold:
                Probability at/above which risk is Critical.

            high_threshold:
                Probability at/above which risk is High.

            medium_threshold:
                Probability at/above which risk is Medium.
        """

        self.critical_threshold = (
            self._validate_threshold(
                critical_threshold,
                self.CRITICAL_THRESHOLD
            )
        )

        self.high_threshold = (
            self._validate_threshold(
                high_threshold,
                self.HIGH_THRESHOLD
            )
        )

        self.medium_threshold = (
            self._validate_threshold(
                medium_threshold,
                self.MEDIUM_THRESHOLD
            )
        )

        # -------------------------------------------------------------
        # Make sure thresholds are logically ordered.
        # -------------------------------------------------------------

        if not (
            self.medium_threshold
            <= self.high_threshold
            <= self.critical_threshold
        ):

            raise ValueError(
                "URL risk thresholds must satisfy: "
                "medium <= high <= critical."
            )

    # =================================================================
    # THRESHOLD VALIDATION
    # =================================================================

    def _validate_threshold(
        self,
        value: Optional[float],
        default: float
    ) -> float:
        """
        Validate a probability threshold.
        """

        if value is None:

            return float(default)

        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            logger.warning(
                "Invalid URL risk threshold '%s'. "
                "Using default %.2f.",
                value,
                default
            )

            return float(default)

        if not (
            0.0 <= numeric_value <= 1.0
        ):

            logger.warning(
                "URL risk threshold %.4f is outside "
                "[0,1]. Using default %.2f.",
                numeric_value,
                default
            )

            return float(default)

        return numeric_value

    # =================================================================
    # MAIN RISK CALCULATION
    # =================================================================

    def calculate_risk(
        self,
        probability: Optional[float],
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the URL-specific risk assessment.

        Args:
            probability:
                ML-generated phishing probability.

            features:
                Normalized URL feature dictionary.

        Returns:
            Dictionary containing:

                risk_score
                risk_level
                triggered_indicators
                indicator_details
                contextual_observations
                probability
                signal_available
        """

        try:

            # =========================================================
            # Validate probability
            # =========================================================

            if probability is None:

                logger.warning(
                    "URL risk scorer received None probability."
                )

                return self._get_unavailable_risk(
                    "URL phishing probability unavailable."
                )

            try:

                normalized_probability = float(
                    probability
                )

            except (
                TypeError,
                ValueError
            ):

                return self._get_unavailable_risk(
                    "Invalid URL phishing probability."
                )

            # ---------------------------------------------------------
            # Protect against NaN / Infinity
            # ---------------------------------------------------------

            if normalized_probability != (
                normalized_probability
            ):

                return self._get_unavailable_risk(
                    "URL phishing probability is NaN."
                )

            if normalized_probability == float(
                "inf"
            ) or normalized_probability == float(
                "-inf"
            ):

                return self._get_unavailable_risk(
                    "URL phishing probability is infinite."
                )

            # ---------------------------------------------------------
            # Clamp probability.
            # ---------------------------------------------------------

            normalized_probability = max(
                0.0,
                min(
                    1.0,
                    normalized_probability
                )
            )

            # =========================================================
            # Base risk score
            # =========================================================

            risk_score = int(
                round(
                    normalized_probability * 100
                )
            )

            # =========================================================
            # Risk level
            # =========================================================

            risk_level = (
                self._determine_risk_level(
                    normalized_probability
                )
            )

            # =========================================================
            # Feature indicators
            # =========================================================

            indicators = (
                self._identify_indicators(
                    features
                )
            )

            # =========================================================
            # Indicator details
            # =========================================================

            indicator_details = (
                self._build_indicator_details(
                    indicators
                )
            )

            # =========================================================
            # Contextual observations
            # =========================================================

            contextual_observations = (
                self._identify_contextual_observations(
                    features
                )
            )

            # =========================================================
            # Build result
            # =========================================================

            return {

                "risk_score":
                    risk_score,

                "risk_level":
                    risk_level,

                "triggered_indicators":
                    indicators,

                "indicator_details":
                    indicator_details,

                "contextual_observations":
                    contextual_observations,

                "probability":
                    round(
                        normalized_probability,
                        6
                    ),

                "signal_available":
                    True,

                "scoring_method":
                    "ml_probability_primary",

                "thresholds": {

                    "critical":
                        self.critical_threshold,

                    "high":
                        self.high_threshold,

                    "medium":
                        self.medium_threshold
                }
            }

        except Exception as e:

            logger.error(
                "Error calculating URL risk score: %s",
                str(e),
                exc_info=True
            )

            return self._get_unavailable_risk(
                f"URL risk scoring failed: {str(e)}"
            )

    # =================================================================
    # RISK LEVEL
    # =================================================================

    def _determine_risk_level(
        self,
        probability: float
    ) -> str:
        """
        Convert phishing probability into a risk category.
        """

        if probability >= (
            self.critical_threshold
        ):

            return "Critical"

        if probability >= (
            self.high_threshold
        ):

            return "High"

        if probability >= (
            self.medium_threshold
        ):

            return "Medium"

        return "Low"

    # =================================================================
    # SAFE NUMERIC VALUE
    # =================================================================

    def _numeric(
        self,
        features: Dict[str, Any],
        key: str,
        default: float = 0.0
    ) -> float:
        """
        Safely retrieve a numeric feature.
        """

        try:

            value = features.get(
                key,
                default
            )

            if value is None:

                return float(default)

            if isinstance(
                value,
                bool
            ):

                return (
                    1.0
                    if value
                    else 0.0
                )

            value = float(
                value
            )

            # NaN
            if value != value:

                return float(default)

            # Infinity
            if value in (
                float("inf"),
                float("-inf")
            ):

                return float(default)

            return value

        except (
            TypeError,
            ValueError
        ):

            return float(default)

    # =================================================================
    # BOOLEAN VALUE
    # =================================================================

    def _boolean(
        self,
        features: Dict[str, Any],
        key: str,
        default: bool = False
    ) -> bool:
        """
        Safely retrieve a boolean-like feature.
        """

        value = features.get(
            key,
            default
        )

        if isinstance(
            value,
            bool
        ):

            return value

        if isinstance(
            value,
            str
        ):

            return value.lower() in {
                "true",
                "1",
                "yes",
                "y"
            }

        try:

            return float(
                value
            ) == 1.0

        except (
            TypeError,
            ValueError
        ):

            return default

    # =================================================================
    # INDICATOR DETECTION
    # =================================================================

    def _identify_indicators(
        self,
        features: Dict[str, Any]
    ) -> List[str]:
        """
        Identify genuine URL security indicators.

        IMPORTANT:

        High path entropy and high query entropy are NOT treated
        as direct phishing indicators.

        They are stored as contextual observations.

        This prevents:

            Google Forms random ID
                    ↓
            high path entropy
                    ↓
            automatic phishing indicator

        """

        indicators: List[str] = []

        if not isinstance(
            features,
            dict
        ):

            return indicators

        # =============================================================
        # IP ADDRESS
        # =============================================================

        if self._boolean(
            features,
            "contains_ip"
        ):

            indicators.append(
                "ip_address_in_domain"
            )

        # =============================================================
        # HTTPS
        # =============================================================

        if not self._boolean(
            features,
            "is_https",
            default=True
        ):

            indicators.append(
                "missing_https"
            )

        # =============================================================
        # DOMAIN ENTROPY
        #
        # IMPORTANT:
        # Domain entropy is stronger than path entropy.
        # =============================================================

        domain_entropy = self._numeric(
            features,
            "domain_entropy"
        )

        if domain_entropy >= 4.5:

            indicators.append(
                "high_domain_entropy"
            )

        # =============================================================
        # GLOBAL URL ENTROPY
        #
        # DO NOT classify this as a strong phishing indicator.
        # =============================================================

        url_entropy = self._numeric(
            features,
            "url_entropy"
        )

        if url_entropy >= 5.0:

            # This is deliberately NOT added to
            # triggered_indicators.

            # It will be returned as a contextual observation.
            pass

        # =============================================================
        # SPECIAL CHARACTERS
        # =============================================================

        special_chars = self._numeric(
            features,
            "num_special_chars"
        )

        if special_chars > 5:

            indicators.append(
                "excessive_special_characters"
            )

        # =============================================================
        # SUSPICIOUS KEYWORDS
        # =============================================================

        suspicious_words = self._numeric(
            features,
            "suspicious_word_count"
        )

        if suspicious_words > 0:

            indicators.append(
                "suspicious_keywords_detected"
            )

        # =============================================================
        # HYPHENS
        # =============================================================

        hyphens = self._numeric(
            features,
            "num_hyphens"
        )

        if hyphens > 3:

            indicators.append(
                "excessive_hyphens_brand_spoofing_risk"
            )

        # =============================================================
        # URL LENGTH
        # =============================================================

        url_length = self._numeric(
            features,
            "url_length"
        )

        if url_length > 100:

            indicators.append(
                "abnormal_url_length"
            )

        # =============================================================
        # SUBDOMAIN DEPTH
        # =============================================================

        subdomain_count = self._numeric(
            features,
            "subdomain_count"
        )

        if subdomain_count >= 4:

            indicators.append(
                "excessive_subdomains"
            )

        # =============================================================
        # PUNYCODE
        # =============================================================

        if self._boolean(
            features,
            "has_punycode"
        ):

            indicators.append(
                "punycode_domain"
            )

        return indicators

    # =================================================================
    # CONTEXTUAL OBSERVATIONS
    # =================================================================

    def _identify_contextual_observations(
        self,
        features: Dict[str, Any]
    ) -> List[str]:
        """
        Identify contextual characteristics that should NOT be treated
        as standalone phishing evidence.

        This is particularly important for dynamic legitimate services.

        Example:

            docs.google.com/forms/d/e/1FAIpQL...

        can naturally have:

            high path entropy
            high global URL entropy

        Those should not automatically become phishing indicators.
        """

        observations: List[str] = []

        if not isinstance(
            features,
            dict
        ):

            return observations

        # =============================================================
        # GLOBAL URL ENTROPY
        # =============================================================

        url_entropy = self._numeric(
            features,
            "url_entropy"
        )

        if url_entropy >= 5.0:

            observations.append(
                "high_url_entropy_context"
            )

        # =============================================================
        # PATH ENTROPY
        # =============================================================

        path_entropy = self._numeric(
            features,
            "path_entropy"
        )

        if path_entropy >= 5.0:

            observations.append(
                "high_path_entropy_context"
            )

        # =============================================================
        # QUERY ENTROPY
        # =============================================================

        query_entropy = self._numeric(
            features,
            "query_entropy"
        )

        if query_entropy >= 5.0:

            observations.append(
                "high_query_entropy_context"
            )

        return observations

    # =================================================================
    # INDICATOR DETAILS
    # =================================================================

    def _build_indicator_details(
        self,
        indicators: List[str]
    ) -> List[Dict[str, str]]:
        """
        Convert indicator names into structured severity information.
        """

        details = []

        for indicator in indicators:

            details.append({

                "indicator":
                    indicator,

                "severity":
                    self.INDICATOR_SEVERITY.get(
                        indicator,
                        "medium"
                    )
            })

        return details

    # =================================================================
    # FALLBACK / UNAVAILABLE
    # =================================================================

    def _get_unavailable_risk(
        self,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Return an unavailable risk result.

        IMPORTANT:

        This is intentionally NOT:

            risk_score = 0
            risk_level = Low

        because that would incorrectly tell Fusion:

            "The URL is safe."

        Instead:

            risk_score = None
            risk_level = Unavailable
            signal_available = False
        """

        return {

            "risk_score":
                None,

            "risk_level":
                "Unavailable",

            "triggered_indicators":
                [],

            "indicator_details":
                [],

            "contextual_observations":
                [],

            "probability":
                None,

            "signal_available":
                False,

            "scoring_method":
                "unavailable",

            "error":
                error_message
        }

    # =================================================================
    # BACKWARD COMPATIBILITY
    # =================================================================

    def get_thresholds(
        self
    ) -> Dict[str, float]:
        """
        Return configured risk thresholds.

        Useful for reports and debugging.
        """

        return {

            "critical":
                self.critical_threshold,

            "high":
                self.high_threshold,

            "medium":
                self.medium_threshold
        }
"""
SSL/TLS Risk Scoring Engine
===========================

Production-grade local risk scoring engine for the SSL/TLS AI Agent.

Responsibilities
----------------
1. Validate SSL feature evidence.
2. Detect strong SSL/TLS security weaknesses.
3. Detect weak/contextual certificate observations.
4. Detect protective SSL/TLS properties.
5. Combine ML phishing probability with SSL policy evidence.
6. Produce a normalized 0-100 SSL risk score.
7. Clearly distinguish:
       - actual risk indicators
       - contextual observations
       - protective indicators
8. Never treat unavailable SSL evidence as legitimate.
9. Never treat a short-lived certificate as standalone phishing evidence.
10. Return a Fusion-Engine-friendly result.

Architecture
------------

SSL Predictor
      |
      | phishing_probability
      v
SSL Risk Scorer
      |
      +--> Strong Risk Indicators
      |
      +--> Contextual Observations
      |
      +--> Protective Indicators
      |
      v
Final SSL Probability
      |
      v
Risk Score 0-100
      |
      v
Decision Fusion Engine
"""

import logging
from typing import Dict, Any, List, Optional


logger = logging.getLogger(__name__)


class SSLRiskScorer:
    """
    Production-grade SSL/TLS local risk scoring engine.

    The scorer evaluates SSL/TLS evidence independently from the
    central multi-agent fusion engine.

    Important design principles
    ----------------------------

    A certificate being short-lived does NOT automatically mean
    that a website is malicious.

    Similarly:

        free automated CA
        recently issued certificate
        short certificate lifetime

    are contextual observations rather than strong phishing
    indicators.

    Strong indicators include:

        expired certificate
        self-signed certificate
        suspicious CA
        weak TLS
        vulnerable cipher
        severely unhealthy cryptography
        explicit insecure connection
        absence of SSL when SSL was successfully checked

    Missing evidence is handled differently from a confirmed
    insecure state.

    Example:

        has_ssl = False

    is different from:

        has_ssl = missing

    Missing means the SSL signal may be unavailable.

    False means the collector explicitly determined that SSL
    is absent.
    """

    # =====================================================================
    # METADATA
    # =====================================================================

    AGENT_NAME = "SSL_AI_Agent"

    SCORER_VERSION = "3.0"

    # =====================================================================
    # RISK THRESHOLDS
    # =====================================================================

    CRITICAL_THRESHOLD = 0.80

    HIGH_THRESHOLD = 0.60

    MEDIUM_THRESHOLD = 0.30

    # =====================================================================
    # PROBABILITY LIMITS
    # =====================================================================

    MIN_PROBABILITY = 0.0

    MAX_PROBABILITY = 1.0

    # =====================================================================
    # POLICY ADJUSTMENT LIMITS
    # =====================================================================

    MAX_POSITIVE_ADJUSTMENT = 0.30

    MAX_NEGATIVE_ADJUSTMENT = -0.15

    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    def __init__(self) -> None:
        """
        Initialize the SSL risk scoring policy.

        The policy intentionally separates three classes of evidence:

        1. Strong risk signals
        2. Contextual observations
        3. Protective signals

        This prevents a normal short-lived certificate from being
        incorrectly classified as a phishing indicator.
        """

        # ================================================================
        # STRONG RISK WEIGHTS
        # ================================================================

        self.strong_risk_weights: Dict[str, float] = {
            "certificate_expired": 0.25,
            "self_signed_certificate": 0.25,
            "suspicious_certificate_authority": 0.20,
            "weak_tls_protocol": 0.20,
            "vulnerable_cipher_suite": 0.15,
            "poor_cryptographic_health": 0.15,
            "no_ssl_detected": 0.20,
            "insecure_connection": 0.15,
        }

        # ================================================================
        # CONTEXTUAL WEIGHTS
        # ================================================================

        self.contextual_weights: Dict[str, float] = {
            "short_lived_certificate_context": 0.03,
            "recently_issued_certificate_context": 0.03,
            "free_automated_ca_context": 0.02,
        }

        # ================================================================
        # PROTECTIVE WEIGHTS
        # ================================================================

        self.protective_weights: Dict[str, float] = {
            "ssl_tls_present": -0.03,
            "trusted_certificate_authority": -0.05,
            "secure_connection": -0.05,
            "strong_cipher_key_length": -0.05,
            "strong_cryptographic_health": -0.04,
        }

        logger.info(
            "%s risk scorer initialized | version=%s",
            self.AGENT_NAME,
            self.SCORER_VERSION,
        )

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def calculate_risk(
        self,
        probability: float,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate the localized SSL/TLS risk.

        Parameters
        ----------
        probability:
            Phishing probability generated by SSLPredictor.

        features:
            Normalized SSL feature dictionary.

        Returns
        -------
        Dict[str, Any]
            Standardized SSL risk result.

        Important
        ---------
        If SSL evidence is unavailable, this method returns:

            signal_available = False
            risk_score = None
            risk_level = "Unknown"

        It will NEVER convert unavailable evidence into:

            risk_score = 0
            risk_level = "Low"

        because:

            unavailable != legitimate
        """

        try:

            # ============================================================
            # STEP 1 — VALIDATE SSL EVIDENCE
            # ============================================================

            validation_error = self._validate_features(features)

            if validation_error:

                logger.warning(
                    "SSL risk scoring unavailable: %s",
                    validation_error,
                )

                return self._get_unavailable_risk(
                    reason=validation_error
                )

            # ============================================================
            # STEP 2 — NORMALIZE ML PROBABILITY
            # ============================================================

            base_probability = self._clamp_probability(
                probability
            )

            # ============================================================
            # STEP 3 — IDENTIFY STRONG RISK INDICATORS
            # ============================================================

            risk_indicators = self._identify_strong_indicators(
                features
            )

            # ============================================================
            # STEP 4 — IDENTIFY CONTEXTUAL OBSERVATIONS
            # ============================================================

            contextual_observations = (
                self._identify_contextual_indicators(
                    features
                )
            )

            # ============================================================
            # STEP 5 — IDENTIFY PROTECTIVE INDICATORS
            # ============================================================

            protective_indicators = (
                self._identify_protective_indicators(
                    features
                )
            )

            # ============================================================
            # STEP 6 — STRONG RISK ADJUSTMENT
            # ============================================================

            strong_risk_adjustment = (
                self._calculate_strong_risk_adjustment(
                    risk_indicators
                )
            )

            # ============================================================
            # STEP 7 — CONTEXTUAL ADJUSTMENT
            # ============================================================

            contextual_adjustment = (
                self._calculate_contextual_adjustment(
                    contextual_observations
                )
            )

            # ============================================================
            # STEP 8 — CERTIFICATE CONTEXT
            # ============================================================

            certificate_context_adjustment = (
                self._calculate_certificate_context_adjustment(
                    features
                )
            )

            # ============================================================
            # STEP 9 — PROTECTIVE ADJUSTMENT
            # ============================================================

            protective_adjustment = (
                self._calculate_protective_adjustment(
                    features
                )
            )

            # ============================================================
            # STEP 10 — TOTAL POLICY ADJUSTMENT
            # ============================================================

            total_adjustment = (
                strong_risk_adjustment
                + contextual_adjustment
                + certificate_context_adjustment
                + protective_adjustment
            )

            # ============================================================
            # STEP 11 — SAFETY CAP
            # ============================================================

            total_adjustment = max(
                self.MAX_NEGATIVE_ADJUSTMENT,
                min(
                    self.MAX_POSITIVE_ADJUSTMENT,
                    total_adjustment,
                ),
            )

            # ============================================================
            # STEP 12 — FINAL PROBABILITY
            # ============================================================

            final_probability = self._clamp_probability(
                base_probability + total_adjustment
            )

            # ============================================================
            # STEP 13 — FINAL RISK SCORE
            # ============================================================

            risk_score = int(
                round(
                    final_probability * 100
                )
            )

            risk_score = max(
                0,
                min(
                    100,
                    risk_score,
                ),
            )

            # ============================================================
            # STEP 14 — RISK LEVEL
            # ============================================================

            risk_level = self._determine_risk_level(
                final_probability
            )

            # ============================================================
            # STEP 15 — SEVERITY
            # ============================================================

            severity_summary = self._build_severity_summary(
                features=features,
                risk_indicators=risk_indicators,
                contextual_observations=contextual_observations,
            )

            # ============================================================
            # STEP 16 — EXPLANATION
            # ============================================================

            scoring_explanation = (
                self._build_scoring_explanation(
                    base_probability=base_probability,
                    strong_risk_adjustment=strong_risk_adjustment,
                    contextual_adjustment=(
                        contextual_adjustment
                        + certificate_context_adjustment
                    ),
                    protective_adjustment=(
                        protective_adjustment
                    ),
                    final_probability=final_probability,
                    risk_indicators=risk_indicators,
                    contextual_observations=(
                        contextual_observations
                    ),
                    protective_indicators=(
                        protective_indicators
                    ),
                )
            )

            logger.info(
                "SSL risk calculated | "
                "Base=%.4f | Strong=%.4f | Context=%.4f | "
                "Protective=%.4f | Final=%.4f | Score=%d | Level=%s",
                base_probability,
                strong_risk_adjustment,
                contextual_adjustment
                + certificate_context_adjustment,
                protective_adjustment,
                final_probability,
                risk_score,
                risk_level,
            )

            # ============================================================
            # STEP 17 — STANDARDIZED RESULT
            # ============================================================

            return {
                "agent_name": self.AGENT_NAME,
                "scorer_version": self.SCORER_VERSION,

                "signal_available": True,

                "risk_score": risk_score,

                "risk_level": risk_level,

                "base_probability": round(
                    base_probability,
                    4,
                ),

                "contextual_adjustment": round(
                    total_adjustment,
                    4,
                ),

                "final_probability": round(
                    final_probability,
                    4,
                ),

                # --------------------------------------------------------
                # IMPORTANT:
                # Actual security weaknesses.
                # --------------------------------------------------------

                "risk_indicators": risk_indicators,

                # --------------------------------------------------------
                # IMPORTANT:
                # Weak/contextual observations.
                # --------------------------------------------------------

                "contextual_observations": (
                    contextual_observations
                ),

                # --------------------------------------------------------
                # Protective evidence.
                # --------------------------------------------------------

                "protective_indicators": (
                    protective_indicators
                ),

                # --------------------------------------------------------
                # Backward compatibility.
                #
                # Existing code may still expect:
                #
                # triggered_indicators
                #
                # Keep it temporarily.
                # --------------------------------------------------------

                "triggered_indicators": (
                    risk_indicators
                    + contextual_observations
                ),

                "severity_summary": severity_summary,

                "scoring_method": (
                    "ml_probability_plus_ssl_policy_context"
                ),

                "scoring_components": {
                    "ml_probability": round(
                        base_probability,
                        4,
                    ),
                    "strong_risk_adjustment": round(
                        strong_risk_adjustment,
                        4,
                    ),
                    "contextual_adjustment": round(
                        contextual_adjustment
                        + certificate_context_adjustment,
                        4,
                    ),
                    "protective_adjustment": round(
                        protective_adjustment,
                        4,
                    ),
                    "total_adjustment": round(
                        total_adjustment,
                        4,
                    ),
                },

                "scoring_explanation": (
                    scoring_explanation
                ),
            }

        except Exception as exc:

            logger.error(
                "SSL risk calculation failed: %s",
                str(exc),
                exc_info=True,
            )

            return self._get_unavailable_risk(
                reason=(
                    f"SSL risk calculation failed: {exc}"
                )
            )

    # =====================================================================
    # VALIDATION
    # =====================================================================

    @staticmethod
    def _validate_features(
        features: Dict[str, Any]
    ) -> Optional[str]:
        """
        Validate the SSL feature payload.

        IMPORTANT:

        Missing optional features are allowed.

        Completely missing SSL evidence is not.

        Explicitly reported collection failures are also rejected.
        """

        if features is None:
            return "SSL feature payload is missing."

        if not isinstance(features, dict):
            return (
                "SSL feature payload must be a dictionary."
            )

        if not features:
            return "SSL feature payload is empty."

        # ---------------------------------------------------------------
        # Explicit error fields
        # ---------------------------------------------------------------

        error_keys = {
            "error",
            "errors",
            "exception",
            "collection_error",
            "extraction_error",
            "error_message",
        }

        for key in error_keys:

            if key not in features:
                continue

            value = features.get(key)

            if value:

                return (
                    f"SSL extraction error: {value}"
                )

        # ---------------------------------------------------------------
        # Explicit success=False
        # ---------------------------------------------------------------

        if features.get("success") is False:

            return (
                "SSL extraction reported success=False."
            )

        # ---------------------------------------------------------------
        # Explicit status
        # ---------------------------------------------------------------

        status = str(
            features.get(
                "status",
                "",
            )
        ).strip().lower()

        if status in {
            "error",
            "failed",
            "failure",
            "timeout",
            "unavailable",
        }:

            return (
                f"SSL extraction status is '{status}'."
            )

        # ---------------------------------------------------------------
        # Explicit signal unavailable
        # ---------------------------------------------------------------

        if features.get(
            "signal_available"
        ) is False:

            return (
                "SSL signal is marked unavailable."
            )

        return None

    # =====================================================================
    # STRONG RISK INDICATORS
    # =====================================================================

    def _identify_strong_indicators(
        self,
        features: Dict[str, Any],
    ) -> List[str]:
        """
        Detect confirmed SSL/TLS security weaknesses.

        IMPORTANT:

        We only classify a feature as a strong risk when the feature
        is explicitly present and evaluates to True.

        Missing data does NOT become a security failure.
        """

        indicators: List[str] = []

        # ---------------------------------------------------------------
        # Certificate expiration
        # ---------------------------------------------------------------

        if self._is_true(
            features.get("is_expired")
        ):

            indicators.append(
                "certificate_expired"
            )

        # ---------------------------------------------------------------
        # Self-signed certificate
        # ---------------------------------------------------------------

        if self._is_true(
            features.get("is_self_signed")
        ):

            indicators.append(
                "self_signed_certificate"
            )

        # ---------------------------------------------------------------
        # Suspicious CA
        # ---------------------------------------------------------------

        if self._is_true(
            features.get("is_suspicious_ca")
        ):

            indicators.append(
                "suspicious_certificate_authority"
            )

        # ---------------------------------------------------------------
        # Weak TLS protocol
        # ---------------------------------------------------------------

        if self._is_true(
            features.get("is_weak_protocol")
        ):

            indicators.append(
                "weak_tls_protocol"
            )

        # ---------------------------------------------------------------
        # Vulnerable cipher
        # ---------------------------------------------------------------

        if self._is_true(
            features.get("is_vulnerable_cipher")
        ):

            indicators.append(
                "vulnerable_cipher_suite"
            )

        # ---------------------------------------------------------------
        # No SSL
        #
        # IMPORTANT FIX:
        #
        # We only report no_ssl_detected if the feature explicitly
        # exists and is False.
        #
        # Missing has_ssl != False.
        # ---------------------------------------------------------------

        if (
            "has_ssl" in features
            and features.get("has_ssl") is not None
            and not self._is_true(
                features.get("has_ssl")
            )
        ):

            indicators.append(
                "no_ssl_detected"
            )

        # ---------------------------------------------------------------
        # Insecure connection
        #
        # Same missing-data protection.
        # ---------------------------------------------------------------

        if (
            "is_secure_connection" in features
            and features.get(
                "is_secure_connection"
            ) is not None
            and not self._is_true(
                features.get(
                    "is_secure_connection"
                )
            )
        ):

            indicators.append(
                "insecure_connection"
            )

        # ---------------------------------------------------------------
        # Cryptographic health
        # ---------------------------------------------------------------

        crypto_health = self._safe_float(
            features.get(
                "crypto_health_score"
            ),
            default=100.0,
        )

        if crypto_health < 50.0:

            indicators.append(
                "poor_cryptographic_health"
            )

        return indicators

    # =====================================================================
    # CONTEXTUAL OBSERVATIONS
    # =====================================================================

    def _identify_contextual_indicators(
        self,
        features: Dict[str, Any],
    ) -> List[str]:
        """
        Detect weak/contextual SSL observations.

        These observations are NOT standalone phishing evidence.

        Example:

            short-lived certificate

        is reported as:

            short_lived_certificate_context

        rather than:

            suspicious_certificate
        """

        observations: List[str] = []

        if self._is_true(
            features.get("is_short_lived")
        ):

            observations.append(
                "short_lived_certificate_context"
            )

        if self._is_true(
            features.get("is_recently_issued")
        ):

            observations.append(
                "recently_issued_certificate_context"
            )

        if self._is_true(
            features.get("is_free_automated_ca")
        ):

            observations.append(
                "free_automated_ca_context"
            )

        return observations

    # =====================================================================
    # PROTECTIVE INDICATORS
    # =====================================================================

    def _identify_protective_indicators(
        self,
        features: Dict[str, Any],
    ) -> List[str]:
        """
        Detect evidence supporting a healthy SSL/TLS configuration.
        """

        indicators: List[str] = []

        # ---------------------------------------------------------------
        # SSL present
        # ---------------------------------------------------------------

        if self._is_true(
            features.get("has_ssl")
        ):

            indicators.append(
                "ssl_tls_present"
            )

        # ---------------------------------------------------------------
        # Trusted CA
        # ---------------------------------------------------------------

        if self._is_true(
            features.get("is_trusted_ca")
        ):

            indicators.append(
                "trusted_certificate_authority"
            )

        # ---------------------------------------------------------------
        # Secure connection
        # ---------------------------------------------------------------

        if self._is_true(
            features.get(
                "is_secure_connection"
            )
        ):

            indicators.append(
                "secure_connection"
            )

        # ---------------------------------------------------------------
        # Strong crypto health
        # ---------------------------------------------------------------

        crypto_health = self._safe_float(
            features.get(
                "crypto_health_score"
            ),
            default=0.0,
        )

        if crypto_health >= 90.0:

            indicators.append(
                "strong_cryptographic_health"
            )

        # ---------------------------------------------------------------
        # Strong cipher
        # ---------------------------------------------------------------

        cipher_strength = self._safe_float(
            features.get(
                "cipher_strength_bits"
            ),
            default=0.0,
        )

        vulnerable_cipher = self._is_true(
            features.get(
                "is_vulnerable_cipher"
            )
        )

        if (
            cipher_strength >= 256.0
            and not vulnerable_cipher
        ):

            indicators.append(
                "strong_cipher_key_length"
            )

        # ---------------------------------------------------------------
        # Modern TLS
        #
        # Only report this when the protocol feature is actually
        # available.
        # ---------------------------------------------------------------

        if (
            "is_weak_protocol" in features
            and features.get(
                "is_weak_protocol"
            ) is not None
            and not self._is_true(
                features.get(
                    "is_weak_protocol"
                )
            )
        ):

            indicators.append(
                "modern_tls_configuration"
            )

        return indicators

    # =====================================================================
    # STRONG RISK ADJUSTMENT
    # =====================================================================

    def _calculate_strong_risk_adjustment(
        self,
        risk_indicators: List[str],
    ) -> float:
        """
        Calculate additional probability from confirmed SSL risks.

        Multiple indicators are accumulated but the final policy
        adjustment is capped later.

        This fixes an important issue in the previous version where
        strong_risk_weights existed but were never actually used.
        """

        adjustment = 0.0

        for indicator in risk_indicators:

            adjustment += self.strong_risk_weights.get(
                indicator,
                0.0,
            )

        return adjustment

    # =====================================================================
    # CONTEXTUAL ADJUSTMENT
    # =====================================================================

    def _calculate_contextual_adjustment(
        self,
        contextual_observations: List[str],
    ) -> float:
        """
        Calculate weak contextual adjustment.

        Contextual observations have deliberately small weights.
        """

        adjustment = 0.0

        for observation in contextual_observations:

            adjustment += self.contextual_weights.get(
                observation,
                0.0,
            )

        return adjustment

    # =====================================================================
    # CERTIFICATE CONTEXT
    # =====================================================================

    def _calculate_certificate_context_adjustment(
        self,
        features: Dict[str, Any],
    ) -> float:
        """
        Apply context-aware certificate logic.

        Critical rule:

            short-lived + trusted CA

        does NOT become a meaningful risk signal.

        However:

            short-lived
            +
            suspicious CA/self-signed

        can add a small contextual amount.
        """

        adjustment = 0.0

        short_lived = self._is_true(
            features.get(
                "is_short_lived"
            )
        )

        suspicious_ca = self._is_true(
            features.get(
                "is_suspicious_ca"
            )
        )

        self_signed = self._is_true(
            features.get(
                "is_self_signed"
            )
        )

        trusted_ca = self._is_true(
            features.get(
                "is_trusted_ca"
            )
        )

        # ---------------------------------------------------------------
        # Short-lived + suspicious identity
        # ---------------------------------------------------------------

        if short_lived and (
            suspicious_ca
            or self_signed
        ):

            adjustment += 0.05

        # ---------------------------------------------------------------
        # Short-lived + trusted CA
        #
        # DO NOT penalize.
        # ---------------------------------------------------------------

        elif short_lived and trusted_ca:

            adjustment -= 0.01

        # ---------------------------------------------------------------
        # Recently issued + suspicious CA
        # ---------------------------------------------------------------

        recently_issued = self._is_true(
            features.get(
                "is_recently_issued"
            )
        )

        if recently_issued and suspicious_ca:

            adjustment += 0.04

        return adjustment

    # =====================================================================
    # PROTECTIVE ADJUSTMENT
    # =====================================================================

    def _calculate_protective_adjustment(
        self,
        features: Dict[str, Any],
    ) -> float:
        """
        Calculate probability reduction from strong protective
        SSL/TLS properties.
        """

        adjustment = 0.0

        # SSL/TLS present
        if self._is_true(
            features.get("has_ssl")
        ):

            adjustment += self.protective_weights[
                "ssl_tls_present"
            ]

        # Trusted CA
        if self._is_true(
            features.get("is_trusted_ca")
        ):

            adjustment += self.protective_weights[
                "trusted_certificate_authority"
            ]

        # Secure connection
        if self._is_true(
            features.get(
                "is_secure_connection"
            )
        ):

            adjustment += self.protective_weights[
                "secure_connection"
            ]

        # Strong cipher
        cipher_strength = self._safe_float(
            features.get(
                "cipher_strength_bits"
            ),
            default=0.0,
        )

        vulnerable_cipher = self._is_true(
            features.get(
                "is_vulnerable_cipher"
            )
        )

        if (
            cipher_strength >= 256.0
            and not vulnerable_cipher
        ):

            adjustment += self.protective_weights[
                "strong_cipher_key_length"
            ]

        # Healthy cryptography
        crypto_health = self._safe_float(
            features.get(
                "crypto_health_score"
            ),
            default=0.0,
        )

        if crypto_health >= 90.0:

            adjustment += self.protective_weights[
                "strong_cryptographic_health"
            ]

        return adjustment

    # =====================================================================
    # RISK LEVEL
    # =====================================================================

    def _determine_risk_level(
        self,
        probability: float,
    ) -> str:
        """
        Convert probability into risk category.
        """

        if probability >= self.CRITICAL_THRESHOLD:
            return "Critical"

        if probability >= self.HIGH_THRESHOLD:
            return "High"

        if probability >= self.MEDIUM_THRESHOLD:
            return "Medium"

        return "Low"

    # =====================================================================
    # SEVERITY SUMMARY
    # =====================================================================

    def _build_severity_summary(
        self,
        features: Dict[str, Any],
        risk_indicators: List[str],
        contextual_observations: List[str],
    ) -> Dict[str, Any]:
        """
        Build a severity summary.

        IMPORTANT:

        Contextual observations are counted separately from
        confirmed security weaknesses.

        Therefore:

            short_lived_certificate_context

        does not become a strong risk indicator.
        """

        critical = 0
        high = 0
        medium = 0
        low = 0

        # ---------------------------------------------------------------
        # Confirmed certificate weaknesses
        # ---------------------------------------------------------------

        if "certificate_expired" in risk_indicators:
            critical += 1

        if "self_signed_certificate" in risk_indicators:
            high += 1

        if (
            "suspicious_certificate_authority"
            in risk_indicators
        ):
            high += 1

        # ---------------------------------------------------------------
        # TLS weaknesses
        # ---------------------------------------------------------------

        if "weak_tls_protocol" in risk_indicators:
            high += 1

        if "vulnerable_cipher_suite" in risk_indicators:
            high += 1

        # ---------------------------------------------------------------
        # Cryptographic health
        # ---------------------------------------------------------------

        crypto_health = self._safe_float(
            features.get(
                "crypto_health_score"
            ),
            default=100.0,
        )

        if crypto_health < 30.0:
            high += 1

        elif crypto_health < 50.0:
            medium += 1

        # ---------------------------------------------------------------
        # No SSL / insecure connection
        # ---------------------------------------------------------------

        if "no_ssl_detected" in risk_indicators:
            high += 1

        if "insecure_connection" in risk_indicators:
            high += 1

        # ---------------------------------------------------------------
        # Contextual observations
        #
        # These are NOT counted as strong risk indicators.
        # ---------------------------------------------------------------

        contextual_count = len(
            contextual_observations
        )

        # If no actual risks and only context exists,
        # classify the context as low-severity observation.
        if (
            not risk_indicators
            and contextual_count > 0
        ):
            low = contextual_count

        elif not risk_indicators and not contextual_count:
            low = 1

        return {
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,

            # Actual security problems only
            "risk_indicator_count": len(
                risk_indicators
            ),

            # Weak/contextual observations only
            "contextual_observation_count": (
                contextual_count
            ),

            # Backward-compatible field
            "total_indicators": (
                len(risk_indicators)
                + contextual_count
            ),
        }

    # =====================================================================
    # SCORING EXPLANATION
    # =====================================================================

    @staticmethod
    def _build_scoring_explanation(
        base_probability: float,
        strong_risk_adjustment: float,
        contextual_adjustment: float,
        protective_adjustment: float,
        final_probability: float,
        risk_indicators: List[str],
        contextual_observations: List[str],
        protective_indicators: List[str],
    ) -> str:
        """
        Generate analyst-friendly scoring explanation.
        """

        parts: List[str] = []

        parts.append(
            f"Base SSL phishing probability was "
            f"{base_probability:.2f}."
        )

        # Strong risk
        if strong_risk_adjustment > 0:

            parts.append(
                f"Confirmed SSL/TLS risk indicators "
                f"increased the probability by "
                f"{strong_risk_adjustment:.2f}."
            )

        # Context
        if contextual_adjustment > 0:

            parts.append(
                f"Contextual certificate observations "
                f"added {contextual_adjustment:.2f}."
            )

        elif contextual_adjustment < 0:

            parts.append(
                f"Trusted certificate context reduced "
                f"the probability by "
                f"{abs(contextual_adjustment):.2f}."
            )

        # Protective
        if protective_adjustment < 0:

            parts.append(
                f"Protective SSL/TLS properties reduced "
                f"the probability by "
                f"{abs(protective_adjustment):.2f}."
            )

        # Strong indicators
        if risk_indicators:

            parts.append(
                "Confirmed risk indicators: "
                + ", ".join(
                    risk_indicators
                )
                + "."
            )

        # Contextual observations
        if contextual_observations:

            parts.append(
                "Contextual observations: "
                + ", ".join(
                    contextual_observations
                )
                + "."
            )

        # Protective indicators
        if protective_indicators:

            parts.append(
                "Protective indicators: "
                + ", ".join(
                    protective_indicators
                )
                + "."
            )

        parts.append(
            f"Final SSL phishing probability is "
            f"{final_probability:.2f}."
        )

        return " ".join(parts)

    # =====================================================================
    # UNAVAILABLE RESPONSE
    # =====================================================================

    def _get_unavailable_risk(
        self,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Return a safe unavailable result.

        IMPORTANT:

            unavailable != legitimate

        Therefore:

            risk_score = None
            risk_level = Unknown
            signal_available = False
        """

        return {
            "agent_name": self.AGENT_NAME,

            "scorer_version": self.SCORER_VERSION,

            "signal_available": False,

            "risk_score": None,

            "risk_level": "Unknown",

            "base_probability": None,

            "contextual_adjustment": None,

            "final_probability": None,

            "risk_indicators": [],

            "contextual_observations": [],

            "protective_indicators": [],

            # Backward compatibility
            "triggered_indicators": [],

            "severity_summary": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "risk_indicator_count": 0,
                "contextual_observation_count": 0,
                "total_indicators": 0,
            },

            "scoring_method": "unavailable",

            "scoring_explanation": (
                "SSL risk scoring was not performed "
                "because the SSL signal was unavailable: "
                f"{reason}"
            ),

            "reason": reason,
        }

    # =====================================================================
    # BOOLEAN CONVERSION
    # =====================================================================

    @staticmethod
    def _is_true(
        value: Any,
    ) -> bool:
        """
        Safely convert boolean-like values.

        Supported examples:

            True
            False
            1
            0
            "true"
            "false"
            "yes"
            "no"
            "1"
            "0"
        """

        if isinstance(
            value,
            bool,
        ):
            return value

        if value is None:
            return False

        if isinstance(
            value,
            (int, float),
        ):
            return value != 0

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value
                .strip()
                .lower()
            )

            return normalized in {
                "true",
                "1",
                "yes",
                "y",
                "on",
            }

        return bool(value)

    # =====================================================================
    # SAFE FLOAT
    # =====================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to finite float.
        """

        try:

            result = float(value)

            if result != result:
                return default

            if result in (
                float("inf"),
                float("-inf"),
            ):
                return default

            return result

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):

            return default

    # =====================================================================
    # PROBABILITY CLAMP
    # =====================================================================

    @classmethod
    def _clamp_probability(
        cls,
        probability: Any,
    ) -> float:
        """
        Clamp probability to [0.0, 1.0].
        """

        value = cls._safe_float(
            probability,
            default=0.0,
        )

        return max(
            cls.MIN_PROBABILITY,
            min(
                cls.MAX_PROBABILITY,
                value,
            ),
        )

    # =====================================================================
    # STATUS
    # =====================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return scorer configuration and policy status.
        """

        return {
            "agent_name": self.AGENT_NAME,

            "scorer_version": self.SCORER_VERSION,

            "status": "ready",

            "thresholds": {
                "medium": self.MEDIUM_THRESHOLD,
                "high": self.HIGH_THRESHOLD,
                "critical": self.CRITICAL_THRESHOLD,
            },

            "policy": {
                "short_lived_certificate":
                    "contextual_only",

                "recently_issued_certificate":
                    "contextual_only",

                "free_automated_ca":
                    "contextual_only",

                "trusted_ca":
                    "protective_signal",

                "self_signed":
                    "strong_risk_signal",

                "expired_certificate":
                    "strong_risk_signal",

                "suspicious_ca":
                    "strong_risk_signal",

                "weak_tls":
                    "strong_risk_signal",

                "vulnerable_cipher":
                    "strong_risk_signal",

                "missing_ssl_evidence":
                    "unavailable_not_legitimate",
            },
        }
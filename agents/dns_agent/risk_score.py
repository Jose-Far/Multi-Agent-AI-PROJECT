"""
DNS Security AI Agent - Risk Scoring Engine
============================================

Converts DNS machine-learning probability and DNS telemetry into a
standardized DNS-specific cybersecurity risk assessment.

The DNS risk score combines:

    1. XGBoost phishing probability
    2. DNS threat indicators
    3. DNS security posture

The ML probability remains the primary signal, while strong DNS
evidence is allowed to raise the risk score when the model probability
is inconsistent with clearly detected DNS indicators.

IMPORTANT
---------

This is a DNS-specific risk signal.

It is NOT the final system-wide phishing decision.

The final Multi-Agent system will later combine:

    URL
    HTML
    SSL
    DNS
    Visual
    Brand
    Behaviour
    Reputation
    Threat Intelligence
    Malware

through the Decision Fusion / Consensus Engine.

Author:
    Multi-Agent AI Cybersecurity Analyst

Agent:
    DNS Security AI Agent

Version:
    2.0.0
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# VERSION
# =============================================================================

RISK_SCORER_VERSION = "2.0.0"


# =============================================================================
# DNS RISK SCORER
# =============================================================================

class DNSRiskScorer:
    """
    Production-grade DNS risk scoring engine.

    The scorer combines:

        ML phishing probability
                    +
        DNS threat evidence
                    +
        DNS security posture
                    ↓
        DNS-specific risk score

    The score is intended for downstream multi-agent fusion.
    """

    # =========================================================================
    # 1. RISK THRESHOLDS
    # =========================================================================

    DEFAULT_CRITICAL_THRESHOLD = 0.80
    DEFAULT_HIGH_THRESHOLD = 0.60
    DEFAULT_MEDIUM_THRESHOLD = 0.30

    # =========================================================================
    # 2. SCORE LIMITS
    # =========================================================================

    MIN_RISK_SCORE = 0
    MAX_RISK_SCORE = 100

    # =========================================================================
    # 3. TTL POLICY
    # =========================================================================

    LOW_TTL_SECONDS = 300

    # =========================================================================
    # 4. POSTURE LIMITS
    # =========================================================================

    MIN_POSTURE_SCORE = 0
    MAX_POSTURE_SCORE = 100

    # =========================================================================
    # 5. INDICATOR SCORE CONTRIBUTIONS
    # =========================================================================
    #
    # These are NOT added blindly.
    #
    # The final scoring algorithm:
    #
    #     base score
    #          +
    #     evidence adjustment
    #          +
    #     posture adjustment
    #
    # A cap prevents DNS indicators from overwhelming the ML model.
    #
    # Strong indicators also establish minimum risk floors.
    # =========================================================================

    INDICATOR_ADJUSTMENTS = {
        "Critical": 18,
        "High": 10,
        "Medium": 4,
        "Low": 1,
    }

    MAX_POSITIVE_INDICATOR_ADJUSTMENT = 45

    # =========================================================================
    # 6. RISK FLOORS
    # =========================================================================
    #
    # These floors solve the exact problem observed in testing:
    #
    # ML probability may be extremely low while strong DNS evidence
    # clearly indicates a suspicious infrastructure posture.
    #
    # The floor does NOT mean "DNS proves phishing".
    #
    # It means:
    #
    #     "DNS evidence is sufficiently suspicious that the DNS signal
    #      should not be represented as zero risk."
    # =========================================================================

    CRITICAL_INDICATOR_MIN_SCORE = 70
    HIGH_INDICATOR_MIN_SCORE = 55
    MULTIPLE_HIGH_MIN_SCORE = 65
    MULTIPLE_MEDIUM_MIN_SCORE = 45

    # =========================================================================
    # 7. PROTECTIVE ADJUSTMENT
    # =========================================================================

    MAX_PROTECTIVE_REDUCTION = 8

    # =========================================================================
    # 8. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
        low_ttl_seconds: float = LOW_TTL_SECONDS,
    ) -> None:
        """
        Initialize the DNS risk scoring policy.
        """

        self._validate_thresholds(
            critical_threshold,
            high_threshold,
            medium_threshold,
        )

        if low_ttl_seconds <= 0:
            raise ValueError(
                "low_ttl_seconds must be greater than zero."
            )

        self.critical_threshold = float(
            critical_threshold
        )

        self.high_threshold = float(
            high_threshold
        )

        self.medium_threshold = float(
            medium_threshold
        )

        self.low_ttl_seconds = float(
            low_ttl_seconds
        )

        logger.info(
            "DNSRiskScorer initialized | "
            "critical=%.2f high=%.2f medium=%.2f ttl=%.1f",
            self.critical_threshold,
            self.high_threshold,
            self.medium_threshold,
            self.low_ttl_seconds,
        )

    # =========================================================================
    # 9. THRESHOLD VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_thresholds(
        critical_threshold: float,
        high_threshold: float,
        medium_threshold: float,
    ) -> None:
        """
        Validate risk threshold ordering.
        """

        thresholds = {
            "critical_threshold": critical_threshold,
            "high_threshold": high_threshold,
            "medium_threshold": medium_threshold,
        }

        for name, value in thresholds.items():

            try:
                numeric_value = float(value)
            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"{name} must be numeric."
                ) from exc

            if not math.isfinite(
                numeric_value
            ):

                raise ValueError(
                    f"{name} must be finite."
                )

            if not (
                0.0 <= numeric_value <= 1.0
            ):

                raise ValueError(
                    f"{name} must be between 0.0 and 1.0."
                )

        if not (
            medium_threshold
            < high_threshold
            < critical_threshold
        ):

            raise ValueError(
                "Risk thresholds must satisfy: "
                "medium < high < critical."
            )

    # =========================================================================
    # 10. PROBABILITY NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_probability(
        probability: Any,
    ) -> float:
        """
        Safely normalize phishing probability.

        Invalid/non-finite values become 0.0.
        """

        try:

            normalized = float(
                probability
            )

        except (
            TypeError,
            ValueError,
        ):

            logger.warning(
                "Invalid DNS probability received: %r. "
                "Using 0.0.",
                probability,
            )

            return 0.0

        if not math.isfinite(
            normalized
        ):

            logger.warning(
                "Non-finite DNS probability received: %r. "
                "Using 0.0.",
                probability,
            )

            return 0.0

        return max(
            0.0,
            min(
                1.0,
                normalized,
            ),
        )

    # =========================================================================
    # 11. CLAMP PROBABILITY
    # =========================================================================

    @staticmethod
    def _clamp_probability(
        probability: float,
    ) -> float:
        """
        Clamp probability into [0, 1].
        """

        try:

            value = float(
                probability
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

        if not math.isfinite(
            value
        ):

            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    # =========================================================================
    # 12. PROBABILITY TO SCORE
    # =========================================================================

    def probability_to_score(
        self,
        probability: float,
    ) -> int:
        """
        Convert phishing probability to a 0-100 score.
        """

        normalized_probability = (
            self._normalize_probability(
                probability
            )
        )

        score = int(
            round(
                normalized_probability
                * self.MAX_RISK_SCORE
            )
        )

        return max(
            self.MIN_RISK_SCORE,
            min(
                self.MAX_RISK_SCORE,
                score,
            ),
        )

    # =========================================================================
    # 13. RISK LEVEL
    # =========================================================================

    def _determine_risk_level(
        self,
        probability: float,
    ) -> str:
        """
        Determine risk tier from normalized probability.
        """

        normalized_probability = (
            self._normalize_probability(
                probability
            )
        )

        if (
            normalized_probability
            >= self.critical_threshold
        ):

            return "Critical"

        if (
            normalized_probability
            >= self.high_threshold
        ):

            return "High"

        if (
            normalized_probability
            >= self.medium_threshold
        ):

            return "Medium"

        return "Low"

    # =========================================================================
    # 14. SAFE FEATURE NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_features(
        features: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Convert feature telemetry into a safe dictionary.
        """

        if features is None:
            return {}

        if not isinstance(
            features,
            Mapping,
        ):

            logger.warning(
                "DNS risk scorer received invalid feature object."
            )

            return {}

        return dict(
            features
        )

    # =========================================================================
    # 15. BOOLEAN FEATURE CHECK
    # =========================================================================

    @staticmethod
    def _is_enabled(
        value: Any,
    ) -> bool:
        """
        Convert common boolean representations into bool.
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

            try:

                return float(
                    value
                ) >= 1.0

            except (
                TypeError,
                ValueError,
            ):

                return False

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value.strip()
                .lower()
            )

            return normalized in {
                "true",
                "1",
                "yes",
                "y",
                "on",
                "enabled",
            }

        return False

    # =========================================================================
    # 16. SAFE FLOAT
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Convert a value into a finite float.
        """

        try:

            numeric = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return float(
                default
            )

        if not math.isfinite(
            numeric
        ):

            return float(
                default
            )

        return numeric

    # =========================================================================
    # 17. INDICATOR CREATION
    # =========================================================================

    @staticmethod
    def _build_indicator(
        code: str,
        title: str,
        description: str,
        severity: str,
        category: str,
        evidence: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Build a standardized DNS threat indicator.
        """

        return {
            "code": str(code),
            "title": str(title),
            "description": str(description),
            "severity": str(severity),
            "category": str(category),
            "evidence": (
                dict(evidence)
                if isinstance(
                    evidence,
                    Mapping,
                )
                else {}
            ),
        }

    # =========================================================================
    # 18. INDICATOR IDENTIFICATION
    # =========================================================================

    def _identify_indicators(
        self,
        features: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Identify DNS-specific security indicators.
        """

        indicators: List[
            Dict[str, Any]
        ] = []

        # ---------------------------------------------------------------------
        # Infrastructure
        # ---------------------------------------------------------------------

        if self._is_enabled(
            features.get(
                "dns_resolution_failed",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="dns_resolution_failed",
                    title="DNS resolution failed",
                    description=(
                        "The domain could not be successfully resolved "
                        "through the DNS resolution process."
                    ),
                    severity="High",
                    category="Infrastructure",
                )
            )

        if self._is_enabled(
            features.get(
                "private_ip_detected",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="private_ip_in_public_dns",
                    title="Private IP address detected",
                    description=(
                        "A DNS response contains an IP address belonging "
                        "to a private or internal address range."
                    ),
                    severity="High",
                    category="Infrastructure",
                )
            )

        if self._is_enabled(
            features.get(
                "has_routing_anomalies",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="dns_routing_anomalies_detected",
                    title="DNS routing anomaly detected",
                    description=(
                        "DNS routing information contains characteristics "
                        "that differ from the expected domain configuration."
                    ),
                    severity="High",
                    category="Infrastructure",
                )
            )

        # ---------------------------------------------------------------------
        # Fast-Flux
        # ---------------------------------------------------------------------

        if self._is_enabled(
            features.get(
                "is_active_fast_flux",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="active_fast_flux_dns_detected",
                    title="Active fast-flux behavior detected",
                    description=(
                        "The domain exhibits characteristics associated "
                        "with rapidly changing DNS infrastructure."
                    ),
                    severity="Critical",
                    category="Fast-Flux",
                )
            )

        elif self._is_enabled(
            features.get(
                "is_fast_flux_candidate",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="fast_flux_evasion_candidate",
                    title="Possible fast-flux behavior",
                    description=(
                        "DNS characteristics suggest that the domain "
                        "may be using infrastructure rotation."
                    ),
                    severity="Medium",
                    category="Fast-Flux",
                )
            )

        ttl_value = self._safe_float(
            features.get(
                "min_ttl_value",
                features.get(
                    "dns_ttl",
                    0.0,
                ),
            )
        )

        if (
            ttl_value > 0
            and ttl_value
            < self.low_ttl_seconds
        ):

            indicators.append(
                self._build_indicator(
                    code="abnormally_low_dns_ttl",
                    title="Abnormally low DNS TTL",
                    description=(
                        "DNS records have a short time-to-live, which "
                        "can be associated with rapidly changing "
                        "infrastructure."
                    ),
                    severity="Medium",
                    category="Fast-Flux",
                    evidence={
                        "ttl_seconds": ttl_value,
                        "threshold_seconds": (
                            self.low_ttl_seconds
                        ),
                    },
                )
            )

        # ---------------------------------------------------------------------
        # Email infrastructure
        # ---------------------------------------------------------------------

        if self._is_enabled(
            features.get(
                "uses_disposable_mail_provider",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="disposable_mail_provider_used",
                    title="Disposable mail infrastructure detected",
                    description=(
                        "DNS-related mail infrastructure is associated "
                        "with a disposable or temporary email provider."
                    ),
                    severity="Medium",
                    category="Email",
                )
            )

        if self._is_enabled(
            features.get(
                "has_suspicious_mx_exchange",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="suspicious_mx_exchange_record",
                    title="Suspicious MX exchange",
                    description=(
                        "The domain's MX configuration contains "
                        "characteristics associated with suspicious "
                        "mail infrastructure."
                    ),
                    severity="High",
                    category="Email",
                )
            )

        # ---------------------------------------------------------------------
        # TXT / Payload
        # ---------------------------------------------------------------------

        if self._is_enabled(
            features.get(
                "has_base64_payload_in_txt",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="base64_payload_in_txt_record",
                    title="Encoded payload in TXT record",
                    description=(
                        "A DNS TXT record contains characteristics "
                        "consistent with encoded data."
                    ),
                    severity="High",
                    category="TXT/Payload",
                )
            )

        if self._is_enabled(
            features.get(
                "has_suspicious_long_txt",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="abnormally_long_txt_payload",
                    title="Abnormally long TXT record",
                    description=(
                        "A DNS TXT record is unusually long and may "
                        "represent an anomalous DNS payload."
                    ),
                    severity="Medium",
                    category="TXT/Payload",
                )
            )

        # ---------------------------------------------------------------------
        # Email authentication
        # ---------------------------------------------------------------------

        if self._is_enabled(
            features.get(
                "is_email_spoofable",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="domain_email_spoofable_weak_dmarc_spf",
                    title="Weak email authentication posture",
                    description=(
                        "SPF/DMARC characteristics suggest that the "
                        "domain may have weaker protection against "
                        "email impersonation."
                    ),
                    severity="Medium",
                    category="Email Authentication",
                )
            )

        if self._is_enabled(
            features.get(
                "has_multiple_spf_records",
                False,
            )
        ):

            indicators.append(
                self._build_indicator(
                    code="multiple_spf_records",
                    title="Multiple SPF records detected",
                    description=(
                        "More than one SPF record appears to be present, "
                        "which can indicate an incorrectly configured "
                        "email authentication policy."
                    ),
                    severity="Medium",
                    category="Email Authentication",
                )
            )

        return indicators

    # =========================================================================
    # 19. INDICATOR SUMMARY
    # =========================================================================

    def _summarize_indicators(
        self,
        indicators: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Summarize indicators by severity and category.
        """

        critical = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "Critical"
        )

        high = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "High"
        )

        medium = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "Medium"
        )

        low = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "Low"
        )

        categories: Dict[
            str,
            int,
        ] = {}

        for indicator in indicators:

            category = str(
                indicator.get(
                    "category",
                    "Unknown",
                )
            )

            categories[category] = (
                categories.get(
                    category,
                    0,
                )
                + 1
            )

        return {
            "total": len(
                indicators
            ),
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "categories": categories,
        }

    # =========================================================================
    # 20. INDICATOR EVIDENCE ADJUSTMENT
    # =========================================================================

    def _calculate_indicator_adjustment(
        self,
        indicators: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Calculate bounded score adjustment from DNS indicators.

        Strong indicators contribute more than contextual indicators.

        The total positive adjustment is capped to prevent double-counting.
        """

        raw_adjustment = 0

        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0

        contributions: List[
            Dict[str, Any]
        ] = []

        for indicator in indicators:

            severity = str(
                indicator.get(
                    "severity",
                    "Low",
                )
            )

            contribution = int(
                self.INDICATOR_ADJUSTMENTS.get(
                    severity,
                    1,
                )
            )

            raw_adjustment += contribution

            if severity == "Critical":
                critical_count += 1
            elif severity == "High":
                high_count += 1
            elif severity == "Medium":
                medium_count += 1
            else:
                low_count += 1

            contributions.append(
                {
                    "code": indicator.get(
                        "code"
                    ),
                    "severity": severity,
                    "adjustment": contribution,
                }
            )

        bounded_adjustment = min(
            raw_adjustment,
            self.MAX_POSITIVE_INDICATOR_ADJUSTMENT,
        )

        return {
            "raw_adjustment": raw_adjustment,
            "applied_adjustment": bounded_adjustment,
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "contributions": contributions,
        }

    # =========================================================================
    # 21. RISK FLOOR
    # =========================================================================

    def _calculate_indicator_risk_floor(
        self,
        indicators: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Calculate the minimum DNS risk score justified by strong evidence.

        This prevents strong DNS indicators from being represented as
        zero risk when the ML model probability is very low.
        """

        critical_count = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "Critical"
        )

        high_count = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "High"
        )

        medium_count = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "Medium"
        )

        risk_floor = 0
        reason = "no_strong_indicator_floor"

        if critical_count >= 1:

            risk_floor = (
                self.CRITICAL_INDICATOR_MIN_SCORE
            )

            reason = (
                "critical_dns_indicator"
            )

        elif high_count >= 2:

            risk_floor = (
                self.MULTIPLE_HIGH_MIN_SCORE
            )

            reason = (
                "multiple_high_dns_indicators"
            )

        elif high_count >= 1:

            risk_floor = (
                self.HIGH_INDICATOR_MIN_SCORE
            )

            reason = (
                "high_dns_indicator"
            )

        elif medium_count >= 2:

            risk_floor = (
                self.MULTIPLE_MEDIUM_MIN_SCORE
            )

            reason = (
                "multiple_medium_dns_indicators"
            )

        return {
            "risk_floor": risk_floor,
            "reason": reason,
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": medium_count,
        }

    # =========================================================================
    # 22. PROTECTIVE SIGNALS
    # =========================================================================

    def _calculate_protective_adjustment(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate a small bounded protective adjustment.

        Protective evidence can reduce risk slightly but can never
        erase strong DNS threat indicators.
        """

        protective_signals: List[
            str
        ] = []

        adjustment = 0

        # Strong SPF configuration.
        if (
            self._is_enabled(
                features.get(
                    "has_spf_record",
                    False,
                )
            )
            and not self._is_enabled(
                features.get(
                    "has_multiple_spf_records",
                    False,
                )
            )
            and self._safe_float(
                features.get(
                    "spf_strictness_score",
                    0.0,
                )
            ) >= 0.75
        ):

            protective_signals.append(
                "strong_spf_configuration"
            )

            adjustment -= 2

        # Strong DMARC policy.
        if (
            self._is_enabled(
                features.get(
                    "has_dmarc_record",
                    False,
                )
            )
            and self._safe_float(
                features.get(
                    "dmarc_policy_score",
                    0.0,
                )
            ) >= 0.75
        ):

            protective_signals.append(
                "strong_dmarc_policy"
            )

            adjustment -= 3

        # Domain verification TXT evidence.
        if self._is_enabled(
            features.get(
                "has_domain_verification",
                False,
            )
        ):

            protective_signals.append(
                "domain_verification_present"
            )

            adjustment -= 1

        adjustment = max(
            -self.MAX_PROTECTIVE_REDUCTION,
            adjustment,
        )

        return {
            "adjustment": adjustment,
            "protective_signals": protective_signals,
        }

    # =========================================================================
    # 23. INFRASTRUCTURE HEALTH
    # =========================================================================

    def _calculate_infrastructure_health(
        self,
        features: Mapping[str, Any],
    ) -> int:
        """
        Calculate infrastructure health from 0-100.
        """

        score = 100

        if self._is_enabled(
            features.get(
                "dns_resolution_failed",
                False,
            )
        ):

            score -= 35

        if self._is_enabled(
            features.get(
                "private_ip_detected",
                False,
            )
        ):

            score -= 30

        if self._is_enabled(
            features.get(
                "has_routing_anomalies",
                False,
            )
        ):

            score -= 25

        return self._clamp_posture_score(
            score
        )

    # =========================================================================
    # 24. FAST-FLUX HEALTH
    # =========================================================================

    def _calculate_fast_flux_health(
        self,
        features: Mapping[str, Any],
    ) -> int:
        """
        Calculate fast-flux health from 0-100.
        """

        score = 100

        if self._is_enabled(
            features.get(
                "is_active_fast_flux",
                False,
            )
        ):

            score -= 70

        elif self._is_enabled(
            features.get(
                "is_fast_flux_candidate",
                False,
            )
        ):

            score -= 35

        ttl_value = self._safe_float(
            features.get(
                "min_ttl_value",
                0.0,
            )
        )

        if (
            ttl_value > 0
            and ttl_value
            < self.low_ttl_seconds
        ):

            score -= 20

        return self._clamp_posture_score(
            score
        )

    # =========================================================================
    # 25. EMAIL AUTHENTICATION HEALTH
    # =========================================================================

    def _calculate_email_auth_health(
        self,
        features: Mapping[str, Any],
    ) -> int:
        """
        Calculate email authentication health from 0-100.
        """

        score = 100

        if self._is_enabled(
            features.get(
                "is_email_spoofable",
                False,
            )
        ):

            score -= 30

        if self._is_enabled(
            features.get(
                "has_multiple_spf_records",
                False,
            )
        ):

            score -= 20

        if not self._is_enabled(
            features.get(
                "has_dmarc_record",
                False,
            )
        ):

            score -= 15

        if self._is_enabled(
            features.get(
                "uses_disposable_mail_provider",
                False,
            )
        ):

            score -= 10

        if self._is_enabled(
            features.get(
                "has_suspicious_mx_exchange",
                False,
            )
        ):

            score -= 25

        return self._clamp_posture_score(
            score
        )

    # =========================================================================
    # 26. POSTURE DECOMPOSITION
    # =========================================================================

    def _decompose_posture(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Decompose DNS security posture into independent dimensions.
        """

        infrastructure_health = (
            self._calculate_infrastructure_health(
                features
            )
        )

        fast_flux_health = (
            self._calculate_fast_flux_health(
                features
            )
        )

        email_auth_health = (
            self._calculate_email_auth_health(
                features
            )
        )

        overall_health = int(
            round(
                (
                    infrastructure_health
                    + fast_flux_health
                    + email_auth_health
                )
                / 3.0
            )
        )

        return {
            "infrastructure_health": (
                infrastructure_health
            ),
            "fast_flux_health": (
                fast_flux_health
            ),
            "email_auth_health": (
                email_auth_health
            ),
            "overall_dns_health": (
                self._clamp_posture_score(
                    overall_health
                )
            ),
        }

    # =========================================================================
    # 27. POSTURE ADJUSTMENT
    # =========================================================================

    def _calculate_posture_adjustment(
        self,
        posture: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate a bounded posture adjustment.

        Poor posture can raise risk slightly.
        Strong posture can reduce risk slightly.

        This is intentionally small because individual posture dimensions
        may overlap with the ML features.
        """

        overall_health = self._safe_float(
            posture.get(
                "overall_dns_health",
                100,
            )
        )

        if overall_health < 30:

            adjustment = 5

        elif overall_health < 50:

            adjustment = 3

        elif overall_health < 70:

            adjustment = 1

        elif overall_health >= 90:

            adjustment = -2

        else:

            adjustment = 0

        return {
            "adjustment": adjustment,
            "overall_dns_health": int(
                round(
                    overall_health
                )
            ),
        }

    # =========================================================================
    # 28. CLAMP POSTURE
    # =========================================================================

    @classmethod
    def _clamp_posture_score(
        cls,
        score: Any,
    ) -> int:
        """
        Clamp posture score to 0-100.
        """

        numeric = cls._safe_float(
            score
        )

        return max(
            cls.MIN_POSTURE_SCORE,
            min(
                cls.MAX_POSTURE_SCORE,
                int(
                    round(
                        numeric
                    )
                ),
            ),
        )

    # =========================================================================
    # 29. HUMAN-READABLE SUMMARY
    # =========================================================================

    def _generate_summary(
        self,
        risk_score: int,
        risk_level: str,
        indicators: List[
            Dict[str, Any]
        ],
        posture: Dict[str, Any],
    ) -> str:
        """
        Generate a concise DNS risk summary.
        """

        if not indicators:

            return (
                f"DNS risk is {risk_level.lower()} "
                f"with a score of {risk_score}/100. "
                "No major DNS threat indicators were detected."
            )

        indicator_count = len(
            indicators
        )

        critical_count = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "Critical"
        )

        high_count = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "High"
        )

        medium_count = sum(
            1
            for indicator in indicators
            if indicator.get(
                "severity"
            ) == "Medium"
        )

        return (
            f"DNS risk is {risk_level.lower()} "
            f"with a score of {risk_score}/100. "
            f"{indicator_count} DNS security indicator(s) were detected, "
            f"including {critical_count} critical, "
            f"{high_count} high, and "
            f"{medium_count} medium-severity indicator(s). "
            f"Overall DNS health is "
            f"{posture.get('overall_dns_health', 0)}/100."
        )

    # =========================================================================
    # 30. PRIMARY RISK CALCULATION
    # =========================================================================

    def calculate_risk(
        self,
        probability: Any,
        features: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Calculate the complete DNS risk assessment.

        Scoring model:

            base_score
                +
            indicator_adjustment
                +
            posture_adjustment
                =
            evidence_adjusted_score

        The final score is also subjected to evidence-based risk floors.
        """

        try:

            # -----------------------------------------------------------------
            # STEP 1 — NORMALIZE PROBABILITY
            # -----------------------------------------------------------------

            normalized_probability = (
                self._normalize_probability(
                    probability
                )
            )

            # -----------------------------------------------------------------
            # STEP 2 — BASE ML SCORE
            # -----------------------------------------------------------------

            base_score = (
                self.probability_to_score(
                    normalized_probability
                )
            )

            # -----------------------------------------------------------------
            # STEP 3 — NORMALIZE FEATURES
            # -----------------------------------------------------------------

            normalized_features = (
                self._normalize_features(
                    features
                )
            )

            # -----------------------------------------------------------------
            # STEP 4 — IDENTIFY INDICATORS
            # -----------------------------------------------------------------

            triggered_indicators = (
                self._identify_indicators(
                    normalized_features
                )
            )

            # -----------------------------------------------------------------
            # STEP 5 — INDICATOR SUMMARY
            # -----------------------------------------------------------------

            indicator_summary = (
                self._summarize_indicators(
                    triggered_indicators
                )
            )

            # -----------------------------------------------------------------
            # STEP 6 — INDICATOR ADJUSTMENT
            # -----------------------------------------------------------------

            indicator_adjustment = (
                self._calculate_indicator_adjustment(
                    triggered_indicators
                )
            )

            # -----------------------------------------------------------------
            # STEP 7 — INDICATOR RISK FLOOR
            # -----------------------------------------------------------------

            indicator_floor = (
                self._calculate_indicator_risk_floor(
                    triggered_indicators
                )
            )

            # -----------------------------------------------------------------
            # STEP 8 — POSTURE
            # -----------------------------------------------------------------

            posture_breakdown = (
                self._decompose_posture(
                    normalized_features
                )
            )

            # -----------------------------------------------------------------
            # STEP 9 — POSTURE ADJUSTMENT
            # -----------------------------------------------------------------

            posture_adjustment = (
                self._calculate_posture_adjustment(
                    posture_breakdown
                )
            )

            # -----------------------------------------------------------------
            # STEP 10 — PROTECTIVE ADJUSTMENT
            # -----------------------------------------------------------------

            protective_adjustment = (
                self._calculate_protective_adjustment(
                    normalized_features
                )
            )

            # -----------------------------------------------------------------
            # STEP 11 — EVIDENCE-ADJUSTED SCORE
            # -----------------------------------------------------------------

            raw_final_score = (
                base_score
                + indicator_adjustment[
                    "applied_adjustment"
                ]
                + posture_adjustment[
                    "adjustment"
                ]
                + protective_adjustment[
                    "adjustment"
                ]
            )

            # -----------------------------------------------------------------
            # STEP 12 — APPLY EVIDENCE FLOOR
            # -----------------------------------------------------------------

            final_score = max(
                raw_final_score,
                indicator_floor[
                    "risk_floor"
                ],
            )

            # -----------------------------------------------------------------
            # STEP 13 — CLAMP FINAL SCORE
            # -----------------------------------------------------------------

            final_score = max(
                self.MIN_RISK_SCORE,
                min(
                    self.MAX_RISK_SCORE,
                    int(
                        round(
                            final_score
                        )
                    ),
                ),
            )

            # -----------------------------------------------------------------
            # STEP 14 — DETERMINE RISK LEVEL
            #
            # Risk level is based on the FINAL DNS risk score rather than
            # only the raw ML probability.
            # -----------------------------------------------------------------

            final_probability = (
                final_score / 100.0
            )

            risk_level = (
                self._determine_risk_level(
                    final_probability
                )
            )

            # -----------------------------------------------------------------
            # STEP 15 — SUMMARY
            # -----------------------------------------------------------------

            summary = (
                self._generate_summary(
                    final_score,
                    risk_level,
                    triggered_indicators,
                    posture_breakdown,
                )
            )

            # -----------------------------------------------------------------
            # STEP 16 — FINAL RESULT
            # -----------------------------------------------------------------

            return {
                "risk_score": final_score,

                "risk_level": risk_level,

                "probability": round(
                    normalized_probability,
                    6,
                ),

                "base_score": base_score,

                "final_probability": round(
                    final_probability,
                    6,
                ),

                "triggered_indicators": (
                    triggered_indicators
                ),

                "indicator_summary": (
                    indicator_summary
                ),

                "indicator_adjustment": (
                    indicator_adjustment
                ),

                "indicator_risk_floor": (
                    indicator_floor
                ),

                "posture_breakdown": (
                    posture_breakdown
                ),

                "posture_adjustment": (
                    posture_adjustment
                ),

                "protective_adjustment": (
                    protective_adjustment
                ),

                "summary": summary,

                "risk_thresholds": {
                    "medium": (
                        self.medium_threshold
                    ),
                    "high": (
                        self.high_threshold
                    ),
                    "critical": (
                        self.critical_threshold
                    ),
                },

                "scorer_version": (
                    RISK_SCORER_VERSION
                ),

                "score_source": (
                    "ml_probability_plus_dns_evidence"
                ),

                "scoring_method": (
                    "evidence_adjusted_dns_risk"
                ),

                "final_decision": (
                    "dns_signal_only"
                ),

                "signal_available": True,
            }

        except Exception as exc:

            logger.exception(
                "Error calculating DNS risk score: %s",
                exc,
            )

            return self._get_fallback_risk(
                str(exc)
            )

    # =========================================================================
    # 31. COMBINED PREDICTION + RISK
    # =========================================================================

    def assess_prediction(
        self,
        prediction_result: Mapping[str, Any],
        features: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Combine DNSPredictor output with DNS risk scoring.
        """

        if not isinstance(
            prediction_result,
            Mapping,
        ):

            raise TypeError(
                "prediction_result must be a Mapping."
            )

        probability = prediction_result.get(
            "phishing_probability",
            0.0,
        )

        if features is None:

            dataframe = prediction_result.get(
                "features_dataframe"
            )

            if dataframe is not None:

                try:

                    if (
                        hasattr(
                            dataframe,
                            "iloc",
                        )
                        and len(
                            dataframe
                        ) == 1
                    ):

                        features = (
                            dataframe.iloc[
                                0
                            ].to_dict()
                        )

                except Exception:

                    logger.debug(
                        "Unable to extract features from "
                        "predictor DataFrame.",
                        exc_info=True,
                    )

        risk_assessment = (
            self.calculate_risk(
                probability,
                features or {},
            )
        )

        return {
            "prediction": dict(
                prediction_result
            ),

            "risk_assessment": (
                risk_assessment
            ),
        }

    # =========================================================================
    # 32. FALLBACK RISK
    # =========================================================================

    def _get_fallback_risk(
        self,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return a safe unavailable risk structure.

        IMPORTANT:

            calculation failure != legitimate

        Therefore risk_score is None rather than zero.
        """

        return {
            "risk_score": None,

            "risk_level": "Unavailable",

            "probability": None,

            "base_score": None,

            "final_probability": None,

            "triggered_indicators": [],

            "indicator_summary": {
                "total": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "categories": {},
            },

            "indicator_adjustment": {
                "raw_adjustment": 0,
                "applied_adjustment": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
                "contributions": [],
            },

            "indicator_risk_floor": {
                "risk_floor": 0,
                "reason": "calculation_error",
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
            },

            "posture_breakdown": {
                "infrastructure_health": None,
                "fast_flux_health": None,
                "email_auth_health": None,
                "overall_dns_health": None,
            },

            "posture_adjustment": {
                "adjustment": 0,
                "overall_dns_health": None,
            },

            "protective_adjustment": {
                "adjustment": 0,
                "protective_signals": [],
            },

            "summary": (
                "DNS risk scoring was unavailable because "
                "the risk calculation failed."
            ),

            "risk_thresholds": {
                "medium": self.medium_threshold,
                "high": self.high_threshold,
                "critical": self.critical_threshold,
            },

            "scorer_version": (
                RISK_SCORER_VERSION
            ),

            "score_source": "unavailable",

            "scoring_method": "unavailable",

            "final_decision": "dns_signal_unavailable",

            "signal_available": False,

            "error": (
                str(error_message)
                if error_message
                else "DNS risk calculation failed."
            ),
        }

    # =========================================================================
    # 33. POLICY
    # =========================================================================

    def get_policy(
        self,
    ) -> Dict[str, Any]:
        """
        Return the active DNS risk-scoring policy.
        """

        return {
            "scorer_version": (
                RISK_SCORER_VERSION
            ),

            "critical_threshold": (
                self.critical_threshold
            ),

            "high_threshold": (
                self.high_threshold
            ),

            "medium_threshold": (
                self.medium_threshold
            ),

            "low_ttl_seconds": (
                self.low_ttl_seconds
            ),

            "indicator_adjustments": dict(
                self.INDICATOR_ADJUSTMENTS
            ),

            "max_positive_indicator_adjustment": (
                self.MAX_POSITIVE_INDICATOR_ADJUSTMENT
            ),

            "risk_floors": {
                "critical_indicator": (
                    self.CRITICAL_INDICATOR_MIN_SCORE
                ),
                "high_indicator": (
                    self.HIGH_INDICATOR_MIN_SCORE
                ),
                "multiple_high": (
                    self.MULTIPLE_HIGH_MIN_SCORE
                ),
                "multiple_medium": (
                    self.MULTIPLE_MEDIUM_MIN_SCORE
                ),
            },

            "max_protective_reduction": (
                self.MAX_PROTECTIVE_REDUCTION
            ),

            "score_source": (
                "ml_probability_plus_dns_evidence"
            ),
        }

    # =========================================================================
    # 34. THRESHOLD UPDATE
    # =========================================================================

    def update_thresholds(
        self,
        critical_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        medium_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Update DNS risk thresholds safely.
        """

        new_critical = (
            self.critical_threshold
            if critical_threshold is None
            else float(
                critical_threshold
            )
        )

        new_high = (
            self.high_threshold
            if high_threshold is None
            else float(
                high_threshold
            )
        )

        new_medium = (
            self.medium_threshold
            if medium_threshold is None
            else float(
                medium_threshold
            )
        )

        self._validate_thresholds(
            new_critical,
            new_high,
            new_medium,
        )

        self.critical_threshold = (
            new_critical
        )

        self.high_threshold = (
            new_high
        )

        self.medium_threshold = (
            new_medium
        )

        return {
            "critical": (
                self.critical_threshold
            ),
            "high": (
                self.high_threshold
            ),
            "medium": (
                self.medium_threshold
            ),
        }

    # =========================================================================
    # 35. RISK LEVEL DESCRIPTION
    # =========================================================================

    @staticmethod
    def get_risk_level_description(
        risk_level: str,
    ) -> str:
        """
        Return a human-readable description of a DNS risk level.
        """

        descriptions = {
            "Low": (
                "Low DNS risk. No strong DNS threat signal "
                "was identified."
            ),

            "Medium": (
                "Moderate DNS risk. Some suspicious DNS "
                "characteristics were identified."
            ),

            "High": (
                "High DNS risk. Significant DNS security "
                "indicators were identified."
            ),

            "Critical": (
                "Critical DNS risk. Strong DNS threat evidence "
                "was identified."
            ),

            "Unavailable": (
                "DNS risk could not be calculated reliably."
            ),

            "Unknown": (
                "DNS risk level is unknown."
            ),
        }

        return descriptions.get(
            str(risk_level),
            "DNS risk level is unknown.",
        )

    # =========================================================================
    # 36. SERIALIZATION
    # =========================================================================

    @classmethod
    def make_serializable(
        cls,
        value: Any,
    ) -> Any:
        """
        Convert common non-JSON-native values into serializable values.
        """

        if value is None:
            return None

        if isinstance(
            value,
            bool,
        ):

            return value

        if isinstance(
            value,
            (str, int),
        ):

            return value

        if isinstance(
            value,
            float,
        ):

            if math.isfinite(
                value
            ):

                return value

            return None

        if isinstance(
            value,
            Mapping,
        ):

            return {
                str(key): cls.make_serializable(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):

            return [
                cls.make_serializable(
                    item
                )
                for item in value
            ]

        # NumPy scalar compatibility.
        try:

            if hasattr(
                value,
                "item",
            ):

                return cls.make_serializable(
                    value.item()
                )

        except Exception:

            pass

        return str(
            value
        )
"""
Threat Intelligence AI Agent - Risk Scoring Engine
===================================================

Deterministic risk scoring layer for the Threat Intelligence Agent.

Architecture
------------

    ThreatFeatureSchema
            |
            v
    ThreatPreprocessor
            |
            v
    ThreatPredictor
            |
            v
    malicious probability
            |
            v
    ThreatRiskScorer
            |
            +--------------------+
            |                    |
            v                    v
       Risk Score           Indicators
        0 - 100             Evidence
            |                    |
            +---------+----------+
                      |
                      v
              Threat Risk Result


IMPORTANT FEATURE CONTRACT
--------------------------

ThreatFeatureSchema is the SINGLE SOURCE OF TRUTH.

The canonical Threat ML schema contains exactly 20 features:

    1.  is_blacklisted
    2.  blacklist_vendors_count
    3.  high_authority_vendor_flagged
    4.  malicious_engines_count
    5.  suspicious_engines_count
    6.  harmless_engines_count
    7.  undetected_engines_count
    8.  malicious_ratio
    9.  suspicion_ratio
    10. has_high_threat_consensus
    11. is_malware_associated
    12. is_c2_node
    13. is_exploit_distributor
    14. malware_threat_score
    15. reputation_score
    16. weighted_threat_score
    17. overall_threat_score
    18. is_zero_day_candidate
    19. days_since_first_submission
    20. days_since_last_analysis

This file does NOT create a second schema list.

It imports the canonical feature names directly from:

    ThreatFeatureSchema

The scorer does not perform ML inference.

The scorer converts the Predictor's malicious probability into:

    risk_score
    risk_level

and extracts explainable threat-intelligence evidence.

No standalone execution hook is included.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .feature_schema import ThreatFeatureSchema


logger = logging.getLogger(__name__)


class ThreatRiskScorer:
    """
    Deterministic policy-based risk scoring engine.

    Primary score
    -------------

        Threat Predictor malicious probability
            ->
        0-100 risk score

    Evidence
    --------

    Threat Intelligence features are analyzed independently to produce
    transparent indicators and an evidence breakdown.

    Risk tiers
    ----------

        0 - 29   -> Low
        30 - 59  -> Medium
        60 - 79  -> High
        80 - 100  -> Critical

    Important
    ---------

    This class does NOT train or execute XGBoost.

    It does NOT replace the ML probability.

    It converts the Predictor's probability into an operational risk
    assessment and explains the supporting threat evidence.
    """

    # =========================================================================
    # RISK THRESHOLDS
    # =========================================================================

    DEFAULT_MEDIUM_THRESHOLD = 0.30
    DEFAULT_HIGH_THRESHOLD = 0.60
    DEFAULT_CRITICAL_THRESHOLD = 0.80

    # =========================================================================
    # SCORE BOUNDARIES
    # =========================================================================

    MIN_RISK_SCORE = 0
    MAX_RISK_SCORE = 100

    # =========================================================================
    # FEATURE SAFETY LIMITS
    # =========================================================================

    MAX_BLACKLIST_VENDOR_COUNT = 50.0
    MAX_ENGINE_COUNT = 500.0

    # =========================================================================
    # SEVERITIES
    # =========================================================================

    SEVERITY_CRITICAL = "critical"
    SEVERITY_HIGH = "high"
    SEVERITY_MEDIUM = "medium"
    SEVERITY_LOW = "low"
    SEVERITY_INFO = "info"

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================

    def __init__(
        self,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
    ) -> None:
        """
        Initialize the Threat Risk Scorer.
        """

        self.critical_threshold = (
            self._validate_threshold(
                critical_threshold,
                "critical_threshold",
            )
        )

        self.high_threshold = (
            self._validate_threshold(
                high_threshold,
                "high_threshold",
            )
        )

        self.medium_threshold = (
            self._validate_threshold(
                medium_threshold,
                "medium_threshold",
            )
        )

        self._validate_threshold_order()

        # ---------------------------------------------------------------------
        # Canonical schema metadata.
        # ---------------------------------------------------------------------

        self.feature_schema: List[str] = (
            ThreatFeatureSchema.get_schema_columns()
        )

        self.feature_count: int = (
            ThreatFeatureSchema.get_feature_count()
        )

        self._validate_feature_schema()

        logger.info(
            "ThreatRiskScorer initialized | "
            "features=%d | medium=%.2f | high=%.2f | critical=%.2f",
            self.feature_count,
            self.medium_threshold,
            self.high_threshold,
            self.critical_threshold,
        )

    # =========================================================================
    # SCHEMA VALIDATION
    # =========================================================================

    def _validate_feature_schema(self) -> None:
        """
        Validate the canonical Threat schema at initialization time.

        The current Threat Agent contract requires exactly 20 features.
        """

        current_schema = (
            ThreatFeatureSchema.get_schema_columns()
        )

        current_count = (
            ThreatFeatureSchema.get_feature_count()
        )

        if len(current_schema) != current_count:

            raise RuntimeError(
                "ThreatFeatureSchema is internally inconsistent: "
                f"list_length={len(current_schema)}, "
                f"feature_count={current_count}."
            )

        if current_count != 20:

            raise RuntimeError(
                "ThreatRiskScorer expects the canonical 20-feature "
                f"Threat schema, but received {current_count} features."
            )

        if list(current_schema) != (
            self.feature_schema
        ):

            raise RuntimeError(
                "ThreatRiskScorer schema snapshot does not match "
                "ThreatFeatureSchema."
            )

    # =========================================================================
    # PUBLIC SCHEMA API
    # =========================================================================

    def get_feature_schema(self) -> List[str]:
        """
        Return the current canonical Threat feature order.
        """

        return list(
            ThreatFeatureSchema.get_schema_columns()
        )

    def get_feature_count(self) -> int:
        """
        Return the canonical Threat feature count.
        """

        return (
            ThreatFeatureSchema.get_feature_count()
        )

    def validate_feature_schema(self) -> Dict[str, Any]:
        """
        Validate this scorer against the canonical Threat schema.
        """

        current_schema = (
            ThreatFeatureSchema.get_schema_columns()
        )

        current_count = (
            ThreatFeatureSchema.get_feature_count()
        )

        return {
            "valid": (
                current_count == 20
                and len(current_schema) == 20
                and current_schema == self.feature_schema
                and current_count == self.feature_count
            ),
            "feature_count": current_count,
            "expected_feature_count": 20,
            "schema_match": (
                current_schema == self.feature_schema
            ),
            "feature_order": list(
                current_schema
            ),
        }

    # =========================================================================
    # THRESHOLD VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_threshold(
        value: Any,
        name: str,
    ) -> float:
        """
        Validate a probability threshold.
        """

        try:
            threshold = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                f"{name} must be numeric."
            ) from exc

        if not math.isfinite(
            threshold
        ):

            raise ValueError(
                f"{name} must be finite."
            )

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                f"{name} must be between 0.0 and 1.0."
            )

        return threshold

    def _validate_threshold_order(self) -> None:
        """
        Ensure:

            medium < high < critical
        """

        if not (
            self.medium_threshold
            < self.high_threshold
            < self.critical_threshold
        ):

            raise ValueError(
                "Risk thresholds must satisfy: "
                "medium < high < critical."
            )

    # =========================================================================
    # SAFE NUMERIC CONVERSION
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Convert telemetry into a finite float.
        """

        if isinstance(
            value,
            bool,
        ):

            return (
                1.0
                if value
                else 0.0
            )

        try:

            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return default

        if not math.isfinite(
            result
        ):

            return default

        return result

    # =========================================================================
    # SAFE FLAG
    # =========================================================================

    @classmethod
    def _is_enabled(
        cls,
        value: Any,
    ) -> bool:
        """
        Interpret a canonical binary feature.
        """

        return (
            cls._safe_float(
                value,
                default=0.0,
            )
            >= 0.5
        )

    # =========================================================================
    # SAFE RATIO
    # =========================================================================

    @classmethod
    def _safe_ratio(
        cls,
        value: Any,
    ) -> float:
        """
        Normalize ratio into [0, 1].
        """

        numeric = cls._safe_float(
            value,
            default=0.0,
        )

        return max(
            0.0,
            min(
                1.0,
                numeric,
            ),
        )

    # =========================================================================
    # SAFE SCORE
    # =========================================================================

    @classmethod
    def _safe_score(
        cls,
        value: Any,
    ) -> float:
        """
        Normalize a 0-100 score.
        """

        numeric = cls._safe_float(
            value,
            default=0.0,
        )

        return max(
            0.0,
            min(
                100.0,
                numeric,
            ),
        )

    # =========================================================================
    # FEATURE ACCESS
    # =========================================================================

    @staticmethod
    def _get_feature(
        features: Mapping[str, Any],
        name: str,
        default: Any = 0.0,
    ) -> Any:
        """
        Safely retrieve a feature.
        """

        if not isinstance(
            features,
            Mapping,
        ):

            return default

        return features.get(
            name,
            default,
        )

    # =========================================================================
    # PROBABILITY NORMALIZATION
    # =========================================================================

    @classmethod
    def _normalize_probability(
        cls,
        probability: Any,
    ) -> float:
        """
        Normalize malicious probability into [0, 1].
        """

        value = cls._safe_float(
            probability,
            default=0.0,
        )

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    # =========================================================================
    # PROBABILITY -> SCORE
    # =========================================================================

    @classmethod
    def _probability_to_score(
        cls,
        probability: Any,
    ) -> int:
        """
        Convert malicious probability to 0-100 risk score.
        """

        normalized = (
            cls._normalize_probability(
                probability
            )
        )

        score = int(
            round(
                normalized * 100.0
            )
        )

        return max(
            cls.MIN_RISK_SCORE,
            min(
                cls.MAX_RISK_SCORE,
                score,
            ),
        )

    # =========================================================================
    # RISK LEVEL
    # =========================================================================

    def _determine_risk_level(
        self,
        probability: Any,
    ) -> str:
        """
        Determine Low / Medium / High / Critical.
        """

        normalized = (
            self._normalize_probability(
                probability
            )
        )

        if normalized >= (
            self.critical_threshold
        ):

            return "Critical"

        if normalized >= (
            self.high_threshold
        ):

            return "High"

        if normalized >= (
            self.medium_threshold
        ):

            return "Medium"

        return "Low"

    # =========================================================================
    # INDICATOR APPEND
    # =========================================================================

    @staticmethod
    def _append_indicator(
        indicators: List[str],
        indicator: str,
    ) -> None:
        """
        Add an indicator once.
        """

        if indicator not in indicators:

            indicators.append(
                indicator
            )

    # =========================================================================
    # STRUCTURED INDICATOR
    # =========================================================================

    @staticmethod
    def _append_detail(
        details: List[Dict[str, Any]],
        indicator: str,
        severity: str,
        description: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add structured evidence information.
        """

        details.append(
            {
                "indicator": indicator,
                "severity": severity,
                "description": description,
                "evidence": (
                    evidence
                    if evidence is not None
                    else {}
                ),
            }
        )

    # =========================================================================
    # BLACKLIST EVIDENCE
    # =========================================================================

    def _evaluate_blacklist(
        self,
        features: Mapping[str, Any],
        indicators: List[str],
        details: List[Dict[str, Any]],
    ) -> None:
        """
        Evaluate blacklist and authoritative-vendor evidence.
        """

        blacklisted = self._is_enabled(
            self._get_feature(
                features,
                "is_blacklisted",
            )
        )

        vendor_count = self._safe_float(
            self._get_feature(
                features,
                "blacklist_vendors_count",
            )
        )

        vendor_count = max(
            0.0,
            min(
                self.MAX_BLACKLIST_VENDOR_COUNT,
                vendor_count,
            ),
        )

        authority_flagged = self._is_enabled(
            self._get_feature(
                features,
                "high_authority_vendor_flagged",
            )
        )

        if blacklisted:

            self._append_indicator(
                indicators,
                "domain_blacklisted_globally",
            )

            self._append_detail(
                details,
                "domain_blacklisted_globally",
                self.SEVERITY_CRITICAL,
                "Threat intelligence reports the entity as blacklisted.",
                {
                    "is_blacklisted": True,
                },
            )

        if authority_flagged:

            self._append_indicator(
                indicators,
                "high_authority_vendor_flagged_domain",
            )

            self._append_detail(
                details,
                "high_authority_vendor_flagged_domain",
                self.SEVERITY_CRITICAL,
                "A high-authority threat intelligence source has flagged the entity.",
                {
                    "high_authority_vendor_flagged": True,
                },
            )

        if vendor_count >= 3:

            self._append_indicator(
                indicators,
                "multiple_blacklist_vendors_reporting",
            )

            self._append_detail(
                details,
                "multiple_blacklist_vendors_reporting",
                self.SEVERITY_HIGH,
                "Multiple blacklist vendors report the entity.",
                {
                    "blacklist_vendors_count": round(
                        vendor_count,
                        2,
                    ),
                },
            )

        elif vendor_count > 0:

            self._append_indicator(
                indicators,
                "blacklist_vendor_reporting",
            )

            self._append_detail(
                details,
                "blacklist_vendor_reporting",
                self.SEVERITY_MEDIUM,
                "At least one blacklist vendor reports the entity.",
                {
                    "blacklist_vendors_count": round(
                        vendor_count,
                        2,
                    ),
                },
            )

    # =========================================================================
    # MULTI-ENGINE EVIDENCE
    # =========================================================================

    def _evaluate_engine_consensus(
        self,
        features: Mapping[str, Any],
        indicators: List[str],
        details: List[Dict[str, Any]],
    ) -> None:
        """
        Evaluate malicious/suspicious engine consensus.
        """

        malicious_count = max(
            0.0,
            min(
                self.MAX_ENGINE_COUNT,
                self._safe_float(
                    self._get_feature(
                        features,
                        "malicious_engines_count",
                    )
                ),
            ),
        )

        suspicious_count = max(
            0.0,
            min(
                self.MAX_ENGINE_COUNT,
                self._safe_float(
                    self._get_feature(
                        features,
                        "suspicious_engines_count",
                    )
                ),
            ),
        )

        harmless_count = max(
            0.0,
            min(
                self.MAX_ENGINE_COUNT,
                self._safe_float(
                    self._get_feature(
                        features,
                        "harmless_engines_count",
                    )
                ),
            ),
        )

        undetected_count = max(
            0.0,
            min(
                self.MAX_ENGINE_COUNT,
                self._safe_float(
                    self._get_feature(
                        features,
                        "undetected_engines_count",
                    )
                ),
            ),
        )

        malicious_ratio = (
            self._safe_ratio(
                self._get_feature(
                    features,
                    "malicious_ratio",
                )
            )
        )

        suspicion_ratio = (
            self._safe_ratio(
                self._get_feature(
                    features,
                    "suspicion_ratio",
                )
            )
        )

        high_consensus = self._is_enabled(
            self._get_feature(
                features,
                "has_high_threat_consensus",
            )
        )

        if high_consensus:

            self._append_indicator(
                indicators,
                "high_threat_vendor_consensus",
            )

            self._append_detail(
                details,
                "high_threat_vendor_consensus",
                self.SEVERITY_CRITICAL,
                "Threat intelligence sources show strong malicious consensus.",
                {
                    "has_high_threat_consensus": True,
                    "malicious_engines_count": malicious_count,
                    "malicious_ratio": round(
                        malicious_ratio,
                        4,
                    ),
                },
            )

        if malicious_ratio >= 0.20:

            self._append_indicator(
                indicators,
                "very_high_malicious_engine_ratio",
            )

            self._append_detail(
                details,
                "very_high_malicious_engine_ratio",
                self.SEVERITY_CRITICAL,
                "A substantial proportion of engines classify the entity as malicious.",
                {
                    "malicious_ratio": round(
                        malicious_ratio,
                        4,
                    ),
                },
            )

        elif malicious_ratio > 0.05:

            self._append_indicator(
                indicators,
                "elevated_malicious_engine_ratio",
            )

            self._append_detail(
                details,
                "elevated_malicious_engine_ratio",
                self.SEVERITY_HIGH,
                "The malicious engine ratio is elevated.",
                {
                    "malicious_ratio": round(
                        malicious_ratio,
                        4,
                    ),
                },
            )

        if malicious_count >= 10:

            self._append_indicator(
                indicators,
                "widespread_malicious_detections",
            )

            self._append_detail(
                details,
                "widespread_malicious_detections",
                self.SEVERITY_CRITICAL,
                "Many independent engines report malicious behavior.",
                {
                    "malicious_engines_count": malicious_count,
                },
            )

        elif malicious_count >= 5:

            self._append_indicator(
                indicators,
                "multiple_malicious_engine_detections",
            )

            self._append_detail(
                details,
                "multiple_malicious_engine_detections",
                self.SEVERITY_HIGH,
                "Multiple independent engines report malicious behavior.",
                {
                    "malicious_engines_count": malicious_count,
                },
            )

        elif malicious_count > 0:

            self._append_indicator(
                indicators,
                "malicious_engine_detection_present",
            )

            self._append_detail(
                details,
                "malicious_engine_detection_present",
                self.SEVERITY_MEDIUM,
                "At least one threat engine reports malicious behavior.",
                {
                    "malicious_engines_count": malicious_count,
                },
            )

        if suspicion_ratio >= 0.10:

            self._append_indicator(
                indicators,
                "elevated_suspicious_engine_ratio",
            )

            self._append_detail(
                details,
                "elevated_suspicious_engine_ratio",
                self.SEVERITY_MEDIUM,
                "A meaningful proportion of engines classify the entity as suspicious.",
                {
                    "suspicion_ratio": round(
                        suspicion_ratio,
                        4,
                    ),
                },
            )

        if suspicious_count >= 5:

            self._append_indicator(
                indicators,
                "multiple_suspicious_engine_detections",
            )

            self._append_detail(
                details,
                "multiple_suspicious_engine_detections",
                self.SEVERITY_MEDIUM,
                "Multiple engines report suspicious characteristics.",
                {
                    "suspicious_engines_count": suspicious_count,
                },
            )

        if (
            harmless_count > 0
            and malicious_count > 0
        ):

            self._append_indicator(
                indicators,
                "mixed_threat_intelligence_consensus",
            )

            self._append_detail(
                details,
                "mixed_threat_intelligence_consensus",
                self.SEVERITY_INFO,
                "Threat intelligence sources disagree about the entity.",
                {
                    "malicious_engines_count": malicious_count,
                    "harmless_engines_count": harmless_count,
                    "undetected_engines_count": undetected_count,
                },
            )

    # =========================================================================
    # MALWARE / C2 / EXPLOIT EVIDENCE
    # =========================================================================

    def _evaluate_malware(
        self,
        features: Mapping[str, Any],
        indicators: List[str],
        details: List[Dict[str, Any]],
    ) -> None:
        """
        Evaluate malware, C2 and exploit infrastructure features.
        """

        malware_associated = self._is_enabled(
            self._get_feature(
                features,
                "is_malware_associated",
            )
        )

        c2_node = self._is_enabled(
            self._get_feature(
                features,
                "is_c2_node",
            )
        )

        exploit_distributor = self._is_enabled(
            self._get_feature(
                features,
                "is_exploit_distributor",
            )
        )

        malware_score = self._safe_score(
            self._get_feature(
                features,
                "malware_threat_score",
            )
        )

        if malware_associated:

            self._append_indicator(
                indicators,
                "active_malware_association",
            )

            self._append_detail(
                details,
                "active_malware_association",
                self.SEVERITY_CRITICAL,
                "The entity is associated with malware activity.",
                {
                    "is_malware_associated": True,
                },
            )

        if c2_node:

            self._append_indicator(
                indicators,
                "active_command_and_control_c2_node",
            )

            self._append_detail(
                details,
                "active_command_and_control_c2_node",
                self.SEVERITY_CRITICAL,
                "The entity is associated with command-and-control infrastructure.",
                {
                    "is_c2_node": True,
                },
            )

        if exploit_distributor:

            self._append_indicator(
                indicators,
                "exploit_kit_distribution_detected",
            )

            self._append_detail(
                details,
                "exploit_kit_distribution_detected",
                self.SEVERITY_CRITICAL,
                "The entity is associated with exploit distribution.",
                {
                    "is_exploit_distributor": True,
                },
            )

        if malware_score >= 75.0:

            self._append_indicator(
                indicators,
                "critical_malware_threat_score",
            )

            self._append_detail(
                details,
                "critical_malware_threat_score",
                self.SEVERITY_CRITICAL,
                "The malware threat score is critically elevated.",
                {
                    "malware_threat_score": round(
                        malware_score,
                        2,
                    ),
                },
            )

        elif malware_score >= 50.0:

            self._append_indicator(
                indicators,
                "high_malware_threat_severity",
            )

            self._append_detail(
                details,
                "high_malware_threat_severity",
                self.SEVERITY_HIGH,
                "The malware threat score is substantially elevated.",
                {
                    "malware_threat_score": round(
                        malware_score,
                        2,
                    ),
                },
            )

    # =========================================================================
    # REPUTATION / COMPOSITE SCORE EVIDENCE
    # =========================================================================

    def _evaluate_scores(
        self,
        features: Mapping[str, Any],
        indicators: List[str],
        details: List[Dict[str, Any]],
    ) -> None:
        """
        Evaluate reputation and composite Threat scores.
        """

        reputation_score = self._safe_score(
            self._get_feature(
                features,
                "reputation_score",
            )
        )

        weighted_score = self._safe_score(
            self._get_feature(
                features,
                "weighted_threat_score",
            )
        )

        overall_score = self._safe_score(
            self._get_feature(
                features,
                "overall_threat_score",
            )
        )

        if overall_score >= 80.0:

            self._append_indicator(
                indicators,
                "critical_overall_threat_score",
            )

            self._append_detail(
                details,
                "critical_overall_threat_score",
                self.SEVERITY_CRITICAL,
                "The composite threat score is critically elevated.",
                {
                    "overall_threat_score": round(
                        overall_score,
                        2,
                    ),
                },
            )

        elif overall_score >= 60.0:

            self._append_indicator(
                indicators,
                "high_overall_threat_score",
            )

            self._append_detail(
                details,
                "high_overall_threat_score",
                self.SEVERITY_HIGH,
                "The composite threat score is highly elevated.",
                {
                    "overall_threat_score": round(
                        overall_score,
                        2,
                    ),
                },
            )

        elif overall_score >= 40.0:

            self._append_indicator(
                indicators,
                "elevated_overall_threat_score",
            )

            self._append_detail(
                details,
                "elevated_overall_threat_score",
                self.SEVERITY_MEDIUM,
                "The composite threat score is elevated.",
                {
                    "overall_threat_score": round(
                        overall_score,
                        2,
                    ),
                },
            )

        if weighted_score >= 75.0:

            self._append_indicator(
                indicators,
                "high_weighted_threat_score",
            )

            self._append_detail(
                details,
                "high_weighted_threat_score",
                self.SEVERITY_HIGH,
                "The weighted threat score is strongly elevated.",
                {
                    "weighted_threat_score": round(
                        weighted_score,
                        2,
                    ),
                },
            )

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # reputation_score polarity is not assumed here.
        #
        # A low reputation score is therefore reported as supporting evidence
        # only, not automatically classified as malicious.
        # ---------------------------------------------------------------------

        if reputation_score <= 20.0:

            self._append_indicator(
                indicators,
                "very_low_reputation_score",
            )

            self._append_detail(
                details,
                "very_low_reputation_score",
                self.SEVERITY_LOW,
                "The entity has a very low normalized reputation score.",
                {
                    "reputation_score": round(
                        reputation_score,
                        2,
                    ),
                },
            )

    # =========================================================================
    # NOVELTY EVIDENCE
    # =========================================================================

    def _evaluate_novelty(
        self,
        features: Mapping[str, Any],
        indicators: List[str],
        details: List[Dict[str, Any]],
    ) -> None:
        """
        Evaluate zero-day and recent-observation characteristics.
        """

        zero_day_candidate = self._is_enabled(
            self._get_feature(
                features,
                "is_zero_day_candidate",
            )
        )

        days_since_first = self._safe_float(
            self._get_feature(
                features,
                "days_since_first_submission",
                -1.0,
            ),
            default=-1.0,
        )

        if zero_day_candidate:

            self._append_indicator(
                indicators,
                "potential_zero_day_threat_candidate",
            )

            self._append_detail(
                details,
                "potential_zero_day_threat_candidate",
                self.SEVERITY_HIGH,
                "The entity has characteristics associated with a newly observed or insufficiently tested threat.",
                {
                    "is_zero_day_candidate": True,
                },
            )

        # -1 is the unknown sentinel.
        if (
            days_since_first >= 0.0
            and days_since_first <= 7.0
        ):

            self._append_indicator(
                indicators,
                "very_recent_threat_intelligence_submission",
            )

            self._append_detail(
                details,
                "very_recent_threat_intelligence_submission",
                self.SEVERITY_LOW,
                "The entity was first observed very recently.",
                {
                    "days_since_first_submission": round(
                        days_since_first,
                        2,
                    ),
                },
            )

    # =========================================================================
    # FRESHNESS
    # =========================================================================

    def _evaluate_freshness(
        self,
        features: Mapping[str, Any],
        indicators: List[str],
        details: List[Dict[str, Any]],
    ) -> None:
        """
        Evaluate threat-intelligence freshness.

        Stale intelligence is not interpreted as safe.
        """

        days_since_analysis = self._safe_float(
            self._get_feature(
                features,
                "days_since_last_analysis",
                -1.0,
            ),
            default=-1.0,
        )

        if days_since_analysis >= 30.0:

            self._append_indicator(
                indicators,
                "stale_threat_intelligence",
            )

            self._append_detail(
                details,
                "stale_threat_intelligence",
                self.SEVERITY_LOW,
                "The latest threat-intelligence analysis is relatively old.",
                {
                    "days_since_last_analysis": round(
                        days_since_analysis,
                        2,
                    ),
                },
            )

    # =========================================================================
    # EVIDENCE EXTRACTION
    # =========================================================================

    def _identify_indicators(
        self,
        features: Mapping[str, Any],
    ) -> Tuple[
        List[str],
        List[Dict[str, Any]],
    ]:
        """
        Extract explainable indicators from canonical Threat features.
        """

        indicators: List[str] = []

        details: List[Dict[str, Any]] = []

        if not isinstance(
            features,
            Mapping,
        ):

            return indicators, details

        self._evaluate_blacklist(
            features,
            indicators,
            details,
        )

        self._evaluate_engine_consensus(
            features,
            indicators,
            details,
        )

        self._evaluate_malware(
            features,
            indicators,
            details,
        )

        self._evaluate_scores(
            features,
            indicators,
            details,
        )

        self._evaluate_novelty(
            features,
            indicators,
            details,
        )

        self._evaluate_freshness(
            features,
            indicators,
            details,
        )

        return indicators, details

    # =========================================================================
    # EVIDENCE STRENGTH
    # =========================================================================

    def _calculate_evidence_strength(
        self,
        details: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate bounded diagnostic evidence strength.

        This does NOT replace model probability.

        Range:

            0.0 - 1.0
        """

        if not details:
            return 0.0

        severity_weights = {
            self.SEVERITY_CRITICAL: 1.00,
            self.SEVERITY_HIGH: 0.75,
            self.SEVERITY_MEDIUM: 0.50,
            self.SEVERITY_LOW: 0.20,
            self.SEVERITY_INFO: 0.00,
        }

        weighted_sum = 0.0

        for detail in details:

            severity = str(
                detail.get(
                    "severity",
                    self.SEVERITY_INFO,
                )
            ).lower()

            weighted_sum += (
                severity_weights.get(
                    severity,
                    0.0,
                )
            )

        return round(
            min(
                1.0,
                weighted_sum / 3.0,
            ),
            4,
        )

    # =========================================================================
    # EVIDENCE SUMMARY
    # =========================================================================

    def _build_evidence_summary(
        self,
        details: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Count evidence by severity.
        """

        summary = {
            self.SEVERITY_CRITICAL: 0,
            self.SEVERITY_HIGH: 0,
            self.SEVERITY_MEDIUM: 0,
            self.SEVERITY_LOW: 0,
            self.SEVERITY_INFO: 0,
        }

        for detail in details:

            severity = str(
                detail.get(
                    "severity",
                    self.SEVERITY_INFO,
                )
            ).lower()

            if severity in summary:

                summary[
                    severity
                ] += 1

        return summary

    # =========================================================================
    # MAIN RISK CALCULATION
    # =========================================================================

    def calculate_risk(
        self,
        probability: Any,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate the complete Threat risk assessment.

        Parameters
        ----------
        probability:
            Malicious probability produced by ThreatPredictor.

        features:
            Canonical Threat feature dictionary.

        Returns
        -------
        dict

        Compatibility keys:

            risk_score
            risk_level
            triggered_indicators

        Additional diagnostic keys:

            model_probability
            indicator_details
            indicator_count
            evidence_strength
            evidence_summary
            scoring_status
        """

        try:

            normalized_probability = (
                self._normalize_probability(
                    probability
                )
            )

            risk_score = (
                self._probability_to_score(
                    normalized_probability
                )
            )

            risk_level = (
                self._determine_risk_level(
                    normalized_probability
                )
            )

            indicators, details = (
                self._identify_indicators(
                    features
                )
            )

            evidence_strength = (
                self._calculate_evidence_strength(
                    details
                )
            )

            evidence_summary = (
                self._build_evidence_summary(
                    details
                )
            )

            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "model_probability": round(
                    normalized_probability,
                    6,
                ),
                "triggered_indicators": indicators,
                "indicator_details": details,
                "indicator_count": len(
                    indicators
                ),
                "evidence_strength": evidence_strength,
                "evidence_summary": evidence_summary,
                "scoring_status": "success",
            }

        except Exception as exc:

            logger.error(
                "Threat risk scoring failed: %s",
                exc,
                exc_info=True,
            )

            return self._fallback_result(
                str(exc)
            )

    # =========================================================================
    # FALLBACK
    # =========================================================================

    @staticmethod
    def _fallback_result(
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return a safe operational fallback.

        IMPORTANT:

        This fallback does not claim the target is malicious or safe.

        The caller receives an explicit scoring failure.
        """

        result: Dict[str, Any] = {
            "risk_score": 0,
            "risk_level": "Low",
            "model_probability": 0.0,
            "triggered_indicators": [],
            "indicator_details": [],
            "indicator_count": 0,
            "evidence_strength": 0.0,
            "evidence_summary": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0,
            },
            "scoring_status": "error_fallback",
        }

        if error_message:

            result["error"] = (
                "Threat risk scoring failed: "
                + error_message
            )

        return result

    # =========================================================================
    # THRESHOLDS
    # =========================================================================

    def get_thresholds(
        self,
    ) -> Dict[str, float]:
        """
        Return active risk thresholds.
        """

        return {
            "medium_threshold": (
                self.medium_threshold
            ),
            "high_threshold": (
                self.high_threshold
            ),
            "critical_threshold": (
                self.critical_threshold
            ),
        }

    # =========================================================================
    # SCORE -> RISK LEVEL
    # =========================================================================

    def get_risk_level_for_score(
        self,
        risk_score: Any,
    ) -> str:
        """
        Convert a 0-100 risk score to a risk tier.
        """

        score = self._safe_float(
            risk_score,
            default=0.0,
        )

        score = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

        return self._determine_risk_level(
            score / 100.0
        )

    # =========================================================================
    # FEATURE SUMMARY
    # =========================================================================

    def summarize_features(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a diagnostic summary using only canonical Threat features.
        """

        if not isinstance(
            features,
            Mapping,
        ):

            features = {}

        return {
            "is_blacklisted": self._is_enabled(
                self._get_feature(
                    features,
                    "is_blacklisted",
                )
            ),

            "blacklist_vendors_count": round(
                max(
                    0.0,
                    self._safe_float(
                        self._get_feature(
                            features,
                            "blacklist_vendors_count",
                        )
                    ),
                ),
                2,
            ),

            "high_authority_vendor_flagged": self._is_enabled(
                self._get_feature(
                    features,
                    "high_authority_vendor_flagged",
                )
            ),

            "malicious_engines_count": round(
                max(
                    0.0,
                    self._safe_float(
                        self._get_feature(
                            features,
                            "malicious_engines_count",
                        )
                    ),
                ),
                2,
            ),

            "suspicious_engines_count": round(
                max(
                    0.0,
                    self._safe_float(
                        self._get_feature(
                            features,
                            "suspicious_engines_count",
                        )
                    ),
                ),
                2,
            ),

            "harmless_engines_count": round(
                max(
                    0.0,
                    self._safe_float(
                        self._get_feature(
                            features,
                            "harmless_engines_count",
                        )
                    ),
                ),
                2,
            ),

            "undetected_engines_count": round(
                max(
                    0.0,
                    self._safe_float(
                        self._get_feature(
                            features,
                            "undetected_engines_count",
                        )
                    ),
                ),
                2,
            ),

            "malicious_ratio": round(
                self._safe_ratio(
                    self._get_feature(
                        features,
                        "malicious_ratio",
                    )
                ),
                4,
            ),

            "suspicion_ratio": round(
                self._safe_ratio(
                    self._get_feature(
                        features,
                        "suspicion_ratio",
                    )
                ),
                4,
            ),

            "has_high_threat_consensus": self._is_enabled(
                self._get_feature(
                    features,
                    "has_high_threat_consensus",
                )
            ),

            "is_malware_associated": self._is_enabled(
                self._get_feature(
                    features,
                    "is_malware_associated",
                )
            ),

            "is_c2_node": self._is_enabled(
                self._get_feature(
                    features,
                    "is_c2_node",
                )
            ),

            "is_exploit_distributor": self._is_enabled(
                self._get_feature(
                    features,
                    "is_exploit_distributor",
                )
            ),

            "malware_threat_score": round(
                self._safe_score(
                    self._get_feature(
                        features,
                        "malware_threat_score",
                    )
                ),
                2,
            ),

            "reputation_score": round(
                self._safe_score(
                    self._get_feature(
                        features,
                        "reputation_score",
                    )
                ),
                2,
            ),

            "weighted_threat_score": round(
                self._safe_score(
                    self._get_feature(
                        features,
                        "weighted_threat_score",
                    )
                ),
                2,
            ),

            "overall_threat_score": round(
                self._safe_score(
                    self._get_feature(
                        features,
                        "overall_threat_score",
                    )
                ),
                2,
            ),

            "is_zero_day_candidate": self._is_enabled(
                self._get_feature(
                    features,
                    "is_zero_day_candidate",
                )
            ),

            "days_since_first_submission": self._safe_float(
                self._get_feature(
                    features,
                    "days_since_first_submission",
                    -1.0,
                ),
                default=-1.0,
            ),

            "days_since_last_analysis": self._safe_float(
                self._get_feature(
                    features,
                    "days_since_last_analysis",
                    -1.0,
                ),
                default=-1.0,
            ),
        }

    # =========================================================================
    # RESULT VALIDATION
    # =========================================================================

    def validate_risk_result(
        self,
        result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate the public risk result contract.
        """

        errors: List[str] = []

        if not isinstance(
            result,
            Mapping,
        ):

            return {
                "valid": False,
                "errors": [
                    "Risk result must be a mapping."
                ],
            }

        required_fields = {
            "risk_score",
            "risk_level",
            "model_probability",
            "triggered_indicators",
            "indicator_details",
            "scoring_status",
        }

        for field in required_fields:

            if field not in result:

                errors.append(
                    f"Missing required field: {field}"
                )

        if "risk_score" in result:

            score = self._safe_float(
                result["risk_score"],
                default=-1.0,
            )

            if not 0.0 <= score <= 100.0:

                errors.append(
                    "risk_score must be between 0 and 100."
                )

        if "model_probability" in result:

            probability = self._safe_float(
                result["model_probability"],
                default=-1.0,
            )

            if not 0.0 <= probability <= 1.0:

                errors.append(
                    "model_probability must be between 0 and 1."
                )

        if "risk_level" in result:

            if result["risk_level"] not in {
                "Low",
                "Medium",
                "High",
                "Critical",
            }:

                errors.append(
                    "Invalid risk_level."
                )

        if "triggered_indicators" in result:

            if not isinstance(
                result["triggered_indicators"],
                list,
            ):

                errors.append(
                    "triggered_indicators must be a list."
                )

        if "indicator_details" in result:

            if not isinstance(
                result["indicator_details"],
                list,
            ):

                errors.append(
                    "indicator_details must be a list."
                )

        return {
            "valid": len(
                errors
            ) == 0,
            "errors": errors,
        }
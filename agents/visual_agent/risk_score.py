"""
agents/visual_agent/risk_score.py

Production-grade risk scoring engine for the Visual AI Agent.

Responsibilities
----------------

    Visual ML probability
            +
    Visual structural features
            +
    Security policy rules
            |
            v
    Local Visual Risk Assessment
            |
            +----------------------+
            |                      |
            v                      v
       0-100 Score             Risk Level
            |
            v
       Triggered Indicators
            |
            v
       Visual Explanation
            |
            v
       Central Fusion Engine


Classification / Risk Contract
------------------------------

Risk score:
    0   - 100

Risk levels:
    Low
    Medium
    High
    Critical


Important Design Principle
--------------------------

The XGBoost phishing probability remains the PRIMARY numerical signal.

Visual indicators are primarily used for:

    - explainability
    - evidence collection
    - policy validation
    - downstream fusion
    - human-readable reporting

The scorer must avoid blindly adding every indicator to the ML probability,
because that would double-count evidence already learned by the XGBoost model.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple


logger = logging.getLogger(__name__)


class VisualRiskScorer:
    """
    Policy-based local risk scoring engine for the Visual AI Agent.

    The scorer converts the Visual Predictor's phishing probability into a
    0-100 risk score and identifies visual security indicators that explain
    why the page may be suspicious.

    Example
    -------

        probability = 0.87

        result = scorer.calculate_risk(
            probability,
            {
                "login_form_detected": 1.0,
                "logo_detected": 1.0,
                "form_area_ratio": 0.45,
                "image_density": 0.30,
                "blank_area_ratio": 0.12,
                "layout_complexity": 72.0,
                "button_count": 8.0,
            },
        )

    Result:

        {
            "risk_score": 87,
            "risk_level": "Critical",
            "phishing_probability": 0.87,
            "triggered_indicators": [...],
            "indicator_count": 3,
            ...
        }
    """

    # =========================================================================
    # 1. RISK LEVELS
    # =========================================================================

    RISK_LOW: str = "Low"

    RISK_MEDIUM: str = "Medium"

    RISK_HIGH: str = "High"

    RISK_CRITICAL: str = "Critical"

    # =========================================================================
    # 2. SCORE BOUNDARIES
    # =========================================================================

    MIN_RISK_SCORE: int = 0

    MAX_RISK_SCORE: int = 100

    # =========================================================================
    # 3. PROBABILITY THRESHOLDS
    # =========================================================================
    #
    # These thresholds are project policy values.
    #
    # They can later be tuned using validation datasets, ROC analysis,
    # precision/recall analysis, or project requirements.
    #
    # =========================================================================

    CRITICAL_THRESHOLD: float = 0.80

    HIGH_THRESHOLD: float = 0.60

    MEDIUM_THRESHOLD: float = 0.30

    # =========================================================================
    # 4. VISUAL INDICATOR THRESHOLDS
    # =========================================================================

    FORM_DENSITY_THRESHOLD: float = 0.30

    IMAGE_DENSITY_THRESHOLD: float = 0.80

    BLANK_AREA_THRESHOLD: float = 0.60

    LAYOUT_COMPLEXITY_THRESHOLD: float = 80.0

    BUTTON_COUNT_THRESHOLD: float = 5.0

    HIGH_ENTROPY_THRESHOLD: float = 7.5

    # =========================================================================
    # 5. INDICATOR SEVERITY WEIGHTS
    # =========================================================================
    #
    # These weights DO NOT directly modify the ML score.
    #
    # They represent the importance of an indicator for explanation and
    # evidence aggregation.
    #
    # =========================================================================

    INDICATOR_WEIGHTS: Dict[
        str,
        float,
    ] = {
        "credential_harvesting_form_visually_detected": 1.00,
        "login_form_with_brand_logo_combination": 1.00,
        "brand_logo_detected_in_layout": 0.45,
        "high_form_density_occupying_screen": 0.70,
        "excessive_image_density_possible_text_obfuscation": 0.65,
        "high_whitespace_minimalist_phishing_template": 0.45,
        "highly_complex_visual_layout": 0.35,
        "multiple_calls_to_action_detected": 0.25,
        "high_visual_entropy": 0.20,
    }

    # =========================================================================
    # 6. INDICATOR HUMAN-READABLE EXPLANATIONS
    # =========================================================================

    INDICATOR_DESCRIPTIONS: Dict[
        str,
        str,
    ] = {
        "credential_harvesting_form_visually_detected": (
            "A visually recognizable login or credential-entry form "
            "was detected."
        ),
        "login_form_with_brand_logo_combination": (
            "A login form appears together with a brand-like logo, "
            "which can be relevant to credential-harvesting pages."
        ),
        "brand_logo_detected_in_layout": (
            "A logo-like element was detected in the page layout."
        ),
        "high_form_density_occupying_screen": (
            "Form elements occupy a relatively large portion of "
            "the visible page."
        ),
        "excessive_image_density_possible_text_obfuscation": (
            "A large proportion of the visible page consists of images."
        ),
        "high_whitespace_minimalist_phishing_template": (
            "The page contains a large amount of blank or whitespace area."
        ),
        "highly_complex_visual_layout": (
            "The rendered page has relatively high visual/layout complexity."
        ),
        "multiple_calls_to_action_detected": (
            "Multiple interactive buttons or calls to action were detected."
        ),
        "high_visual_entropy": (
            "The screenshot contains relatively high visual entropy."
        ),
    }

    # =========================================================================
    # 7. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        *,
        critical_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        medium_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize the Visual Risk Scorer.

        Optional threshold overrides are useful for experiments and
        validation without modifying the class-level policy.
        """

        self.critical_threshold = (
            float(
                critical_threshold
            )
            if critical_threshold is not None
            else self.CRITICAL_THRESHOLD
        )

        self.high_threshold = (
            float(
                high_threshold
            )
            if high_threshold is not None
            else self.HIGH_THRESHOLD
        )

        self.medium_threshold = (
            float(
                medium_threshold
            )
            if medium_threshold is not None
            else self.MEDIUM_THRESHOLD
        )

        self._validate_thresholds()

        logger.info(
            "VisualRiskScorer initialized. "
            "thresholds=medium:%.2f high:%.2f critical:%.2f",
            self.medium_threshold,
            self.high_threshold,
            self.critical_threshold,
        )

    # =========================================================================
    # 8. THRESHOLD VALIDATION
    # =========================================================================

    def _validate_thresholds(
        self,
    ) -> None:
        """
        Validate the risk-level probability thresholds.
        """

        thresholds = {
            "medium_threshold": self.medium_threshold,
            "high_threshold": self.high_threshold,
            "critical_threshold": self.critical_threshold,
        }

        for name, value in thresholds.items():

            if not math.isfinite(
                value
            ):

                raise ValueError(
                    f"{name} must be a finite number."
                )

            if not 0.0 <= value <= 1.0:

                raise ValueError(
                    f"{name} must be between 0.0 and 1.0."
                )

        if not (
            self.medium_threshold
            <= self.high_threshold
            <= self.critical_threshold
        ):

            raise ValueError(
                "Risk thresholds must satisfy: "
                "medium <= high <= critical."
            )

    # =========================================================================
    # 9. PROBABILITY NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_probability(
        probability: Any,
    ) -> float:
        """
        Safely normalize phishing probability to [0.0, 1.0].

        Raises
        ------
        ValueError
            If the supplied value cannot be interpreted as a finite number.
        """

        try:

            value = float(
                probability
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "Visual phishing probability must be numeric."
            ) from exc

        if not math.isfinite(
            value
        ):

            raise ValueError(
                "Visual phishing probability must be finite."
            )

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    # =========================================================================
    # 10. FEATURE NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_features(
        features: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Convert the feature payload into a regular dictionary.

        Missing or invalid feature dictionaries become an empty dictionary.
        """

        if features is None:

            return {}

        if not isinstance(
            features,
            Mapping,
        ):

            logger.warning(
                "Visual risk scorer received invalid feature payload type: %s",
                type(features).__name__,
            )

            return {}

        return dict(
            features
        )

    # =========================================================================
    # 11. SAFE NUMERIC FEATURE
    # =========================================================================

    @staticmethod
    def _numeric_feature(
        features: Mapping[str, Any],
        name: str,
        default: float = 0.0,
    ) -> float:
        """
        Safely retrieve a numerical feature.
        """

        value = features.get(
            name,
            default,
        )

        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            logger.debug(
                "Visual feature '%s' could not be converted to float.",
                name,
            )

            return float(
                default
            )

        if not math.isfinite(
            numeric_value
        ):

            return float(
                default
            )

        return numeric_value

    # =========================================================================
    # 12. BOOLEAN FEATURE
    # =========================================================================

    @staticmethod
    def _boolean_feature(
        features: Mapping[str, Any],
        name: str,
    ) -> bool:
        """
        Safely interpret a Visual boolean feature.

        Supports:

            True / False
            1 / 0
            "true" / "false"
            "yes" / "no"
        """

        value = features.get(
            name,
            False,
        )

        if isinstance(
            value,
            bool,
        ):

            return value

        if isinstance(
            value,
            (int, float),
        ):

            return float(
                value
            ) >= 0.5

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value.strip()
                .lower()
            )

            if normalized in {
                "true",
                "1",
                "yes",
                "y",
                "detected",
            }:

                return True

            if normalized in {
                "false",
                "0",
                "no",
                "n",
                "not_detected",
            }:

                return False

        return False

    # =========================================================================
    # 13. RISK LEVEL
    # =========================================================================

    def _determine_risk_level(
        self,
        probability: float,
    ) -> str:
        """
        Convert phishing probability into a policy risk level.

        Boundaries:

            >= critical → Critical
            >= high     → High
            >= medium   → Medium
            otherwise   → Low
        """

        probability = (
            self._normalize_probability(
                probability
            )
        )

        if (
            probability
            >= self.critical_threshold
        ):

            return self.RISK_CRITICAL

        if (
            probability
            >= self.high_threshold
        ):

            return self.RISK_HIGH

        if (
            probability
            >= self.medium_threshold
        ):

            return self.RISK_MEDIUM

        return self.RISK_LOW

    # =========================================================================
    # 14. INDICATOR DETECTION
    # =========================================================================

    def _identify_indicators(
        self,
        features: Optional[
            Mapping[str, Any]
        ],
    ) -> List[str]:
        """
        Identify visual security indicators.

        These indicators are evidence for explanation and downstream fusion.
        They do not automatically increase the ML probability.
        """

        normalized_features = (
            self._normalize_features(
                features
            )
        )

        if not normalized_features:

            return []

        indicators: List[str] = []

        login_form = (
            self._boolean_feature(
                normalized_features,
                "login_form_detected",
            )
        )

        logo_detected = (
            self._boolean_feature(
                normalized_features,
                "logo_detected",
            )
        )

        form_area_ratio = (
            self._numeric_feature(
                normalized_features,
                "form_area_ratio",
            )
        )

        image_density = (
            self._numeric_feature(
                normalized_features,
                "image_density",
            )
        )

        blank_area_ratio = (
            self._numeric_feature(
                normalized_features,
                "blank_area_ratio",
            )
        )

        layout_complexity = (
            self._numeric_feature(
                normalized_features,
                "layout_complexity",
            )
        )

        button_count = (
            self._numeric_feature(
                normalized_features,
                "button_count",
            )
        )

        entropy = (
            self._numeric_feature(
                normalized_features,
                "image_entropy",
            )
        )

        # ---------------------------------------------------------------------
        # Credential harvesting form.
        # ---------------------------------------------------------------------

        if login_form:

            indicators.append(
                "credential_harvesting_form_visually_detected"
            )

        # ---------------------------------------------------------------------
        # Login form + brand logo.
        #
        # This combination is more informative than either feature alone.
        # ---------------------------------------------------------------------

        if (
            login_form
            and logo_detected
        ):

            indicators.append(
                "login_form_with_brand_logo_combination"
            )

        # ---------------------------------------------------------------------
        # Brand logo.
        # ---------------------------------------------------------------------

        if logo_detected:

            indicators.append(
                "brand_logo_detected_in_layout"
            )

        # ---------------------------------------------------------------------
        # Large form area.
        # ---------------------------------------------------------------------

        if (
            form_area_ratio
            > self.FORM_DENSITY_THRESHOLD
        ):

            indicators.append(
                "high_form_density_occupying_screen"
            )

        # ---------------------------------------------------------------------
        # Image-heavy page.
        # ---------------------------------------------------------------------

        if (
            image_density
            > self.IMAGE_DENSITY_THRESHOLD
        ):

            indicators.append(
                "excessive_image_density_possible_text_obfuscation"
            )

        # ---------------------------------------------------------------------
        # Large whitespace area.
        # ---------------------------------------------------------------------

        if (
            blank_area_ratio
            > self.BLANK_AREA_THRESHOLD
        ):

            indicators.append(
                "high_whitespace_minimalist_phishing_template"
            )

        # ---------------------------------------------------------------------
        # Complex layout.
        # ---------------------------------------------------------------------

        if (
            layout_complexity
            > self.LAYOUT_COMPLEXITY_THRESHOLD
        ):

            indicators.append(
                "highly_complex_visual_layout"
            )

        # ---------------------------------------------------------------------
        # Multiple buttons.
        # ---------------------------------------------------------------------

        if (
            button_count
            > self.BUTTON_COUNT_THRESHOLD
        ):

            indicators.append(
                "multiple_calls_to_action_detected"
            )

        # ---------------------------------------------------------------------
        # High image entropy.
        # ---------------------------------------------------------------------

        if (
            entropy
            > self.HIGH_ENTROPY_THRESHOLD
        ):

            indicators.append(
                "high_visual_entropy"
            )

        return indicators

    # =========================================================================
    # 15. INDICATOR DETAILS
    # =========================================================================

    def _build_indicator_details(
        self,
        indicators: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Convert indicator names into structured explanation objects.
        """

        details: List[
            Dict[str, Any]
        ] = []

        for indicator in indicators:

            details.append(
                {
                    "indicator": indicator,
                    "description": (
                        self.INDICATOR_DESCRIPTIONS.get(
                            indicator,
                            "Visual security indicator detected.",
                        )
                    ),
                    "weight": round(
                        float(
                            self.INDICATOR_WEIGHTS.get(
                                indicator,
                                0.0,
                            )
                        ),
                        4,
                    ),
                }
            )

        return details

    # =========================================================================
    # 16. INDICATOR EVIDENCE SCORE
    # =========================================================================

    def _calculate_indicator_evidence(
        self,
        indicators: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate a bounded visual evidence score.

        Important:
            This is NOT added directly to the ML phishing probability.

        It is a separate evidence signal that can be consumed by the central
        Decision Fusion Engine.
        """

        if not indicators:

            return {
                "indicator_evidence_score": 0,
                "indicator_evidence_ratio": 0.0,
                "indicator_weight_sum": 0.0,
            }

        weights = [
            float(
                self.INDICATOR_WEIGHTS.get(
                    indicator,
                    0.0,
                )
            )
            for indicator in indicators
        ]

        # ---------------------------------------------------------------------
        # We intentionally cap the raw sum.
        #
        # Otherwise several correlated visual indicators could create an
        # artificial score above the intended range.
        # ---------------------------------------------------------------------

        weight_sum = sum(
            weights
        )

        max_possible_weight = sum(
            self.INDICATOR_WEIGHTS.values()
        )

        if max_possible_weight <= 0.0:

            ratio = 0.0

        else:

            ratio = min(
                1.0,
                weight_sum
                / max_possible_weight,
            )

        evidence_score = int(
            round(
                ratio * 100
            )
        )

        return {
            "indicator_evidence_score": evidence_score,
            "indicator_evidence_ratio": round(
                ratio,
                4,
            ),
            "indicator_weight_sum": round(
                weight_sum,
                4,
            ),
        }

    # =========================================================================
    # 17. PRIMARY RISK SCORE
    # =========================================================================

    @staticmethod
    def _probability_to_score(
        probability: float,
    ) -> int:
        """
        Convert normalized probability to an integer 0-100 score.
        """

        probability = max(
            0.0,
            min(
                1.0,
                float(
                    probability
                ),
            ),
        )

        return int(
            round(
                probability * 100
            )
        )

    # =========================================================================
    # 18. INDICATOR-BASED POLICY CHECK
    # =========================================================================

    def _apply_policy_guard(
        self,
        probability: float,
        indicators: List[str],
    ) -> Tuple[
        float,
        Optional[str],
    ]:
        """
        Apply a conservative policy guard.

        The policy guard does NOT manufacture a high probability from visual
        indicators alone.

        It only prevents a clearly concerning combination from being treated
        as completely low-risk when the ML model is uncertain.

        Current policy:

            login form + logo
                AND
            ML probability >= 0.25

        can raise the effective probability to 0.30.

        This is deliberately conservative.

        The purpose is policy safety, not replacing the trained model.
        """

        effective_probability = (
            float(
                probability
            )
        )

        policy_reason: Optional[
            str
        ] = None

        high_value_combination = (
            "login_form_with_brand_logo_combination"
            in indicators
        )

        if (
            high_value_combination
            and effective_probability
            >= self.medium_threshold
            * 0.83
        ):

            minimum_policy_probability = (
                self.medium_threshold
            )

            if (
                effective_probability
                < minimum_policy_probability
            ):

                effective_probability = (
                    minimum_policy_probability
                )

                policy_reason = (
                    "login_form_and_brand_logo_policy_guard"
                )

        return (
            min(
                1.0,
                effective_probability,
            ),
            policy_reason,
        )

    # =========================================================================
    # 19. MAIN RISK CALCULATION
    # =========================================================================

    def calculate_risk(
        self,
        probability: float,
        features: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Calculate the localized Visual Agent risk assessment.

        Parameters
        ----------
        probability:
            XGBoost phishing probability in [0, 1].

        features:
            Normalized visual feature dictionary.

        Returns
        -------
        Dict[str, Any]
            Structured risk assessment.

        Important
        ---------
        The returned `risk_score` is based primarily on phishing probability.
        Visual indicators are returned separately as supporting evidence.
        """

        try:

            # -----------------------------------------------------------------
            # STEP 1 — Normalize probability.
            # -----------------------------------------------------------------

            normalized_probability = (
                self._normalize_probability(
                    probability
                )
            )

            # -----------------------------------------------------------------
            # STEP 2 — Normalize features.
            # -----------------------------------------------------------------

            normalized_features = (
                self._normalize_features(
                    features
                )
            )

            # -----------------------------------------------------------------
            # STEP 3 — Identify visual indicators.
            # -----------------------------------------------------------------

            triggered_indicators = (
                self._identify_indicators(
                    normalized_features
                )
            )

            # -----------------------------------------------------------------
            # STEP 4 — Apply conservative policy guard.
            # -----------------------------------------------------------------

            (
                effective_probability,
                policy_reason,
            ) = self._apply_policy_guard(
                normalized_probability,
                triggered_indicators,
            )

            # -----------------------------------------------------------------
            # STEP 5 — Convert probability to 0-100 score.
            # -----------------------------------------------------------------

            risk_score = (
                self._probability_to_score(
                    effective_probability
                )
            )

            # -----------------------------------------------------------------
            # STEP 6 — Determine risk level.
            # -----------------------------------------------------------------

            risk_level = (
                self._determine_risk_level(
                    effective_probability
                )
            )

            # -----------------------------------------------------------------
            # STEP 7 — Build detailed indicator information.
            # -----------------------------------------------------------------

            indicator_details = (
                self._build_indicator_details(
                    triggered_indicators
                )
            )

            # -----------------------------------------------------------------
            # STEP 8 — Calculate independent visual evidence score.
            # -----------------------------------------------------------------

            indicator_evidence = (
                self._calculate_indicator_evidence(
                    triggered_indicators
                )
            )

            # -----------------------------------------------------------------
            # STEP 9 — Return structured assessment.
            # -----------------------------------------------------------------

            result: Dict[str, Any] = {
                "risk_score": risk_score,
                "risk_level": risk_level,

                # Original ML probability.
                "phishing_probability": round(
                    normalized_probability,
                    6,
                ),

                # Probability after conservative policy guard.
                "effective_probability": round(
                    effective_probability,
                    6,
                ),

                "triggered_indicators": (
                    triggered_indicators
                ),

                "indicator_details": (
                    indicator_details
                ),

                "indicator_count": len(
                    triggered_indicators
                ),

                "policy_adjustment": (
                    policy_reason
                ),

                "score_source": (
                    "visual_ml_probability"
                    if policy_reason is None
                    else "visual_ml_probability_with_policy_guard"
                ),

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

                **indicator_evidence,
            }

            logger.info(
                "Visual risk assessment completed. "
                "probability=%.4f "
                "score=%d "
                "level=%s "
                "indicators=%d",
                normalized_probability,
                risk_score,
                risk_level,
                len(
                    triggered_indicators
                ),
            )

            return result

        except Exception as exc:

            logger.error(
                "Error calculating Visual risk score: %s",
                exc,
                exc_info=True,
            )

            return self._get_fallback_risk(
                error=str(
                    exc
                )
            )

    # =========================================================================
    # 20. COMPLETE PREDICTION + RISK METHOD
    # =========================================================================

    def score_prediction(
        self,
        prediction_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Convenience method that consumes the output of VisualPredictor.

        Expected predictor fields:

            phishing_probability
            features

        This makes the connection between predictor.py and risk_score.py
        explicit.
        """

        if not isinstance(
            prediction_result,
            Mapping,
        ):

            raise TypeError(
                "prediction_result must be a dictionary-like mapping."
            )

        probability = prediction_result.get(
            "phishing_probability",
            0.0,
        )

        features = prediction_result.get(
            "features"
        )

        # ---------------------------------------------------------------------
        # Backward compatibility:
        #
        # Older predictor implementations may expose only
        # `features_dataframe`.
        # ---------------------------------------------------------------------

        if (
            not features
            and isinstance(
                prediction_result.get(
                    "features_dataframe"
                ),
                object,
            )
        ):

            dataframe = prediction_result.get(
                "features_dataframe"
            )

            if isinstance(
                dataframe,
                # Avoid importing another dataframe type here.
                # This is simply a safe structural check.
                type(None),
            ):

                dataframe = None

            if (
                dataframe is not None
                and hasattr(
                    dataframe,
                    "iloc",
                )
                and len(
                    dataframe
                ) > 0
            ):

                features = (
                    dataframe.iloc[
                        0
                    ].to_dict()
                )

        return self.calculate_risk(
            probability=probability,
            features=features,
        )

    # =========================================================================
    # 21. INDICATOR QUERY
    # =========================================================================

    def has_indicator(
        self,
        risk_result: Mapping[str, Any],
        indicator: str,
    ) -> bool:
        """
        Check whether a particular indicator was triggered.
        """

        if not isinstance(
            risk_result,
            Mapping,
        ):

            return False

        indicators = (
            risk_result.get(
                "triggered_indicators",
                [],
            )
        )

        if not isinstance(
            indicators,
            list,
        ):

            return False

        return indicator in indicators

    # =========================================================================
    # 22. HIGH-RISK CHECK
    # =========================================================================

    def is_high_risk(
        self,
        risk_result: Mapping[str, Any],
    ) -> bool:
        """
        Return True when the Visual risk level is High or Critical.
        """

        if not isinstance(
            risk_result,
            Mapping,
        ):

            return False

        level = str(
            risk_result.get(
                "risk_level",
                self.RISK_LOW,
            )
        )

        return level in {
            self.RISK_HIGH,
            self.RISK_CRITICAL,
        }

    # =========================================================================
    # 23. CRITICAL-RISK CHECK
    # =========================================================================

    def is_critical(
        self,
        risk_result: Mapping[str, Any],
    ) -> bool:
        """
        Return True only for Critical Visual risk.
        """

        if not isinstance(
            risk_result,
            Mapping,
        ):

            return False

        return (
            risk_result.get(
                "risk_level"
            )
            == self.RISK_CRITICAL
        )

    # =========================================================================
    # 24. RISK LEVEL ORDER
    # =========================================================================

    @staticmethod
    def risk_level_rank(
        risk_level: str,
    ) -> int:
        """
        Convert risk level into an ordinal rank.

            Low      = 0
            Medium   = 1
            High     = 2
            Critical = 3
        """

        ranks = {
            "Low": 0,
            "Medium": 1,
            "High": 2,
            "Critical": 3,
        }

        return ranks.get(
            str(
                risk_level
            ),
            0,
        )

    # =========================================================================
    # 25. HUMAN-READABLE SUMMARY
    # =========================================================================

    def build_summary(
        self,
        risk_result: Mapping[str, Any],
    ) -> str:
        """
        Build a concise human-readable Visual risk summary.
        """

        if not isinstance(
            risk_result,
            Mapping,
        ):

            return (
                "Visual risk assessment unavailable."
            )

        risk_score = risk_result.get(
            "risk_score",
            0,
        )

        risk_level = risk_result.get(
            "risk_level",
            self.RISK_LOW,
        )

        phishing_probability = (
            risk_result.get(
                "phishing_probability",
                0.0,
            )
        )

        indicators = (
            risk_result.get(
                "triggered_indicators",
                [],
            )
        )

        if indicators:

            indicator_text = (
                f"{len(indicators)} visual security indicator(s) detected."
            )

        else:

            indicator_text = (
                "No configured visual security indicators were detected."
            )

        return (
            "Visual assessment: "
            f"{risk_level} risk "
            f"(score {risk_score}/100), "
            f"ML phishing probability "
            f"{float(phishing_probability) * 100:.1f}%. "
            f"{indicator_text}"
        )

    # =========================================================================
    # 26. CONFIGURATION
    # =========================================================================

    def get_configuration(
        self,
    ) -> Dict[str, Any]:
        """
        Return the complete active risk-scoring configuration.
        """

        return {
            "thresholds": {
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
            "indicator_thresholds": {
                "form_area_ratio": (
                    self.FORM_DENSITY_THRESHOLD
                ),
                "image_density": (
                    self.IMAGE_DENSITY_THRESHOLD
                ),
                "blank_area_ratio": (
                    self.BLANK_AREA_THRESHOLD
                ),
                "layout_complexity": (
                    self.LAYOUT_COMPLEXITY_THRESHOLD
                ),
                "button_count": (
                    self.BUTTON_COUNT_THRESHOLD
                ),
                "image_entropy": (
                    self.HIGH_ENTROPY_THRESHOLD
                ),
            },
            "indicator_weights": dict(
                self.INDICATOR_WEIGHTS
            ),
            "indicator_descriptions": dict(
                self.INDICATOR_DESCRIPTIONS
            ),
        }

    # =========================================================================
    # 27. HEALTH CHECK
    # =========================================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:
        """
        Verify that the Visual risk scorer is correctly configured.
        """

        try:

            self._validate_thresholds()

            return {
                "status": "healthy",
                "component": "VisualRiskScorer",
                "thresholds": {
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
                "indicator_rules": len(
                    self.INDICATOR_WEIGHTS
                ),
            }

        except Exception as exc:

            logger.error(
                "VisualRiskScorer health check failed: %s",
                exc,
                exc_info=True,
            )

            return {
                "status": "error",
                "component": "VisualRiskScorer",
                "error": str(
                    exc
                ),
            }

    # =========================================================================
    # 28. SAFE FALLBACK
    # =========================================================================

    def _get_fallback_risk(
        self,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return a conservative, stable fallback structure.

        The fallback deliberately does not claim that the page is legitimate.
        An inference failure is represented as an unavailable/low-information
        state so that the central fusion engine can make the final decision.
        """

        result: Dict[str, Any] = {
            "risk_score": 0,
            "risk_level": self.RISK_LOW,
            "phishing_probability": 0.0,
            "effective_probability": 0.0,
            "triggered_indicators": [],
            "indicator_details": [],
            "indicator_count": 0,
            "indicator_evidence_score": 0,
            "indicator_evidence_ratio": 0.0,
            "indicator_weight_sum": 0.0,
            "policy_adjustment": None,
            "score_source": "fallback",
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
        }

        if error:

            result["error"] = error

        return result
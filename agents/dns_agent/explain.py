"""
DNS Security AI Agent - Explainability Engine
==============================================

This module provides Explainable AI (XAI) capabilities for the DNS
Security AI Agent.

Primary explainability technology
----------------------------------
SHAP (SHapley Additive exPlanations)

SHAP is used to explain how individual DNS features contributed to the
machine-learning model's prediction.

The module also provides a deterministic heuristic explanation fallback
when SHAP or the trained model is unavailable.

Architecture
------------

                    DNS Features
                         |
                         v
                  DNS Predictor
                         |
                         v
                  XGBoost Model
                         |
                         v
                  Phishing Score
                         |
                         v
                  DNS Explainer
                    /        \
                   /          \
                  v            v
              SHAP XAI     Heuristic XAI
                  |            |
                  └──────┬─────┘
                         v
                 Explanation Result
                         |
             ┌───────────┼───────────┐
             v           v           v
         Summary     Top Factors   Evidence
                         |
                         v
                  AI Reasoning Layer
                         |
                         v
                  Final DNS Report


Important Security Principle
----------------------------

The explanation engine explains the DNS model's decision.

It does NOT independently determine whether a website is phishing.

For example:

    "Active fast-flux behavior increased phishing probability"

means:

    The DNS model found this feature influential.

It does NOT mean:

    The domain is definitely malicious.

The final phishing decision will later be produced by the
Multi-Agent Decision Fusion / Consensus Engine.


Supported Explanation Sources
-----------------------------

1. SHAP
   Provides model-specific feature attribution.

2. Heuristic fallback
   Provides rule-based explanations when SHAP cannot be used.

Every explanation identifies its source so downstream systems know
whether the explanation is ML-derived or heuristic.


Output Design
-------------

The engine returns structured information suitable for:

    - DNS Agent
    - Decision Fusion Engine
    - AI Reasoning Engine
    - Dashboard
    - SOC analyst interface
    - PDF report
    - JSON API
    - Explainable AI section


Author:
    Multi-Agent AI Cybersecurity Analyst

Agent:
    DNS Security AI Agent

Version:
    1.0.0
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_schema import DNSFeatureSchema


# =============================================================================
# OPTIONAL SHAP DEPENDENCY
# =============================================================================

try:
    import shap

    SHAP_AVAILABLE = True

except ImportError:

    shap = None

    SHAP_AVAILABLE = False


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# VERSION
# =============================================================================

EXPLAINER_VERSION = "1.0.0"


# =============================================================================
# DNS EXPLAINER
# =============================================================================

class DNSExplainer:
    """
    Production-oriented Explainable AI engine for the DNS Security Agent.

    The class explains a single DNS prediction using either:

        1. SHAP TreeExplainer
        2. Deterministic DNS heuristic fallback

    The class is intentionally independent from:

        - DNS resolution
        - model training
        - risk scoring
        - final multi-agent fusion

    This separation makes the architecture easier to maintain and test.
    """

    # =========================================================================
    # 1. HUMAN-READABLE FEATURE DESCRIPTIONS
    # =========================================================================

    FEATURE_DESCRIPTIONS: Dict[str, str] = {

        # ---------------------------------------------------------------------
        # Routing / Infrastructure
        # ---------------------------------------------------------------------

        "has_a_records":
            "Presence of IPv4 A records",

        "has_aaaa_records":
            "Presence of IPv6 AAAA records",

        "has_cname_record":
            "Presence of a CNAME record",

        "a_record_count":
            "Number of IPv4 addresses returned by DNS",

        "aaaa_record_count":
            "Number of IPv6 addresses returned by DNS",

        "ns_count":
            "Number of authoritative nameservers",

        "cname_count":
            "Number of CNAME records",

        "total_resolved_ips":
            "Total number of resolved IP addresses",

        "has_routing_anomalies":
            "DNS routing or resolution anomalies",

        # ---------------------------------------------------------------------
        # TTL / Fast Flux
        # ---------------------------------------------------------------------

        "min_ttl_value":
            "Minimum DNS record TTL",

        "max_ttl_value":
            "Maximum DNS record TTL",

        "avg_ttl_value":
            "Average DNS record TTL",

        "is_fast_flux_candidate":
            "DNS characteristics associated with possible fast-flux behavior",

        "is_active_fast_flux":
            "Active fast-flux infrastructure rotation",

        # ---------------------------------------------------------------------
        # MX / Email
        # ---------------------------------------------------------------------

        "has_mx_records":
            "Presence of mail-exchange records",

        "mx_record_count":
            "Number of mail-exchange records",

        "uses_free_mail_provider":
            "Use of a free or consumer-oriented mail provider",

        "uses_disposable_mail_provider":
            "Use of a disposable or temporary mail provider",

        "has_suspicious_mx_exchange":
            "Suspicious mail-exchange infrastructure",

        "lowest_mx_preference":
            "Lowest MX preference value",

        # ---------------------------------------------------------------------
        # TXT
        # ---------------------------------------------------------------------

        "has_txt_records":
            "Presence of DNS TXT records",

        "txt_record_count":
            "Number of TXT records",

        "has_domain_verification":
            "Presence of a recognized domain-verification record",

        "has_suspicious_long_txt":
            "Unusually long or suspicious TXT record",

        "has_base64_payload_in_txt":
            "Potential Base64-encoded payload in TXT records",

        "avg_txt_length":
            "Average TXT record length",

        # ---------------------------------------------------------------------
        # SPF
        # ---------------------------------------------------------------------

        "has_spf_record":
            "Presence of an SPF record",

        "spf_record_count":
            "Number of SPF records",

        "has_multiple_spf_records":
            "Multiple SPF records detected",

        "spf_includes_count":
            "Number of SPF include mechanisms",

        "spf_strictness_score":
            "Strength of SPF policy configuration",

        # ---------------------------------------------------------------------
        # DMARC
        # ---------------------------------------------------------------------

        "has_dmarc_record":
            "Presence of a DMARC record",

        "dmarc_policy_score":
            "Strength of DMARC enforcement policy",

        "dmarc_subdomain_policy_score":
            "Strength of DMARC subdomain policy",

        "is_email_spoofable":
            "Domain characteristics associated with email spoofing risk",

        # ---------------------------------------------------------------------
        # Legacy aliases
        # ---------------------------------------------------------------------

        "has_a_record":
            "Presence of IPv4 routing",

        "has_aaaa_record":
            "Presence of IPv6 routing",

        "has_mx_record":
            "Presence of a mail server",

        "dns_ttl":
            "DNS time-to-live value",

        "dns_resolution_failed":
            "DNS resolution failure",

        "private_ip_detected":
            "Private or internal IP detected in public DNS",

        "dns_health_score":
            "Composite DNS infrastructure health score",
    }

    # =========================================================================
    # 2. CLASS DEFINITIONS
    # =========================================================================

    CLASS_INDEX_MAP: Dict[str, int] = {
        "legitimate": 0,
        "phishing": 1,
    }

    INDEX_CLASS_MAP: Dict[int, str] = {
        0: "legitimate",
        1: "phishing",
    }

    # =========================================================================
    # 3. HEURISTIC IMPACT WEIGHTS
    # =========================================================================

    HEURISTIC_WEIGHTS: Dict[str, float] = {

        "is_active_fast_flux":
            0.40,

        "is_fast_flux_candidate":
            0.20,

        "has_routing_anomalies":
            0.35,

        "private_ip_detected":
            0.30,

        "dns_resolution_failed":
            0.30,

        "has_base64_payload_in_txt":
            0.35,

        "has_suspicious_long_txt":
            0.25,

        "uses_disposable_mail_provider":
            0.25,

        "has_suspicious_mx_exchange":
            0.30,

        "is_email_spoofable":
            0.20,

        "has_multiple_spf_records":
            0.15,
    }

    # =========================================================================
    # 4. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model: Optional[Any] = None,
        max_factors: int = 5,
        shap_tolerance: float = 0.0001,
    ) -> None:
        """
        Initialize the DNS XAI engine.

        Parameters
        ----------
        model:
            Trained XGBoost model.

        max_factors:
            Maximum number of important features returned in the explanation.

        shap_tolerance:
            Minimum absolute SHAP value considered meaningful.
        """

        if max_factors <= 0:

            raise ValueError(
                "max_factors must be greater than zero."
            )

        if shap_tolerance < 0:

            raise ValueError(
                "shap_tolerance cannot be negative."
            )

        self.model = model

        self.max_factors = int(
            max_factors
        )

        self.shap_tolerance = float(
            shap_tolerance
        )

        self.explainer = None

        self.shap_initialized = False

        self.explanation_count = 0

        self.shap_explanation_count = 0

        self.heuristic_explanation_count = 0

        self._initialize_explainer()

        logger.info(
            "DNSExplainer initialized. "
            "SHAP available=%s, model supplied=%s",
            SHAP_AVAILABLE,
            model is not None,
        )

    # =========================================================================
    # 5. SHAP INITIALIZATION
    # =========================================================================

    def _initialize_explainer(self) -> None:
        """
        Initialize SHAP TreeExplainer for the supplied tree-based model.

        XGBoost is the expected model family for this DNS Agent.
        """

        if not SHAP_AVAILABLE:

            logger.warning(
                "SHAP is not installed. "
                "DNS explanations will use heuristic fallback."
            )

            return

        if self.model is None:

            logger.info(
                "No DNS ML model supplied. "
                "SHAP initialization skipped."
            )

            return

        try:

            self.explainer = (
                shap.TreeExplainer(
                    self.model
                )
            )

            self.shap_initialized = True

            logger.info(
                "DNS SHAP TreeExplainer initialized successfully."
            )

        except Exception as exc:

            self.explainer = None

            self.shap_initialized = False

            logger.warning(
                "Failed to initialize SHAP TreeExplainer: %s",
                exc,
            )

    # =========================================================================
    # 6. MODEL UPDATE
    # =========================================================================

    def set_model(
        self,
        model: Optional[Any],
    ) -> bool:
        """
        Replace the underlying ML model and reinitialize SHAP.

        Useful when the DNS model is reloaded after training.
        """

        self.model = model

        self.explainer = None

        self.shap_initialized = False

        self._initialize_explainer()

        return self.shap_initialized

    # =========================================================================
    # 7. INPUT NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_features(
        features: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Safely normalize feature input.
        """

        if features is None:

            return {}

        if not isinstance(
            features,
            Mapping,
        ):

            raise TypeError(
                "DNS features must be a mapping/dictionary."
            )

        return dict(
            features
        )

    # =========================================================================
    # 8. FEATURE ALIGNMENT
    # =========================================================================

    def _align_features(
        self,
        raw_features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Align raw DNS features to the canonical DNS ML schema.

        This guarantees:

            - correct column names
            - correct feature order
            - one-row DataFrame
            - numeric values
            - compatibility with XGBoost/SHAP
        """

        features = dict(
            raw_features
        )

        # ---------------------------------------------------------------------
        # Prefer the canonical schema's normalization implementation.
        # ---------------------------------------------------------------------

        if hasattr(
            DNSFeatureSchema,
            "align_and_validate",
        ):

            dataframe = (
                DNSFeatureSchema.align_and_validate(
                    features
                )
            )

        elif hasattr(
            DNSFeatureSchema,
            "validate_and_normalize",
        ):

            normalized = (
                DNSFeatureSchema.validate_and_normalize(
                    features
                )
            )

            feature_names = (
                DNSFeatureSchema.get_schema()
                if hasattr(
                    DNSFeatureSchema,
                    "get_schema",
                )
                else DNSFeatureSchema.get_feature_names()
            )

            dataframe = pd.DataFrame(
                [
                    {
                        name: normalized.get(
                            name,
                            0.0,
                        )
                        for name in feature_names
                    }
                ]
            )

        else:

            raise AttributeError(
                "DNSFeatureSchema does not expose a supported "
                "normalization method."
            )

        # ---------------------------------------------------------------------
        # Ensure all values are finite.
        # ---------------------------------------------------------------------

        dataframe = dataframe.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        dataframe = dataframe.fillna(
            0.0
        )

        # ---------------------------------------------------------------------
        # Ensure numerical dtype.
        # ---------------------------------------------------------------------

        for column in dataframe.columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).fillna(
                0.0
            )

        return dataframe.astype(
            float
        )

    # =========================================================================
    # 9. FEATURE NAMES
    # =========================================================================

    @staticmethod
    def _get_feature_names() -> List[str]:
        """
        Retrieve the authoritative feature ordering from DNSFeatureSchema.
        """

        if hasattr(
            DNSFeatureSchema,
            "get_schema",
        ):

            return list(
                DNSFeatureSchema.get_schema()
            )

        if hasattr(
            DNSFeatureSchema,
            "get_feature_names",
        ):

            return list(
                DNSFeatureSchema.get_feature_names()
            )

        if hasattr(
            DNSFeatureSchema,
            "EXPECTED_FEATURE_ORDER",
        ):

            return list(
                DNSFeatureSchema.EXPECTED_FEATURE_ORDER
            )

        raise AttributeError(
            "DNSFeatureSchema does not expose feature names."
        )

    # =========================================================================
    # 10. HUMAN DESCRIPTION
    # =========================================================================

    def _get_human_description(
        self,
        feature_name: str,
    ) -> str:
        """
        Convert a machine feature name into an SOC-friendly description.
        """

        if feature_name in (
            self.FEATURE_DESCRIPTIONS
        ):

            return self.FEATURE_DESCRIPTIONS[
                feature_name
            ]

        return (
            feature_name
            .replace(
                "_",
                " ",
            )
            .strip()
            .title()
        )

    # =========================================================================
    # 11. SAFE NUMERIC CONVERSION
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to a finite float.
        """

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
    # 12. BOOLEAN FEATURE CHECK
    # =========================================================================

    @staticmethod
    def _is_enabled(
        value: Any,
    ) -> bool:
        """
        Safely interpret boolean-like feature values.
        """

        if isinstance(
            value,
            bool,
        ):

            return value

        numeric = DNSExplainer._safe_float(
            value,
            default=0.0,
        )

        return numeric >= 1.0

    # =========================================================================
    # 13. SHAP VALUE EXTRACTION
    # =========================================================================

    def _extract_shap_values(
        self,
        shap_values: Any,
        target_class_idx: int,
    ) -> np.ndarray:
        """
        Normalize SHAP's multiple possible output formats.

        SHAP versions can return:

            1. list of arrays
            2. 2D numpy array
            3. 3D numpy array
            4. Explanation object

        This method converts those variants into:

            1D array = one SHAP value per feature
        """

        # ---------------------------------------------------------------------
        # New SHAP Explanation API.
        # ---------------------------------------------------------------------

        if hasattr(
            shap_values,
            "values",
        ):

            values = shap_values.values

        else:

            values = shap_values

        # ---------------------------------------------------------------------
        # List output:
        #
        # [class_0_values, class_1_values]
        # ---------------------------------------------------------------------

        if isinstance(
            values,
            list,
        ):

            if not values:

                return np.array(
                    [],
                    dtype=float,
                )

            if (
                target_class_idx
                < len(values)
            ):

                selected = values[
                    target_class_idx
                ]

            else:

                selected = values[
                    0
                ]

            selected = np.asarray(
                selected
            )

            if selected.ndim >= 2:

                selected = selected[
                    0
                ]

            return np.asarray(
                selected,
                dtype=float,
            ).reshape(
                -1
            )

        # ---------------------------------------------------------------------
        # NumPy output.
        # ---------------------------------------------------------------------

        array = np.asarray(
            values
        )

        # ---------------------------------------------------------------------
        # 1D:
        #
        # [feature_1, feature_2, ...]
        # ---------------------------------------------------------------------

        if array.ndim == 1:

            return array.astype(
                float
            )

        # ---------------------------------------------------------------------
        # 2D:
        #
        # [sample, feature]
        # ---------------------------------------------------------------------

        if array.ndim == 2:

            return array[
                0
            ].astype(
                float
            )

        # ---------------------------------------------------------------------
        # 3D:
        #
        # possible:
        #
        # [sample, feature, class]
        #
        # or:
        #
        # [class, sample, feature]
        # ---------------------------------------------------------------------

        if array.ndim == 3:

            # Most recent SHAP format:
            #
            # [samples, features, classes]

            if (
                array.shape[0] == 1
                and target_class_idx
                < array.shape[2]
            ):

                return array[
                    0,
                    :,
                    target_class_idx
                ].astype(
                    float
                )

            # Legacy/class-first format:
            #
            # [classes, samples, features]

            if (
                target_class_idx
                < array.shape[0]
                and array.shape[1] == 1
            ):

                return array[
                    target_class_idx,
                    0,
                    :
                ].astype(
                    float
                )

            # Conservative fallback.
            return array.reshape(
                -1
            ).astype(
                float
            )

        # ---------------------------------------------------------------------
        # Unsupported shape.
        # ---------------------------------------------------------------------

        logger.warning(
            "Unsupported SHAP value shape: %s",
            array.shape,
        )

        return array.reshape(
            -1
        ).astype(
            float
        )

    # =========================================================================
    # 14. SHAP BASE VALUE EXTRACTION
    # =========================================================================

    def _extract_base_value(
        self,
        shap_result: Any,
        target_class_idx: int,
    ) -> Optional[float]:
        """
        Attempt to extract the SHAP expected/base value.

        This value is useful for research reports but is optional because
        SHAP output formats vary between versions.
        """

        try:

            base_values = getattr(
                shap_result,
                "base_values",
                None,
            )

            if base_values is None:

                return None

            array = np.asarray(
                base_values
            )

            if array.ndim == 0:

                return self._safe_float(
                    array.item()
                )

            if array.ndim == 1:

                if (
                    target_class_idx
                    < len(array)
                ):

                    return self._safe_float(
                        array[
                            target_class_idx
                        ]
                    )

                return self._safe_float(
                    array[0]
                )

            if array.ndim >= 2:

                return self._safe_float(
                    array.reshape(
                        -1
                    )[0]
                )

        except Exception:

            logger.debug(
                "Unable to extract SHAP base value.",
                exc_info=True,
            )

        return None

    # =========================================================================
    # 15. IMPACT DIRECTION
    # =========================================================================

    @staticmethod
    def _impact_direction(
        impact: float,
    ) -> Tuple[str, str]:
        """
        Convert SHAP sign into human-readable direction.

        Positive:
            pushes prediction toward phishing

        Negative:
            pushes prediction toward legitimate

        Note:
            This interpretation assumes the SHAP values correspond to
            the phishing/class-1 output.
        """

        if impact > 0:

            return (
                "increases risk",
                "phishing",
            )

        if impact < 0:

            return (
                "decreases risk",
                "legitimate",
            )

        return (
            "neutral",
            "neutral",
        )

    # =========================================================================
    # 16. SHAP FEATURE FACTORS
    # =========================================================================

    def _build_shap_factors(
        self,
        feature_names: Sequence[str],
        shap_values: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Convert raw SHAP values into structured factor records.
        """

        factors: List[
            Dict[str, Any]
        ] = []

        limit = min(
            len(feature_names),
            len(shap_values),
        )

        for index in range(
            limit
        ):

            feature_name = (
                feature_names[
                    index
                ]
            )

            impact = self._safe_float(
                shap_values[
                    index
                ]
            )

            if (
                abs(impact)
                <= self.shap_tolerance
            ):

                continue

            direction, supports_class = (
                self._impact_direction(
                    impact
                )
            )

            factors.append(
                {
                    "feature": feature_name,

                    "description": (
                        self._get_human_description(
                            feature_name
                        )
                    ),

                    "impact": round(
                        impact,
                        6,
                    ),

                    "absolute_impact": round(
                        abs(impact),
                        6,
                    ),

                    "direction": direction,

                    "supports_class": (
                        supports_class
                    ),

                    "explanation_source": (
                        "SHAP"
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # Most influential factors first.
        # ---------------------------------------------------------------------

        factors.sort(
            key=lambda item: item[
                "absolute_impact"
            ],
            reverse=True,
        )

        return factors[
            :self.max_factors
        ]

    # =========================================================================
    # 17. HEURISTIC FACTOR BUILDER
    # =========================================================================

    def _build_heuristic_factor(
        self,
        feature: str,
        impact: float,
        direction: str,
        supports_class: str,
    ) -> Dict[str, Any]:
        """
        Build a standardized heuristic explanation factor.
        """

        return {
            "feature": feature,

            "description": (
                self._get_human_description(
                    feature
                )
            ),

            "impact": round(
                impact,
                4,
            ),

            "absolute_impact": round(
                abs(impact),
                4,
            ),

            "direction": direction,

            "supports_class": supports_class,

            "explanation_source": (
                "heuristic"
            ),
        }

    # =========================================================================
    # 18. HEURISTIC EXPLANATION
    # =========================================================================

    def _heuristic_explanation(
        self,
        features: Mapping[str, Any],
        prediction: str,
    ) -> Dict[str, Any]:
        """
        Generate a deterministic DNS explanation without SHAP.

        This fallback is deliberately transparent.

        It should never be presented as a SHAP explanation.
        """

        normalized_prediction = (
            str(
                prediction
            )
            .strip()
            .lower()
        )

        if normalized_prediction not in (
            "legitimate",
            "phishing",
        ):

            normalized_prediction = (
                "phishing"
            )

        top_factors: List[
            Dict[str, Any]
        ] = []

        # ---------------------------------------------------------------------
        # Rule evaluation.
        # ---------------------------------------------------------------------

        for feature_name, weight in (
            self.HEURISTIC_WEIGHTS.items()
        ):

            value = features.get(
                feature_name,
                0.0,
            )

            if not self._is_enabled(
                value
            ):

                continue

            top_factors.append(
                self._build_heuristic_factor(
                    feature=feature_name,
                    impact=weight,
                    direction="increases risk",
                    supports_class="phishing",
                )
            )

        # ---------------------------------------------------------------------
        # Strong DNS infrastructure can support legitimate classification.
        #
        # Note:
        # This is only used as a heuristic explanation and not as proof
        # of legitimacy.
        # ---------------------------------------------------------------------

        has_routing = (
            self._is_enabled(
                features.get(
                    "has_a_records",
                    False,
                )
            )
        )

        has_nameservers = (
            self._safe_float(
                features.get(
                    "ns_count",
                    0,
                )
            )
            > 0
        )

        no_routing_anomaly = not (
            self._is_enabled(
                features.get(
                    "has_routing_anomalies",
                    False,
                )
            )
        )

        no_fast_flux = not (
            self._is_enabled(
                features.get(
                    "is_active_fast_flux",
                    False,
                )
            )
        )

        if (
            has_routing
            and has_nameservers
            and no_routing_anomaly
            and no_fast_flux
        ):

            top_factors.append(
                self._build_heuristic_factor(
                    feature="dns_baseline",
                    impact=-0.10,
                    direction="decreases risk",
                    supports_class="legitimate",
                )
            )

        # ---------------------------------------------------------------------
        # Sort.
        # ---------------------------------------------------------------------

        top_factors.sort(
            key=lambda item: item[
                "absolute_impact"
            ],
            reverse=True,
        )

        top_factors = top_factors[
            :self.max_factors
        ]

        # ---------------------------------------------------------------------
        # Baseline if nothing significant was detected.
        # ---------------------------------------------------------------------

        if not top_factors:

            top_factors.append(
                {
                    "feature": "dns_baseline",

                    "description": (
                        "General DNS routing characteristics"
                    ),

                    "impact": 0.0,

                    "absolute_impact": 0.0,

                    "direction": "neutral",

                    "supports_class": (
                        normalized_prediction
                    ),

                    "explanation_source": (
                        "heuristic"
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # Summary.
        # ---------------------------------------------------------------------

        phishing_factors = [
            factor
            for factor in top_factors
            if factor.get(
                "supports_class"
            ) == "phishing"
        ]

        legitimate_factors = [
            factor
            for factor in top_factors
            if factor.get(
                "supports_class"
            ) == "legitimate"
        ]

        if phishing_factors:

            names = [
                factor[
                    "description"
                ]
                for factor in phishing_factors[
                    :2
                ]
            ]

            summary = (
                f"Heuristic DNS analysis identified "
                f"risk-increasing characteristics associated "
                f"with {', '.join(names)}."
            )

        elif legitimate_factors:

            names = [
                factor[
                    "description"
                ]
                for factor in legitimate_factors[
                    :2
                ]
            ]

            summary = (
                f"Heuristic DNS analysis identified "
                f"relatively normal characteristics such as "
                f"{', '.join(names)}."
            )

        else:

            summary = (
                "No strong DNS-specific heuristic indicators "
                "were identified."
            )

        return {
            "summary": summary,

            "top_factors": top_factors,

            "explanation_source": (
                "heuristic"
            ),

            "model_specific": False,

            "shap_available": (
                SHAP_AVAILABLE
            ),

            "shap_used": False,

            "prediction_class": (
                normalized_prediction
            ),
        }

    # =========================================================================
    # 19. SHAP EXPLANATION
    # =========================================================================

    def _shap_explanation(
        self,
        features: Mapping[str, Any],
        prediction: str,
    ) -> Dict[str, Any]:
        """
        Generate a local SHAP explanation for a single DNS prediction.
        """

        if not self.shap_initialized:

            raise RuntimeError(
                "SHAP explainer is not initialized."
            )

        # ---------------------------------------------------------------------
        # Align features.
        # ---------------------------------------------------------------------

        dataframe = (
            self._align_features(
                features
            )
        )

        feature_names = (
            self._get_feature_names()
        )

        # ---------------------------------------------------------------------
        # Ensure DataFrame columns match expected feature names.
        # ---------------------------------------------------------------------

        dataframe = dataframe.reindex(
            columns=feature_names,
            fill_value=0.0,
        )

        # ---------------------------------------------------------------------
        # Determine target class.
        # ---------------------------------------------------------------------

        normalized_prediction = (
            str(
                prediction
            )
            .strip()
            .lower()
        )

        target_class_idx = (
            self.CLASS_INDEX_MAP.get(
                normalized_prediction,
                self.CLASS_INDEX_MAP[
                    "phishing"
                ],
            )
        )

        # ---------------------------------------------------------------------
        # Calculate SHAP values.
        # ---------------------------------------------------------------------

        shap_result = (
            self.explainer(
                dataframe
            )
        )

        shap_values = (
            self._extract_shap_values(
                shap_result,
                target_class_idx,
            )
        )

        # ---------------------------------------------------------------------
        # Build feature factors.
        # ---------------------------------------------------------------------

        top_factors = (
            self._build_shap_factors(
                feature_names,
                shap_values,
            )
        )

        # ---------------------------------------------------------------------
        # Base value.
        # ---------------------------------------------------------------------

        base_value = (
            self._extract_base_value(
                shap_result,
                target_class_idx,
            )
        )

        # ---------------------------------------------------------------------
        # Positive / negative contributors.
        # ---------------------------------------------------------------------

        positive_contributors = [
            factor
            for factor in top_factors
            if factor[
                "impact"
            ] > 0
        ]

        negative_contributors = [
            factor
            for factor in top_factors
            if factor[
                "impact"
            ] < 0
        ]

        # ---------------------------------------------------------------------
        # Natural language summary.
        # ---------------------------------------------------------------------

        if positive_contributors:

            names = [
                factor[
                    "description"
                ]
                for factor in positive_contributors[
                    :3
                ]
            ]

            summary = (
                f"The DNS model classified the instance as "
                f"'{normalized_prediction}'. "
                f"The strongest risk-increasing contributors were: "
                f"{', '.join(names)}."
            )

        elif negative_contributors:

            names = [
                factor[
                    "description"
                ]
                for factor in negative_contributors[
                    :3
                ]
            ]

            summary = (
                f"The DNS model classified the instance as "
                f"'{normalized_prediction}'. "
                f"The strongest risk-reducing contributors were: "
                f"{', '.join(names)}."
            )

        else:

            summary = (
                f"The DNS model classified the instance as "
                f"'{normalized_prediction}', but no individual "
                f"feature produced a significant SHAP contribution "
                f"under the configured tolerance."
            )

        # ---------------------------------------------------------------------
        # Return.
        # ---------------------------------------------------------------------

        return {
            "summary": summary,

            "top_factors": top_factors,

            "explanation_source": (
                "SHAP"
            ),

            "model_specific": True,

            "shap_available": True,

            "shap_used": True,

            "prediction_class": (
                normalized_prediction
            ),

            "target_class_index": (
                target_class_idx
            ),

            "base_value": (
                round(
                    base_value,
                    6,
                )
                if base_value is not None
                else None
            ),

            "positive_contributors": (
                positive_contributors
            ),

            "negative_contributors": (
                negative_contributors
            ),

            "feature_count": (
                len(feature_names)
            ),

            "explainer_version": (
                EXPLAINER_VERSION
            ),
        }

    # =========================================================================
    # 20. MAIN EXPLANATION METHOD
    # =========================================================================

    def generate_explanation(
        self,
        features: Optional[
            Mapping[str, Any]
        ],
        prediction: str,
    ) -> Dict[str, Any]:
        """
        Generate a complete explanation for one DNS prediction.

        Preferred path:

            SHAP

        Fallback path:

            heuristic

        The fallback is automatic and transparent.
        """

        self.explanation_count += 1

        normalized_features = (
            self._normalize_features(
                features
            )
        )

        normalized_prediction = (
            str(
                prediction
            )
            .strip()
            .lower()
        )

        if normalized_prediction not in (
            "legitimate",
            "phishing",
        ):

            logger.warning(
                "Unknown DNS prediction class '%s'. "
                "Defaulting to phishing for explanation.",
                prediction,
            )

            normalized_prediction = (
                "phishing"
            )

        # ---------------------------------------------------------------------
        # Preferred SHAP explanation.
        # ---------------------------------------------------------------------

        if (
            self.shap_initialized
            and self.explainer is not None
            and SHAP_AVAILABLE
        ):

            try:

                result = (
                    self._shap_explanation(
                        normalized_features,
                        normalized_prediction,
                    )
                )

                self.shap_explanation_count += 1

                return result

            except Exception as exc:

                logger.warning(
                    "SHAP explanation failed. "
                    "Switching to heuristic explanation: %s",
                    exc,
                    exc_info=True,
                )

        # ---------------------------------------------------------------------
        # Heuristic fallback.
        # ---------------------------------------------------------------------

        self.heuristic_explanation_count += 1

        return (
            self._heuristic_explanation(
                normalized_features,
                normalized_prediction,
            )
        )

    # =========================================================================
    # 21. EXPLAIN PREDICTOR RESULT
    # =========================================================================

    def explain_prediction(
        self,
        prediction_result: Mapping[str, Any],
        features: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Explain a DNSPredictor result.

        Expected predictor result:

            {
                "class_label": "phishing",
                "phishing_probability": 0.87,
                "features_dataframe": ...
            }

        This method automatically extracts the prediction class and,
        where possible, the processed feature DataFrame.
        """

        if not isinstance(
            prediction_result,
            Mapping,
        ):

            raise TypeError(
                "prediction_result must be a mapping."
            )

        # ---------------------------------------------------------------------
        # Determine prediction class.
        # ---------------------------------------------------------------------

        prediction = prediction_result.get(
            "class_label",
            "phishing",
        )

        # ---------------------------------------------------------------------
        # Use explicitly supplied features first.
        # ---------------------------------------------------------------------

        if features is None:

            dataframe = prediction_result.get(
                "features_dataframe"
            )

            if isinstance(
                dataframe,
                pd.DataFrame,
            ) and not dataframe.empty:

                try:

                    features = (
                        dataframe.iloc[
                            0
                        ].to_dict()
                    )

                except Exception:

                    logger.debug(
                        "Unable to extract features from prediction DataFrame.",
                        exc_info=True,
                    )

        # ---------------------------------------------------------------------
        # Generate explanation.
        # ---------------------------------------------------------------------

        explanation = (
            self.generate_explanation(
                features or {},
                str(
                    prediction
                ),
            )
        )

        # ---------------------------------------------------------------------
        # Attach prediction context.
        # ---------------------------------------------------------------------

        explanation[
            "phishing_probability"
        ] = self._safe_float(
            prediction_result.get(
                "phishing_probability",
                0.0,
            )
        )

        explanation[
            "confidence"
        ] = self._safe_float(
            prediction_result.get(
                "confidence",
                0.0,
            )
        )

        explanation[
            "model_status"
        ] = prediction_result.get(
            "model_status",
            "unknown",
        )

        explanation[
            "prediction_source"
        ] = prediction_result.get(
            "prediction_source",
            "unknown",
        )

        return explanation

    # =========================================================================
    # 22. EXPLAIN WITH RISK CONTEXT
    # =========================================================================

    def explain_assessment(
        self,
        prediction_result: Mapping[str, Any],
        risk_result: Optional[
            Mapping[str, Any]
        ] = None,
        features: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete XAI assessment containing:

            - model prediction
            - SHAP/heuristic factors
            - risk score
            - risk level
            - triggered indicators
            - posture context

        This method is especially useful for dns_agent.py.
        """

        explanation = (
            self.explain_prediction(
                prediction_result,
                features,
            )
        )

        if risk_result is not None:

            explanation[
                "risk_score"
            ] = risk_result.get(
                "risk_score",
                0,
            )

            explanation[
                "risk_level"
            ] = risk_result.get(
                "risk_level",
                "Unknown",
            )

            explanation[
                "triggered_indicators"
            ] = risk_result.get(
                "triggered_indicators",
                [],
            )

            explanation[
                "posture_breakdown"
            ] = risk_result.get(
                "posture_breakdown",
                {},
            )

            explanation[
                "indicator_summary"
            ] = risk_result.get(
                "indicator_summary",
                {},
            )

        return explanation

    # =========================================================================
    # 23. GET TOP FACTORS
    # =========================================================================

    def get_top_factors(
        self,
        explanation: Mapping[str, Any],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract the strongest explanatory factors.

        Useful for dashboards where only the top 3-5 factors should be shown.
        """

        factors = explanation.get(
            "top_factors",
            [],
        )

        if not isinstance(
            factors,
            list,
        ):

            return []

        factor_limit = (
            self.max_factors
            if limit is None
            else max(
                1,
                int(
                    limit
                ),
            )
        )

        return factors[
            :factor_limit
        ]

    # =========================================================================
    # 24. GET RISK-INCREASING FACTORS
    # =========================================================================

    def get_risk_increasing_factors(
        self,
        explanation: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Return only features that increase phishing risk.
        """

        factors = explanation.get(
            "top_factors",
            [],
        )

        return [
            factor
            for factor in factors
            if (
                factor.get(
                    "direction"
                )
                == "increases risk"
            )
        ]

    # =========================================================================
    # 25. GET RISK-DECREASING FACTORS
    # =========================================================================

    def get_risk_decreasing_factors(
        self,
        explanation: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Return only features that decrease phishing risk.
        """

        factors = explanation.get(
            "top_factors",
            [],
        )

        return [
            factor
            for factor in factors
            if (
                factor.get(
                    "direction"
                )
                == "decreases risk"
            )
        ]

    # =========================================================================
    # 26. FEATURE IMPORTANCE SUMMARY
    # =========================================================================

    def get_feature_importance_summary(
        self,
        explanation: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Return a compact mapping:

            feature -> absolute importance

        Useful for charts and dashboards.
        """

        factors = explanation.get(
            "top_factors",
            [],
        )

        summary: Dict[
            str,
            float
        ] = {}

        for factor in factors:

            feature = factor.get(
                "feature"
            )

            if not feature:

                continue

            impact = self._safe_float(
                factor.get(
                    "absolute_impact",
                    0.0,
                )
            )

            summary[
                str(
                    feature
                )
            ] = round(
                impact,
                6,
            )

        return summary

    # =========================================================================
    # 27. EXPLAINABILITY STATUS
    # =========================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return the current XAI engine status.
        """

        if (
            self.shap_initialized
            and self.explainer is not None
        ):

            explanation_mode = "SHAP"

        else:

            explanation_mode = "heuristic"

        return {
            "explainer_version": (
                EXPLAINER_VERSION
            ),

            "shap_available": (
                SHAP_AVAILABLE
            ),

            "shap_initialized": (
                self.shap_initialized
            ),

            "model_available": (
                self.model is not None
            ),

            "explanation_mode": (
                explanation_mode
            ),

            "max_factors": (
                self.max_factors
            ),

            "shap_tolerance": (
                self.shap_tolerance
            ),

            "total_explanations": (
                self.explanation_count
            ),

            "shap_explanations": (
                self.shap_explanation_count
            ),

            "heuristic_explanations": (
                self.heuristic_explanation_count
            ),
        }

    # =========================================================================
    # 28. RESET STATISTICS
    # =========================================================================

    def reset_statistics(
        self,
    ) -> None:
        """
        Reset runtime explanation counters.
        """

        self.explanation_count = 0

        self.shap_explanation_count = 0

        self.heuristic_explanation_count = 0

        logger.debug(
            "DNS XAI statistics reset."
        )

    # =========================================================================
    # 29. JSON-SAFE CONVERSION
    # =========================================================================

    @staticmethod
    def make_serializable(
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert NumPy/Pandas values into JSON-compatible Python objects.

        This is useful before sending XAI results to:

            - Flask/FastAPI
            - PostgreSQL
            - Elasticsearch
            - JSON files
            - dashboard APIs
        """

        def convert(
            value: Any,
        ) -> Any:

            # -------------------------------------------------------------
            # Dictionary
            # -------------------------------------------------------------

            if isinstance(
                value,
                dict,
            ):

                return {
                    str(key): convert(
                        item
                    )
                    for key, item in value.items()
                }

            # -------------------------------------------------------------
            # List
            # -------------------------------------------------------------

            if isinstance(
                value,
                list,
            ):

                return [
                    convert(item)
                    for item in value
                ]

            # -------------------------------------------------------------
            # Tuple
            # -------------------------------------------------------------

            if isinstance(
                value,
                tuple,
            ):

                return [
                    convert(item)
                    for item in value
                ]

            # -------------------------------------------------------------
            # NumPy scalar
            # -------------------------------------------------------------

            if hasattr(
                value,
                "item",
            ):

                try:

                    return value.item()

                except Exception:

                    pass

            # -------------------------------------------------------------
            # Pandas scalar
            # -------------------------------------------------------------

            if pd.isna(
                value
            ):

                return None

            return value

        return convert(
            result
        )
"""
SSL/TLS Security AI Agent - Explainability Engine
==================================================

Production-grade Explainable AI (XAI) engine for the SSL/TLS Security
AI Agent.

Purpose
-------

This module converts the SSL AI model prediction into an analyst-friendly
explanation.

Primary explanation method:

    SHAP TreeExplainer

Fallback explanation method:

    Rule-based SSL/TLS heuristic explanation


Architecture
------------

    SSL Feature Vector
            |
            v
       SSL Predictor
            |
            v
       XGBoost Model
            |
            +------------------+
            |                  |
            v                  v
       Probability         SHAP Values
            |                  |
            +--------+---------+
                     |
                     v
              SSLExplainer
                     |
                     v
       Human-readable explanation
                     |
          +----------+----------+
          |          |          |
          v          v          v
      Summary    Top Factors  Evidence
                     |
                     v
              PDF / Dashboard
              / Fusion Engine


Important
---------

This module does NOT calculate the final SSL risk score.

Risk scoring belongs to:

    agents/ssl_agent/risk_score.py

This module only explains why the model/rules produced a particular
prediction.
"""

import logging

from typing import (
    Dict,
    Any,
    List,
    Optional,
    Tuple
)

import math

import numpy as np

import pandas as pd

from .feature_schema import SSLFeatureSchema


# ==========================================================================
# OPTIONAL SHAP DEPENDENCY
# ==========================================================================

try:

    import shap

    SHAP_AVAILABLE = True

except ImportError:

    shap = None

    SHAP_AVAILABLE = False


logger = logging.getLogger(__name__)


class SSLExplainer:
    """
    Production-grade Explainable AI engine for the SSL/TLS AI Agent.

    Main responsibilities
    ---------------------

        • Initialize SHAP TreeExplainer
        • Validate model availability
        • Align SSL features with training schema
        • Calculate local SHAP values
        • Handle different SHAP output formats
        • Identify positive risk contributors
        • Identify protective/legitimate contributors
        • Generate human-readable explanations
        • Provide deterministic heuristic fallback
        • Return structured XAI data
        • Support dashboard/PDF report generation

    The class is intentionally independent from:

        • SSL feature extraction
        • SSL model training
        • Risk scoring
        • Decision fusion
    """

    # ======================================================================
    # METADATA
    # ======================================================================

    EXPLAINER_NAME = (
        "SSL Explainability Engine"
    )

    EXPLAINER_VERSION = "1.0.0"

    # ======================================================================
    # CLASS LABELS
    # ======================================================================

    CLASS_INDEX_MAP: Dict[
        str,
        int
    ] = {

        "legitimate":
            0,

        "phishing":
            1,

        "suspicious":
            1,

        "malicious":
            1
    }

    CLASS_LABEL_MAP: Dict[
        int,
        str
    ] = {

        0:
            "legitimate",

        1:
            "phishing"
    }

    # ======================================================================
    # FEATURE DESCRIPTIONS
    # ======================================================================

    FEATURE_DESCRIPTIONS: Dict[
        str,
        str
    ] = {

        "has_ssl":
            "Active SSL/TLS connection",

        "is_expired":
            "Certificate expiration status",

        "cert_age_days":
            "Certificate age in days",

        "days_until_expiry":
            "Remaining certificate validity window",

        "total_lifespan_days":
            "Total authorized certificate lifespan",

        "is_self_signed":
            "Self-signed certificate without a trusted CA",

        "is_recently_issued":
            "Recently provisioned certificate",

        "is_short_lived":
            "Short-lived certificate",

        "is_trusted_ca":
            "Certificate issued by a trusted certificate authority",

        "is_free_automated_ca":
            "Certificate issued by a free or automated certificate authority",

        "is_suspicious_ca":
            "Suspicious or untrusted certificate authority",

        "issuer_trust_score":
            "Certificate authority trustworthiness score",

        "is_weak_protocol":
            "Weak or deprecated TLS protocol",

        "is_vulnerable_cipher":
            "Potentially vulnerable cryptographic cipher",

        "cipher_strength_bits":
            "Cryptographic key strength in bits",

        "crypto_health_score":
            "Overall cryptographic configuration health",

        "is_secure_connection":
            "Overall secure connection status"
    }

    # ======================================================================
    # FEATURE CATEGORIES
    # ======================================================================

    FEATURE_CATEGORIES: Dict[
        str,
        str
    ] = {

        "has_ssl":
            "connection",

        "is_expired":
            "certificate",

        "cert_age_days":
            "certificate",

        "days_until_expiry":
            "certificate",

        "total_lifespan_days":
            "certificate",

        "is_self_signed":
            "certificate",

        "is_recently_issued":
            "certificate",

        "is_short_lived":
            "certificate",

        "is_trusted_ca":
            "trust",

        "is_free_automated_ca":
            "trust",

        "is_suspicious_ca":
            "trust",

        "issuer_trust_score":
            "trust",

        "is_weak_protocol":
            "cryptography",

        "is_vulnerable_cipher":
            "cryptography",

        "cipher_strength_bits":
            "cryptography",

        "crypto_health_score":
            "cryptography",

        "is_secure_connection":
            "connection"
    }

    # ======================================================================
    # FEATURE EXPLANATION WEIGHTS
    # ======================================================================

    HEURISTIC_WEIGHTS: Dict[
        str,
        float
    ] = {

        "is_expired":
            0.40,

        "is_self_signed":
            0.40,

        "is_suspicious_ca":
            0.30,

        "is_weak_protocol":
            0.30,

        "is_vulnerable_cipher":
            0.20,

        "is_secure_connection":
            0.20,

        "crypto_health_score":
            0.20,

        "issuer_trust_score":
            0.15,

        "is_recently_issued":
            0.15,

        "is_short_lived":
            0.10,

        "is_free_automated_ca":
            0.05
    }

    # ======================================================================
    # FEATURES TO SUPPRESS
    # ======================================================================

    """
    Some features can be technically associated with phishing without
    being inherently malicious.

    For example:

        has_ssl = True

    should not automatically be explained as:

        "Having SSL increases phishing risk."

    That would be misleading to a security analyst.

    Therefore these features are filtered when their positive SHAP
    contribution would create a misleading explanation.
    """

    MISLEADING_POSITIVE_FEATURES = {

        "has_ssl",

        "is_secure_connection"
    }

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(
        self,
        model: Any = None,
        max_factors: int = 5,
        shap_enabled: bool = True
    ):
        """
        Initialize the SSL explanation engine.

        Args:
            model:
                Fitted XGBoost model.

            max_factors:
                Maximum number of top explanation factors returned.

            shap_enabled:
                Whether SHAP should be used when available.
        """

        self.model = model

        self.max_factors = max(
            1,
            int(
                max_factors
            )
        )

        self.shap_enabled = bool(
            shap_enabled
        )

        self.explainer = None

        self.last_error: Optional[
            str
        ] = None

        self._initialize_explainer()

    # ======================================================================
    # INITIALIZE SHAP
    # ======================================================================

    def _initialize_explainer(
        self
    ) -> None:
        """
        Initialize SHAP TreeExplainer.

        If SHAP or the model is unavailable, the heuristic explanation
        engine remains available.
        """

        # ------------------------------------------------------------------
        # SHAP disabled
        # ------------------------------------------------------------------

        if not self.shap_enabled:

            logger.info(
                "SSL SHAP explainability is disabled."
            )

            return

        # ------------------------------------------------------------------
        # SHAP unavailable
        # ------------------------------------------------------------------

        if not SHAP_AVAILABLE:

            logger.warning(
                (
                    "SHAP library is not installed. "
                    "SSLExplainer will use heuristic explanations."
                )
            )

            return

        # ------------------------------------------------------------------
        # Model unavailable
        # ------------------------------------------------------------------

        if self.model is None:

            logger.warning(
                (
                    "No SSL model supplied to SSLExplainer. "
                    "Heuristic explanations will be used."
                )
            )

            return

        # ------------------------------------------------------------------
        # Initialize TreeExplainer
        # ------------------------------------------------------------------

        try:

            self.explainer = (
                shap.TreeExplainer(
                    self.model
                )
            )

            logger.info(
                (
                    "SSL SHAP TreeExplainer initialized successfully."
                )
            )

        except Exception as e:

            self.explainer = None

            self.last_error = str(
                e
            )

            logger.warning(
                (
                    "Failed to initialize SSL SHAP TreeExplainer: %s"
                ),
                str(e)
            )

    # ======================================================================
    # UPDATE MODEL
    # ======================================================================

    def set_model(
        self,
        model: Any
    ) -> bool:
        """
        Set or replace the underlying ML model.

        Useful when the predictor loads a model after the explainer
        has already been initialized.
        """

        self.model = model

        self.explainer = None

        self.last_error = None

        self._initialize_explainer()

        return bool(
            self.explainer is not None
        )

    # ======================================================================
    # GET MODEL
    # ======================================================================

    def get_model(
        self
    ) -> Any:
        """
        Return the underlying model.
        """

        return self.model

    # ======================================================================
    # FEATURE ALIGNMENT
    # ======================================================================

    def _align_features(
        self,
        raw_features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Align raw SSL features to the canonical SSLFeatureSchema.

        The output always contains the exact ordered feature vector
        expected by the model.
        """

        # ------------------------------------------------------------------
        # Normalize using central schema
        # ------------------------------------------------------------------

        clean_features = (
            SSLFeatureSchema.validate_and_normalize(
                raw_features
            )
        )

        feature_names = (
            SSLFeatureSchema.get_feature_names()
        )

        ordered_features = {}

        for feature_name in (
            feature_names
        ):

            value = (
                clean_features.get(
                    feature_name,
                    0
                )
            )

            ordered_features[
                feature_name
            ] = self._safe_numeric(
                value
            )

        # ------------------------------------------------------------------
        # Build one-row DataFrame
        # ------------------------------------------------------------------

        df = pd.DataFrame(
            [
                ordered_features
            ],
            columns=feature_names
        )

        # ------------------------------------------------------------------
        # Final numeric cleanup
        # ------------------------------------------------------------------

        df = df.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        df = df.fillna(
            0.0
        )

        return df.astype(
            float
        )

    # ======================================================================
    # SAFE NUMERIC
    # ======================================================================

    @staticmethod
    def _safe_numeric(
        value: Any,
        default: float = 0.0
    ) -> float:
        """
        Safely convert a feature value to a finite float.
        """

        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return default

        if not math.isfinite(
            numeric_value
        ):

            return default

        return numeric_value

    # ======================================================================
    # HUMAN DESCRIPTION
    # ======================================================================

    def _get_human_description(
        self,
        feature_name: str
    ) -> str:
        """
        Convert a raw feature name into a SOC analyst-friendly description.
        """

        return (
            self.FEATURE_DESCRIPTIONS.get(
                feature_name,
                feature_name
                .replace(
                    "_",
                    " "
                )
                .title()
            )
        )

    # ======================================================================
    # FEATURE CATEGORY
    # ======================================================================

    def _get_feature_category(
        self,
        feature_name: str
    ) -> str:
        """
        Return the security category associated with a feature.
        """

        return (
            self.FEATURE_CATEGORIES.get(
                feature_name,
                "ssl_tls"
            )
        )

    # ======================================================================
    # CLASS NORMALIZATION
    # ======================================================================

    def _normalize_prediction(
        self,
        prediction: Any
    ) -> str:
        """
        Normalize prediction labels.

        Supported labels:

            legitimate
            phishing
            suspicious
            malicious
        """

        if prediction is None:

            return "phishing"

        prediction_string = str(
            prediction
        ).strip().lower()

        if prediction_string in {
            "legitimate",
            "safe",
            "benign",
            "0"
        }:

            return "legitimate"

        if prediction_string in {
            "phishing",
            "suspicious",
            "malicious",
            "unsafe",
            "1"
        }:

            return "phishing"

        return prediction_string

    # ======================================================================
    # CLASS INDEX
    # ======================================================================

    def _get_target_class_index(
        self,
        prediction: str
    ) -> int:
        """
        Convert prediction label into model class index.
        """

        normalized = (
            self._normalize_prediction(
                prediction
            )
        )

        return int(
            self.CLASS_INDEX_MAP.get(
                normalized,
                1
            )
        )

    # ======================================================================
    # SHAP VALUE EXTRACTION
    # ======================================================================

    def _extract_instance_shap_values(
        self,
        shap_values: Any,
        target_class_index: int
    ) -> np.ndarray:
        """
        Normalize SHAP's multiple possible output formats into:

            shape = (number_of_features,)

        SHAP versions and model configurations can produce:

            • list of arrays
            • 2D ndarray
            • 3D ndarray
            • Explanation object
        """

        # ------------------------------------------------------------------
        # SHAP Explanation object
        # ------------------------------------------------------------------

        if hasattr(
            shap_values,
            "values"
        ):

            values = (
                shap_values.values
            )

        else:

            values = shap_values

        # ------------------------------------------------------------------
        # List output
        # ------------------------------------------------------------------

        if isinstance(
            values,
            list
        ):

            if not values:

                raise ValueError(
                    "SHAP returned an empty list."
                )

            if target_class_index < len(
                values
            ):

                selected = values[
                    target_class_index
                ]

            else:

                selected = values[
                    -1
                ]

            selected = np.asarray(
                selected
            )

            if selected.ndim == 2:

                return selected[
                    0
                ].astype(
                    float
                )

            return selected.reshape(
                -1
            ).astype(
                float
            )

        # ------------------------------------------------------------------
        # Convert ndarray
        # ------------------------------------------------------------------

        values = np.asarray(
            values
        )

        # ------------------------------------------------------------------
        # Empty
        # ------------------------------------------------------------------

        if values.size == 0:

            raise ValueError(
                "SHAP returned an empty array."
            )

        # ------------------------------------------------------------------
        # 1D
        # ------------------------------------------------------------------

        if values.ndim == 1:

            return values.astype(
                float
            )

        # ------------------------------------------------------------------
        # 2D
        #
        # Usually:
        #
        #     samples × features
        # ------------------------------------------------------------------

        if values.ndim == 2:

            return values[
                0
            ].astype(
                float
            )

        # ------------------------------------------------------------------
        # 3D
        #
        # Possible:
        #
        #     samples × features × classes
        #
        # or:
        #
        #     classes × samples × features
        # ------------------------------------------------------------------

        if values.ndim == 3:

            # --------------------------------------------------------------
            # samples × features × classes
            # --------------------------------------------------------------

            if (
                values.shape[0] == 1
                and
                values.shape[2] > 1
            ):

                class_index = min(
                    target_class_index,
                    values.shape[2] - 1
                )

                return values[
                    0,
                    :,
                    class_index
                ].astype(
                    float
                )

            # --------------------------------------------------------------
            # classes × samples × features
            # --------------------------------------------------------------

            if (
                values.shape[1] == 1
                and
                values.shape[0] > 1
            ):

                class_index = min(
                    target_class_index,
                    values.shape[0] - 1
                )

                return values[
                    class_index,
                    0,
                    :
                ].astype(
                    float
                )

            # --------------------------------------------------------------
            # Generic fallback
            # --------------------------------------------------------------

            return values.reshape(
                -1
            ).astype(
                float
            )

        # ------------------------------------------------------------------
        # Unsupported dimensionality
        # ------------------------------------------------------------------

        raise ValueError(
            (
                "Unsupported SHAP output dimensions: "
                f"{values.shape}"
            )
        )

    # ======================================================================
    # SHAP IMPACT DIRECTION
    # ======================================================================

    @staticmethod
    def _get_impact_direction(
        impact: float
    ) -> Tuple[str, str]:
        """
        Translate SHAP sign into direction and supporting class.

        Positive SHAP:

            pushes toward phishing

        Negative SHAP:

            pushes toward legitimate
        """

        if impact > 0:

            return (
                "increases risk",
                "phishing"
            )

        if impact < 0:

            return (
                "decreases risk",
                "legitimate"
            )

        return (
            "neutral",
            "neutral"
        )

    # ======================================================================
    # SHAP FACTOR EXTRACTION
    # ======================================================================

    def _extract_shap_factors(
        self,
        df_features: pd.DataFrame,
        prediction: str
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Calculate and structure local SHAP feature contributions.
        """

        if self.explainer is None:

            raise RuntimeError(
                "SHAP explainer is not initialized."
            )

        feature_names = (
            SSLFeatureSchema.get_feature_names()
        )

        target_class_index = (
            self._get_target_class_index(
                prediction
            )
        )

        # ------------------------------------------------------------------
        # Compute SHAP values
        # ------------------------------------------------------------------

        shap_values = (
            self.explainer.shap_values(
                df_features
            )
        )

        instance_shap = (
            self._extract_instance_shap_values(
                shap_values,
                target_class_index
            )
        )

        # ------------------------------------------------------------------
        # Validate number of values
        # ------------------------------------------------------------------

        if len(
            instance_shap
        ) != len(
            feature_names
        ):

            raise ValueError(
                (
                    "SHAP feature count does not match "
                    "SSL schema. "
                    f"SHAP={len(instance_shap)}, "
                    f"Schema={len(feature_names)}"
                )
            )

        feature_impacts = []

        # ------------------------------------------------------------------
        # Build factor list
        # ------------------------------------------------------------------

        for index, feature_name in enumerate(
            feature_names
        ):

            impact = self._safe_numeric(
                instance_shap[
                    index
                ]
            )

            # --------------------------------------------------------------
            # Ignore mathematically negligible values
            # --------------------------------------------------------------

            if abs(
                impact
            ) < 0.0001:

                continue

            # --------------------------------------------------------------
            # Guardrail against misleading explanations
            # --------------------------------------------------------------

            if (
                feature_name
                in
                self.MISLEADING_POSITIVE_FEATURES
                and
                impact > 0
            ):

                continue

            direction, supports_class = (
                self._get_impact_direction(
                    impact
                )
            )

            feature_impacts.append({

                "feature":
                    feature_name,

                "description":
                    self._get_human_description(
                        feature_name
                    ),

                "category":
                    self._get_feature_category(
                        feature_name
                    ),

                "impact":
                    round(
                        impact,
                        6
                    ),

                "absolute_impact":
                    round(
                        abs(
                            impact
                        ),
                        6
                    ),

                "direction":
                    direction,

                "supports_class":
                    supports_class,

                "value":
                    self._safe_numeric(
                        df_features.iloc[
                            0
                        ][
                            feature_name
                        ]
                    ),

                "method":
                    "SHAP"
            })

        # ------------------------------------------------------------------
        # Sort by absolute SHAP impact
        # ------------------------------------------------------------------

        feature_impacts.sort(
            key=lambda item:
                item[
                    "absolute_impact"
                ],
            reverse=True
        )

        return feature_impacts

    # ======================================================================
    # HEURISTIC FACTOR
    # ======================================================================

    def _create_heuristic_factor(
        self,
        feature: str,
        impact: float,
        direction: str,
        supports_class: str,
        value: Any,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build one standardized heuristic explanation factor.
        """

        return {

            "feature":
                feature,

            "description":
                description
                or
                self._get_human_description(
                    feature
                ),

            "category":
                self._get_feature_category(
                    feature
                ),

            "impact":
                round(
                    float(
                        impact
                    ),
                    4
                ),

            "absolute_impact":
                round(
                    abs(
                        float(
                            impact
                        )
                    ),
                    4
                ),

            "direction":
                direction,

            "supports_class":
                supports_class,

            "value":
                value,

            "method":
                "heuristic"
        }

    # ======================================================================
    # HEURISTIC EXPLANATION
    # ======================================================================

    def _heuristic_explanation(
        self,
        features: Dict[str, Any],
        prediction: str
    ) -> Dict[str, Any]:
        """
        Generate deterministic SSL/TLS explanation when SHAP is unavailable.

        The heuristic engine is not a replacement for SHAP.

        It provides understandable security evidence so that the complete
        agent remains operational even without SHAP.
        """

        normalized_features = (
            SSLFeatureSchema.validate_and_normalize(
                features
            )
        )

        prediction = (
            self._normalize_prediction(
                prediction
            )
        )

        top_factors: List[
            Dict[str, Any]
        ] = []

        # ==============================================================
        # EXPIRED CERTIFICATE
        # ==============================================================

        if self._is_enabled(
            normalized_features,
            "is_expired"
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="is_expired",
                    impact=0.40,
                    direction="increases risk",
                    supports_class="phishing",
                    value=1.0
                )
            )

        # ==============================================================
        # SELF-SIGNED CERTIFICATE
        # ==============================================================

        if self._is_enabled(
            normalized_features,
            "is_self_signed"
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="is_self_signed",
                    impact=0.40,
                    direction="increases risk",
                    supports_class="phishing",
                    value=1.0
                )
            )

        # ==============================================================
        # SUSPICIOUS CA
        # ==============================================================

        if self._is_enabled(
            normalized_features,
            "is_suspicious_ca"
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="is_suspicious_ca",
                    impact=0.30,
                    direction="increases risk",
                    supports_class="phishing",
                    value=1.0
                )
            )

        # ==============================================================
        # WEAK TLS PROTOCOL
        # ==============================================================

        if self._is_enabled(
            normalized_features,
            "is_weak_protocol"
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="is_weak_protocol",
                    impact=0.30,
                    direction="increases risk",
                    supports_class="phishing",
                    value=1.0
                )
            )

        # ==============================================================
        # VULNERABLE CIPHER
        # ==============================================================

        if self._is_enabled(
            normalized_features,
            "is_vulnerable_cipher"
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="is_vulnerable_cipher",
                    impact=0.20,
                    direction="increases risk",
                    supports_class="phishing",
                    value=1.0
                )
            )

        # ==============================================================
        # LOW TRUST SCORE
        # ==============================================================

        issuer_trust_score = (
            self._safe_numeric(
                normalized_features.get(
                    "issuer_trust_score",
                    0
                )
            )
        )

        if issuer_trust_score < 30:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="issuer_trust_score",
                    impact=0.15,
                    direction="increases risk",
                    supports_class="phishing",
                    value=issuer_trust_score
                )
            )

        # ==============================================================
        # POOR CRYPTO HEALTH
        # ==============================================================

        crypto_health = (
            self._safe_numeric(
                normalized_features.get(
                    "crypto_health_score",
                    0
                )
            )
        )

        if crypto_health < 50:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="crypto_health_score",
                    impact=0.20,
                    direction="increases risk",
                    supports_class="phishing",
                    value=crypto_health
                )
            )

        # ==============================================================
        # RECENTLY ISSUED
        # ==============================================================

        if self._is_enabled(
            normalized_features,
            "is_recently_issued"
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="is_recently_issued",
                    impact=0.15,
                    direction="increases risk",
                    supports_class="phishing",
                    value=1.0
                )
            )

        # ==============================================================
        # SHORT LIVED
        # ==============================================================

        if self._is_enabled(
            normalized_features,
            "is_short_lived"
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="is_short_lived",
                    impact=0.10,
                    direction="increases risk",
                    supports_class="phishing",
                    value=1.0
                )
            )

        # ==============================================================
        # SECURE CONNECTION
        # ==============================================================

        if (
            self._is_enabled(
                normalized_features,
                "is_secure_connection"
            )
            and
            self._is_enabled(
                normalized_features,
                "is_trusted_ca"
            )
            and
            not self._is_enabled(
                normalized_features,
                "is_expired"
            )
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="is_secure_connection",
                    impact=-0.20,
                    direction="decreases risk",
                    supports_class="legitimate",
                    value=1.0
                )
            )

        # ==============================================================
        # STRONG CRYPTO HEALTH
        # ==============================================================

        if crypto_health >= 90:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="crypto_health_score",
                    impact=-0.20,
                    direction="decreases risk",
                    supports_class="legitimate",
                    value=crypto_health,
                    description=(
                        "Strong cryptographic configuration"
                    )
                )
            )

        # ==============================================================
        # TRUSTED CA
        # ==============================================================

        if self._is_enabled(
            normalized_features,
            "is_trusted_ca"
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="is_trusted_ca",
                    impact=-0.10,
                    direction="decreases risk",
                    supports_class="legitimate",
                    value=1.0
                )
            )

        # ==============================================================
        # SORT FACTORS
        # ==============================================================

        top_factors.sort(
            key=lambda item:
                item[
                    "absolute_impact"
                ],
            reverse=True
        )

        top_factors = (
            top_factors[
                :self.max_factors
            ]
        )

        # ==============================================================
        # BASELINE FACTOR
        # ==============================================================

        if not top_factors:

            top_factors.append({

                "feature":
                    "ssl_baseline",

                "description":
                    "General SSL/TLS transport-layer characteristics.",

                "category":
                    "ssl_tls",

                "impact":
                    0.0,

                "absolute_impact":
                    0.0,

                "direction":
                    "neutral",

                "supports_class":
                    prediction,

                "value":
                    0.0,

                "method":
                    "heuristic"
            })

        # ==============================================================
        # SUMMARY
        # ==============================================================

        positive_factors = [

            factor[
                "description"
            ]

            for factor
            in top_factors

            if factor[
                "impact"
            ] > 0
        ]

        negative_factors = [

            factor[
                "description"
            ]

            for factor
            in top_factors

            if factor[
                "impact"
            ] < 0
        ]

        if positive_factors:

            summary = (
                (
                    f"Heuristic SSL/TLS analysis supports a "
                    f"'{prediction}' verdict. "
                    f"Primary risk indicators include: "
                    f"{', '.join(positive_factors[:3])}."
                )
            )

        elif negative_factors:

            summary = (
                (
                    f"Heuristic SSL/TLS analysis supports a "
                    f"'{prediction}' verdict based primarily on "
                    f"protective indicators such as "
                    f"{', '.join(negative_factors[:3])}."
                )
            )

        else:

            summary = (
                (
                    f"Heuristic SSL/TLS analysis produced a "
                    f"'{prediction}' verdict without strong "
                    f"feature-specific evidence."
                )
            )

        return {

            "summary":
                summary,

            "top_factors":
                top_factors,

            "method":
                "heuristic",

            "shap_available":
                SHAP_AVAILABLE,

            "model_available":
                self.model is not None
        }

    # ======================================================================
    # MAIN EXPLANATION METHOD
    # ======================================================================

    def generate_explanation(
        self,
        features: Dict[str, Any],
        prediction: str
    ) -> Dict[str, Any]:
        """
        Generate a complete SSL/TLS explanation.

        SHAP is attempted first.

        If SHAP fails or is unavailable, deterministic heuristic
        explanation is returned.

        Args:
            features:
                Raw or normalized SSL feature dictionary.

            prediction:
                Model prediction, normally:

                    legitimate
                    phishing

        Returns:
            Structured explanation dictionary.
        """

        prediction = (
            self._normalize_prediction(
                prediction
            )
        )

        # ------------------------------------------------------------------
        # Attempt SHAP
        # ------------------------------------------------------------------

        if (
            self.shap_enabled
            and
            SHAP_AVAILABLE
            and
            self.explainer is not None
        ):

            try:

                # ==========================================================
                # ALIGN FEATURES
                # ==========================================================

                df_features = (
                    self._align_features(
                        features
                    )
                )

                # ==========================================================
                # EXTRACT SHAP FACTORS
                # ==========================================================

                shap_factors = (
                    self._extract_shap_factors(
                        df_features,
                        prediction
                    )
                )

                # ==========================================================
                # LIMIT TOP FACTORS
                # ==========================================================

                top_factors = (
                    shap_factors[
                        :self.max_factors
                    ]
                )

                # ==========================================================
                # GENERATE SUMMARY
                # ==========================================================

                summary = (
                    self._generate_shap_summary(
                        prediction,
                        top_factors
                    )
                )

                # ==========================================================
                # RETURN XAI RESULT
                # ==========================================================

                result = {

                    "summary":
                        summary,

                    "top_factors":
                        top_factors,

                    "all_factors":
                        shap_factors,

                    "method":
                        "SHAP",

                    "prediction":
                        prediction,

                    "feature_count":
                        len(
                            SSLFeatureSchema
                            .get_feature_names()
                        ),

                    "shap_available":
                        True,

                    "model_available":
                        self.model is not None,

                    "explainer_available":
                        self.explainer is not None,

                    "explainer_name":
                        self.EXPLAINER_NAME,

                    "explainer_version":
                        self.EXPLAINER_VERSION
                }

                self.last_error = None

                return result

            except Exception as e:

                self.last_error = str(
                    e
                )

                logger.error(
                    (
                        "SSL SHAP explanation failed: %s"
                    ),
                    str(e),
                    exc_info=True
                )

        # ------------------------------------------------------------------
        # Fallback
        # ------------------------------------------------------------------

        result = (
            self._heuristic_explanation(
                features,
                prediction
            )
        )

        result.update({

            "prediction":
                prediction,

            "feature_count":
                len(
                    SSLFeatureSchema
                    .get_feature_names()
                ),

            "explainer_name":
                self.EXPLAINER_NAME,

            "explainer_version":
                self.EXPLAINER_VERSION,

            "fallback_reason":
                self.last_error
                if self.last_error
                else (
                    "SHAP unavailable or not initialized."
                )
        })

        return result

    # ======================================================================
    # SHAP SUMMARY
    # ======================================================================

    def _generate_shap_summary(
        self,
        prediction: str,
        top_factors: List[
            Dict[str, Any]
        ]
    ) -> str:
        """
        Generate concise SOC-friendly summary from SHAP factors.
        """

        risk_factors = [

            factor[
                "description"
            ]

            for factor
            in top_factors

            if factor[
                "impact"
            ] > 0
        ]

        protective_factors = [

            factor[
                "description"
            ]

            for factor
            in top_factors

            if factor[
                "impact"
            ] < 0
        ]

        if risk_factors:

            return (
                (
                    f"The SSL/TLS model classified the connection "
                    f"as '{prediction}'. "
                    f"The strongest risk-driving features were: "
                    f"{', '.join(risk_factors[:3])}."
                )
            )

        if protective_factors:

            return (
                (
                    f"The SSL/TLS model classified the connection "
                    f"as '{prediction}'. "
                    f"The prediction was primarily supported by "
                    f"protective indicators: "
                    f"{', '.join(protective_factors[:3])}."
                )
            )

        return (
            (
                f"The SSL/TLS model produced a '{prediction}' "
                f"classification without strong individual "
                f"feature-level SHAP contributions."
            )
        )

    # ======================================================================
    # EXPLANATION FOR PREDICTOR RESULT
    # ======================================================================

    def explain_prediction(
        self,
        prediction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convenience method that consumes the dictionary returned by
        SSLPredictor.predict().

        Expected predictor payload:

            {
                "class_label": "phishing",
                "phishing_probability": 0.87,
                "features": {...}
            }

        This avoids making the caller manually extract fields.
        """

        if not isinstance(
            prediction_result,
            dict
        ):

            return self._heuristic_explanation(
                {},
                "phishing"
            )

        prediction = (
            prediction_result.get(
                "class_label",
                "phishing"
            )
        )

        features = (
            prediction_result.get(
                "features"
            )
        )

        # ------------------------------------------------------------------
        # Some predictors may return only a DataFrame.
        # ------------------------------------------------------------------

        if not isinstance(
            features,
            dict
        ):

            dataframe = (
                prediction_result.get(
                    "features_dataframe"
                )
            )

            if isinstance(
                dataframe,
                pd.DataFrame
            ) and not dataframe.empty:

                features = (
                    dataframe
                    .iloc[
                        0
                    ]
                    .to_dict()
                )

            else:

                features = {}

        return self.generate_explanation(
            features,
            prediction
        )

    # ======================================================================
    # TOP RISK FACTORS
    # ======================================================================

    def get_top_risk_factors(
        self,
        explanation: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Extract only positive/risk-increasing explanation factors.
        """

        if not isinstance(
            explanation,
            dict
        ):

            return []

        factors = (
            explanation.get(
                "top_factors",
                []
            )
        )

        if not isinstance(
            factors,
            list
        ):

            return []

        risk_factors = [

            factor

            for factor
            in factors

            if self._safe_numeric(
                factor.get(
                    "impact",
                    0
                )
            ) > 0
        ]

        maximum = (
            limit
            if limit is not None
            else self.max_factors
        )

        return risk_factors[
            :max(
                1,
                int(
                    maximum
                )
            )
        ]

    # ======================================================================
    # PROTECTIVE FACTORS
    # ======================================================================

    def get_protective_factors(
        self,
        explanation: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Extract features that decrease phishing risk.
        """

        if not isinstance(
            explanation,
            dict
        ):

            return []

        factors = (
            explanation.get(
                "top_factors",
                []
            )
        )

        if not isinstance(
            factors,
            list
        ):

            return []

        protective_factors = [

            factor

            for factor
            in factors

            if self._safe_numeric(
                factor.get(
                    "impact",
                    0
                )
            ) < 0
        ]

        maximum = (
            limit
            if limit is not None
            else self.max_factors
        )

        return protective_factors[
            :max(
                1,
                int(
                    maximum
                )
            )
        ]

    # ======================================================================
    # GENERATE SECURITY RECOMMENDATIONS
    # ======================================================================

    def generate_recommendations(
        self,
        features: Dict[str, Any]
    ) -> List[str]:
        """
        Generate practical SSL/TLS security recommendations based on
        detected feature conditions.

        This is intentionally rule-based and separate from SHAP.
        """

        normalized = (
            SSLFeatureSchema.validate_and_normalize(
                features
            )
        )

        recommendations: List[
            str
        ] = []

        # ==============================================================
        # SSL
        # ==============================================================

        if not self._is_enabled(
            normalized,
            "has_ssl"
        ):

            recommendations.append(
                (
                    "Enable HTTPS and deploy a valid SSL/TLS "
                    "certificate."
                )
            )

        # ==============================================================
        # EXPIRATION
        # ==============================================================

        if self._is_enabled(
            normalized,
            "is_expired"
        ):

            recommendations.append(
                (
                    "Renew the expired SSL/TLS certificate "
                    "immediately."
                )
            )

        # ==============================================================
        # SELF SIGNED
        # ==============================================================

        if self._is_enabled(
            normalized,
            "is_self_signed"
        ):

            recommendations.append(
                (
                    "Replace the self-signed certificate with a "
                    "certificate issued by a trusted CA."
                )
            )

        # ==============================================================
        # SUSPICIOUS CA
        # ==============================================================

        if self._is_enabled(
            normalized,
            "is_suspicious_ca"
        ):

            recommendations.append(
                (
                    "Investigate the certificate authority and "
                    "replace suspicious certificates."
                )
            )

        # ==============================================================
        # WEAK TLS
        # ==============================================================

        if self._is_enabled(
            normalized,
            "is_weak_protocol"
        ):

            recommendations.append(
                (
                    "Disable obsolete TLS protocols and enforce "
                    "modern TLS versions."
                )
            )

        # ==============================================================
        # VULNERABLE CIPHER
        # ==============================================================

        if self._is_enabled(
            normalized,
            "is_vulnerable_cipher"
        ):

            recommendations.append(
                (
                    "Disable vulnerable cipher suites and use "
                    "modern authenticated encryption."
                )
            )

        # ==============================================================
        # CRYPTO HEALTH
        # ==============================================================

        crypto_health = (
            self._safe_numeric(
                normalized.get(
                    "crypto_health_score",
                    0
                )
            )
        )

        if crypto_health < 50:

            recommendations.append(
                (
                    "Review the complete TLS cryptographic "
                    "configuration and improve its security posture."
                )
            )

        # ==============================================================
        # TRUST
        # ==============================================================

        issuer_trust_score = (
            self._safe_numeric(
                normalized.get(
                    "issuer_trust_score",
                    0
                )
            )
        )

        if issuer_trust_score < 30:

            recommendations.append(
                (
                    "Verify the certificate issuer and use a "
                    "well-established trusted CA."
                )
            )

        # ==============================================================
        # EXPIRATION WINDOW
        # ==============================================================

        days_until_expiry = (
            self._safe_numeric(
                normalized.get(
                    "days_until_expiry",
                    0
                )
            )
        )

        if (
            days_until_expiry > 0
            and
            days_until_expiry <= 30
        ):

            recommendations.append(
                (
                    "Schedule certificate renewal because the "
                    "certificate is approaching expiration."
                )
            )

        # ==============================================================
        # SECURE CONNECTION
        # ==============================================================

        if not self._is_enabled(
            normalized,
            "is_secure_connection"
        ):

            recommendations.append(
                (
                    "Review the connection configuration and "
                    "ensure HTTPS is enforced."
                )
            )

        # ==============================================================
        # DEFAULT
        # ==============================================================

        if not recommendations:

            recommendations.append(
                (
                    "No major SSL/TLS remediation actions were "
                    "identified from the available features."
                )
            )

        return recommendations

    # ======================================================================
    # FULL XAI REPORT
    # ======================================================================

    def generate_full_report(
        self,
        features: Dict[str, Any],
        prediction: str,
        risk_assessment: Optional[
            Dict[str, Any]
        ] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete SSL XAI report.

        Combines:

            • prediction
            • SHAP/heuristic explanation
            • risk assessment
            • recommendations

        This method is particularly useful for the future PDF report
        generation layer.
        """

        explanation = (
            self.generate_explanation(
                features,
                prediction
            )
        )

        recommendations = (
            self.generate_recommendations(
                features
            )
        )

        report = {

            "prediction":
                self._normalize_prediction(
                    prediction
                ),

            "explanation":
                explanation,

            "recommendations":
                recommendations,

            "risk_assessment":
                risk_assessment,

            "feature_count":
                len(
                    SSLFeatureSchema
                    .get_feature_names()
                ),

            "schema_version":
                SSLFeatureSchema.SCHEMA_VERSION,

            "explainer_name":
                self.EXPLAINER_NAME,

            "explainer_version":
                self.EXPLAINER_VERSION
        }

        return report

    # ======================================================================
    # STATUS
    # ======================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return the current XAI engine status.
        """

        return {

            "explainer_name":
                self.EXPLAINER_NAME,

            "explainer_version":
                self.EXPLAINER_VERSION,

            "shap_installed":
                SHAP_AVAILABLE,

            "shap_enabled":
                self.shap_enabled,

            "shap_explainer_ready":
                self.explainer is not None,

            "model_available":
                self.model is not None,

            "fallback_available":
                True,

            "max_factors":
                self.max_factors,

            "feature_count":
                len(
                    SSLFeatureSchema
                    .get_feature_names()
                ),

            "schema_version":
                SSLFeatureSchema.SCHEMA_VERSION,

            "last_error":
                self.last_error
        }

    # ======================================================================
    # BOOLEAN HELPERS
    # ======================================================================

    @classmethod
    def _is_enabled(
        cls,
        features: Dict[str, Any],
        feature_name: str
    ) -> bool:
        """
        Safely evaluate a boolean/numeric feature.
        """

        value = features.get(
            feature_name,
            0
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

            return (
                value
                .strip()
                .lower()
                in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                    "enabled"
                }
            )

        return (
            cls._safe_numeric(
                value
            )
            >
            0
        )
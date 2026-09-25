"""
HTML AI Agent - Explainable AI Engine
=====================================

Provides human-readable explanations for HTML AI Agent predictions.

Primary method:
    SHAP TreeExplainer

Fallback:
    Deterministic HTML structural explanation

Classification:
    0 = legitimate
    1 = phishing

IMPORTANT
---------
This module explains an existing prediction.

It does NOT create the final phishing verdict.
"""


import logging

from typing import (
    Dict,
    Any,
    List,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd

from .feature_schema import HTMLFeatureSchema


logger = logging.getLogger(__name__)


# =========================================================================
# OPTIONAL SHAP
# =========================================================================

try:
    import shap

    SHAP_AVAILABLE = True

except ImportError:
    shap = None
    SHAP_AVAILABLE = False


class HTMLExplainer:
    """
    Explainable AI engine for the HTML AI Agent.

    Uses the centralized 50-feature HTML schema.
    """

    # =====================================================================
    # CLASS MAPPING
    # =====================================================================

    CLASS_INDEX_MAP = {
        "legitimate": 0,
        "phishing": 1,
    }

    INDEX_CLASS_MAP = {
        0: "legitimate",
        1: "phishing",
    }

    # =====================================================================
    # CONFIGURATION
    # =====================================================================

    DEFAULT_TOP_FACTORS = 5
    MIN_SHAP_IMPACT = 0.0001

    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    def __init__(
        self,
        model: Any = None,
        max_factors: int = DEFAULT_TOP_FACTORS,
    ):

        self.model = model

        try:
            max_factors = int(max_factors)

        except (
            TypeError,
            ValueError,
        ):

            max_factors = self.DEFAULT_TOP_FACTORS

        self.max_factors = max(
            1,
            max_factors,
        )

        self.explainer = None

        self._initialize_explainer()

    # =====================================================================
    # SHAP INITIALIZATION
    # =====================================================================

    def _initialize_explainer(self) -> None:

        if not SHAP_AVAILABLE:

            logger.warning(
                "SHAP is not installed. "
                "Using heuristic explanation mode."
            )

            return

        if self.model is None:

            logger.warning(
                "No trained HTML model supplied. "
                "Using heuristic explanation mode."
            )

            return

        try:

            self.explainer = shap.TreeExplainer(
                self.model
            )

            logger.info(
                "HTML SHAP TreeExplainer initialized."
            )

        except Exception as exc:

            self.explainer = None

            logger.warning(
                "Failed to initialize SHAP TreeExplainer: %s",
                str(exc),
            )

    # =====================================================================
    # PUBLIC EXPLANATION
    # =====================================================================

    def generate_explanation(
        self,
        features: Dict[str, Any],
        prediction: Any,
    ) -> Dict[str, Any]:

        if not isinstance(features, dict):

            logger.warning(
                "HTMLExplainer received invalid feature input."
            )

            features = {}

        normalized_prediction = (
            self._normalize_prediction(
                prediction
            )
        )

        # ---------------------------------------------------------------
        # SHAP
        # ---------------------------------------------------------------

        if (
            self.explainer is not None
            and SHAP_AVAILABLE
        ):

            try:

                return self._generate_shap_explanation(
                    features,
                    normalized_prediction,
                )

            except Exception as exc:

                logger.error(
                    "SHAP explanation failed: %s. "
                    "Using heuristic explanation.",
                    str(exc),
                    exc_info=True,
                )

        # ---------------------------------------------------------------
        # Heuristic fallback
        # ---------------------------------------------------------------

        return self._heuristic_explanation(
            features,
            normalized_prediction,
        )

    # =====================================================================
    # SHAP EXPLANATION
    # =====================================================================

    def _generate_shap_explanation(
        self,
        features: Dict[str, Any],
        prediction: str,
    ) -> Dict[str, Any]:

        # ---------------------------------------------------------------
        # Align with centralized 50-feature schema
        # ---------------------------------------------------------------

        df_features = HTMLFeatureSchema.align_and_validate(
            features
        )

        feature_order = HTMLFeatureSchema.get_schema()

        descriptions = HTMLFeatureSchema.get_descriptions()

        # ---------------------------------------------------------------
        # Target class
        # ---------------------------------------------------------------

        target_class_idx = self.CLASS_INDEX_MAP[
            prediction
        ]

        # ---------------------------------------------------------------
        # SHAP values
        # ---------------------------------------------------------------

        shap_output = self.explainer.shap_values(
            df_features
        )

        instance_shap = (
            self._extract_instance_shap_values(
                shap_output,
                target_class_idx,
            )
        )

        # ---------------------------------------------------------------
        # Ensure SHAP vector corresponds to 50 features
        # ---------------------------------------------------------------

        if len(instance_shap) != len(feature_order):

            raise RuntimeError(
                "SHAP feature count mismatch. "
                f"Expected {len(feature_order)}, "
                f"found {len(instance_shap)}."
            )

        # ---------------------------------------------------------------
        # Build impacts
        # ---------------------------------------------------------------

        feature_impacts: List[
            Tuple[str, float]
        ] = []

        for index, feature_name in enumerate(
            feature_order
        ):

            impact = self._safe_float(
                instance_shap[index]
            )

            if abs(impact) <= self.MIN_SHAP_IMPACT:
                continue

            feature_impacts.append(
                (
                    feature_name,
                    impact,
                )
            )

        feature_impacts.sort(
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        # ---------------------------------------------------------------
        # Top factors
        # ---------------------------------------------------------------

        top_factors = []

        for feature_name, impact in feature_impacts[
            :self.max_factors
        ]:

            feature_value = self._safe_float(
                df_features.iloc[0].get(
                    feature_name,
                    0.0,
                )
            )

            if impact > 0:

                direction = "increases risk"
                supports_class = "phishing"

            else:

                direction = "decreases risk"
                supports_class = "legitimate"

            description = descriptions.get(
                feature_name,
                self._humanize_feature_name(
                    feature_name
                ),
            )

            top_factors.append(
                {
                    "feature": feature_name,
                    "value": round(
                        feature_value,
                        4,
                    ),
                    "impact": round(
                        impact,
                        4,
                    ),
                    "direction": direction,
                    "supports_class": supports_class,
                    "description": description,
                }
            )

        summary = self._build_shap_summary(
            prediction,
            top_factors,
        )

        return {
            "summary": summary,
            "top_factors": top_factors,
            "explanation_method": "shap",
            "predicted_class": prediction,
            "feature_count": len(feature_order),
            "explained_feature_count": len(
                feature_impacts
            ),
        }

    # =====================================================================
    # SHAP OUTPUT NORMALIZATION
    # =====================================================================

    def _extract_instance_shap_values(
        self,
        shap_output: Any,
        target_class_idx: int,
    ) -> np.ndarray:

        if hasattr(
            shap_output,
            "values",
        ):

            shap_output = shap_output.values

        # ---------------------------------------------------------------
        # List
        # ---------------------------------------------------------------

        if isinstance(
            shap_output,
            list,
        ):

            if not shap_output:
                return np.array([], dtype=float)

            if target_class_idx < len(shap_output):
                values = shap_output[
                    target_class_idx
                ]
            else:
                values = shap_output[0]

            values = np.asarray(
                values,
                dtype=float,
            )

            if values.ndim == 2:
                return values[0]

            return values.reshape(-1)

        # ---------------------------------------------------------------
        # ndarray
        # ---------------------------------------------------------------

        values = np.asarray(
            shap_output,
            dtype=float,
        )

        if values.size == 0:
            return np.array([], dtype=float)

        # ---------------------------------------------------------------
        # 1D
        # ---------------------------------------------------------------

        if values.ndim == 1:
            return values

        # ---------------------------------------------------------------
        # 2D
        # ---------------------------------------------------------------

        if values.ndim == 2:
            return values[0]

        # ---------------------------------------------------------------
        # 3D
        # ---------------------------------------------------------------

        if values.ndim == 3:

            first_sample = values[0]

            if (
                target_class_idx
                <
                first_sample.shape[-1]
            ):

                return first_sample[
                    :,
                    target_class_idx,
                ]

            return first_sample[:, 0]

        logger.warning(
            "Unexpected SHAP output shape: %s",
            values.shape,
        )

        return values.reshape(-1)

    # =====================================================================
    # SHAP SUMMARY
    # =====================================================================

    def _build_shap_summary(
        self,
        prediction: str,
        top_factors: List[Dict[str, Any]],
    ) -> str:

        if not top_factors:

            if prediction == "phishing":

                return (
                    "The HTML model classified the page "
                    "as phishing, but no significant SHAP "
                    "feature contributions were identified."
                )

            return (
                "The HTML model classified the page as "
                "legitimate, with no significant SHAP "
                "feature contributions identified."
            )

        phishing_factors = [
            factor["description"]
            for factor in top_factors
            if factor.get("supports_class")
            == "phishing"
        ]

        legitimate_factors = [
            factor["description"]
            for factor in top_factors
            if factor.get("supports_class")
            == "legitimate"
        ]

        if prediction == "phishing":

            if phishing_factors:

                return (
                    "HTML structural analysis classified "
                    "the page as phishing. The strongest "
                    "risk-increasing factors were: "
                    + ", ".join(
                        phishing_factors[:2]
                    )
                    + "."
                )

            return (
                "The HTML model classified the page as "
                "phishing, although the strongest displayed "
                "SHAP factors were not directly risk-increasing."
            )

        if legitimate_factors:

            return (
                "HTML structural analysis classified "
                "the page as legitimate. The strongest "
                "factors reducing phishing risk were: "
                + ", ".join(
                    legitimate_factors[:2]
                )
                + "."
            )

        return (
            "HTML structural analysis classified the "
            "page as legitimate."
        )

    # =====================================================================
    # HEURISTIC EXPLANATION
    # =====================================================================

    def _heuristic_explanation(
        self,
        features: Dict[str, Any],
        prediction: str,
    ) -> Dict[str, Any]:

        descriptions = (
            HTMLFeatureSchema.get_descriptions()
        )

        top_factors = []

        # -----------------------------------------------------------------
        # Cross-domain forms
        # -----------------------------------------------------------------

        cross_domain_forms = self._safe_numeric(
            features.get(
                "cross_domain_form_actions",
                0,
            )
        )

        if cross_domain_forms > 0:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="cross_domain_form_actions",
                    value=cross_domain_forms,
                    impact=min(
                        0.50,
                        cross_domain_forms * 0.15,
                    ),
                    description=descriptions.get(
                        "cross_domain_form_actions",
                        "Forms submit data to another domain.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Credential hidden input
        # -----------------------------------------------------------------

        credential_hidden = self._safe_numeric(
            features.get(
                "credential_hidden_input_count",
                0,
            )
        )

        if credential_hidden > 0:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="credential_hidden_input_count",
                    value=credential_hidden,
                    impact=min(
                        0.45,
                        credential_hidden * 0.15,
                    ),
                    description=descriptions.get(
                        "credential_hidden_input_count",
                        "Credential-related hidden inputs detected.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Suspicious hidden input
        # -----------------------------------------------------------------

        suspicious_hidden = self._safe_numeric(
            features.get(
                "suspicious_hidden_input_count",
                0,
            )
        )

        if suspicious_hidden > 0:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="suspicious_hidden_input_count",
                    value=suspicious_hidden,
                    impact=min(
                        0.40,
                        suspicious_hidden * 0.15,
                    ),
                    description=descriptions.get(
                        "suspicious_hidden_input_count",
                        "Suspicious hidden input patterns detected.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Invisible iframe
        # -----------------------------------------------------------------

        invisible_iframes = self._safe_numeric(
            features.get(
                "invisible_iframes",
                0,
            )
        )

        if invisible_iframes > 0:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="invisible_iframes",
                    value=invisible_iframes,
                    impact=min(
                        0.40,
                        invisible_iframes * 0.15,
                    ),
                    description=descriptions.get(
                        "invisible_iframes",
                        "Invisible iframe elements detected.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Suspicious JavaScript
        # -----------------------------------------------------------------

        suspicious_scripts = self._safe_numeric(
            features.get(
                "suspicious_inline_scripts",
                0,
            )
        )

        if suspicious_scripts > 0:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="suspicious_inline_scripts",
                    value=suspicious_scripts,
                    impact=min(
                        0.40,
                        suspicious_scripts * 0.20,
                    ),
                    description=descriptions.get(
                        "suspicious_inline_scripts",
                        "Suspicious inline JavaScript detected.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # JavaScript redirect
        # -----------------------------------------------------------------

        javascript_redirects = self._safe_numeric(
            features.get(
                "javascript_redirect_count",
                0,
            )
        )

        if javascript_redirects > 0:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="javascript_redirect_count",
                    value=javascript_redirects,
                    impact=min(
                        0.40,
                        javascript_redirects * 0.15,
                    ),
                    description=descriptions.get(
                        "javascript_redirect_count",
                        "JavaScript redirect behavior detected.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # HTTP forms
        # -----------------------------------------------------------------

        http_forms = self._safe_numeric(
            features.get(
                "http_form_actions",
                0,
            )
        )

        if http_forms > 0:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="http_form_actions",
                    value=http_forms,
                    impact=min(
                        0.30,
                        http_forms * 0.10,
                    ),
                    description=descriptions.get(
                        "http_form_actions",
                        "Forms submit data using HTTP.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Meta refresh
        # -----------------------------------------------------------------

        if self._safe_boolean(
            features.get(
                "has_meta_refresh",
                False,
            )
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="has_meta_refresh",
                    value=1.0,
                    impact=0.20,
                    description=descriptions.get(
                        "has_meta_refresh",
                        "Automatic meta refresh detected.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Hidden elements
        # -----------------------------------------------------------------

        hidden_elements = self._safe_numeric(
            features.get(
                "hidden_elements_count",
                0,
            )
        )

        if hidden_elements > 5:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="hidden_elements_count",
                    value=hidden_elements,
                    impact=min(
                        0.25,
                        hidden_elements * 0.02,
                    ),
                    description=descriptions.get(
                        "hidden_elements_count",
                        "Multiple hidden HTML elements detected.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Urgency keywords
        # -----------------------------------------------------------------

        urgency_keywords = self._safe_numeric(
            features.get(
                "urgency_keyword_count",
                0,
            )
        )

        if urgency_keywords > 0:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="urgency_keyword_count",
                    value=urgency_keywords,
                    impact=min(
                        0.25,
                        urgency_keywords * 0.05,
                    ),
                    description=descriptions.get(
                        "urgency_keyword_count",
                        "Urgency-related language detected.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Credential keywords
        # -----------------------------------------------------------------

        credential_keywords = self._safe_numeric(
            features.get(
                "credential_keyword_count",
                0,
            )
        )

        if credential_keywords >= 2:

            top_factors.append(
                self._create_heuristic_factor(
                    feature="credential_keyword_count",
                    value=credential_keywords,
                    impact=min(
                        0.25,
                        credential_keywords * 0.05,
                    ),
                    description=descriptions.get(
                        "credential_keyword_count",
                        "Credential-related keywords detected.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Right-click disabled
        # -----------------------------------------------------------------

        if self._safe_boolean(
            features.get(
                "right_click_disabled",
                False,
            )
        ):

            top_factors.append(
                self._create_heuristic_factor(
                    feature="right_click_disabled",
                    value=1.0,
                    impact=0.05,
                    description=descriptions.get(
                        "right_click_disabled",
                        "Right-click inspection is disabled.",
                    ),
                    supports_class="phishing",
                )
            )

        # -----------------------------------------------------------------
        # Sort
        # -----------------------------------------------------------------

        top_factors.sort(
            key=lambda factor: abs(
                self._safe_numeric(
                    factor.get(
                        "impact",
                        0,
                    )
                )
            ),
            reverse=True,
        )

        top_factors = top_factors[
            :self.max_factors
        ]

        summary = self._build_heuristic_summary(
            prediction,
            top_factors,
        )

        return {
            "summary": summary,
            "top_factors": top_factors,
            "explanation_method": "heuristic",
            "predicted_class": prediction,
            "feature_count": len(
                HTMLFeatureSchema.get_schema()
            ),
            "explained_feature_count": len(
                top_factors
            ),
        }

    # =====================================================================
    # HEURISTIC FACTOR
    # =====================================================================

    @staticmethod
    def _create_heuristic_factor(
        feature: str,
        value: float,
        impact: float,
        description: str,
        supports_class: str,
    ) -> Dict[str, Any]:

        impact = max(
            0.0,
            min(
                1.0,
                float(impact),
            ),
        )

        return {
            "feature": feature,
            "value": round(
                float(value),
                4,
            ),
            "impact": round(
                impact,
                4,
            ),
            "direction": (
                "increases risk"
                if supports_class == "phishing"
                else "decreases risk"
            ),
            "supports_class": supports_class,
            "description": description,
        }

    # =====================================================================
    # HEURISTIC SUMMARY
    # =====================================================================

    def _build_heuristic_summary(
        self,
        prediction: str,
        factors: List[Dict[str, Any]],
    ) -> str:

        risk_factors = [
            factor.get(
                "description",
                "",
            )
            for factor in factors
            if factor.get(
                "supports_class"
            ) == "phishing"
        ]

        if risk_factors:

            return (
                "HTML structural analysis supports a "
                f"'{prediction}' assessment. The strongest "
                "detected structural indicators were: "
                + ", ".join(
                    risk_factors[:2]
                )
                + "."
            )

        if prediction == "phishing":

            return (
                "The HTML model classified the page as "
                "phishing, but the heuristic explanation "
                "did not identify strong individual HTML "
                "indicators."
            )

        return (
            "HTML structural analysis found no strong "
            "phishing-oriented indicators among the "
            "available HTML features."
        )

    # =====================================================================
    # PREDICTION NORMALIZATION
    # =====================================================================

    def _normalize_prediction(
        self,
        prediction: Any,
    ) -> str:
        """
        Normalize prediction safely.

        IMPORTANT:
            Unknown values are NOT silently converted to legitimate.

            An invalid prediction causes an exception so the caller
            cannot accidentally treat an unknown/error state as safe.
        """

        if isinstance(
            prediction,
            str,
        ):

            normalized = (
                prediction
                .strip()
                .lower()
            )

            if normalized in {
                "legitimate",
                "phishing",
            }:

                return normalized

            if normalized == "0":
                return "legitimate"

            if normalized == "1":
                return "phishing"

        if isinstance(
            prediction,
            (int, np.integer),
        ):

            if int(prediction) == 0:
                return "legitimate"

            if int(prediction) == 1:
                return "phishing"

        if isinstance(
            prediction,
            (float, np.floating),
        ):

            if not np.isfinite(
                prediction
            ):

                raise ValueError(
                    "HTML prediction is not finite."
                )

            if float(prediction) == 0.0:
                return "legitimate"

            if float(prediction) == 1.0:
                return "phishing"

        raise ValueError(
            f"Invalid HTML prediction: {prediction!r}. "
            "Expected legitimate, phishing, 0, or 1."
        )

    # =====================================================================
    # SAFE FLOAT
    # =====================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            value = float(value)

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):

            return default

        if not np.isfinite(value):
            return default

        return value

    # =====================================================================
    # SAFE NUMERIC
    # =====================================================================

    @staticmethod
    def _safe_numeric(
        value: Any,
        default: float = 0.0,
    ) -> float:

        return HTMLExplainer._safe_float(
            value,
            default,
        )

    # =====================================================================
    # SAFE BOOLEAN
    # =====================================================================

    @staticmethod
    def _safe_boolean(
        value: Any,
    ) -> bool:

        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            (int, float),
        ):
            return value != 0

        if isinstance(
            value,
            str,
        ):

            return (
                value
                .strip()
                .lower()
                in {
                    "true",
                    "1",
                    "yes",
                    "y",
                    "on",
                }
            )

        return False

    # =====================================================================
    # HUMANIZE FEATURE
    # =====================================================================

    @staticmethod
    def _humanize_feature_name(
        feature_name: str,
    ) -> str:

        return (
            str(feature_name)
            .replace(
                "_",
                " ",
            )
            .strip()
            .title()
        )

    # =====================================================================
    # STATUS
    # =====================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:

        if (
            self.explainer is not None
            and SHAP_AVAILABLE
        ):

            method = "shap"

        else:

            method = "heuristic"

        return {
            "shap_available": SHAP_AVAILABLE,
            "model_available": self.model is not None,
            "shap_explainer_initialized": (
                self.explainer is not None
            ),
            "active_explanation_method": method,
            "max_factors": self.max_factors,
            "feature_count": len(
                HTMLFeatureSchema.get_schema()
            ),
            "feature_schema": "50 HTML features",
        }
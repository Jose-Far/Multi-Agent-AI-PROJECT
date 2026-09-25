import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .feature_schema import URLFeatureSchema


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Optional SHAP dependency
# ---------------------------------------------------------------------

try:
    import shap

    SHAP_AVAILABLE = True

except ImportError:
    shap = None
    SHAP_AVAILABLE = False


class URLExplainer:
    """
    Production-grade Explainable AI engine for the URL AI Agent.

    This component is responsible for converting the prediction produced by
    the URL XGBoost model into human-readable feature-level explanations.

    IMPORTANT DESIGN PRINCIPLE
    ---------------------------

    The explainer MUST use the same feature schema as:

        URLFeatureSchema
              ↓
        URLPreprocessor
              ↓
        URLModel
              ↓
        URLPredictor
              ↓
        URLExplainer

    Therefore this class does NOT maintain an independent hard-coded
    feature list.

    This prevents feature drift between:

        training
        inference
        SHAP
        risk scoring
        reporting

    Updated URL entropy interpretation:

        domain_entropy
            → strong contextual security signal

        path_entropy
            → weaker contextual signal

        query_entropy
            → contextual signal

        url_entropy
            → legacy/global entropy feature, if still present

    A high path entropy alone should NOT automatically be described as
    strong evidence of phishing because legitimate services such as
    Google Forms, Microsoft services, cloud applications, CDNs and
    dynamically generated links can naturally contain long random IDs.
    """

    # =================================================================
    # HUMAN-READABLE FEATURE DESCRIPTIONS
    # =================================================================

    FEATURE_DESCRIPTIONS: Dict[str, str] = {

        # -------------------------------------------------------------
        # URL size
        # -------------------------------------------------------------

        "url_length":
            "Overall URL Character Length",

        "domain_length":
            "Domain Name Length",

        "path_length":
            "URL Path Length",

        "query_length":
            "URL Query Length",

        # -------------------------------------------------------------
        # URL structural features
        # -------------------------------------------------------------

        "num_dots":
            "Number of Domain/Subdomain Separators",

        "num_hyphens":
            "Number of Hyphens in URL",

        "num_underscores":
            "Number of Underscores in URL",

        "num_slashes":
            "Number of Path Separators",

        "num_question_marks":
            "Number of Query Delimiters",

        "num_equal_signs":
            "Number of Query Assignment Operators",

        "num_at_symbols":
            "Number of @ Symbols",

        "num_percent_signs":
            "Number of URL Encoding Characters",

        "num_digits":
            "Number of Numeric Characters",

        "num_special_chars":
            "Number of Special Characters",

        # -------------------------------------------------------------
        # Entropy
        # -------------------------------------------------------------

        "url_entropy":
            "Overall URL String Randomness",

        "domain_entropy":
            "Domain Name Randomness",

        "path_entropy":
            "URL Path Randomness",

        "query_entropy":
            "URL Query Randomness",

        # -------------------------------------------------------------
        # Security indicators
        # -------------------------------------------------------------

        "contains_ip":
            "IP Address Used Instead of Domain",

        "is_https":
            "HTTPS Protocol Status",

        "has_punycode":
            "Internationalized/Punycode Domain Indicator",

        "subdomain_count":
            "Number of Subdomains",

        "tld_length":
            "Top-Level Domain Length",

        "suspicious_word_count":
            "Number of Suspicious/Social Engineering Keywords",

        # -------------------------------------------------------------
        # Additional possible features
        # -------------------------------------------------------------

        "is_shortened_url":
            "URL Shortening Service Indicator",

        "has_port":
            "Non-Standard Port Indicator",

        "domain_has_digits":
            "Numeric Characters in Domain",

        "path_has_digits":
            "Numeric Characters in Path",

        "query_parameter_count":
            "Number of Query Parameters",

        "is_trusted_domain":
            "Trusted Domain Indicator",

        "is_known_brand":
            "Known Brand Domain Indicator"
    }

    # =================================================================
    # CLASS MAPPING
    # =================================================================

    CLASS_INDEX_MAP: Dict[str, int] = {

        "legitimate": 0,

        "phishing": 1,

        "suspicious": 1,

        "malicious": 1,

        "unknown": 1
    }

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __init__(
        self,
        model: Any = None
    ):
        """
        Initialize the URL explainer.

        Args:
            model:
                Trained XGBoost model.

        The model may be None when:

            - model artifact is unavailable
            - training has not happened yet
            - fallback inference is being used

        In such cases the explainer automatically uses heuristic
        explanations.
        """

        self.model = model

        self.explainer = None

        # Store schema version when available.
        self.schema_version = getattr(
            URLFeatureSchema,
            "SCHEMA_VERSION",
            "unknown"
        )

        self.feature_names = self._get_feature_names()

        self._initialize_explainer()

    # =================================================================
    # FEATURE SCHEMA
    # =================================================================

    def _get_feature_names(self) -> List[str]:
        """
        Retrieve the authoritative feature order from URLFeatureSchema.

        This is extremely important.

        The old implementation contained a manually maintained list.
        That could become different from the actual XGBoost feature order.

        Now:

            URLFeatureSchema
                    ↓
              feature_names
                    ↓
                SHAP

        Returns:
            Ordered feature list.
        """

        try:

            names = URLFeatureSchema.get_feature_names()

            if not names:
                raise ValueError(
                    "URLFeatureSchema returned an empty feature list."
                )

            return list(names)

        except Exception as e:

            logger.error(
                "Unable to retrieve URL feature schema: %s",
                str(e),
                exc_info=True
            )

            # This should only be a last-resort compatibility fallback.
            return [
                "url_length",
                "domain_length",
                "num_dots",
                "num_hyphens",
                "num_underscores",
                "num_slashes",
                "num_question_marks",
                "num_equal_signs",
                "num_at_symbols",
                "num_percent_signs",
                "num_digits",
                "num_special_chars",
                "url_entropy",
                "domain_entropy",
                "path_entropy",
                "query_entropy",
                "contains_ip",
                "is_https",
                "has_punycode",
                "subdomain_count",
                "tld_length",
                "suspicious_word_count"
            ]

    # =================================================================
    # SHAP INITIALIZATION
    # =================================================================

    def _initialize_explainer(self) -> None:
        """
        Initialize SHAP TreeExplainer when possible.

        If SHAP cannot initialize, the system does NOT crash.

        Instead:

            SHAP unavailable
                    ↓
            heuristic explanation
        """

        if not SHAP_AVAILABLE:

            logger.warning(
                "URLExplainer: SHAP is not installed. "
                "Using heuristic explanation mode."
            )

            return

        if self.model is None:

            logger.warning(
                "URLExplainer: No trained model supplied. "
                "Using heuristic explanation mode."
            )

            return

        try:

            self.explainer = shap.TreeExplainer(
                self.model
            )

            logger.info(
                "URLExplainer: SHAP TreeExplainer initialized."
            )

        except Exception as e:

            logger.warning(
                "URLExplainer: SHAP initialization failed: %s",
                str(e)
            )

            self.explainer = None

    # =================================================================
    # FEATURE ALIGNMENT
    # =================================================================

    def _align_features(
        self,
        raw_features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Convert incoming feature dictionary into the exact feature
        matrix expected by the model.

        The order comes from URLFeatureSchema.

        Missing features receive their schema defaults.

        Extra features are ignored.

        Returns:
            One-row pandas DataFrame.
        """

        if not isinstance(
            raw_features,
            dict
        ):

            raw_features = {}

        try:

            # ---------------------------------------------------------
            # Use authoritative schema normalization when available.
            # ---------------------------------------------------------

            normalized_dataframe = URLFeatureSchema.align_and_validate(
                raw_features
            )

            if normalized_dataframe.empty:
                raise ValueError(
                    "URL schema alignment returned an empty DataFrame."
                )

            normalized = normalized_dataframe.iloc[0].to_dict()

        except Exception as e:

            logger.warning(
                "URL schema normalization failed: %s. "
                "Using direct alignment.",
                str(e)
            )

            normalized = raw_features

        row_vector: Dict[str, float] = {}

        for feature_name in self.feature_names:

            value = normalized.get(
                feature_name,
                0.0
            )

            try:

                if value is None:

                    numeric_value = 0.0

                elif isinstance(
                    value,
                    bool
                ):

                    numeric_value = (
                        1.0
                        if value
                        else 0.0
                    )

                else:

                    numeric_value = float(
                        value
                    )

                if not np.isfinite(
                    numeric_value
                ):

                    numeric_value = 0.0

            except (
                ValueError,
                TypeError
            ):

                numeric_value = 0.0

            row_vector[
                feature_name
            ] = numeric_value

        return pd.DataFrame(
            [row_vector],
            columns=self.feature_names
        )

    # =================================================================
    # HUMAN DESCRIPTION
    # =================================================================

    def _get_human_description(
        self,
        feature_name: str
    ) -> str:
        """
        Convert internal feature name into an analyst-friendly
        description.
        """

        return self.FEATURE_DESCRIPTIONS.get(
            feature_name,
            feature_name.replace(
                "_",
                " "
            ).title()
        )

    # =================================================================
    # FEATURE IMPORTANCE / CONTEXT
    # =================================================================

    def _is_entropy_feature(
        self,
        feature_name: str
    ) -> bool:
        """
        Determine whether a feature represents entropy.

        Used to prevent the explanation layer from overstating
        path/query randomness.
        """

        return feature_name in {
            "url_entropy",
            "domain_entropy",
            "path_entropy",
            "query_entropy"
        }

    def _entropy_explanation(
        self,
        feature_name: str
    ) -> str:
        """
        Generate context-aware entropy explanations.

        This is specifically designed to reduce misleading explanations
        for legitimate dynamic URLs.
        """

        if feature_name == "domain_entropy":

            return (
                "The domain contains relatively high randomness. "
                "Domain-level randomness is more security-relevant "
                "because phishing domains often use generated or "
                "obfuscated domain names."
            )

        if feature_name == "path_entropy":

            return (
                "The URL path contains high randomness. "
                "Path randomness is treated as contextual evidence "
                "because legitimate services may generate long "
                "random identifiers."
            )

        if feature_name == "query_entropy":

            return (
                "The query component contains random-looking data. "
                "This is contextual evidence and should be evaluated "
                "with other URL indicators."
            )

        if feature_name == "url_entropy":

            return (
                "The overall URL contains random-looking characters. "
                "Global URL entropy alone is not sufficient to establish "
                "phishing because legitimate dynamic services can also "
                "produce high-entropy URLs."
            )

        return self._get_human_description(
            feature_name
        )

    # =================================================================
    # HEURISTIC EXPLANATION
    # =================================================================

    def _heuristic_explanation(
        self,
        features: Dict[str, Any],
        prediction: str
    ) -> Dict[str, Any]:
        """
        Generate a deterministic explanation when SHAP is unavailable.

        IMPORTANT:

        Entropy is interpreted differently depending on where it occurs.

            domain entropy
                stronger signal

            path entropy
                contextual signal

            query entropy
                contextual signal

        This prevents legitimate dynamic URLs from being automatically
        described as malicious.
        """

        top_factors: List[
            Dict[str, Any]
        ] = []

        prediction = str(
            prediction or "unknown"
        ).lower()

        # -------------------------------------------------------------
        # IP address
        # -------------------------------------------------------------

        try:

            contains_ip = float(
                features.get(
                    "contains_ip",
                    0
                )
            )

        except Exception:

            contains_ip = 0.0

        if contains_ip == 1:

            top_factors.append({

                "feature":
                    "contains_ip",

                "impact":
                    0.40,

                "direction":
                    "increases risk",

                "supports_class":
                    "phishing",

                "description":
                    (
                        "The URL uses an IP address instead of "
                        "a conventional domain name."
                    )
            })

        # -------------------------------------------------------------
        # Domain entropy
        # -------------------------------------------------------------

        try:

            domain_entropy = float(
                features.get(
                    "domain_entropy",
                    0.0
                )
            )

        except Exception:

            domain_entropy = 0.0

        if domain_entropy > 4.0:

            top_factors.append({

                "feature":
                    "domain_entropy",

                "impact":
                    0.35,

                "direction":
                    "increases risk",

                "supports_class":
                    "phishing",

                "description":
                    self._entropy_explanation(
                        "domain_entropy"
                    )
            })

        # -------------------------------------------------------------
        # PATH entropy
        #
        # IMPORTANT:
        # This is intentionally weak.
        # -------------------------------------------------------------

        try:

            path_entropy = float(
                features.get(
                    "path_entropy",
                    0.0
                )
            )

        except Exception:

            path_entropy = 0.0

        if path_entropy > 5.0:

            top_factors.append({

                "feature":
                    "path_entropy",

                "impact":
                    0.08,

                "direction":
                    "contextual",

                "supports_class":
                    "unknown",

                "description":
                    self._entropy_explanation(
                        "path_entropy"
                    )
            })

        # -------------------------------------------------------------
        # Query entropy
        # -------------------------------------------------------------

        try:

            query_entropy = float(
                features.get(
                    "query_entropy",
                    0.0
                )
            )

        except Exception:

            query_entropy = 0.0

        if query_entropy > 5.0:

            top_factors.append({

                "feature":
                    "query_entropy",

                "impact":
                    0.10,

                "direction":
                    "contextual",

                "supports_class":
                    "unknown",

                "description":
                    self._entropy_explanation(
                        "query_entropy"
                    )
            })

        # -------------------------------------------------------------
        # Suspicious words
        # -------------------------------------------------------------

        try:

            suspicious_words = float(
                features.get(
                    "suspicious_word_count",
                    0
                )
            )

        except Exception:

            suspicious_words = 0.0

        if suspicious_words > 0:

            top_factors.append({

                "feature":
                    "suspicious_word_count",

                "impact":
                    0.30,

                "direction":
                    "increases risk",

                "supports_class":
                    "phishing",

                "description":
                    (
                        "The URL contains keywords commonly "
                        "associated with social engineering or "
                        "credential harvesting."
                    )
            })

        # -------------------------------------------------------------
        # HTTPS
        # -------------------------------------------------------------

        try:

            https = float(
                features.get(
                    "is_https",
                    0
                )
            )

        except Exception:

            https = 0.0

        if https == 1:

            top_factors.append({

                "feature":
                    "is_https",

                "impact":
                    0.10,

                "direction":
                    "decreases risk",

                "supports_class":
                    "legitimate",

                "description":
                    (
                        "The URL uses HTTPS. This is a positive "
                        "transport-security indicator, although HTTPS "
                        "alone does not prove that a website is legitimate."
                    )
            })

        # -------------------------------------------------------------
        # Hyphens
        # -------------------------------------------------------------

        try:

            hyphens = float(
                features.get(
                    "num_hyphens",
                    0
                )
            )

        except Exception:

            hyphens = 0.0

        if hyphens >= 3:

            top_factors.append({

                "feature":
                    "num_hyphens",

                "impact":
                    0.12,

                "direction":
                    "increases risk",

                "supports_class":
                    "phishing",

                "description":
                    (
                        "The URL contains multiple hyphens. "
                        "This may indicate obfuscation or brand "
                        "spoofing, but should be interpreted with "
                        "other domain characteristics."
                    )
            })

        # -------------------------------------------------------------
        # If nothing significant was found
        # -------------------------------------------------------------

        if not top_factors:

            top_factors.append({

                "feature":
                    "baseline_url_analysis",

                "impact":
                    0.0,

                "direction":
                    "neutral",

                "supports_class":
                    prediction,

                "description":
                    (
                        "No dominant heuristic URL indicator "
                        "was identified."
                    )
            })

        # -------------------------------------------------------------
        # Summary
        # -------------------------------------------------------------

        meaningful_factors = [
            item["description"]
            for item in top_factors
            if item.get(
                "direction"
            ) != "neutral"
        ]

        if meaningful_factors:

            summary = (
                f"Heuristic URL analysis classified the URL as "
                f"'{prediction}' based primarily on: "
                f"{', '.join(meaningful_factors[:2])}."
            )

        else:

            summary = (
                f"Heuristic URL analysis produced a "
                f"'{prediction}' classification without a "
                f"dominant risk indicator."
            )

        return {

            "summary":
                summary,

            "top_factors":
                top_factors,

            "explanation_source":
                "heuristic",

            "schema_version":
                self.schema_version
        }

    # =================================================================
    # SHAP EXTRACTION
    # =================================================================

    def _extract_shap_values(
        self,
        shap_result: Any,
        target_class_idx: int
    ) -> np.ndarray:
        """
        Normalize SHAP output across different SHAP versions.

        SHAP may return:

            list
            numpy.ndarray
            shap.Explanation

        This method converts all supported formats into:

            1-dimensional numpy array

        representing one instance.
        """

        # -------------------------------------------------------------
        # New SHAP Explanation API
        # -------------------------------------------------------------

        if hasattr(
            shap_result,
            "values"
        ):

            values = shap_result.values

        else:

            values = shap_result

        # -------------------------------------------------------------
        # Convert to numpy
        # -------------------------------------------------------------

        values = np.asarray(
            values,
            dtype=float
        )

        # -------------------------------------------------------------
        # Empty output
        # -------------------------------------------------------------

        if values.size == 0:

            return np.array(
                [],
                dtype=float
            )

        # -------------------------------------------------------------
        # 1D
        # -------------------------------------------------------------

        if values.ndim == 1:

            return values

        # -------------------------------------------------------------
        # 2D
        #
        # Usually:
        #
        # samples × features
        # -------------------------------------------------------------

        if values.ndim == 2:

            return values[0]

        # -------------------------------------------------------------
        # 3D
        #
        # Possible formats:
        #
        # samples × features × classes
        #
        # OR
        #
        # samples × classes × features
        # -------------------------------------------------------------

        if values.ndim == 3:

            # samples × features × classes

            if (
                values.shape[0] == 1
                and
                values.shape[1] == len(
                    self.feature_names
                )
            ):

                class_index = min(
                    target_class_idx,
                    values.shape[2] - 1
                )

                return values[
                    0,
                    :,
                    class_index
                ]

            # samples × classes × features

            if (
                values.shape[0] == 1
                and
                values.shape[2] == len(
                    self.feature_names
                )
            ):

                class_index = min(
                    target_class_idx,
                    values.shape[1] - 1
                )

                return values[
                    0,
                    class_index,
                    :
                ]

            # Last-resort flattening

            return values.reshape(
                -1
            )

        # -------------------------------------------------------------
        # Unexpected dimensionality
        # -------------------------------------------------------------

        return values.reshape(
            -1
        )

    # =================================================================
    # SHAP EXPLANATION
    # =================================================================

    def generate_explanation(
        self,
        features: Dict[str, Any],
        prediction: str
    ) -> Dict[str, Any]:
        """
        Generate a local explanation for the prediction.

        Preferred order:

            SHAP
              ↓
            heuristic fallback

        The method NEVER allows XAI failure to crash the URL Agent.
        """

        # -------------------------------------------------------------
        # Validate input
        # -------------------------------------------------------------

        if not isinstance(
            features,
            dict
        ):

            features = {}

        prediction = str(
            prediction or "unknown"
        ).lower()

        # -------------------------------------------------------------
        # SHAP unavailable
        # -------------------------------------------------------------

        if (
            not SHAP_AVAILABLE
            or self.explainer is None
        ):

            return self._heuristic_explanation(
                features,
                prediction
            )

        try:

            # =========================================================
            # Build exact model input
            # =========================================================

            df_features = (
                self._align_features(
                    features
                )
            )

            target_class_idx = (
                self.CLASS_INDEX_MAP.get(
                    prediction,
                    1
                )
            )

            # =========================================================
            # Calculate SHAP
            # =========================================================

            shap_result = (
                self.explainer.shap_values(
                    df_features
                )
            )

            instance_shap_values = (
                self._extract_shap_values(
                    shap_result,
                    target_class_idx
                )
            )

            if len(
                instance_shap_values
            ) < len(
                self.feature_names
            ):

                logger.warning(
                    "SHAP returned %d values for %d features. "
                    "Falling back to heuristic explanation.",
                    len(instance_shap_values),
                    len(self.feature_names)
                )

                return self._heuristic_explanation(
                    features,
                    prediction
                )

            # =========================================================
            # Build feature impact list
            # =========================================================

            feature_impacts = []

            for index, feature_name in enumerate(
                self.feature_names
            ):

                if index >= len(
                    instance_shap_values
                ):

                    continue

                try:

                    impact_value = float(
                        instance_shap_values[index]
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

                if not np.isfinite(
                    impact_value
                ):

                    continue

                if abs(
                    impact_value
                ) <= 0.0001:

                    continue

                # -----------------------------------------------------
                # HTTPS guardrail
                # -----------------------------------------------------

                if (
                    feature_name == "is_https"
                    and
                    impact_value > 0
                ):

                    continue

                # -----------------------------------------------------
                # Path/query entropy guardrail
                #
                # Do NOT remove the model's actual SHAP value.
                #
                # Instead, mark it contextual later.
                # -----------------------------------------------------

                feature_impacts.append(
                    (
                        feature_name,
                        impact_value
                    )
                )

            # =========================================================
            # Sort strongest factors
            # =========================================================

            feature_impacts.sort(
                key=lambda item:
                    abs(item[1]),
                reverse=True
            )

            # =========================================================
            # Generate top factors
            # =========================================================

            top_factors = []

            supporting_contributors = []

            for feature_name, impact in feature_impacts[:10]:

                impact_value = round(
                    float(impact),
                    4
                )

                # -----------------------------------------------------
                # Standard SHAP direction
                #
                # Positive → phishing
                # Negative → legitimate
                # -----------------------------------------------------

                if impact_value > 0:

                    supports_class = (
                        "phishing"
                    )

                    direction = (
                        "increases risk"
                    )

                else:

                    supports_class = (
                        "legitimate"
                    )

                    direction = (
                        "decreases risk"
                    )

                # -----------------------------------------------------
                # Entropy-specific interpretation
                # -----------------------------------------------------

                if self._is_entropy_feature(
                    feature_name
                ):

                    description = (
                        self._entropy_explanation(
                            feature_name
                        )
                    )

                    # Path/query entropy is contextual rather than
                    # direct evidence.

                    if feature_name in {
                        "path_entropy",
                        "query_entropy"
                    }:

                        direction = (
                            "contextual"
                        )

                        supports_class = (
                            "contextual"
                        )

                else:

                    description = (
                        self._get_human_description(
                            feature_name
                        )
                    )

                # -----------------------------------------------------
                # Add factor
                # -----------------------------------------------------

                top_factors.append({

                    "feature":
                        feature_name,

                    "impact":
                        impact_value,

                    "direction":
                        direction,

                    "supports_class":
                        supports_class,

                    "description":
                        description
                })

                # -----------------------------------------------------
                # Summary contributors
                #
                # Do not allow contextual entropy to dominate the
                # natural-language summary.
                # -----------------------------------------------------

                if (
                    prediction == "phishing"
                    and
                    impact_value > 0
                    and
                    feature_name not in {
                        "path_entropy",
                        "query_entropy"
                    }
                ):

                    supporting_contributors.append(
                        description
                    )

                elif (
                    prediction == "legitimate"
                    and
                    impact_value < 0
                ):

                    supporting_contributors.append(
                        description
                    )

            # =========================================================
            # Natural-language summary
            # =========================================================

            if supporting_contributors:

                summary = (
                    f"The URL classification of "
                    f"'{prediction}' was primarily influenced "
                    f"by: "
                    f"{', '.join(supporting_contributors[:2])}."
                )

            else:

                summary = (
                    f"The URL classification of "
                    f"'{prediction}' was based primarily on "
                    f"the combined lexical and structural "
                    f"characteristics of the URL."
                )

            # =========================================================
            # Return explanation
            # =========================================================

            return {

                "summary":
                    summary,

                "top_factors":
                    top_factors[:5],

                "explanation_source":
                    "SHAP",

                "schema_version":
                    self.schema_version,

                "feature_count":
                    len(
                        self.feature_names
                    )
            }

        except Exception as e:

            logger.error(
                "URL XAI SHAP extraction failed: %s",
                str(e),
                exc_info=True
            )

            # ---------------------------------------------------------
            # IMPORTANT:
            # XAI failure must NEVER crash the entire URL Agent.
            # ---------------------------------------------------------

            return self._heuristic_explanation(
                features,
                prediction
            )
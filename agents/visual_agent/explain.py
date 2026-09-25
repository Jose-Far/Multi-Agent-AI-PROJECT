"""
Production-grade Explainable AI (XAI) engine for the Visual AI Agent.

Responsibilities
----------------

    Trained Visual XGBoost Model
                |
                v
          SHAP TreeExplainer
                |
                v
       Per-instance attribution
                |
                v
        Feature contributions
                |
                v
        Human-readable evidence
                |
                v
       Visual Agent explanation


Classification Contract
------------------------

Current trained Visual model:

    0 = legitimate
    1 = phishing

The explainer also tolerates legacy "suspicious" references by mapping
them safely to the phishing class when necessary.

Visual Feature Contract
-----------------------

Exactly 12 features are expected:

    1.  screenshot_width
    2.  screenshot_height
    3.  image_entropy
    4.  dominant_color_count
    5.  text_density
    6.  image_density
    7.  form_area_ratio
    8.  button_count
    9.  logo_detected
    10. login_form_detected
    11. layout_complexity
    12. blank_area_ratio


Explainability Modes
--------------------

    SHAP
        Actual model-specific feature attribution.

    HEURISTIC
        Deterministic rule-based explanation used when SHAP or the model
        is unavailable.

    ERROR_FALLBACK
        Safe explanation when an unexpected XAI failure occurs.


Important
---------

This module explains an already-generated prediction.

It does NOT:

    - train the XGBoost model
    - perform Visual feature extraction
    - calculate the final project-wide risk score
    - replace the predictor
    - replace the central fusion engine

No standalone execution hook is included.
"""

from __future__ import annotations


# =============================================================================
# STANDARD LIBRARY
# =============================================================================

import logging
import math

from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# =============================================================================
# THIRD-PARTY
# =============================================================================

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT IMPORT
# =============================================================================

from .feature_schema import VisualFeatureSchema


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# OPTIONAL SHAP IMPORT
# =============================================================================

try:

    import shap

    SHAP_AVAILABLE = True

except ImportError:

    shap = None

    SHAP_AVAILABLE = False

    logger.warning(
        "SHAP is not installed. "
        "VisualExplainer will use heuristic explanations."
    )


# =============================================================================
# VISUAL EXPLAINER
# =============================================================================

class VisualExplainer:
    """
    Explainability engine for the Visual AI Agent.

    Supports the current binary Visual model:

        0 = legitimate
        1 = phishing

    The implementation remains backward compatible with older project code
    that may refer to "suspicious".
    """

    # =========================================================================
    # 1. CURRENT CLASSIFICATION CONTRACT
    # =========================================================================

    CLASS_INDEX_MAP: ClassVar[Dict[str, int]] = {
        "legitimate": 0,
        "phishing": 1,
    }

    CLASS_LABELS: ClassVar[Dict[int, str]] = {
        0: "legitimate",
        1: "phishing",
    }

    NUM_CLASSES: ClassVar[int] = 2

    # -------------------------------------------------------------------------
    # Legacy compatibility.
    #
    # "suspicious" is not a trained class in the current Visual model.
    # -------------------------------------------------------------------------

    LEGACY_CLASS_ALIASES: ClassVar[Dict[str, str]] = {
        "suspicious": "phishing",
        "malicious": "phishing",
        "unsafe": "phishing",
    }

    # =========================================================================
    # 2. EXPLANATION MODES
    # =========================================================================

    EXPLANATION_MODE_SHAP: ClassVar[str] = "shap"

    EXPLANATION_MODE_HEURISTIC: ClassVar[str] = (
        "heuristic"
    )

    EXPLANATION_MODE_ERROR: ClassVar[str] = (
        "error_fallback"
    )

    # =========================================================================
    # 3. CONFIGURATION
    # =========================================================================

    DEFAULT_TOP_K: ClassVar[int] = 5

    MIN_ABSOLUTE_SHAP_IMPACT: ClassVar[float] = (
        0.0001
    )

    # =========================================================================
    # 4. HEURISTIC THRESHOLDS
    # =========================================================================

    FORM_AREA_THRESHOLD: ClassVar[float] = 0.30

    IMAGE_DENSITY_THRESHOLD: ClassVar[float] = 0.80

    BLANK_AREA_THRESHOLD: ClassVar[float] = 0.60

    LAYOUT_COMPLEXITY_THRESHOLD: ClassVar[float] = 80.0

    BUTTON_COUNT_THRESHOLD: ClassVar[float] = 5.0

    ENTROPY_THRESHOLD: ClassVar[float] = 7.5

    # =========================================================================
    # 5. FALLBACK FEATURE DESCRIPTIONS
    # =========================================================================

    FALLBACK_FEATURE_DESCRIPTIONS: ClassVar[
        Dict[str, str]
    ] = {

        "screenshot_width":
            "Total rendered width of the webpage screenshot in pixels.",

        "screenshot_height":
            "Total rendered height of the webpage screenshot in pixels.",

        "image_entropy":
            "Measurement of visual randomness and image complexity.",

        "dominant_color_count":
            "Number of dominant colors detected in the visual layout.",

        "text_density":
            "Ratio of visible screen area occupied by text.",

        "image_density":
            "Ratio of visible screen area occupied by images.",

        "form_area_ratio":
            "Ratio of visible screen area occupied by forms.",

        "button_count":
            "Number of visible interactive buttons.",

        "logo_detected":
            "Indicates whether a logo-like visual element was detected.",

        "login_form_detected":
            "Indicates whether a login or credential-entry interface "
            "was visually detected.",

        "layout_complexity":
            "Estimated complexity of the rendered visual layout.",

        "blank_area_ratio":
            "Ratio of relatively empty or whitespace screen area.",
    }

    # =========================================================================
    # 6. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model: Any = None,
        *,
        default_top_k: int = DEFAULT_TOP_K,
        min_absolute_shap_impact: float = (
            MIN_ABSOLUTE_SHAP_IMPACT
        ),
    ) -> None:

        self.model = model

        self.explainer: Optional[Any] = None

        self.default_top_k = int(
            default_top_k
        )

        self.min_absolute_shap_impact = float(
            min_absolute_shap_impact
        )

        self.shap_initialized = False

        self.shap_initialization_error: Optional[
            str
        ] = None

        # ---------------------------------------------------------------------
        # Validate schema.
        # ---------------------------------------------------------------------

        VisualFeatureSchema.verify_schema_integrity()

        self.feature_schema = (
            VisualFeatureSchema.get_schema()
        )

        # ---------------------------------------------------------------------
        # Initialize SHAP.
        # ---------------------------------------------------------------------

        self._initialize_explainer()

        logger.info(
            "VisualExplainer initialized: "
            "model_available=%s, "
            "shap_available=%s, "
            "shap_initialized=%s, "
            "feature_count=%s, "
            "num_classes=%s",
            self.model is not None,
            SHAP_AVAILABLE,
            self.shap_initialized,
            len(self.feature_schema),
            self.NUM_CLASSES,
        )

    # =========================================================================
    # 7. CONFIGURATION VALIDATION
    # =========================================================================

    def _validate_configuration(
        self,
    ) -> None:

        if self.default_top_k <= 0:

            raise ValueError(
                "default_top_k must be greater than zero."
            )

        if not math.isfinite(
            self.min_absolute_shap_impact
        ):

            raise ValueError(
                "min_absolute_shap_impact must be finite."
            )

        if self.min_absolute_shap_impact < 0.0:

            raise ValueError(
                "min_absolute_shap_impact cannot be negative."
            )

    # =========================================================================
    # 8. SHAP INITIALIZATION
    # =========================================================================

    def _initialize_explainer(
        self,
    ) -> None:

        self.explainer = None

        self.shap_initialized = False

        self.shap_initialization_error = None

        if not SHAP_AVAILABLE:

            logger.warning(
                "VisualExplainer: SHAP library unavailable."
            )

            return

        if self.model is None:

            logger.info(
                "VisualExplainer: no model supplied."
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
                "VisualExplainer: SHAP TreeExplainer initialized."
            )

        except Exception as exc:

            self.shap_initialization_error = str(
                exc
            )

            self.explainer = None

            logger.warning(
                "VisualExplainer: SHAP initialization failed: %s",
                exc,
                exc_info=True,
            )

    # =========================================================================
    # 9. MODEL UPDATE
    # =========================================================================

    def set_model(
        self,
        model: Any,
    ) -> None:

        self.model = model

        self._initialize_explainer()

    # =========================================================================
    # 10. CLASS NORMALIZATION
    # =========================================================================

    def _normalize_prediction_class(
        self,
        prediction: Any,
    ) -> Tuple[int, str]:
        """
        Normalize prediction to:

            (class_index, class_label)

        Current contract:

            0 -> legitimate
            1 -> phishing

        Legacy "suspicious" is accepted and mapped to phishing.
        """

        # ---------------------------------------------------------------------
        # String label.
        # ---------------------------------------------------------------------

        if isinstance(
            prediction,
            str,
        ):

            normalized = (
                prediction
                .strip()
                .lower()
            )

            normalized = (
                self.LEGACY_CLASS_ALIASES.get(
                    normalized,
                    normalized,
                )
            )

            if normalized in self.CLASS_INDEX_MAP:

                index = (
                    self.CLASS_INDEX_MAP[
                        normalized
                    ]
                )

                return (
                    index,
                    self.CLASS_LABELS[
                        index
                    ],
                )

            raise ValueError(
                "Unknown Visual prediction label: "
                f"{prediction}"
            )

        # ---------------------------------------------------------------------
        # Numeric class.
        # ---------------------------------------------------------------------

        try:

            index = int(
                prediction
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "Visual prediction must be "
                "'legitimate'/'phishing' or class index 0/1."
            ) from exc

        if index not in self.CLASS_LABELS:

            raise ValueError(
                "Visual prediction class index must be 0 or 1."
            )

        return (
            index,
            self.CLASS_LABELS[
                index
            ],
        )

    # =========================================================================
    # 11. FEATURE NORMALIZATION
    # =========================================================================

    def _normalize_features(
        self,
        features: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:

        if features is None:

            return {}

        if not isinstance(
            features,
            Mapping,
        ):

            raise TypeError(
                "Visual explanation features must be a mapping."
            )

        return dict(
            features
        )

    # =========================================================================
    # 12. FEATURE DATAFRAME
    # =========================================================================

    def _prepare_feature_dataframe(
        self,
        features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Prepare exactly one 12-feature model row.
        """

        df_features = (
            VisualFeatureSchema.align_and_validate(
                features
            )
        )

        if not isinstance(
            df_features,
            pd.DataFrame,
        ):

            raise TypeError(
                "VisualFeatureSchema.align_and_validate() "
                "must return a pandas DataFrame."
            )

        if len(
            df_features
        ) != 1:

            raise ValueError(
                "Visual XAI requires exactly one feature row."
            )

        if list(
            df_features.columns
        ) != list(
            self.feature_schema
        ):

            raise ValueError(
                "Visual XAI feature order does not match "
                "the trained schema."
            )

        if df_features.shape[1] != 12:

            raise ValueError(
                "Visual XAI requires exactly 12 features."
            )

        df_features = (
            df_features
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(
                0.0
            )
            .astype(
                "float64"
            )
        )

        if not np.isfinite(
            df_features.to_numpy(
                dtype=float
            )
        ).all():

            raise ValueError(
                "Visual XAI feature matrix contains "
                "non-finite values."
            )

        return df_features

    # =========================================================================
    # 13. FEATURE DESCRIPTION
    # =========================================================================

    def _get_feature_description(
        self,
        feature_name: str,
    ) -> str:
        """
        Safely obtain a feature description.

        Primary source:
            VisualFeatureSchema.get_descriptions()

        Fallback:
            local description dictionary.

        This prevents XAI failure if an older feature_schema.py is loaded.
        """

        # ---------------------------------------------------------------------
        # Primary schema method.
        # ---------------------------------------------------------------------

        try:

            get_descriptions = getattr(
                VisualFeatureSchema,
                "get_descriptions",
                None,
            )

            if callable(
                get_descriptions
            ):

                descriptions = (
                    get_descriptions()
                )

                if isinstance(
                    descriptions,
                    Mapping,
                ):

                    description = (
                        descriptions.get(
                            feature_name
                        )
                    )

                    if description:

                        return str(
                            description
                        )

        except Exception as exc:

            logger.debug(
                "Could not obtain feature description from schema: %s",
                exc,
            )

        # ---------------------------------------------------------------------
        # Secondary schema method.
        # ---------------------------------------------------------------------

        try:

            get_feature_description = getattr(
                VisualFeatureSchema,
                "get_feature_description",
                None,
            )

            if callable(
                get_feature_description
            ):

                description = (
                    get_feature_description(
                        feature_name
                    )
                )

                if description:

                    return str(
                        description
                    )

        except Exception as exc:

            logger.debug(
                "Schema single-feature description lookup failed: %s",
                exc,
            )

        # ---------------------------------------------------------------------
        # Local fallback.
        # ---------------------------------------------------------------------

        return (
            self.FALLBACK_FEATURE_DESCRIPTIONS.get(
                feature_name,
                feature_name.replace(
                    "_",
                    " ",
                ).title(),
            )
        )

    # =========================================================================
    # 14. SAFE FLOAT
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            converted = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return default

        if not math.isfinite(
            converted
        ):

            return default

        return converted

    # =========================================================================
    # 15. SHAP CALCULATION
    # =========================================================================

    def _calculate_shap_values(
        self,
        df_features: pd.DataFrame,
    ) -> Any:
        """
        Support both modern and legacy SHAP APIs.
        """

        if self.explainer is None:

            raise RuntimeError(
                "SHAP explainer is not initialized."
            )

        # ---------------------------------------------------------------------
        # Modern SHAP Explanation API.
        # ---------------------------------------------------------------------

        try:

            return self.explainer(
                df_features
            )

        except Exception as modern_exc:

            logger.debug(
                "Modern SHAP API failed: %s",
                modern_exc,
            )

        # ---------------------------------------------------------------------
        # Legacy API.
        # ---------------------------------------------------------------------

        try:

            return self.explainer.shap_values(
                df_features
            )

        except Exception as legacy_exc:

            raise RuntimeError(
                "Both modern and legacy SHAP APIs failed."
            ) from legacy_exc

    # =========================================================================
    # 16. SHAP VECTOR EXTRACTION
    # =========================================================================

    def _extract_class_shap_vector(
        self,
        shap_output: Any,
        target_class_index: int,
    ) -> np.ndarray:
        """
        Extract one feature-attribution vector.

        Handles:

            SHAP Explanation
            list-based multiclass output
            2D arrays
            3D arrays

        Also handles binary SHAP outputs where only one output vector is
        returned.
        """

        # ---------------------------------------------------------------------
        # Extract values from Explanation object.
        # ---------------------------------------------------------------------

        if hasattr(
            shap_output,
            "values",
        ):

            values = (
                shap_output.values
            )

        else:

            values = shap_output

        # ---------------------------------------------------------------------
        # List-based SHAP output.
        # ---------------------------------------------------------------------

        if isinstance(
            values,
            list,
        ):

            if len(
                values
            ) == 0:

                raise ValueError(
                    "SHAP returned an empty list."
                )

            # ---------------------------------------------------------------
            # Current binary model.
            # ---------------------------------------------------------------

            if len(
                values
            ) == 2:

                selected = np.asarray(
                    values[
                        target_class_index
                    ],
                    dtype=float,
                )

            # ---------------------------------------------------------------
            # Legacy 3-class model.
            # ---------------------------------------------------------------

            elif len(
                values
            ) == 3:

                legacy_index = min(
                    target_class_index,
                    2,
                )

                selected = np.asarray(
                    values[
                        legacy_index
                    ],
                    dtype=float,
                )

            # ---------------------------------------------------------------
            # Unexpected number of outputs.
            # ---------------------------------------------------------------

            else:

                selected = np.asarray(
                    values[
                        min(
                            target_class_index,
                            len(values) - 1,
                        )
                    ],
                    dtype=float,
                )

            if selected.ndim == 2:

                selected = selected[
                    0
                ]

            return selected.reshape(
                -1
            )

        # ---------------------------------------------------------------------
        # ndarray.
        # ---------------------------------------------------------------------

        values = np.asarray(
            values,
            dtype=float,
        )

        # ---------------------------------------------------------------------
        # 1D:
        #
        # (features,)
        # ---------------------------------------------------------------------

        if values.ndim == 1:

            return values.reshape(
                -1
            )

        # ---------------------------------------------------------------------
        # 2D:
        #
        # Usually:
        #
        # (samples, features)
        # ---------------------------------------------------------------------

        if values.ndim == 2:

            if values.shape[0] < 1:

                raise ValueError(
                    "SHAP output contains no samples."
                )

            return values[
                0
            ].reshape(
                -1
            )

        # ---------------------------------------------------------------------
        # 3D:
        #
        # Possible:
        #
        # (samples, features, classes)
        #
        # OR
        #
        # (samples, classes, features)
        # ---------------------------------------------------------------------

        if values.ndim == 3:

            if values.shape[0] < 1:

                raise ValueError(
                    "SHAP output contains no samples."
                )

            feature_count = len(
                self.feature_schema
            )

            class_count = (
                self.NUM_CLASSES
            )

            # ---------------------------------------------------------------
            # (1, features, classes)
            # ---------------------------------------------------------------

            if (
                values.shape[1]
                == feature_count
                and values.shape[2]
                >= 2
            ):

                class_index = min(
                    target_class_index,
                    values.shape[2] - 1,
                )

                return values[
                    0,
                    :,
                    class_index,
                ].reshape(
                    -1
                )

            # ---------------------------------------------------------------
            # (1, classes, features)
            # ---------------------------------------------------------------

            if (
                values.shape[2]
                == feature_count
                and values.shape[1]
                >= 2
            ):

                class_index = min(
                    target_class_index,
                    values.shape[1] - 1,
                )

                return values[
                    0,
                    class_index,
                    :,
                ].reshape(
                    -1
                )

            raise ValueError(
                "Unsupported SHAP 3D shape: "
                f"{values.shape}"
            )

        raise ValueError(
            "Unsupported SHAP output dimensions: "
            f"{values.ndim}"
        )

    # =========================================================================
    # 17. BASE VALUE
    # =========================================================================

    def _extract_base_value(
        self,
        shap_output: Any,
        target_class_index: int,
    ) -> Optional[float]:

        if not hasattr(
            shap_output,
            "base_values",
        ):

            return None

        try:

            base_values = np.asarray(
                shap_output.base_values,
                dtype=float,
            )

            if base_values.ndim == 0:

                return self._safe_float(
                    base_values
                )

            if base_values.ndim == 1:

                if len(
                    base_values
                ) == 1:

                    return self._safe_float(
                        base_values[
                            0
                        ]
                    )

                class_index = min(
                    target_class_index,
                    len(base_values) - 1,
                )

                return self._safe_float(
                    base_values[
                        class_index
                    ]
                )

            if base_values.ndim == 2:

                if (
                    base_values.shape[0]
                    < 1
                ):

                    return None

                if (
                    base_values.shape[1]
                    == 1
                ):

                    return self._safe_float(
                        base_values[
                            0,
                            0,
                        ]
                    )

                class_index = min(
                    target_class_index,
                    base_values.shape[1] - 1,
                )

                return self._safe_float(
                    base_values[
                        0,
                        class_index,
                    ]
                )

        except Exception:

            return None

        return None

    # =========================================================================
    # 18. BUILD SHAP FEATURE IMPACTS
    # =========================================================================

    def _build_shap_feature_impacts(
        self,
        shap_vector: np.ndarray,
        df_features: pd.DataFrame,
        prediction_label: str,
    ) -> List[
        Dict[str, Any]
    ]:

        if len(
            shap_vector
        ) != len(
            self.feature_schema
        ):

            raise ValueError(
                "SHAP feature vector length "
                f"{len(shap_vector)} does not match "
                f"Visual schema length "
                f"{len(self.feature_schema)}."
            )

        feature_impacts: List[
            Dict[str, Any]
        ] = []

        for index, feature_name in enumerate(
            self.feature_schema
        ):

            impact = self._safe_float(
                shap_vector[
                    index
                ]
            )

            if (
                abs(
                    impact
                )
                < self.min_absolute_shap_impact
            ):

                continue

            feature_value = self._safe_float(
                df_features.iloc[
                    0,
                    index,
                ]
            )

            if impact > 0:

                direction = (
                    "supports_prediction"
                )

            elif impact < 0:

                direction = (
                    "opposes_prediction"
                )

            else:

                direction = "neutral"

            feature_impacts.append(
                {
                    "feature":
                        feature_name,

                    "value":
                        round(
                            feature_value,
                            6,
                        ),

                    "shap_value":
                        round(
                            impact,
                            6,
                        ),

                    "absolute_impact":
                        round(
                            abs(
                                impact
                            ),
                            6,
                        ),

                    "direction":
                        direction,

                    "supports_class":
                        (
                            prediction_label
                            if impact > 0
                            else self._opposing_class(
                                prediction_label
                            )
                        ),

                    "description":
                        self._get_feature_description(
                            feature_name
                        ),
                }
            )

        feature_impacts.sort(
            key=lambda item: item[
                "absolute_impact"
            ],
            reverse=True,
        )

        return feature_impacts

    # =========================================================================
    # 19. OPPOSING CLASS
    # =========================================================================

    def _opposing_class(
        self,
        prediction_label: str,
    ) -> str:

        if prediction_label == "phishing":

            return "legitimate"

        return "phishing"

    # =========================================================================
    # 20. HEURISTIC NUMERIC FEATURE
    # =========================================================================

    @staticmethod
    def _feature_number(
        features: Mapping[str, Any],
        name: str,
        default: float = 0.0,
    ) -> float:

        try:

            value = float(
                features.get(
                    name,
                    default,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return default

        if not math.isfinite(
            value
        ):

            return default

        return value

    # =========================================================================
    # 21. HEURISTIC BOOLEAN FEATURE
    # =========================================================================

    @staticmethod
    def _feature_boolean(
        features: Mapping[str, Any],
        name: str,
    ) -> bool:

        value = features.get(
            name,
            False,
        )

        if isinstance(
            value,
            bool,
        ):

            return value

        try:

            return float(
                value
            ) >= 0.5

        except (
            TypeError,
            ValueError,
        ):

            return False

    # =========================================================================
    # 22. HEURISTIC EXPLANATION
    # =========================================================================

    def _heuristic_explanation(
        self,
        features: Mapping[str, Any],
        prediction: Any,
    ) -> Dict[str, Any]:

        _, prediction_label = (
            self._normalize_prediction_class(
                prediction
            )
        )

        normalized_features = (
            self._normalize_features(
                features
            )
        )

        top_factors: List[
            Dict[str, Any]
        ] = []

        # ---------------------------------------------------------------------
        # Login form.
        # ---------------------------------------------------------------------

        if self._feature_boolean(
            normalized_features,
            "login_form_detected",
        ):

            top_factors.append(
                {
                    "feature":
                        "login_form_detected",

                    "value":
                        1.0,

                    "impact":
                        0.40,

                    "direction":
                        "increases_phishing_risk",

                    "supports_class":
                        "phishing",

                    "description":
                        self._get_feature_description(
                            "login_form_detected"
                        ),

                    "reason":
                        (
                            "A credential-entry interface "
                            "was visually detected."
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Logo.
        # ---------------------------------------------------------------------

        if self._feature_boolean(
            normalized_features,
            "logo_detected",
        ):

            top_factors.append(
                {
                    "feature":
                        "logo_detected",

                    "value":
                        1.0,

                    "impact":
                        0.25,

                    "direction":
                        "increases_contextual_risk",

                    "supports_class":
                        "phishing",

                    "description":
                        self._get_feature_description(
                            "logo_detected"
                        ),

                    "reason":
                        (
                            "A logo-like visual element "
                            "was detected."
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Login + logo.
        # ---------------------------------------------------------------------

        if (
            self._feature_boolean(
                normalized_features,
                "login_form_detected",
            )
            and
            self._feature_boolean(
                normalized_features,
                "logo_detected",
            )
        ):

            top_factors.append(
                {
                    "feature":
                        "login_form_detected + logo_detected",

                    "value":
                        1.0,

                    "impact":
                        0.45,

                    "direction":
                        "increases_phishing_risk",

                    "supports_class":
                        "phishing",

                    "description":
                        (
                            "A credential-entry interface and "
                            "logo-like element occur together."
                        ),

                    "reason":
                        (
                            "This combination can be relevant "
                            "to impersonation-style pages."
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Form area.
        # ---------------------------------------------------------------------

        form_area = (
            self._feature_number(
                normalized_features,
                "form_area_ratio",
            )
        )

        if (
            form_area
            > self.FORM_AREA_THRESHOLD
        ):

            top_factors.append(
                {
                    "feature":
                        "form_area_ratio",

                    "value":
                        round(
                            form_area,
                            6,
                        ),

                    "impact":
                        0.30,

                    "direction":
                        "increases_contextual_risk",

                    "supports_class":
                        "phishing",

                    "description":
                        self._get_feature_description(
                            "form_area_ratio"
                        ),

                    "reason":
                        (
                            "Forms occupy a relatively large "
                            "portion of the visible page."
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Image density.
        # ---------------------------------------------------------------------

        image_density = (
            self._feature_number(
                normalized_features,
                "image_density",
            )
        )

        if (
            image_density
            > self.IMAGE_DENSITY_THRESHOLD
        ):

            top_factors.append(
                {
                    "feature":
                        "image_density",

                    "value":
                        round(
                            image_density,
                            6,
                        ),

                    "impact":
                        0.20,

                    "direction":
                        "increases_contextual_risk",

                    "supports_class":
                        "phishing",

                    "description":
                        self._get_feature_description(
                            "image_density"
                        ),

                    "reason":
                        (
                            "High image density was detected "
                            "in the visual layout."
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Blank area.
        # ---------------------------------------------------------------------

        blank_area = (
            self._feature_number(
                normalized_features,
                "blank_area_ratio",
            )
        )

        if (
            blank_area
            > self.BLANK_AREA_THRESHOLD
        ):

            top_factors.append(
                {
                    "feature":
                        "blank_area_ratio",

                    "value":
                        round(
                            blank_area,
                            6,
                        ),

                    "impact":
                        0.15,

                    "direction":
                        "increases_contextual_risk",

                    "supports_class":
                        "phishing",

                    "description":
                        self._get_feature_description(
                            "blank_area_ratio"
                        ),

                    "reason":
                        (
                            "The page contains substantial "
                            "empty visual space."
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Layout complexity.
        # ---------------------------------------------------------------------

        layout_complexity = (
            self._feature_number(
                normalized_features,
                "layout_complexity",
            )
        )

        if (
            layout_complexity
            > self.LAYOUT_COMPLEXITY_THRESHOLD
        ):

            top_factors.append(
                {
                    "feature":
                        "layout_complexity",

                    "value":
                        round(
                            layout_complexity,
                            6,
                        ),

                    "impact":
                        0.15,

                    "direction":
                        "increases_contextual_risk",

                    "supports_class":
                        "phishing",

                    "description":
                        self._get_feature_description(
                            "layout_complexity"
                        ),

                    "reason":
                        (
                            "The visual layout has relatively "
                            "high complexity."
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Button count.
        # ---------------------------------------------------------------------

        button_count = (
            self._feature_number(
                normalized_features,
                "button_count",
            )
        )

        if (
            button_count
            > self.BUTTON_COUNT_THRESHOLD
        ):

            top_factors.append(
                {
                    "feature":
                        "button_count",

                    "value":
                        round(
                            button_count,
                            6,
                        ),

                    "impact":
                        0.10,

                    "direction":
                        "increases_contextual_risk",

                    "supports_class":
                        "phishing",

                    "description":
                        self._get_feature_description(
                            "button_count"
                        ),

                    "reason":
                        (
                            "Multiple interactive controls "
                            "were detected."
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Entropy.
        # ---------------------------------------------------------------------

        entropy = (
            self._feature_number(
                normalized_features,
                "image_entropy",
            )
        )

        if (
            entropy
            > self.ENTROPY_THRESHOLD
        ):

            top_factors.append(
                {
                    "feature":
                        "image_entropy",

                    "value":
                        round(
                            entropy,
                            6,
                        ),

                    "impact":
                        0.08,

                    "direction":
                        "increases_visual_complexity",

                    "supports_class":
                        "phishing",

                    "description":
                        self._get_feature_description(
                            "image_entropy"
                        ),

                    "reason":
                        (
                            "The screenshot contains relatively "
                            "high visual entropy."
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Sort.
        # ---------------------------------------------------------------------

        top_factors.sort(
            key=lambda item: abs(
                float(
                    item.get(
                        "impact",
                        0.0,
                    )
                )
            ),
            reverse=True,
        )

        top_factors = (
            top_factors[
                :self.default_top_k
            ]
        )

        # ---------------------------------------------------------------------
        # Baseline.
        # ---------------------------------------------------------------------

        if not top_factors:

            top_factors = [
                {
                    "feature":
                        "baseline_visual",

                    "value":
                        0.0,

                    "impact":
                        0.0,

                    "direction":
                        "neutral",

                    "supports_class":
                        prediction_label,

                    "description":
                        (
                            "No configured high-signal visual "
                            "anomalies were detected."
                        ),

                    "reason":
                        (
                            "The observed visual features did not "
                            "trigger the heuristic explanation rules."
                        ),
                }
            ]

        positive_factors = [
            factor
            for factor in top_factors
            if float(
                factor.get(
                    "impact",
                    0.0,
                )
            ) > 0
        ]

        if positive_factors:

            factor_names = [
                factor[
                    "feature"
                ]
                for factor
                in positive_factors[
                    :3
                ]
            ]

            summary = (
                f"Heuristic visual analysis classified the "
                f"layout as '{prediction_label}' with the "
                f"strongest observed indicators being: "
                f"{', '.join(factor_names)}."
            )

        else:

            summary = (
                f"Heuristic visual analysis found no strong "
                f"structural indicators for the "
                f"'{prediction_label}' result."
            )

        return {
            "explanation_mode":
                self.EXPLANATION_MODE_HEURISTIC,

            "summary":
                summary,

            "prediction":
                prediction_label,

            "prediction_class_index":
                self.CLASS_INDEX_MAP[
                    prediction_label
                ],

            "top_factors":
                top_factors,

            "positive_contributors":
                [
                    factor[
                        "feature"
                    ]
                    for factor
                    in positive_factors
                ],

            "negative_contributors":
                [],

            "shap_available":
                SHAP_AVAILABLE,

            "shap_initialized":
                self.shap_initialized,

            "explanation_reliability":
                "rule_based",
        }

    # =========================================================================
    # 23. SHAP EXPLANATION
    # =========================================================================

    def _generate_shap_explanation(
        self,
        features: Mapping[str, Any],
        prediction: Any,
        top_k: int,
    ) -> Dict[str, Any]:

        target_class_index, prediction_label = (
            self._normalize_prediction_class(
                prediction
            )
        )

        df_features = (
            self._prepare_feature_dataframe(
                features
            )
        )

        shap_output = (
            self._calculate_shap_values(
                df_features
            )
        )

        instance_shap = (
            self._extract_class_shap_vector(
                shap_output,
                target_class_index,
            )
        )

        instance_shap = np.asarray(
            instance_shap,
            dtype=float,
        ).reshape(
            -1
        )

        if len(
            instance_shap
        ) != len(
            self.feature_schema
        ):

            raise ValueError(
                "SHAP attribution count "
                f"{len(instance_shap)} does not match "
                f"Visual feature count "
                f"{len(self.feature_schema)}."
            )

        all_impacts = (
            self._build_shap_feature_impacts(
                instance_shap,
                df_features,
                prediction_label,
            )
        )

        top_factors = all_impacts[
            :top_k
        ]

        positive_contributors = [
            item
            for item
            in all_impacts
            if float(
                item[
                    "shap_value"
                ]
            ) > 0
        ]

        negative_contributors = [
            item
            for item
            in all_impacts
            if float(
                item[
                    "shap_value"
                ]
            ) < 0
        ]

        if positive_contributors:

            positive_names = [
                item[
                    "feature"
                ]
                for item
                in positive_contributors[
                    :3
                ]
            ]

            summary = (
                f"SHAP analysis indicates that the "
                f"'{prediction_label}' classification was "
                f"primarily supported by "
                f"{', '.join(positive_names)}."
            )

        elif negative_contributors:

            negative_names = [
                item[
                    "feature"
                ]
                for item
                in negative_contributors[
                    :3
                ]
            ]

            summary = (
                f"SHAP analysis found no strong positive "
                f"contributors to the '{prediction_label}' "
                f"classification. The strongest opposing "
                f"features were "
                f"{', '.join(negative_names)}."
            )

        else:

            summary = (
                f"SHAP analysis found no feature attribution "
                f"above the configured threshold for the "
                f"'{prediction_label}' classification."
            )

        base_value = (
            self._extract_base_value(
                shap_output,
                target_class_index,
            )
        )

        total_absolute_impact = float(
            np.sum(
                np.abs(
                    instance_shap
                )
            )
        )

        return {
            "explanation_mode":
                self.EXPLANATION_MODE_SHAP,

            "summary":
                summary,

            "prediction":
                prediction_label,

            "prediction_class_index":
                target_class_index,

            "top_factors":
                top_factors,

            "all_feature_impacts":
                all_impacts,

            "positive_contributors":
                [
                    item[
                        "feature"
                    ]
                    for item
                    in positive_contributors
                ],

            "negative_contributors":
                [
                    item[
                        "feature"
                    ]
                    for item
                    in negative_contributors
                ],

            "base_value":
                base_value,

            "total_absolute_shap_impact":
                round(
                    total_absolute_impact,
                    6,
                ),

            "shap_available":
                SHAP_AVAILABLE,

            "shap_initialized":
                self.shap_initialized,

            "explanation_reliability":
                "model_specific",

            "feature_count":
                len(
                    self.feature_schema
                ),

            "class_mapping":
                dict(
                    self.CLASS_LABELS
                ),
        }

    # =========================================================================
    # 24. PUBLIC EXPLANATION
    # =========================================================================

    def generate_explanation(
        self,
        features: Dict[str, Any],
        prediction: Any,
        *,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:

        try:

            requested_top_k = int(
                top_k
                if top_k is not None
                else self.default_top_k
            )

            if requested_top_k <= 0:

                raise ValueError(
                    "top_k must be greater than zero."
                )

            requested_top_k = min(
                requested_top_k,
                len(
                    self.feature_schema
                ),
            )

            _, prediction_label = (
                self._normalize_prediction_class(
                    prediction
                )
            )

            normalized_features = (
                self._normalize_features(
                    features
                )
            )

            # -----------------------------------------------------------------
            # SHAP path.
            # -----------------------------------------------------------------

            if (
                self.shap_initialized
                and self.explainer is not None
                and self.model is not None
            ):

                try:

                    return (
                        self._generate_shap_explanation(
                            normalized_features,
                            prediction_label,
                            requested_top_k,
                        )
                    )

                except Exception as exc:

                    logger.error(
                        "Visual SHAP explanation failed: %s",
                        exc,
                        exc_info=True,
                    )

                    heuristic_result = (
                        self._heuristic_explanation(
                            normalized_features,
                            prediction_label,
                        )
                    )

                    heuristic_result[
                        "fallback_reason"
                    ] = (
                        f"SHAP explanation failed: {exc}"
                    )

                    return heuristic_result

            # -----------------------------------------------------------------
            # Heuristic path.
            # -----------------------------------------------------------------

            return (
                self._heuristic_explanation(
                    normalized_features,
                    prediction_label,
                )
            )

        except Exception as exc:

            logger.error(
                "Visual explanation generation failed: %s",
                exc,
                exc_info=True,
            )

            return (
                self._error_explanation(
                    features,
                    prediction,
                    exc,
                )
            )

    # =========================================================================
    # 25. PREDICTOR RESULT EXPLANATION
    # =========================================================================

    def explain_prediction(
        self,
        prediction_result: Mapping[str, Any],
        *,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:

        if not isinstance(
            prediction_result,
            Mapping,
        ):

            raise TypeError(
                "prediction_result must be a mapping."
            )

        prediction = (
            prediction_result.get(
                "class_label"
            )
        )

        if prediction is None:

            prediction = (
                prediction_result.get(
                    "class_index",
                    0,
                )
            )

        features = (
            prediction_result.get(
                "features"
            )
        )

        # ---------------------------------------------------------------------
        # Backward-compatible feature fields.
        # ---------------------------------------------------------------------

        if not features:

            dataframe = (
                prediction_result.get(
                    "features_dataframe"
                )
            )

            if (
                isinstance(
                    dataframe,
                    pd.DataFrame,
                )
                and not dataframe.empty
            ):

                features = (
                    dataframe.iloc[
                        0
                    ].to_dict()
                )

        if not features:

            features = (
                prediction_result.get(
                    "normalized_features",
                    {}
                )
            )

        if not features:

            features = {}

        explanation = (
            self.generate_explanation(
                features=features,
                prediction=prediction,
                top_k=top_k,
            )
        )

        explanation[
            "phishing_probability"
        ] = self._safe_float(
            prediction_result.get(
                "phishing_probability",
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

        explanation[
            "model_based"
        ] = bool(
            prediction_result.get(
                "model_based",
                False,
            )
        )

        explanation[
            "fallback_used"
        ] = bool(
            prediction_result.get(
                "fallback_used",
                False,
            )
        )

        return explanation

    # =========================================================================
    # 26. ERROR FALLBACK
    # =========================================================================

    def _error_explanation(
        self,
        features: Any,
        prediction: Any,
        error: Exception,
    ) -> Dict[str, Any]:

        try:

            class_index, prediction_label = (
                self._normalize_prediction_class(
                    prediction
                )
            )

        except Exception:

            class_index = 0

            prediction_label = (
                "legitimate"
            )

        return {
            "explanation_mode":
                self.EXPLANATION_MODE_ERROR,

            "summary":
                (
                    "Visual explanation could not be generated. "
                    "The prediction should be interpreted using "
                    "the available model and risk outputs."
                ),

            "prediction":
                prediction_label,

            "prediction_class_index":
                class_index,

            "top_factors":
                [],

            "all_feature_impacts":
                [],

            "positive_contributors":
                [],

            "negative_contributors":
                [],

            "shap_available":
                SHAP_AVAILABLE,

            "shap_initialized":
                self.shap_initialized,

            "explanation_reliability":
                "unavailable",

            "error":
                str(
                    error
                ),
        }

    # =========================================================================
    # 27. TOP FACTORS
    # =========================================================================

    def get_top_factors(
        self,
        explanation: Mapping[str, Any],
        *,
        top_k: int = 5,
    ) -> List[
        Dict[str, Any]
    ]:

        if not isinstance(
            explanation,
            Mapping,
        ):

            return []

        try:

            top_k = max(
                1,
                int(
                    top_k
                ),
            )

        except (
            TypeError,
            ValueError,
        ):

            top_k = 5

        factors = (
            explanation.get(
                "top_factors",
                [],
            )
        )

        if not isinstance(
            factors,
            list,
        ):

            return []

        return factors[
            :top_k
        ]

    # =========================================================================
    # 28. POSITIVE CONTRIBUTORS
    # =========================================================================

    def get_positive_contributors(
        self,
        explanation: Mapping[str, Any],
    ) -> List[str]:

        if not isinstance(
            explanation,
            Mapping,
        ):

            return []

        contributors = (
            explanation.get(
                "positive_contributors",
                [],
            )
        )

        if not isinstance(
            contributors,
            list,
        ):

            return []

        return [
            str(
                contributor
            )
            for contributor
            in contributors
        ]

    # =========================================================================
    # 29. NEGATIVE CONTRIBUTORS
    # =========================================================================

    def get_negative_contributors(
        self,
        explanation: Mapping[str, Any],
    ) -> List[str]:

        if not isinstance(
            explanation,
            Mapping,
        ):

            return []

        contributors = (
            explanation.get(
                "negative_contributors",
                [],
            )
        )

        if not isinstance(
            contributors,
            list,
        ):

            return []

        return [
            str(
                contributor
            )
            for contributor
            in contributors
        ]

    # =========================================================================
    # 30. HUMAN SUMMARY
    # =========================================================================

    def build_human_summary(
        self,
        explanation: Mapping[str, Any],
    ) -> str:

        if not isinstance(
            explanation,
            Mapping,
        ):

            return (
                "Visual explanation unavailable."
            )

        summary = explanation.get(
            "summary"
        )

        if summary:

            return str(
                summary
            )

        prediction = explanation.get(
            "prediction",
            "unknown",
        )

        return (
            f"Visual explanation for '{prediction}' "
            "is unavailable."
        )

    # =========================================================================
    # 31. HEALTH CHECK
    # =========================================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        if (
            self.model is not None
            and self.shap_initialized
        ):

            status = "ready"

        elif (
            SHAP_AVAILABLE
            and self.model is not None
        ):

            status = (
                "heuristic_fallback"
            )

        else:

            status = (
                "heuristic_only"
            )

        return {
            "status":
                status,

            "component":
                "VisualExplainer",

            "model_available":
                self.model is not None,

            "shap_available":
                SHAP_AVAILABLE,

            "shap_initialized":
                self.shap_initialized,

            "shap_initialization_error":
                self.shap_initialization_error,

            "feature_count":
                len(
                    self.feature_schema
                ),

            "features":
                list(
                    self.feature_schema
                ),

            "classes":
                dict(
                    self.CLASS_LABELS
                ),

            "class_mapping":
                dict(
                    self.CLASS_INDEX_MAP
                ),

            "num_classes":
                self.NUM_CLASSES,

            "default_top_k":
                self.default_top_k,

            "minimum_shap_impact":
                self.min_absolute_shap_impact,
        }

    # =========================================================================
    # 32. CONFIGURATION
    # =========================================================================

    def get_configuration(
        self,
    ) -> Dict[str, Any]:

        return {
            "component":
                "VisualExplainer",

            "explanation_modes": {
                "shap":
                    self.EXPLANATION_MODE_SHAP,

                "heuristic":
                    self.EXPLANATION_MODE_HEURISTIC,

                "error":
                    self.EXPLANATION_MODE_ERROR,
            },

            "classification": {
                "num_classes":
                    self.NUM_CLASSES,

                "class_mapping":
                    dict(
                        self.CLASS_INDEX_MAP
                    ),

                "class_labels":
                    dict(
                        self.CLASS_LABELS
                    ),
            },

            "shap": {
                "available":
                    SHAP_AVAILABLE,

                "initialized":
                    self.shap_initialized,

                "minimum_absolute_impact":
                    self.min_absolute_shap_impact,
            },

            "top_k":
                self.default_top_k,

            "feature_schema":
                list(
                    self.feature_schema
                ),

            "heuristic_thresholds": {
                "form_area_ratio":
                    self.FORM_AREA_THRESHOLD,

                "image_density":
                    self.IMAGE_DENSITY_THRESHOLD,

                "blank_area_ratio":
                    self.BLANK_AREA_THRESHOLD,

                "layout_complexity":
                    self.LAYOUT_COMPLEXITY_THRESHOLD,

                "button_count":
                    self.BUTTON_COUNT_THRESHOLD,

                "image_entropy":
                    self.ENTROPY_THRESHOLD,
            },
        }
"""
Authoritative feature contract for the Visual AI Agent.

The Visual Agent uses exactly 12 numerical ML features:

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

This schema is shared by:

    VisualFeatureExtractor
            |
            v
    VisualFeatureSchema
            |
            v
    VisualPreprocessor
            |
            v
    VisualXGBoostModel
            |
            v
    VisualPredictor
            |
            v
    VisualRiskScorer
            |
            v
    VisualExplainer

Responsibilities
----------------
- Define canonical feature names.
- Define exact feature order.
- Resolve feature aliases.
- Provide default values.
- Provide human-readable feature descriptions.
- Identify missing features.
- Identify extra features.
- Normalize booleans.
- Normalize numerical values.
- Build the final 12-column DataFrame.
- Provide schema diagnostics.
- Maintain backward compatibility with existing project files.

Classification contract
-----------------------

    0 = legitimate
    1 = phishing

No standalone execution hook is included.
"""

from __future__ import annotations


# ============================================================================
# STANDARD LIBRARY
# ============================================================================

import logging
import math

from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
)


# ============================================================================
# THIRD-PARTY
# ============================================================================

import numpy as np
import pandas as pd

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# VISUAL FEATURE SCHEMA
# ============================================================================

class VisualFeatureSchema(BaseModel):
    """
    Canonical Visual AI Agent feature schema.

    The model contains the 12 features consumed by the Visual XGBoost
    classifier.

    IMPORTANT
    ---------

    FEATURE_ORDER is part of the ML model contract.

    Do not reorder these features after a model has been trained unless
    the model is retrained with the new order.
    """

    # ========================================================================
    # PYDANTIC CONFIGURATION
    # ========================================================================

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # ========================================================================
    # FEATURE 1
    # ========================================================================

    screenshot_width: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Width of the captured screenshot in pixels."
        ),
    )

    # ========================================================================
    # FEATURE 2
    # ========================================================================

    screenshot_height: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Height of the captured screenshot in pixels."
        ),
    )

    # ========================================================================
    # FEATURE 3
    # ========================================================================

    image_entropy: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Entropy of visual/image information."
        ),
    )

    # ========================================================================
    # FEATURE 4
    # ========================================================================

    dominant_color_count: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Number of dominant colors detected."
        ),
    )

    # ========================================================================
    # FEATURE 5
    # ========================================================================

    text_density: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated ratio of screenshot occupied by text."
        ),
    )

    # ========================================================================
    # FEATURE 6
    # ========================================================================

    image_density: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated ratio of screenshot occupied by images."
        ),
    )

    # ========================================================================
    # FEATURE 7
    # ========================================================================

    form_area_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated proportion of screenshot occupied by forms."
        ),
    )

    # ========================================================================
    # FEATURE 8
    # ========================================================================

    button_count: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Number of detected visual buttons."
        ),
    )

    # ========================================================================
    # FEATURE 9
    # ========================================================================

    logo_detected: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Binary indicator for detected visual logo."
        ),
    )

    # ========================================================================
    # FEATURE 10
    # ========================================================================

    login_form_detected: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Binary indicator for visually detected login form."
        ),
    )

    # ========================================================================
    # FEATURE 11
    # ========================================================================

    layout_complexity: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Numerical visual layout complexity score."
        ),
    )

    # ========================================================================
    # FEATURE 12
    # ========================================================================

    blank_area_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated ratio of blank/unused screenshot area."
        ),
    )

    # ========================================================================
    # SCHEMA METADATA
    # ========================================================================

    SCHEMA_VERSION: ClassVar[str] = "2.2.0"

    SCHEMA_NAME: ClassVar[str] = (
        "visual_agent_feature_schema"
    )

    DEFAULT_FEATURE_VALUE: ClassVar[float] = 0.0

    # ========================================================================
    # EXACT ML FEATURE ORDER
    # ========================================================================

    FEATURE_ORDER: ClassVar[List[str]] = [

        "screenshot_width",

        "screenshot_height",

        "image_entropy",

        "dominant_color_count",

        "text_density",

        "image_density",

        "form_area_ratio",

        "button_count",

        "logo_detected",

        "login_form_detected",

        "layout_complexity",

        "blank_area_ratio",
    ]

    # ========================================================================
    # DEFAULT VALUES
    # ========================================================================

    DEFAULT_FEATURES: ClassVar[Dict[str, float]] = {

        "screenshot_width":
            0.0,

        "screenshot_height":
            0.0,

        "image_entropy":
            0.0,

        "dominant_color_count":
            0.0,

        "text_density":
            0.0,

        "image_density":
            0.0,

        "form_area_ratio":
            0.0,

        "button_count":
            0.0,

        "logo_detected":
            0.0,

        "login_form_detected":
            0.0,

        "layout_complexity":
            0.0,

        "blank_area_ratio":
            0.0,
    }

    # ========================================================================
    # HUMAN-READABLE FEATURE DESCRIPTIONS
    # ========================================================================

    FEATURE_DESCRIPTIONS: ClassVar[Dict[str, str]] = {

        "screenshot_width": (
            "Width of the captured screenshot in pixels."
        ),

        "screenshot_height": (
            "Height of the captured screenshot in pixels."
        ),

        "image_entropy": (
            "Entropy of visual/image information."
        ),

        "dominant_color_count": (
            "Number of dominant colors detected in the screenshot."
        ),

        "text_density": (
            "Estimated ratio of the screenshot occupied by text."
        ),

        "image_density": (
            "Estimated ratio of the screenshot occupied by images."
        ),

        "form_area_ratio": (
            "Estimated proportion of the screenshot occupied by forms."
        ),

        "button_count": (
            "Number of detected visual buttons."
        ),

        "logo_detected": (
            "Binary indicator showing whether a visual logo was detected."
        ),

        "login_form_detected": (
            "Binary indicator showing whether a login or credential-entry "
            "form was visually detected."
        ),

        "layout_complexity": (
            "Numerical score representing the complexity of the visual "
            "page layout."
        ),

        "blank_area_ratio": (
            "Estimated ratio of blank or unused screenshot area."
        ),
    }

    # ========================================================================
    # BOOLEAN FEATURES
    # ========================================================================

    BOOLEAN_FEATURES: ClassVar[List[str]] = [

        "logo_detected",

        "login_form_detected",
    ]

    # ========================================================================
    # RATIO FEATURES
    # ========================================================================

    RATIO_FEATURES: ClassVar[List[str]] = [

        "text_density",

        "image_density",

        "form_area_ratio",

        "blank_area_ratio",
    ]

    # ========================================================================
    # NON-NEGATIVE FEATURES
    # ========================================================================

    NON_NEGATIVE_FEATURES: ClassVar[List[str]] = [

        "screenshot_width",

        "screenshot_height",

        "image_entropy",

        "dominant_color_count",

        "button_count",

        "layout_complexity",
    ]

    # ========================================================================
    # FEATURE ALIASES
    # ========================================================================

    FEATURE_ALIASES: ClassVar[Dict[str, str]] = {

        # --------------------------------------------------------------------
        # SCREENSHOT WIDTH
        # --------------------------------------------------------------------

        "width":
            "screenshot_width",

        "screenshot_w":
            "screenshot_width",

        "screen_width":
            "screenshot_width",

        "viewport_width":
            "screenshot_width",

        "image_width":
            "screenshot_width",

        "capture_width":
            "screenshot_width",

        "screenshot_width_px":
            "screenshot_width",

        # --------------------------------------------------------------------
        # SCREENSHOT HEIGHT
        # --------------------------------------------------------------------

        "height":
            "screenshot_height",

        "screenshot_h":
            "screenshot_height",

        "screen_height":
            "screenshot_height",

        "viewport_height":
            "screenshot_height",

        "image_height":
            "screenshot_height",

        "capture_height":
            "screenshot_height",

        "screenshot_height_px":
            "screenshot_height",

        # --------------------------------------------------------------------
        # IMAGE ENTROPY
        # --------------------------------------------------------------------

        "entropy":
            "image_entropy",

        "visual_entropy":
            "image_entropy",

        "image_complexity":
            "image_entropy",

        "visual_image_entropy":
            "image_entropy",

        "screenshot_entropy":
            "image_entropy",

        # --------------------------------------------------------------------
        # DOMINANT COLORS
        # --------------------------------------------------------------------

        "color_count":
            "dominant_color_count",

        "colors_count":
            "dominant_color_count",

        "dominant_colors_count":
            "dominant_color_count",

        "dominant_color_num":
            "dominant_color_count",

        "dominant_colour_count":
            "dominant_color_count",

        "num_dominant_colors":
            "dominant_color_count",

        # --------------------------------------------------------------------
        # TEXT DENSITY
        # --------------------------------------------------------------------

        "text_ratio":
            "text_density",

        "text_area_ratio":
            "text_density",

        "visual_text_density":
            "text_density",

        "ocr_text_density":
            "text_density",

        "text_percentage":
            "text_density",

        # --------------------------------------------------------------------
        # IMAGE DENSITY
        # --------------------------------------------------------------------

        "image_ratio":
            "image_density",

        "image_area_ratio":
            "image_density",

        "visual_image_density":
            "image_density",

        "image_percentage":
            "image_density",

        # --------------------------------------------------------------------
        # FORM AREA
        # --------------------------------------------------------------------

        "form_ratio":
            "form_area_ratio",

        "form_density":
            "form_area_ratio",

        "form_area_percentage":
            "form_area_ratio",

        "visual_form_area_ratio":
            "form_area_ratio",

        # --------------------------------------------------------------------
        # BUTTON COUNT
        # --------------------------------------------------------------------

        "num_buttons":
            "button_count",

        "buttons_count":
            "button_count",

        "button_count_visual":
            "button_count",

        "visual_button_count":
            "button_count",

        "detected_button_count":
            "button_count",

        # --------------------------------------------------------------------
        # LOGO DETECTION
        # --------------------------------------------------------------------

        "has_logo":
            "logo_detected",

        "logo_present":
            "logo_detected",

        "logo_detected_visual":
            "logo_detected",

        "visual_logo_detected":
            "logo_detected",

        "has_visual_logo":
            "logo_detected",

        # --------------------------------------------------------------------
        # LOGIN FORM
        # --------------------------------------------------------------------

        "has_login_form":
            "login_form_detected",

        "has_login_form_visual":
            "login_form_detected",

        "login_form_visual":
            "login_form_detected",

        "login_detected":
            "login_form_detected",

        "visual_login_form_detected":
            "login_form_detected",

        "login_form_present":
            "login_form_detected",

        "login_form":
            "login_form_detected",

        # --------------------------------------------------------------------
        # LAYOUT COMPLEXITY
        # --------------------------------------------------------------------

        "layout_complexity_score":
            "layout_complexity",

        "visual_layout_complexity":
            "layout_complexity",

        "layout_score":
            "layout_complexity",

        "complexity_score":
            "layout_complexity",

        "visual_complexity":
            "layout_complexity",

        # --------------------------------------------------------------------
        # BLANK AREA
        # --------------------------------------------------------------------

        "whitespace_ratio":
            "blank_area_ratio",

        "white_space_ratio":
            "blank_area_ratio",

        "blank_ratio":
            "blank_area_ratio",

        "empty_area_ratio":
            "blank_area_ratio",

        "empty_space_ratio":
            "blank_area_ratio",

        "whitespace_area_ratio":
            "blank_area_ratio",

        "blank_space_ratio":
            "blank_area_ratio",
    }

    # =========================================================================
    # SCHEMA INTEGRITY
    # =========================================================================

    @classmethod
    def verify_schema_integrity(
        cls,
    ) -> bool:
        """
        Verify that the schema configuration is internally consistent.
        """

        # ---------------------------------------------------------------------
        # Feature count
        # ---------------------------------------------------------------------

        if len(
            cls.FEATURE_ORDER
        ) != 12:

            raise RuntimeError(
                (
                    "Visual feature schema must contain "
                    f"exactly 12 features, got "
                    f"{len(cls.FEATURE_ORDER)}."
                )
            )

        # ---------------------------------------------------------------------
        # Duplicate detection
        # ---------------------------------------------------------------------

        if len(
            set(
                cls.FEATURE_ORDER
            )
        ) != len(
            cls.FEATURE_ORDER
        ):

            raise RuntimeError(
                "Visual feature schema contains duplicate features."
            )

        # ---------------------------------------------------------------------
        # Defaults
        # ---------------------------------------------------------------------

        missing_defaults = [
            feature
            for feature in cls.FEATURE_ORDER
            if feature
            not in cls.DEFAULT_FEATURES
        ]

        if missing_defaults:

            raise RuntimeError(
                (
                    "Visual schema features missing defaults: "
                    f"{missing_defaults}"
                )
            )

        # ---------------------------------------------------------------------
        # Descriptions
        # ---------------------------------------------------------------------

        missing_descriptions = [
            feature
            for feature in cls.FEATURE_ORDER
            if feature
            not in cls.FEATURE_DESCRIPTIONS
        ]

        if missing_descriptions:

            raise RuntimeError(
                (
                    "Visual schema features missing descriptions: "
                    f"{missing_descriptions}"
                )
            )

        # ---------------------------------------------------------------------
        # Boolean feature membership
        # ---------------------------------------------------------------------

        for feature in cls.BOOLEAN_FEATURES:

            if feature not in cls.FEATURE_ORDER:

                raise RuntimeError(
                    (
                        f"Boolean feature '{feature}' "
                        "is not present in FEATURE_ORDER."
                    )
                )

        # ---------------------------------------------------------------------
        # Ratio feature membership
        # ---------------------------------------------------------------------

        for feature in cls.RATIO_FEATURES:

            if feature not in cls.FEATURE_ORDER:

                raise RuntimeError(
                    (
                        f"Ratio feature '{feature}' "
                        "is not present in FEATURE_ORDER."
                    )
                )

        # ---------------------------------------------------------------------
        # Non-negative feature membership
        # ---------------------------------------------------------------------

        for feature in cls.NON_NEGATIVE_FEATURES:

            if feature not in cls.FEATURE_ORDER:

                raise RuntimeError(
                    (
                        f"Non-negative feature '{feature}' "
                        "is not present in FEATURE_ORDER."
                    )
                )

        return True

    # =========================================================================
    # GET SCHEMA
    # =========================================================================

    @classmethod
    def get_schema(
        cls,
    ) -> List[str]:
        """
        Return the canonical 12-feature ML schema.
        """

        cls.verify_schema_integrity()

        return list(
            cls.FEATURE_ORDER
        )

    # =========================================================================
    # GET SCHEMA COLUMNS
    # =========================================================================

    @classmethod
    def get_schema_columns(
        cls,
    ) -> List[str]:
        """
        Return the canonical feature columns.
        """

        return list(
            cls.FEATURE_ORDER
        )

    # =========================================================================
    # GET FEATURE NAMES
    # =========================================================================

    @classmethod
    def get_feature_names(
        cls,
    ) -> List[str]:
        """
        Compatibility alias for modules that request feature names.
        """

        return list(
            cls.FEATURE_ORDER
        )

    # =========================================================================
    # GET FEATURE COUNT
    # =========================================================================

    @classmethod
    def get_feature_count(
        cls,
    ) -> int:
        """
        Return the number of ML features.
        """

        return len(
            cls.FEATURE_ORDER
        )

    # =========================================================================
    # GET DEFAULT FEATURES
    # =========================================================================

    @classmethod
    def get_default_features(
        cls,
    ) -> Dict[str, float]:
        """
        Return a fresh copy of the schema defaults.
        """

        return dict(
            cls.DEFAULT_FEATURES
        )

    # =========================================================================
    # GET FEATURE DESCRIPTIONS
    # =========================================================================

    @classmethod
    def get_descriptions(
        cls,
    ) -> Dict[str, str]:
        """
        Return human-readable descriptions for all canonical Visual
        ML features.

        This method is required by VisualExplainer.

        VisualExplainer calls:

            VisualFeatureSchema.get_descriptions()

        during SHAP explanation generation.
        """

        cls.verify_schema_integrity()

        return dict(
            cls.FEATURE_DESCRIPTIONS
        )

    # =========================================================================
    # GET SINGLE FEATURE DESCRIPTION
    # =========================================================================

    @classmethod
    def get_feature_description(
        cls,
        feature_name: Any,
    ) -> str:
        """
        Return the description for one feature.

        Aliases are resolved automatically.
        """

        canonical_name = (
            cls.resolve_feature_name(
                feature_name
            )
        )

        if canonical_name is None:

            normalized_name = (
                cls._normalize_feature_name(
                    feature_name
                )
            )

            return (
                normalized_name
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

        return cls.FEATURE_DESCRIPTIONS.get(
            canonical_name,
            canonical_name.replace(
                "_",
                " ",
            ).title(),
        )

    # =========================================================================
    # GET SCHEMA METADATA
    # =========================================================================

    @classmethod
    def get_schema_metadata(
        cls,
    ) -> Dict[str, Any]:
        """
        Return descriptive metadata about the Visual schema.
        """

        cls.verify_schema_integrity()

        return {

            "schema_name":
                cls.SCHEMA_NAME,

            "schema_version":
                cls.SCHEMA_VERSION,

            "agent":
                "Visual_AI_Agent",

            "classification_contract": {
                "0":
                    "legitimate",
                "1":
                    "phishing",
            },

            "feature_count":
                len(
                    cls.FEATURE_ORDER
                ),

            "feature_order":
                list(
                    cls.FEATURE_ORDER
                ),

            "features":
                list(
                    cls.FEATURE_ORDER
                ),

            "descriptions":
                dict(
                    cls.FEATURE_DESCRIPTIONS
                ),

            "default_value":
                cls.DEFAULT_FEATURE_VALUE,

            "defaults":
                dict(
                    cls.DEFAULT_FEATURES
                ),

            "boolean_features":
                list(
                    cls.BOOLEAN_FEATURES
                ),

            "ratio_features":
                list(
                    cls.RATIO_FEATURES
                ),

            "non_negative_features":
                list(
                    cls.NON_NEGATIVE_FEATURES
                ),

            "alias_count":
                len(
                    cls.FEATURE_ALIASES
                ),

            "strict_column_order":
                True,

            "unknown_features_ignored":
                True,
        }

    # =========================================================================
    # NORMALIZE FEATURE NAME
    # =========================================================================

    @classmethod
    def _normalize_feature_name(
        cls,
        feature_name: Any,
    ) -> str:
        """
        Normalize feature-name formatting.
        """

        return (
            str(
                feature_name
            )
            .strip()
            .lower()
            .replace(
                "-",
                "_",
            )
            .replace(
                " ",
                "_",
            )
        )

    # =========================================================================
    # RESOLVE FEATURE NAME
    # =========================================================================

    @classmethod
    def resolve_feature_name(
        cls,
        feature_name: Any,
    ) -> Optional[str]:
        """
        Resolve a raw feature name to a canonical feature.

        Returns
        -------
        str or None
            Canonical feature name, or None if the feature is unknown.
        """

        if feature_name is None:

            return None

        normalized_name = (
            cls._normalize_feature_name(
                feature_name
            )
        )

        # ---------------------------------------------------------------------
        # Already canonical
        # ---------------------------------------------------------------------

        if normalized_name in (
            cls.FEATURE_ORDER
        ):

            return normalized_name

        # ---------------------------------------------------------------------
        # Alias
        # ---------------------------------------------------------------------

        return cls.FEATURE_ALIASES.get(
            normalized_name
        )

    # =========================================================================
    # NORMALIZE FEATURES
    # =========================================================================

    @classmethod
    def normalize_features(
        cls,
        raw_features: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Normalize raw feature names into canonical names.

        Unknown fields are retained for diagnostics and telemetry.
        """

        if raw_features is None:

            return {}

        if not isinstance(
            raw_features,
            Mapping,
        ):

            raise TypeError(
                (
                    "raw_features must be "
                    "a dictionary or Mapping."
                )
            )

        normalized: Dict[str, Any] = {}

        for raw_name, value in (
            raw_features.items()
        ):

            canonical_name = (
                cls.resolve_feature_name(
                    raw_name
                )
            )

            # ---------------------------------------------------------------
            # Unknown field
            # ---------------------------------------------------------------

            if canonical_name is None:

                normalized[
                    str(
                        raw_name
                    )
                ] = value

                continue

            normalized_raw_name = (
                cls._normalize_feature_name(
                    raw_name
                )
            )

            # ---------------------------------------------------------------
            # Prefer explicit canonical names over aliases.
            # ---------------------------------------------------------------

            if (
                canonical_name in normalized
                and normalized_raw_name
                != canonical_name
            ):

                continue

            normalized[
                canonical_name
            ] = value

        return normalized

    # =========================================================================
    # GET MISSING FEATURES
    # =========================================================================

    @classmethod
    def get_missing_features(
        cls,
        raw_features: Optional[
            Mapping[str, Any]
        ],
    ) -> List[str]:
        """
        Return canonical ML features absent from the input.
        """

        normalized = (
            cls.normalize_features(
                raw_features
            )
        )

        return [
            feature
            for feature
            in cls.FEATURE_ORDER
            if feature
            not in normalized
        ]

    # =========================================================================
    # GET EXTRA FEATURES
    # =========================================================================

    @classmethod
    def get_extra_features(
        cls,
        raw_features: Optional[
            Mapping[str, Any]
        ],
    ) -> List[str]:
        """
        Return normalized fields that are not part of the 12-feature
        ML contract.
        """

        normalized = (
            cls.normalize_features(
                raw_features
            )
        )

        return sorted(
            [
                feature
                for feature
                in normalized
                if feature
                not in cls.FEATURE_ORDER
            ]
        )

    # =========================================================================
    # SAFE BOOLEAN
    # =========================================================================

    @classmethod
    def _safe_boolean(
        cls,
        value: Any,
    ) -> float:
        """
        Convert common boolean representations to 0.0 or 1.0.
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

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value
                .strip()
                .lower()
            )

            if normalized in {
                "true",
                "yes",
                "y",
                "1",
                "on",
                "detected",
                "present",
            }:

                return 1.0

            if normalized in {
                "false",
                "no",
                "n",
                "0",
                "off",
                "not_detected",
                "absent",
                "",
                "none",
                "null",
            }:

                return 0.0

        try:

            numeric = float(
                value
            )

            if not math.isfinite(
                numeric
            ):

                return 0.0

            return (
                1.0
                if numeric > 0.0
                else 0.0
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    # =========================================================================
    # SAFE NUMERIC
    # =========================================================================

    @classmethod
    def _safe_numeric(
        cls,
        feature_name: str,
        value: Any,
    ) -> float:
        """
        Convert a feature value into a finite numerical value.
        """

        default = float(
            cls.DEFAULT_FEATURES.get(
                feature_name,
                cls.DEFAULT_FEATURE_VALUE,
            )
        )

        # ---------------------------------------------------------------------
        # Boolean features
        # ---------------------------------------------------------------------

        if feature_name in (
            cls.BOOLEAN_FEATURES
        ):

            return cls._safe_boolean(
                value
            )

        # ---------------------------------------------------------------------
        # Numeric conversion
        # ---------------------------------------------------------------------

        try:

            numeric = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            logger.warning(
                (
                    "Visual feature '%s' "
                    "received invalid value %r. "
                    "Using default %.4f."
                ),
                feature_name,
                value,
                default,
            )

            return default

        # ---------------------------------------------------------------------
        # NaN / Infinity
        # ---------------------------------------------------------------------

        if not math.isfinite(
            numeric
        ):

            logger.warning(
                (
                    "Visual feature '%s' "
                    "received NaN/Infinity. "
                    "Using default %.4f."
                ),
                feature_name,
                default,
            )

            return default

        # ---------------------------------------------------------------------
        # Ratio features
        # ---------------------------------------------------------------------

        if feature_name in (
            cls.RATIO_FEATURES
        ):

            numeric = float(
                np.clip(
                    numeric,
                    0.0,
                    1.0,
                )
            )

        # ---------------------------------------------------------------------
        # Non-negative features
        # ---------------------------------------------------------------------

        if feature_name in (
            cls.NON_NEGATIVE_FEATURES
        ):

            numeric = max(
                0.0,
                numeric,
            )

        return numeric

    # =========================================================================
    # ALIGN AND VALIDATE
    # =========================================================================

    @classmethod
    def align_and_validate(
        cls,
        raw_features: Optional[
            Mapping[str, Any]
        ],
        strict: bool = False,
        allow_unknown: bool = True,
    ) -> pd.DataFrame:
        """
        Convert raw Visual features into the exact 12-feature ML DataFrame.

        Missing features receive schema defaults when strict=False.

        The resulting DataFrame always contains exactly 12 columns in the
        trained feature order.
        """

        if raw_features is None:

            raw_features = {}

        if not isinstance(
            raw_features,
            Mapping,
        ):

            raise TypeError(
                (
                    "raw_features must be "
                    "a dictionary or Mapping."
                )
            )

        # ====================================================================
        # NORMALIZE
        # ====================================================================

        normalized = (
            cls.normalize_features(
                raw_features
            )
        )

        # ====================================================================
        # MISSING
        # ====================================================================

        missing_features = [
            feature
            for feature
            in cls.FEATURE_ORDER
            if feature
            not in normalized
        ]

        if (
            strict
            and missing_features
        ):

            raise ValueError(
                (
                    "Visual feature schema is incomplete. "
                    "Missing features: "
                    f"{missing_features}"
                )
            )

        # ====================================================================
        # EXTRA
        # ====================================================================

        extra_features = [
            feature
            for feature
            in normalized
            if feature
            not in cls.FEATURE_ORDER
        ]

        if (
            extra_features
            and not allow_unknown
        ):

            raise ValueError(
                (
                    "Unknown Visual features received: "
                    f"{sorted(extra_features)}"
                )
            )

        # ====================================================================
        # BUILD ALIGNED VECTOR
        # ====================================================================

        aligned: Dict[str, float] = (
            cls.get_default_features()
        )

        for feature_name in (
            cls.FEATURE_ORDER
        ):

            if feature_name in normalized:

                aligned[
                    feature_name
                ] = cls._safe_numeric(
                    feature_name,
                    normalized[
                        feature_name
                    ],
                )

        # ====================================================================
        # DATAFRAME
        # ====================================================================

        dataframe = pd.DataFrame(
            [
                [
                    aligned[
                        feature_name
                    ]
                    for feature_name
                    in cls.FEATURE_ORDER
                ]
            ],
            columns=cls.FEATURE_ORDER,
        )

        # ====================================================================
        # INVALID VALUES
        # ====================================================================

        dataframe = (
            dataframe.replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

        dataframe = (
            dataframe.fillna(
                0.0
            )
        )

        # ====================================================================
        # FINAL FLOAT TYPE
        # ====================================================================

        dataframe = (
            dataframe.astype(
                float
            )
        )

        # ====================================================================
        # FINAL COLUMN ORDER
        # ====================================================================

        dataframe = (
            dataframe[
                cls.FEATURE_ORDER
            ]
        )

        # ====================================================================
        # FINAL VALIDATION
        # ====================================================================

        if dataframe.shape[1] != 12:

            raise RuntimeError(
                (
                    "Visual ML DataFrame must contain "
                    f"12 features, got "
                    f"{dataframe.shape[1]}."
                )
            )

        if list(
            dataframe.columns
        ) != list(
            cls.FEATURE_ORDER
        ):

            raise RuntimeError(
                (
                    "Visual ML feature ordering "
                    "does not match FEATURE_ORDER."
                )
            )

        if not np.isfinite(
            dataframe.to_numpy(
                dtype=float
            )
        ).all():

            raise RuntimeError(
                "Visual ML DataFrame contains non-finite values."
            )

        return dataframe

    # =========================================================================
    # VALIDATION REPORT
    # =========================================================================

    @classmethod
    def get_validation_report(
        cls,
        raw_features: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Produce a detailed schema diagnostic report.
        """

        if raw_features is None:

            raw_features = {}

        if not isinstance(
            raw_features,
            Mapping,
        ):

            return {

                "valid":
                    False,

                "schema_complete":
                    False,

                "model_ready":
                    False,

                "expected_feature_count":
                    len(
                        cls.FEATURE_ORDER
                    ),

                "received_feature_count":
                    0,

                "canonical_feature_count":
                    0,

                "missing_features":
                    list(
                        cls.FEATURE_ORDER
                    ),

                "extra_features":
                    [],

                "alias_resolutions":
                    {},

                "error":
                    (
                        "raw_features must be "
                        "a Mapping."
                    ),
            }

        normalized = (
            cls.normalize_features(
                raw_features
            )
        )

        missing = (
            cls.get_missing_features(
                raw_features
            )
        )

        extra = (
            cls.get_extra_features(
                raw_features
            )
        )

        # ====================================================================
        # ALIAS RESOLUTION REPORT
        # ====================================================================

        alias_resolutions: Dict[
            str,
            str,
        ] = {}

        for raw_name in (
            raw_features.keys()
        ):

            canonical = (
                cls.resolve_feature_name(
                    raw_name
                )
            )

            normalized_raw = (
                cls._normalize_feature_name(
                    raw_name
                )
            )

            if (
                canonical
                and canonical
                != normalized_raw
            ):

                alias_resolutions[
                    str(
                        raw_name
                    )
                ] = canonical

        canonical_present = [
            feature
            for feature
            in cls.FEATURE_ORDER
            if feature
            in normalized
        ]

        return {

            "valid":
                True,

            "schema_name":
                cls.SCHEMA_NAME,

            "schema_version":
                cls.SCHEMA_VERSION,

            "expected_feature_count":
                len(
                    cls.FEATURE_ORDER
                ),

            "received_feature_count":
                len(
                    raw_features
                ),

            "canonical_feature_count":
                len(
                    canonical_present
                ),

            "canonical_features":
                list(
                    cls.FEATURE_ORDER
                ),

            "canonical_features_present":
                canonical_present,

            "missing_features":
                missing,

            "missing_feature_count":
                len(
                    missing
                ),

            "extra_features":
                extra,

            "extra_feature_count":
                len(
                    extra
                ),

            "alias_resolutions":
                alias_resolutions,

            "schema_complete":
                len(
                    missing
                ) == 0,

            "model_ready":
                True,
        }

    # =========================================================================
    # FEATURE VECTOR
    # =========================================================================

    @classmethod
    def to_feature_vector(
        cls,
        raw_features: Optional[
            Mapping[str, Any]
        ],
    ) -> List[float]:
        """
        Convert raw features to an ordered 12-value numerical list.
        """

        dataframe = (
            cls.align_and_validate(
                raw_features
            )
        )

        return [
            float(
                value
            )
            for value
            in dataframe.iloc[
                0
            ].tolist()
        ]
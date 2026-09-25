"""
agents/visual_agent/preprocessing.py

Production-grade preprocessing layer for the Visual AI Agent.

Architecture
------------

    Screenshot
        |
        v
    Visual Feature Extractor
        |
        v
    Raw Visual Features
        |
        v
    VisualFeatureSchema
        |
        v
    VisualPreprocessor
        |
        +--> Missing-value handling
        +--> Infinite-value handling
        +--> Numeric conversion
        +--> Outlier capping
        +--> Boolean normalization
        +--> Exact feature ordering
        +--> Final validation
        |
        v
    ML Model
        |
        v
    Visual Risk Score


The preprocessor is intentionally designed around the definitive
VisualFeatureSchema.

The ML model currently expects exactly 12 visual features:

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

IMPORTANT
---------
The preprocessing logic used during model training should match the
preprocessing logic used during live inference.

If the training pipeline caps a feature, the inference pipeline should
apply the same cap.

If the training pipeline converts values to float64, the inference
pipeline should do the same.

This prevents training/inference distribution mismatch.
"""

from __future__ import annotations

import logging
import math
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .feature_schema import VisualFeatureSchema


logger = logging.getLogger(__name__)


class VisualPreprocessor:
    """
    Production preprocessing engine for the Visual AI Agent.

    Responsibilities
    ----------------

    1. Schema alignment
    2. Feature-name normalization
    3. Alias resolution
    4. Missing-column handling
    5. Missing-value imputation
    6. Infinite-value handling
    7. Numeric type conversion
    8. Boolean normalization
    9. Feature-specific outlier capping
    10. Exact feature ordering
    11. Final ML-input validation
    12. Preprocessing diagnostics

    The preprocessor does NOT perform:

        - screenshot extraction
        - HTML parsing
        - DNS analysis
        - SSL analysis
        - URL analysis
        - ML prediction
        - risk scoring

    Those responsibilities belong to other components of the Visual
    AI Agent architecture.
    """

    # =========================================================================
    # 1. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        *,
        enable_outlier_capping: bool = True,
        strict_validation: bool = False,
    ) -> None:
        """
        Initialize the VisualPreprocessor.

        Parameters
        ----------
        enable_outlier_capping:
            Whether feature-specific maximum caps should be applied.

        strict_validation:
            Controls semantic validation behavior.

            False:
                Invalid semantic values are logged as warnings.

            True:
                Invalid semantic values raise ValueError.

        Notes
        -----
        The defaults are intentionally suitable for live inference.
        """

        self.enable_outlier_capping = bool(
            enable_outlier_capping
        )

        self.strict_validation = bool(
            strict_validation
        )

        # ---------------------------------------------------------------------
        # Feature-specific upper bounds.
        #
        # These limits prevent extreme extracted values from dominating
        # downstream model decisions.
        #
        # They are preprocessing safety bounds, not claims about what the
        # feature extractor is mathematically capable of producing.
        # ---------------------------------------------------------------------

        self.outlier_caps: Dict[str, float] = {
            # Screenshot dimensions
            "screenshot_width": 7680.0,
            "screenshot_height": 10000.0,

            # Visual complexity
            "image_entropy": 10.0,

            # Visual counts
            "dominant_color_count": 100.0,
            "button_count": 50.0,

            # Layout complexity
            "layout_complexity": 100.0,

            # Ratio features
            "text_density": 1.0,
            "image_density": 1.0,
            "form_area_ratio": 1.0,
            "blank_area_ratio": 1.0,
        }

        # ---------------------------------------------------------------------
        # Boolean features are naturally bounded between 0 and 1.
        # ---------------------------------------------------------------------

        self.boolean_features = set(
            VisualFeatureSchema.BOOLEAN_FEATURES
        )

        # ---------------------------------------------------------------------
        # Ratio features should normally be between 0 and 1.
        # ---------------------------------------------------------------------

        self.ratio_features = set(
            VisualFeatureSchema.RATIO_FEATURES
        )

        # ---------------------------------------------------------------------
        # Store the expected schema locally as an immutable tuple.
        #
        # This prevents accidental modification of the expected order
        # during runtime.
        # ---------------------------------------------------------------------

        self._schema: tuple[str, ...] = tuple(
            VisualFeatureSchema.get_schema()
        )

        logger.debug(
            "VisualPreprocessor initialized. "
            "features=%d capping=%s strict=%s",
            len(self._schema),
            self.enable_outlier_capping,
            self.strict_validation,
        )

    # =========================================================================
    # 2. PUBLIC CONFIGURATION METHODS
    # =========================================================================

    def get_schema(self) -> List[str]:
        """
        Return the exact feature order used by this preprocessor.
        """

        return list(self._schema)

    def get_outlier_caps(self) -> Dict[str, float]:
        """
        Return a copy of the active feature caps.
        """

        return dict(self.outlier_caps)

    def get_configuration(self) -> Dict[str, Any]:
        """
        Return the current preprocessing configuration.
        """

        return {
            "feature_count": len(self._schema),
            "features": list(self._schema),
            "enable_outlier_capping": (
                self.enable_outlier_capping
            ),
            "strict_validation": (
                self.strict_validation
            ),
            "outlier_caps": dict(
                self.outlier_caps
            ),
        }

    # =========================================================================
    # 3. SINGLE-ROW TRANSFORMATION
    # =========================================================================

    def transform_single(
        self,
        raw_features: Optional[Mapping[str, Any]],
    ) -> pd.DataFrame:
        """
        Transform one raw Visual Feature Extractor output.

        Parameters
        ----------
        raw_features:
            Dictionary-like object containing raw visual telemetry.

        Returns
        -------
        pandas.DataFrame
            A one-row DataFrame containing exactly the 12 model features.

        Processing pipeline
        -------------------

            Raw dictionary
                ↓
            Schema alignment
                ↓
            Numeric conversion
                ↓
            Missing/NaN handling
                ↓
            Boolean normalization
                ↓
            Outlier capping
                ↓
            Exact column ordering
                ↓
            Final validation
                ↓
            ML-ready DataFrame

        Notes
        -----
        This method does not perform model prediction.
        """

        try:

            # -----------------------------------------------------------------
            # 1. Validate basic input
            # -----------------------------------------------------------------

            if raw_features is not None and not isinstance(
                raw_features,
                Mapping,
            ):
                raise TypeError(
                    "raw_features must be a mapping/dictionary. "
                    f"Received: {type(raw_features).__name__}"
                )

            # -----------------------------------------------------------------
            # 2. Use the definitive schema for structural alignment.
            #
            # Missing features are filled by the schema.
            # Aliases are resolved by the schema.
            # -----------------------------------------------------------------

            df_aligned = (
                VisualFeatureSchema.align_and_validate(
                    raw_features,
                    strict=self.strict_validation,
                    allow_unknown=True,
                )
            )

            # -----------------------------------------------------------------
            # 3. Apply numerical preprocessing.
            # -----------------------------------------------------------------

            df_processed = self._apply_transformations(
                df_aligned
            )

            # -----------------------------------------------------------------
            # 4. Final validation.
            # -----------------------------------------------------------------

            self._validate_final_dataframe(
                df_processed,
                expect_single_row=True,
            )

            logger.debug(
                "Single Visual feature vector "
                "preprocessed successfully."
            )

            return df_processed

        except Exception as exc:

            logger.error(
                "Failed to preprocess single Visual "
                "feature vector: %s",
                exc,
                exc_info=True,
            )

            # -------------------------------------------------------------
            # Safe fallback.
            #
            # We return a structurally valid 1-row DataFrame rather than
            # returning an empty DataFrame whose columns may not match the
            # model contract.
            # -------------------------------------------------------------

            return self._create_safe_fallback_dataframe()

    # =========================================================================
    # 4. BATCH TRANSFORMATION
    # =========================================================================

    def transform_batch(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform a batch DataFrame into model-ready Visual features.

        IMPORTANT:
        Only the 12 canonical ML features are processed. Source metadata
        columns (for example image_width/image_height) are deliberately
        excluded before alias normalization. This prevents metadata aliases
        from colliding with screenshot_width/screenshot_height.

        Returns exactly the 12 canonical features in schema order.
        """

        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError(
                "df_raw must be a pandas DataFrame."
            )

        if df_raw.empty:
            raise ValueError(
                "Visual training DataFrame is empty."
            )

        try:
            logger.debug(
                "Starting Visual batch preprocessing. "
                "Rows=%d, Columns=%d",
                len(df_raw),
                len(df_raw.columns),
            )

            # -------------------------------------------------------------
            # CRITICAL FIX:
            # Select canonical ML features BEFORE alias normalization.
            #
            # The CSV also contains metadata:
            #     image_width
            #     image_height
            #
            # VisualFeatureSchema can resolve those aliases to:
            #     screenshot_width
            #     screenshot_height
            #
            # If metadata is passed through normalization together with the
            # canonical columns, duplicate feature names are created.
            # -------------------------------------------------------------

            canonical_features = list(self._schema)

            missing_features = [
                feature
                for feature in canonical_features
                if feature not in df_raw.columns
            ]

            if missing_features:
                raise ValueError(
                    "Visual batch is missing required canonical "
                    f"features: {missing_features}"
                )

            df_canonical = df_raw[
                canonical_features
            ].copy(deep=True)

            # -------------------------------------------------------------
            # Align to the exact canonical schema.
            # -------------------------------------------------------------

            df_aligned = self._align_batch_schema(
                df_canonical
            )

            # -------------------------------------------------------------
            # Apply numeric/boolean/ratio transformations.
            # -------------------------------------------------------------

            df_processed = self._apply_transformations(
                df_aligned
            )

            # -------------------------------------------------------------
            # Force exact canonical order and numeric dtype.
            # -------------------------------------------------------------

            df_processed = df_processed[
                canonical_features
            ].copy()

            for feature in canonical_features:
                df_processed[feature] = pd.to_numeric(
                    df_processed[feature],
                    errors="coerce",
                ).astype("float64")

            # -------------------------------------------------------------
            # Final validation.
            # -------------------------------------------------------------

            self._validate_final_dataframe(
                df_processed,
                expect_single_row=False,
            )

            logger.debug(
                "Visual batch preprocessing completed successfully. "
                "Rows=%d, Columns=%d",
                len(df_processed),
                len(df_processed.columns),
            )

            return df_processed

        except Exception as exc:
            logger.error(
                "Failed to preprocess Visual batch dataset: %s",
                exc,
                exc_info=True,
            )
            raise

    # =========================================================================
    # 5. DATAFRAME COLUMN NORMALIZATION
    # =========================================================================

    def _normalize_dataframe_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize DataFrame column names using VisualFeatureSchema.

        Examples:

            width
                →
            screenshot_width

            has_logo
                →
            logo_detected

            Has Login Form
                →
            login_form_detected

        Unknown columns are preserved at this stage and removed later during
        exact schema alignment.
        """

        df = dataframe.copy(
            deep=True
        )

        normalized_columns: List[str] = []

        for column in df.columns:

            canonical_name = (
                VisualFeatureSchema.resolve_feature_name(
                    column
                )
            )

            if not canonical_name:

                logger.warning(
                    "Ignoring empty DataFrame column name: %r",
                    column,
                )

                canonical_name = str(
                    column
                )

            normalized_columns.append(
                canonical_name
            )

        # ---------------------------------------------------------------------
        # Detect duplicate columns created by alias normalization.
        #
        # Example:
        #
        #     width
        #     screenshot_width
        #
        # both resolve to:
        #
        #     screenshot_width
        #
        # Pandas allows duplicate columns, but duplicate model features are
        # dangerous and ambiguous.
        # ---------------------------------------------------------------------

        duplicates = self._find_duplicates(
            normalized_columns
        )

        if duplicates:

            raise ValueError(
                "Duplicate feature columns detected after "
                f"normalization: {sorted(duplicates)}"
            )

        df.columns = normalized_columns

        return df

    # =========================================================================
    # 6. BATCH SCHEMA ALIGNMENT
    # =========================================================================

    def _align_batch_schema(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align a batch DataFrame to the exact 12-feature schema.

        Missing features:
            Added using the schema default.

        Extra features:
            Removed.

        Feature order:
            Forced to EXPECTED_FEATURE_ORDER.
        """

        df = dataframe.copy(
            deep=True
        )

        # ---------------------------------------------------------------------
        # Add missing expected columns.
        # ---------------------------------------------------------------------

        missing_columns = [
            feature
            for feature in self._schema
            if feature not in df.columns
        ]

        if missing_columns:

            logger.warning(
                "Visual batch is missing %d expected feature(s): %s. "
                "Default values will be inserted.",
                len(missing_columns),
                missing_columns,
            )

            for feature in missing_columns:

                default_value = (
                    VisualFeatureSchema.DEFAULT_FEATURES.get(
                        feature,
                        VisualFeatureSchema.DEFAULT_FEATURE_VALUE,
                    )
                )

                df[feature] = default_value

        # ---------------------------------------------------------------------
        # Identify extra columns.
        # ---------------------------------------------------------------------

        extra_columns = [
            column
            for column in df.columns
            if column not in self._schema
        ]

        if extra_columns:

            logger.debug(
                "Dropping %d non-schema Visual batch columns: %s",
                len(extra_columns),
                extra_columns,
            )

        # ---------------------------------------------------------------------
        # Exact feature order.
        # ---------------------------------------------------------------------

        df = df.loc[
            :,
            list(self._schema),
        ]

        return df

    # =========================================================================
    # 7. MAIN TRANSFORMATION ENGINE
    # =========================================================================

    def _apply_transformations(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply all numerical preprocessing transformations.

        Processing order:

            1. Copy DataFrame
            2. Normalize numeric values
            3. Replace infinities
            4. Impute missing values
            5. Normalize Boolean features
            6. Apply feature-specific caps
            7. Re-enforce ratio boundaries
            8. Convert everything to float64
            9. Reorder columns
            10. Return clean DataFrame
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "_apply_transformations expects a pandas DataFrame."
            )

        df = dataframe.copy(
            deep=True
        )

        # ---------------------------------------------------------------------
        # 1. Ensure exact feature order before transformation.
        # ---------------------------------------------------------------------

        df = self._ensure_schema_columns(
            df
        )

        # ---------------------------------------------------------------------
        # 2. Convert all model features to numeric.
        #
        # Invalid strings become NaN and are handled by the next step.
        # ---------------------------------------------------------------------

        for feature in self._schema:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

        # ---------------------------------------------------------------------
        # 3. Replace positive/negative infinity with NaN.
        # ---------------------------------------------------------------------

        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # ---------------------------------------------------------------------
        # 4. Replace missing numeric values with safe defaults.
        #
        # Use per-feature defaults rather than assuming that every default
        # will always be identical.
        # ---------------------------------------------------------------------

        for feature in self._schema:

            default_value = (
                VisualFeatureSchema.DEFAULT_FEATURES.get(
                    feature,
                    VisualFeatureSchema.DEFAULT_FEATURE_VALUE,
                )
            )

            df[feature] = df[feature].fillna(
                default_value
            )

        # ---------------------------------------------------------------------
        # 5. Normalize Boolean features.
        # ---------------------------------------------------------------------

        self._normalize_boolean_columns(
            df
        )

        # ---------------------------------------------------------------------
        # 6. Apply outlier caps.
        # ---------------------------------------------------------------------

        if self.enable_outlier_capping:

            self._apply_outlier_caps(
                df
            )

        # ---------------------------------------------------------------------
        # 7. Enforce ratio boundaries.
        #
        # Ratio features represent proportions and should remain within
        # [0.0, 1.0].
        # ---------------------------------------------------------------------

        self._apply_ratio_bounds(
            df
        )

        # ---------------------------------------------------------------------
        # 8. Convert all features to float64.
        # ---------------------------------------------------------------------

        for feature in self._schema:

            df[feature] = df[
                feature
            ].astype("float64")

        # ---------------------------------------------------------------------
        # 9. Reorder columns one final time.
        # ---------------------------------------------------------------------

        df = df.loc[
            :,
            list(self._schema),
        ]

        return df

    # =========================================================================
    # 8. BOOLEAN NORMALIZATION
    # =========================================================================

    def _normalize_boolean_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Normalize Boolean features to 0.0 / 1.0.

        Supported values include:

            True
            False
            1
            0
            "true"
            "false"
            "yes"
            "no"
            "on"
            "off"

        The DataFrame is modified in place.
        """

        for feature in self.boolean_features:

            if feature not in dataframe.columns:
                continue

            series = dataframe[
                feature
            ]

            # -------------------------------------------------------------
            # Numeric conversion.
            # -------------------------------------------------------------

            numeric_series = pd.to_numeric(
                series,
                errors="coerce",
            )

            # -------------------------------------------------------------
            # String representation for values that failed numeric
            # conversion.
            # -------------------------------------------------------------

            string_series = (
                series.astype(str)
                .str.strip()
                .str.lower()
            )

            true_mask = string_series.isin(
                {
                    "true",
                    "yes",
                    "y",
                    "on",
                }
            )

            false_mask = string_series.isin(
                {
                    "false",
                    "no",
                    "n",
                    "off",
                }
            )

            # -------------------------------------------------------------
            # IMPORTANT:
            # pd.to_numeric(bool_series) can preserve pandas bool dtype.
            # We must explicitly create a float64 Series before assigning
            # 1.0 / 0.0, otherwise newer pandas versions raise:
            #
            #   TypeError: Invalid value '1.0' for dtype 'bool'
            # -------------------------------------------------------------

            result = pd.Series(
                numeric_series.to_numpy(copy=True),
                index=series.index,
                dtype="float64",
            )

            # -------------------------------------------------------------
            # Apply string Boolean values.
            # -------------------------------------------------------------

            result.loc[
                true_mask
            ] = 1.0

            result.loc[
                false_mask
            ] = 0.0

            # -------------------------------------------------------------
            # Invalid Boolean values become 0.0.
            # -------------------------------------------------------------

            result = result.fillna(
                VisualFeatureSchema.DEFAULT_FEATURE_VALUE
            )

            # -------------------------------------------------------------
            # Boolean features must remain 0/1.
            # -------------------------------------------------------------

            result = result.clip(
                lower=0.0,
                upper=1.0,
            )

            # -------------------------------------------------------------
            # Convert all non-zero values to 1.0.
            #
            # This guarantees a binary representation.
            # -------------------------------------------------------------

            result = (
                result > 0.0
            ).astype("float64")

            dataframe[
                feature
            ] = result

    # =========================================================================
    # 9. OUTLIER CAPPING
    # =========================================================================

    def _apply_outlier_caps(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Apply feature-specific upper bounds.

        The DataFrame is modified in place.

        Lower bounds are handled separately for ratio/non-negative features.
        """

        for feature, max_cap in self.outlier_caps.items():

            if feature not in dataframe.columns:
                continue

            if not math.isfinite(
                float(max_cap)
            ):

                raise ValueError(
                    f"Invalid outlier cap for "
                    f"'{feature}': {max_cap}"
                )

            dataframe[
                feature
            ] = dataframe[
                feature
            ].clip(
                upper=float(max_cap)
            )

    # =========================================================================
    # 10. RATIO BOUNDS
    # =========================================================================

    def _apply_ratio_bounds(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Ensure ratio features remain mathematically bounded.

        Ratio features are constrained to:

            0.0 <= value <= 1.0
        """

        for feature in self.ratio_features:

            if feature not in dataframe.columns:
                continue

            dataframe[
                feature
            ] = dataframe[
                feature
            ].clip(
                lower=0.0,
                upper=1.0,
            )

    # =========================================================================
    # 11. SCHEMA ENFORCEMENT
    # =========================================================================

    def _ensure_schema_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ensure that a DataFrame contains every expected feature.

        Missing features are filled using DEFAULT_FEATURES.

        Extra features are removed.
        """

        df = dataframe.copy(
            deep=True
        )

        for feature in self._schema:

            if feature not in df.columns:

                default_value = (
                    VisualFeatureSchema.DEFAULT_FEATURES.get(
                        feature,
                        VisualFeatureSchema.DEFAULT_FEATURE_VALUE,
                    )
                )

                df[feature] = default_value

        # ---------------------------------------------------------------------
        # Keep only model features.
        # ---------------------------------------------------------------------

        df = df.loc[
            :,
            list(self._schema),
        ]

        return df

    # =========================================================================
    # 12. FINAL DATAFRAME VALIDATION
    # =========================================================================

    def _validate_final_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        expect_single_row: bool,
    ) -> None:
        """
        Perform final structural and numerical validation.

        The schema's validate_dataframe() method is designed primarily for
        single-row inference, so batch validation is handled here separately.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Preprocessed output must be a pandas DataFrame."
            )

        # ---------------------------------------------------------------------
        # Exact column count.
        # ---------------------------------------------------------------------

        if len(dataframe.columns) != len(
            self._schema
        ):

            raise ValueError(
                "Preprocessed Visual feature count mismatch. "
                f"Expected {len(self._schema)}, "
                f"received {len(dataframe.columns)}."
            )

        # ---------------------------------------------------------------------
        # Exact column order.
        # ---------------------------------------------------------------------

        if list(
            dataframe.columns
        ) != list(
            self._schema
        ):

            raise ValueError(
                "Preprocessed Visual feature order mismatch. "
                f"Expected {list(self._schema)}, "
                f"received {list(dataframe.columns)}."
            )

        # ---------------------------------------------------------------------
        # Single-row validation.
        # ---------------------------------------------------------------------

        if expect_single_row:

            if len(dataframe) != 1:

                raise ValueError(
                    "Single Visual inference requires exactly "
                    f"one row. Received {len(dataframe)}."
                )

        # ---------------------------------------------------------------------
        # Numeric validation.
        # ---------------------------------------------------------------------

        non_numeric = [
            feature
            for feature in self._schema
            if not pd.api.types.is_numeric_dtype(
                dataframe[feature]
            )
        ]

        if non_numeric:

            raise ValueError(
                "Non-numeric features found after preprocessing: "
                f"{non_numeric}"
            )

        # ---------------------------------------------------------------------
        # Null validation.
        # ---------------------------------------------------------------------

        if dataframe[
            list(self._schema)
        ].isnull().any().any():

            null_columns = dataframe.columns[
                dataframe.isnull().any()
            ].tolist()

            raise ValueError(
                "Preprocessed Visual DataFrame contains "
                f"null values in: {null_columns}"
            )

        # ---------------------------------------------------------------------
        # NaN / infinity validation.
        # ---------------------------------------------------------------------

        matrix = dataframe[
            list(self._schema)
        ].to_numpy(
            dtype="float64"
        )

        if not np.isfinite(
            matrix
        ).all():

            raise ValueError(
                "Preprocessed Visual DataFrame contains "
                "NaN or infinite values."
            )

        # ---------------------------------------------------------------------
        # Semantic validation.
        #
        # For batches, validate every feature column using vectorized
        # checks rather than calling the single-row schema validator.
        # ---------------------------------------------------------------------

        self._validate_semantic_ranges(
            dataframe
        )

    # =========================================================================
    # 13. SEMANTIC RANGE VALIDATION
    # =========================================================================

    def _validate_semantic_ranges(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate non-negative, ratio, and Boolean constraints.
        """

        # ---------------------------------------------------------------------
        # Non-negative features.
        # ---------------------------------------------------------------------

        for feature in (
            VisualFeatureSchema.NON_NEGATIVE_FEATURES
        ):

            if feature not in dataframe.columns:
                continue

            if (
                dataframe[
                    feature
                ] < 0
            ).any():

                message = (
                    f"Feature '{feature}' contains "
                    "negative values after preprocessing."
                )

                if self.strict_validation:
                    raise ValueError(message)

                logger.warning(message)

        # ---------------------------------------------------------------------
        # Ratio features.
        # ---------------------------------------------------------------------

        for feature in self.ratio_features:

            if feature not in dataframe.columns:
                continue

            invalid_mask = (
                (dataframe[feature] < 0.0)
                | (dataframe[feature] > 1.0)
            )

            if invalid_mask.any():

                message = (
                    f"Ratio feature '{feature}' contains "
                    "values outside [0.0, 1.0]."
                )

                if self.strict_validation:
                    raise ValueError(message)

                logger.warning(message)

        # ---------------------------------------------------------------------
        # Boolean features.
        # ---------------------------------------------------------------------

        for feature in self.boolean_features:

            if feature not in dataframe.columns:
                continue

            invalid_mask = ~dataframe[
                feature
            ].isin(
                [0.0, 1.0]
            )

            if invalid_mask.any():

                message = (
                    f"Boolean feature '{feature}' contains "
                    "values other than 0.0 or 1.0."
                )

                if self.strict_validation:
                    raise ValueError(message)

                logger.warning(message)

    # =========================================================================
    # 14. SAFE FALLBACK
    # =========================================================================

    def _create_safe_fallback_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Create a structurally valid one-row fallback DataFrame.

        Every feature receives its schema-defined default value.

        This ensures downstream components receive the correct columns even
        when the live preprocessing operation encounters malformed input.
        """

        fallback = {
            feature: float(
                VisualFeatureSchema.DEFAULT_FEATURES.get(
                    feature,
                    VisualFeatureSchema.DEFAULT_FEATURE_VALUE,
                )
            )
            for feature in self._schema
        }

        dataframe = pd.DataFrame(
            [fallback],
            columns=list(self._schema),
        )

        dataframe = dataframe.astype(
            "float64"
        )

        return dataframe

    # =========================================================================
    # 15. DUPLICATE DETECTION
    # =========================================================================

    @staticmethod
    def _find_duplicates(
        values: List[str],
    ) -> set[str]:
        """
        Return duplicate values from a list.
        """

        seen: set[str] = set()
        duplicates: set[str] = set()

        for value in values:

            if value in seen:
                duplicates.add(value)

            seen.add(value)

        return duplicates

    # =========================================================================
    # 16. PREPROCESSING STATISTICS
    # =========================================================================

    def get_statistics(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate descriptive statistics for a preprocessed DataFrame.

        Useful for:
            - debugging
            - dataset inspection
            - model development
            - detecting unexpected distributions

        Returns
        -------
        Dict[str, Dict[str, float]]

        Example:

            {
                "text_density": {
                    "min": 0.1,
                    "max": 0.9,
                    "mean": 0.5,
                    "std": 0.2
                }
            }
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        df = self._ensure_schema_columns(
            dataframe
        )

        statistics: Dict[
            str,
            Dict[str, float],
        ] = {}

        for feature in self._schema:

            numeric_values = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            statistics[feature] = {
                "min": float(
                    numeric_values.min()
                ),
                "max": float(
                    numeric_values.max()
                ),
                "mean": float(
                    numeric_values.mean()
                ),
                "std": float(
                    numeric_values.std(
                        ddof=0
                    )
                ),
            }

        return statistics

    # =========================================================================
    # 17. OUTLIER REPORT
    # =========================================================================

    def get_outlier_report(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Report values that exceed configured preprocessing caps.

        This method does not modify the DataFrame.

        Returns
        -------
        Dict[str, Dict[str, Any]]

        Example:

            {
                "button_count": {
                    "cap": 50.0,
                    "affected_rows": 3,
                    "affected_percentage": 1.5
                }
            }
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        df = self._ensure_schema_columns(
            dataframe
        )

        report: Dict[
            str,
            Dict[str, Any],
        ] = {}

        total_rows = len(df)

        for feature, cap in self.outlier_caps.items():

            if feature not in df.columns:
                continue

            values = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            affected_mask = (
                values > cap
            )

            affected_rows = int(
                affected_mask.sum()
            )

            affected_percentage = (
                (
                    affected_rows
                    / total_rows
                    * 100.0
                )
                if total_rows > 0
                else 0.0
            )

            report[feature] = {
                "cap": float(cap),
                "affected_rows": affected_rows,
                "affected_percentage": (
                    float(
                        affected_percentage
                    )
                ),
            }

        return report

    # =========================================================================
    # 18. PREPROCESSING HEALTH CHECK
    # =========================================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:
        """
        Perform a static health check of the preprocessor configuration.

        This does not process any data.
        """

        schema = set(
            self._schema
        )

        cap_features = set(
            self.outlier_caps.keys()
        )

        unknown_caps = (
            cap_features - schema
        )

        missing_caps = (
            {
                feature
                for feature in schema
                if feature
                not in self.boolean_features
            }
            - cap_features
        )

        return {
            "status": (
                "healthy"
                if not unknown_caps
                else "warning"
            ),
            "feature_count": len(
                self._schema
            ),
            "schema_features": list(
                self._schema
            ),
            "unknown_cap_features": sorted(
                unknown_caps
            ),
            "features_without_caps": sorted(
                missing_caps
            ),
            "outlier_capping_enabled": (
                self.enable_outlier_capping
            ),
            "strict_validation": (
                self.strict_validation
            ),
        }

    # =========================================================================
    # 19. MODEL MATRIX CONVERSION
    # =========================================================================

    def to_model_matrix(
        self,
        raw_features: Optional[Mapping[str, Any]],
    ) -> np.ndarray:
        """
        Transform raw features directly into a NumPy model matrix.

        Returns
        -------
        numpy.ndarray

        Shape:

            (1, 12)

        This is suitable for models such as:

            XGBoost
            RandomForest
            ExtraTrees
            LogisticRegression
            SVM
            Neural-network classifiers
        """

        dataframe = self.transform_single(
            raw_features
        )

        matrix = dataframe.to_numpy(
            dtype="float64",
            copy=True,
        )

        if matrix.shape != (
            1,
            len(self._schema),
        ):

            raise ValueError(
                "Unexpected Visual model matrix shape. "
                f"Expected (1, {len(self._schema)}), "
                f"received {matrix.shape}."
            )

        return matrix

    # =========================================================================
    # 20. SINGLE-ROW DIAGNOSTIC
    # =========================================================================

    def get_transformation_summary(
        self,
        raw_features: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Return a diagnostic summary showing how a raw feature vector is
        interpreted by the preprocessor.

        Useful for debugging the Visual Agent pipeline.
        """

        processed = self.transform_single(
            raw_features
        )

        row = processed.iloc[0]

        feature_values: Dict[
            str,
            float,
        ] = {}

        for feature in self._schema:

            feature_values[feature] = float(
                row[feature]
            )

        return {
            "schema_version": (
                VisualFeatureSchema.get_schema_version()
            ),
            "feature_count": len(
                self._schema
            ),
            "features": feature_values,
            "outlier_capping_enabled": (
                self.enable_outlier_capping
            ),
            "strict_validation": (
                self.strict_validation
            ),
        }
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from .feature_schema import URLFeatureSchema


logger = logging.getLogger(__name__)


class URLPreprocessor:
    """
    Production-grade preprocessing layer for the URL AI Agent.

    Responsibilities
    ----------------
    1. Enforce the central URLFeatureSchema.
    2. Preserve domain/path/query feature separation.
    3. Handle missing, NaN and infinite values.
    4. Apply safe upper-bound clipping.
    5. Guarantee deterministic feature ordering.
    6. Guarantee numeric values for XGBoost.
    7. Keep training and live inference preprocessing identical.

    IMPORTANT
    ---------
    The following entropy features MUST remain separate:

        url_entropy
        domain_entropy
        path_entropy
        query_entropy

    This is important because a legitimate website may contain a
    highly random generated path or query parameter.

    Example:

        docs.google.com/forms/d/e/1FAIp...

    can naturally have high path entropy without the domain itself
    being suspicious.

    Therefore this preprocessor does NOT merge these features.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(self):
        """
        Initialize preprocessing configuration.

        The caps are intentionally conservative. They prevent abnormal
        values from dominating the model while preserving useful
        differences between legitimate and suspicious URLs.
        """

        self.outlier_caps: Dict[str, float] = {

            # ----------------------------------------------------------
            # URL dimensions
            # ----------------------------------------------------------

            "url_length": 2048.0,

            "domain_length": 255.0,

            "path_length": 4096.0,

            "query_length": 4096.0,

            # ----------------------------------------------------------
            # Structural counts
            # ----------------------------------------------------------

            "num_dots": 25.0,

            "num_hyphens": 25.0,

            "num_underscores": 25.0,

            "num_slashes": 100.0,

            "num_question_marks": 20.0,

            "num_equal_signs": 100.0,

            "num_at_symbols": 10.0,

            "num_percent_signs": 100.0,

            "num_digits": 500.0,

            "num_special_chars": 200.0,

            # ----------------------------------------------------------
            # Entropy
            # ----------------------------------------------------------

            # Global URL entropy.
            #
            # This is intentionally kept as a contextual feature.
            "url_entropy": 8.0,

            # Domain entropy.
            #
            # This is generally more meaningful for detecting
            # algorithmically generated domains.
            "domain_entropy": 8.0,

            # Path entropy.
            #
            # High path entropy is NOT automatically malicious.
            "path_entropy": 8.0,

            # Query entropy.
            #
            # High query entropy can be legitimate for tracking,
            # authentication and application parameters.
            "query_entropy": 8.0,

            # ----------------------------------------------------------
            # Binary features
            # ----------------------------------------------------------

            "contains_ip": 1.0,

            "is_https": 1.0,

            "has_punycode": 1.0,

            # ----------------------------------------------------------
            # Domain structure
            # ----------------------------------------------------------

            "subdomain_count": 20.0,

            "tld_length": 20.0,

            # ----------------------------------------------------------
            # Suspicious vocabulary
            # ----------------------------------------------------------

            "suspicious_word_count": 30.0,
        }

        # --------------------------------------------------------------
        # Expected schema
        # --------------------------------------------------------------

        self.feature_names = (
            URLFeatureSchema.get_schema()
        )

        logger.info(
            "URLPreprocessor initialized with %d features.",
            len(self.feature_names)
        )

        logger.debug(
            "URL preprocessing feature order: %s",
            self.feature_names
        )

    # ==================================================================
    # SINGLE SAMPLE TRANSFORMATION
    # ==================================================================

    def transform_single(
        self,
        raw_features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Transform one live URL feature dictionary into a model-ready
        DataFrame.

        Pipeline:

            Raw URL features
                    ↓
            URLFeatureSchema
                    ↓
            Schema alignment
                    ↓
            Missing-value handling
                    ↓
            Outlier clipping
                    ↓
            Numeric conversion
                    ↓
            XGBoost-ready DataFrame

        Args:
            raw_features:
                Raw feature dictionary produced by the URL feature
                extractor.

        Returns:
            Exactly one-row pandas DataFrame whose columns exactly
            match URLFeatureSchema.
        """

        try:

            if not isinstance(
                raw_features,
                dict
            ):

                logger.warning(
                    "URLPreprocessor received invalid input type: %s. "
                    "Using schema defaults.",
                    type(raw_features).__name__
                )

                raw_features = {}

            # ==========================================================
            # STEP 1
            # Central schema alignment
            # ==========================================================

            df_aligned = (
                URLFeatureSchema.align_and_validate(
                    raw_features
                )
            )

            # ==========================================================
            # STEP 2
            # Apply common transformations
            # ==========================================================

            df_processed = (
                self._apply_transformations(
                    df_aligned
                )
            )

            # ==========================================================
            # STEP 3
            # Final schema validation
            # ==========================================================

            self._validate_output(
                df_processed
            )

            # ==========================================================
            # Preserve useful metadata
            # ==========================================================

            df_processed.attrs.update(
                df_aligned.attrs
            )

            df_processed.attrs[
                "preprocessor_version"
            ] = "2.0.0"

            return df_processed

        except Exception as e:

            logger.error(
                "Failed to preprocess single URL feature vector: %s",
                str(e),
                exc_info=True
            )

            # ----------------------------------------------------------
            # Safe fallback
            # ----------------------------------------------------------

            fallback = pd.DataFrame(
                [
                    URLFeatureSchema.get_default_features()
                ],
                columns=self.feature_names
            )

            fallback = (
                self._apply_transformations(
                    fallback
                )
            )

            fallback.attrs[
                "preprocessing_error"
            ] = str(e)

            fallback.attrs[
                "preprocessor_version"
            ] = "2.0.0"

            return fallback

    # ==================================================================
    # BATCH TRANSFORMATION
    # ==================================================================

    def transform_batch(
        self,
        df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform a complete training dataset.

        IMPORTANT:
        Training preprocessing MUST be identical to live inference
        preprocessing.

        Therefore this method uses:

            URLFeatureSchema.align_dataframe()

        rather than maintaining a second independent schema-alignment
        implementation.

        Args:
            df_raw:
                Raw training DataFrame.

        Returns:
            Fully aligned and preprocessed DataFrame.
        """

        if not isinstance(
            df_raw,
            pd.DataFrame
        ):

            raise TypeError(
                "transform_batch() expects a pandas DataFrame."
            )

        if df_raw.empty:

            raise ValueError(
                "Cannot preprocess an empty URL dataset."
            )

        try:

            logger.info(
                "Preprocessing URL batch: %d rows × %d columns",
                df_raw.shape[0],
                df_raw.shape[1]
            )

            # ==========================================================
            # STEP 1
            # Use the SAME central schema used by live inference.
            # ==========================================================

            df_aligned = (
                URLFeatureSchema.align_dataframe(
                    df_raw
                )
            )

            # ==========================================================
            # STEP 2
            # Apply transformations.
            # ==========================================================

            df_processed = (
                self._apply_transformations(
                    df_aligned
                )
            )

            # ==========================================================
            # STEP 3
            # Validate exact output.
            # ==========================================================

            self._validate_output(
                df_processed,
                expected_rows=len(df_raw)
            )

            logger.info(
                "URL batch preprocessing completed successfully: "
                "%d rows × %d features",
                df_processed.shape[0],
                df_processed.shape[1]
            )

            return df_processed

        except Exception as e:

            logger.error(
                "Failed to preprocess URL batch dataset: %s",
                str(e),
                exc_info=True
            )

            raise

    # ==================================================================
    # CORE TRANSFORMATION PIPELINE
    # ==================================================================

    def _apply_transformations(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply deterministic preprocessing operations.

        Operations:

            1. Copy input
            2. Replace infinity
            3. Numeric conversion
            4. NaN imputation
            5. Outlier clipping
            6. Binary normalization
            7. Float conversion
            8. Exact column ordering
        """

        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise TypeError(
                "_apply_transformations() expects a DataFrame."
            )

        processed = df.copy()

        # ==============================================================
        # STEP 1
        # Ensure exact feature set
        # ==============================================================

        processed = (
            self._ensure_schema(
                processed
            )
        )

        # ==============================================================
        # STEP 2
        # Replace infinity
        # ==============================================================

        processed = processed.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # ==============================================================
        # STEP 3
        # Numeric conversion
        # ==============================================================

        for feature in self.feature_names:

            processed[
                feature
            ] = pd.to_numeric(
                processed[
                    feature
                ],
                errors="coerce"
            )

        # ==============================================================
        # STEP 4
        # Missing value imputation
        # ==============================================================

        processed = processed.fillna(
            0.0
        )

        # ==============================================================
        # STEP 5
        # Prevent negative values for inherently non-negative
        # features.
        # ==============================================================

        non_negative_features = {

            "url_length",
            "domain_length",
            "path_length",
            "query_length",

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

            "subdomain_count",
            "tld_length",
            "suspicious_word_count"
        }

        for feature in non_negative_features:

            if feature in processed.columns:

                processed[
                    feature
                ] = processed[
                    feature
                ].clip(
                    lower=0.0
                )

        # ==============================================================
        # STEP 6
        # Outlier capping
        # ==============================================================

        for feature, maximum in (
            self.outlier_caps.items()
        ):

            if feature not in processed.columns:
                continue

            processed[
                feature
            ] = processed[
                feature
            ].clip(
                upper=maximum
            )

        # ==============================================================
        # STEP 7
        # Binary feature normalization
        # ==============================================================

        binary_features = {

            "contains_ip",
            "is_https",
            "has_punycode"
        }

        for feature in binary_features:

            if feature in processed.columns:

                processed[
                    feature
                ] = (
                    processed[
                        feature
                    ]
                    .astype(float)
                    .apply(
                        lambda value:
                            1.0
                            if value != 0.0
                            else 0.0
                    )
                )

        # ==============================================================
        # STEP 8
        # Final float conversion
        # ==============================================================

        for feature in self.feature_names:

            processed[
                feature
            ] = processed[
                feature
            ].astype(
                np.float64
            )

        # ==============================================================
        # STEP 9
        # Final infinity / NaN protection
        # ==============================================================

        processed = processed.replace(
            [np.inf, -np.inf],
            0.0
        )

        processed = processed.fillna(
            0.0
        )

        # ==============================================================
        # STEP 10
        # EXACT FEATURE ORDER
        # ==============================================================

        processed = processed[
            self.feature_names
        ]

        return processed

    # ==================================================================
    # SCHEMA ENFORCEMENT
    # ==================================================================

    def _ensure_schema(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Ensure that the DataFrame contains exactly the central schema.

        Missing features receive schema defaults.

        Extra columns are discarded.

        This method is deliberately defensive because training datasets
        can contain metadata columns such as:

            url
            label
            source
            timestamp

        Those must never accidentally enter the XGBoost feature matrix.
        """

        result = df.copy()

        # --------------------------------------------------------------
        # Add missing schema features
        # --------------------------------------------------------------

        defaults = (
            URLFeatureSchema.get_default_features()
        )

        for feature in self.feature_names:

            if feature not in result.columns:

                logger.warning(
                    "Missing URL feature '%s'. "
                    "Using default value.",
                    feature
                )

                result[
                    feature
                ] = defaults.get(
                    feature,
                    0.0
                )

        # --------------------------------------------------------------
        # Drop everything outside the schema
        # --------------------------------------------------------------

        result = result[
            self.feature_names
        ].copy()

        return result

    # ==================================================================
    # OUTPUT VALIDATION
    # ==================================================================

    def _validate_output(
        self,
        df: pd.DataFrame,
        expected_rows: int = None
    ) -> None:
        """
        Validate the final model-ready DataFrame.

        Raises RuntimeError if the output does not exactly match the
        feature schema.
        """

        # --------------------------------------------------------------
        # Type
        # --------------------------------------------------------------

        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise RuntimeError(
                "Preprocessor output is not a DataFrame."
            )

        # --------------------------------------------------------------
        # Column order
        # --------------------------------------------------------------

        actual_columns = list(
            df.columns
        )

        if actual_columns != self.feature_names:

            raise RuntimeError(
                "URL preprocessing schema mismatch.\n"
                f"Expected: {self.feature_names}\n"
                f"Actual:   {actual_columns}"
            )

        # --------------------------------------------------------------
        # Row count
        # --------------------------------------------------------------

        if expected_rows is not None:

            if len(df) != expected_rows:

                raise RuntimeError(
                    "URL preprocessing changed the number "
                    f"of rows: expected {expected_rows}, "
                    f"received {len(df)}."
                )

        # --------------------------------------------------------------
        # Numeric values
        # --------------------------------------------------------------

        for feature in self.feature_names:

            if not pd.api.types.is_numeric_dtype(
                df[feature]
            ):

                raise RuntimeError(
                    f"URL feature '{feature}' is not numeric."
                )

        # --------------------------------------------------------------
        # NaN check
        # --------------------------------------------------------------

        if df.isna().any().any():

            raise RuntimeError(
                "Preprocessed URL feature matrix still "
                "contains NaN values."
            )

        # --------------------------------------------------------------
        # Infinity check
        # --------------------------------------------------------------

        numeric_values = (
            df.to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            numeric_values
        ).all():

            raise RuntimeError(
                "Preprocessed URL feature matrix contains "
                "infinite values."
            )

    # ==================================================================
    # FEATURE INSPECTION
    # ==================================================================

    def get_feature_names(self):
        """
        Return the exact feature order used by XGBoost.
        """

        return self.feature_names.copy()

    # ==================================================================
    # FEATURE SUMMARY
    # ==================================================================

    def get_feature_summary(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate a numerical summary useful for debugging and
        model-training verification.
        """

        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise TypeError(
                "get_feature_summary() expects a DataFrame."
            )

        aligned = (
            URLFeatureSchema.align_dataframe(
                df
            )
        )

        processed = (
            self._apply_transformations(
                aligned
            )
        )

        summary = {}

        for feature in self.feature_names:

            values = processed[
                feature
            ]

            summary[
                feature
            ] = {

                "min":
                    float(values.min()),

                "max":
                    float(values.max()),

                "mean":
                    float(values.mean()),

                "std":
                    float(values.std())
                    if len(values) > 1
                    else 0.0
            }

        return summary
"""
HTML AI Agent - Preprocessing
=============================

Responsible for converting extracted HTML features into a deterministic,
numerically safe ML feature matrix.

Pipeline:

    Raw HTML Features
            ↓
    HTMLFeatureSchema
            ↓
    Schema Alignment
            ↓
    Missing Value Handling
            ↓
    Numeric Conversion
            ↓
    Boolean Normalization
            ↓
    Ratio Normalization
            ↓
    Outlier Clipping
            ↓
    NaN / Infinity Protection
            ↓
    Float Conversion
            ↓
    Final Schema Validation
            ↓
    XGBoost Model


IMPORTANT
---------
This module does NOT determine whether a website is phishing.

For example:

    hidden_input_count = 23

does NOT automatically mean phishing.

The preprocessing layer only prepares the feature representation
for the machine-learning model.
"""

import logging

from typing import Dict, Any, Set

import numpy as np
import pandas as pd

from .feature_schema import HTMLFeatureSchema


# =========================================================================
# LOGGER
# =========================================================================

logger = logging.getLogger(__name__)


class HTMLPreprocessor:
    """
    Production-grade preprocessing layer for the HTML AI Agent.

    Responsibilities
    ----------------
    1. Align raw features to the central feature schema.
    2. Add missing features with safe defaults.
    3. Ignore unexpected features.
    4. Convert values to numeric form.
    5. Normalize boolean features.
    6. Normalize ratio features.
    7. Clip extreme numerical values.
    8. Remove NaN and infinity values.
    9. Preserve exact feature ordering.
    10. Validate the final ML-ready DataFrame.

    IMPORTANT
    ---------
    The exact same preprocessing pipeline is used during:

        Training
            AND
        Live Inference

    This prevents training/inference preprocessing drift.
    """

    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    def __init__(self):
        """
        Initialize the HTML preprocessing configuration.
        """

        # -----------------------------------------------------------------
        # Numerical outlier limits
        #
        # These values are safety limits.
        #
        # They are NOT phishing thresholds.
        # -----------------------------------------------------------------

        self.outlier_caps: Dict[str, float] = {

            # =============================================================
            # BASIC HTML / DOCUMENT
            # =============================================================

            "html_length_bytes":
                5_000_000.0,

            # =============================================================
            # FORM FEATURES
            # =============================================================

            "form_count":
                50.0,

            "forms_with_action":
                50.0,

            "empty_form_action_count":
                50.0,

            "password_field_count":
                20.0,

            "email_field_count":
                20.0,

            "username_field_count":
                20.0,

            "payment_field_count":
                20.0,

            "otp_field_count":
                20.0,

            "credential_field_count":
                50.0,

            # =============================================================
            # FORM DESTINATION
            # =============================================================

            "cross_domain_form_actions":
                50.0,

            "same_domain_form_actions":
                50.0,

            "http_form_actions":
                50.0,

            "https_form_actions":
                50.0,

            # =============================================================
            # DOMAIN RELATIONSHIP
            # =============================================================

            "subdomain_match_count":
                100.0,

            "registrable_domain_match_count":
                100.0,

            "cross_domain_count":
                100.0,

            # =============================================================
            # HIDDEN INPUTS
            # =============================================================

            "hidden_input_count":
                200.0,

            "credential_hidden_input_count":
                100.0,

            "suspicious_hidden_input_count":
                100.0,

            # =============================================================
            # IFRAMES
            # =============================================================

            "iframe_count":
                100.0,

            "invisible_iframes":
                50.0,

            # =============================================================
            # JAVASCRIPT
            # =============================================================

            "script_count":
                500.0,

            "external_scripts_count":
                300.0,

            "suspicious_inline_scripts":
                100.0,

            # =============================================================
            # LINKS
            # =============================================================

            "total_links":
                5_000.0,

            "external_links_count":
                3_000.0,

            "empty_links_count":
                500.0,

            # =============================================================
            # DOM / EVASION
            # =============================================================

            "hidden_elements_count":
                1_000.0,

            "html_comments_count":
                1_000.0
        }

        # -----------------------------------------------------------------
        # Ratio features
        #
        # These values MUST remain in:
        #
        #       0.0 <= value <= 1.0
        #
        # The feature schema currently defines the same ratio features.
        # -----------------------------------------------------------------

        self.ratio_features: Set[str] = {

            "external_form_action_ratio",

            "hidden_input_ratio"
        }

        # -----------------------------------------------------------------
        # Retrieve the authoritative schema.
        # -----------------------------------------------------------------

        self.feature_schema = (
            HTMLFeatureSchema.get_schema()
        )

        logger.info(
            "HTMLPreprocessor initialized with %d features.",
            len(
                self.feature_schema
            )
        )

    # =====================================================================
    # SINGLE SAMPLE TRANSFORMATION
    # =====================================================================

    def transform_single(
        self,
        raw_features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Transform one raw HTML feature dictionary into an ML-ready
        DataFrame.

        Parameters
        ----------
        raw_features:
            Raw feature dictionary produced by the HTML Feature Extractor.

        Returns
        -------
        pandas.DataFrame
            Exactly one row containing the complete HTML feature schema.

        Notes
        -----
        If preprocessing fails, a safe zero-valued feature vector is
        returned. The Predictor can then determine whether the resulting
        signal should be treated as usable.
        """

        try:

            # =============================================================
            # 1. INPUT VALIDATION
            # =============================================================

            if not isinstance(
                raw_features,
                dict
            ):

                logger.warning(
                    "HTMLPreprocessor received invalid feature payload. "
                    "Expected dictionary."
                )

                raw_features = {}

            # =============================================================
            # 2. SCHEMA ALIGNMENT
            # =============================================================

            df_aligned = (
                HTMLFeatureSchema
                .align_and_validate(
                    raw_features
                )
            )

            # =============================================================
            # 3. COMMON TRANSFORMATION PIPELINE
            # =============================================================

            df_processed = (
                self._apply_transformations(
                    df_aligned
                )
            )

            # =============================================================
            # 4. FINAL VALIDATION
            # =============================================================

            if not self.validate_processed_features(
                df_processed
            ):

                raise ValueError(
                    "Processed HTML feature DataFrame "
                    "failed final validation."
                )

            return df_processed

        except Exception as e:

            logger.error(
                "Failed to preprocess HTML feature vector: %s",
                str(e),
                exc_info=True
            )

            # -------------------------------------------------------------
            # Return safe schema-compatible vector.
            # -------------------------------------------------------------

            return self._get_safe_dataframe()

    # =====================================================================
    # BATCH TRANSFORMATION
    # =====================================================================

    def transform_batch(
        self,
        df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform a training dataset using exactly the same preprocessing
        logic used during live inference.

        Parameters
        ----------
        df_raw:
            Raw training feature DataFrame.

        Returns
        -------
        pandas.DataFrame
            Fully processed feature matrix.

        Raises
        ------
        TypeError
            If df_raw is not a DataFrame.

        ValueError
            If preprocessing or final validation fails.
        """

        # =============================================================
        # 1. INPUT TYPE VALIDATION
        # =============================================================

        if not isinstance(
            df_raw,
            pd.DataFrame
        ):

            raise TypeError(
                "transform_batch expects a pandas DataFrame."
            )

        if df_raw.empty:

            raise ValueError(
                "transform_batch received an empty DataFrame."
            )

        try:

            df = (
                df_raw
                .copy()
            )

            # =========================================================
            # 2. GET AUTHORITATIVE SCHEMA
            # =========================================================

            schema = (
                HTMLFeatureSchema.get_schema()
            )

            # =========================================================
            # 3. ADD MISSING FEATURES
            # =========================================================

            for feature_name in schema:

                if feature_name not in df.columns:

                    logger.warning(
                        "Training dataset missing feature '%s'. "
                        "Filling with 0.0.",
                        feature_name
                    )

                    df[feature_name] = 0.0

            # =========================================================
            # 4. REMOVE EXTRA FEATURES
            #
            # This prevents accidental data leakage.
            # =========================================================

            df = df[
                schema
            ].copy()

            # =========================================================
            # 5. COMMON TRANSFORMATION PIPELINE
            # =========================================================

            df = (
                self._apply_transformations(
                    df
                )
            )

            # =========================================================
            # 6. FINAL VALIDATION
            # =========================================================

            if not self.validate_processed_features(
                df
            ):

                raise ValueError(
                    "Batch preprocessing produced "
                    "an invalid HTML feature matrix."
                )

            return df

        except Exception as e:

            logger.error(
                "HTML batch preprocessing failed: %s",
                str(e),
                exc_info=True
            )

            raise

    # =====================================================================
    # COMMON TRANSFORMATION PIPELINE
    # =====================================================================

    def _apply_transformations(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply the complete numerical safety transformation pipeline.

        This method MUST remain identical in behavior for:

            Training
            and
            Inference
        """

        # =============================================================
        # COPY INPUT
        # =============================================================

        df = (
            df
            .copy()
        )

        # =============================================================
        # 1. ENSURE EXACT SCHEMA
        # =============================================================

        schema = (
            HTMLFeatureSchema.get_schema()
        )

        for feature_name in schema:

            if feature_name not in df.columns:

                df[feature_name] = 0.0

        # -------------------------------------------------------------
        # Keep ONLY expected model features.
        # -------------------------------------------------------------

        df = df[
            schema
        ].copy()

        # =============================================================
        # 2. REPLACE INFINITY WITH NaN
        # =============================================================

        df = df.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        # =============================================================
        # 3. NUMERIC CONVERSION
        # =============================================================

        for column in schema:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # =============================================================
        # 4. MISSING VALUE IMPUTATION
        # =============================================================

        df = df.fillna(
            0.0
        )

        # =============================================================
        # 5. BOOLEAN NORMALIZATION
        # =============================================================

        boolean_features = (
            HTMLFeatureSchema
            .get_boolean_features()
        )

        for feature_name in boolean_features:

            if feature_name not in df.columns:

                continue

            # ---------------------------------------------------------
            # Any non-zero value becomes 1.
            #
            # Zero becomes 0.
            # ---------------------------------------------------------

            df[feature_name] = (
                df[feature_name]
                .apply(
                    lambda value:
                    1.0
                    if float(value) != 0.0
                    else 0.0
                )
            )

        # =============================================================
        # 6. RATIO NORMALIZATION
        # =============================================================

        # Use the central schema as the source of truth where possible.
        # The local set is kept for compatibility with the current
        # preprocessing configuration.

        ratio_features = set(
            self.ratio_features
        )

        try:

            schema_ratio_features = (
                HTMLFeatureSchema
                .get_ratio_features()
            )

            ratio_features.update(
                schema_ratio_features
            )

        except AttributeError:

            logger.debug(
                "HTMLFeatureSchema does not expose "
                "get_ratio_features()."
            )

        for feature_name in ratio_features:

            if feature_name not in df.columns:

                continue

            df[feature_name] = (
                df[feature_name]
                .clip(
                    lower=0.0,
                    upper=1.0
                )
            )

        # =============================================================
        # 7. NUMERICAL OUTLIER CLIPPING
        # =============================================================

        for (
            feature_name,
            maximum
        ) in self.outlier_caps.items():

            if feature_name not in df.columns:

                continue

            df[feature_name] = (
                df[feature_name]
                .clip(
                    lower=0.0,
                    upper=maximum
                )
            )

        # =============================================================
        # 8. FINAL FLOAT CONVERSION
        # =============================================================

        for column in schema:

            df[column] = (
                df[column]
                .astype(float)
            )

        # =============================================================
        # 9. FINAL NaN / INFINITY PROTECTION
        # =============================================================

        df = df.replace(
            [
                np.inf,
                -np.inf
            ],
            0.0
        )

        df = df.fillna(
            0.0
        )

        # =============================================================
        # 10. FINAL SCHEMA ORDERING
        # =============================================================

        df = df[
            schema
        ].copy()

        return df

    # =====================================================================
    # SAFE FALLBACK DATAFRAME
    # =====================================================================

    @staticmethod
    def _get_safe_dataframe() -> pd.DataFrame:
        """
        Return a completely valid zero-valued feature vector.

        IMPORTANT
        ---------
        This is only a schema-safe fallback.

        It does NOT mean that the target webpage is legitimate.

        The higher-level Predictor/Agent must know that preprocessing
        failed if this fallback is used.
        """

        schema = (
            HTMLFeatureSchema.get_schema()
        )

        data = {
            feature_name:
                0.0
            for feature_name
            in schema
        }

        return pd.DataFrame(
            [data],
            columns=schema
        )

    # =====================================================================
    # VALIDATE PROCESSED FEATURES
    # =====================================================================

    def validate_processed_features(
        self,
        df: pd.DataFrame
    ) -> bool:
        """
        Validate a processed feature matrix before ML inference.

        Validation includes:

            1. DataFrame type
            2. Non-empty input
            3. Exact schema
            4. Exact column order
            5. No NaN
            6. No infinity
            7. Numeric data types
            8. Valid ratio ranges
            9. Valid boolean values
        """

        # =============================================================
        # 1. TYPE CHECK
        # =============================================================

        if not isinstance(
            df,
            pd.DataFrame
        ):

            logger.error(
                "Processed HTML features are not a DataFrame."
            )

            return False

        # =============================================================
        # 2. EMPTY CHECK
        # =============================================================

        if df.empty:

            logger.error(
                "Processed HTML feature DataFrame is empty."
            )

            return False

        # =============================================================
        # 3. SCHEMA VALIDATION
        # =============================================================

        if not HTMLFeatureSchema.validate_dataframe(
            df
        ):

            logger.error(
                "HTML feature schema validation failed."
            )

            return False

        # =============================================================
        # 4. EXACT COLUMN ORDER
        # =============================================================

        expected_schema = (
            HTMLFeatureSchema.get_schema()
        )

        actual_schema = list(
            df.columns
        )

        if actual_schema != expected_schema:

            logger.error(
                "HTML feature order mismatch.\n"
                "Expected: %s\n"
                "Received: %s",
                expected_schema,
                actual_schema
            )

            return False

        # =============================================================
        # 5. NaN CHECK
        # =============================================================

        if df.isna().any().any():

            logger.error(
                "Processed HTML features contain NaN values."
            )

            return False

        # =============================================================
        # 6. INFINITY CHECK
        # =============================================================

        try:

            numeric_values = (
                df.to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                numeric_values
            ).all():

                logger.error(
                    "Processed HTML features contain "
                    "infinite or non-finite values."
                )

                return False

        except Exception as e:

            logger.error(
                "Unable to validate numerical HTML features: %s",
                str(e)
            )

            return False

        # =============================================================
        # 7. NUMERIC DTYPE CHECK
        # =============================================================

        for column in expected_schema:

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):

                logger.error(
                    "HTML feature '%s' is not numeric.",
                    column
                )

                return False

        # =============================================================
        # 8. RATIO VALIDATION
        # =============================================================

        ratio_features = set(
            self.ratio_features
        )

        try:

            ratio_features.update(
                HTMLFeatureSchema
                .get_ratio_features()
            )

        except AttributeError:

            pass

        for feature_name in ratio_features:

            if feature_name not in df.columns:

                continue

            values = (
                df[feature_name]
            )

            if (
                (values < 0.0).any()
                or
                (values > 1.0).any()
            ):

                logger.error(
                    "Ratio feature '%s' contains "
                    "values outside [0, 1].",
                    feature_name
                )

                return False

        # =============================================================
        # 9. BOOLEAN VALIDATION
        # =============================================================

        boolean_features = (
            HTMLFeatureSchema
            .get_boolean_features()
        )

        for feature_name in boolean_features:

            if feature_name not in df.columns:

                continue

            unique_values = set(
                df[feature_name]
                .astype(float)
                .unique()
                .tolist()
            )

            invalid_values = (
                unique_values
                -
                {
                    0.0,
                    1.0
                }
            )

            if invalid_values:

                logger.error(
                    "Boolean feature '%s' contains "
                    "invalid values: %s",
                    feature_name,
                    invalid_values
                )

                return False

        return True

    # =====================================================================
    # FEATURE STATISTICS
    # =====================================================================

    def get_feature_statistics(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Return basic statistics for a processed HTML feature matrix.

        Useful for:

            - Debugging
            - Model evaluation
            - Dataset analysis
            - Research documentation
        """

        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise TypeError(
                "get_feature_statistics expects "
                "a pandas DataFrame."
            )

        if df.empty:

            return {
                "rows": 0,
                "features": 0,
                "columns": []
            }

        return {

            "rows":
                int(
                    len(df)
                ),

            "features":
                int(
                    len(df.columns)
                ),

            "columns":
                list(
                    df.columns
                ),

            "missing_values":
                int(
                    df.isna()
                    .sum()
                    .sum()
                ),

            "infinite_values":
                int(
                    np.isinf(
                        df.to_numpy(
                            dtype=float
                        )
                    )
                    .sum()
                ),

            "all_numeric":
                bool(
                    all(
                        pd.api.types
                        .is_numeric_dtype(
                            dtype
                        )
                        for dtype
                        in df.dtypes
                    )
                )
        }

    # =====================================================================
    # SCHEMA INFORMATION
    # =====================================================================

    def get_schema(
        self
    ):
        """
        Return the exact feature schema used by the preprocessor.
        """

        return list(
            self.feature_schema
        )

    # =====================================================================
    # PREPROCESSOR STATUS
    # =====================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return preprocessing configuration and status.
        """

        return {

            "status":
                "ready",

            "feature_count":
                len(
                    self.feature_schema
                ),

            "feature_schema":
                list(
                    self.feature_schema
                ),

            "ratio_features":
                sorted(
                    self.ratio_features
                ),

            "outlier_cap_count":
                len(
                    self.outlier_caps
                ),

            "outlier_caps":
                dict(
                    self.outlier_caps
                )
        }
"""
SSL/TLS Security AI Agent - Preprocessing
=========================================

Production-grade preprocessing layer for the SSL/TLS AI Agent.

Responsibilities
----------------

    Raw SSL/TLS telemetry
            ↓
    SSLFeatureSchema
            ↓
    Canonical feature names
            ↓
    Type normalization
            ↓
    Missing-value handling
            ↓
    Infinity / NaN handling
            ↓
    Range validation
            ↓
    Outlier clipping
            ↓
    Boolean → numerical conversion
            ↓
    Deterministic feature ordering
            ↓
    XGBoost-ready DataFrame


IMPORTANT
---------
Training and inference MUST use the same preprocessing pipeline.

Therefore:

    trainer.py
        ↓
    SSLPreprocessor.transform_batch()

and:

    ssl_agent.py
        ↓
    SSLPreprocessor.transform_single()

must produce identical feature columns and transformations.

The final model input is always:

    17 numerical features
    in the exact order defined by SSLFeatureSchema.
"""

import logging

from typing import (
    Dict,
    Any,
    List,
    Optional
)

import numpy as np

import pandas as pd

from .feature_schema import SSLFeatureSchema


logger = logging.getLogger(__name__)


class SSLPreprocessor:
    """
    Production-grade preprocessing engine for the SSL/TLS AI Agent.

    This class is responsible for converting raw SSL/TLS telemetry into
    deterministic numerical feature matrices suitable for machine-learning
    training and inference.

    Main responsibilities:

        1. Schema alignment
        2. Alias normalization
        3. Missing-feature handling
        4. Unknown-feature removal
        5. Boolean normalization
        6. Numeric conversion
        7. NaN handling
        8. Infinity handling
        9. Range enforcement
        10. Outlier clipping
        11. Feature ordering
        12. Training/inference consistency
        13. Input validation
        14. Preprocessing metadata

    The class deliberately does NOT perform machine-learning prediction.
    That responsibility belongs to predictor.py.
    """

    # ======================================================================
    # PREPROCESSOR METADATA
    # ======================================================================

    PREPROCESSOR_NAME = (
        "SSLPreprocessor"
    )

    PREPROCESSOR_VERSION = "1.0.0"

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(
        self,
        outlier_caps: Optional[
            Dict[str, float]
        ] = None
    ):
        """
        Initialize the SSL preprocessing engine.

        Args:
            outlier_caps:
                Optional custom upper-bound clipping values.

                If omitted, production-safe defaults are used.

        Notes:
            The feature schema itself also defines valid feature ranges.
            These caps are intended to prevent extreme telemetry values
            from reaching the model.
        """

        # ------------------------------------------------------------------
        # Canonical schema
        # ------------------------------------------------------------------

        self.feature_names = (
            SSLFeatureSchema.get_feature_names()
        )

        # ------------------------------------------------------------------
        # Expected feature count
        # ------------------------------------------------------------------

        self.feature_count = (
            SSLFeatureSchema.get_feature_count()
        )

        # ------------------------------------------------------------------
        # Outlier caps
        # ------------------------------------------------------------------

        self.outlier_caps: Dict[
            str,
            float
        ] = {

            # Certificate lifetime
            "cert_age_days":
                3650.0,

            "days_until_expiry":
                3650.0,

            "total_lifespan_days":
                3650.0,

            # Trust
            "issuer_trust_score":
                100.0,

            # Cryptographic key strength
            "cipher_strength_bits":
                8192.0,

            # Overall crypto health
            "crypto_health_score":
                100.0
        }

        # ------------------------------------------------------------------
        # Apply custom caps if provided
        # ------------------------------------------------------------------

        if outlier_caps is not None:

            if not isinstance(
                outlier_caps,
                dict
            ):

                raise TypeError(
                    "outlier_caps must be a dictionary."
                )

            for feature, cap in (
                outlier_caps.items()
            ):

                if feature not in (
                    self.feature_names
                ):

                    logger.warning(
                        "Ignoring outlier cap for unknown "
                        "SSL feature '%s'.",
                        feature
                    )

                    continue

                try:

                    numeric_cap = float(
                        cap
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    logger.warning(
                        "Invalid outlier cap for '%s': %r. "
                        "Using default.",
                        feature,
                        cap
                    )

                    continue

                if not np.isfinite(
                    numeric_cap
                ):

                    logger.warning(
                        "Non-finite outlier cap for '%s'. "
                        "Using default.",
                        feature
                    )

                    continue

                self.outlier_caps[
                    feature
                ] = numeric_cap

        # ------------------------------------------------------------------
        # Validate configuration
        # ------------------------------------------------------------------

        self._validate_configuration()

        logger.info(
            "%s initialized successfully. "
            "Features=%d Version=%s",
            self.PREPROCESSOR_NAME,
            self.feature_count,
            self.PREPROCESSOR_VERSION
        )

    # ======================================================================
    # CONFIGURATION VALIDATION
    # ======================================================================

    def _validate_configuration(
        self
    ) -> None:
        """
        Validate preprocessing configuration against the central schema.

        Raises:
            ValueError:
                If the preprocessing configuration is inconsistent.
        """

        # ------------------------------------------------------------------
        # Feature count
        # ------------------------------------------------------------------

        if self.feature_count != len(
            self.feature_names
        ):

            raise ValueError(
                (
                    "SSL schema feature count mismatch: "
                    f"declared={self.feature_count}, "
                    f"actual={len(self.feature_names)}"
                )
            )

        # ------------------------------------------------------------------
        # Schema integrity
        # ------------------------------------------------------------------

        try:

            integrity_valid, integrity_errors = (
                SSLFeatureSchema.check_schema_integrity()
            )

        except AttributeError:

            # Defensive compatibility if an older schema is used.
            integrity_valid = True
            integrity_errors = []

        if not integrity_valid:

            raise ValueError(
                (
                    "SSLFeatureSchema integrity check failed: "
                    f"{integrity_errors}"
                )
            )

        # ------------------------------------------------------------------
        # Validate caps
        # ------------------------------------------------------------------

        for feature, cap in (
            self.outlier_caps.items()
        ):

            if feature not in (
                self.feature_names
            ):

                continue

            if not np.isfinite(
                cap
            ):

                raise ValueError(
                    (
                        f"Outlier cap for '{feature}' "
                        "must be finite."
                    )
                )

            if cap < 0:

                raise ValueError(
                    (
                        f"Outlier cap for '{feature}' "
                        "cannot be negative."
                    )
                )

    # ======================================================================
    # SINGLE RECORD TRANSFORMATION
    # ======================================================================

    def transform_single(
        self,
        raw_features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Transform one raw SSL feature dictionary into an ML-ready DataFrame.

        This method is intended for live inference.

        Args:
            raw_features:
                Raw SSL/TLS telemetry dictionary.

        Returns:
            pandas.DataFrame:
                One-row DataFrame containing exactly 17 numerical
                features in canonical order.

        Example input:

            {
                "has_https": True,
                "certificate_expired": False,
                "certificate_age_days": 120,
                "trust_score": 95
            }

        Example output columns:

            [
                "has_ssl",
                "is_expired",
                "cert_age_days",
                ...
                "is_secure_connection"
            ]
        """

        try:

            # ==============================================================
            # INPUT VALIDATION
            # ==============================================================

            if not isinstance(
                raw_features,
                dict
            ):

                logger.warning(
                    "SSL single-record input must be a dictionary. "
                    "Received %s. Using schema defaults.",
                    type(
                        raw_features
                    ).__name__
                )

                raw_features = {}

            # ==============================================================
            # SCHEMA NORMALIZATION
            # ==============================================================

            normalized_features = (
                SSLFeatureSchema.validate_and_normalize(
                    raw_features
                )
            )

            # ==============================================================
            # DATAFRAME CREATION
            # ==============================================================

            df = pd.DataFrame(
                [
                    normalized_features
                ]
            )

            # ==============================================================
            # FINAL TRANSFORMATION
            # ==============================================================

            df_processed = (
                self._apply_transformations(
                    df
                )
            )

            # ==============================================================
            # FINAL STRUCTURE CHECK
            # ==============================================================

            self._validate_processed_dataframe(
                df_processed
            )

            return df_processed

        except Exception as e:

            logger.error(
                "SSL single-record preprocessing failed: %s",
                str(e),
                exc_info=True
            )

            # --------------------------------------------------------------
            # Safe deterministic fallback
            # --------------------------------------------------------------

            fallback_features = (
                SSLFeatureSchema.get_default_features()
            )

            fallback_df = pd.DataFrame(
                [
                    fallback_features
                ],
                columns=self.feature_names
            )

            fallback_df = (
                self._apply_transformations(
                    fallback_df
                )
            )

            return fallback_df

    # ======================================================================
    # BATCH TRANSFORMATION
    # ======================================================================

    def transform_batch(
        self,
        df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform a batch DataFrame for model training.

        This method guarantees that the final DataFrame has exactly the
        same columns and ordering used during live inference.

        Args:
            df_raw:
                Raw training DataFrame.

        Returns:
            pandas.DataFrame:
                Cleaned, numerical, deterministic feature matrix.

        Raises:
            TypeError:
                If input is not a DataFrame.

            ValueError:
                If the dataset cannot be safely transformed.
        """

        # ==============================================================
        # INPUT VALIDATION
        # ==============================================================

        if not isinstance(
            df_raw,
            pd.DataFrame
        ):

            raise TypeError(
                (
                    "SSLPreprocessor.transform_batch() expects "
                    "a pandas DataFrame."
                )
            )

        try:

            logger.debug(
                "Starting SSL batch preprocessing. "
                "Rows=%d Columns=%d",
                len(df_raw),
                len(df_raw.columns)
            )

            # ==========================================================
            # COPY
            # ==========================================================

            df = df_raw.copy()

            # ==========================================================
            # NORMALIZE COLUMN ALIASES
            # ==========================================================

            df = self._normalize_dataframe_columns(
                df
            )

            # ==========================================================
            # ALIGN SCHEMA
            # ==========================================================

            df = self._align_dataframe_schema(
                df
            )

            # ==========================================================
            # TRANSFORM
            # ==========================================================

            df_processed = (
                self._apply_transformations(
                    df
                )
            )

            # ==========================================================
            # FINAL VALIDATION
            # ==========================================================

            self._validate_processed_dataframe(
                df_processed
            )

            logger.debug(
                "SSL batch preprocessing completed. "
                "Rows=%d Columns=%d",
                len(df_processed),
                len(df_processed.columns)
            )

            return df_processed

        except Exception as e:

            logger.error(
                "SSL batch preprocessing failed: %s",
                str(e),
                exc_info=True
            )

            raise

    # ======================================================================
    # DATAFRAME COLUMN NORMALIZATION
    # ======================================================================

    def _normalize_dataframe_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize alternate SSL feature names to canonical names.

        Example:

            has_https
                ↓
            has_ssl

        This operates on column names only.
        """

        if df.empty:

            return df.copy()

        alias_map = (
            SSLFeatureSchema.FEATURE_ALIASES
        )

        rename_map = {}

        existing_columns = set(
            df.columns
        )

        for column in df.columns:

            canonical_name = (
                alias_map.get(
                    column
                )
            )

            if canonical_name is None:

                continue

            # ----------------------------------------------------------
            # Canonical column already exists.
            #
            # Preserve canonical data rather than overwriting it.
            # ----------------------------------------------------------

            if canonical_name in existing_columns:

                logger.debug(
                    (
                        "Both alias '%s' and canonical feature "
                        "'%s' exist. Keeping canonical column."
                    ),
                    column,
                    canonical_name
                )

                continue

            rename_map[
                column
            ] = canonical_name

        if rename_map:

            logger.debug(
                "Normalizing SSL feature columns: %s",
                rename_map
            )

            df = df.rename(
                columns=rename_map
            )

        return df

    # ======================================================================
    # SCHEMA ALIGNMENT
    # ======================================================================

    def _align_dataframe_schema(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align DataFrame to the exact SSL schema.

        Missing columns receive schema defaults.

        Extra columns are removed.

        Final order exactly matches SSLFeatureSchema.
        """

        df = df.copy()

        defaults = (
            SSLFeatureSchema.get_default_features()
        )

        # ==============================================================
        # ADD MISSING FEATURES
        # ==============================================================

        missing_features = []

        for feature_name in (
            self.feature_names
        ):

            if feature_name not in df.columns:

                missing_features.append(
                    feature_name
                )

                df[
                    feature_name
                ] = defaults[
                    feature_name
                ]

        if missing_features:

            logger.debug(
                "Added missing SSL features with defaults: %s",
                missing_features
            )

        # ==============================================================
        # REMOVE EXTRA FEATURES
        # ==============================================================

        extra_features = [

            column

            for column in df.columns

            if column not in self.feature_names
        ]

        if extra_features:

            logger.debug(
                "Dropping extra SSL feature columns: %s",
                extra_features
            )

            df = df.drop(
                columns=extra_features
            )

        # ==============================================================
        # EXACT FEATURE ORDER
        # ==============================================================

        df = df[
            self.feature_names
        ]

        return df

    # ======================================================================
    # MAIN TRANSFORMATION PIPELINE
    # ======================================================================

    def _apply_transformations(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply all numerical preprocessing operations.

        Order:

            1. Schema alignment
            2. Boolean conversion
            3. Numeric conversion
            4. Infinity replacement
            5. NaN imputation
            6. Range clipping
            7. Outlier capping
            8. Final float conversion
            9. Feature ordering
        """

        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise TypeError(
                "_apply_transformations expects DataFrame."
            )

        # ==============================================================
        # COPY
        # ==============================================================

        df = df.copy()

        # ==============================================================
        # SCHEMA ALIGNMENT
        # ==============================================================

        df = self._align_dataframe_schema(
            df
        )

        # ==============================================================
        # BOOLEAN CONVERSION
        # ==============================================================

        df = self._convert_boolean_columns(
            df
        )

        # ==============================================================
        # NUMERICAL CONVERSION
        # ==============================================================

        df = self._convert_numeric_columns(
            df
        )

        # ==============================================================
        # INFINITY → NaN
        # ==============================================================

        df = df.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        # ==============================================================
        # MISSING VALUE IMPUTATION
        # ==============================================================

        df = self._impute_missing_values(
            df
        )

        # ==============================================================
        # SCHEMA RANGE ENFORCEMENT
        # ==============================================================

        df = self._apply_schema_ranges(
            df
        )

        # ==============================================================
        # OUTLIER CAPPING
        # ==============================================================

        df = self._apply_outlier_caps(
            df
        )

        # ==============================================================
        # FINAL FLOAT CONVERSION
        # ==============================================================

        df = self._convert_all_to_float(
            df
        )

        # ==============================================================
        # FINAL COLUMN ORDER
        # ==============================================================

        df = df[
            self.feature_names
        ]

        return df

    # ======================================================================
    # BOOLEAN CONVERSION
    # ======================================================================

    def _convert_boolean_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert boolean features to numerical 0/1 values.

        Handles:

            True
            False
            "true"
            "false"
            "yes"
            "no"
            "1"
            "0"
            1
            0
        """

        df = df.copy()

        boolean_features = (
            SSLFeatureSchema.BOOLEAN_FEATURES
        )

        for feature_name in boolean_features:

            if feature_name not in df.columns:

                continue

            df[
                feature_name
            ] = df[
                feature_name
            ].apply(
                self._safe_boolean_to_numeric
            )

        return df

    # ======================================================================
    # SAFE BOOLEAN CONVERSION
    # ======================================================================

    @staticmethod
    def _safe_boolean_to_numeric(
        value: Any
    ) -> float:
        """
        Convert a value into numerical boolean representation.

        Returns:

            True  → 1.0
            False → 0.0
        """

        if value is None:

            return 0.0

        if isinstance(
            value,
            bool
        ):

            return (
                1.0
                if value
                else 0.0
            )

        if isinstance(
            value,
            (int, float)
        ):

            try:

                numeric = float(
                    value
                )

                if not np.isfinite(
                    numeric
                ):

                    return 0.0

                return (
                    1.0
                    if numeric != 0
                    else 0.0
                )

            except (
                TypeError,
                ValueError
            ):

                return 0.0

        if isinstance(
            value,
            str
        ):

            normalized = (
                value
                .strip()
                .lower()
            )

            if normalized in {
                "true",
                "1",
                "yes",
                "y",
                "on",
                "enabled",
                "valid",
                "secure"
            }:

                return 1.0

            if normalized in {
                "false",
                "0",
                "no",
                "n",
                "off",
                "disabled",
                "invalid",
                "insecure",
                "",
                "none",
                "null"
            }:

                return 0.0

        return 0.0

    # ======================================================================
    # NUMERIC CONVERSION
    # ======================================================================

    def _convert_numeric_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert all non-boolean SSL features to numeric values.

        Invalid values become NaN and are subsequently imputed.
        """

        df = df.copy()

        numeric_features = (

            set(
                SSLFeatureSchema.INTEGER_FEATURES
            )

            |

            set(
                SSLFeatureSchema.FLOAT_FEATURES
            )
        )

        for feature_name in numeric_features:

            if feature_name not in df.columns:

                continue

            df[
                feature_name
            ] = pd.to_numeric(
                df[
                    feature_name
                ],
                errors="coerce"
            )

        return df

    # ======================================================================
    # MISSING VALUE IMPUTATION
    # ======================================================================

    def _impute_missing_values(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Replace NaN values with schema-defined defaults.

        This is preferable to using a universal zero because the schema
        explicitly defines the semantic default for every feature.
        """

        df = df.copy()

        defaults = (
            SSLFeatureSchema.get_default_features()
        )

        for feature_name in (
            self.feature_names
        ):

            if feature_name not in df.columns:

                df[
                    feature_name
                ] = defaults[
                    feature_name
                ]

                continue

            default_value = (
                defaults[
                    feature_name
                ]
            )

            # Convert NaN to default
            df[
                feature_name
            ] = df[
                feature_name
            ].fillna(
                default_value
            )

        return df

    # ======================================================================
    # SCHEMA RANGE ENFORCEMENT
    # ======================================================================

    def _apply_schema_ranges(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply the minimum/maximum ranges defined by SSLFeatureSchema.

        This protects the model from impossible numerical values.
        """

        df = df.copy()

        feature_ranges = (
            SSLFeatureSchema.get_feature_ranges()
        )

        for feature_name, (
            minimum,
            maximum
        ) in feature_ranges.items():

            if feature_name not in df.columns:

                continue

            # ----------------------------------------------------------
            # Boolean values
            # ----------------------------------------------------------

            if feature_name in (
                SSLFeatureSchema.BOOLEAN_FEATURES
            ):

                df[
                    feature_name
                ] = df[
                    feature_name
                ].clip(
                    lower=0.0,
                    upper=1.0
                )

                continue

            # ----------------------------------------------------------
            # Numeric values
            # ----------------------------------------------------------

            if minimum is not None:

                df[
                    feature_name
                ] = df[
                    feature_name
                ].clip(
                    lower=minimum
                )

            if maximum is not None:

                df[
                    feature_name
                ] = df[
                    feature_name
                ].clip(
                    upper=maximum
                )

        return df

    # ======================================================================
    # OUTLIER CAPS
    # ======================================================================

    def _apply_outlier_caps(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply additional conservative outlier caps.

        These are intentionally separate from schema validity ranges.

        Example:

            Schema permits:
                cert_age_days <= 100000

            Preprocessor model cap:
                cert_age_days <= 3650

        This allows the schema to remain broadly valid while keeping
        extreme telemetry from dominating the model.
        """

        df = df.copy()

        for feature_name, max_cap in (
            self.outlier_caps.items()
        ):

            if feature_name not in df.columns:

                continue

            df[
                feature_name
            ] = df[
                feature_name
            ].clip(
                upper=max_cap
            )

        return df

    # ======================================================================
    # FINAL FLOAT CONVERSION
    # ======================================================================

    def _convert_all_to_float(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert every model feature to float.

        XGBoost receives a purely numerical matrix.
        """

        df = df.copy()

        for feature_name in (
            self.feature_names
        ):

            df[
                feature_name
            ] = pd.to_numeric(
                df[
                    feature_name
                ],
                errors="coerce"
            )

        # --------------------------------------------------------------
        # Any values that became NaN during conversion are replaced
        # with zero as the final safety layer.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Force float32 for efficient ML processing.
        # --------------------------------------------------------------

        for feature_name in (
            self.feature_names
        ):

            df[
                feature_name
            ] = df[
                feature_name
            ].astype(
                np.float32
            )

        return df

    # ======================================================================
    # PROCESSED DATAFRAME VALIDATION
    # ======================================================================

    def _validate_processed_dataframe(
        self,
        df: pd.DataFrame
    ) -> None:
        """
        Validate the final ML-ready DataFrame.

        Raises:
            ValueError:
                If the DataFrame does not satisfy the expected model
                input contract.
        """

        # ==============================================================
        # DATAFRAME TYPE
        # ==============================================================

        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise ValueError(
                "Processed SSL output must be a pandas DataFrame."
            )

        # ==============================================================
        # COLUMN COUNT
        # ==============================================================

        if len(
            df.columns
        ) != self.feature_count:

            raise ValueError(
                (
                    "Processed SSL feature count mismatch. "
                    f"Expected {self.feature_count}, "
                    f"received {len(df.columns)}."
                )
            )

        # ==============================================================
        # COLUMN ORDER
        # ==============================================================

        if list(
            df.columns
        ) != self.feature_names:

            raise ValueError(
                (
                    "Processed SSL feature order mismatch. "
                    "Expected canonical schema order."
                )
            )

        # ==============================================================
        # NUMERICAL DATA
        # ==============================================================

        for feature_name in (
            self.feature_names
        ):

            if not pd.api.types.is_numeric_dtype(
                df[
                    feature_name
                ]
            ):

                raise ValueError(
                    (
                        f"SSL feature '{feature_name}' "
                        "is not numerical after preprocessing."
                    )
                )

        # ==============================================================
        # FINITE VALUES
        # ==============================================================

        numeric_values = (
            df[
                self.feature_names
            ].to_numpy(
                dtype=np.float64
            )
        )

        if not np.isfinite(
            numeric_values
        ).all():

            raise ValueError(
                (
                    "Processed SSL feature matrix contains "
                    "NaN or infinite values."
                )
            )

    # ======================================================================
    # TRANSFORM TO NUMPY
    # ======================================================================

    def transform_to_numpy(
        self,
        raw_features: Dict[str, Any]
    ) -> np.ndarray:
        """
        Transform one raw feature dictionary directly into a NumPy
        model-ready matrix.

        Shape:

            (1, 17)
        """

        df = self.transform_single(
            raw_features
        )

        return df[
            self.feature_names
        ].to_numpy(
            dtype=np.float32
        )

    # ======================================================================
    # TRANSFORM BATCH TO NUMPY
    # ======================================================================

    def batch_to_numpy(
        self,
        df_raw: pd.DataFrame
    ) -> np.ndarray:
        """
        Transform a training DataFrame into a NumPy matrix.

        Shape:

            (number_of_samples, 17)
        """

        df = self.transform_batch(
            df_raw
        )

        return df[
            self.feature_names
        ].to_numpy(
            dtype=np.float32
        )

    # ======================================================================
    # FEATURE VECTOR DICTIONARY
    # ======================================================================

    def transform_single_to_dict(
        self,
        raw_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Transform one raw feature dictionary and return a numerical
        model-ready dictionary.

        Useful for debugging and API payloads.
        """

        df = self.transform_single(
            raw_features
        )

        row = df.iloc[
            0
        ]

        return {

            feature_name:
                float(
                    row[
                        feature_name
                    ]
                )

            for feature_name
            in self.feature_names
        }

    # ======================================================================
    # GET FEATURE NAMES
    # ======================================================================

    def get_feature_names(
        self
    ) -> List[str]:
        """
        Return the exact feature order used by the preprocessor.
        """

        return self.feature_names.copy()

    # ======================================================================
    # GET FEATURE COUNT
    # ======================================================================

    def get_feature_count(
        self
    ) -> int:
        """
        Return the number of model features.
        """

        return self.feature_count

    # ======================================================================
    # GET STATUS
    # ======================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return preprocessing component status.

        Useful for the SSL Agent health endpoint and debugging.
        """

        return {

            "component":
                self.PREPROCESSOR_NAME,

            "version":
                self.PREPROCESSOR_VERSION,

            "status":
                "ready",

            "schema_name":
                SSLFeatureSchema.SCHEMA_NAME,

            "schema_version":
                SSLFeatureSchema.SCHEMA_VERSION,

            "feature_count":
                self.feature_count,

            "feature_names":
                self.get_feature_names(),

            "outlier_caps":
                self.outlier_caps.copy()
        }